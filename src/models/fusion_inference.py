from __future__ import annotations

import gc
import hashlib
import os
import re
import concurrent.futures
import threading
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
from loguru import logger

from src.config import LABEL_LIST, PROMPT_TEMPLATE
from src.database.opensearch import OpenSearchKB
from src.llm_call import rewrite_claim
from src.retrieval.retrieval import QueryExpander, RetrievalResult, TemporalScorer

# Nhãn hiển thị theo đúng thứ tự index của model (0=Đúng, 1=Sai, 2=Chưa chắc chắn),
# tách biệt với self.label_list (A/B/C) vốn chỉ dùng cho prompt LLM.
VERDICT_ORDER = ["Đúng", "Sai", "Chưa chắc chắn"]

_URL_KEYS: Tuple[str, ...] = ("article_url", "url", "link", "source_url")


def _extract_url(meta: Dict[str, Any]) -> str:
    """Lấy URL đầu tiên tìm được trong metadata theo thứ tự ưu tiên _URL_KEYS."""
    return next((str(meta[k]).strip() for k in _URL_KEYS if meta.get(k)), "")


def _looks_like_listing_page(url: str) -> bool:
    """Đoán xem URL có phải trang danh mục/chuyên mục (không phải bài báo thật) hay không."""
    if not url:
        return False
    path = urlparse(url).path
    last_segment = path.rstrip("/").rsplit("/", 1)[-1]
    return not re.search(r"\d{6,}", last_segment) # Các bài báo thật thường có ID/Timestamp ở cuối URL


def _resolve_fusion_model_path(path_or_repo: str, filename: str = "fusion_gold.pt") -> str:
    """Trả về đường dẫn tới file model fusion: dùng file local nếu có sẵn,
    ngược lại tải về từ HuggingFace Hub theo repo_id."""
    if os.path.isfile(path_or_repo):
        return path_or_repo
    try:
        from huggingface_hub import hf_hub_download

        print(path_or_repo)
        print(filename)

        local_path = hf_hub_download(repo_id=path_or_repo, filename=filename)
        logger.info(
            f"[fusion_inference] Downloaded {filename} from HF repo '{path_or_repo}' -> {local_path}"
        )
        return local_path
    except Exception as exc:
        raise FileNotFoundError(
            f"Cannot resolve fusion model path '{path_or_repo}': {exc}"
        ) from exc


try:
    from src.models.fusion import ConfidenceAwareFusion, RetrievalFeatureEncoder
except ImportError:
    ConfidenceAwareFusion = None  # type: ignore
    RetrievalFeatureEncoder = None  # type: ignore

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # type: ignore


def _parse_timestamp(value: Any) -> datetime:
    """Cố gắng parse timestamp từ nhiều định dạng và chuẩn hoá về UTC."""
    now = datetime.now(timezone.utc)
    if value is None:
        return now

    if isinstance(value, datetime):
        ts = value
    elif isinstance(value, (int, float)):
        ts = datetime.fromtimestamp(float(value), tz=timezone.utc)
    else:
        raw = str(value).strip()
        if not raw:
            return now
        raw = raw.replace("Z", "+00:00")
        try:
            ts = datetime.fromisoformat(raw)
        except ValueError:
            try:
                ts = datetime.fromtimestamp(float(raw), tz=timezone.utc)
            except (ValueError, OverflowError):
                return now

    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def extract_date_range(text: str) -> Tuple[Optional[str], Optional[str], str]:
    """Trích khoảng thời gian (min, max) từ các mốc ngày/tháng/năm trong text.

    Bỏ qua nếu ngày nằm trong tương lai. Trả về (min_iso, max_iso, text_đã_bỏ_ngày);
    nếu không tìm thấy ngày hợp lệ thì trả về (None, None, text_gốc).
    """
    now_utc = datetime.now(timezone.utc)
    current_year = now_utc.year
    current_month = now_utc.month
    current_day = now_utc.day

    pat = re.compile(
        r"(?:ngày\s+)?(\d{1,2})(?:\s*[-/]\s*|\s+tháng\s+)(\d{1,2})(?:(?:\s*[-/]\s*|\s+năm\s+)(\d{2,4}))?", 
        re.IGNORECASE
    )
    
    matches = list(pat.finditer(text))
    dates = []  # (datetime, match_object) đã sắp theo thời gian
    for mo in matches:
        d_str, m_str, y_str = mo.group(1), mo.group(2), mo.group(3)
        d = int(d_str) if d_str else current_day
        m_val = int(m_str) if m_str else current_month
        y = int(y_str) if y_str else current_year
        if y < 100:
            y += 2000

        try:
            dt = datetime(y, m_val, d, tzinfo=timezone.utc)
            dates.append((dt, mo))
        except ValueError:
            pass

    if not dates:
        return None, None, text

    dates.sort(key=lambda x: x[0])

    if dates[-1][0].date() > now_utc.date():
        return None, None, text

    first_dt, first_mo = dates[0]
    last_dt = dates[-1][0]

    # "Từ/Kể từ/Bắt đầu từ/Sau <ngày>" là mốc MỞ: khoảng [ngày, hiện tại],
    # không phải chỉ đúng một ngày. Trước đây một mốc đơn bị co về [ngày, 23:59]
    # cùng ngày, làm lọt mất mọi bài đăng sau đó (chính là evidence cần nhất).
    prefix = text[: first_mo.start()].rstrip()
    open_ended = bool(
        re.search(r"(?i)(k[ểe]\s+t[ừu]|b[ắa]t\s+đ[ầa]u\s+t[ừu]|t[ừu]|sau)\s*$", prefix)
    )

    min_dt = first_dt
    if open_ended:
        max_dt = now_utc
    elif len(dates) == 1:
        max_dt = first_dt.replace(hour=23, minute=59, second=59)
    else:
        max_dt = last_dt.replace(hour=23, minute=59, second=59)

    # Bỏ chuỗi ngày khỏi text truy vấn, đồng thời dọn liên từ thời gian mồ côi
    # còn sót ở đầu ("Từ ,", "Kể từ ...") để query không bị cụt như "Từ , ...".
    cleaned_text = pat.sub("", text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    cleaned_text = re.sub(
        r"(?i)^(k[ểe]\s+t[ừu]|b[ắa]t\s+đ[ầa]u\s+t[ừu]|t[ừu]|sau|ng[àa]y)\b[\s,\.]*",
        "",
        cleaned_text,
    )
    cleaned_text = re.sub(r"^[\s,\.;:–—-]+", "", cleaned_text).strip()

    return min_dt.isoformat(), max_dt.isoformat(), cleaned_text


def _select_doc_text(source: Dict[str, Any]) -> str:
    """Chọn đoạn text làm evidence từ document OpenSearch (ưu tiên text/content/description/title)."""
    for key in ("text", "content", "description", "title"):
        value = source.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return ""


_MIN_EVIDENCE_BODY_CHARS = int(os.getenv("FUSION_MIN_EVIDENCE_BODY_CHARS", "160"))


def _clean_body_text(source: Dict[str, Any]) -> str:
    """Trả về phần thân bài đã bỏ tiêu đề ở đầu và loại nhiễu BOM/khoảng trắng."""
    content = str(source.get("content") or source.get("text") or "")
    content = content.replace("﻿", " ")
    title = str(source.get("title") or "").strip()
    body = content.strip()
    if title and body.startswith(title):
        body = body[len(title):]
    return re.sub(r"\s+", " ", body).strip()


def _has_usable_content(source: Dict[str, Any]) -> bool:
    """True nếu document có đủ độ dài thân bài để coi là evidence đáng tin."""
    return len(_clean_body_text(source)) >= _MIN_EVIDENCE_BODY_CHARS


# Bài dài được chia thành nhiều chunk-document lúc crawl (xem crawler.py
# _split_sentences). Một số "document" trong index thực ra là trang danh
# mục/chuyên mục (ghép nhiều tiêu đề rời rạc, không phải một bài thật) — loại
# này có thể sinh ra nhiều chunk hơn bất kỳ bài thật nào. Giới hạn cả số chunk
# lẫn độ dài sau khi ghép để không đưa loại nhiễu đó vào LLM như một bài liền mạch.
_MAX_CHUNKS_TO_MERGE = 8
_MAX_MERGED_ARTICLE_CHARS = 8000


def _merge_article_chunks(title: str, chunk_hits: List[Any]) -> Optional[str]:
    """ghép evidence từ chunk"""
    if len(chunk_hits) <= 1:
        return None
    if len(chunk_hits) > _MAX_CHUNKS_TO_MERGE:
        return None
    if any(h.source.get("chunk_idx") is None for h in chunk_hits):
        return None

    ordered = sorted(chunk_hits, key=lambda h: h.source.get("chunk_idx"))
    pieces = []
    for h in ordered:
        content = str(h.source.get("content") or "").strip()
        if title and content.startswith(title):
            content = content[len(title):].strip()
        if content:
            pieces.append(content)
    if not pieces:
        return None

    merged = (f"{title} " if title else "") + " ".join(pieces)
    merged = merged.strip()
    if len(merged) > _MAX_MERGED_ARTICLE_CHARS:
        merged = merged[:_MAX_MERGED_ARTICLE_CHARS].rstrip() + "…"
    return merged or None



_EVIDENCE_MIN_SENTENCE_CHARS = 12 #min input cho sentence
_EVIDENCE_MAX_SENTENCES_SCANNED = int(os.getenv("FUSION_EVIDENCE_MAX_SENTENCES", "80")) #max input cho sentence 
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?…])\s+")# tách câu


def _split_sentences(text: str) -> List[str]:
    """Tách text evidence thành các câu (gần đúng) để phục vụ trích câu liên quan."""
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return []
    parts = _SENTENCE_BOUNDARY_RE.split(normalized)
    return [p.strip() for p in parts if len(p.strip()) >= _EVIDENCE_MIN_SENTENCE_CHARS]


# Khối menu/điều hướng của trang báo (danh sách nhiều tiêu đề rời ghép lại,
# gần như không có dấu chấm câu) thường lọt vào full-article dưới dạng MỘT
# "câu" dài bất thường. Nhận diện bằng: rất dài VÀ mật độ dấu phẩy thấp — câu
# tin thật dù dài vẫn có dấu phẩy ngắt mệnh đề, còn menu là chuỗi tiêu đề nối
# nhau gần như không ngắt. (Đo thực tế: menu ~124 từ / density 0.02, câu tin
# 24–36 từ / density 0.07–0.13.)
_BOILERPLATE_MIN_WORDS = int(os.getenv("FUSION_BOILERPLATE_MIN_WORDS", "50"))
_BOILERPLATE_MAX_COMMA_DENSITY = float(
    os.getenv("FUSION_BOILERPLATE_MAX_COMMA_DENSITY", "0.05")
)
# Câu evidence trích một ngày lệch quá xa ngày của claim (± ngần này ngày) bị
# coi là nội dung cũ/ngược hướng và loại khỏi evidence.
_EVIDENCE_DATE_TOLERANCE_DAYS = int(os.getenv("FUSION_EVIDENCE_DATE_TOLERANCE_DAYS", "2"))
_DATE_IN_TEXT_RE = re.compile(
    r"(?:ngày\s+)?(\d{1,2})(?:\s*[-/]\s*|\s+tháng\s+)(\d{1,2})(?:(?:\s*[-/]\s*|\s+năm\s+)(\d{2,4}))?",
    re.IGNORECASE,
)


def _looks_like_boilerplate(sentence: str) -> bool:
    """True nếu 'câu' thực chất là khối menu/điều hướng ghép nhiều tiêu đề:
    dài bất thường VÀ mật độ dấu phẩy thấp."""
    n = len(sentence.split())
    if n < _BOILERPLATE_MIN_WORDS:
        return False
    return sentence.count(",") / n < _BOILERPLATE_MAX_COMMA_DENSITY


def _dates_in_text(text: str) -> List[datetime]:
    """Trích danh sách mốc ngày (UTC) xuất hiện trong text. Ngày không có năm
    mặc định lấy năm hiện tại; mốc không hợp lệ bị bỏ qua."""
    now = datetime.now(timezone.utc)
    out: List[datetime] = []
    for d_str, m_str, y_str in _DATE_IN_TEXT_RE.findall(text or ""):
        try:
            y = int(y_str) if y_str else now.year
            if y < 100:
                y += 2000
            out.append(datetime(y, int(m_str), int(d_str), tzinfo=timezone.utc))
        except (ValueError, TypeError):
            continue
    return out


def _claim_date_window(
    min_ts: Optional[str], max_ts: Optional[str]
) -> "Optional[Tuple[datetime, datetime]]":
    """Dựng cửa sổ [lo, hi] quanh ngày của claim (± _EVIDENCE_DATE_TOLERANCE_DAYS)
    để lọc câu evidence lệch ngày. Trả về None nếu claim không kèm ngày."""
    if not min_ts and not max_ts:
        return None
    tol = timedelta(days=_EVIDENCE_DATE_TOLERANCE_DAYS)
    lo = _parse_timestamp(min_ts) if min_ts else _parse_timestamp(max_ts)
    hi = _parse_timestamp(max_ts) if max_ts else _parse_timestamp(min_ts)
    return lo - tol, hi + tol


def _sentence_in_window(
    sentence: str, window: "Optional[Tuple[datetime, datetime]]"
) -> bool:
    """Giữ câu nếu không có cửa sổ, câu không trích ngày nào, hoặc có ít nhất
    một ngày nằm trong cửa sổ. Loại câu chỉ trích ngày ngoài cửa sổ."""
    if window is None:
        return True
    dates = _dates_in_text(sentence)
    if not dates:
        return True
    lo, hi = window
    return any(lo <= d <= hi for d in dates)


def _env_flag(name: str, default: bool = False) -> bool:
    """Đọc biến môi trường dạng cờ bật/tắt (1/true/yes/y/on = True)."""
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _truncate(text: str, max_chars: int) -> str:
    """Cắt bớt chuỗi về tối đa max_chars ký tự, thêm dấu … ở cuối (max_chars<=0 = không cắt)."""
    s = str(text)
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + "…"


def _normalize_query_text(text: str) -> str:
    """Chuẩn hoá text truy vấn: gộp khoảng trắng thừa và cắt đầu/cuối."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _tokenize_for_overlap(text: str) -> List[str]:
    """Tách chuỗi thành các token chữ-số viết thường để tính độ trùng lặp."""
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _is_verbatim_query(query: str) -> bool:
    """Nhận biết truy vấn là kiểu dán nguyên cả bài (rất dài) thay vì một claim ngắn."""
    normalized = _normalize_query_text(query)
    token_count = len(_tokenize_for_overlap(normalized))
    if not normalized:
        return False
    # Chỉ coi là dán nguyên cả bài khi nó đặc biệt dài
    if len(normalized) >= 1200:
        return True
    return token_count >= 200


def _token_overlap_ratio(query_text: str, doc_text: str) -> float:
    """Tỉ lệ token của truy vấn cũng xuất hiện trong document (0..1)."""
    q_tokens = set(_tokenize_for_overlap(query_text))
    d_tokens = set(_tokenize_for_overlap(doc_text))
    if not q_tokens or not d_tokens:
        return 0.0
    return float(len(q_tokens & d_tokens) / len(q_tokens))




def _build_retrieval_features_train_compatible(
    retriever: Any,
    text: str,
    top_k: int,
    rrf_top_k: int = 100,
    precomputed_vector: Optional[List[float]] = None,
    score_features: int = 5,
    min_timestamp: Optional[str] = None,
    max_timestamp: Optional[str] = None,
    evidence_top_k: Optional[int] = None,
) -> "tuple[np.ndarray, Optional[np.ndarray], List[str], List[RetrievalResult]]":
    """
    Dựng feature giống hệt lúc training.
    Trả về (mảng_score_features, interaction_features, evidence_texts, results).
    interaction_features có shape [2*emb_dim] = concat(q⊙mean_d, |q-mean_d|) hoặc None.

    `top_k` quy định kích thước ma trận feature dạng số (được pad thành đúng
    `top_k` hàng ở dưới — bắt buộc, vì MLP nhánh retrieval được train với shape
    cố định này). `evidence_top_k`, nếu nhỏ hơn, giới hạn độc lập số evidence
    text (đã dedup/ghép full-article) được giữ lại — nó không bao giờ làm thay
    đổi shape ma trận feature đã pad.
    """
    results = retriever.retrieve(
        text,
        top_k=top_k,
        rrf_top_k=rrf_top_k,
        precomputed_vector=precomputed_vector,
        min_timestamp=min_timestamp,
        max_timestamp=max_timestamp,
        evidence_top_k=evidence_top_k,
    )
    features = []
    doc_embs = []
    evidence_texts = []

    for r in results:
        row = [r.score, r.rrf_score, r.recency_score, r.cyclicity_score]
        if score_features >= 5:
            row.append(r.cosine_similarity)
        features.append(row)
        if r.embedding is not None:
            doc_embs.append(r.embedding)
        evidence_texts.append(r.text)

    pad = top_k - len(features)
    if pad > 0:
        base_dim = min(score_features, 5)
        features.extend([[0.0] * base_dim] * pad)

    interaction = None
    q_emb = getattr(retriever, "_last_query_embedding", None)
    if q_emb is not None and doc_embs:
        mean_d = np.array(doc_embs, dtype=np.float32).mean(axis=0)
        interaction = np.concatenate(
            [q_emb * mean_d, np.abs(q_emb - mean_d)], dtype=np.float32
        )

    return np.array(features, dtype=np.float32), interaction, evidence_texts, results


class OpenSearchHybridRetriever:
    """
    Lớp bọc retrieval giữ nguyên pipeline tính điểm như lúc training, nhưng
    dùng OpenSearch để làm BM25 và tìm kiếm vector.
    """

    def __init__(
        self,
        kb: OpenSearchKB,
        embedding_model: str,
        alpha: float = 0.7,
        lambda_decay: float = 0.1,
        gamma: float = 0.5,
        use_query_expansion: bool = True,
        rrf_k: int = 60,
        device: Optional[str] = None,
    ):
        """Khởi tạo retriever: nạp encoder SentenceTransformer, bộ chấm điểm
        theo thời gian (TemporalScorer) và (tuỳ chọn) bộ mở rộng truy vấn."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for OpenSearch vector retrieval."
            )

        self.kb = kb
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.temporal_scorer = TemporalScorer(
            alpha=alpha, lambda_decay=lambda_decay, gamma=gamma
        )
        self.query_expander = QueryExpander() if use_query_expansion else None
        _st_device = device or os.getenv("DEVICE", "cpu")
        self.encoder = SentenceTransformer(embedding_model, device=_st_device)
        self.embedding_dim = int(self.encoder.get_sentence_embedding_dimension())

        # Đảm bảo phần kiểm tra số chiều trong wrapper OpenSearch khớp với encoder đang dùng.
        self.kb.embedding_dim = self.embedding_dim

    def _select_relevant_sentences(
        self,
        claim: str,
        text: str,
        top_n: int,
        title: str = "",
        date_window: "Optional[Tuple[datetime, datetime]]" = None,
    ) -> str:
        """Giữ lại những câu trong `text` giống `claim` nhất, ghép lại theo đúng
        thứ tự gốc, và luôn đặt tiêu đề bài `title` lên đầu.

        Tiêu đề là bản tóm tắt của chính bài báo về nội dung nó đưa tin, nên
        luôn được giữ trong mọi evidence bất kể độ tương đồng và chiếm một trong
        `top_n` suất (top_n=3 kèm tiêu đề ⇒ tiêu đề + 2 câu thân bài). Các suất
        còn lại dành cho những câu thân bài gần claim nhất, chấm điểm bằng chính
        encoder retrieval đang có sẵn. Chỉ giữ những câu này giúp loại bỏ các con
        số/giọng điệu gây nhiễu khiến NLI đọc thành "mâu thuẫn" và làm phình
        evidence đưa cho LLM. Trả về text gốc nếu có lỗi xảy ra.
        """
        if top_n <= 0 or not claim:
            return text
        title = _normalize_query_text(title)
        body = _normalize_query_text(text)
        # Bài đã ghép bắt đầu bằng tiêu đề — bỏ bản sao ở đầu này để tiêu đề
        # không bị lặp lại và không chiếm mất một suất câu thân bài.
        if title and body.startswith(title):
            body = body[len(title):].strip()
        sentences = _split_sentences(body)
        # Loại khối menu/điều hướng và câu trích ngày lệch xa ngày claim (nội
        # dung cũ/ngược hướng) trước khi chấm điểm liên quan.
        drop_boilerplate = _env_flag(
            "FUSION_EVIDENCE_BOILERPLATE_FILTER_ENABLED", default=True
        )
        drop_stale_dates = _env_flag(
            "FUSION_EVIDENCE_DATE_FILTER_ENABLED", default=True
        )
        if drop_boilerplate:
            sentences = [s for s in sentences if not _looks_like_boilerplate(s)]
        if drop_stale_dates:
            sentences = [s for s in sentences if _sentence_in_window(s, date_window)]
        pick_n = max(0, top_n - (1 if title else 0))
        if len(sentences) <= pick_n:
            chosen = sentences
        else:
            scan = sentences[:_EVIDENCE_MAX_SENTENCES_SCANNED]
            try:
                embs = self.encoder.encode(
                    [claim] + scan,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=32,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"[fusion_inference] sentence_extract_failed: {exc}")
                chosen = scan[:pick_n]
            else:
                sims = embs[1:] @ embs[0]
                top_idx = np.argsort(-sims)[:pick_n]
                chosen = [scan[i] for i in sorted(int(i) for i in top_idx)]
        parts = ([title] if title else []) + chosen
        return " ".join(parts) if parts else text

    def _dedupe_and_fetch_full_articles(
        self,
        ranked_items: List[RetrievalResult],
        top_k: int,
        claim: str = "",
        date_window: "Optional[Tuple[datetime, datetime]]" = None,
    ) -> List[RetrievalResult]:
        """Bỏ các chunk trùng nhau của cùng một bài, dựng lại full-article cho
        mỗi item được giữ, rồi chỉ giữ những câu liên quan nhất tới claim làm
        evidence text.

        Việc dedup chỉ xét trong cửa sổ top_k ban đầu — một chunk trùng bị loại
        sẽ KHÔNG được bù bằng ứng viên xếp hạng thấp hơn. Full-article được dựng
        lại trước để bước trích câu liên quan có thể nhìn thấy mọi chunk của bài
        (câu xác nhận có thể nằm ở chunk không phải chunk được retrieve), sau đó
        `_select_relevant_sentences` cắt gọn lại còn vài câu thực sự nói tới
        claim — đưa evidence cô đọng cho cả LLM lẫn bộ chấm NLI thay vì cả bài
        đầy những con số cạnh tranh nhau. Đặt
        FUSION_EVIDENCE_SENTENCE_EXTRACT_ENABLED=0 để giữ nguyên full-article.

        `ranked_items` cần được sắp xếp tốt-nhất-trước để chunk điểm cao nhất
        của mỗi bài là chunk sống sót sau dedup.
        """
        extract_enabled = _env_flag(
            "FUSION_EVIDENCE_SENTENCE_EXTRACT_ENABLED", default=True
        )
        top_n = int(os.getenv("FUSION_EVIDENCE_SENTENCE_TOP_N", "3"))

        def _focus(items: List[RetrievalResult]) -> List[RetrievalResult]:
            if extract_enabled and claim:
                for it in items:
                    title = str((it.metadata or {}).get("title") or "").strip()
                    it.text = self._select_relevant_sentences(
                        claim, it.text, top_n, title=title, date_window=date_window
                    )
            return items

        deduped: List[RetrievalResult] = []
        seen_urls: set = set()
        for item in ranked_items[:top_k]:
            url = _extract_url(item.metadata or {})
            if url:
                if url in seen_urls:
                    continue
                seen_urls.add(url)
            deduped.append(item)

        urls_to_fetch = sorted(
            {_extract_url(item.metadata or {}) for item in deduped} - {""}
        )
        if not urls_to_fetch:
            return _focus(deduped)

        try:
            chunk_hits = self.kb.get_by_field_values("article_url", urls_to_fetch)
        except Exception as exc:
            logger.warning(f"[fusion_inference] full_article_fetch_failed: {exc}")
            return _focus(deduped)

        hits_by_url: Dict[str, List[Any]] = {}
        for hit in chunk_hits:
            hit_url = str((hit.source or {}).get("article_url") or "").strip()
            if hit_url:
                hits_by_url.setdefault(hit_url, []).append(hit)

        for item in deduped:
            url = _extract_url(item.metadata or {})
            group = hits_by_url.get(url) if url else None
            if not group:
                continue
            title = str((item.metadata or {}).get("title") or "").strip()
            merged = _merge_article_chunks(title, group)
            if merged:
                item.text = merged

        return _focus(deduped)

    def _get_search_pool_size(self, rrf_top_k: int) -> int:
        """
        Lấy một pool đủ lớn để xếp hạng RRF mà không cần tải toàn bộ DB.
        Lúc inference, tải 10k document đầy đủ mất hơn 100 giây qua HTTP.
        """
        hard_cap = int(os.getenv("RETRIEVAL_POOL_SIZE_CAP", "100"))
        pool = min(max(rrf_top_k * 2, 50), hard_cap)
        try:
            count = self.kb.count_docs()
            if count > 0:
                pool = min(count, pool)
        except Exception as exc:
            logger.warning(
                f"Could not fetch OpenSearch count, using pool={pool}: {exc}"
            )
        return pool

    def _doc_group_key(self, source: Dict[str, Any]) -> str:
        """Khoá nhóm của document (theo type/source) dùng cho chấm điểm thời gian."""
        return str(source.get("type") or source.get("source") or "default")

    def _doc_timestamp(self, source: Dict[str, Any]) -> datetime:
        """Lấy timestamp của document từ các trường thời gian có thể có; mặc định là hiện tại."""
        for key in ("timestamp", "published_at", "created_at", "fetched_at"):
            if key in source:
                return _parse_timestamp(source.get(key))
        return datetime.now(timezone.utc)

    def _encode_query(self, query: str) -> List[float]:
        """Encode một truy vấn thành vector embedding (chưa chuẩn hoá)."""
        vector = self.encoder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=False
        )[0]
        return vector.astype(np.float32).tolist()

    def batch_encode(self, queries: List[str]) -> List[List[float]]:
        """Encode nhiều queries trong một lần gọi — nhanh hơn N lần gọi đơn lẻ."""
        vectors = self.encoder.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
            batch_size=min(len(queries), 32),
        )
        return [v.astype(np.float32).tolist() for v in vectors]

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_temporal: bool = True,
        expand_query: bool = True,
        use_semantic: bool = True,
        rrf_top_k: int = 20,
        precomputed_vector: Optional[List[float]] = None,
        min_timestamp: Optional[str] = None,
        max_timestamp: Optional[str] = None,
        evidence_top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Chạy pipeline retrieval lai (BM25 + vector) rồi hợp nhất bằng RRF và
        chấm điểm theo thời gian, trả về danh sách RetrievalResult đã sắp xếp.

        Hỗ trợ lọc theo khoảng thời gian, mở rộng truy vấn, và xử lý riêng cho
        truy vấn dán nguyên cả bài (verbatim). `top_k` quy định số hàng feature,
        `evidence_top_k` giới hạn số evidence text được giữ lại.
        """
        debug = _env_flag("FUSION_INFERENCE_DEBUG", default=False) or _env_flag(
            "FUSION_INFERENCE_LOG_ALL", default=False
        )
        # KHÔNG gán self.temporal_scorer.reference_date ở đây — property đó đã tự
        # trả về datetime.now(timezone.utc) khi _reference_date là None, và việc
        # thay đổi thuộc tính dùng chung sẽ gây race condition khi có nhiều
        # request đồng thời.

        t_retrieve_start = perf_counter()

        normalized_query = _normalize_query_text(query)
        if not normalized_query:
            return []

        verbatim_query = _is_verbatim_query(query)
        expanded_query = normalized_query
        t_expand0 = perf_counter()
        if expand_query and self.query_expander is not None and not verbatim_query:
            expanded_query = self.query_expander.expand_query(normalized_query)
        t_expand1 = perf_counter()

        t_pool0 = perf_counter()
        effective_rrf_top_k = max(rrf_top_k, top_k * 3)
        search_pool_k = self._get_search_pool_size(rrf_top_k=effective_rrf_top_k)
        t_pool1 = perf_counter()

        semantic_enabled = use_semantic and not verbatim_query
        if debug:
            logger.info(
                "[fusion_inference] retrieve"
                f" | query={query!r}"
                f" | normalized_query={normalized_query!r}"
                f" | expanded_query={expanded_query!r}"
                f" | top_k={top_k}"
                f" | rrf_top_k={effective_rrf_top_k}"
                f" | search_pool_k={search_pool_k}"
                f" | verbatim_query={verbatim_query}"
                f" | use_temporal={use_temporal}"
                f" | use_semantic={use_semantic}"
                f" | semantic_enabled={semantic_enabled}"
            )

        t_bm25_0 = perf_counter()
        bm25_hits = self.kb.search_bm25(
            query=expanded_query,
            k=search_pool_k,
            fields=["title^3", "description^2", "content", "text"],
            min_timestamp=min_timestamp,
            max_timestamp=max_timestamp,
        )
        t_bm25_1 = perf_counter()

        vector_hits = []
        t_encode_ms = 0.0
        t_vector_ms = 0.0
        if semantic_enabled:
            if precomputed_vector is not None:
                query_vec = precomputed_vector
            else:
                t_enc0 = perf_counter()
                query_vec = self._encode_query(expanded_query)
                t_encode_ms = 1000.0 * (perf_counter() - t_enc0)
            t_vec0 = perf_counter()
            vector_hits = self.kb.search_vector(
                query_vector=query_vec,
                k=search_pool_k,
                min_timestamp=min_timestamp,
                max_timestamp=max_timestamp,
            )
            t_vector_ms = 1000.0 * (perf_counter() - t_vec0)

        if debug:
            logger.info(
                f"[fusion_inference] retrieve_hits | bm25={len(bm25_hits)} | vector={len(vector_hits)}"
                f" | pool_ms={1000.0*(t_pool1-t_pool0):.1f}"
                f" | expand_ms={1000.0*(t_expand1-t_expand0):.1f}"
                f" | bm25_ms={1000.0*(t_bm25_1-t_bm25_0):.1f}"
                f" | encode_ms={t_encode_ms:.1f}"
                f" | vector_ms={t_vector_ms:.1f}"
            )

        if not bm25_hits and not vector_hits:
            return []

        vector_cosine = {hit.id: hit.score for hit in vector_hits}

        hit_by_id = {}
        for hit in bm25_hits:
            hit_by_id[hit.id] = hit
        for hit in vector_hits:
            if hit.id not in hit_by_id:
                hit_by_id[hit.id] = hit

        if _env_flag("FUSION_LISTING_PAGE_FILTER_ENABLED", default=True):
            hit_by_id = {
                doc_id: hit
                for doc_id, hit in hit_by_id.items()
                if not _looks_like_listing_page(_extract_url(hit.source or {}))
            }
            if not hit_by_id:
                return []

        if _env_flag("FUSION_MIN_CONTENT_FILTER_ENABLED", default=True):
            # Drop crawl-failure docs (headline-only / BOM-only body) before
            # they pollute RRF and the evidence fed to the LLM. Applies to
            # already-indexed data too, so no re-index is needed to benefit.
            hit_by_id = {
                doc_id: hit
                for doc_id, hit in hit_by_id.items()
                if _has_usable_content(hit.source or {})
            }
            if not hit_by_id:
                return []

        bm25_ranks = {hit.id: rank for rank, hit in enumerate(bm25_hits)}
        vector_ranks = {hit.id: rank for rank, hit in enumerate(vector_hits)}
        missing_rank = max(search_pool_k, len(hit_by_id))

        # Bước 2: Reciprocal Rank Fusion (cùng công thức với retriever lúc training).
        t_rrf0 = perf_counter()
        rrf_scores = {}
        bm25_weight = 1.35 if verbatim_query else 1.0
        vector_weight = 1.0
        for doc_id in hit_by_id:
            bm25_rank = bm25_ranks.get(doc_id, missing_rank)
            vector_rank = vector_ranks.get(doc_id, missing_rank)
            rrf_scores[doc_id] = (bm25_weight / (self.rrf_k + bm25_rank)) + (
                vector_weight / (self.rrf_k + vector_rank)
            )

        rrf_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        rrf_candidates = rrf_ranked[:effective_rrf_top_k]
        if not rrf_candidates:
            return []

        max_rrf = max(score for _, score in rrf_candidates)
        max_rrf = max(max_rrf, 1e-8)
        rrf_scores_norm = {doc_id: score / max_rrf for doc_id, score in rrf_candidates}

        # Dựng lịch sử thời gian theo từng nhóm để phục vụ chấm điểm thời gian.
        docs_by_group: Dict[str, List[datetime]] = {}
        for hit in hit_by_id.values():
            source = hit.source or {}
            group = self._doc_group_key(source)
            docs_by_group.setdefault(group, []).append(self._doc_timestamp(source))

        final_items = []
        for doc_id, _ in rrf_candidates:
            hit = hit_by_id[doc_id]
            source = hit.source or {}
            timestamp = self._doc_timestamp(source)
            group = self._doc_group_key(source)

            if use_temporal:
                temporal, recency, cyclicity = (
                    self.temporal_scorer.calculate_temporal_score(
                        timestamp,
                        docs_by_group.get(group, []),
                        use_adaptive_lambda=True,
                    )
                )
                final_score = (
                    self.alpha * rrf_scores_norm[doc_id] + (1.0 - self.alpha) * temporal
                )
            else:
                recency, cyclicity = 0.5, 0.5
                final_score = rrf_scores_norm[doc_id]

            final_items.append(
                RetrievalResult(
                    document_id=doc_id,
                    text=_select_doc_text(source),
                    score=float(final_score),
                    rrf_score=float(rrf_scores_norm[doc_id]),
                    recency_score=float(recency),
                    cyclicity_score=float(cyclicity),
                    cosine_similarity=float(vector_cosine.get(doc_id, 0.0)),
                    timestamp=timestamp,
                    metadata=source,
                )
            )
        t_rrf1 = perf_counter()

        if verbatim_query:
            query_lower = normalized_query.lower()
            for item in final_items:
                doc_text = _normalize_query_text(item.text)
                doc_lower = doc_text.lower()
                if query_lower and len(query_lower) >= 40 and query_lower in doc_lower:
                    item.score += 0.35
                else:
                    item.score += 0.15 * _token_overlap_ratio(query_lower, doc_lower)

        final_items.sort(key=lambda x: x.score, reverse=True)

        # top_k quy định feature dạng số kích thước cố định của nhánh retrieval
        # (model được train với đúng self.top_k hàng, được zero-pad bởi
        # _build_retrieval_features_train_compatible nếu trả về ít hơn — xem hàm
        # đó). evidence_top_k là giới hạn độc lập, nhỏ-hơn-hoặc-bằng, cho số
        # evidence *text* (đã dedup/full-article) thực sự được giữ — nó không bao
        # giờ được mở cửa sổ rộng quá top_k.
        dedup_window = top_k
        if evidence_top_k is not None:
            dedup_window = min(top_k, max(1, int(evidence_top_k)))

        t_fullart0 = perf_counter()
        if _env_flag("FUSION_FULL_ARTICLE_ENABLED", default=True):
            final_items = self._dedupe_and_fetch_full_articles(
                final_items,
                dedup_window,
                claim=normalized_query,
                date_window=_claim_date_window(min_timestamp, max_timestamp),
            )
        else:
            final_items = final_items[:dedup_window]
        t_fullart1 = perf_counter()

        if debug:
            t_total = 1000.0 * (perf_counter() - t_retrieve_start)
            logger.info(
                f"[fusion_inference] retrieve_timing"
                f" | pool_ms={1000.0*(t_pool1-t_pool0):.1f}"
                f" | expand_ms={1000.0*(t_expand1-t_expand0):.1f}"
                f" | bm25_ms={1000.0*(t_bm25_1-t_bm25_0):.1f}"
                f" | encode_ms={t_encode_ms:.1f}"
                f" | vector_ms={t_vector_ms:.1f}"
                f" | rrf_temporal_ms={1000.0*(t_rrf1-t_rrf0):.1f}"
                f" | full_article_ms={1000.0*(t_fullart1-t_fullart0):.1f}"
                f" | n_results={len(final_items)}"
                f" | total_ms={t_total:.1f}"
            )

        return final_items


@dataclass
class ClaimPrediction:
    """Kết quả dự đoán cho một claim: verdict, nhãn, độ tin cậy, evidence và nguồn."""
    claim: str
    verdict: str  # "Đúng" | "Sai"
    label: str  # Token nhãn của model, ví dụ "Đúng" | "Sai"
    label_id: int
    confidence: float
    evidence: List[str]
    source_links: List[str]
    timing_ms: Optional[Dict[str, float]] = None
    label_probs: Optional[Dict[str, float]] = None


class FusionClaimVerifier:
    """
    Bộ trợ giúp inference cho từng claim:
      claim -> lấy evidence -> logits LLM -> fusion -> verdict Đúng/Sai.
    """

    def __init__(
        self,
        fusion_model_path: str,
        opensearch_index: Optional[str] = None,
        llm_model_path: Optional[str] = None,
        retriever_model_path: Optional[str] = None,
        device: Optional[str] = None,
        alpha: float = 0.7,
        lambda_decay: float = 0.1,
        gamma: float = 0.5,
        rrf_k: int = 60,
        llm_evidence_top_k: Optional[int] = None,
        debug: Optional[bool] = None,
        log_evidence_chars: int = 240,
    ):
        """Nạp checkpoint fusion, các nhánh model (retrieval encoder + fusion),
        retriever OpenSearch, LLM scorer và cấu hình các guard NLI/grounding."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for fusion inference.")
        if ConfidenceAwareFusion is None or RetrievalFeatureEncoder is None:
            raise ImportError(
                "Fusion PyTorch modules are unavailable. Ensure torch is installed."
            )

        requested = (device or "cpu").strip().lower()
        if requested != "cpu" and not torch.cuda.is_available():
            logger.warning(
                f"[fusion_inference] device='{requested}' requested but CUDA not available, falling back to 'cpu'."
            )
            requested = "cpu"
        self.device = requested
        self.checkpoint = torch.load(
            fusion_model_path,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        self.saved_config = self.checkpoint.get("config", {})
        fusion_state = self.checkpoint.get("fusion", {})
        has_adaptive_state = any(
            str(k).startswith("beta_gate.") for k in fusion_state.keys()
        )
        saved_adaptive = bool(
            self.saved_config.get("adaptive_beta", has_adaptive_state)
        )
        if saved_adaptive != has_adaptive_state:
            logger.warning(
                "[fusion_inference] adaptive_beta mismatch between saved_config "
                f"({saved_adaptive}) and checkpoint fusion state ({has_adaptive_state}). "
                "Using fusion state architecture for safe loading."
            )
        effective_adaptive_beta = has_adaptive_state

        self.top_k = int(self.saved_config.get("top_k", 10))
        if llm_evidence_top_k is None:
            llm_evidence_top_k = int(os.getenv("FUSION_LLM_EVIDENCE_TOP_K", "10"))
        self.llm_evidence_top_k = max(1, int(llm_evidence_top_k))
        llm_batch_env = os.getenv("LLM_INFER_BATCH_SIZE") or os.getenv("FUSION_LLM_INFER_BATCH_SIZE", "1")
        try:
            llm_infer_batch_size = int(llm_batch_env)
        except ValueError:
            llm_infer_batch_size = 1
            logger.warning(
                f"[fusion_inference] Invalid LLM_INFER_BATCH_SIZE={llm_batch_env!r}, fallback to 1"
            )
        self.llm_infer_batch_size = max(1, llm_infer_batch_size)
        self.label_list = list(self.saved_config.get("label_list", LABEL_LIST))
        log_all = _env_flag("FUSION_INFERENCE_LOG_ALL", default=False)
        self.debug = (
            _env_flag("FUSION_INFERENCE_DEBUG", default=False)
            if debug is None
            else bool(debug)
        )
        if log_all:
            self.debug = True
        self.log_evidence_chars = int(
            os.getenv("FUSION_INFERENCE_LOG_EVIDENCE_CHARS", str(log_evidence_chars))
        )
        self.log_full_evidence = _env_flag(
            "FUSION_INFERENCE_LOG_FULL_EVIDENCE", default=False
        )
        if log_all:
            self.log_full_evidence = True
            # 0 means "no truncate" for _truncate()
            self.log_evidence_chars = 0
        if not self.debug:
            logger.debug(
                "Fusion inference debug is OFF. Enable with env FUSION_INFERENCE_DEBUG=1 "
                "or pass debug=True to verify_claim_true_false()/FusionClaimVerifier."
            )

        self.score_features = int(self.saved_config.get("score_features", 5))
        self.interaction_dim = int(self.saved_config.get("interaction_dim", 0))
        self.nli_model_name: Optional[str] = self.saved_config.get("nli_model") or None
        self._nli_scorer = None  # lazy-loaded on first predict call
        self._nli_lock = threading.Lock()
        # Số doc tối đa đưa vào NLI — đủ để lấy signal, không cần tất cả top_k.
        # Giảm từ top_k (10) xuống 5 cắt ~50% NLI time, accuracy giảm không đáng kể.
        self._nli_top_k = int(os.getenv("NLI_EVIDENCE_TOP_K", str(min(5, self.top_k))))
        self.nli_override_enabled = _env_flag("NLI_OVERRIDE_ENABLED", default=True)
        self.nli_override_threshold = float(os.getenv("NLI_OVERRIDE_THRESHOLD", "0.5"))
        # Grounding guard: một verdict Đúng/Sai tự tin phải được chống lưng bởi ít
        # nhất một evidence thực sự kéo theo (A) hoặc mâu thuẫn (B) với claim. Nếu
        # không có evidence nào như vậy thì verdict là vô căn cứ → hạ về C.
        self.grounding_guard_enabled = _env_flag("GROUNDING_GUARD_ENABLED", default=True)
        self.grounding_support_threshold = float(
            os.getenv("GROUNDING_SUPPORT_THRESHOLD", "0.45")
        )
        self.retrieval_encoder = RetrievalFeatureEncoder(
            num_retrieved=self.top_k,
            score_features=self.score_features,
            hidden_dim=64,
            output_dim=64,
            interaction_dim=self.interaction_dim,
        ).to(self.device)

        self.fusion = ConfidenceAwareFusion(
            retrieval_input_dim=64,
            hidden_dim=128,
            num_classes=int(self.saved_config.get("num_classes", len(self.label_list))),
            initial_beta=float(self.saved_config.get("initial_beta", 0.8)),
            lambda_reg=float(self.saved_config.get("lambda_reg", 0.01)),
            normalize_branch_logits=bool(
                self.saved_config.get("normalize_branch_logits", False)
            ),
            adaptive_beta=effective_adaptive_beta,
        ).to(self.device)

        self.retrieval_encoder.load_state_dict(self.checkpoint["retrieval_encoder"])
        self.fusion.load_state_dict(self.checkpoint["fusion"])
        self.retrieval_encoder.eval()
        self.fusion.eval()

        # torch.compile tăng tốc ~20% cho fusion/retrieval_encoder (model nhỏ,
        # compile nhanh). Chỉ bật trên GPU vì overhead compile trên CPU không đáng.
        if self.device != "cpu" and hasattr(torch, "compile"):
            try:
                self.retrieval_encoder = torch.compile(self.retrieval_encoder)
                self.fusion = torch.compile(self.fusion)
                logger.info("[fusion_inference] torch.compile enabled for fusion models.")
            except Exception as exc:
                logger.warning(f"[fusion_inference] torch.compile failed (non-fatal): {exc}")

        retriever_model = retriever_model_path or self.saved_config.get(
            "retriever_model", "bge-vi-base"
        )

        index_name = (
            opensearch_index
            or os.getenv("OPENSEARCH_INDEX_NAME")
            or os.getenv("OP_KB_NAME")
        )
        if not index_name:
            raise ValueError(
                "Missing OpenSearch index name. Set OPENSEARCH_INDEX_NAME/OP_KB_NAME or pass opensearch_index."
            )

        kb = OpenSearchKB(index_name=index_name, embedding_dim=768)
        self.retriever = OpenSearchHybridRetriever(
            kb=kb,
            embedding_model=retriever_model,
            alpha=alpha,
            lambda_decay=lambda_decay,
            gamma=gamma,
            use_query_expansion=True,
            rrf_k=rrf_k,
            device=self.device,
        )
        
        model_name = llm_model_path or self.saved_config.get("model_name")
        if not model_name:
            raise ValueError(
                "Missing LLM path. Provide llm_model_path or save model_name in fusion checkpoint."
            )

        from src.llm_scorer import LLMScorer

        self.llm = LLMScorer(
            model_name=model_name,
            device=self.device,
            max_length=int(os.getenv("LLM_MAX_LENGTH", "6144")),
            labels=self.label_list,
            prompt_template=PROMPT_TEMPLATE,
        )

        self._log_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------
    def warmup(self) -> None:
        """Kích hoạt việc biên dịch Triton (lazy) của torch.compile ngay lúc khởi động.
        """
        if self.device == "cpu":
            return

        logger.info(
            "[fusion_inference] warmup: triggering torch.compile kernel compilation"
            " — may take a few minutes on first boot …"
        )
        t0 = perf_counter()
        with torch.inference_mode():
            dummy_retrieval = torch.zeros(
                1, self.top_k, self.score_features, dtype=torch.float32, device=self.device
            )
            dummy_encoded = self.retrieval_encoder(dummy_retrieval)
            dummy_llm = torch.zeros(
                1, len(self.label_list), dtype=torch.float32, device=self.device
            )
            self.fusion(dummy_llm, dummy_encoded)

            dummy_ids = torch.zeros(1, 16, dtype=torch.long, device=self.device)
            dummy_mask = torch.ones(1, 16, dtype=torch.long, device=self.device)
            self.llm.model(input_ids=dummy_ids, attention_mask=dummy_mask)

        logger.info(
            f"[fusion_inference] warmup done | elapsed_ms={1000.0 * (perf_counter() - t0):.0f}"
        )

    # ------------------------------------------------------------------
    # NLI helpers
    # ------------------------------------------------------------------
    def _ensure_nli_loaded(self) -> None:
        """Nạp NLIScorer một lần duy nhất (lazy, thread-safe) khi cần dùng lần đầu."""
        if self._nli_scorer is not None or not self.nli_model_name:
            return
        with self._nli_lock:
            if self._nli_scorer is not None:
                return
            from src.models.nli_scorer import NLIScorer
            # NLI_MAX_LENGTH: độ phức tạp attention là O(n²) — 256 vs 512 = ít tính toán hơn 4 lần.
            # Evidence liên quan tới fact-checking hầu như luôn nằm trong ~200 token đầu.
            nli_max_length = int(os.getenv("NLI_MAX_LENGTH", "256"))
            # NLI_DEVICE mặc định là CPU để không tranh VRAM với LLM.
            # Trên GPU 6 GB, riêng LLM đã cần ~3 GB + activations; thêm DeBERTa sẽ gây OOM.
            nli_device = os.getenv("NLI_DEVICE", "cpu")
            self._nli_scorer = NLIScorer(
                model_name=self.nli_model_name,
                device=nli_device,
                max_length=nli_max_length,
            ).load()

    def _nli_for_claim(
        self, claim: str, results: List[RetrievalResult]
    ) -> Optional[Tuple[np.ndarray, int]]:
        """Trả về (NLI features [top_k, 3], số_evidence_thật) hoặc None nếu NLI không khả dụng."""
        if not self.nli_model_name:
            return None
        self._ensure_nli_loaded()
        real_docs = [r.text for r in results[: self._nli_top_k] if r.text]
        nli_padded = np.full((self.top_k, 3), 1.0 / 3.0, dtype=np.float32)
        if real_docs:
            scores = self._nli_scorer.score(
                premises=real_docs, hypotheses=[claim] * len(real_docs)
            )
            nli_padded[: len(real_docs)] = scores
        return nli_padded, len(real_docs)

    def _nli_for_batch(
        self, claims: List[str], results_list: List[List[RetrievalResult]]
    ) -> List[Optional[Tuple[np.ndarray, int]]]:
        """Trả về danh sách (NLI features [top_k, 3], số_evidence_thật), mỗi phần tử cho một claim."""
        if not self.nli_model_name:
            return [None] * len(claims)
        self._ensure_nli_loaded()

        flat_docs: List[str] = []
        flat_claims: List[str] = []
        offsets: List[tuple] = []
        for claim, results in zip(claims, results_list):
            real_docs = [r.text for r in results[: self._nli_top_k] if r.text]
            offsets.append((len(flat_docs), len(real_docs)))
            flat_docs.extend(real_docs)
            flat_claims.extend([claim] * len(real_docs))

        if not flat_docs:
            return [
                (np.full((self.top_k, 3), 1.0 / 3.0, dtype=np.float32), 0)
                for _ in claims
            ]

        nli_flat = self._nli_scorer.score(premises=flat_docs, hypotheses=flat_claims)
        out = []
        for start, n_real in offsets:
            nli_padded = np.full((self.top_k, 3), 1.0 / 3.0, dtype=np.float32)
            if n_real > 0:
                nli_padded[:n_real] = nli_flat[start : start + n_real]
            out.append((nli_padded, n_real))
        return out

    def _apply_nli_conflict_guard(
        self,
        pred_id: int,
        probs: "torch.Tensor",
        nli_feats: Optional[np.ndarray],
        n_real_evidence: int,
    ) -> Tuple[int, float, bool]:
        """Hạ verdict về 'Chưa chắc chắn' khi verdict của fusion bị đa số evidence
        (được chấm điểm độc lập) mâu thuẫn.
        """
        confidence = float(probs[pred_id].item())
        if (
            not self.nli_override_enabled
            or nli_feats is None
            or pred_id == 2
            or n_real_evidence == 0
        ):
            return pred_id, confidence, False

        target_label = 2 if pred_id == 0 else 0  # cần mâu thuẫn với Đúng, kéo theo với Sai
        real_nli = nli_feats[:n_real_evidence]
        argmaxes = np.argmax(real_nli, axis=-1)
        top_confidences = real_nli[np.arange(len(real_nli)), argmaxes]
        strong = top_confidences >= self.nli_override_threshold
        n_strong = int(strong.sum())
        if n_strong == 0:
            return pred_id, confidence, False

        agree = int((argmaxes[strong] == target_label).sum())
        min_votes = 2 if n_real_evidence >= 2 else 1
        if agree >= min_votes and agree > n_strong / 2:
            return 2, float(probs[2].item()), True
        return pred_id, confidence, False

    def _apply_evidence_grounding_guard(
        self,
        pred_id: int,
        probs: "torch.Tensor",
        nli_feats: Optional[np.ndarray],
        n_real_evidence: int,
    ) -> Tuple[int, float, bool]:
        """Hạ một verdict Đúng/Sai tự tin về 'Chưa chắc chắn' khi KHÔNG evidence
        nào trong số được lấy về cung cấp tín hiệu NLI mà verdict đó cần —
        kéo theo (entailment) cho Đúng (cột 0), mâu thuẫn (contradiction) cho Sai (cột 2).
        """
        confidence = float(probs[pred_id].item())
        if (
            not self.grounding_guard_enabled
            or nli_feats is None
            or pred_id == 2
            or n_real_evidence == 0
        ):
            return pred_id, confidence, False

        support_col = 0 if pred_id == 0 else 2  # kéo theo cho A, mâu thuẫn cho B
        real_nli = nli_feats[:n_real_evidence]
        max_support = float(real_nli[:, support_col].max())
        if max_support < self.grounding_support_threshold:
            return 2, float(probs[2].item()), True
        return pred_id, confidence, False

    # ------------------------------------------------------------------
    # Claims index
    # ------------------------------------------------------------------
    @property
    def _claims_kb(self) -> "OpenSearchKB":
        """OpenSearchKB (khởi tạo lazy) trỏ tới index 'claims'."""
        if not hasattr(self, "_claims_kb_instance"):
            claims_index = os.getenv("OP_CLAIMS_INDEX", "claims")
            self._claims_kb_instance = OpenSearchKB(
                index_name=claims_index,
                embedding_dim=self.retriever.embedding_dim,
            )
        return self._claims_kb_instance

    def _log_claims_to_opensearch(self, claims_data: List[Dict]) -> None:
        """
        Lưu danh sách claims vào OpenSearch index 'claims' (upsert).

        Mỗi doc cần có _id; ở đây dùng SHA-1 của claim text để claim
        giống nhau sẽ upsert thay vì duplicate.

        claims_data: list of dict với ít nhất key 'claim' và các meta
                     tuỳ ý (verdict, confidence, checked_at, …).
        """
        if not claims_data:
            return

        def _make_id(text: str) -> str:
            return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()

        docs = []
        for item in claims_data:
            doc = dict(item)
            if "_id" not in doc and "id" not in doc:
                doc["_id"] = _make_id(str(doc.get("claim", "")))
            docs.append(doc)

        try:
            result = self._claims_kb.insert_many(docs, upsert=True)
            if self.debug:
                logger.info(
                    f"[fusion_inference] claims_logged | inserted={result.get('inserted')} | errors={result.get('errors')}"
                )
        except Exception as exc:
            logger.warning(
                f"[fusion_inference] Failed to log claims to OpenSearch: {exc}"
            )

    # ------------------------------------------------------------------
    # Claim splitting
    # ------------------------------------------------------------------
    def split_long_claim(self, claim: str) -> Optional[List[str]]:
        """Viết lại/tách claim dài khi bật REWRITE_CLAIM; trả về None nếu tắt."""
        if not _env_flag("REWRITE_CLAIM", default=False):
            return None
        rewritten = rewrite_claim(claim)
        return [rewritten]

    def _prepare_sub_claims(self, claim: str) -> List[str]:
        """
        Chuẩn hoá output từ splitter và fallback về claim gốc nếu splitter
        chưa được implement hoặc trả về dữ liệu không hợp lệ.
        """
        text = str(claim).strip()
        if not text:
            return []

        try:
            raw_sub_claims = self.split_long_claim(text)
        except Exception as exc:
            logger.warning(
                f"[fusion_inference] split_long_claim failed, fallback to original claim: {exc}"
            )
            return [text]

        if raw_sub_claims is None:
            return [text]

        if isinstance(raw_sub_claims, str):
            candidates = [raw_sub_claims]
        else:
            try:
                candidates = list(raw_sub_claims)
            except TypeError:
                candidates = [str(raw_sub_claims)]

        normalized: List[str] = []
        seen = set()
        for item in candidates:
            part = str(item).strip()
            if not part or part in seen:
                continue
            seen.add(part)
            normalized.append(part)

        # Cap số sub-claims để tránh OOM khi claim rất dài và splitter trả về nhiều items.
        _max_sub = int(os.getenv("MAX_SUB_CLAIMS", "5"))
        if len(normalized) > _max_sub:
            logger.warning(
                f"[fusion_inference] sub-claims capped {len(normalized)} -> {_max_sub} (MAX_SUB_CLAIMS={_max_sub})"
            )
            normalized = normalized[:_max_sub]

        return normalized or [text]

    def _aggregate_sub_claim_predictions(
        self, claim: str, sub_preds: List[ClaimPrediction], sub_claim_count: int
    ) -> ClaimPrediction:
        """Gộp dự đoán của các sub-claim thành một verdict cho claim gốc.

        Quy tắc: chỉ cần một sub-claim 'Sai' thì tổng thể là 'Sai'; nếu không có
        'Sai' nhưng có 'Chưa chắc chắn' thì là 'Chưa chắc chắn'; còn lại là 'Đúng'.
        Độ tin cậy lấy trung bình, evidence và source link được gộp lại.
        """
        has_sai = False
        has_chua_chac_chan = False
        all_evidence: List[str] = []
        all_source_links: List[str] = []

        avg_conf = (
            sum(p.confidence for p in sub_preds) / len(sub_preds) if sub_preds else 0.0
        )
        avg_label_probs: Optional[Dict[str, float]] = None
        probs_sources = [p.label_probs for p in sub_preds if p.label_probs]
        if probs_sources:
            avg_label_probs = {
                label: sum(probs.get(label, 0.0) for probs in probs_sources) / len(probs_sources)
                for label in VERDICT_ORDER
            }

        for p in sub_preds:
            if p.verdict == "Sai":
                has_sai = True
            elif p.verdict == "Chưa chắc chắn":
                has_chua_chac_chan = True

            all_evidence.extend(p.evidence)
            for link in p.source_links:
                if link not in all_source_links:
                    all_source_links.append(link)

        if has_sai:
            final_verdict = "Sai"
            final_label = "Sai"
            final_label_id = (
                self.label_list.index("Sai") if "Sai" in self.label_list else 1
            )
        elif has_chua_chac_chan:
            final_verdict = "Chưa chắc chắn"
            final_label = "Chưa chắc chắn"
            final_label_id = (
                self.label_list.index("Chưa chắc chắn")
                if "Chưa chắc chắn" in self.label_list
                else 2
            )
        else:
            final_verdict = "Đúng"
            final_label = "Đúng"
            final_label_id = (
                self.label_list.index("Đúng") if "Đúng" in self.label_list else 0
            )

        max_evidence = self.llm_evidence_top_k * max(1, int(sub_claim_count))
        return ClaimPrediction(
            claim=claim,
            verdict=final_verdict,
            label=final_label,
            label_id=final_label_id,
            confidence=avg_conf,
            evidence=all_evidence[:max_evidence],
            source_links=all_source_links,
            label_probs=avg_label_probs,
        )

    def _log_predictions_to_claims_index(
        self, predictions: List[ClaimPrediction]
    ) -> None:
        """Ghi (bất đồng bộ) danh sách dự đoán vào index 'claims' của OpenSearch."""
        if not predictions:
            return

        self._log_executor.submit(
            self._log_claims_to_opensearch,
            [
                {
                    "claim": p.claim,
                    "verdict": p.verdict,
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                }
                for p in predictions
            ],
        )

    def _predict_batch_without_split(self, claims: List[str]) -> List[ClaimPrediction]:
        """Dự đoán theo lô cho danh sách claim (không tách sub-claim).

        Chuẩn hoá ngày trong claim, retrieve evidence, chạy NLI + LLM + fusion
        theo batch rồi áp các guard NLI/grounding để ra verdict cuối cho từng claim.
        """
        t0 = perf_counter()
        if not claims:
            return []

        now_utc = datetime.now(timezone.utc)
        if self.debug:
            logger.info(
                f"[fusion_inference] start _predict_batch_without_split | batch_size={len(claims)} | llm_infer_batch_size={self.llm_infer_batch_size} | now_utc={now_utc.isoformat()}"
            )

        valid_items: List[tuple[int, str]] = []
        results: List[Optional[ClaimPrediction]] = [None] * len(claims)

        for i, claim in enumerate(claims):
            text = str(claim).strip()
            if not text:
                continue
            valid_items.append((i, text))

        if not valid_items:
            return []

        total_llm_ms = 0.0

        with torch.inference_mode():
            for start in range(0, len(valid_items), self.llm_infer_batch_size):
                batch = valid_items[start : start + self.llm_infer_batch_size]
                valid_indices = []
                valid_claims = []
                all_retrieval_features = []
                all_llm_evidences = []
                all_source_links = []

                # Chèn ngày hôm nay vào các claim không có ngày cụ thể.
                _today_str = now_utc.strftime("%d/%m/%Y")
                _yesterday_str = (now_utc - timedelta(days=1)).strftime("%d/%m/%Y")
                _day_before_str = (now_utc - timedelta(days=2)).strftime("%d/%m/%Y")
                _tomorrow_str = (now_utc + timedelta(days=1)).strftime("%d/%m/%Y")
                _date_pat = re.compile(r"\d{1,2}[/\-]\d{1,2}|tháng\s+\d", re.IGNORECASE)
                
                new_batch = []
                batch_date_ranges = {}
                batch_cleaned_texts = {}
                for idx, text in batch:
                    text = re.sub(r'(?i)(?:ngày\s+)?\bhôm nay\b', f'ngày {_today_str}', text)
                    text = re.sub(r'(?i)(?:ngày\s+)?\bhôm qua\b', f'ngày {_yesterday_str}', text)
                    text = re.sub(r'(?i)(?:ngày\s+)?\bhôm kia\b', f'ngày {_day_before_str}', text)
                    text = re.sub(r'(?i)(?:ngày\s+)?\bngày mai\b', f'ngày {_tomorrow_str}', text)
                    
                    min_ts, max_ts, cleaned_text = extract_date_range(text)
                    batch_date_ranges[idx] = (min_ts, max_ts)
                    batch_cleaned_texts[idx] = cleaned_text
                    
                    if not _date_pat.search(text):
                        text = f"{text} (ngày {_today_str})"
                    new_batch.append((idx, text))
                batch = new_batch

                # Encode trước tất cả query trong một lần gọi SentenceTransformer
                batch_texts_list = [batch_cleaned_texts[idx] for idx, _ in batch]
                for idx, text in batch:
                    logger.info(f"[fusion_inference] final_claim_to_llm (batch) | idx={idx} | claim={text!r}")
                t_benc0 = perf_counter()
                batch_vectors = self.retriever.batch_encode(batch_texts_list)
                t_benc1 = perf_counter()
                if self.debug:
                    logger.info(
                        f"[fusion_inference] batch_encode | n={len(batch_texts_list)}"
                        f" | elapsed_ms={1000.0*(t_benc1-t_benc0):.1f}"
                    )
                vec_map: dict[int, List[float]] = {i: vec for i, vec in enumerate(batch_vectors)}

                all_retrieval_interactions = []

                def _retrieve_one(
                    item: Tuple[int, str],
                    batch_pos: int,
                ) -> Tuple[int, str, Any, Any, List[str], List[str], List[RetrievalResult]]:
                    """Retrieve evidence cho một claim và dựng feature + danh sách source link."""
                    _idx, _text = item
                    _t0 = perf_counter()
                    min_ts, max_ts = batch_date_ranges.get(_idx, (None, None))
                    cleaned_text = batch_cleaned_texts.get(_idx, _text)
                    _feat, _interaction, _evidence, _results = _build_retrieval_features_train_compatible(
                        self.retriever, cleaned_text, self.top_k,
                        precomputed_vector=vec_map.get(batch_pos),
                        score_features=self.score_features,
                        min_timestamp=min_ts,
                        max_timestamp=max_ts,
                        evidence_top_k=self.llm_evidence_top_k,
                    )
                    if self.debug:
                        logger.info(
                            f"[fusion_inference] retrieve_one | idx={_idx}"
                            f" | elapsed_ms={1000.0*(perf_counter()-_t0):.1f}"
                        )
                    _links: List[str] = []
                    for _r in _results[: self.llm_evidence_top_k]:
                        _meta = _r.metadata or {}
                        _url = _extract_url(_meta)
                        if _url and _url not in _links:
                            _links.append(_url)
                    return _idx, _text, _feat, _interaction, _evidence[: self.llm_evidence_top_k], _links, _results

                t_retrieve_all0 = perf_counter()
                retrieved: dict[int, Tuple] = {}
                for batch_pos, item in enumerate(batch):
                    r_idx, r_text, r_feat, r_int, r_ev, r_links, r_results = _retrieve_one(item, batch_pos)
                    retrieved[r_idx] = (r_text, r_feat, r_int, r_ev, r_links, r_results)
                t_retrieve_all1 = perf_counter()
                if self.debug:
                    logger.info(
                        f"[fusion_inference] retrieve_all | n={len(batch)}"
                        f" | total_ms={1000.0*(t_retrieve_all1-t_retrieve_all0):.1f}"
                        f" | avg_ms={1000.0*(t_retrieve_all1-t_retrieve_all0)/max(len(batch),1):.1f}"
                    )

                all_retrieval_results_list: List[List[RetrievalResult]] = []
                for idx, text in batch:
                    valid_indices.append(idx)
                    valid_claims.append(text)
                    _, feat, interaction, ev, links, r_results = retrieved[idx]
                    all_retrieval_features.append(feat)
                    all_retrieval_interactions.append(interaction)
                    all_llm_evidences.append(ev)
                    all_source_links.append(links)
                    all_retrieval_results_list.append(r_results)

                # Gộp NLI features vào score features nếu checkpoint được train kèm NLI
                t_nli0 = perf_counter()
                nli_list: Optional[List[Tuple[np.ndarray, int]]] = None
                if self.nli_model_name:
                    nli_list = self._nli_for_batch(valid_claims, all_retrieval_results_list)
                    all_retrieval_features = [
                        np.concatenate([f, n[0]], axis=-1)
                        for f, n in zip(all_retrieval_features, nli_list)
                    ]
                t_nli1 = perf_counter()
                if self.debug and self.nli_model_name:
                    logger.info(
                        f"[fusion_inference] nli_batch | n={len(valid_claims)}"
                        f" | elapsed_ms={1000.0*(t_nli1-t_nli0):.1f}"
                    )

                retrieval_features = torch.tensor(
                    np.stack(all_retrieval_features),
                    dtype=torch.float32,
                    device=self.device,
                )
                interaction_tensor = None
                if all_retrieval_interactions and all_retrieval_interactions[0] is not None:
                    interaction_tensor = torch.tensor(
                        np.stack(all_retrieval_interactions), dtype=torch.float32, device=self.device
                    )

                t_llm0 = perf_counter()
                llm_logits = self.llm.score_logits(valid_claims, all_llm_evidences).to(
                    self.device
                )
                t_llm1 = perf_counter()
                total_llm_ms += 1000.0 * (t_llm1 - t_llm0)

                t_fusion0 = perf_counter()
                retrieval_encoded = self.retrieval_encoder(retrieval_features, interaction_tensor)
                fusion_output = self.fusion(llm_logits, retrieval_encoded)
                t_fusion1 = perf_counter()
                if self.debug:
                    logger.info(
                        f"[fusion_inference] fusion | elapsed_ms={1000.0*(t_fusion1-t_fusion0):.1f}"
                    )

                probs_batch = fusion_output.final_probs
                pred_ids = torch.argmax(probs_batch, dim=-1).cpu().tolist()
                confidences = torch.max(probs_batch, dim=-1)[0].cpu().tolist()

                for pos, (v_idx, text, llm_ev, links, pred_id, conf) in enumerate(zip(
                    valid_indices,
                    valid_claims,
                    all_llm_evidences,
                    all_source_links,
                    pred_ids,
                    confidences,
                )):
                    pred_id_int = int(pred_id)
                    nli_feats_i, nli_n_real_i = (
                        nli_list[pos] if nli_list is not None else (None, 0)
                    )
                    pred_id_int, conf, nli_overridden = self._apply_nli_conflict_guard(
                        pred_id_int, probs_batch[pos], nli_feats_i, nli_n_real_i
                    )
                    if nli_overridden:
                        logger.info(
                            "[fusion_inference] nli_conflict_guard_triggered (batch)"
                            f" | idx={v_idx} | fusion_pred_id={int(pred_id)}"
                            f" | n_real_evidence={nli_n_real_i}"
                            f" | nli_feats={np.array2string(nli_feats_i[:nli_n_real_i], precision=4)}"
                            " | downgraded_to=C (Chưa chắc chắn)"
                        )
                    pred_id_int, conf, grounding_overridden = self._apply_evidence_grounding_guard(
                        pred_id_int, probs_batch[pos], nli_feats_i, nli_n_real_i
                    )
                    if grounding_overridden:
                        logger.info(
                            "[fusion_inference] grounding_guard_triggered (batch)"
                            f" | idx={v_idx} | fusion_pred_id={int(pred_id)}"
                            f" | n_real_evidence={nli_n_real_i}"
                            f" | nli_feats={np.array2string(nli_feats_i[:nli_n_real_i], precision=4)}"
                            " | downgraded_to=C (Chưa chắc chắn) | reason=no_supporting_evidence"
                        )
                    pred_label = self.label_list[pred_id_int]
                    if pred_id_int == 0:
                        verdict = "Đúng"
                    elif pred_id_int == 1:
                        verdict = "Sai"
                    else:
                        verdict = "Chưa chắc chắn"
                    label_probs = {
                        label: float(probs_batch[pos][i].item())
                        for i, label in enumerate(VERDICT_ORDER)
                    }
                    results[v_idx] = ClaimPrediction(
                        claim=text,
                        verdict=verdict,
                        label=pred_label,
                        label_id=pred_id_int,
                        confidence=float(conf),
                        evidence=llm_ev,
                        source_links=links,
                        label_probs=label_probs,
                    )

                del retrieval_features
                del llm_logits
                del retrieval_encoded
                del fusion_output
                del probs_batch
                gc.collect()

        batch_elapsed_ms = 1000.0 * (perf_counter() - t0)
        self._last_batch_timing = {
            "llm_ms": round(total_llm_ms, 1),
            "batch_ms": round(batch_elapsed_ms, 1),
        }

        if self.debug:
            logger.info(
                f"[fusion_inference] _predict_batch_without_split done | llm_elapsed_ms={total_llm_ms:.2f} | elapsed_ms={batch_elapsed_ms:.2f}"
            )

        return [r for r in results if r is not None]

    def predict(self, claim: str) -> ClaimPrediction:
        """Dự đoán verdict cho một claim đơn.

        Nếu claim được tách thành nhiều sub-claim thì dự đoán từng cái rồi gộp
        lại; ngược lại chạy trực tiếp pipeline retrieve → NLI → LLM → fusion và
        áp guard, trả về ClaimPrediction kèm thông tin timing.
        """
        t0 = perf_counter()
        text = str(claim).strip()
        if not text:
            raise ValueError("Claim is empty.")

        t_split0 = perf_counter()
        sub_claims = self._prepare_sub_claims(text)
        t_split1 = perf_counter()
        if self.debug:
            logger.info(
                f"[fusion_inference] prepare_sub_claims | n_sub={len(sub_claims)}"
                f" | elapsed_ms={1000.0*(t_split1-t_split0):.1f}"
            )

        if len(sub_claims) > 1:
            if self.debug:
                logger.info(
                    f"[fusion_inference] splitting claim into {len(sub_claims)} sub-claims"
                )
            sub_preds = self._predict_batch_without_split(sub_claims)
            aggregated = self._aggregate_sub_claim_predictions(
                text, sub_preds, sub_claim_count=len(sub_claims)
            )

            total_ms = 1000.0 * (perf_counter() - t0)
            batch_timing = getattr(self, "_last_batch_timing", {})
            aggregated.timing_ms = {
                "split_ms": round(1000.0 * (t_split1 - t_split0), 1),
                "llm_ms": batch_timing.get("llm_ms", 0.0),
                "batch_ms": batch_timing.get("batch_ms", 0.0),
                "n_sub_claims": len(sub_claims),
                "total_ms": round(total_ms, 1),
            }

            if self.debug:
                logger.info(
                    f"[fusion_inference] done predict (aggregated) | verdict={aggregated.verdict!r} | confidence={aggregated.confidence:.6f} | elapsed_ms={total_ms:.2f}"
                    f" | split_ms={aggregated.timing_ms['split_ms']}"
                    f" | llm_ms={aggregated.timing_ms['llm_ms']}"
                )

            self._log_predictions_to_claims_index([aggregated])
            return aggregated

        model_text = sub_claims[0]

        now_utc = datetime.now(timezone.utc)
        if self.debug:
            logger.info(
                f"[fusion_inference] start predict | now_utc={now_utc.isoformat()} | top_k={self.top_k} | llm_evidence_top_k={self.llm_evidence_top_k} | labels={self.label_list}"
            )
            logger.info(f"[fusion_inference] claim_input={model_text!r}")
            if model_text != text:
                logger.info(f"[fusion_inference] original_claim={text!r}")

        today_str = now_utc.strftime("%d/%m/%Y")
        yesterday_str = (now_utc - timedelta(days=1)).strftime("%d/%m/%Y")
        day_before_str = (now_utc - timedelta(days=2)).strftime("%d/%m/%Y")
        tomorrow_str = (now_utc + timedelta(days=1)).strftime("%d/%m/%Y")

        model_text = re.sub(r'(?i)(?:ngày\s+)?\bhôm nay\b', f'ngày {today_str}', model_text)
        model_text = re.sub(r'(?i)(?:ngày\s+)?\bhôm qua\b', f'ngày {yesterday_str}', model_text)
        model_text = re.sub(r'(?i)(?:ngày\s+)?\bhôm kia\b', f'ngày {day_before_str}', model_text)
        model_text = re.sub(r'(?i)(?:ngày\s+)?\bngày mai\b', f'ngày {tomorrow_str}', model_text)

        min_ts, max_ts, search_text = extract_date_range(model_text)

        has_date = bool(re.search(r"\d{1,2}[/\-]\d{1,2}", model_text) or
                        re.search(r"tháng\s+\d", model_text, re.IGNORECASE))
        if not has_date:
            model_text = f"{model_text} (ngày {today_str})"
            if self.debug:
                logger.info(f"[fusion_inference] no date in claim, injected today: {today_str!r}")

        logger.info(f"[fusion_inference] final_claim_to_llm={model_text!r}")

        t_retrieval0 = perf_counter()
        retrieval_features_np, doc_emb_np, retrieved_evidence, retrieval_results = (
            _build_retrieval_features_train_compatible(
                self.retriever, search_text, self.top_k,
                score_features=self.score_features,
                min_timestamp=min_ts,
                max_timestamp=max_ts,
                evidence_top_k=self.llm_evidence_top_k,
            )
        )
        t_retrieval1 = perf_counter()

        seen_ev_urls: set = set()
        llm_evidence: list = []
        for ev_text, r in zip(retrieved_evidence, retrieval_results):
            meta = r.metadata or {}
            url = (
                meta.get("article_url")
                or meta.get("url")
                or meta.get("link")
                or ""
            )
            if url and url in seen_ev_urls:
                continue
            if url:
                seen_ev_urls.add(url)
            llm_evidence.append(ev_text)
            if len(llm_evidence) >= self.llm_evidence_top_k:
                break

        if self.debug:
            logger.info(
                f"[fusion_inference] retrieval_done | n_results={len(retrieval_results)} | elapsed_ms={1000.0 * (t_retrieval1 - t_retrieval0):.2f}"
            )
            if retrieval_results:
                for idx, r in enumerate(retrieval_results, start=1):
                    ts = (
                        r.timestamp.astimezone(timezone.utc)
                        if isinstance(r.timestamp, datetime)
                        else _parse_timestamp(r.timestamp)
                    )
                    age_s = (now_utc - ts).total_seconds()
                    meta = r.metadata or {}
                    title = _truncate(str(meta.get("title") or ""), 120)
                    url = _truncate(_extract_url(meta), 200)
                    source_name = str(meta.get("source") or meta.get("type") or "")
                    logger.info(
                        "[fusion_inference] retrieved"
                        f" | rank={idx}"
                        f" | doc_id={r.document_id}"
                        f" | ts_utc={ts.isoformat()}"
                        f" | age_hours={age_s / 3600.0:.2f}"
                        f" | score={r.score:.6f}"
                        f" | rrf={r.rrf_score:.6f}"
                        f" | recency={r.recency_score:.6f}"
                        f" | cyclicity={r.cyclicity_score:.6f}"
                        f" | cosine={r.cosine_similarity:.6f}"
                        + (f" | source={source_name!r}" if source_name else "")
                        + (f" | title={title!r}" if title else "")
                        + (f" | url={url!r}" if url else "")
                    )

            logger.info(
                "[fusion_inference] retrieval_features_np shape="
                f"{retrieval_features_np.shape} | cols=[score, rrf, recency, cyclicity, cosine]"
            )
            logger.info(
                "[fusion_inference] retrieval_features_np="
                + np.array2string(
                    retrieval_features_np,
                    precision=6,
                    suppress_small=False,
                    separator=", ",
                )
            )

            if llm_evidence:
                logger.info(
                    f"[fusion_inference] llm_evidence_input | n_items={len(llm_evidence)} | selected_from={len(retrieved_evidence)} | log_full={self.log_full_evidence} | max_chars={self.log_evidence_chars}"
                )
                for idx, ev in enumerate(llm_evidence, start=1):
                    ev_text = str(ev)
                    if not self.log_full_evidence:
                        ev_text = _truncate(ev_text, self.log_evidence_chars)
                    logger.info(f"[fusion_inference] evidence[{idx}]={ev_text!r}")

        # Nối thêm NLI features nếu checkpoint được train kèm NLI
        t_nli0 = perf_counter()
        nli_feats: Optional[np.ndarray] = None
        nli_n_real = 0
        if self.nli_model_name:
            nli_result = self._nli_for_claim(model_text, retrieval_results)
            if nli_result is not None:
                nli_feats, nli_n_real = nli_result
                retrieval_features_np = np.concatenate(
                    [retrieval_features_np, nli_feats], axis=-1
                )
        t_nli1 = perf_counter()
        if self.debug and self.nli_model_name:
            logger.info(
                f"[fusion_inference] nli_single | elapsed_ms={1000.0*(t_nli1-t_nli0):.1f}"
            )

        retrieval_features = torch.tensor(
            retrieval_features_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.inference_mode():
            t_llm0 = perf_counter()
            llm_logits = self.llm.score_logits([model_text], [llm_evidence]).to(
                self.device
            )
            t_llm1 = perf_counter()

            if self.debug:
                llm_probs = torch.softmax(llm_logits, dim=-1)
                logger.info(
                    f"[fusion_inference] llm_done | logits_shape={tuple(llm_logits.shape)} | elapsed_ms={1000.0 * (t_llm1 - t_llm0):.2f}"
                )
                logger.info(
                    "[fusion_inference] llm_logits="
                    + np.array2string(
                        llm_logits.detach().cpu().float().numpy(),
                        precision=6,
                        suppress_small=False,
                        separator=", ",
                    )
                )
                logger.info(
                    "[fusion_inference] llm_probs="
                    + np.array2string(
                        llm_probs.detach().cpu().float().numpy(),
                        precision=6,
                        suppress_small=False,
                        separator=", ",
                    )
                )

            interaction_tensor = (
                torch.tensor(doc_emb_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                if doc_emb_np is not None else None
            )
            t_fusion0 = perf_counter()
            retrieval_encoded = self.retrieval_encoder(retrieval_features, interaction_tensor)
            if self.debug:
                enc = retrieval_encoded.detach().cpu()
                logger.info(
                    f"[fusion_inference] retrieval_encoder_out | shape={tuple(enc.shape)} | mean={enc.mean().item():.6f} | std={enc.std(unbiased=False).item():.6f}"
                )

            fusion_output = self.fusion(llm_logits, retrieval_encoded)
            t_fusion1 = perf_counter()
            probs = fusion_output.final_probs[0]
            pred_id = int(torch.argmax(probs).item())
            confidence = float(probs[pred_id].item())

            pred_id, confidence, nli_overridden = self._apply_nli_conflict_guard(
                pred_id, probs, nli_feats, nli_n_real
            )
            if nli_overridden:
                logger.info(
                    "[fusion_inference] nli_conflict_guard_triggered"
                    f" | fusion_pred_id={int(torch.argmax(fusion_output.final_probs[0]).item())}"
                    f" | n_real_evidence={nli_n_real}"
                    f" | nli_feats={np.array2string(nli_feats[:nli_n_real], precision=4)}"
                    " | downgraded_to=C (Chưa chắc chắn)"
                )

            pred_id, confidence, grounding_overridden = self._apply_evidence_grounding_guard(
                pred_id, probs, nli_feats, nli_n_real
            )
            if grounding_overridden:
                logger.info(
                    "[fusion_inference] grounding_guard_triggered"
                    f" | fusion_pred_id={int(torch.argmax(fusion_output.final_probs[0]).item())}"
                    f" | n_real_evidence={nli_n_real}"
                    f" | nli_feats={np.array2string(nli_feats[:nli_n_real], precision=4)}"
                    " | downgraded_to=C (Chưa chắc chắn) | reason=no_supporting_evidence"
                )

            if self.debug:
                logger.info(
                    f"[fusion_inference] fusion_done | lm_weight={fusion_output.lm_weight:.6f} | retrieval_weight={fusion_output.retrieval_weight:.6f}"
                    f" | elapsed_ms={1000.0*(t_fusion1-t_fusion0):.1f}"
                )
                logger.info(
                    "[fusion_inference] fused_logits="
                    + np.array2string(
                        fusion_output.fused_logits.detach().cpu().float().numpy(),
                        precision=6,
                        suppress_small=False,
                        separator=", ",
                    )
                )
                logger.info(
                    "[fusion_inference] final_probs="
                    + np.array2string(
                        fusion_output.final_probs.detach().cpu().float().numpy(),
                        precision=6,
                        suppress_small=False,
                        separator=", ",
                    )
                )

        pred_label = self.label_list[pred_id]
        if pred_id == 0:
            verdict = "Đúng"
        elif pred_id == 1:
            verdict = "Sai"
        else:
            verdict = "Chưa chắc chắn"

        if self.debug:
            logger.info(
                f"[fusion_inference] done predict | verdict={verdict!r} | label={pred_label!r} | confidence={confidence:.6f} | elapsed_ms={1000.0 * (perf_counter() - t0):.2f}"
            )

        source_links = []
        for r in retrieval_results[: self.llm_evidence_top_k]:
            meta = r.metadata or {}
            url = _extract_url(meta)
            logger.info(
                f"[fusion_inference:predict] link_debug | url={url} | article_url={meta.get('article_url')} | url={meta.get('url')} | link={meta.get('link')} | source_url={meta.get('source_url')} | all_meta_keys={list(meta.keys())}"
            )
            if url and url not in source_links:
                source_links.append(url)

        t_total = perf_counter() - t0
        timing_ms = {
            "split_ms": round(1000.0 * (t_split1 - t_split0), 1),
            "retrieval_ms": round(1000.0 * (t_retrieval1 - t_retrieval0), 1),
            "nli_ms": round(1000.0 * (t_nli1 - t_nli0), 1),
            "llm_ms": round(1000.0 * (t_llm1 - t_llm0), 1),
            "fusion_ms": round(1000.0 * (t_fusion1 - t_fusion0), 1),
            "total_ms": round(1000.0 * t_total, 1),
        }
        if self.debug:
            logger.info(
                f"[fusion_inference] timing_summary"
                f" | split_ms={timing_ms['split_ms']}"
                f" | retrieval_ms={timing_ms['retrieval_ms']}"
                f" | nli_ms={timing_ms['nli_ms']}"
                f" | llm_ms={timing_ms['llm_ms']}"
                f" | fusion_ms={timing_ms['fusion_ms']}"
                f" | total_ms={timing_ms['total_ms']}"
            )

        label_probs = {
            label: float(probs[i].item()) for i, label in enumerate(VERDICT_ORDER)
        }
        prediction = ClaimPrediction(
            claim=text,
            verdict=verdict,
            label=pred_label,
            label_id=pred_id,
            confidence=confidence,
            evidence=llm_evidence,
            source_links=source_links,
            timing_ms=timing_ms,
            label_probs=label_probs,
        )

        self._log_predictions_to_claims_index([prediction])

        return prediction

    def predict_batch(self, claims: List[str]) -> List[ClaimPrediction]:
        """Dự đoán verdict cho nhiều claim cùng lúc.

        Tách từng claim thành sub-claim, dự đoán toàn bộ sub-claim trong một lô
        phẳng để tận dụng batch, rồi gom kết quả về đúng claim gốc (gộp nếu claim
        có nhiều sub-claim).
        """
        t0 = perf_counter()
        if not claims:
            return []

        expanded: List[tuple[int, str, List[str]]] = []
        flat_sub_claims: List[str] = []
        total_sub_claims = 0

        for i, claim in enumerate(claims):
            text = str(claim).strip()
            if not text:
                continue
            sub_claims = self._prepare_sub_claims(text)
            expanded.append((i, text, sub_claims))
            flat_sub_claims.extend(sub_claims)
            total_sub_claims += len(sub_claims)

        if not expanded:
            return []

        if self.debug:
            now_utc = datetime.now(timezone.utc)
            split_count = sum(1 for _, _, subs in expanded if len(subs) > 1)
            logger.info(
                f"[fusion_inference] start predict_batch | batch_size={len(claims)} | valid_claims={len(expanded)} | split_claims={split_count} | total_sub_claims={total_sub_claims} | now_utc={now_utc.isoformat()}"
            )

        sub_predictions = self._predict_batch_without_split(flat_sub_claims)
        results: List[Optional[ClaimPrediction]] = [None] * len(claims)
        cursor = 0
        for original_idx, original_claim, sub_claims in expanded:
            count = len(sub_claims)
            group_preds = sub_predictions[cursor : cursor + count]
            cursor += count
            if not group_preds:
                continue

            if count == 1:
                p = group_preds[0]
                results[original_idx] = ClaimPrediction(
                    claim=original_claim,
                    verdict=p.verdict,
                    label=p.label,
                    label_id=p.label_id,
                    confidence=p.confidence,
                    evidence=list(p.evidence),
                    source_links=list(p.source_links),
                    label_probs=dict(p.label_probs) if p.label_probs else None,
                )
            else:
                results[original_idx] = self._aggregate_sub_claim_predictions(
                    original_claim, group_preds, sub_claim_count=count
                )

        if cursor != len(sub_predictions):
            logger.warning(
                f"[fusion_inference] predict_batch regroup mismatch: consumed={cursor}, produced={len(sub_predictions)}"
            )

        if self.debug:
            logger.info(
                f"[fusion_inference] batch done | elapsed_ms={1000.0 * (perf_counter() - t0):.2f}"
            )

        final_results = [r for r in results if r is not None]
        self._log_predictions_to_claims_index(final_results)

        return final_results


_VERIFIER_CACHE: Dict[str, FusionClaimVerifier] = {}
_VERIFIER_LOCK = threading.Lock()


def _get_or_create_verifier(
    fusion_model_path: Optional[str],
    opensearch_index: Optional[str],
    llm_model_path: Optional[str],
    retriever_model_path: Optional[str],
    device: Optional[str],
    llm_evidence_top_k: Optional[int],
    effective_debug: bool,
    use_cache: bool,
) -> FusionClaimVerifier:
    cache_key = "|".join([
        fusion_model_path or "",
        opensearch_index or "",
        llm_model_path or "",
        retriever_model_path or "",
        device or "",
        str(llm_evidence_top_k or ""),
        f"debug={int(effective_debug)}",
    ])
    resolved_fusion_path = _resolve_fusion_model_path(
        fusion_model_path or os.getenv("FUSION_MODEL")
    )
    resolved_llm_path = llm_model_path or os.getenv("LLM_FINETUNE")
    logger.info(f"resolved_fusion_path: {resolved_fusion_path}")
    logger.info(f"resolved_llm_path: {resolved_llm_path}")
    logger.info(f"opensearch_index: {opensearch_index}")
    logger.info(f"retriever_model_path: {retriever_model_path}")
    logger.info(f"device: {device}")
    logger.info(f"use_cache: {use_cache}")
    verifier = _VERIFIER_CACHE.get(cache_key) if use_cache else None
    if verifier is None:
        with _VERIFIER_LOCK:
            verifier = _VERIFIER_CACHE.get(cache_key)
            if verifier is None:
                verifier = FusionClaimVerifier(
                    fusion_model_path=resolved_fusion_path,
                    opensearch_index=opensearch_index,
                    llm_model_path=resolved_llm_path,
                    retriever_model_path=retriever_model_path,
                    device=device,
                    llm_evidence_top_k=llm_evidence_top_k,
                    debug=effective_debug,
                )
                if use_cache:
                    _VERIFIER_CACHE[cache_key] = verifier
    return verifier


def verify_claim_true_false(
    claim: str,
    fusion_model_path: Optional[str] = None,
    opensearch_index: Optional[str] = None,
    llm_model_path: Optional[str] = None,
    retriever_model_path: Optional[str] = None,
    device: Optional[str] = None,
    use_cache: bool = True,
    llm_evidence_top_k: Optional[int] = None,
    debug: Optional[bool] = None,
) -> str:
    """
    Convenience function requested:
      input: claim text
      output: "Đúng" hoặc "Sai"
    """
    effective_debug = (
        _env_flag("FUSION_INFERENCE_DEBUG", default=False)
        if debug is None
        else bool(debug)
    )
    logger.info(f"fusion_model_path: {fusion_model_path}")
    verifier = _get_or_create_verifier(
        fusion_model_path, opensearch_index, llm_model_path,
        retriever_model_path, device, llm_evidence_top_k,
        effective_debug, use_cache,
    )
    return verifier.predict(claim).verdict


def verify_claims_true_false(
    claims: List[str],
    fusion_model_path: Optional[str] = None,
    opensearch_index: Optional[str] = None,
    llm_model_path: Optional[str] = None,
    retriever_model_path: Optional[str] = None,
    device: Optional[str] = None,
    use_cache: bool = True,
    llm_evidence_top_k: Optional[int] = None,
    debug: Optional[bool] = None,
    batch_size: int = 4,
) -> List[str]:
    """
    Batch convenience function requested:
      input: list of claim texts
      output: list of "Đúng" hoặc "Sai"
    """
    effective_debug = (
        _env_flag("FUSION_INFERENCE_DEBUG", default=False)
        if debug is None
        else bool(debug)
    )
    verifier = _get_or_create_verifier(
        fusion_model_path, opensearch_index, llm_model_path,
        retriever_model_path, device, llm_evidence_top_k,
        effective_debug, use_cache,
    )
    all_verdicts = []
    for i in range(0, len(claims), batch_size):
        if effective_debug:
            logger.info(
                f"[verify_claims_true_false] Processing batch {i // batch_size + 1} (size: {min(batch_size, len(claims) - i)})"
            )
        batch_claims = claims[i : i + batch_size]
        predictions = verifier.predict_batch(batch_claims)
        all_verdicts.extend([p.verdict for p in predictions])

    return all_verdicts
