import argparse
import html
import logging
import os
import random
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from exa_py import Exa

from dotenv import load_dotenv

# Thư mục gốc project (4 cấp trên file này — giống crawl_24hmoney.py)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

load_dotenv(os.path.join(_PROJECT_ROOT, ".env"), override=True)

LOGGER = logging.getLogger(__name__)

# 1. Khởi tạo Exa với hỗ trợ nhiều API key (fallback khi key bị lỗi/hết quota)
# EXA_API_KEY trong .env có thể là 1 key hoặc nhiều key cách nhau bằng dấu phẩy:
#   EXA_API_KEY=key1,key2,key3
_RAW_KEYS = os.getenv("EXA_API_KEY", "")
API_KEYS = [k.strip() for k in _RAW_KEYS.split(",") if k.strip()]
if not API_KEYS:
    raise ValueError("Lỗi: Chưa thiết lập biến môi trường EXA_API_KEY.")

# Index key đang dùng + danh sách key đã bị đánh dấu hỏng trong session này
_current_key_idx = 0
_dead_keys = set()
_key_lock = threading.Lock()  # bảo vệ rotate khi nhiều thread cùng fail
exa = Exa(api_key=API_KEYS[_current_key_idx])
LOGGER.info("Đã load %d Exa API key, đang dùng key #%d.",
            len(API_KEYS), _current_key_idx + 1)


def _rotate_key():
    """
    Chuyển sang key kế tiếp chưa bị đánh dấu hỏng.
    Trả về True nếu rotate thành công, False nếu hết key khả dụng.
    Thread-safe: nhiều thread cùng fail sẽ không skip key hoặc rotate quá đà.
    """
    global _current_key_idx, exa
    with _key_lock:
        _dead_keys.add(_current_key_idx)

        for offset in range(1, len(API_KEYS) + 1):
            idx = (_current_key_idx + offset) % len(API_KEYS)
            if idx not in _dead_keys:
                _current_key_idx = idx
                exa = Exa(api_key=API_KEYS[idx])
                LOGGER.info("  → Chuyển sang Exa API key #%d", idx + 1)
                return True
        return False


# Các pattern lỗi cho thấy key có vấn đề (sai key / hết quota / bị chặn) → cần rotate
_KEY_ERROR_PATTERNS = ("401", "403", "429", "unauthorized", "forbidden",
                       "quota", "rate limit", "invalid api key", "payment")


def _is_key_error(err):
    msg = str(err).lower()
    return any(p in msg for p in _KEY_ERROR_PATTERNS)


TARGET_COUNT = 50
MAX_RESULTS_PER_QUERY = 15
RETRY = 3

# Cấu hình runtime, sẽ được override từ CLI args (giống Crawl24HMoneyV2)
HOURS_BACK = 24.0          # cào bài đăng trong N giờ gần nhất
CRAWL_WORKERS = 4          # số luồng song song khi gọi Exa search
DELAY_BETWEEN = 0.3        # giây nghỉ giữa các batch query (giảm rate-limit)
# Lưu ý: exa-py SDK chưa expose timeout ở constructor, nên --timeout hiện chỉ
# có ý nghĩa biểu trưng cho phép tương lai pass xuống. Không dùng trực tiếp ở đây.

# Excerpt = 1 câu đầu tiên đủ dài, ghép với title để tạo đầu ra chuẩn
# (title thường là đề tựa, câu đầu thường là lead/sapo — ghép lại đủ ý cho LLM phân loại).
SENTENCE_MIN_CHARS = 30      # câu ngắn hơn ngưỡng này coi như chưa đủ ý → thử câu kế tiếp
SENTENCE_MAX_CHARS = 300     # nếu câu quá dài thì cắt mềm tại khoảng trắng

# 2. BỘ TỪ KHÓA ĐƯỢC THIẾT KẾ ĐẶC TRỊ CHO TÀI CHÍNH VIỆT NAM
# Chia theo nhóm chủ đề để đảm bảo 50 tin lấy được phủ đều, không bị dồn vào 1 chủ đề.
QUERY_GROUPS = {
    "stock": [
        "Tin đồn úp bô trên sàn chứng khoán mã cổ phiếu đội lái xả hàng",
        "Sự thật đằng sau việc cổ phiếu bị thao túng giá rớt thảm hại F319",
        "Hóng biến công ty chứng khoán call margin hàng loạt tài khoản",
    ],
    "banking": [
        "Tin gầm giường ngân hàng mất thanh khoản không rút được tiền",
        "Giám đốc chi nhánh ngân hàng ôm tiền bỏ trốn hóng biến",
        "Sự thật tin đồn sáp nhập ngân hàng yếu kém bị mua lại 0 đồng",
    ],
    "real_estate_bonds": [
        "Bom nợ trái phiếu doanh nghiệp bất động sản chủ tịch bùng tiền",
        "Dự án ma bất động sản phân lô bán nền lùa gà nhà đầu tư",
        "Hóng phốt tập đoàn bất động sản bị phong tỏa tài khoản",
    ],
    "crypto_forex_scam": [
        "Sập sàn forex lừa đảo cháy tài khoản nhà đầu tư kêu cứu",
        "Phốt app đầu tư tài chính đa cấp sập rút tiền không được",
        "Chủ sàn coin đa cấp lùa gà bị bế đi trong đêm",
    ],
    "macro": [
        "Thuyết âm mưu tỷ giá USD tăng vọt dòng tiền tháo chạy",
        "Hóng tin mật bắt bớ đại gia tài chính thao túng thị trường",
    ],
}

# Quota mặc định: chia đều TARGET_COUNT cho các nhóm, dư thì cộng vào nhóm cuối.
def _build_quota(target, groups):
    n = len(groups)
    base = target // n
    quota = {g: base for g in groups}
    quota[list(groups)[-1]] += target - base * n
    return quota

# 3. DANH SÁCH CHẶN (BLACKLIST)
URLS_TO_EXCLUDE = [
    "https://www.sbv.gov.vn/", "https://div.gov.vn/", "https://baochinhphu.vn/", 
    "https://mof.gov.vn/", "https://cic.gov.vn/", "https://tapchinganhang.gov.vn/",
    "https://thoibaonganhang.vn/", "https://baodautu.vn/", "https://thesaigontimes.vn/",
    "https://vietstock.vn/", "https://bnews.vn/", "https://tapchicongthuong.vn/",
    "https://vnba.org.vn/", "https://www.customs.gov.vn/", "https://www.vietinbank.vn/",
    "https://bidv.com.vn/", "https://www.agribank.com.vn/", "https://techcombank.com/", 
    "https://www.vpbank.com.vn/", "https://www.mbbank.com.vn/", "https://acb.com.vn/",
    "https://www.sacombank.com.vn/", "https://hdbank.com.vn/", "https://tpb.vn/",
    "https://www.ocb.com.vn/", "https://www.shb.com.vn/", "https://www.seabank.com.vn/", 
    "https://www.vietcombank.com.vn/", "https://cafef.vn/", "https://vneconomy.vn/",
    "https://vietnambiz.vn/", "https://vietnamnet.vn/", "https://nld.com.vn/",
    "https://www.sggp.org.vn/", "https://baotintuc.vn/", "https://tinnhanhchungkhoan.vn/",
    "https://tapchikinhtetaichinh.vn/", "https://cafebiz.vn/", "https://vnexpress.net/",
    "https://tuoitre.vn/", "https://thanhnien.vn/", "https://dantri.com.vn/",
    "https://laodong.vn/", "https://plo.vn/", "https://znews.vn/", "https://vtv.vn/", 
    "https://vov.vn/", "https://tinnhiemmang.vn/", "https://ssc.gov.vn/", "https://www.vietnam.vn/",
    "https://www.hsx.vn/", "https://www.hnx.vn/", "https://vsd.vn/", "https://bocongan.gov.vn/"
]

BLACKLIST_DOMAINS = list(set(
    urlparse(url).netloc.lower().replace("www.", "")
    for url in URLS_TO_EXCLUDE if urlparse(url).netloc
))

def clean_text(text):
    """
    Dọn rác trong content cào về từ web (markdown leftover, HTML entity,
    navigation chain, timestamp rác, emoji, lặp tiêu đề).
    Chỉ dùng regex, không LLM.
    """
    if not text:
        return ""

    # 1. Decode HTML entity: &agrave; → à, &amp; → &, &nbsp; → space
    text = html.unescape(text)

    # 2. Bỏ markdown heading markers (#, ##, ###...) ở đầu cụm/sau xuống dòng
    text = re.sub(r"(^|[\n\s])#{1,6}\s+", r"\1", text)

    # 3. Bỏ horizontal rule (---, ***, ___)
    text = re.sub(r"(^|\n)\s*[-*_]{3,}\s*(\n|$)", r"\1", text)

    # 4. Bỏ emoji & ký tự đặc biệt (giữ chữ Latin/Việt, số, dấu câu cơ bản)
    text = re.sub(r"[^\w\s\.,;:!?\-–—'\"%/()\[\]&@$+°²³…]", " ",
                  text, flags=re.UNICODE)

    # 5. Bỏ timestamp rác: "--:--:-- PM", "07:00", "30/05/2026 07:00"
    text = re.sub(r"-{2,}:-{2,}:-{2,}\s*[AP]M", " ", text)
    text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}\b", " ", text)
    text = re.sub(r"\b\d{1,2}:\d{2}(:\d{2})?\s*([AP]M)?\b", " ", text)

    # 6. Bỏ navigation chain: " - X - Y - Z - ..." nơi mỗi mục ngắn (< 40 ký tự)
    # Pattern: ≥ 3 cụm ngắn nối bằng " - " liên tiếp.
    text = re.sub(r"(?:\s-\s[^-\n]{1,40}){3,}", " ", text)

    # 7. Bỏ chuỗi "Đăng nhập | Đăng ký | ..." kiểu menu
    menu_words = (r"Đăng nhập|Đăng ký|Trang chủ|Bình luận|Chia sẻ|Gửi mail|"
                  r"Facebook|Google|Email|Mật khẩu|Xin chào|Hủy|Đối nội|Đối ngoại|"
                  r"Bảo vệ nền tảng tư tưởng|Kết luận thanh tra|Diễn đàn|Ban chỉ đạo")
    text = re.sub(rf"(?:\b(?:{menu_words})\b[\s,|×!\-]*){{2,}}", " ", text,
                  flags=re.IGNORECASE)

    # 8. Bỏ marker "TIN MỚI NHẬN", "TPO -", "Mới | N giờ trước"
    text = re.sub(r"\bTIN MỚI NHẬN\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bTPO\s*[-–]\s*", " ", text)
    text = re.sub(r"\bMới\s*\|\s*\d+\s*(giờ|phút|ngày)\s*trước", " ", text,
                  flags=re.IGNORECASE)

    # 9. Bỏ lỗi server: 502/503/504 Bad Gateway / nginx / cloudflare
    text = re.sub(r"\b50[234]\s+(Bad Gateway|Service Unavailable|Gateway Timeout)\b",
                  " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(nginx|cloudflare|apache)\b", " ", text, flags=re.IGNORECASE)

    # 10. Gộp khoảng trắng / xuống dòng dư
    text = re.sub(r"\s+", " ", text).strip()

    # 11. Bỏ dấu câu lặp: "..", ",,", "!!", " . . ."
    text = re.sub(r"([.,;:!?\-])\1{2,}", r"\1", text)
    text = re.sub(r"(\s[.,;:!?]){2,}", " ", text)

    return text


def first_sentence(text, min_chars=SENTENCE_MIN_CHARS, max_chars=SENTENCE_MAX_CHARS):
    """
    Trích câu đầu tiên đủ ý từ văn bản.
    - Bỏ qua các câu quá ngắn (< min_chars) — thường là tiêu đề phụ, ngày tháng, "Đọc thêm:"...
    - Nếu câu vượt max_chars, cắt mềm tại khoảng trắng cuối cùng + "…"
    - Trả về chuỗi rỗng nếu không tìm được câu phù hợp.
    """
    if not text:
        return ""

    # Chuẩn hoá: bỏ xuống dòng thừa, gộp khoảng trắng
    cleaned = " ".join(text.split())

    # Tách câu thô theo . ! ? — đủ tốt cho tiếng Việt
    parts = re.split(r"(?<=[.!?])\s+", cleaned)

    for sent in parts:
        sent = sent.strip()
        if len(sent) < min_chars:
            continue
        if len(sent) <= max_chars:
            return sent
        # Câu quá dài → cắt mềm
        window = sent[:max_chars]
        space = window.rfind(" ")
        if space >= int(max_chars * 0.5):
            return window[:space].rstrip(",;:") + "…"
        return window + "…"

    return ""


def build_record_text(title, content):
    """
    Đầu ra chuẩn = title + câu đầu tiên đủ ý của content (đã được làm sạch).
    Tránh lặp nếu câu đầu đã chứa title (hoặc ngược lại).
    """
    title = clean_text(title or "")
    cleaned_content = clean_text(content or "")

    # Nếu title xuất hiện 2+ lần trong content (header lặp), chỉ giữ phần sau lần lặp cuối
    if title and len(title) >= 15:
        occurrences = [m.start() for m in re.finditer(re.escape(title), cleaned_content)]
        if len(occurrences) >= 2:
            cleaned_content = cleaned_content[occurrences[-1] + len(title):].strip()

    sent = first_sentence(cleaned_content)

    if not sent:
        return title
    if not title:
        return sent

    # Tránh lặp: nếu câu đầu chứa toàn bộ title (hoặc gần đúng) thì chỉ trả về câu đó
    if title.lower() in sent.lower():
        return sent

    # Đảm bảo title có dấu kết câu trước khi nối
    if title[-1] not in ".!?":
        title = title + "."
    return f"{title} {sent}"


def search(query):
    # start_published_date dùng HOURS_BACK (CLI override được)
    cutoff = (datetime.now(tz=timezone.utc) - timedelta(hours=HOURS_BACK)).isoformat()

    for attempt in range(RETRY):
        try:
            # SỬ DỤNG AUTO (KEYWORD SEARCH) THAY VÌ NEURAL
            # Lý do: Khi cần độ chính xác cao về chủ đề chuyên biệt, tìm kiếm theo từ khóa (keyword)
            # sẽ ghim chặt kết quả vào "tài chính", "ngân hàng", "chứng khoán" hơn là để Neural tự phiêu.
            response = exa.search(
                query=query,
                type="auto",
                num_results=MAX_RESULTS_PER_QUERY,
                start_published_date=cutoff,
                exclude_domains=BLACKLIST_DOMAINS,
                contents={"text": True}
            )
            return response.results

        except Exception as e:
            # Nếu là lỗi key (401/403/429/quota...) → rotate sang key kế tiếp và retry ngay
            # mà không tốn lượt RETRY của lỗi mạng thông thường.
            if _is_key_error(e):
                LOGGER.warning("Key #%d có vấn đề: %s", _current_key_idx + 1, e)
                if _rotate_key():
                    continue  # retry ngay với key mới, không sleep
                LOGGER.error("Hết key khả dụng, dừng query này.")
                return []

            wait = (attempt + 1) * 5
            LOGGER.warning("Lỗi gọi Exa API: %s. Đang thử lại sau %ds...", e, wait)
            time.sleep(wait)

    return []

def is_financial_context(text):
    """
    BỘ LỌC CỨNG: ÉP BUỘC VĂN BẢN PHẢI CÓ TỪ KHÓA TÀI CHÍNH
    """
    finance_keywords = [
        "cổ phiếu", "chứng khoán", "ngân hàng", "lãi suất", "tỷ giá", "usd", "vnd",
        "trái phiếu", "thanh khoản", "đáo hạn", "tín dụng", "huy động vốn", 
        "dự án", "đội lái", "f319", "bất động sản", "tiền gửi", "đầu tư", "forex", "coin"
    ]
    text_lower = text.lower()
    
    # Phải chứa ít nhất 2 từ khóa tài chính thì mới cho qua
    match_count = sum(1 for kw in finance_keywords if kw in text_lower)
    return match_count >= 2

def keep(item):
    text = item.text or ""
    url = item.url or ""
    
    # 1. Bỏ qua bài quá ngắn
    if len(text) < 150:
        return False

    # 2. Bỏ qua trang danh mục
    if any(path in url.lower() for path in ["/tag/", "/chu-de/", "/category/", "/danh-muc/"]):
        return False

    # 3. CHỐT CHẶN MỚI: Ép nội dung phải thuộc chủ đề Tài chính/Ngân hàng
    if not is_financial_context(text):
        return False

    return True

def _norm_for_dedup(s):
    """Chuẩn hoá chuỗi để so sánh dedup: lower, gộp khoảng trắng."""
    return " ".join((s or "").lower().split())


def _published_to_unix(date_str):
    """Convert published_date ISO string từ Exa sang unix timestamp (int) hoặc None."""
    if not date_str:
        return None
    try:
        text = date_str.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except (ValueError, AttributeError):
        return None


def crawl():
    """
    Cào Exa song song theo CRAWL_WORKERS, dedup trong session, trả về list articles.
    Mỗi article = {"title": str, "url": str, "published_at": int|None}.
    """
    seen_url = set()
    seen_content = set()
    state_lock = threading.Lock()

    quota = _build_quota(TARGET_COUNT, QUERY_GROUPS)
    collected_per_group = {g: 0 for g in QUERY_GROUPS}
    articles: list[dict] = []

    # Mỗi nhóm có pool query riêng, shuffle độc lập
    group_queries = {g: random.sample(qs, len(qs)) for g, qs in QUERY_GROUPS.items()}

    def _process_results(group: str, results) -> int:
        """Merge kết quả 1 query vào state chung. Trả về số tin mới thêm."""
        added = 0
        with state_lock:
            for item in results:
                if len(articles) >= TARGET_COUNT:
                    break
                if collected_per_group[group] >= quota[group]:
                    break
                url = item.url
                if not url or url in seen_url:
                    continue
                if not keep(item):
                    continue

                content = build_record_text(item.title, item.text)
                key = _norm_for_dedup(content)
                if not key or key in seen_content:
                    continue

                seen_url.add(url)
                seen_content.add(key)
                articles.append({
                    "title": content,  # title + 1 câu đầu, đã clean — dùng làm claim
                    "url": url,
                    "published_at": _published_to_unix(item.published_date),
                })
                collected_per_group[group] += 1
                added += 1
                LOGGER.info(" + [%s] %s", group, url)
        return added

    empty_passes = 0
    while len(articles) < TARGET_COUNT and empty_passes < 3:
        # Mỗi pass: chọn 1 query/nhóm còn thiếu quota, submit song song
        batch: list[tuple[str, str]] = []  # [(group, query), ...]
        for group in QUERY_GROUPS:
            if collected_per_group[group] >= quota[group]:
                continue
            if not group_queries[group]:
                group_queries[group] = random.sample(
                    QUERY_GROUPS[group], len(QUERY_GROUPS[group]))
            batch.append((group, group_queries[group].pop()))

        if not batch:
            break

        LOGGER.info("Pass: %d query song song (workers=%d)", len(batch), CRAWL_WORKERS)
        for g, q in batch:
            LOGGER.info("  [%s %d/%d] %s", g, collected_per_group[g], quota[g], q)

        progressed = False
        with ThreadPoolExecutor(max_workers=CRAWL_WORKERS) as pool:
            futures = {pool.submit(search, q): (g, q) for g, q in batch}
            for fut in as_completed(futures):
                group, query = futures[fut]
                try:
                    results = fut.result()
                except Exception as exc:
                    LOGGER.warning("Query [%s] '%s' lỗi: %s", group, query, exc)
                    continue
                if _process_results(group, results) > 0:
                    progressed = True
                if len(articles) >= TARGET_COUNT:
                    break

        empty_passes = 0 if progressed else empty_passes + 1
        time.sleep(DELAY_BETWEEN)

    # Nếu nhóm nào không đủ quota, lấp bằng các nhóm khác (cũng chạy song song)
    if len(articles) < TARGET_COUNT:
        LOGGER.info("Một số nhóm không đủ quota, lấp thêm cho đủ %d...", TARGET_COUNT)
        all_queries = [(g, q) for g, qs in QUERY_GROUPS.items() for q in qs]
        random.shuffle(all_queries)

        # Tạm gỡ giới hạn quota cho phase fill — chỉ cần đủ TARGET_COUNT
        for g in collected_per_group:
            quota[g] = TARGET_COUNT

        for i in range(0, len(all_queries), CRAWL_WORKERS):
            if len(articles) >= TARGET_COUNT:
                break
            chunk = all_queries[i: i + CRAWL_WORKERS]
            with ThreadPoolExecutor(max_workers=CRAWL_WORKERS) as pool:
                futures = {pool.submit(search, q): (g, q) for g, q in chunk}
                for fut in as_completed(futures):
                    group, _ = futures[fut]
                    try:
                        results = fut.result()
                    except Exception:
                        continue
                    _process_results(group, results)
                    if len(articles) >= TARGET_COUNT:
                        break
            time.sleep(DELAY_BETWEEN)

    LOGGER.info("Crawl xong: %d tin tài chính từ Exa", len(articles))
    return articles


def _load_verifier():
    """Khởi tạo FusionClaimVerifier từ biến môi trường (giống crawl_24hmoney.py)."""
    from src.models.fusion_inference import (  # noqa: PLC0415
        FusionClaimVerifier,
        _resolve_fusion_model_path,
    )

    fusion_path = _resolve_fusion_model_path(os.getenv("FUSION_MODEL"))
    return FusionClaimVerifier(
        fusion_model_path=fusion_path,
        opensearch_index=os.getenv("OPENSEARCH_INDEX_NAME") or os.getenv("OP_KB_NAME", "news_kb"),
        llm_model_path=os.getenv("LLM_FINETUNE"),
        retriever_model_path=os.getenv("RETRIEVER_MODEL", "AITeamVN/Vietnamese_Embedding"),
        device=os.getenv("DEVICE", "cpu"),
        llm_evidence_top_k=int(os.getenv("FUSION_LLM_EVIDENCE_TOP_K", "3")),
        debug=False,
    )


def predict_and_index(articles: list[dict]) -> dict:
    """
    Predict từng mini-batch (llm_infer_batch_size) rồi insert ngay vào OP_CLAIMS_INDEX.
    """
    from src.database.opensearch import OpenSearchKB  # noqa: PLC0415

    valid = [(art, art["title"].strip()) for art in articles if (art.get("title") or "").strip()]
    if not valid:
        LOGGER.warning("Không có tiêu đề hợp lệ để predict.")
        return {"inserted": 0, "errors": 0}

    verifier = _load_verifier()
    kb = OpenSearchKB(index_name=os.getenv("OP_CLAIMS_INDEX", "claims"), embedding_dim=1)
    batch_size = getattr(verifier, "llm_infer_batch_size", 4)

    total_inserted = 0
    total_errors = 0

    for start in range(0, len(valid), batch_size):
        mini = valid[start: start + batch_size]
        mini_arts, mini_titles = zip(*mini)

        LOGGER.info("Predict mini-batch %d–%d / %d ...",
                    start + 1, start + len(mini), len(valid))
        preds = verifier._predict_batch_without_split(list(mini_titles))

        checked_at = datetime.now(timezone.utc).isoformat()
        docs = [
            {
                "id": str(uuid.uuid4()),
                "claim": title,
                "verdict": pred.verdict,
                "confidence": pred.confidence,
                "source_links": pred.source_links,
                "checked_at": checked_at,
                "source": "exa_social",
                "url": art.get("url", ""),
                "published_at": art.get("published_at"),
            }
            for art, title, pred in zip(mini_arts, mini_titles, preds)
            if pred is not None
        ]

        if not docs:
            continue

        result = kb.insert_many(docs, upsert=True)
        ins = result.get("inserted", 0)
        err = result.get("errors", 0)
        total_inserted += ins
        total_errors += err
        LOGGER.info("Batch %d–%d: inserted=%d errors=%d",
                    start + 1, start + len(mini), ins, err)

    return {"inserted": total_inserted, "errors": total_errors}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Crawler Exa – cào tin đồn tài chính VN, predict và index vào OpenSearch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--hours-back", type=float, default=24.0, metavar="N",
                   help="Lấy tin đăng trong N giờ gần nhất")
    p.add_argument("--limit", type=int, default=50, metavar="N",
                   help="Số tin tối đa cần cào (0 = không giới hạn → dùng TARGET_COUNT mặc định)")
    p.add_argument("--workers", type=int, default=4, metavar="N",
                   help="Số luồng song song khi gọi Exa search")
    p.add_argument("--delay", type=float, default=0.3, metavar="SEC",
                   help="Giây nghỉ giữa mỗi vòng query")
    p.add_argument("--timeout", type=int, default=20, metavar="SEC",
                   help="Timeout mỗi request Exa (giây)")
    p.add_argument("--no-index", action="store_true",
                   help="Chỉ cào, không predict/index vào OpenSearch")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main() -> None:
    global TARGET_COUNT, HOURS_BACK, CRAWL_WORKERS, DELAY_BETWEEN
    args = build_arg_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Override config từ CLI
    HOURS_BACK = args.hours_back
    CRAWL_WORKERS = max(1, args.workers)
    DELAY_BETWEEN = args.delay
    if args.timeout != 20:
        LOGGER.warning("--timeout %d được set nhưng exa-py SDK không hỗ trợ, "
                       "flag này hiện không có hiệu lực.", args.timeout)
    if args.limit > 0:
        TARGET_COUNT = args.limit

    articles = crawl()

    if args.no_index:
        LOGGER.info("Bỏ qua predict/index (--no-index). Đã cào %d tin.", len(articles))
        return

    if not articles:
        LOGGER.warning("Không có tin nào để index.")
        return

    result = predict_and_index(articles)
    print(
        f"\nKết quả: insert {result.get('inserted', 0)} / {len(articles)} tin "
        f"vào OpenSearch (lỗi: {result.get('errors', 0)})."
    )


if __name__ == "__main__":
    main()