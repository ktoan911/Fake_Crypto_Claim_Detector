from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

load_dotenv(os.path.join(_PROJECT_ROOT, ".env"), override=True)

from src.database.opensearch import OpenSearchKB  # noqa: E402
from src.llm_call import generate_rumor_claims_from_news  # noqa: E402

LOGGER = logging.getLogger(__name__)

DEFAULT_QUERY = "tài chính ngân hàng chứng khoán trái phiếu bất động sản vàng tiền điện tử"
DEFAULT_SOURCE = "llm_news_kb_rumor"
SOURCE_FIELDS = [
    "title",
    "description",
    "content",
    "text",
    "published_at",
    "timestamp",
    "article_url",
    "source_url",
    "url",
    "link",
]


def _compact_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _source_url(source: Dict[str, Any]) -> str:
    for key in ("article_url", "url", "link", "source_url"):
        value = _compact_text(source.get(key))
        if value:
            return value
    return ""


def _published_to_unix(value: Any) -> Optional[int]:
    if value is None:
        return None

    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        number = float(value)
        if number > 10_000_000_000:
            number = number / 1000.0
        return int(number)
    else:
        raw = str(value).strip()
        if not raw:
            return None
        try:
            number = float(raw)
            if number > 10_000_000_000:
                number = number / 1000.0
            return int(number)
        except ValueError:
            pass

        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.astimezone(timezone.utc).timestamp())


def _kb_embedding_dim() -> int:
    try:
        return int(os.getenv("OP_EMBEDDING_DIM", "768"))
    except ValueError:
        return 768


def search_news_kb(
    query: str,
    source_limit: int = 20,
    days_back: Optional[float] = None,
    index_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search read-only documents from the news knowledge-base index."""
    index_name = index_name or os.getenv("OP_KB_NAME", "news_kb")
    kb = OpenSearchKB(index_name=index_name, embedding_dim=_kb_embedding_dim())

    filter_clauses: List[Dict[str, Any]] = []
    if days_back and days_back > 0:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
        filter_clauses.append({"range": {"published_at": {"gte": cutoff}}})

    normalized_query = _compact_text(query)
    query_clause: Dict[str, Any]
    if normalized_query:
        query_clause = {
            "multi_match": {
                "query": normalized_query,
                "fields": ["title^3", "description^2", "content"],
                "type": "best_fields",
                "minimum_should_match": "30%",
            }
        }
    else:
        query_clause = {"match_all": {}}

    body: Dict[str, Any] = {
        "size": max(1, int(source_limit)),
        "_source": SOURCE_FIELDS,
        "query": {
            "bool": {
                "must": [query_clause],
                "filter": filter_clauses,
            }
        },
    }

    LOGGER.info("Search OpenSearch index=%s source_limit=%d", index_name, source_limit)
    resp = kb.client.search(index=kb.index, body=body)
    hits = resp.get("hits", {}).get("hits", [])

    items: List[Dict[str, Any]] = []
    seen = set()
    for hit in hits:
        source = hit.get("_source", {}) or {}
        title = _compact_text(source.get("title"))
        content = _compact_text(
            source.get("content") or source.get("description") or source.get("text")
        )
        if not title and not content:
            continue

        url = _source_url(source)
        dedup_key = url or f"{title}\n{content[:180]}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        published_at = source.get("published_at") or source.get("timestamp")
        items.append(
            {
                "source_ref": len(items) + 1,
                "source_news_id": str(hit.get("_id") or ""),
                "title": title,
                "description": _compact_text(source.get("description")),
                "content": content,
                "published_at": published_at,
                "published_at_unix": _published_to_unix(published_at),
                "url": url,
            }
        )

    LOGGER.info("Collected %d source news docs from %s", len(items), index_name)
    return items


def _first_two_sentences(text: str) -> str:
    text = _compact_text(text).strip("-*•` ")
    parts = re.split(r"(?<=[.!?。])\s+", text)
    return " ".join(part for part in parts[:2] if part) or text


def _parse_source_ref(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        match = re.search(r"\d+", str(value))
        return int(match.group()) if match else None


def parse_rumor_claims(text: str) -> List[Dict[str, Any]]:
    """Parse output JSON array từ LLM thành list {"claim", "source_ref"}."""
    if not text:
        return []

    match = re.search(r"\[.*\]", text.strip(), re.DOTALL)
    raw = match.group() if match else text.strip()
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(items, list):
        return []

    rumors: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        if isinstance(item, dict):
            claim = item.get("claim") or item.get("text") or item.get("rumor")
            source_ref = _parse_source_ref(item.get("source_ref") or item.get("ref"))
        else:
            claim = item
            source_ref = None

        claim_text = _first_two_sentences(str(claim or ""))
        key = claim_text.lower()
        if len(claim_text) < 10 or key in seen:
            continue
        seen.add(key)
        rumors.append({"claim": claim_text, "source_ref": source_ref})

    return rumors


def build_rumor_articles(
    news_items: List[Dict[str, Any]],
    target_count: int,
    max_retries: int = 2,
) -> List[Dict[str, Any]]:
    rumors: List[Dict[str, Any]] = []
    total_attempts = max(1, max_retries + 1)
    for attempt in range(1, total_attempts + 1):
        raw_output = generate_rumor_claims_from_news(
            news_items=news_items,
            target_count=target_count,
        )
        rumors = parse_rumor_claims(raw_output)
        if rumors:
            if attempt > 1:
                LOGGER.info("LLM parse OK ở lần thử %d/%d", attempt, total_attempts)
            break
        preview = (raw_output or "")[:300] if isinstance(raw_output, str) else repr(raw_output)[:300]
        LOGGER.warning(
            "Lần thử %d/%d: không parse được rumor claim từ LLM: %r",
            attempt,
            total_attempts,
            preview,
        )
    else:
        LOGGER.error(
            "Bỏ qua batch (%d tin nguồn) sau %d lần LLM không trả về JSON hợp lệ.",
            len(news_items),
            total_attempts,
        )
        return []

    ref_map = {item["source_ref"]: item for item in news_items}

    articles: List[Dict[str, Any]] = []
    skipped_no_ref = 0
    for rumor in rumors:
        claim = _compact_text(rumor.get("claim"))
        if not claim:
            continue

        source_ref = rumor.get("source_ref")
        source_item = ref_map.get(source_ref) if source_ref is not None else None
        if source_item is None:
            skipped_no_ref += 1
            continue

        articles.append(
            {
                "title": claim,
                "url": source_item.get("url", ""),
                "published_at_unix": source_item.get("published_at_unix"),
                "source_ref": source_ref,
                "source_news_id": source_item.get("source_news_id", ""),
                "source_news_title": source_item.get("title", ""),
            }
        )

    if skipped_no_ref:
        LOGGER.warning(
            "Bỏ %d rumor không khớp source_ref hợp lệ (tránh gán sai nguồn).",
            skipped_no_ref,
        )

    return articles[:target_count]


def _load_verifier():
    from src.models.fusion_inference import (  # noqa: PLC0415
        FusionClaimVerifier,
        _resolve_fusion_model_path,
    )

    fusion_path = _resolve_fusion_model_path(os.getenv("FUSION_MODEL"))
    return FusionClaimVerifier(
        fusion_model_path=fusion_path,
        opensearch_index=os.getenv("OPENSEARCH_INDEX_NAME")
        or os.getenv("OP_KB_NAME", "news_kb"),
        llm_model_path=os.getenv("LLM_FINETUNE"),
        retriever_model_path=os.getenv(
            "RETRIEVER_MODEL", "AITeamVN/Vietnamese_Embedding"
        ),
        device=os.getenv("DEVICE", "cpu"),
        llm_evidence_top_k=int(os.getenv("FUSION_LLM_EVIDENCE_TOP_K", "3")),
        debug=False,
    )


def predict_and_index(
    articles: List[Dict[str, Any]], kb_index_name: str
) -> Dict[str, Any]:
    valid = []
    for article in articles:
        claim = _compact_text(article.get("title"))
        if claim:
            valid.append((article, claim))
    if not valid:
        LOGGER.warning("Không có rumor claim hợp lệ để predict.")
        return {"inserted": 0, "errors": 0}

    verifier = _load_verifier()
    claims_kb = OpenSearchKB(
        index_name=os.getenv("OP_CLAIMS_INDEX", "claims"),
        embedding_dim=1,
    )
    batch_size = max(1, int(getattr(verifier, "llm_infer_batch_size", 4)))

    total_inserted = 0
    total_errors = 0

    for start in range(0, len(valid), batch_size):
        mini = valid[start : start + batch_size]
        mini_articles, mini_claims = zip(*mini)

        LOGGER.info(
            "Predict mini-batch %d-%d / %d ...",
            start + 1,
            start + len(mini),
            len(valid),
        )
        preds = verifier._predict_batch_without_split(list(mini_claims))
        checked_at = datetime.now(timezone.utc).isoformat()

        docs = [
            {
                "id": str(uuid.uuid4()),
                "claim": claim,
                "verdict": pred.verdict,
                "confidence": pred.confidence,
                "source_links": pred.source_links,
                "checked_at": checked_at,
                "source": DEFAULT_SOURCE,
                "url": article.get("url", ""),
                "published_at": article.get("published_at_unix"),
                "generated": True,
                "generated_from_index": kb_index_name,
                "source_news_id": article.get("source_news_id", ""),
                "source_news_title": article.get("source_news_title", ""),
            }
            for article, claim, pred in zip(mini_articles, mini_claims, preds)
            if pred is not None
        ]

        if not docs:
            continue

        result = claims_kb.insert_many(docs, upsert=True)
        inserted = int(result.get("inserted", 0) or 0)
        errors = int(result.get("errors", 0) or 0)
        total_inserted += inserted
        total_errors += errors
        LOGGER.info(
            "Batch %d-%d: inserted=%d errors=%d",
            start + 1,
            start + len(mini),
            inserted,
            errors,
        )

    return {"inserted": total_inserted, "errors": total_errors}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sinh tin đồn từ OpenSearch news_kb, fact-check và index vào claims.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Query BM25 dùng để lấy tin nguồn từ news_kb.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Số tin đồn/claim cần sinh.",
    )
    parser.add_argument(
        "--source-limit",
        type=int,
        default=20,
        help="Số tin nguồn lấy từ news_kb để đưa vào prompt LLM.",
    )
    parser.add_argument(
        "--days-back",
        type=float,
        default=0.0,
        help="Chỉ lấy tin nguồn trong N ngày gần nhất; 0 là không lọc ngày.",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Chỉ search + sinh tin đồn, không predict/index vào OpenSearch claims.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Số lần retry LLM nếu parse fail; vượt ngưỡng thì bỏ batch để khỏi tốn quota.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.limit <= 0:
        raise ValueError("--limit phải lớn hơn 0")
    if args.source_limit <= 0:
        raise ValueError("--source-limit phải lớn hơn 0")
    if args.max_retries < 0:
        raise ValueError("--max-retries không được âm")

    kb_index_name = os.getenv("OP_KB_NAME", "news_kb")
    news_items = search_news_kb(
        query=args.query,
        source_limit=args.source_limit,
        days_back=args.days_back,
        index_name=kb_index_name,
    )
    if not news_items:
        LOGGER.warning("Không tìm thấy tin nguồn phù hợp trong %s.", kb_index_name)
        return

    articles = build_rumor_articles(
        news_items,
        target_count=args.limit,
        max_retries=args.max_retries,
    )
    LOGGER.info("Generated %d rumor claims", len(articles))

    if args.no_index:
        print(json.dumps(articles, ensure_ascii=False, indent=2))
        return

    if not articles:
        LOGGER.warning("Không có rumor claim nào để predict/index.")
        return

    result = predict_and_index(articles, kb_index_name=kb_index_name)
    print(
        f"\nKết quả: insert {result.get('inserted', 0)} / {len(articles)} "
        f"tin đồn vào OpenSearch claims (lỗi: {result.get('errors', 0)})."
    )


if __name__ == "__main__":
    main()
