"""
Crawler bài viết 24hmoney.vn (v2)
----------------------------------
Nguồn index: sitemap-news.xml

Chỉ lấy tin tức chứng khoán, ngân hàng và tài chính Việt Nam.
Các danh mục được phép cào cố định trong ALLOWED_CATEGORY_IDS.

Tham số chính
-------------
--hours-back      : lấy bài đăng trong N giờ gần nhất (mặc định 2)
--limit           : tối đa bao nhiêu bài (mặc định 0 = không giới hạn)
--workers         : số luồng song song khi tải nội dung bài (mặc định 4)
--delay           : giây nghỉ giữa mỗi request (mặc định 0.3)
--timeout         : timeout mỗi request (mặc định 20 giây)
Luồng xử lý: sitemap → lọc tiêu đề → _predict_batch_without_split (1 lần) → bulk-insert OP_CLAIMS_INDEX
"""

import argparse
import logging
import os
import re
import sys
import time
import uuid
import xml.etree.ElementTree as ET

from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from bs4 import BeautifulSoup

# Thư mục gốc project (4 cấp trên file này)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


LOGGER = logging.getLogger(__name__)

SITEMAP_URL = "https://24hmoney.vn/sitemap-news.xml"

ALLOWED_CATEGORY_IDS: frozenset[str] = frozenset({"1", "4", "27", "30", "50", "81"})

NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "news": "http://www.google.com/schemas/sitemap-news/0.9",
}

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
}

CONTENT_SELECTORS = [
    ".content-detail",
    ".article-content",
    ".post-content",
    ".news-content",
    "article",
    "main",
]

URL_CAT_RE = re.compile(r"/news/[^/]+-c(\d+)a\d+\.html")


def _extract_category_id(url: str) -> Optional[str]:
    m = URL_CAT_RE.search(url)
    return m.group(1) if m else None


def _parse_iso(date_str: str) -> Optional[datetime]:
    if not date_str:
        return None
    text = date_str.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _to_unix(dt: Optional[datetime]) -> Optional[int]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


class Crawl24HMoneyV2:
    def __init__(
        self,
        hours_back: float = 24.0,
        limit: int = 50,
        workers: int = 4,
        delay: float = 0.3,
        timeout: int = 20,
    ):
        self.hours_back = hours_back
        self.limit = limit  # 0 = unlimited
        self.workers = workers
        self.delay = delay
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    def _fetch_bytes(self, url: str) -> bytes:
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.content

    def _parse_sitemap(self, xml_bytes: bytes) -> list[dict]:
        root = ET.fromstring(xml_bytes)
        entries: list[dict] = []
        for url_el in root.findall("sm:url", NS):
            loc = url_el.findtext("sm:loc", namespaces=NS) or ""
            pub_date_raw = url_el.findtext(
                "news:news/news:publication_date", namespaces=NS
            ) or ""
            title = url_el.findtext("news:news/news:title", namespaces=NS) or ""
            dt = _parse_iso(pub_date_raw)
            entries.append(
                {
                    "url": loc,
                    "title": title,
                    "published_at_dt": dt,
                    "published_at": _to_unix(dt),
                    "category_id": _extract_category_id(loc),
                }
            )
        return entries

    def _get_candidate_urls(self) -> list[dict]:
        LOGGER.info("Đang tải sitemap từ %s", SITEMAP_URL)
        try:
            xml_bytes = self._fetch_bytes(SITEMAP_URL)
        except Exception as exc:
            LOGGER.error("Không tải được sitemap: %s", exc)
            return []

        entries = self._parse_sitemap(xml_bytes)
        LOGGER.info("Sitemap có %d bài viết", len(entries))

        cutoff = _now_utc() - timedelta(hours=self.hours_back)
        LOGGER.info("Lọc bài đăng sau %s (%.1f giờ gần nhất)", cutoff.isoformat(), self.hours_back)

        filtered: list[dict] = []
        for entry in entries:
            dt = entry["published_at_dt"]
            if dt is None:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt < cutoff:
                continue
            if entry["category_id"] not in ALLOWED_CATEGORY_IDS:
                continue
            filtered.append(entry)

        filtered.sort(key=lambda e: e["published_at"] or 0, reverse=True)

        if self.limit > 0:
            filtered = filtered[: self.limit]

        LOGGER.info("Còn %d bài sau khi lọc", len(filtered))
        return filtered

    def _fetch_article(self, entry: dict) -> dict:
        url = entry["url"]
        time.sleep(self.delay)
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception as exc:
            LOGGER.debug("Lỗi khi tải %s: %s", url, exc)
            return {}

        return {
            "title": self._extract_title(soup, entry.get("title") or url),
            "content": self._extract_content(soup),
            "url": url,
            "published_at": entry.get("published_at"),
        }

    def _extract_title(self, soup: BeautifulSoup, fallback: str) -> str:
        og = soup.select_one("meta[property='og:title']")
        if og and og.get("content"):
            return og["content"].strip()
        h1 = soup.select_one("h1")
        if h1:
            t = h1.get_text(strip=True)
            if t:
                return t
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        return fallback

    def _extract_content(self, soup: BeautifulSoup) -> str:
        best = ""
        for sel in CONTENT_SELECTORS:
            node = soup.select_one(sel)
            if not node:
                continue
            for tag in node(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
                tag.decompose()
            text = node.get_text("\n", strip=True)
            if len(text) > len(best):
                best = text

        if len(best) < 200 and soup.body:
            for tag in soup.body(["script", "style", "noscript", "header", "footer", "nav"]):
                tag.decompose()
            best = soup.body.get_text("\n", strip=True)

        return re.sub(r"\n{3,}", "\n\n", best).strip()

    def crawl(self) -> list[dict]:
        candidates = self._get_candidate_urls()
        if not candidates:
            LOGGER.warning("Không có bài nào phù hợp với bộ lọc.")
            return []

        articles = [
            {
                "title": (entry.get("title") or entry["url"]).strip(),
                "url": entry["url"],
                "published_at": entry.get("published_at"),
            }
            for entry in candidates
            if (entry.get("title") or "").strip()
        ]

        LOGGER.info("Crawl hoàn thành: %d tiêu đề từ sitemap", len(articles))
        return articles


def _load_verifier():
    """Khởi tạo FusionClaimVerifier từ biến môi trường (giống api_server.py)."""
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
    Gọi _predict_batch_without_split trên toàn bộ tiêu đề một lần,
    rồi bulk-insert kết quả vào OP_CLAIMS_INDEX.
    """
    from src.database.opensearch import OpenSearchKB  # noqa: PLC0415

    valid = [(art, art["title"].strip()) for art in articles if (art.get("title") or "").strip()]
    if not valid:
        LOGGER.warning("Không có tiêu đề hợp lệ để predict.")
        return {"inserted": 0, "errors": 0}

    verifier = _load_verifier()

    arts, titles = zip(*valid)
    LOGGER.info("Predict batch %d tiêu đề ...", len(titles))
    preds = verifier._predict_batch_without_split(list(titles))

    checked_at = datetime.now(timezone.utc).isoformat()
    docs = [
        {
            "id": str(uuid.uuid4()),
            "claim": title,
            "verdict": pred.verdict,
            "confidence": pred.confidence,
            "evidence": pred.evidence,
            "source_links": pred.source_links,
            "checked_at": checked_at,
            "source": "24hmoney",
            "url": art.get("url", ""),
            "published_at": art.get("published_at"),
        }
        for art, title, pred in zip(arts, titles, preds)
        if pred is not None
    ]

    LOGGER.info("Predict xong: %d / %d tiêu đề có kết quả.", len(docs), len(titles))

    if not docs:
        LOGGER.warning("Không có doc nào để insert vào OpenSearch.")
        return {"inserted": 0, "errors": 0}

    kb = OpenSearchKB(index_name=os.getenv("OP_CLAIMS_INDEX", "claims"), embedding_dim=1)
    result = kb.insert_many(docs, upsert=True)
    LOGGER.info(
        "Đã insert %d docs vào index '%s'. Lỗi: %d.",
        result.get("inserted", 0),
        kb.index,
        result.get("errors", 0),
    )
    return result


_CRAWL_WORKER_OPTIONS = [1, 2, 4, 8, 12]
_SEP = "─" * 64


def _benchmark_crawl(hours_back: float, limit: int) -> tuple[list[dict], int]:
    """
    Phase 1: đo tốc độ cào với từng mức crawl_workers.
    Trả về (articles từ lần chạy có rate cao nhất, best_workers).
    """
    print(f"\n{'═'*64}")
    print("PHASE 1 – Crawl workers  (request thực tới 24hmoney.vn)")
    print(_SEP)
    print(f"{'Workers':>8} │ {'Fetched':>7} │ {'Time (s)':>9} │ {'Art/s':>8} │ Speedup vs 1")
    print(_SEP)

    rows: list[tuple[int, int, float, float]] = []
    best_articles: list[dict] = []

    for w in _CRAWL_WORKER_OPTIONS:
        crawler = Crawl24HMoneyV2(hours_back=hours_back, limit=limit, workers=w, delay=0.1, timeout=15)
        t0 = time.perf_counter()
        try:
            articles = crawler.crawl()
            elapsed = time.perf_counter() - t0
            rate = len(articles) / elapsed if elapsed > 0 else 0.0
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            articles = []
            rate = 0.0
            print(f"{w:>8} │  ERROR: {exc}")
            rows.append((w, 0, elapsed, 0.0))
            continue

        baseline = rows[0][3] if rows else rate
        speedup = rate / baseline if baseline > 0 else 1.0
        print(f"{w:>8} │ {len(articles):>7} │ {elapsed:>9.2f} │ {rate:>8.2f} │ {speedup:>8.2f}x")
        rows.append((w, len(articles), elapsed, rate))

        if not best_articles or rate >= max(r[3] for r in rows):
            best_articles = articles

    best_row = max(rows, key=lambda r: r[3])
    print(_SEP)
    print(f"✅ Tối ưu crawl : --workers {best_row[0]}  ({best_row[3]:.2f} bài/giây)")
    return best_articles, best_row[0]


def run_benchmark(hours_back: float, limit: int) -> None:
    """Benchmark crawl workers và in khuyến nghị."""
    print("=" * 64)
    print("  BENCHMARK: 24hmoney crawl workers")
    print(f"  hours_back={hours_back}  article_limit={limit}")
    print("=" * 64)

    _, best_crawl_w = _benchmark_crawl(hours_back, limit)

    print(f"\n{'═'*64}")
    print("KHUYẾN NGHỊ SỬ DỤNG")
    print(_SEP)
    print("  python -m src.data_process.crawlers.news_crawler.crawl_24hmoney \\")
    print(f"      --workers {best_crawl_w} \\")
    print("      --hours-back 2")
    print()
    print("Ghi chú:")
    print("  • Crawl workers : I/O-bound → nhiều workers nhanh hơn (đến giới hạn mạng)")
    print("  • Predict        : batch một lần duy nhất, không có tham số workers")
    print(_SEP)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Crawler 24hmoney.vn – lấy bài viết mới nhất theo khoảng thời gian",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--hours-back", type=float, default=2.0, metavar="N",
                   help="Lấy bài đăng trong N giờ gần nhất")
    p.add_argument("--limit", type=int, default=0, metavar="N",
                   help="Số bài tối đa (0 = không giới hạn)")
    p.add_argument("--workers", type=int, default=4, metavar="N",
                   help="Số luồng song song khi tải nội dung")
    p.add_argument("--delay", type=float, default=0.3, metavar="SEC",
                   help="Giây nghỉ giữa mỗi request")
    p.add_argument("--timeout", type=int, default=20, metavar="SEC",
                   help="Timeout mỗi request (giây)")
    p.add_argument("--benchmark", action="store_true",
                   help="Chạy benchmark tìm crawl workers tối ưu thay vì pipeline thật")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.benchmark:
        run_benchmark(
            hours_back=args.hours_back,
            limit=args.limit if args.limit > 0 else 20,
        )
        return

    crawler = Crawl24HMoneyV2(
        hours_back=args.hours_back,
        limit=args.limit,
        workers=args.workers,
        delay=args.delay,
        timeout=args.timeout,
    )

    articles = crawler.crawl()

    result = predict_and_index(articles)
    print(
        f"\nKết quả: insert {result.get('inserted', 0)} / {len(articles)} bài "
        f"vào OpenSearch (lỗi: {result.get('errors', 0)})."
    )


if __name__ == "__main__":
    main()
