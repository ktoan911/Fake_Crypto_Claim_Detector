import argparse
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://24hmoney.vn/"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
}

BLOCKED_PATH_HINTS = {
    "/dang-nhap",
    "/dang-ky",
    "/login",
    "/register",
    "/privacy",
    "/gioi-thieu",
    "/lien-he",
    "/rss",
}

ARTICLE_CONTENT_SELECTORS = [
    "article",
    "main article",
    ".article-content",
    ".post-content",
    ".content-detail",
    ".news-content",
    ".entry-content",
    "main",
]

DATE_META_SELECTORS = [
    "meta[property='article:published_time']",
    "meta[name='publishdate']",
    "meta[name='pubdate']",
    "meta[itemprop='datePublished']",
    "time[datetime]",
]

DATE_TEXT_SELECTORS = [
    "time",
    ".date",
    ".post-date",
    ".publish-date",
    ".article-date",
]

TOPIC_META_SELECTORS = [
    "meta[property='article:section']",
    "meta[name='news_keywords']",
    "meta[name='keywords']",
]

TOPIC_LINK_SELECTORS = [
    "a[rel='category tag']",
    ".breadcrumb a",
    ".tags a",
    ".post-tags a",
    ".article-tags a",
]

DATE_PATTERNS = [
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
    re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b"),
    re.compile(r"\b(\d{1,2}-\d{1,2}-\d{4})\b"),
]


def _normalize_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    cleaned = parsed._replace(fragment="")
    return cleaned.geturl()


def _is_same_domain(base_netloc: str, candidate_netloc: str) -> bool:
    base = base_netloc.lower().replace("www.", "")
    candidate = candidate_netloc.lower().replace("www.", "")
    return candidate == base or candidate.endswith(f".{base}")


def _looks_like_article(path: str) -> bool:
    if not path or path == "/":
        return False

    path_lower = path.lower()
    if any(hint in path_lower for hint in BLOCKED_PATH_HINTS):
        return False

    if path_lower.endswith((
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".svg",
        ".pdf",
        ".zip",
        ".css",
        ".js",
    )):
        return False

    segments = [s for s in path_lower.split("/") if s]
    if not segments:
        return False

    if len(segments) >= 2:
        return True

    last = segments[-1]
    return "-" in last or any(char.isdigit() for char in last)


def _parse_date(date_raw: str) -> Optional[str]:
    text = (date_raw or "").strip()
    if not text:
        return None

    text = text.replace("Z", "+00:00")
    text = re.sub(r"\s+", " ", text).strip()

    try:
        return datetime.fromisoformat(text).isoformat()
    except ValueError:
        pass

    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d/%m/%Y %H:%M",
        "%d-%m-%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.isoformat()
        except ValueError:
            continue

    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        candidate = match.group(1)
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                dt = datetime.strptime(candidate, fmt)
                return dt.isoformat()
            except ValueError:
                continue

    return None


def _split_keywords(raw: str) -> list[str]:
    if not raw:
        return []
    parts = re.split(r"[,;|/]", raw)
    return [p.strip() for p in parts if p.strip()]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_topic_value(value: Any) -> list[str]:
    topics: list[str] = []
    for item in _as_list(value):
        if isinstance(item, dict):
            name = item.get("name") or item.get("@id") or item.get("headline")
            if isinstance(name, str):
                topics.extend(_split_keywords(name))
        elif isinstance(item, str):
            topics.extend(_split_keywords(item))
    return topics


class Crawl24HMoney:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: int = 20):
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self.base_netloc = urlparse(self.base_url).netloc

    def _fetch_soup(self, url: str) -> BeautifulSoup:
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        return BeautifulSoup(response.text, "html.parser")

    def _extract_article_links(self, soup: BeautifulSoup, link_limit: int) -> list[str]:
        links: list[str] = []
        seen: set[str] = set()

        for anchor in soup.select("a[href]"):
            href = anchor.get("href", "").strip()
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue

            full_url = _normalize_url(urljoin(self.base_url, href))
            parsed = urlparse(full_url)

            if parsed.scheme not in ("http", "https"):
                continue

            if not _is_same_domain(self.base_netloc, parsed.netloc):
                continue

            if not _looks_like_article(parsed.path):
                continue

            if full_url in seen:
                continue

            seen.add(full_url)
            links.append(full_url)

            if len(links) >= link_limit:
                break

        return links

    def _extract_title(self, soup: BeautifulSoup, fallback_url: str) -> str:
        og_title = soup.select_one("meta[property='og:title']")
        if og_title and og_title.get("content"):
            return og_title.get("content", "").strip()

        h1 = soup.select_one("h1")
        if h1:
            title = h1.get_text(strip=True)
            if title:
                return title

        if soup.title and soup.title.string:
            return soup.title.string.strip()

        return fallback_url

    def _extract_json_ld_items(self, soup: BeautifulSoup) -> list[dict]:
        items: list[dict] = []
        for node in soup.select("script[type='application/ld+json']"):
            raw = (node.string or node.get_text() or "").strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if isinstance(data, dict) and isinstance(data.get("@graph"), list):
                for entry in data["@graph"]:
                    if isinstance(entry, dict):
                        items.append(entry)
                continue

            if isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict):
                        items.append(entry)
                continue

            if isinstance(data, dict):
                items.append(data)
        return items

    def _extract_published_at(self, soup: BeautifulSoup) -> Optional[str]:
        json_ld_items = self._extract_json_ld_items(soup)
        for item in json_ld_items:
            for key in ("datePublished", "dateModified", "uploadDate", "dateCreated"):
                raw = item.get(key)
                if isinstance(raw, str):
                    parsed = _parse_date(raw)
                    if parsed:
                        return parsed

        for selector in DATE_META_SELECTORS:
            node = soup.select_one(selector)
            if not node:
                continue

            raw = (
                node.get("content")
                or node.get("datetime")
                or node.get_text(strip=True)
            )
            parsed = _parse_date(raw)
            if parsed:
                return parsed

        for selector in DATE_TEXT_SELECTORS:
            node = soup.select_one(selector)
            if not node:
                continue
            parsed = _parse_date(node.get_text(" ", strip=True))
            if parsed:
                return parsed

        return None

    def _extract_topics(self, soup: BeautifulSoup) -> list[str]:
        topics: list[str] = []
        seen: set[str] = set()

        def _add(values: list[str]) -> None:
            for value in values:
                cleaned = re.sub(r"\s+", " ", value).strip()
                if not cleaned:
                    continue
                key = cleaned.lower()
                if key in seen:
                    continue
                seen.add(key)
                topics.append(cleaned)

        json_ld_items = self._extract_json_ld_items(soup)
        for item in json_ld_items:
            _add(_normalize_topic_value(item.get("articleSection")))
            _add(_normalize_topic_value(item.get("keywords")))
            _add(_normalize_topic_value(item.get("about")))
            _add(_normalize_topic_value(item.get("genre")))

        for selector in TOPIC_META_SELECTORS:
            node = soup.select_one(selector)
            if not node:
                continue
            _add(_split_keywords(node.get("content", "")))

        for selector in TOPIC_LINK_SELECTORS:
            for node in soup.select(selector):
                text = node.get_text(" ", strip=True)
                _add(_split_keywords(text))

        filtered: list[str] = []
        for topic in topics:
            if len(topic) < 2:
                continue
            if topic.lower() in {"home", "trang chủ"}:
                continue
            filtered.append(topic)

        return filtered[:10]

    def _extract_content(self, soup: BeautifulSoup) -> str:
        best_text = ""

        for selector in ARTICLE_CONTENT_SELECTORS:
            section = soup.select_one(selector)
            if not section:
                continue

            for tag in section(["script", "style", "noscript", "header", "footer", "nav"]):
                tag.decompose()

            text = section.get_text("\n", strip=True)
            if len(text) > len(best_text):
                best_text = text

        if len(best_text) < 150 and soup.body:
            body = soup.body
            for tag in body(["script", "style", "noscript", "header", "footer", "nav"]):
                tag.decompose()
            best_text = body.get_text("\n", strip=True)

        best_text = re.sub(r"\n{3,}", "\n\n", best_text).strip()
        return best_text

    def crawl(self, limit: int = 20) -> list[dict]:
        LOGGER.info("Đang crawl danh sách từ %s", self.base_url)
        listing_soup = self._fetch_soup(self.base_url)

        # Thu rộng hơn để tránh hụt do có nhiều link menu/quảng cáo.
        candidate_links = self._extract_article_links(listing_soup, link_limit=max(limit * 20, 100))
        LOGGER.info("Tìm thấy %d link nghi ngờ là bài viết", len(candidate_links))

        articles: list[dict] = []

        for link in candidate_links:
            if len(articles) >= limit:
                break

            try:
                article_soup = self._fetch_soup(link)
            except Exception as exc:
                LOGGER.debug("Bỏ qua link lỗi %s: %s", link, exc)
                continue

            content = self._extract_content(article_soup)
            if len(content) < 300:
                continue

            article = {
                "source": "24hmoney.vn",
                "url": link,
                "title": self._extract_title(article_soup, link),
                "published_at": self._extract_published_at(article_soup),
                "topics": self._extract_topics(article_soup),
                "content": content,
                "crawled_at": datetime.utcnow().isoformat() + "Z",
            }
            articles.append(article)

        LOGGER.info("Crawl thành công %d bài", len(articles))
        return articles


def save_to_json(articles: list[dict], output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(articles, file_obj, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawler tin tức từ 24hmoney.vn")
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="Trang bắt đầu crawl")
    parser.add_argument("--limit", type=int, default=20, help="Số bài viết tối đa")
    parser.add_argument(
        "--output",
        default="data/24hmoney_news.json",
        help="Đường dẫn file JSON đầu ra",
    )
    parser.add_argument("--timeout", type=int, default=20, help="Timeout request (giây)")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Mức log",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    crawler = Crawl24HMoney(base_url=args.url, timeout=args.timeout)
    articles = crawler.crawl(limit=args.limit)
    save_to_json(articles, args.output)

    print(f"Đã lưu {len(articles)} bài viết vào: {args.output}")


if __name__ == "__main__":
    main()
