"""Polite crawler restricted to the official Bonum website."""
from __future__ import annotations

import logging
import re
import time
from collections import deque
from urllib.parse import parse_qs, urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 BonumChatbotIngest/1.0"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}
ALLOWED_HOSTS = {"bonum.in", "www.bonum.in"}
SKIP_PREFIXES = ("/admin", "/uploads", "/images", "/assets", "/css", "/js")


def normalize_url(url: str) -> str | None:
    """Return a canonical crawlable Bonum URL, or None if it should be skipped."""
    url, _ = urldefrag(url.strip())
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or parsed.netloc.lower() not in ALLOWED_HOSTS:
        return None
    path = parsed.path or "/"
    if path == "/":
        path = "/index.php"
    if path.startswith(SKIP_PREFIXES) or not path.endswith(".php"):
        return None
    query = ""
    if path.endswith("blog_detail.php"):
        blog_id = parse_qs(parsed.query).get("id", [""])[0]
        if not blog_id.isdigit():
            return None
        query = f"?id={blog_id}"
    return f"{config.BASE_URL}{path}{query}"


def cache_path(url: str):
    """Where a page's raw HTML is cached. Manually saved pages use the same names."""
    parsed = urlparse(url)
    slug = re.sub(r"[^a-z0-9]+", "_", f"{parsed.path}_{parsed.query}".lower()).strip("_")
    return config.RAW_DIR / f"{slug}.html"


def _make_session() -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=(409, 429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.headers.update(HEADERS)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _download(session: requests.Session, url: str) -> str | None:
    try:
        response = session.get(url, timeout=config.REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Could not download %s (%s)", url, exc)
        return None
    if "text/html" not in response.headers.get("Content-Type", ""):
        return None
    return response.content.decode("utf-8", errors="replace")


def _extract_links(page_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = (normalize_url(urljoin(page_url, a["href"])) for a in soup.find_all("a", href=True))
    return [link for link in links if link]


def crawl(offline: bool = False, max_pages: int = config.MAX_PAGES) -> dict[str, str]:
    """Return {url: html}. Falls back to data/raw/<slug>.html when a download fails."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    queue = deque(
        url for url in (normalize_url(f"{config.BASE_URL}/{p}") for p in config.SEED_PATHS) if url
    )
    queued = set(queue)
    pages: dict[str, str] = {}
    session = None if offline else _make_session()

    while queue and len(pages) < max_pages:
        url = queue.popleft()
        html = _download(session, url) if session else None
        if html:
            cache_path(url).write_text(html, encoding="utf-8")
        else:
            path = cache_path(url)
            if not path.exists():
                log.warning("Skipped %s (not downloadable and not cached)", url)
                continue
            log.info("Using cached copy for %s", url)
            html = path.read_text(encoding="utf-8")
        pages[url] = html
        for link in _extract_links(url, html):
            if link not in queued:
                queued.add(link)
                queue.append(link)
        if session:
            time.sleep(config.REQUEST_DELAY_SECONDS)
    return pages
