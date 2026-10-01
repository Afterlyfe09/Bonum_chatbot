"""HTML -> clean text lines, plus separation of site-wide header/footer text."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

from . import config

HEADING_PREFIX = "## "
_NOISE_TAGS = ["script", "style", "noscript", "svg", "iframe", "input", "select", "textarea", "option"]
_HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]
# Filler text that appears on the live site and must never reach the knowledge base.
_PLACEHOLDER = re.compile(r"interdum et malesuada|lorem ipsum", re.IGNORECASE)


@dataclass
class Page:
    url: str
    title: str
    lines: list[str] = field(default_factory=list)
    page_name: str = ""
    kind: str = "page"  # "page" or "sitewide"


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\\'", "'")
    return re.sub(r"\s+", " ", text).strip()


def extract_page(url: str, html: str) -> Page:
    """Extract title and text lines; headings are marked with a '## ' prefix."""
    soup = BeautifulSoup(html, "html.parser")
    title = clean_text(soup.title.get_text(" ")) if soup.title else ""
    if soup.head:
        soup.head.decompose()
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    for heading in soup.find_all(_HEADING_TAGS):
        text = clean_text(heading.get_text(" "))
        if text:
            heading.string = HEADING_PREFIX + text
        else:
            heading.decompose()

    lines: list[str] = []
    seen: set[str] = set()
    for raw in soup.get_text("\n").splitlines():
        line = clean_text(raw)
        if not re.search(r"\w", line) or _PLACEHOLDER.search(line) or line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return Page(url=url, title=title, lines=lines)


def split_boilerplate(
    pages: list[Page], ratio: float = 0.6, min_pages: int = 4
) -> tuple[list[Page], Page | None]:
    """Move lines repeated on most pages (menu, header, footer) into one 'sitewide' page.

    Tag-agnostic on purpose: contact details live in the header/footer, so they are kept
    once as a dedicated document instead of being deleted or duplicated on every page.
    """
    if len(pages) < min_pages:
        return pages, None
    counts = Counter(line for page in pages for line in set(page.lines))
    cutoff = max(2, math.ceil(len(pages) * ratio))
    common = {line for line, count in counts.items() if count >= cutoff}
    if not common:
        return pages, None

    sitewide_lines: list[str] = []
    seen: set[str] = set()
    for page in pages:
        for line in page.lines:
            if line in common and line not in seen:
                seen.add(line)
                sitewide_lines.append(line)
        page.lines = [line for line in page.lines if line not in common]
    sitewide = Page(
        url=config.HOME_URL,
        title="Bonum eDesigns - site header, footer and contact details",
        lines=sitewide_lines,
        page_name="Contact and site information",
        kind="sitewide",
    )
    return pages, sitewide


def assign_page_names(pages: list[Page]) -> None:
    """Human-readable page name = first remaining heading, else the <title>."""
    for page in pages:
        if page.url == config.HOME_URL:
            page.page_name = "Home"
            continue
        heading = next(
            (line[len(HEADING_PREFIX):] for line in page.lines if line.startswith(HEADING_PREFIX)), ""
        )
        page.page_name = heading or page.title or page.url
