"""Split cleaned pages into retrieval-sized chunks that keep their heading context."""
from __future__ import annotations

import re

from .cleaner import HEADING_PREFIX, Page

MAX_CHARS = 900
MIN_SECTION_CHARS = 150  # tiny sections are merged into their neighbour
MIN_CHUNK_CHARS = 40
OVERLAP_CHARS = 200


def _split_long(line: str, max_chars: int) -> list[str]:
    if len(line) <= max_chars:
        return [line]
    pieces: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", line):
        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def _make_chunk(page: Page, section: str, body: str) -> dict:
    label = page.page_name if section == page.page_name else f"{page.page_name} - {section}"
    return {
        "id": -1,
        "title": page.title,
        "page_name": page.page_name,
        "section": section,
        "url": page.url,
        "kind": page.kind,
        "text": f"{label}\n{body}",
    }


def chunk_page(page: Page) -> list[dict]:
    chunks: list[dict] = []
    section = page.page_name
    buffer: list[str] = []

    def size() -> int:
        return sum(len(line) + 1 for line in buffer)

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if len(body) >= MIN_CHUNK_CHARS:
            chunks.append(_make_chunk(page, section, body))

    for raw in page.lines:
        if raw.startswith(HEADING_PREFIX):
            heading = raw[len(HEADING_PREFIX):]
            if size() >= MIN_SECTION_CHARS:
                flush()
                buffer = []
                section = heading
            else:
                buffer.append(heading)
            continue
        for piece in _split_long(raw, MAX_CHARS):
            if buffer and size() + len(piece) > MAX_CHARS:
                flush()
                tail = buffer[-1]
                buffer = [tail] if len(tail) <= OVERLAP_CHARS else []
            buffer.append(piece)
    flush()
    return chunks


def chunk_pages(pages: list[Page]) -> list[dict]:
    chunks = [chunk for page in pages for chunk in chunk_page(page)]
    for index, chunk in enumerate(chunks):
        chunk["id"] = index
    return chunks
