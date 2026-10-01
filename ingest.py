"""Crawl bonum.in, clean and chunk the pages, embed them and build the FAISS index.

Usage:
    python ingest.py             # download pages (falls back to data/raw cache) and rebuild
    python ingest.py --offline   # rebuild only from pages already saved in data/raw
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict

from src import chunker, cleaner, config, retriever, scraper


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Bonum knowledge base.")
    parser.add_argument("--offline", action="store_true", help="use only cached HTML in data/raw")
    parser.add_argument("--max-pages", type=int, default=config.MAX_PAGES)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("1/4 Crawling bonum.in ...")
    html_by_url = scraper.crawl(offline=args.offline, max_pages=args.max_pages)
    if not html_by_url:
        print("No pages could be downloaded or found in data/raw. Check your connection and retry.")
        return 1
    print(f"    {len(html_by_url)} pages fetched")
    for url, html in html_by_url.items():
        print(f"    raw {len(html):>7} bytes  {url}")

    print("2/4 Cleaning ...")
    pages = [cleaner.extract_page(url, html) for url, html in html_by_url.items()]
    pages = [page for page in pages if page.lines]
    pages, sitewide = cleaner.split_boilerplate(pages)
    if sitewide:
        pages.append(sitewide)
    cleaner.assign_page_names(pages)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (config.PROCESSED_DIR / "pages.json").write_text(
        json.dumps([asdict(page) for page in pages], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for page in pages:
        chars = sum(len(line) for line in page.lines)
        print(f"    {page.page_name[:40]:<40} {len(page.lines):>3} lines {chars:>6} chars  {page.url}")

    print("3/4 Chunking ...")
    chunks = chunker.chunk_pages(pages)
    if not chunks:
        print("No usable text was extracted. Inspect data/raw and data/processed/pages.json.")
        return 1
    print(f"    {len(chunks)} chunks")

    print("4/4 Embedding and building FAISS index (first run downloads the embedding model) ...")
    retriever.build_index(chunks)
    print(f"Done. Index saved to {config.VECTOR_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
