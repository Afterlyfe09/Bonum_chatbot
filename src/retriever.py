"""FAISS index build/load and hybrid (semantic + light keyword) search."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from . import config
from .embeddings import embed_query, embed_texts


class VectorStoreError(Exception):
    """Raised with a message that is safe to show to end users."""


@dataclass(frozen=True)
class Hit:
    chunk: dict
    score: float


# (query pattern, URL fragment to favour, also favour the site-wide contact chunk?)
# These only nudge ranking toward the right page; they contain no business facts.
_INTENT_HINTS: list[tuple[re.Pattern, str, bool]] = [
    (re.compile(r"\b(contact|e-?mail|phone|call|number|address|located|location|office|reach|whatsapp)\b", re.I), "contact.php", True),
    (re.compile(r"\b(faq|faqs|frequently|questions?)\b", re.I), "faq.php", False),
    (re.compile(r"\b(refund|cancel\w*|money back)\b", re.I), "refund.php", False),
    (re.compile(r"\b(privacy|personal data)\b", re.I), "privacy.php", False),
    (re.compile(r"\b(terms|conditions|usage)\b", re.I), "terms.php", False),
    (re.compile(r"\b(blog|blogs|article|articles)\b", re.I), "blog", False),
    (re.compile(r"\b(about|who are|company|team)\b", re.I), "about.php", False),
    (re.compile(r"web\s?-?(site|dev\w*|design)|\bwebsite\b", re.I), "webdev.php", False),
    (re.compile(r"\b(apps?|mobile|android|ios)\b", re.I), "appdev.php", False),
    (re.compile(r"\b(software|erp|crm)\b", re.I), "software.php", False),
    (re.compile(r"\b(analytics|dashboards?|bi)\b", re.I), "data.php", False),
    (re.compile(r"\b(marketing|seo|social media|ads)\b", re.I), "digital.php", False),
]
_INTENT_BOOST = 0.12
_KEYWORD_BOOST = 0.05
_LEXICAL_BOOST = 0.10
_STOPWORDS = {
    "what", "does", "tell", "about", "with", "your", "have", "that", "this", "they", "from",
    "which", "offer", "provide", "bonum", "please", "would", "could", "there", "their", "much",
}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 3 and t not in _STOPWORDS}


def _boost(query: str, query_tokens: set[str], chunk: dict, text_tokens: frozenset | set = frozenset()) -> float:
    boost = 0.0
    for pattern, url_fragment, include_sitewide in _INTENT_HINTS:
        if not pattern.search(query):
            continue
        if url_fragment in chunk["url"] or (include_sitewide and chunk.get("kind") == "sitewide"):
            boost = max(boost, _INTENT_BOOST)
    label_tokens = _tokens(f"{chunk['page_name']} {chunk['section']}")
    lexical = _LEXICAL_BOOST * len(query_tokens & text_tokens) / len(query_tokens) if query_tokens else 0.0
    return boost + min(2, len(query_tokens & label_tokens)) * _KEYWORD_BOOST + lexical


def _expand_neighbors(hits: list[Hit], chunks: list[dict], limit: int) -> list[Hit]:
    """Append the chunks right before/after each hit (same page) so long sections stay whole."""
    seen = {hit.chunk["id"] for hit in hits}
    expanded = list(hits)
    for hit in hits:
        for j in (hit.chunk["id"] - 1, hit.chunk["id"] + 1):
            if 0 <= j < len(chunks) and j not in seen and chunks[j]["url"] == hit.chunk["url"] \
                    and chunks[j]["kind"] == hit.chunk["kind"]:
                seen.add(j)
                expanded.append(Hit(chunks[j], hit.score - 0.01))
    return expanded[:limit]


def build_index(chunks: list[dict]) -> None:
    """Embed all chunks and persist the FAISS index plus chunk metadata."""
    import faiss

    vectors = embed_texts([chunk["text"] for chunk in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    config.VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.INDEX_PATH))
    config.META_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")


class Retriever:
    def __init__(self, index_path: Path = config.INDEX_PATH, meta_path: Path = config.META_PATH):
        if not index_path.exists() or not meta_path.exists():
            raise VectorStoreError(
                "The Bonum knowledge base has not been built yet. Run `python ingest.py` first."
            )
        try:
            import faiss

            self._index = faiss.read_index(str(index_path))
            self._chunks: list[dict] = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise VectorStoreError(
                "The knowledge base files look corrupted. Re-run `python ingest.py` to rebuild them."
            ) from exc
        if self._index.ntotal != len(self._chunks) or not self._chunks:
            raise VectorStoreError(
                "The knowledge base is inconsistent. Re-run `python ingest.py` to rebuild it."
            )
        self._text_tokens = [_tokens(chunk["text"]) for chunk in self._chunks]

    def search(
        self, query: str, top_k: int = config.TOP_K, threshold: float = config.RELEVANCE_THRESHOLD
    ) -> list[Hit]:
        """Top chunks whose (cosine + small boost) score clears the relevance threshold."""
        if not query.strip():
            return []
        candidates = min(self._index.ntotal, top_k * 4)
        scores, ids = self._index.search(embed_query(query), candidates)
        query_tokens = _tokens(query)
        hits = [
            Hit(self._chunks[i], float(score) + _boost(query, query_tokens, self._chunks[i], self._text_tokens[i]))
            for score, i in zip(scores[0], ids[0])
            if i >= 0
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        selected = [hit for hit in hits[:top_k] if hit.score >= threshold]
        return _expand_neighbors(selected, self._chunks, top_k * 2)