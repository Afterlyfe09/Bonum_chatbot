"""Local sentence-transformer embeddings (no API cost, loaded once per process)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import config


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> np.ndarray:
    """L2-normalised float32 vectors, so inner product equals cosine similarity."""
    vectors = _model().encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
    )
    return np.asarray(vectors, dtype="float32")


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])
