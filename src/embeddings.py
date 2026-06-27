"""OpenAI embedding generation.

Vectors are L2-normalized so that an inner-product search (FAISS IndexFlatIP)
yields cosine similarity directly. Embeddings are generated in a single batched
API call for all chunks.
"""
from __future__ import annotations

import numpy as np

from src.config import Settings
from src.llm import call_with_retry, get_client


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit L2 norm (so dot product == cosine similarity)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def generate_embeddings(texts, client=None, settings=None) -> np.ndarray:
    """Embed a list of texts in one batch call. Returns float32 (n, dim), normalized."""
    settings = settings or Settings.from_env()
    client = client or get_client(settings)
    resp = call_with_retry(
        lambda: client.embeddings.create(model=settings.embedding_model, input=texts)
    )
    vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return _normalize(vectors)


def embed_query(text, client=None, settings=None) -> np.ndarray:
    """Embed a single query string. Returns a float32 (dim,) normalized vector."""
    return generate_embeddings([text], client=client, settings=settings)[0]
