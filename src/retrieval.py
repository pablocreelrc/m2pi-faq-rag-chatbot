"""Hybrid retrieval: dense vector search + lexical BM25, fused with RRF.

- Dense: FAISS ``IndexFlatIP`` over L2-normalized embeddings, so inner product is
  cosine similarity (exact k-NN; the corpus is small enough not to need ANN).
- Sparse: BM25Okapi over whitespace tokens, which catches exact terms the dense
  model can blur (product names, acronyms like SSO/MFA/SLA, numbers, prices).
- Fusion: Reciprocal Rank Fusion combines both rankings without needing the two
  score scales to be comparable.
"""
from __future__ import annotations

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src.config import MIN_CHUNKS, MIN_RELEVANCE, RRF_K, TOP_K


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Explicit cosine similarity: dot(a, b) / (||a|| * ||b||)."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b)) / denom if denom else 0.0


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """Exact inner-product index; with normalized vectors IP == cosine."""
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def _tokenize(text: str) -> list[str]:
    return [tok for tok in text.lower().split() if tok]


def build_bm25(chunk_texts: list[str]) -> BM25Okapi:
    """Build a BM25 index over the tokenized chunk texts."""
    return BM25Okapi([_tokenize(t) for t in chunk_texts])


def faiss_search(index: faiss.Index, query_vec: np.ndarray, k: int) -> list[int]:
    """Return chunk indices ranked by descending cosine similarity."""
    _, idx = index.search(np.array([query_vec], dtype=np.float32), k)
    return [int(i) for i in idx[0] if i != -1]


def bm25_search(bm25: BM25Okapi, query: str, k: int) -> list[int]:
    """Return chunk indices ranked by descending BM25 score."""
    scores = bm25.get_scores(_tokenize(query))
    return [int(i) for i in np.argsort(scores)[::-1][:k]]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Fuse ranked id-lists: score(doc) = sum over lists of 1 / (k + rank)."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused, key=lambda d: fused[d], reverse=True)


def _trim_by_relevance(ids, query_vec, faiss_index, chunks, min_chunks, min_relevance):
    """Drop trailing chunks whose cosine to the query is below the relevance floor.

    Keeps every chunk that clears the floor (in fused-rank order). If fewer than
    min_chunks clear it, falls back to the most cosine-relevant chunks so the 2-5
    rubric floor is always met.
    """
    scored = [(i, cosine_similarity(query_vec, faiss_index.reconstruct(int(i)))) for i in ids]
    keep = [i for i, sim in scored if sim >= min_relevance]
    if len(keep) < min_chunks:
        keep = [i for i, _ in sorted(scored, key=lambda t: t[1], reverse=True)[:min_chunks]]
    return [chunks[i] for i in keep]


def hybrid_search(query, query_vec, faiss_index, bm25, chunks, top_k=TOP_K) -> list[dict]:
    """Dense (FAISS cosine) + sparse (BM25), fused with RRF; return 2..top_k chunks.

    After ranking, trailing chunks whose cosine to the query falls below MIN_RELEVANCE
    are dropped so short keyword queries are not padded with off-topic context.
    """
    pool = max(top_k * 2, 10)
    dense = faiss_search(faiss_index, query_vec, pool)
    sparse = bm25_search(bm25, query, pool)
    fused = reciprocal_rank_fusion([dense, sparse])[:top_k]
    return _trim_by_relevance(fused, query_vec, faiss_index, chunks, MIN_CHUNKS, MIN_RELEVANCE)
