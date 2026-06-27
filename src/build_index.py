"""Indexing data pipeline: load -> chunk -> embed -> store.

Stages (rubric "Data Pipeline de Indexacion"):
  1. Load the document (UTF-8 aware read).
  2. Segment into chunks (sentence-aware, 20+ chunks).
  3. Generate embeddings for every chunk.
  4. Store the FAISS index + chunk text + metadata under index/.

Run from the repo root:  python -m src.build_index
"""
from __future__ import annotations

import json
import os
import pickle

import faiss

from src.chunking import load_and_chunk_document
from src.config import FAQ_PATH, INDEX_DIR, Settings
from src.embeddings import generate_embeddings
from src.retrieval import build_faiss_index


def _persist(index_dir: str, chunks: list[dict], vectors) -> dict:
    """Write the FAISS index, the chunk texts, and a metadata summary to disk."""
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(build_faiss_index(vectors), os.path.join(index_dir, "faiss.index"))
    with open(os.path.join(index_dir, "chunks.pkl"), "wb") as fh:
        pickle.dump(chunks, fh)
    meta = {
        "n_chunks": len(chunks),
        "n_embeddings": int(vectors.shape[0]),
        "embedding_dim": int(vectors.shape[1]),
        "min_tokens": min(c["n_tokens"] for c in chunks),
        "max_tokens": max(c["n_tokens"] for c in chunks),
    }
    with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def build(faq_path: str = FAQ_PATH, index_dir: str = INDEX_DIR):
    """Run the full indexing pipeline and return (chunks, vectors, meta)."""
    settings = Settings.from_env()
    chunks = load_and_chunk_document(faq_path)                  # Stage 1-2
    vectors = generate_embeddings([c["text"] for c in chunks], settings=settings)  # Stage 3
    meta = _persist(index_dir, chunks, vectors)                 # Stage 4
    return chunks, vectors, meta


def main() -> None:
    _, _, meta = build()
    print(
        f"Indexed {meta['n_chunks']} chunks ({meta['n_embeddings']} embeddings, "
        f"dim={meta['embedding_dim']}) | tokens {meta['min_tokens']}-{meta['max_tokens']} "
        f"-> {INDEX_DIR}/"
    )


if __name__ == "__main__":
    main()
