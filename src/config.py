"""Central configuration for the M2PI RAG FAQ chatbot.

All runtime settings come from the environment (loaded from ``.env``); secrets are
never hard-coded. Tunable RAG parameters live here as documented constants so the
chunking and retrieval behaviour can be adjusted in one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env once at import time. override=False so a real environment variable wins.
load_dotenv(override=False)

# --- Models ---
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536-dim OpenAI embeddings
OPENAI_MODEL = "gpt-5.4-mini"               # chat model for answer + evaluator
TEMPERATURE = 0.2                           # low -> grounded, low-variance answers

# --- Chunking (sentence-aware with token overlap) ---
CHUNK_TARGET_TOKENS = 90    # target chunk size; lands comfortably inside 50-500
CHUNK_OVERLAP_TOKENS = 20    # trailing-sentence overlap preserves cross-chunk context
CHUNK_MIN_TOKENS = 50       # rubric floor per chunk
CHUNK_MAX_TOKENS = 500      # rubric ceiling per chunk

# --- Retrieval ---
TOP_K = 4    # chunks fed to the LLM (rubric expects 2-5)
RRF_K = 60   # reciprocal-rank-fusion constant (standard from the IR literature)

# --- Paths ---
INDEX_DIR = "index"
FAQ_PATH = "data/faq_document.txt"


@dataclass(frozen=True)
class Settings:
    """Immutable run configuration, sourced from the environment."""

    api_key: str
    embedding_model: str
    chat_model: str
    temperature: float
    top_k: int

    @staticmethod
    def from_env() -> "Settings":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        embedding_model = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL).strip() or EMBEDDING_MODEL
        chat_model = os.getenv("OPENAI_MODEL", OPENAI_MODEL).strip() or OPENAI_MODEL
        try:
            top_k = int(os.getenv("TOP_K", str(TOP_K)))
        except ValueError:
            top_k = TOP_K
        top_k = max(2, min(5, top_k))  # clamp to the rubric's 2-5 window
        return Settings(
            api_key=api_key,
            embedding_model=embedding_model,
            chat_model=chat_model,
            temperature=TEMPERATURE,
            top_k=top_k,
        )
