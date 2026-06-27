"""Core tests — run WITHOUT an API key (the OpenAI calls are mocked).

Covered:
- Chunking: 20+ chunks, every chunk within the 50-500 token window, full coverage.
- Retrieval: explicit cosine similarity, RRF ordering, hybrid returns 2-5 chunks.
- Output contract: generate_answer returns exactly the three required keys.
- Evaluator: returns an int 0-10 score with a >=50-char reason (and clamps bad input).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.chunking import count_tokens, load_and_chunk_document
from src.config import CHUNK_MAX_TOKENS, CHUNK_MIN_TOKENS, FAQ_PATH
from src.evaluator import evaluate_answer
from src.generation import REQUIRED_KEYS, generate_answer
from src.retrieval import (
    build_bm25,
    build_faiss_index,
    cosine_similarity,
    hybrid_search,
    reciprocal_rank_fusion,
)


# --- mock OpenAI client ------------------------------------------------------

def _chat_response(payload: dict):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
    )


class _FakeClient:
    """Stands in for openai.OpenAI; returns a fixed chat payload."""

    def __init__(self, chat_payload: dict):
        completions = SimpleNamespace(create=lambda **_kw: _chat_response(chat_payload))
        self.chat = SimpleNamespace(completions=completions)


# --- chunking ----------------------------------------------------------------

def test_chunking_meets_count_and_token_bounds():
    chunks = load_and_chunk_document(FAQ_PATH)
    assert len(chunks) >= 20
    for c in chunks:
        assert CHUNK_MIN_TOKENS <= c["n_tokens"] <= CHUNK_MAX_TOKENS


def test_chunking_covers_start_and_end_of_document():
    raw = open(FAQ_PATH, encoding="utf-8").read()
    chunks = load_and_chunk_document(FAQ_PATH)
    assert "PeopleFlow" in chunks[0]["text"]
    assert "permanently deleted" in chunks[-1]["text"]
    assert raw.strip()  # document is non-empty


def test_chunk_ids_are_sequential_and_unique():
    chunks = load_and_chunk_document(FAQ_PATH)
    ids = [c["id"] for c in chunks]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)


# --- retrieval ---------------------------------------------------------------

def test_cosine_similarity_identical_and_orthogonal():
    v = np.array([0.3, 0.4, 0.5], dtype=np.float32)
    assert cosine_similarity(v, v) == pytest.approx(1.0)
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_rrf_orders_by_fused_rank():
    # doc 2 appears near the top of both rankings -> should win.
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 3, 1]])
    assert fused[0] == 2
    assert set(fused) == {1, 2, 3}


def _toy_corpus():
    chunks = [
        {"id": f"chunk_{i:03d}", "text": t, "n_tokens": count_tokens(t)}
        for i, t in enumerate(
            [
                "Employees accrue 15 vacation days in their first year.",
                "Enable SSO with SAML under Settings, then Security.",
                "Payroll runs semi-monthly on the 15th and last day.",
                "Multi-factor authentication uses TOTP or hardware keys.",
                "The Business plan costs 12 dollars per seat per month.",
            ]
        )
    ]
    # Deterministic unit vectors (no API) so FAISS has something to search.
    vecs = np.eye(5, dtype=np.float32)
    return chunks, build_faiss_index(vecs), build_bm25([c["text"] for c in chunks]), vecs


def test_hybrid_search_returns_between_two_and_five():
    chunks, faiss_index, bm25, vecs = _toy_corpus()
    out = hybrid_search("vacation days", vecs[0], faiss_index, bm25, chunks, top_k=4)
    assert 2 <= len(out) <= 5
    assert all(set(c) == {"id", "text", "n_tokens"} for c in out)


def test_hybrid_search_finds_keyword_match_via_bm25():
    chunks, faiss_index, bm25, vecs = _toy_corpus()
    # Query vector points to an unrelated chunk; BM25 should still surface "SSO".
    out = hybrid_search("How do I enable SSO?", vecs[0], faiss_index, bm25, chunks, top_k=3)
    assert any("SSO" in c["text"] for c in out)


# --- generation contract -----------------------------------------------------

def test_generate_answer_returns_exact_three_keys():
    chunks = [{"id": "chunk_000", "text": "New hires accrue 15 vacation days.", "n_tokens": 8}]
    client = _FakeClient({"system_answer": "New hires get 15 vacation days."})
    result = generate_answer("How many vacation days?", chunks, client=client, settings=_settings())
    assert set(result) == set(REQUIRED_KEYS)
    assert result["user_question"] == "How many vacation days?"
    assert result["chunks_related"] == ["New hires accrue 15 vacation days."]


# --- evaluator ---------------------------------------------------------------

def test_evaluator_returns_valid_score_and_reason():
    client = _FakeClient({"score": 8, "reason": "x" * 60})
    verdict = evaluate_answer("q", "a", ["chunk text"], client=client, settings=_settings())
    assert isinstance(verdict["score"], int)
    assert 0 <= verdict["score"] <= 10
    assert len(verdict["reason"]) >= 50


def test_evaluator_clamps_and_pads_bad_output():
    client = _FakeClient({"score": 42, "reason": "short"})
    verdict = evaluate_answer("q", "a", ["c"], client=client, settings=_settings())
    assert verdict["score"] == 10
    assert len(verdict["reason"]) >= 50


# --- helpers -----------------------------------------------------------------

def _settings():
    from src.config import Settings

    return Settings(
        api_key="test-key",
        embedding_model="text-embedding-3-small",
        chat_model="gpt-test",
        temperature=0.0,
        top_k=4,
    )
