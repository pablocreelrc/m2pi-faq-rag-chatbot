"""Core tests — run WITHOUT an API key (all OpenAI calls are mocked).

Covered:
- Chunking: 20+ chunks, every chunk 50-500 tokens, full coverage, oversize-sentence
  splitting, and no cross-section heading bleed.
- Retrieval: explicit cosine, RRF ordering, BM25 keyword match, relevance trim/floor.
- Generation: exact three-key contract + graceful handling of bad model output.
- Evaluator: valid score/reason, clamping, non-numeric score, invalid JSON.
- llm.call_with_retry: retries transient errors, surfaces after exhaustion, skips 4xx.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from openai import APIConnectionError

from src.chunking import count_tokens, load_and_chunk_document
from src.config import CHUNK_MAX_TOKENS, CHUNK_MIN_TOKENS, FAQ_PATH
from src.evaluator import evaluate_answer
from src.generation import _NO_ANSWER, REQUIRED_KEYS, generate_answer
from src.llm import call_with_retry
from src.retrieval import (
    bm25_search,
    build_bm25,
    build_faiss_index,
    cosine_similarity,
    hybrid_search,
    reciprocal_rank_fusion,
)


# --- mock OpenAI client ------------------------------------------------------

def _client(content):
    """A fake OpenAI client whose chat.completions.create returns fixed content."""
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    create = lambda **_kw: resp  # noqa: E731
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _json_client(payload):
    return _client(json.dumps(payload))


def _settings():
    from src.config import Settings

    return Settings(
        api_key="test-key",
        embedding_model="text-embedding-3-small",
        chat_model="gpt-test",
        temperature=0.0,
        top_k=4,
    )


def _conn_error():
    return APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/x"))


# --- chunking ----------------------------------------------------------------

def test_chunking_meets_count_and_token_bounds():
    chunks = load_and_chunk_document(FAQ_PATH)
    assert len(chunks) >= 20
    for c in chunks:
        assert CHUNK_MIN_TOKENS <= c["n_tokens"] <= CHUNK_MAX_TOKENS


def test_chunking_covers_start_and_end_of_document():
    chunks = load_and_chunk_document(FAQ_PATH)
    assert "PeopleFlow" in chunks[0]["text"]
    assert "permanently deleted" in chunks[-1]["text"]


def test_chunk_ids_are_sequential_and_unique():
    ids = [c["id"] for c in load_and_chunk_document(FAQ_PATH)]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)


def test_chunking_splits_oversized_sentence(tmp_path):
    # One sentence (no terminal punctuation) far above the ceiling must be split.
    doc = "Big Section\n\n" + " ".join(["word"] * 1200)
    p = tmp_path / "doc.txt"
    p.write_text(doc, encoding="utf-8")
    chunks = load_and_chunk_document(str(p))
    assert len(chunks) > 1
    assert all(c["n_tokens"] <= CHUNK_MAX_TOKENS for c in chunks)


def test_chunking_ceiling_holds_for_adjacent_long_sentences(tmp_path):
    # Two large sentences must not combine (via overlap) into an over-ceiling chunk.
    doc = "Big Section\n\n" + " ".join(["alpha"] * 300) + ". " + " ".join(["beta"] * 300) + "."
    p = tmp_path / "c.txt"
    p.write_text(doc, encoding="utf-8")
    assert all(c["n_tokens"] <= CHUNK_MAX_TOKENS for c in load_and_chunk_document(str(p)))


def test_chunking_floor_holds_with_tiny_section(tmp_path):
    # A tiny standalone section must be merged so no chunk falls below the floor.
    doc = "Tiny\n\nHi.\n\nMain Section\n\n" + ("This is a normal policy sentence. " * 12)
    p = tmp_path / "f.txt"
    p.write_text(doc, encoding="utf-8")
    assert all(c["n_tokens"] >= CHUNK_MIN_TOKENS for c in load_and_chunk_document(str(p)))


def test_chunking_empty_document_returns_no_chunks(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("   \n\n  \n", encoding="utf-8")
    assert load_and_chunk_document(str(p)) == []


def test_chunking_has_no_cross_section_bleed(tmp_path):
    # Sections large enough to clear the floor must never share a chunk.
    doc = (
        "Alpha Section\n\n" + ("Apple is a sweet red fruit grown in orchards. " * 10) + "\n\n"
        "Beta Section\n\n" + ("Banana is a soft yellow tropical fruit. " * 10)
    )
    p = tmp_path / "d.txt"
    p.write_text(doc, encoding="utf-8")
    for c in load_and_chunk_document(str(p)):
        assert not ("Apple" in c["text"] and "Banana" in c["text"])


# --- retrieval ---------------------------------------------------------------

def test_cosine_similarity_identical_and_orthogonal():
    v = np.array([0.3, 0.4, 0.5], dtype=np.float32)
    assert cosine_similarity(v, v) == pytest.approx(1.0)
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_rrf_orders_by_fused_rank():
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
    vecs = np.eye(5, dtype=np.float32)
    return chunks, build_faiss_index(vecs), build_bm25([c["text"] for c in chunks]), vecs


def test_hybrid_search_returns_between_two_and_five():
    chunks, faiss_index, bm25, vecs = _toy_corpus()
    out = hybrid_search("vacation days", vecs[0], faiss_index, bm25, chunks, top_k=4)
    assert 2 <= len(out) <= 5
    assert all(set(c) == {"id", "text", "n_tokens"} for c in out)


def test_bm25_search_finds_keyword_match():
    chunks, _faiss_index, bm25, _vecs = _toy_corpus()
    ranked = bm25_search(bm25, "How do I enable SSO?", k=3)
    assert chunks[ranked[0]]["text"].count("SSO") >= 1


def test_hybrid_search_trims_irrelevant_chunks_to_floor():
    # Only one chunk matches the query vector; the trim falls back to the 2-chunk
    # floor with the true match ranked first.
    chunks, faiss_index, bm25, vecs = _toy_corpus()
    out = hybrid_search("zzz no lexical overlap", vecs[0], faiss_index, bm25, chunks, top_k=4)
    assert len(out) == 2
    assert out[0]["text"] == chunks[0]["text"]


# --- generation contract + error handling ------------------------------------

def test_generate_answer_returns_exact_three_keys():
    chunks = [{"id": "chunk_000", "text": "New hires accrue 15 vacation days.", "n_tokens": 8}]
    client = _json_client({"system_answer": "New hires get 15 vacation days."})
    result = generate_answer("How many vacation days?", chunks, client=client, settings=_settings())
    assert set(result) == set(REQUIRED_KEYS)
    assert result["user_question"] == "How many vacation days?"
    assert result["chunks_related"] == ["New hires accrue 15 vacation days."]


@pytest.mark.parametrize("content", [None, "not valid json {", json.dumps({"wrong_key": "x"})])
def test_generate_answer_degrades_gracefully_on_bad_output(content):
    chunks = [{"id": "c", "text": "x", "n_tokens": 5}]
    result = generate_answer("q", chunks, client=_client(content), settings=_settings())
    assert set(result) == set(REQUIRED_KEYS)
    assert result["system_answer"] == _NO_ANSWER  # never crashes, always valid contract


# --- evaluator ---------------------------------------------------------------

def test_evaluator_returns_valid_score_and_reason():
    client = _json_client({"score": 8, "reason": "x" * 60})
    verdict = evaluate_answer("q", "a", ["chunk text"], client=client, settings=_settings())
    assert isinstance(verdict["score"], int)
    assert 0 <= verdict["score"] <= 10
    assert len(verdict["reason"]) >= 50


def test_evaluator_clamps_and_pads_bad_output():
    client = _json_client({"score": 42, "reason": "short"})
    verdict = evaluate_answer("q", "a", ["c"], client=client, settings=_settings())
    assert verdict["score"] == 10
    assert len(verdict["reason"]) >= 50


def test_evaluator_handles_non_numeric_score_and_invalid_json():
    bad_score = evaluate_answer(
        "q", "a", ["c"], client=_json_client({"score": "high", "reason": "z" * 60}),
        settings=_settings(),
    )
    assert bad_score["score"] == 0
    invalid = evaluate_answer("q", "a", ["c"], client=_client(None), settings=_settings())
    assert invalid["score"] == 0
    assert len(invalid["reason"]) >= 50


# --- llm.call_with_retry -----------------------------------------------------

def test_retry_succeeds_after_transient_errors():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _conn_error()
        return "ok"

    assert call_with_retry(flaky, max_attempts=3, base_delay=0.0) == "ok"
    assert calls["n"] == 3


def test_retry_raises_runtimeerror_after_exhaustion():
    with pytest.raises(RuntimeError):
        call_with_retry(lambda: (_ for _ in ()).throw(_conn_error()), max_attempts=2, base_delay=0.0)


def test_retry_does_not_retry_non_retryable():
    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise ValueError("hard 4xx-like error")

    with pytest.raises(ValueError):
        call_with_retry(bad, max_attempts=3, base_delay=0.0)
    assert calls["n"] == 1  # not retried
