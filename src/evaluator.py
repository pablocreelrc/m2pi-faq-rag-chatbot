"""Bonus evaluator agent: an LLM-as-judge that scores answer quality.

Given the question, the system's answer, and the retrieved chunks, it returns a
``{"score": int 0-10, "reason": str}`` verdict assessing three dimensions: chunk
relevance, answer quality (is the answer grounded in the chunks), and completeness.
This adds automated quality assurance / hallucination screening to the pipeline.
"""
from __future__ import annotations

import json

from src.config import Settings
from src.llm import call_with_retry, get_client

JUDGE_PROMPT = (
    "You are a strict QA evaluator for a RAG-based FAQ system. Given a user question, "
    "the system's answer, and the retrieved context chunks, score the answer from 0 to "
    "10 considering three dimensions: (1) chunk relevance - do the chunks relate to the "
    "question; (2) answer quality - does the answer use information from the chunks "
    "without inventing facts; (3) completeness - does it fully address the question. "
    'Return a JSON object with keys "score" (integer 0-10) and "reason" (at least 50 '
    "characters, citing specific observations such as which chunks were used)."
)

_FALLBACK = (
    " Score reflects chunk relevance to the question, how well the answer is grounded "
    "in those chunks, and the completeness of the response."
)


def _normalize_verdict(payload: dict) -> dict:
    """Coerce the model output into a valid score (0-10 int) and >=50-char reason."""
    try:
        score = int(round(float(payload.get("score", 0))))
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(10, score))
    reason = str(payload.get("reason", "")).strip()
    if len(reason) < 50:
        reason = (reason + _FALLBACK).strip()
    return {"score": score, "reason": reason}


def evaluate_answer(
    user_question: str,
    system_answer: str,
    chunks_related: list[str],
    client=None,
    settings=None,
) -> dict:
    """Return {"score": int 0-10, "reason": str} judging the RAG answer."""
    settings = settings or Settings.from_env()
    client = client or get_client(settings)
    context = "\n\n".join(f"[D{i}] {c}" for i, c in enumerate(chunks_related, start=1))
    user_msg = f"Question: {user_question}\n\nAnswer: {system_answer}\n\nRetrieved chunks:\n{context}"
    resp = call_with_retry(
        lambda: client.chat.completions.create(
            model=settings.chat_model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
    )
    return _normalize_verdict(json.loads(resp.choices[0].message.content))
