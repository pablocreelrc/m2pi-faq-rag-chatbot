"""Grounded answer generation (the "generation" half of RAG).

The retrieved chunks are assembled into a numbered context block and passed to the
LLM, which is instructed to answer using only that context. The returned object is
built deterministically to guarantee the exact three-key output contract required
by the brief: ``user_question``, ``system_answer``, ``chunks_related``.
"""
from __future__ import annotations

import json

from src.config import Settings
from src.llm import call_with_retry, get_client

REQUIRED_KEYS = ("user_question", "system_answer", "chunks_related")

SYSTEM_PROMPT = (
    "You are an HR-SaaS FAQ support assistant. Answer the user's question using ONLY "
    "the provided context passages. If the answer is not in the context, say you do "
    "not have that information. Be concise, accurate, and do not invent details."
)


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks as a numbered [D1]..[Dn] context block."""
    return "\n\n".join(f"[D{i}] {c['text']}" for i, c in enumerate(chunks, start=1))


def _validate(result: dict) -> dict:
    """Assert the output matches the exact three-key contract."""
    if set(result) != set(REQUIRED_KEYS):
        raise ValueError(f"Answer JSON must have exactly {REQUIRED_KEYS}, got {list(result)}")
    if not isinstance(result["system_answer"], str) or not result["system_answer"].strip():
        raise ValueError("system_answer must be a non-empty string")
    if not isinstance(result["chunks_related"], list):
        raise ValueError("chunks_related must be a list")
    return result


def generate_answer(question: str, chunks: list[dict], client=None, settings=None) -> dict:
    """Retrieve-then-generate: return {user_question, system_answer, chunks_related}."""
    settings = settings or Settings.from_env()
    client = client or get_client(settings)
    user_msg = (
        f"Context passages:\n{build_context(chunks)}\n\nQuestion: {question}\n\n"
        'Respond with a JSON object containing a single key "system_answer" whose '
        "value is your grounded answer to the question."
    )
    resp = call_with_retry(
        lambda: client.chat.completions.create(
            model=settings.chat_model,
            temperature=settings.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
    )
    payload = json.loads(resp.choices[0].message.content)
    result = {
        "user_question": question,
        "system_answer": str(payload.get("system_answer", "")).strip(),
        "chunks_related": [c["text"] for c in chunks],
    }
    return _validate(result)
