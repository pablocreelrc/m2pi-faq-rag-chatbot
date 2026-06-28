"""Query pipeline: embed query -> hybrid search -> generate -> JSON (+ optional eval).

Stages (rubric "Query Pipeline"):
  1. Embed the user's question (same model/dimension as the chunks).
  2. Hybrid vector + BM25 search retrieves the most relevant chunks.
  3. Assemble the chunks into context.
  4. The LLM generates a grounded answer; output is the strict three-key JSON.

The bonus evaluator runs unless --no-eval is passed; its verdict is printed
separately (to stderr) so the answer on stdout stays exactly three keys, and its
failure can never suppress the answer.

Run from the repo root:  python -m src.query "How many vacation days do new hires get?"
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import faiss

from src.config import INDEX_DIR, MAX_QUESTION_CHARS, Settings
from src.embeddings import embed_query
from src.evaluator import evaluate_answer
from src.generation import generate_answer
from src.llm import get_client
from src.retrieval import build_bm25, hybrid_search


def load_index(index_dir: str = INDEX_DIR):
    """Load the persisted FAISS index + chunks (JSON) and rebuild the BM25 index."""
    index_path = os.path.join(index_dir, "faiss.index")
    chunks_path = os.path.join(index_dir, "chunks.json")
    if not (os.path.exists(index_path) and os.path.exists(chunks_path)):
        raise RuntimeError(
            f"No index found in '{index_dir}'. Run 'python -m src.build_index' first."
        )
    faiss_index = faiss.read_index(index_path)
    with open(chunks_path, encoding="utf-8") as fh:
        chunks = json.load(fh)
    bm25 = build_bm25([c["text"] for c in chunks])
    return faiss_index, bm25, chunks


def answer_question(question, faiss_index, bm25, chunks, client, settings, evaluate=True):
    """Run the query pipeline; return (answer_dict, evaluation_dict_or_None).

    The evaluator is isolated: if it fails, the answer is still returned (eval=None).
    """
    qvec = embed_query(question, client=client, settings=settings)                 # Stage 1
    retrieved = hybrid_search(question, qvec, faiss_index, bm25, chunks, settings.top_k)  # Stage 2-3
    answer = generate_answer(question, retrieved, client=client, settings=settings)  # Stage 4
    if not evaluate:
        return answer, None
    try:
        evaluation = evaluate_answer(
            answer["user_question"], answer["system_answer"], answer["chunks_related"],
            client=client, settings=settings,
        )
    except Exception as exc:  # a bonus QA step must never suppress the answer
        print(f"# Evaluator skipped (error: {exc})", file=sys.stderr)
        evaluation = None
    return answer, evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the FAQ RAG chatbot a question.")
    parser.add_argument("question", help="the user question to answer")
    parser.add_argument("--no-eval", action="store_true", help="skip the bonus evaluator agent")
    args = parser.parse_args()

    question = args.question.strip()
    if not question:
        print(json.dumps({"error": "Question must not be empty."}), file=sys.stderr)
        raise SystemExit(2)
    if len(question) > MAX_QUESTION_CHARS:
        print(
            json.dumps({"error": f"Question exceeds {MAX_QUESTION_CHARS} characters."}),
            file=sys.stderr,
        )
        raise SystemExit(2)

    settings = Settings.from_env()
    client = get_client(settings)
    faiss_index, bm25, chunks = load_index()
    answer, evaluation = answer_question(
        question, faiss_index, bm25, chunks, client, settings, evaluate=not args.no_eval
    )
    print(json.dumps(answer, indent=2))  # ensure_ascii=True keeps control bytes inert
    if evaluation is not None:
        print("\n# Evaluator agent (bonus):", file=sys.stderr)
        print(json.dumps(evaluation, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
