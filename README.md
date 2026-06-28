# FAQ Support Chatbot (RAG)

> SoyHenry **AI Engineering** — Module 2 Integrator Project (M2PI).

An intelligent **FAQ support chatbot** for a fictional HR-SaaS company ("PeopleFlow"). Its
customer-support team receives hundreds of repetitive questions a day about policies, features,
and procedures that are already documented. This system answers those questions instantly: it
ingests the company's plain-text documentation, splits it into chunks, embeds them, and stores
them for retrieval. For each user question it runs a **hybrid vector + keyword search** over the
chunks, retrieves the most relevant ones, and has an LLM generate a grounded answer — returned as
**structured JSON** for transparency and easy integration:

```json
{
  "user_question": "How do I enable Single Sign-On (SSO)?",
  "system_answer": "SSO is available on the Business and Enterprise plans. To enable it, an Owner uploads the identity provider metadata in Settings > Security > SSO...",
  "chunks_related": ["SSO is available on the Business and Enterprise plans. To enable it...", "..."]
}
```

## What it does

- **Indexing pipeline** — loads the FAQ document, chunks it (heading-aware, 23 chunks of
  55–126 tokens), embeds every chunk with OpenAI, and stores a FAISS index + chunk text (JSON) on disk.
- **Query pipeline** — embeds the question, runs **hybrid retrieval** (dense FAISS cosine search
  fused with lexical BM25 via Reciprocal Rank Fusion), assembles the top chunks into context, and
  the LLM generates a grounded answer as strict three-key JSON.
- **Evaluator agent (bonus)** — an LLM-as-judge scores each answer `0–10` with a written reason
  across three dimensions (chunk relevance, answer grounding, completeness).

## Why this is RAG

The system **retrieves** relevant chunks first and only **then generates** an answer from them
(a visible two-step flow: `hybrid_search` → `generate_answer`). The LLM is instructed to use only
the retrieved context, which keeps answers **grounded** in the company's own documentation and
makes them **auditable** — the exact source chunks are returned alongside every answer. RAG also
lets the knowledge base be **updated by re-running the indexer**, with no model retraining.

## Setup

Requires **Python ≥ 3.11**.

```bash
# 1. Install dependencies
pip install -r requirements.txt          # or:  uv sync --extra dev

# 2. Configure your API key
cp .env.example .env                      # then edit .env and paste your real key
export OPENAI_API_KEY=sk-...              # (alternative to editing .env)
```

> `requirements.txt` pins are verified installable on Linux/Windows/macOS (x86_64 + Apple
> Silicon), CPython 3.11–3.13; `uv.lock` is committed for exact `uv` reproduction.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | — | OpenAI API key. Loaded from `.env`; never committed. |
| `EMBEDDING_MODEL` | no | `text-embedding-3-small` | Model for chunk + query embeddings. |
| `OPENAI_MODEL` | no | `gpt-5.4-mini` | Chat model for the answer + evaluator. |
| `TOP_K` | no | `4` | Max chunks per query; results below a cosine relevance floor are trimmed (min 2). |

## Run

Run from the repository root:

```bash
# Step 1 — build the index from data/faq_document.txt
python -m src.build_index
# -> Indexed 23 chunks (23 embeddings, dim=1536) | tokens 55-126 -> index/

# Step 2 — ask a question (answer prints as JSON to stdout)
python -m src.query "How many vacation days do new employees get?"
```

Example output (the evaluator verdict prints separately, to stderr, so stdout stays exactly the
three required keys):

```json
{
  "user_question": "How many vacation days do new employees get?",
  "system_answer": "Full-time employees accrue 15 paid vacation days during their first year of employment.",
  "chunks_related": ["...the relevant source passages..."]
}
```

Pass `--no-eval` to skip the evaluator agent. Five worked end-to-end examples (with evaluator
scores) are committed under `outputs/`.

## Technical decisions

- **Chunking — heading-aware, sentence-based, with token overlap.** The document is split into
  sections by its headings (so a heading never bleeds into a neighbouring topic), and within each
  section sentences are packed to ~90 tokens with a ~20-token overlap; each chunk is prefixed with
  its heading. Over-long sentences are hard-split and sub-floor chunks merged, so every chunk stays
  in the 50–500 token window (measured with `tiktoken`).
- **Search — hybrid (FAISS cosine + BM25, RRF fusion).** Dense embeddings capture semantic meaning
  but blur exact terms; BM25 nails literal matches the dense model misses — product names and
  acronyms like `SSO`, `MFA`, `SLA`, prices, and dates. Reciprocal Rank Fusion combines the two
  rankings into a candidate pool; embeddings are L2-normalized so FAISS inner product equals cosine.
  The final chunks are then selected by a **relative cosine gap** (keep chunks within `RELEVANCE_GAP`
  of the top hit, min two), which adapts to each query's score scale and surfaces the genuinely
  closest chunk even when RRF ranked it lower — so answers get tight, on-topic context, not padding.

More detail in [`reports/technical-decisions.md`](reports/technical-decisions.md).

## Project layout

```
data/faq_document.txt        source FAQ document (>=1000 words, yields 23 chunks)
src/config.py                settings (env) + tunable chunking/retrieval constants
src/llm.py                   OpenAI client + retry-with-backoff wrapper
src/chunking.py              load + heading-aware chunking (load_and_chunk_document)
src/embeddings.py            OpenAI embeddings, L2-normalized (generate_embeddings/embed_query)
src/retrieval.py             cosine_similarity, FAISS, BM25, RRF, hybrid_search
src/generation.py            context assembly + generate_answer (strict 3-key JSON)
src/evaluator.py             bonus LLM-as-judge (evaluate_answer)
src/build_index.py           indexing pipeline entrypoint  (python -m src.build_index)
src/query.py                 query pipeline entrypoint      (python -m src.query "...")
outputs/sample_queries.json  5 example query/answer pairs (proof it runs end-to-end)
outputs/sample_evaluations.json  evaluator scores for those same 5 queries
index/                       persisted FAISS index + chunks.json (git-ignored, regenerated)
reports/technical-decisions.md  deeper write-up of chunking + retrieval choices
tests/test_core.py           mocked unit + integration tests (no API key needed)
```

## Test

```bash
pytest -q          # or:  uv run pytest -q
```

Tests run **without an API key** — the OpenAI calls are mocked. The 20 tests cover chunk count and
token bounds, oversize-sentence splitting, no cross-section heading bleed, explicit cosine, RRF
ordering, BM25 keyword match, the relevance trim/floor, the exact three-key contract, graceful
handling of malformed model output, the evaluator's score/reason validation, and the retry wrapper
(retries transient errors, surfaces after exhaustion, does not retry 4xx).

## Robustness & security

- Chunks are stored as **JSON, not pickle** (pickle would execute code on load).
- Retrieved context is fenced as "reference data only" and the model is told never to follow
  instructions inside it; the strict three-key output is assembled in code. Direct prompt-injection
  attempts are refused in testing. (For fully untrusted documents this reduces, not eliminates,
  indirect-injection risk.)
- The user question is length-capped, ingested text is stripped of control characters, and the
  bonus evaluator is isolated so its failure can never suppress the answer.

## Known limitations

- Answers are only as good as `data/faq_document.txt`; out-of-scope questions are answered with
  "I don't have that information" by design.
- The corpus is small, so retrieval uses an exact FAISS index (`IndexFlatIP`); a production system
  with millions of chunks would switch to an approximate (ANN) index.
- The evaluator is an LLM-as-judge and is therefore advisory, not a guarantee of correctness.
- `tiktoken` downloads its encoding on first use, so the initial indexing run needs network access.

## Rubric reference

Built against the official M2PI brief + rubric (`../M2/L5-proyecto-integrador/fuente-oficial.md`).
Submission window on the campus: **2026-06-26 18:00 → 2026-06-29 18:00**.
