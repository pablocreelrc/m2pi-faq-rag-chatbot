# Technical decisions

This note explains the main design choices behind the M2PI FAQ RAG chatbot and why each was made.

## Chunking: sentence-aware with token overlap

**Choice.** Split the document on sentence boundaries, then greedily pack sentences into chunks of
~90 tokens (`CHUNK_TARGET_TOKENS`) with a ~20-token overlap (`CHUNK_OVERLAP_TOKENS`). A too-small
trailing chunk is merged into its predecessor so every chunk stays within the 50–500 token window.

**Why.** Fixed-size character splits cut sentences in half, which fragments meaning and hurts both
embedding quality and human readability of the retrieved context. Packing whole sentences keeps each
chunk self-contained. The overlap means a fact that spans a boundary still appears intact in at
least one chunk. Token counting uses `tiktoken` (cl100k_base) so chunk sizes are measured in the
same units the embedding model consumes. On the supplied document this yields **25 chunks of 55–90
tokens** — comfortably above the 20-chunk minimum and inside the size window.

## Embeddings: OpenAI `text-embedding-3-small`, L2-normalized

**Choice.** Embed all chunks in a single batched API call; normalize each vector to unit length.

**Why.** `text-embedding-3-small` (1536 dims) is a strong, inexpensive default and matches the
brief's `.env.example`. Normalizing vectors makes inner product equal to cosine similarity, which
lets the FAISS `IndexFlatIP` index return cosine scores directly with no extra computation.

## Vector search: FAISS `IndexFlatIP` (exact cosine)

**Choice.** An exact, brute-force inner-product index.

**Why.** With only 25 chunks, approximate nearest-neighbour (ANN) indexes add complexity and a
recall penalty for no benefit — exact search is instant at this scale. Inner product over
normalized vectors is exactly cosine similarity. The repository also includes an explicit
`cosine_similarity(a, b) = dot(a, b) / (||a|| · ||b||)` function to make the similarity math
visible and testable.

## Lexical search: BM25

**Choice.** A BM25Okapi index over whitespace-tokenized chunk text, run alongside the dense search.

**Why.** Dense embeddings capture meaning but can blur exact tokens. HR-SaaS questions frequently
hinge on literal strings — product names, acronyms (`SSO`, `MFA`, `SLA`, `SAML`), prices (`$12`),
and dates — where exact-term matching outperforms semantics. BM25 covers that gap.

## Fusion: Reciprocal Rank Fusion (RRF)

**Choice.** Combine the dense and sparse rankings with RRF: `score(doc) = Σ 1 / (k + rank)`,
`k = 60`, then take the top `TOP_K` (default 4, clamped to 2–5).

**Why.** Dense cosine scores and BM25 scores live on different, non-comparable scales, so they
cannot simply be added. RRF fuses by *rank* instead of raw score, which is robust and parameter-light.
The `k = 60` constant is the standard value from the IR literature (Cormack et al., 2009).

## Output contract and the evaluator

**Choice.** `generate_answer` builds the result dict deterministically so the output always has
**exactly** `user_question`, `system_answer`, `chunks_related` — the model only supplies the answer
text. The bonus evaluator is a separate LLM-as-judge returning `{score, reason}`; its verdict is
printed to stderr so the answer on stdout stays exactly three keys.

**Why.** Enforcing the contract in code (rather than trusting the model to emit the right keys)
guarantees valid JSON for every query. Keeping the evaluator output separate avoids polluting the
required three-key schema while still providing automated quality assurance.
