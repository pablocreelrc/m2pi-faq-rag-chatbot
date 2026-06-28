# Technical decisions

This note explains the main design choices behind the M2PI FAQ RAG chatbot and why each was made.

## Chunking: heading-aware, sentence-based, with token overlap

**Choice.** Split the document into sections by its headings (blank-line-delimited lines without
terminal punctuation), then within each section greedily pack sentences into ~90-token chunks
(`CHUNK_TARGET_TOKENS`) with a ~20-token overlap (`CHUNK_OVERLAP_TOKENS`). Text is never carried
across a section boundary, and each chunk is prefixed with its section heading. Sentences above the
500-token ceiling are hard-split, and any sub-floor chunk is merged with a neighbour in the same
section, so every chunk stays inside the 50–500 token window for any input document.

**Why.** Fixed-size character splits cut sentences mid-thought; packing whole sentences keeps each
chunk self-contained, and the overlap preserves facts that straddle a boundary. Making it
*heading-aware* matters more than it looks: headings have no terminal punctuation, so a naïve
sentence splitter glues a heading onto the previous section's text — e.g. a security paragraph
ending "...revokes their access. Single Sign-On and Security PeopleFlow supports..." would let a
security chunk score highly on a vacation query (cross-topic bleed). Sectioning prevents that and
gives each chunk its own heading as context. Token counts use `tiktoken` (cl100k_base), matching how
the embedding model sees text. On the supplied document this yields **21 chunks of 57–126 tokens**.

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

## Relevance selection: relative cosine gap over the fused pool

**Choice.** BM25 + dense fusion builds a candidate pool (recall); the final chunks are then selected
by cosine over that whole pool — keep chunks whose cosine trails the top hit by no more than
`RELEVANCE_GAP` (0.08), ordered by cosine, capped at `TOP_K` (4), with a `MIN_CHUNKS` (2) floor.

**Why.** A *fixed* cosine floor (e.g. 0.30) is the wrong tool: `text-embedding-3-small` scores
unrelated HR text in the ~0.16–0.35 band and related text in ~0.43–0.69, so any absolute cutoff
sits inside the noise overlap — it keeps junk on one query and clips good chunks on another. A
*relative* gap adapts to each query's own score scale. Selecting over the full fused pool (not just
RRF's top-k) also matters: RRF sometimes ranks the genuinely closest chunk below lexical near-misses
(an "annual discount" query had the correct billing chunk at cosine 0.46 but RRF buried it under
0.33 vacation chunks); selecting by cosine across the pool surfaces it. Net effect: the discount
query now returns two billing chunks instead of vacation noise, and the overtime query returns two
on-topic chunks instead of four. The 2-chunk floor keeps the rubric's 2–5 range satisfied.

## Output contract and the evaluator

**Choice.** `generate_answer` builds the result dict deterministically so the output always has
**exactly** `user_question`, `system_answer`, `chunks_related` (the retrieved passages supplied as
context) — the model only supplies the answer text, and bad/empty model output degrades to a safe
"I don't have that information" rather than crashing. The bonus evaluator is a separate LLM-as-judge
returning `{score, reason}`; it is **isolated** (a failure prints a note to stderr and never
suppresses the answer) and its verdict goes to stderr so stdout stays exactly three keys. The judge
is instructed that a correct "information not available" answer is high-quality when the chunks
genuinely lack the fact, so it does not penalize safe refusals.

**Why.** Enforcing the contract in code (rather than trusting the model to emit the right keys)
guarantees valid JSON for every query, and isolating the bonus QA step means it can never take down
the core deliverable.

## Robustness & security

**Choice.** Chunks are persisted as **JSON, not pickle**; the user question is length-capped before
any API call; ingested text is stripped of control characters; and retrieved context is fenced as
explicit "reference data only" with a system-prompt instruction never to follow instructions inside
it.

**Why.** `pickle.load` executes arbitrary code, so a tampered index file would be an RCE vector —
JSON cannot. Capping input avoids oversized-request crashes and wasted spend. Fencing context plus
the deterministic output assembly limits prompt-injection blast radius (direct injection attempts
are refused in testing), which matters as soon as the system ingests documents it does not fully
control.
