"""Document loading and sentence-aware chunking with token-based overlap.

Strategy: split the document into sentences, then greedily pack sentences into a
chunk until it reaches ~CHUNK_TARGET_TOKENS, carrying a CHUNK_OVERLAP_TOKENS tail
of sentences into the next chunk. Sentence boundaries keep each chunk semantically
coherent (versus blind fixed-size splits that cut mid-thought), while the overlap
preserves context that straddles a boundary. Token counts use tiktoken so the
50-500 token rubric window is measured the same way the embedding model sees text.
"""
from __future__ import annotations

import re

import tiktoken

from src.config import CHUNK_MIN_TOKENS, CHUNK_OVERLAP_TOKENS, CHUNK_TARGET_TOKENS

_ENCODER = tiktoken.get_encoding("cl100k_base")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def count_tokens(text: str) -> int:
    """Number of tokens in ``text`` under the cl100k_base encoding."""
    return len(_ENCODER.encode(text))


def _split_sentences(text: str) -> list[str]:
    """Split text into trimmed sentences, ignoring blank lines."""
    sentences: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            sentences.extend(s.strip() for s in _SENTENCE_RE.split(line) if s.strip())
    return sentences


def _overlap_tail(sentences: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    """Return the trailing sentences (and their token sum) up to overlap_tokens."""
    tail: list[str] = []
    total = 0
    for sent in reversed(sentences):
        st = count_tokens(sent)
        if tail and total + st > overlap_tokens:
            break
        tail.insert(0, sent)
        total += st
    return tail, total


def _pack_sentences(sentences: list[str], target: int, overlap: int) -> list[str]:
    """Greedily pack sentences into ~target-token chunks with sentence overlap."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sent in sentences:
        st = count_tokens(sent)
        if current and current_tokens + st > target:
            chunks.append(" ".join(current))
            current, current_tokens = _overlap_tail(current, overlap)
        current.append(sent)
        current_tokens += st
    if current:
        chunks.append(" ".join(current))
    return _merge_short_tail(chunks)


def _merge_short_tail(chunks: list[str]) -> list[str]:
    """Fold a too-small final chunk into the previous one to honour the 50-token floor."""
    if len(chunks) >= 2 and count_tokens(chunks[-1]) < CHUNK_MIN_TOKENS:
        chunks[-2] = chunks[-2] + " " + chunks.pop()
    return chunks


def load_and_chunk_document(
    path: str,
    target_tokens: int = CHUNK_TARGET_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[dict]:
    """Load a UTF-8 text file and return a list of chunk dicts.

    Each chunk: ``{"id": "chunk_000", "text": str, "n_tokens": int}``.
    """
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    packed = _pack_sentences(_split_sentences(text), target_tokens, overlap_tokens)
    return [
        {"id": f"chunk_{i:03d}", "text": chunk, "n_tokens": count_tokens(chunk)}
        for i, chunk in enumerate(packed)
    ]
