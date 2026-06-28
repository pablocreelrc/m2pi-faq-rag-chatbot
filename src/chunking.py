"""Document loading and heading-aware, sentence-based chunking with token overlap.

Strategy: the document is first split into sections by its headings (blank-line
delimited lines without terminal punctuation). Within each section, sentences are
greedily packed into ~CHUNK_TARGET_TOKENS chunks with a CHUNK_OVERLAP_TOKENS
sentence overlap, and text is NEVER carried across a section boundary -- so a
heading stays with its own content and never bleeds into a neighbouring topic.
Each chunk is prefixed with its section heading so it carries its own context.
A final pass hard-splits any over-ceiling chunk and merges any sub-floor chunk
into a neighbour, so every chunk stays within the 50-500 token window for any
document with at least ~50 tokens of content (a shorter whole document collapses
to a single chunk; the merge is a last resort that may join two tiny sections).
"""
from __future__ import annotations

import re

import tiktoken

from src.config import (
    CHUNK_MAX_TOKENS,
    CHUNK_MIN_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
)

_ENCODER = tiktoken.get_encoding("cl100k_base")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
# Control chars (C0/C1) except tab/newline -- defuses terminal escapes / byte injection.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def count_tokens(text: str) -> int:
    """Number of tokens in ``text`` under the cl100k_base encoding."""
    return len(_ENCODER.encode(text))


def _sanitize(text: str) -> str:
    """Remove control characters from ingested text."""
    return _CONTROL_RE.sub("", text)


def _is_heading(block: str) -> bool:
    """True if a block is a single short line without terminal punctuation."""
    return (
        "\n" not in block
        and len(block.split()) <= 12
        and not block.rstrip().endswith((".", "!", "?", ":"))
    )


def _split_sentences(text: str) -> list[str]:
    """Split a body block into trimmed sentences."""
    return [s.strip() for s in _SENTENCE_RE.split(text.replace("\n", " ")) if s.strip()]


def _parse_sections(text: str) -> list[tuple[str, list[str]]]:
    """Group the document into (heading, sentences) sections, keyed by its headings."""
    sections: list[tuple[str, list[str]]] = []
    heading, pending = "", []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        if _is_heading(block):
            if pending:
                sections.append((heading, pending))
                pending = []
            heading = block
        else:
            pending.extend(_split_sentences(block))
    if pending:
        sections.append((heading, pending))
    return sections


def _enforce_ceiling(chunks: list[str]) -> list[str]:
    """Hard-split any chunk above the ceiling into token windows (final safety net)."""
    out: list[str] = []
    for chunk in chunks:
        tokens = _ENCODER.encode(chunk)
        if len(tokens) <= CHUNK_MAX_TOKENS:
            out.append(chunk)
            continue
        step = CHUNK_TARGET_TOKENS
        out.extend(
            _ENCODER.decode(tokens[i : i + step]).strip()
            for i in range(0, len(tokens), step)
        )
    return out


def _overlap_tail(sentences: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    """Return the trailing sentences (and their token sum) up to overlap_tokens."""
    tail, total = [], 0
    for sent in reversed(sentences):
        st = count_tokens(sent)
        if total + st > overlap_tokens:  # never carry an oversize sentence as overlap
            break
        tail.insert(0, sent)
        total += st
    return tail, total


def _pack(sentences: list[str], target: int, overlap: int) -> list[str]:
    """Greedily pack sentences into ~target-token chunks with sentence overlap."""
    chunks, current, cur_tok = [], [], 0
    for sent in sentences:
        st = count_tokens(sent)
        if current and cur_tok + st > target:
            chunks.append(" ".join(current))
            current, cur_tok = _overlap_tail(current, overlap)
        current.append(sent)
        cur_tok += st
    if current:
        chunks.append(" ".join(current))
    return chunks


def _merge_short(chunks: list[str]) -> list[str]:
    """Fold any sub-floor chunk into an adjacent one so the 50-token floor holds."""
    i = 0
    while i < len(chunks):
        if len(chunks) > 1 and count_tokens(chunks[i]) < CHUNK_MIN_TOKENS:
            j = i - 1 if i > 0 else i + 1
            chunks[min(i, j)] = chunks[min(i, j)] + " " + chunks[max(i, j)]
            chunks.pop(max(i, j))
        else:
            i += 1
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
        text = _sanitize(fh.read())
    chunks: list[str] = []
    for heading, sentences in _parse_sections(text):
        for body in _pack(sentences, target_tokens, overlap_tokens):
            chunks.append(f"{heading}. {body}" if heading else body)
    # Final safety net: guarantee 50-500 tokens for ANY input (ceiling then floor).
    chunks = _merge_short(_enforce_ceiling(chunks))
    return [
        {"id": f"chunk_{i:03d}", "text": chunk, "n_tokens": count_tokens(chunk)}
        for i, chunk in enumerate(chunks)
    ]
