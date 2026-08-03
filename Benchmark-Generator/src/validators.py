"""Stage 06 — Validation functions (pure, no I/O, no side effects).

Each check returns (passed: bool, reason: str).
On pass, reason is "".
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

from src.schema import Question

NGRAM_N = 8


# ── helpers ─────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


# ── 6.1 Schema ──────────────────────────────────────────────────────────────

def check_schema(record: dict) -> tuple[bool, str]:
    """Validate record against the Question Pydantic schema."""
    try:
        Question(**record)
        return True, ""
    except Exception as e:
        return False, f"schema: {e}"


# ── 6.2 Evidence quote substring ────────────────────────────────────────────

def check_evidence_quotes(
    record: dict,
    passage_text_by_id: dict[str, str],
) -> tuple[bool, str]:
    """Every evidence.quote must be a case-insensitive substring of its passage text."""
    evidence = record.get("evidence", [])
    if not evidence:
        return False, "evidence: no evidence items"
    for ev in evidence:
        if not isinstance(ev, dict):
            return False, "evidence: item is not a dict"
        pid = ev.get("passage_id", "")
        quote = ev.get("quote", "")
        if not pid:
            return False, "evidence: missing passage_id"
        if not quote:
            return False, f"evidence: empty quote for {pid!r}"
        passage = passage_text_by_id.get(pid, "")
        if not passage:
            return False, f"evidence: passage_id {pid!r} not found in corpus"
        if quote.lower() not in passage.lower():
            return False, f"evidence: quote not a substring of passage {pid!r}"
    return True, ""


# ── 6.3 Memorization-leak ───────────────────────────────────────────────────

def build_ngram_set(texts: list[str], n: int = NGRAM_N) -> frozenset[str]:
    """Build a frozenset of all word-level n-grams from a list of texts."""
    ngrams: set[str] = set()
    for text in texts:
        words = _tokenize(text)
        for i in range(len(words) - n + 1):
            ngrams.add(" ".join(words[i : i + n]))
    return frozenset(ngrams)


def check_memorization_ngram(
    reference_answer: str,
    training_ngrams: frozenset[str],
    n: int = NGRAM_N,
) -> tuple[bool, str]:
    """Reject if any n-gram of reference_answer appears verbatim in training text."""
    words = _tokenize(reference_answer)
    for i in range(len(words) - n + 1):
        ng = " ".join(words[i : i + n])
        if ng in training_ngrams:
            return False, f"memorization-ngram: verbatim {n}-gram in training: {ng!r}"
    return True, ""


def check_memorization_cosine(
    ref_embedding: np.ndarray,        # shape (D,), L2-normalised
    passage_embeddings: np.ndarray,   # shape (N, D), L2-normalised
    threshold: float = 0.92,
) -> tuple[bool, str]:
    """Reject if reference_answer embedding is too close to any training passage."""
    sims = passage_embeddings @ ref_embedding   # (N,)
    max_sim = float(sims.max())
    if max_sim > threshold:
        return False, f"memorization-cosine: max_sim={max_sim:.4f} > {threshold}"
    return True, ""


# ── 6.4 Embedding dedup ─────────────────────────────────────────────────────

def check_dedup(
    question_embedding: np.ndarray,              # shape (D,), L2-normalised
    accepted_embeddings: Optional[np.ndarray],   # shape (M, D) or None
    threshold: float = 0.88,
) -> tuple[bool, str]:
    """Reject if question is too similar to any already-accepted question."""
    if accepted_embeddings is None or len(accepted_embeddings) == 0:
        return True, ""
    sims = accepted_embeddings @ question_embedding   # (M,)
    max_sim = float(sims.max())
    if max_sim > threshold:
        return False, f"dedup: cosine={max_sim:.4f} > {threshold}"
    return True, ""
