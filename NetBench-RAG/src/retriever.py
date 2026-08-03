"""
Retrieval pipeline for the NetBench-RAG system.

Three stages:
  1. Dual search      — vector (Qdrant) + keyword (BM25), top-30 each
  2. RRF fusion       — Reciprocal Rank Fusion, top-20 candidates
  3. Cross-encoder    — BAAI/bge-reranker-v2-m3, final top-5

Acronym expansion is applied to the raw query before BM25 only; the vector
query uses the original text (the embedding model handles semantics).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import CrossEncoder

from .chunker import Chunk
from .embedder import Embedder
from .store import BM25Index, VectorStore


# ═══════════════════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalResult:
    """
    The output of one retrieval call.

    ``chunks``        : final top-N Chunk objects, ordered by reranker score
    ``reranker_scores``: parallel list of cross-encoder scores
    ``rrf_candidates`` : IDs of the top-M candidates passed to the reranker
    """
    chunks: list[Chunk]
    reranker_scores: list[float]
    rrf_candidates: list[int] = field(default_factory=list)

    def top_chunks(self, n: int | None = None) -> list[Chunk]:
        """Return the top-n chunks (default: all)."""
        return self.chunks[:n] if n else self.chunks


# ═══════════════════════════════════════════════════════════════════════════════
# Acronym expansion
# ═══════════════════════════════════════════════════════════════════════════════

def _build_acronym_pattern(expansion_map: dict[str, str]) -> re.Pattern | None:
    """
    Build a single compiled regex that matches any key in *expansion_map*
    as a whole word (case-sensitive, word boundaries on both sides).

    Returns None if the map is empty.
    """
    if not expansion_map:
        return None
    # Escape each key and join as alternatives, longest first to avoid
    # partial matches (e.g. "MPTCP" before "TCP")
    keys = sorted(expansion_map.keys(), key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b"
    return re.compile(pattern)


def expand_acronyms(query: str, expansion_map: dict[str, str]) -> str:
    """
    Replace HPN acronyms in *query* with their full forms.

    Each matched acronym is replaced by "acronym full_form" so the original
    term is preserved for exact-match BM25 while also adding the expanded terms.

    Example
    -------
    "How does BBR estimate RTT?" →
    "How does BBR bottleneck bandwidth and round-trip propagation estimate RTT round trip time?"
    """
    if not expansion_map:
        return query

    pattern = _build_acronym_pattern(expansion_map)
    if pattern is None:
        return query

    def _replace(m: re.Match) -> str:
        acronym = m.group(1)
        expansion = expansion_map.get(acronym, acronym)
        return f"{acronym} {expansion}"

    return pattern.sub(_replace, query)


# ═══════════════════════════════════════════════════════════════════════════════
# RRF
# ═══════════════════════════════════════════════════════════════════════════════

def _reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = 60,
    top_n: int = 20,
) -> list[int]:
    """
    Merge multiple ranked lists of candidate IDs using Reciprocal Rank Fusion.

    Parameters
    ----------
    ranked_lists : each inner list is a ranked sequence of chunk IDs
                   (index 0 = highest rank)
    k            : RRF smoothing constant (standard: 60)
    top_n        : number of top candidates to return

    Returns
    -------
    List of chunk IDs sorted by descending RRF score, truncated to top_n.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):   # 1-based rank
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)
    return sorted_ids[:top_n]


# ═══════════════════════════════════════════════════════════════════════════════
# Retriever
# ═══════════════════════════════════════════════════════════════════════════════

class Retriever:
    """
    Full three-stage retrieval pipeline.

    Parameters
    ----------
    vector_store    : VectorStore (Qdrant wrapper)
    bm25_index      : BM25Index
    chunks_cache    : list[Chunk] — positional lookup; index == Qdrant point ID
    embedder        : Embedder — for query vector
    reranker        : CrossEncoder — for stage 3
    vector_top_k    : candidates from vector search
    bm25_top_k      : candidates from BM25 search
    rrf_k           : RRF smoothing constant
    rrf_top_n       : candidates passed to reranker
    reranker_top_n  : final chunks returned to the caller
    acronym_map     : dict of acronym→expansion applied to BM25 query only
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        chunks_cache: list[Chunk],
        embedder: Embedder,
        reranker: CrossEncoder,
        vector_top_k: int = 30,
        bm25_top_k: int = 30,
        rrf_k: int = 60,
        rrf_top_n: int = 20,
        reranker_top_n: int = 5,
        acronym_map: dict[str, str] | None = None,
    ) -> None:
        self._vs = vector_store
        self._bm25 = bm25_index
        self._cache = chunks_cache
        self._embedder = embedder
        self._reranker = reranker
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k
        self.rrf_top_n = rrf_top_n
        self.reranker_top_n = reranker_top_n
        self._acronym_map: dict[str, str] = acronym_map or {}

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Run the full retrieval pipeline for *query*.

        Returns a RetrievalResult with the final top chunks in reranker-score order.
        """
        # ── Stage 1: Dual search ─────────────────────────────────────────────
        vec_ids = self._vector_search(query)
        bm25_ids = self._bm25_search(query)

        # ── Stage 2: RRF fusion ──────────────────────────────────────────────
        rrf_ids = _reciprocal_rank_fusion(
            [vec_ids, bm25_ids],
            k=self.rrf_k,
            top_n=self.rrf_top_n,
        )

        # ── Stage 3: Cross-encoder reranking ─────────────────────────────────
        candidates = [self._cache[i] for i in rrf_ids]
        pairs = [(query, c.text) for c in candidates]
        raw_scores: np.ndarray = self._reranker.predict(pairs, show_progress_bar=False)

        # Sort by score descending, take top reranker_top_n
        order = np.argsort(raw_scores)[::-1][: self.reranker_top_n]
        final_chunks = [candidates[i] for i in order]
        final_scores = [float(raw_scores[i]) for i in order]

        return RetrievalResult(
            chunks=final_chunks,
            reranker_scores=final_scores,
            rrf_candidates=rrf_ids,
        )

    # ── Stage 1 helpers ───────────────────────────────────────────────────────

    def _vector_search(self, query: str) -> list[int]:
        """Embed the query and return top-K chunk IDs from Qdrant."""
        qvec = self._embedder.embed_query(query)
        hits = self._vs.search(qvec, top_k=self.vector_top_k)
        return [h["id"] for h in hits]

    def _bm25_search(self, query: str) -> list[int]:
        """Expand acronyms, run BM25, return top-K chunk IDs."""
        expanded = expand_acronyms(query, self._acronym_map)
        hits = self._bm25.search(expanded, top_k=self.bm25_top_k)
        return [idx for idx, _score in hits]


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

def retriever_from_config(cfg: dict) -> Retriever:
    """
    Build a fully-loaded Retriever from config.yaml.

    Loads: Qdrant collection, BM25 index, chunks cache, embedding model,
    cross-encoder reranker.  All models are placed on the configured device.

    Parameters
    ----------
    cfg : full parsed config dict (yaml.safe_load output)
    """
    from .embedder import embedder_from_config
    from .store import BM25Index, load_chunks_cache, vector_store_from_config

    r = cfg.get("retrieval", {})
    paths = cfg.get("paths", {})

    vector_store = vector_store_from_config(cfg)
    bm25_index = BM25Index.load(paths["bm25_index"])
    chunks_cache = load_chunks_cache(paths["chunks_cache"])
    embedder = embedder_from_config(cfg)

    reranker = CrossEncoder(
        r.get("reranker_model", "BAAI/bge-reranker-v2-m3"),
        device=r.get("reranker_device", "cuda"),
    )

    return Retriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        chunks_cache=chunks_cache,
        embedder=embedder,
        reranker=reranker,
        vector_top_k=r.get("vector_top_k", 30),
        bm25_top_k=r.get("bm25_top_k", 30),
        rrf_k=r.get("rrf_k", 60),
        rrf_top_n=r.get("rrf_top_n", 20),
        reranker_top_n=r.get("reranker_top_n", 5),
        acronym_map=r.get("acronym_expansion", {}),
    )
