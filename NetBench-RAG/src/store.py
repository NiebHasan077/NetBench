"""
Persistence layer for the NetBench-RAG pipeline.

Two storage backends:

VectorStore  — Qdrant local disk mode.
               Stores chunk vectors + full metadata payload.
               Point ID == position in chunks_cache (invariant for BM25 alignment).

BM25Index    — rank_bm25.BM25Okapi.
               Built from all chunks in the cache; pickled to disk.
               Position in BM25 corpus == Qdrant point ID.

Helper functions save_chunks_cache / load_chunks_cache pickle the list[Chunk]
that ties the two indices together.

Invariant
---------
  Qdrant point ID  ==  chunks_cache index  ==  BM25 corpus position

Always rebuild BM25 + save chunks_cache together after every indexing run.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi

from .chunker import Chunk


# ═══════════════════════════════════════════════════════════════════════════════
# Qdrant vector store
# ═══════════════════════════════════════════════════════════════════════════════

# Payload fields stored for every chunk point
_PAYLOAD_FIELDS = (
    "text",
    "context_prefix",
    "paper_title",
    "authors",
    "source_file",
    "parse_confidence",
    "parse_method",
    "section_heading",
    "section_number",
    "is_abstract",
    "chunk_index",
    "total_chunks",
)

_DISTANCE_MAP = {
    "Cosine": Distance.COSINE,
    "Dot": Distance.DOT,
    "Euclid": Distance.EUCLID,
}


class VectorStore:
    """
    Qdrant local-mode vector store.

    Parameters
    ----------
    qdrant_dir      : local path for on-disk Qdrant storage
    collection_name : name of the Qdrant collection
    dimension       : embedding vector dimension
    distance        : "Cosine" | "Dot" | "Euclid"
    indexed_fields  : payload field names to create keyword/float indices for
                      (speeds up metadata-filtered search)
    upsert_batch    : number of PointStructs per Qdrant upsert call
    """

    def __init__(
        self,
        qdrant_dir: str,
        collection_name: str,
        dimension: int = 1024,
        distance: str = "Cosine",
        indexed_fields: Optional[list[str]] = None,
        upsert_batch: int = 256,
    ) -> None:
        Path(qdrant_dir).mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=qdrant_dir)
        self.collection_name = collection_name
        self.dimension = dimension
        self.distance = _DISTANCE_MAP.get(distance, Distance.COSINE)
        self.indexed_fields = indexed_fields or []
        self.upsert_batch = upsert_batch

    # ── Collection lifecycle ──────────────────────────────────────────────────

    def create_collection_if_missing(self) -> None:
        """Create the collection + payload indices if they do not exist yet."""
        if self.client.collection_exists(self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.dimension,
                distance=self.distance,
            ),
        )

        # Build payload indices for fast metadata filtering
        for field in self.indexed_fields:
            # parse_confidence is a float; everything else is keyword/string
            if field == "parse_confidence":
                schema = PayloadSchemaType.FLOAT
            else:
                schema = PayloadSchemaType.KEYWORD
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=schema,
            )

    def drop_collection(self) -> None:
        """Permanently delete the collection (used by --force rebuild)."""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

    # ── Querying ──────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Return the number of points currently in the collection."""
        if not self.client.collection_exists(self.collection_name):
            return 0
        result = self.client.count(collection_name=self.collection_name)
        return result.count

    def get_indexed_source_files(self) -> set[str]:
        """
        Return the set of source_file values for all indexed chunks.
        Used to determine which papers can be skipped on a resumed run.
        """
        if not self.client.collection_exists(self.collection_name):
            return set()

        source_files: set[str] = set()
        offset = None

        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=["source_file"],
                with_vectors=False,
                limit=1000,
                offset=offset,
            )
            for point in results:
                sf = (point.payload or {}).get("source_file", "")
                if sf:
                    source_files.add(sf)
            if next_offset is None:
                break
            offset = next_offset

        return source_files

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 30,
    ) -> list[dict]:
        """
        Nearest-neighbour vector search.

        Returns
        -------
        list of dicts with keys: id, score, payload (all _PAYLOAD_FIELDS)
        """
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        )
        return [
            {"id": h.id, "score": h.score, "payload": h.payload}
            for h in response.points
        ]

    # ── Upsert ────────────────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        vectors: np.ndarray,
        start_id: int,
    ) -> None:
        """
        Insert chunks into Qdrant.

        Parameters
        ----------
        chunks   : list of Chunk objects
        vectors  : float32 array of shape (len(chunks), dim)
        start_id : first Qdrant point ID to use (must equal len(existing cache)
                   to maintain the ID == cache-index invariant)
        """
        for batch_start in range(0, len(chunks), self.upsert_batch):
            batch_chunks = chunks[batch_start : batch_start + self.upsert_batch]
            batch_vectors = vectors[batch_start : batch_start + self.upsert_batch]

            points = [
                PointStruct(
                    id=start_id + batch_start + i,
                    vector=vec.tolist(),
                    payload=_chunk_to_payload(chunk),
                )
                for i, (chunk, vec) in enumerate(zip(batch_chunks, batch_vectors))
            ]
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )


def _chunk_to_payload(chunk: Chunk) -> dict:
    """Serialise a Chunk to a flat Qdrant payload dict."""
    return {field: getattr(chunk, field) for field in _PAYLOAD_FIELDS}


def chunk_from_payload(payload: dict, point_id: int) -> Chunk:
    """
    Reconstruct a Chunk from a Qdrant payload dict.
    Used by the retriever to return full Chunk objects from search results.
    """
    return Chunk(
        text=payload.get("text", ""),
        context_prefix=payload.get("context_prefix", ""),
        paper_title=payload.get("paper_title", ""),
        authors=payload.get("authors", ""),
        source_file=payload.get("source_file", ""),
        parse_confidence=payload.get("parse_confidence", 0.0),
        parse_method=payload.get("parse_method", ""),
        section_heading=payload.get("section_heading", ""),
        section_number=payload.get("section_number", ""),
        is_abstract=payload.get("is_abstract", False),
        chunk_index=payload.get("chunk_index", point_id),
        total_chunks=payload.get("total_chunks", 0),
    )


# ── Factory ───────────────────────────────────────────────────────────────────

def vector_store_from_config(cfg: dict) -> VectorStore:
    """
    Instantiate a VectorStore from config.yaml.

    Parameters
    ----------
    cfg : full parsed config dict
    """
    paths = cfg.get("paths", {})
    qdrant_cfg = cfg.get("qdrant", {})
    embed_cfg = cfg.get("embedding", {})
    return VectorStore(
        qdrant_dir=paths.get("qdrant_dir", "data/qdrant_store/"),
        collection_name=qdrant_cfg.get("collection_name", "netbench_rag"),
        dimension=embed_cfg.get("dimension", 1024),
        distance=qdrant_cfg.get("distance", "Cosine"),
        indexed_fields=qdrant_cfg.get("indexed_fields", []),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# BM25 index
# ═══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str, lowercase: bool = True) -> list[str]:
    """
    Tokenise *text* for BM25.

    Splits on non-alphanumeric characters, removes single-char tokens,
    optionally lowercases.  Keeps numbers (useful for networking metrics).
    """
    if lowercase:
        text = text.lower()
    tokens = re.split(r"[^a-zA-Z0-9]+", text)
    return [t for t in tokens if len(t) > 1]


class BM25Index:
    """
    Wrapper around BM25Okapi with serialisation support.

    The BM25 model is built over the ``text`` field of each Chunk (body text
    only, no context prefix) so that term frequencies reflect paper content.

    Position in the BM25 corpus == Qdrant point ID == chunks_cache index.
    """

    def __init__(
        self,
        model: BM25Okapi,
        lowercase: bool = True,
    ) -> None:
        self._model = model
        self._lowercase = lowercase

    # ── Build ─────────────────────────────────────────────────────────────────

    @classmethod
    def build(cls, chunks: list[Chunk], lowercase: bool = True) -> "BM25Index":
        """Build a BM25 index from a list of Chunk objects."""
        corpus = [_tokenize(c.text, lowercase) for c in chunks]
        model = BM25Okapi(corpus)
        return cls(model, lowercase)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 30) -> list[tuple[int, float]]:
        """
        BM25 retrieval.

        Returns
        -------
        list of (corpus_position, score) sorted by score descending,
        truncated to top_k.  corpus_position == Qdrant point ID.
        """
        tokens = _tokenize(query, self._lowercase)
        scores = self._model.get_scores(tokens)

        # Partial sort: O(n + k log k) instead of O(n log n)
        top_indices = np.argpartition(scores, -min(top_k, len(scores)))[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(int(idx), float(scores[idx])) for idx in top_indices]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Pickle the BM25 model to *path*."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "lowercase": self._lowercase}, f)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        """Load a previously saved BM25Index from *path*."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["model"], data.get("lowercase", True))


# ═══════════════════════════════════════════════════════════════════════════════
# Chunk cache persistence
# ═══════════════════════════════════════════════════════════════════════════════

def save_chunks_cache(chunks: list[Chunk], path: str) -> None:
    """Pickle the full list of Chunk objects to *path*."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(chunks, f)


def load_chunks_cache(path: str) -> list[Chunk]:
    """Load and return the list of Chunk objects from *path*."""
    with open(path, "rb") as f:
        return pickle.load(f)
