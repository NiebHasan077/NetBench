"""
Embedding module for the NetBench-RAG pipeline.

Wraps BAAI/bge-large-en-v1.5 via sentence-transformers for:
  - Passage embedding during indexing  (empty passage instruction)
  - Query embedding during retrieval   (BGE query instruction prefix)

Both return L2-normalised float32 vectors suitable for cosine similarity.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from .chunker import Chunk


class Embedder:
    """
    Thin wrapper around SentenceTransformer for the RAG pipeline.

    Parameters
    ----------
    model_name          : HuggingFace model id (e.g. "BAAI/bge-large-en-v1.5")
    device              : "cuda" | "cpu"
    batch_size          : chunks per forward pass during indexing
    query_instruction   : prefix prepended to every query string
    passage_instruction : prefix prepended to passage text (empty for BGE large)
    normalize           : L2-normalise output vectors (required for cosine sim)
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 64,
        query_instruction: str = "",
        passage_instruction: str = "",
        normalize: bool = True,
    ) -> None:
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.passage_instruction = passage_instruction
        self.normalize = normalize

    # ── Indexing ──────────────────────────────────────────────────────────────

    def embed_chunks(
        self,
        chunks: list[Chunk],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of Chunk objects for indexing.

        Uses ``chunk.text_for_embedding`` (context_prefix + body text).
        If a non-empty passage_instruction is configured it is prepended here
        (BGE large v1.5 does not need one, so the default is empty).

        Returns
        -------
        np.ndarray of shape (len(chunks), embedding_dim), dtype float32
        """
        texts = [
            (self.passage_instruction + c.text_for_embedding).strip()
            for c in chunks
        ]
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string for retrieval.

        Prepends the BGE query instruction prefix before encoding.

        Returns
        -------
        np.ndarray of shape (embedding_dim,), dtype float32
        """
        text = (self.query_instruction + query).strip()
        vector = self.model.encode(
            [text],
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        return vector.astype(np.float32)


# ── Factory ───────────────────────────────────────────────────────────────────

def embedder_from_config(cfg: dict) -> Embedder:
    """
    Instantiate an Embedder from the ``embedding`` section of config.yaml.

    Parameters
    ----------
    cfg : full parsed config dict (yaml.safe_load output)
    """
    e = cfg.get("embedding", {})
    return Embedder(
        model_name=e.get("model", "BAAI/bge-large-en-v1.5"),
        device=e.get("device", "cuda"),
        batch_size=e.get("batch_size", 64),
        query_instruction=e.get("query_instruction", ""),
        passage_instruction=e.get("passage_instruction", ""),
        normalize=e.get("normalize", True),
    )
