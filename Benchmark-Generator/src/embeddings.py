"""Sentence-transformer wrapper used by Stage 02 (and later Phase 4 clustering,
Phase 6 dedup/leak checks). Local-only — no API calls.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("embeddings")


def detect_device(preferred: Optional[str] = None) -> str:
    """Return a torch device string. Prefer the explicit override; else cuda:0 if available; else cpu."""
    if preferred:
        return preferred
    try:
        import torch  # local import so the module loads without torch installed (e.g., schema-only tests)
        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"


class Embedder:
    """Thin wrapper around SentenceTransformer.encode with sane defaults.

    Embeddings are L2-normalized by default so cosine similarity reduces to a dot product.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = detect_device(device)
        logger.info("loading embedding model: %s on %s", model_name, self.device)
        self.model = SentenceTransformer(model_name, device=self.device)
        self._dim: Optional[int] = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.model.get_sentence_embedding_dimension())
        return self._dim

    def embed(
        self,
        texts: Sequence[str],
        normalize: bool = True,
        show_progress: bool = True,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self.model.encode(
            list(texts),
            batch_size=batch_size or self.batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
