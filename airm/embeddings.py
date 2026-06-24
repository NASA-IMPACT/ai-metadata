"""Local embeddings + in-memory cosine vector store.

Uses sentence-transformers (default: BAAI/bge-small-en-v1.5) so retrieval is
free, reproducible, and offline. The model is loaded lazily and cached per
process so importing this module is cheap.

Two index modes:

* single-encoder -- one vector per record from a chosen renderer (Experiments 1, 4, 5).
* dual-encoder   -- separate content + metadata vectors, scores combined
                    (Experiment 3, "connected" representation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=4)
def _load_model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def embed(texts: list[str], model: str = DEFAULT_MODEL) -> np.ndarray:
    """Return L2-normalized embeddings (n, d) for cosine via dot product."""
    m = _load_model(model)
    vecs = m.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


@dataclass
class Index:
    """A single-encoder cosine index over rendered records."""

    model: str = DEFAULT_MODEL
    ids: list[str] = field(default_factory=list)
    matrix: np.ndarray | None = None

    @classmethod
    def build(
        cls,
        records: list[dict],
        render_fn: Callable[[dict], str],
        *,
        model: str = DEFAULT_MODEL,
        id_fn: Callable[[dict], str] = lambda r: r["meta"]["concept-id"],
    ) -> "Index":
        ids = [id_fn(r) for r in records]
        texts = [render_fn(r) for r in records]
        matrix = embed(texts, model=model)
        return cls(model=model, ids=ids, matrix=matrix)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return [(concept_id, score)] for the top-k records by cosine."""
        if self.matrix is None:
            raise RuntimeError("Index not built")
        q = embed([query], model=self.model)[0]
        scores = self.matrix @ q
        order = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in order]


@dataclass
class DualIndex:
    """A dual-encoder index: content and metadata embedded separately.

    Operationalizes the "connected" representation in Experiment 3. A metadata
    update only re-embeds the metadata side, which the experiment times against
    the single-encoder "embedded" baseline (which must re-embed the whole chunk).
    """

    model: str = DEFAULT_MODEL
    alpha: float = 0.5  # weight on the content score vs metadata score
    ids: list[str] = field(default_factory=list)
    content_matrix: np.ndarray | None = None
    metadata_matrix: np.ndarray | None = None

    @classmethod
    def build(
        cls,
        records: list[dict],
        content_fn: Callable[[dict], str],
        metadata_fn: Callable[[dict], str],
        *,
        model: str = DEFAULT_MODEL,
        alpha: float = 0.5,
        id_fn: Callable[[dict], str] = lambda r: r["meta"]["concept-id"],
    ) -> "DualIndex":
        ids = [id_fn(r) for r in records]
        content_matrix = embed([content_fn(r) for r in records], model=model)
        metadata_matrix = embed([metadata_fn(r) for r in records], model=model)
        return cls(
            model=model,
            alpha=alpha,
            ids=ids,
            content_matrix=content_matrix,
            metadata_matrix=metadata_matrix,
        )

    def reembed_metadata(
        self,
        records: list[dict],
        metadata_fn: Callable[[dict], str],
    ) -> None:
        """Re-embed only the metadata side (simulated update path)."""
        self.metadata_matrix = embed([metadata_fn(r) for r in records], model=self.model)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if self.content_matrix is None or self.metadata_matrix is None:
            raise RuntimeError("DualIndex not built")
        q = embed([query], model=self.model)[0]
        scores = self.alpha * (self.content_matrix @ q) + (1 - self.alpha) * (
            self.metadata_matrix @ q
        )
        order = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in order]
