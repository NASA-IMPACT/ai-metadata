"""Local embeddings + in-memory cosine vector store.

Uses sentence-transformers (default: BAAI/bge-small-en-v1.5) so retrieval is
free, reproducible, and offline. The model is loaded lazily and cached per
process so importing this module is cheap.

Embeddings are **content-addressed cached** on disk (see ``embed``): the cache
key is a hash of the embedding-model name + the exact texts, so a hit is only
ever returned for byte-identical input. This makes the redundant rebuilds across
a run cheap (e.g. the retrieval and robustness stages build the *same* index for
a given representation) while staying correct under change: edit the corpus or a
renderer (including each auto-tune template) and the texts differ, the hash
differs, and the vectors are recomputed. Set ``AIRM_NO_EMBED_CACHE=1`` to bypass.

Two index modes:

* single-encoder -- one vector per record from a chosen renderer (Experiments 1, 4, 5).
* dual-encoder   -- separate content + metadata vectors, scores combined
                    (Experiment 3, "connected" representation).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import numpy as np

from . import DATA_DIR

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_CACHE_DIR = DATA_DIR / "embed_cache"


@lru_cache(maxsize=4)
def _load_model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def _cache_key(texts: list[str], model: str) -> str:
    """SHA-256 over the model name + every text (newline-joined, length-prefixed).

    Content-addressed: identical (model, texts) -> identical key. Texts are
    length-prefixed so no concatenation of distinct lists can collide.
    """
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    for t in texts:
        h.update(str(len(t)).encode("utf-8"))
        h.update(b":")
        h.update(t.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def embed(texts: list[str], model: str = DEFAULT_MODEL) -> np.ndarray:
    """Return L2-normalized embeddings (n, d) for cosine via dot product.

    Results are cached on disk under ``data/embed_cache/`` keyed on a hash of
    ``(model, texts)`` so repeated/identical embedding passes within and across
    runs are loaded instead of recomputed. The cache is correct-by-construction:
    any change to the texts (new corpus, edited renderer, tuned template) yields
    a new key. Bypass entirely with ``AIRM_NO_EMBED_CACHE=1``.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    use_cache = os.environ.get("AIRM_NO_EMBED_CACHE", "") not in ("1", "true", "True")
    cache_path = None
    if use_cache:
        cache_path = EMBED_CACHE_DIR / f"{_cache_key(texts, model)}.npy"
        if cache_path.exists():
            try:
                return np.load(cache_path)
            except Exception:
                pass  # corrupt/partial cache file -> fall through and recompute

    m = _load_model(model)
    vecs = m.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    arr = np.asarray(vecs, dtype=np.float32)

    if cache_path is not None:
        try:
            EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".npy.tmp")
            np.save(tmp, arr)
            tmp.replace(cache_path)  # atomic publish; readers never see a partial file
        except Exception:
            pass  # caching is best-effort; never fail an embed over disk issues
    return arr


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
