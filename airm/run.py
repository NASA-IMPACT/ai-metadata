"""Experiment runner scaffolding: config, seeding, retrieval eval, JSONL logging.

This module provides the *common* machinery the per-experiment scripts will call
(coded later). It already implements the core retrieval-evaluation loop used by
Experiments 1, 3, and 4, so the smoke test can exercise the full pipeline.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from . import metrics
from .embeddings import Index
from .queries import Query


def run_timestamp() -> str:
    """UTC timestamp slug for a run directory, e.g. ``20260623T124800Z``."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def new_run_dir(base: Path, *, stamp: str | None = None) -> Path:
    """Create and return a fresh timestamped run directory under ``base``.

    Each experiment run writes into its own ``base/<timestamp>/`` folder so
    subsequent runs are logged separately rather than overwriting each other.
    Also (best-effort) repoints a ``base/latest`` symlink at the new directory.
    """
    stamp = stamp or run_timestamp()
    run_dir = base / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = base / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_dir.name)
    except OSError:
        pass  # symlinks may be unsupported; the timestamped dir is the source of truth
    return run_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class ExperimentConfig:
    name: str
    representations: list[str]  # keys into representations.RENDERERS
    models: list[str] = field(default_factory=list)
    k: int = 10
    seed: int = 0
    embedding_model: str = "BAAI/bge-small-en-v1.5"


@dataclass
class RetrievalResult:
    representation: str
    query_id: str
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    top_ids: list[str]


@dataclass
class RelevanceCoverage:
    """Ground-truth vs. corpus coverage — the retrieval eval's silent-failure map."""

    missing_relevant: dict[str, list[str]]  # query_id -> relevant ids absent from corpus
    empty_relevant: list[str]  # query_ids with no ground truth at all
    n_queries: int
    n_ok: int  # queries with at least one in-corpus relevant id


def check_relevance_coverage(
    records: list[dict], queries: list[Query]
) -> RelevanceCoverage:
    """Flag queries whose ground truth can never be retrieved from ``records``.

    A relevant concept-id that is not in the corpus (stale/typo'd id, or a
    filtered corpus) makes that query score a hard 0 across *every* representation
    with no error — quiet, systematic score depression. Likewise a query with an
    empty ``relevant`` list. Callers should ``log`` the result rather than let it
    pass silently (see CLAUDE.md coverage convention).
    """
    corpus_ids = {r["meta"]["concept-id"] for r in records}
    missing: dict[str, list[str]] = {}
    empty: list[str] = []
    n_ok = 0
    for q in queries:
        if not q.relevant:
            empty.append(q.id)
            continue
        absent = [c for c in q.relevant if c not in corpus_ids]
        if absent:
            missing[q.id] = absent
        if any(c in corpus_ids for c in q.relevant):
            n_ok += 1
    return RelevanceCoverage(missing, empty, len(queries), n_ok)


def evaluate_retrieval(
    records: list[dict],
    queries: list[Query],
    render_fn: Callable[[dict], str],
    representation: str,
    *,
    k: int = 10,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
) -> list[RetrievalResult]:
    """Build an index for one representation and score every query against it."""
    index = Index.build(records, render_fn, model=embedding_model)
    results: list[RetrievalResult] = []
    for q in queries:
        ranked = [cid for cid, _ in index.search(q.question, k=k)]
        rel = set(q.relevant)
        results.append(
            RetrievalResult(
                representation=representation,
                query_id=q.id,
                recall_at_k=metrics.recall_at_k(ranked, rel, k),
                mrr=metrics.mrr(ranked, rel),
                ndcg_at_k=metrics.ndcg_at_k(ranked, rel, k),
                top_ids=ranked,
            )
        )
    return results


def summarize(
    results: list[RetrievalResult],
    *,
    clusters: dict[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Aggregate per-representation means with bootstrap CIs on recall.

    When ``clusters`` (query_id -> group id, e.g. the query's ground-truth
    dataset) is given, the recall CI is computed with a cluster bootstrap so
    correlated paraphrase clusters don't shrink the interval artificially. Falls
    back to the per-query bootstrap when ``clusters`` is omitted.
    """
    by_rep: dict[str, list[RetrievalResult]] = {}
    for r in results:
        by_rep.setdefault(r.representation, []).append(r)

    summary: dict[str, dict[str, float]] = {}
    for rep, rows in by_rep.items():
        recalls = [r.recall_at_k for r in rows]
        if clusters is not None:
            cluster_ids = [clusters.get(r.query_id, r.query_id) for r in rows]
            mean, lo, hi = metrics.bootstrap_ci_clustered(recalls, cluster_ids)
        else:
            mean, lo, hi = metrics.bootstrap_ci(recalls)
        summary[rep] = {
            "recall_at_k_mean": mean,
            "recall_at_k_lo": lo,
            "recall_at_k_hi": hi,
            "mrr_mean": float(np.mean([r.mrr for r in rows])),
            "ndcg_mean": float(np.mean([r.ndcg_at_k for r in rows])),
            "n_queries": len(rows),
        }
    return summary


def write_jsonl(rows: list, path: Path, *, append: bool = False) -> None:
    """Write dataclass/dict rows as JSONL.

    Overwrites ``path`` by default so re-running an experiment produces fresh
    results instead of stacking onto a previous run's file. Pass ``append=True``
    to stream rows across multiple calls into one file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w") as fh:
        for row in rows:
            fh.write(json.dumps(asdict(row) if hasattr(row, "__dataclass_fields__") else row) + "\n")
