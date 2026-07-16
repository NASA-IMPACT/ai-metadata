"""Evaluation metrics shared across experiments.

* Retrieval:   recall_at_k, mrr, ndcg_at_k
* Tool-use:    field_selection_f1, value_format_valid (ISO-8601, bbox)
* Faithfulness: hallucination scaffold (annotated correct / gap / fabrication)
* Cost:        token counting, Pareto frontier
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

# --------------------------------------------------------------------------- #
# Retrieval metrics. ``ranked`` is a list of concept-ids best-first;
# ``relevant`` is the ground-truth set.
# --------------------------------------------------------------------------- #


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    # Count *distinct* relevant ids hit, so a ranked list with a repeated id
    # (possible if the corpus has duplicate concept-ids; search does not dedup)
    # can never score above 1.0.
    hits = len(set(ranked[:k]) & relevant)
    return hits / len(relevant)


def mrr(ranked: list[str], relevant: set[str]) -> float:
    for i, cid in enumerate(ranked, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    # Credit each relevant id at most once, at its first (best) rank, so a
    # duplicated id in ``ranked`` cannot inflate DCG past the ideal.
    seen: set[str] = set()
    dcg = 0.0
    for i, cid in enumerate(ranked[:k], start=1):
        if cid in relevant and cid not in seen:
            seen.add(cid)
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


# --------------------------------------------------------------------------- #
# Tool-use metrics (Experiment 2).
# --------------------------------------------------------------------------- #


def field_selection_f1(predicted: set[str], gold: set[str]) -> float:
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    tp = len(predicted & gold)
    prec = tp / len(predicted)
    rec = tp / len(gold)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def _parse_iso8601(s: str) -> datetime | None:
    """Parse an ISO-8601 UTC timestamp, tolerating the forms CMR actually emits.

    Accepts a trailing ``Z``, fractional seconds, and numeric ``+HH:MM`` offsets;
    range-checks month/day/hour/minute/second via ``datetime`` (so ``2020-13-40``
    is rejected). Returns a ``datetime`` on success, ``None`` otherwise.
    """
    s = s.strip()
    if not s:
        return None
    # datetime.fromisoformat accepts numeric offsets and fractional seconds; it
    # only chokes on the literal ``Z``, so normalize that to ``+00:00`` first.
    normalized = s[:-1] + "+00:00" if s.endswith("Z") else s
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def value_format_valid(field: str, value: str) -> bool:
    """Validate a few high-signal CMR value formats (temporal, bounding_box)."""
    value = value.strip()
    if field == "temporal":
        # "start[,end]" — start required, end optional (open-ended range). Each
        # endpoint must be a valid ISO-8601 datetime and start must not follow end.
        parts = [p.strip() for p in value.split(",")]
        if len(parts) not in (1, 2):
            return False
        start = _parse_iso8601(parts[0])
        if start is None:
            return False
        if len(parts) == 2 and parts[1]:
            end = _parse_iso8601(parts[1])
            if end is None or end < start:
                return False
        return True
    if field == "bounding_box":
        parts = value.split(",")
        if len(parts) != 4:
            return False
        try:
            w, s, e, n = (float(p) for p in parts)
        except ValueError:
            return False
        return -180 <= w <= 180 and -180 <= e <= 180 and -90 <= s <= 90 and -90 <= n <= 90
    # default: non-empty is acceptable
    return bool(value)


# --------------------------------------------------------------------------- #
# Faithfulness scaffold (Experiment 4). Labels follow the report's
# "Input Matters" annotation method: correct / correct-gap / fabrication.
# --------------------------------------------------------------------------- #

FAITHFULNESS_LABELS = ("correct", "correct_gap_report", "fabrication")


@dataclass
class FaithfulnessTally:
    correct: int = 0
    correct_gap_report: int = 0
    fabrication: int = 0

    def add(self, label: str) -> None:
        if label not in FAITHFULNESS_LABELS:
            raise ValueError(f"unknown label {label!r}")
        setattr(self, label, getattr(self, label) + 1)

    @property
    def total(self) -> int:
        return self.correct + self.correct_gap_report + self.fabrication

    def rate(self, label: str) -> float:
        return getattr(self, label) / self.total if self.total else 0.0


# --------------------------------------------------------------------------- #
# Cost + Pareto.
# --------------------------------------------------------------------------- #


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Approximate token count via tiktoken (provider-agnostic proxy)."""
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # fallback: ~4 chars/token
        return max(1, len(text) // 4)


def pareto_frontier(points: list[tuple[str, float, float]]) -> list[str]:
    """Return labels on the accuracy-vs-cost Pareto frontier.

    points: [(label, accuracy_higher_better, cost_lower_better)].
    A point is dominated if another has >= accuracy and <= cost (strictly
    better on at least one axis).
    """
    frontier = []
    for label, acc, cost in points:
        dominated = any(
            (a2 >= acc and c2 <= cost and (a2 > acc or c2 < cost))
            for l2, a2, c2 in points
            if l2 != label
        )
        if not dominated:
            frontier.append(label)
    return frontier


# --------------------------------------------------------------------------- #
# Bootstrap CIs (used by the robustness loop).
# --------------------------------------------------------------------------- #


def bootstrap_ci(
    values: list[float], n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) percentile bootstrap CI for a list of per-query scores."""
    import numpy as np

    if not values:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = arr[rng.integers(0, len(arr), size=(n_resamples, len(arr)))].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (float(arr.mean()), lo, hi)


def bootstrap_ci_clustered(
    values: list[float],
    clusters: list,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Cluster (block) bootstrap CI for correlated per-query scores.

    When many queries are near-duplicate paraphrases of the same ground-truth
    dataset, treating them as independent (:func:`bootstrap_ci`) understates the
    CI width. This resamples whole *clusters* (e.g. one per gold dataset) with
    replacement and pools their member scores, so correlated within-cluster
    duplicates no longer inflate the effective sample size.

    ``values`` and ``clusters`` are parallel lists; ``clusters[i]`` is the group
    id (hashable) of ``values[i]``. Falls back to the per-observation bootstrap
    when there is 0 or 1 distinct cluster.
    """
    import numpy as np

    if not values:
        return (0.0, 0.0, 0.0)
    groups: dict = {}
    for v, c in zip(values, clusters):
        groups.setdefault(c, []).append(float(v))
    keys = list(groups)
    point = float(np.mean(values))
    if len(keys) < 2:
        return bootstrap_ci(values, n_resamples=n_resamples, alpha=alpha, seed=seed)

    rng = np.random.default_rng(seed)
    members = [np.asarray(groups[k], dtype=float) for k in keys]
    means = np.empty(n_resamples, dtype=float)
    n = len(keys)
    for r in range(n_resamples):
        picked = rng.integers(0, n, size=n)
        pooled = np.concatenate([members[i] for i in picked])
        means[r] = pooled.mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (point, lo, hi)
