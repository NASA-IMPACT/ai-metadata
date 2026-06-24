"""Evaluation metrics shared across experiments.

* Retrieval:   recall_at_k, mrr, ndcg_at_k
* Tool-use:    field_selection_f1, value_format_valid (ISO-8601, bbox)
* Faithfulness: hallucination scaffold (annotated correct / gap / fabrication)
* Cost:        token counting, Pareto frontier
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Retrieval metrics. ``ranked`` is a list of concept-ids best-first;
# ``relevant`` is the ground-truth set.
# --------------------------------------------------------------------------- #


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for cid in ranked[:k] if cid in relevant)
    return hits / len(relevant)


def mrr(ranked: list[str], relevant: set[str]) -> float:
    for i, cid in enumerate(ranked, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, cid in enumerate(ranked[:k], start=1)
        if cid in relevant
    )
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


_ISO_RANGE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?,\s*"
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?)?$"
)


def value_format_valid(field: str, value: str) -> bool:
    """Validate a few high-signal CMR value formats (temporal, bounding_box)."""
    value = value.strip()
    if field == "temporal":
        return bool(_ISO_RANGE.match(value))
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
