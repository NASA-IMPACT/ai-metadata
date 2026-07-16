"""Query workload + ground-truth relevance.

A query is a natural-language question plus the set of CMR concept-ids that are
the correct answer(s). Stored in ``data/queries.yaml`` so it can be hand-checked
and version-controlled. The robustness loop perturbs queries (paraphrase /
reshuffle) using these primitives.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from . import QUERIES_PATH


@dataclass
class Query:
    id: str
    question: str
    relevant: list[str] = field(default_factory=list)  # ground-truth concept-ids
    notes: str = ""
    difficulty: str = ""  # "easy" | "medium" | "hard" (benchmark stratum)
    fields: list[str] = field(default_factory=list)  # CMR fields needed to answer
    paper: str = ""  # source paper the query was derived from

    def to_dict(self) -> dict:
        d = {"id": self.id, "question": self.question, "relevant": self.relevant}
        if self.difficulty:
            d["difficulty"] = self.difficulty
        if self.fields:
            d["fields"] = self.fields
        if self.paper:
            d["paper"] = self.paper
        if self.notes:
            d["notes"] = self.notes
        return d


def load_queries(path: Path = QUERIES_PATH) -> list[Query]:
    if not path.exists():
        raise FileNotFoundError(f"No query set at {path}.")
    raw = yaml.safe_load(path.read_text()) or []
    return [
        Query(
            id=q["id"],
            question=q["question"],
            relevant=q.get("relevant", []),
            notes=q.get("notes", ""),
            difficulty=q.get("difficulty", ""),
            fields=q.get("fields", []),
            paper=q.get("paper", ""),
        )
        for q in raw
    ]


def save_queries(queries: list[Query], path: Path = QUERIES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            [q.to_dict() for q in queries], sort_keys=False, allow_unicode=True
        )
    )


def _gold_cluster(q: Query) -> str:
    """Group key for a query: its ground-truth dataset (fallback to its id).

    Many queries are near-duplicate paraphrases that share one gold concept-id;
    grouping by that id lets callers keep a whole paraphrase cluster on one side
    of a split (no leakage) and cluster-bootstrap honest CIs.
    """
    return q.relevant[0] if q.relevant else q.id


def split_by_cluster(
    queries: list[Query],
    *,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
    cluster_key: Callable[[Query], str] = _gold_cluster,
) -> dict[str, list[Query]]:
    """Partition queries into train/val/test by *cluster*, not by row.

    Assigning whole clusters (paraphrases of the same gold dataset) to a single
    split prevents train/test leakage — otherwise a paraphrase of a tuning query
    can land in the test set and inflate the reported score. Returns
    ``{"train": [...], "val": [...], "test": [...]}``. The split is deterministic
    for a given ``seed``.
    """
    if not abs(sum(ratios) - 1.0) < 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")
    clusters: dict[str, list[Query]] = {}
    for q in queries:
        clusters.setdefault(cluster_key(q), []).append(q)
    keys = sorted(clusters)  # stable order before shuffling for reproducibility
    random.Random(seed).shuffle(keys)

    n = len(keys)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    buckets = {
        "train": keys[:n_train],
        "val": keys[n_train : n_train + n_val],
        "test": keys[n_train + n_val :],
    }
    return {
        split: [q for key in key_list for q in clusters[key]]
        for split, key_list in buckets.items()
    }


def seed_queries_from_corpus(records: list[dict]) -> list[Query]:
    """Heuristically seed a query set from corpus records.

    For each record, build a question from its title/variables and mark that
    record as the ground-truth relevant id. These are *starting points* meant
    to be hand-checked and broadened (a real query may have several relevant
    collections); they make the harness runnable before manual curation.
    """
    from .representations import extract_title, extract_variables

    queries: list[Query] = []
    for i, rec in enumerate(records):
        cid = rec["meta"]["concept-id"]
        variables = extract_variables(rec)
        title = extract_title(rec)
        if variables:
            question = f"Which dataset provides {variables[0].lower()} measurements?"
        else:
            question = f"Find the dataset titled '{title}'."
        queries.append(
            Query(
                id=f"q{i:03d}",
                question=question,
                relevant=[cid],
                notes="auto-seeded; verify and broaden relevance set",
            )
        )
    return queries
