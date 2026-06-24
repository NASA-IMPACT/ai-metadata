"""Query workload + ground-truth relevance.

A query is a natural-language question plus the set of CMR concept-ids that are
the correct answer(s). Stored in ``data/queries.yaml`` so it can be hand-checked
and version-controlled. The robustness loop perturbs queries (paraphrase /
reshuffle) using these primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
