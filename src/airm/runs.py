"""Finding a past run by what it *was*, not by what its folder is called.

Run directories are named with a plain UTC timestamp, so the folder tells you
when a run happened and nothing else. Everything that identifies it -- payload,
pooling rule, query set, encoder, model matrix -- lives in the summary it wrote.
That is the right way round: a descriptive folder name is a second source of
truth that drifts from the artefact and cannot be extended without renaming
directories.

The cost is that callers can no longer construct a path from a convention. This
module pays that cost once::

    latest("chunked_summary.json", payload="faceted", pool="max")

Ties break on ``created``, so the newest matching run wins and re-running an
experiment supersedes its predecessor without anyone deleting anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .config import RUNS_DIR


class RunNotFound(LookupError):
    """No run matched. Carries what was searched for, because the usual cause is
    a typo in a filter rather than a genuinely missing run."""


def iter_summaries(filename: str, *, runs_dir: Path | None = None) -> Iterator[tuple[Path, dict]]:
    """Every readable ``filename`` under ``runs/``, as ``(path, parsed)``.

    A malformed or partial summary is skipped rather than raised on: a crashed
    run should not stop you finding the completed ones.
    """
    root = runs_dir or RUNS_DIR
    if not root.exists():
        return
    for path in sorted(root.glob(f"*/{filename}")):
        try:
            yield path, json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue


def find(filename: str, *, runs_dir: Path | None = None, **match) -> list[tuple[Path, dict]]:
    """All runs whose summary matches every ``key=value`` in ``match``.

    Newest first by ``created``. A summary written before provenance existed has
    no ``created`` and sorts last -- superseded by any run that records one,
    which is the behaviour you want when re-running an experiment.
    """
    hits = [
        (path, data)
        for path, data in iter_summaries(filename, runs_dir=runs_dir)
        if all(data.get(k) == v for k, v in match.items())
    ]
    return sorted(hits, key=lambda pair: pair[1].get("created") or "", reverse=True)


def latest(filename: str, *, runs_dir: Path | None = None, **match) -> tuple[Path, dict]:
    """The newest matching run, or :class:`RunNotFound`."""
    hits = find(filename, runs_dir=runs_dir, **match)
    if not hits:
        criteria = ", ".join(f"{k}={v!r}" for k, v in match.items()) or "(no filters)"
        raise RunNotFound(f"No {filename} under {runs_dir or RUNS_DIR} matching {criteria}.")
    return hits[0]
