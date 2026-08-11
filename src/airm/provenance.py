"""The header every experiment summary carries.

A run's numbers are only reproducible if you can say *when* they were produced
and *from which code*. Every experiment in this study already stamped a
``run_id``, which orders runs but does not identify them: two runs of the same
module a commit apart look identical in the artefact, and a run made against a
dirty tree looks exactly like one made from a clean checkout.

So one block, written by every experiment, at the top of every summary:

``run_id``
    The caller's id, or a fresh UTC timestamp. Unchanged from before.
``created``
    UTC ISO-8601, to the second. ``run_id`` is usually a timestamp too, but not
    when a caller passes ``--run-id faceted_full_max`` -- and a named run is
    exactly the case where the date is otherwise lost.
``git_commit`` / ``git_dirty``
    Which code produced this. ``git_dirty`` matters more than the sha: a summary
    written from an edited working tree cannot be reproduced from any commit, and
    that fact belongs in the artefact rather than in someone's memory.

Git is consulted through ``subprocess`` and every failure is swallowed to
``None`` -- provenance is metadata about a result, and failing to collect it must
never destroy the result itself.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .config import REPO_ROOT, new_run_id

#: Long enough to be unambiguous in this repo, short enough to read.
SHA_LENGTH = 12


def _git(*args: str, root: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(root or REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_commit(root: Path | None = None) -> str | None:
    sha = _git("rev-parse", "HEAD", root=root)
    return sha[:SHA_LENGTH] if sha else None


def git_dirty(root: Path | None = None) -> bool | None:
    """Whether tracked files differ from HEAD. ``None`` if git could not be read.

    ``None`` and ``False`` are deliberately different: "clean" is a claim, and an
    absent git is not evidence for it.
    """
    status = _git("status", "--porcelain", "--untracked-files=no", root=root)
    return None if status is None else bool(status)


def provenance(run_id: str | None = None, **extra) -> dict:
    """The standard header, ready to splat into a summary dict.

    Put it first so ``run_id`` leads the file::

        summary = {**provenance(run_id), "payload": ..., "by_source": ...}
    """
    return {
        "run_id": run_id or new_run_id(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        **extra,
    }
