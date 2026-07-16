"""Validate + improve loop framework (Report.md §6, applied per experiment).

Two stages, both requested by the user:

A. Robustness  -- re-run a scoring function under perturbations (paraphrased
                  queries, shuffled corpus order, multiple seeds), then report
                  bootstrap CIs and flag findings whose CIs overlap as "not
                  robust". No design changes.

B. Auto-tune   -- an agentic optimizer: read the metric report, ask an LLM to
                  critique and rewrite the weakest representation's renderer text
                  / prompt, apply to a *versioned* candidate, re-run the scorer,
                  keep the edit only if the headline metric improves, and stop
                  after ``patience`` rounds without gain.

This module is the *framework*. Each experiment supplies:
  * ``scorer(variant) -> float`` for the headline metric, and
  * for auto-tune, a ``Candidate`` describing the editable text + how to score it.

Guardrails: auto-tune only ever mutates the candidate text passed to it; it
never writes source files directly. The caller decides whether to persist a
winning candidate.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from . import metrics
from .queries import Query

# --------------------------------------------------------------------------- #
# Stage A: robustness.
# --------------------------------------------------------------------------- #


@dataclass
class RobustnessReport:
    label: str
    mean: float
    ci_lo: float
    ci_hi: float
    per_perturbation: list[float] = field(default_factory=list)


def paraphrase_queries(
    queries: list[Query], style: str = "shuffle_words", *, seed: int | None = None
) -> list[Query]:
    """Cheap deterministic perturbation of query text for robustness checks.

    ``style="shuffle_words"`` lightly reorders tokens; an LLM paraphraser can be
    swapped in by the caller for a stronger test. Ground-truth relevance is
    preserved.

    Pass ``seed`` for a self-contained, reproducible shuffle (a local RNG); when
    omitted it uses the global ``random`` state, so the caller must seed it first
    (e.g. via :func:`airm.run.set_seed`) to get reproducible output.
    """
    rng = random.Random(seed) if seed is not None else random
    out: list[Query] = []
    for q in queries:
        words = q.question.split()
        if style == "shuffle_words" and len(words) > 3:
            mid = words[1:-1]
            rng.shuffle(mid)
            text = " ".join([words[0], *mid, words[-1]])
        else:
            text = q.question
        out.append(
            Query(
                id=q.id,
                question=text,
                relevant=q.relevant,
                notes=q.notes,
                difficulty=q.difficulty,
                fields=q.fields,
                paper=q.paper,
            )
        )
    return out


def robustness_check(
    label: str,
    scorer: Callable[[int], list[float]],
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> RobustnessReport:
    """Run ``scorer(seed)`` (returns per-query scores) across seeds; pool + CI.

    The per-seed score lists are repeated measures of the *same* queries, so
    pooling them as if independent understates the CI. We cluster-bootstrap by
    query position (each query is one cluster observed across all seeds), which
    keeps a query's correlated seed measurements together. This assumes every
    seed returns scores in the same query order — which the runner guarantees.
    """
    pooled: list[float] = []
    clusters: list[int] = []
    per_perturbation: list[float] = []
    for seed in seeds:
        scores = scorer(seed)
        pooled.extend(scores)
        clusters.extend(range(len(scores)))  # query index -> cluster id
        per_perturbation.append(sum(scores) / len(scores) if scores else 0.0)
    mean, lo, hi = metrics.bootstrap_ci_clustered(pooled, clusters)
    return RobustnessReport(label, mean, lo, hi, per_perturbation)


def cis_overlap(a: RobustnessReport, b: RobustnessReport) -> bool:
    """True if two findings' CIs overlap (difference not robust)."""
    return not (a.ci_hi < b.ci_lo or b.ci_hi < a.ci_lo)


# --------------------------------------------------------------------------- #
# Stage B: agentic auto-tune.
# --------------------------------------------------------------------------- #


@dataclass
class Candidate:
    """An editable artifact (renderer template or prompt) under optimization."""

    text: str
    score: float = float("-inf")
    round: int = 0


@dataclass
class TuneResult:
    best: Candidate
    history: list[Candidate]
    plateaued: bool


_CRITIQUE_SYSTEM = (
    "You optimize how scientific metadata is rendered for LLM retrieval. "
    "Given the current rendering template/prompt and its measured score, propose "
    "an improved version. Return ONLY the new template text, no commentary."
)


def auto_tune(
    initial: str,
    score_fn: Callable[[str], float],
    *,
    model: str,
    rounds: int = 5,
    patience: int = 2,
    context: str = "",
) -> TuneResult:
    """Agentic hill-climb on a text artifact.

    Each round: score current text, ask ``model`` (via airm.llm) for a revision,
    keep it only if it scores higher. Stop after ``patience`` non-improving
    rounds or ``rounds`` total. The LLM call is lazily imported so robustness-only
    use needs no provider configured.
    """
    from . import llm

    best = Candidate(text=initial, score=score_fn(initial), round=0)
    history = [best]
    stale = 0

    for r in range(1, rounds + 1):
        if stale >= patience:
            return TuneResult(best=best, history=history, plateaued=True)

        prompt = (
            f"{context}\n\n"
            f"Current template (score={best.score:.4f}):\n{best.text}\n\n"
            "Propose a better template."
        )
        try:
            revised = llm.complete(model, prompt, system=_CRITIQUE_SYSTEM).text.strip()
        except llm.ProviderUnavailable:
            # cannot tune without a model; return what we have
            return TuneResult(best=best, history=history, plateaued=False)

        cand = Candidate(text=revised, score=score_fn(revised), round=r)
        history.append(cand)
        if cand.score > best.score:
            best = cand
            stale = 0
        else:
            stale += 1

    return TuneResult(best=best, history=history, plateaued=stale >= patience)
