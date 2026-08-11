"""Experiment 2 — how does model variation affect retrieval across formats?

The matrix is ``format x model x query``. Each cell:

1. retrieves the top-k records from that format's Chroma collection,
2. asks the model which collections answer the question, given only those,
3. scores the result on exact retrieval metrics and the five DeepEval metrics.

Retrieval depends only on the format, so it is computed **once per (format,
query)** and shared across models -- otherwise every model would re-run an
identical vector search and the retrieval columns would be duplicated N times
with identical values, at N times the cost.

Every cell is checkpointed to JSONL as it completes. A matrix of this size will
be interrupted -- by a rate limit, a laptop lid, an OOM -- and a run that has to
restart from zero is a run that never finishes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import index
from .config import (
    FORMATS,
    RUNS_DIR,
    TOP_K,
    ModelSpec,
    ensure_dirs,
    model_matrix,
    new_run_id,
)
from .provenance import provenance
from .evaluate import judge_description, retrieval_metrics, score_with_deepeval
from .llm import PURPOSE_ANSWER, CallLogger, LLMError, ask, resolve_models
from .queries import Query, load_eval_queries

ANSWER_SYSTEM = """\
You help Earth scientists find NASA data. You are given a question and a numbered \
list of candidate dataset descriptions retrieved from NASA's Common Metadata \
Repository.

Name the candidates that genuinely help answer the question, and say briefly why \
each one does.

Rules:
- Use ONLY the candidates shown. Do not mention datasets that are not listed.
- If none of the candidates are relevant, say so plainly.
- Refer to each dataset by its title.
- Be concise: a sentence per relevant dataset."""


def answer_prompt(question: str, contexts: list[str]) -> str:
    blocks = "\n\n".join(f"[{i}]\n{c}" for i, c in enumerate(contexts, start=1))
    return f"Question: {question}\n\nCandidates:\n\n{blocks}"


def expected_answer(query: Query) -> str:
    """The reference answer DeepEval's correctness metric grades against."""
    titles = [t for t in query.expected_titles if t]
    if titles:
        return "The relevant collections are: " + "; ".join(titles) + "."
    return "The relevant collections are: " + "; ".join(query.expected_concept_ids) + "."


@dataclass
class Cell:
    run_id: str
    query_id: str
    source: str
    topic: str | None
    fmt: str
    provider: str
    model: str
    retrieved: list[str] = field(default_factory=list)
    expected: list[str] = field(default_factory=list)
    answer: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    latency_s: float | None = None
    metrics: dict = field(default_factory=dict)
    judge_scores: dict = field(default_factory=dict)
    judge_errors: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.query_id, self.fmt, f"{self.provider}:{self.model}")


def retrieve_for(
    queries: list[Query],
    *,
    formats: tuple[str, ...] = FORMATS,
    k: int = TOP_K,
    needed: set[tuple[str, str]] | None = None,
) -> dict[tuple[str, str], list[dict]]:
    """``{(query_id, format): hits}`` -- computed once, reused for every model.

    Scoped to the formats actually being run, and -- on a resume -- to the
    ``needed`` pairs that still have at least one uncheckpointed cell. Without
    that scope, resuming a fully-completed matrix would re-run every vector
    search before discovering there is nothing left to do.
    """
    out: dict[tuple[str, str], list[dict]] = {}
    for query in queries:
        for fmt in formats:
            pair = (query.query_id, fmt)
            if needed is not None and pair not in needed:
                continue
            out[pair] = index.query(fmt, query.text, k=k)
    return out


def load_checkpoint(path: Path) -> dict[tuple[str, str, str], dict]:
    if not path.exists():
        return {}
    done: dict[tuple[str, str, str], dict] = {}
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            done[(row["query_id"], row["fmt"], f"{row['provider']}:{row['model']}")] = row
    return done


def run(
    *,
    formats: tuple[str, ...] = FORMATS,
    models: list[ModelSpec] | None = None,
    queries: list[Query] | None = None,
    limit: int | None = None,
    k: int = TOP_K,
    judge: bool = True,
    run_id: str | None = None,
    out_dir: Path | None = None,
    resume: bool = True,
) -> dict:
    """Run the matrix, checkpointing every completed cell."""
    ensure_dirs()
    run_id = run_id or new_run_id()
    out_dir = out_dir or (RUNS_DIR / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = queries if queries is not None else load_eval_queries()
    if limit:
        queries = queries[:limit]
    models = models if models is not None else model_matrix()

    usable, problems = resolve_models(models)
    logger = CallLogger(run_id=run_id)

    judge_obj = None
    metrics_obj = None
    if judge:
        from .config import judge_spec
        from .evaluate import _deepeval_judge, build_metrics

        # Fail fast if the judge's provider is down or the model retired.
        # resolve_models() protects the answer models, but the judge bypasses
        # the matrix -- and a broken judge key would otherwise burn hours of
        # retried 401s across every metric of every cell, then "complete" with
        # every judge score None.
        _, judge_problems = resolve_models([judge_spec()])
        if judge_problems:
            raise LLMError(
                "DeepEval judge is unavailable: "
                + "; ".join(judge_problems)
                + ". Fix the credentials/model, or pass --no-judge for "
                "retrieval metrics only."
            )
        judge_obj = _deepeval_judge(None, logger)
        metrics_obj = build_metrics(judge_obj)

    checkpoint = out_dir / "exp2_cells.jsonl"
    done = load_checkpoint(checkpoint) if resume else {}

    needed = {
        (query.query_id, fmt)
        for query in queries
        for fmt in formats
        if any((query.query_id, fmt, spec.key) not in done for spec in usable)
    }
    hits = retrieve_for(queries, formats=formats, k=k, needed=needed)

    cells: list[Cell] = [Cell(**_from_row(row)) for row in done.values()]

    with checkpoint.open("a") as sink:
        for spec in usable:
            for fmt in formats:
                for query in queries:
                    key = (query.query_id, fmt, spec.key)
                    if key in done:
                        continue

                    candidates = hits[(query.query_id, fmt)]
                    contexts = [h["document"] for h in candidates]
                    retrieved = [h["concept_id"] for h in candidates]

                    cell = Cell(
                        run_id=run_id,
                        query_id=query.query_id,
                        source=query.source,
                        topic=query.topic,
                        fmt=fmt,
                        provider=spec.provider,
                        model=spec.model,
                        retrieved=retrieved,
                        expected=query.expected_concept_ids,
                        metrics=retrieval_metrics(retrieved, query.expected_concept_ids),
                    )

                    try:
                        response = ask(
                            spec,
                            answer_prompt(query.text, contexts),
                            system=ANSWER_SYSTEM,
                            purpose=PURPOSE_ANSWER,
                            logger=logger,
                            temperature=0.0,
                            fmt=fmt,
                            query_id=query.query_id,
                        )
                        cell.answer = response.text
                        cell.prompt_tokens = response.prompt_tokens
                        cell.completion_tokens = response.completion_tokens
                        cell.cost_usd = response.cost_usd
                        cell.latency_s = response.latency_s
                    except LLMError as exc:
                        cell.error = str(exc)

                    if judge and cell.answer:
                        result = score_with_deepeval(
                            question=query.text,
                            answer=cell.answer,
                            expected=expected_answer(query),
                            contexts=contexts,
                            judge=judge_obj,
                            metrics=metrics_obj,
                            logger=logger,
                            fmt=fmt,
                            query_id=query.query_id,
                            model=spec.model,
                        )
                        cell.judge_scores = result.scores
                        cell.judge_errors = result.errors

                    cells.append(cell)
                    sink.write(json.dumps(asdict(cell), ensure_ascii=False) + "\n")
                    sink.flush()

    summary = summarise(cells)
    result = {
        **provenance(run_id),
        "formats": list(formats),
        "models": [s.key for s in usable],
        "model_problems": problems,
        "queries": len(queries),
        "cells": len(cells),
        "top_k": k,
        "judge": judge_description() if judge else None,
        "log_counts": logger.counts(),
        "summary": summary,
    }
    (out_dir / "exp2_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _from_row(row: dict) -> dict:
    return {k: v for k, v in row.items() if k in Cell.__dataclass_fields__}


def summarise(cells: list[Cell]) -> dict:
    """Mean metrics per ``(format, model)`` and per format overall."""
    import statistics

    def mean(values):
        values = [v for v in values if v is not None and v == v]
        return round(statistics.fmean(values), 4) if values else None

    per_cell: dict[str, dict] = {}
    for cell in cells:
        bucket = per_cell.setdefault(
            f"{cell.fmt}|{cell.provider}:{cell.model}", {"n": 0, "n_answered": 0, "vals": {}}
        )
        bucket["n"] += 1
        # Retrieval metrics cover every cell; answer/judge columns cover only
        # the cells the model actually answered. Reporting both denominators
        # keeps a partly-failed run from reading as a fully-scored one.
        if cell.answer and not cell.error:
            bucket["n_answered"] += 1
        for name, value in {**cell.metrics, **cell.judge_scores}.items():
            bucket["vals"].setdefault(name, []).append(value)
        for name, value in (
            ("prompt_tokens", cell.prompt_tokens),
            ("completion_tokens", cell.completion_tokens),
            ("latency_s", cell.latency_s),
            ("cost_usd", cell.cost_usd),
        ):
            bucket["vals"].setdefault(name, []).append(value)

    return {
        key: {
            "n": b["n"],
            "n_answered": b["n_answered"],
            **{name: mean(vals) for name, vals in b["vals"].items()},
        }
        for key, b in sorted(per_cell.items())
    }


if __name__ == "__main__":  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 2: model x format retrieval.")
    parser.add_argument("--smoke", action="store_true", help="2 formats, 1 model, 5 queries")
    parser.add_argument("--no-judge", action="store_true", help="retrieval metrics only")
    parser.add_argument("--limit", type=int, help="cap the number of queries")
    parser.add_argument("--provider", help="restrict to one provider")
    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        help="continue an interrupted run under runs/RUN_ID, skipping "
        "checkpointed cells (without this, every invocation starts a fresh run)",
    )
    args = parser.parse_args()

    specs = model_matrix()
    if args.provider:
        specs = [s for s in specs if s.provider == args.provider]

    kwargs: dict = {"judge": not args.no_judge, "models": specs}
    if args.smoke:
        kwargs.update(formats=("json", "mat"), models=specs[:1], limit=5)
    elif args.limit:
        kwargs["limit"] = args.limit
    if args.resume:
        if not (RUNS_DIR / args.resume / "exp2_cells.jsonl").exists():
            parser.error(
                f"no checkpoint at runs/{args.resume}/exp2_cells.jsonl; "
                f"existing runs: {sorted(d.name for d in RUNS_DIR.glob('*') if d.is_dir())}"
            )
        kwargs["run_id"] = args.resume

    res = run(**kwargs)

    print(f"run           : {res['run_id']}")
    print(f"models        : {', '.join(res['models']) or '(none available)'}")
    for problem in res["model_problems"]:
        print(f"  ! {problem}")
    print(f"queries       : {res['queries']}   cells: {res['cells']}")
    print(f"judge         : {res['judge']}")
    print(f"log records   : {res['log_counts']}")
    print("\nmean scores per format x model:")
    for key, row in res["summary"].items():
        keep = {k: v for k, v in row.items() if k in ("n", "n_answered", "recall@10", "mrr", "ndcg@10", "correctness", "faithfulness")}
        print(f"  {key:<34} {keep}")
    print(f"\nwrote {RUNS_DIR / res['run_id']}/exp2_cells.jsonl")
