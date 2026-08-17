#!/usr/bin/env python
"""Stage J -- DeepEval judging of a Stage A answers run.

Reads the frozen ``runs/<rid>/answers.jsonl`` a Stage A run produced, re-assembles
each cell's contexts from the logged concept-ids (fingerprint-checked against what
Stage R pinned, exactly as Stage A did), and scores every answered cell with the
five DeepEval metrics under one fixed judge:

    correctness, faithfulness, answer_relevancy, contextual_relevancy, contextual_recall

Judge scores append to ``runs/<rid>/judgements.jsonl``; every judge call is logged
to ``runs/<rid>/llm/judge.jsonl``. Because this stage only reads frozen answers it
can run incrementally (one model, one payload, a --limit slice) and can be re-run
under a different judge without touching the answers -- rows carry the judge's
identity and the checkpoint key includes it, so a nano pass and a mini calibration
pass coexist in the same file and are compared with ``--report``.

A judge failure is recorded as an error with ``None`` scores, never as a zero:
a broken judge and a bad answer are different events.

**Self-judging caveat.** If the judge is also one of the models under test in the
answers run, it grades its own output on half the matrix. The run summary records
this as ``self_judge_models`` rather than refusing -- but treat those cells'
scores as compromised for model-vs-model comparison.

## Usage

    # Judge everything answered so far with gpt-5.4-nano, 4 concurrent workers:
    uv run python scripts/judge_stage.py --answers 20260814T210927Z

    # Calibration slice: same 60 cells under two judges, then compare:
    uv run python scripts/judge_stage.py --answers RID --judge openai:gpt-5.4-nano --limit 60
    uv run python scripts/judge_stage.py --answers RID --judge openai:gpt-5-mini  --limit 60
    uv run python scripts/judge_stage.py --answers RID --report

    --judge      provider:model (default openai:gpt-5.4-nano)
    --models     judge only these answer models' cells
    --payloads   faceted / unfaceted
    --formats    subset of the six formats
    --metrics    subset of the five metric names (e.g. drop contextual_relevancy,
                 which alone costs ~10 of the ~21 judge calls per cell)
    --workers    concurrent cells (default 4; each worker has its own judge+metrics)
    --limit      cap the number of cells this invocation judges
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airm import format_cache, unfaceted  # noqa: E402
from airm.config import FORMATS, RUNS_DIR, ModelSpec  # noqa: E402
from airm.evaluate import METRIC_NAMES, _deepeval_judge, build_metrics, score_with_deepeval  # noqa: E402
from airm.exp2 import expected_answer  # noqa: E402
from airm.llm import CallLogger, LLMError, resolve_models  # noqa: E402
from airm.provenance import provenance  # noqa: E402
from airm.queries import load_queries  # noqa: E402

QUERIES_PATH = Path("data/queries_full.jsonl")

#: Ollama judges need an explicit context window (see llm.complete); sized per
#: cell from the contexts, with room for DeepEval's own prompt scaffolding.
OLLAMA_CTX_MARGIN = 4096
OLLAMA_CTX_MAX = 131072


class RunLogger(CallLogger):
    """Judge calls beside the answers they grade: ``runs/<rid>/llm/judge.jsonl``."""

    @property
    def dir(self) -> Path:
        return RUNS_DIR / self.run_id / "llm"


# --------------------------------------------------------------------------- #
# Inputs.
# --------------------------------------------------------------------------- #


def parse_model(spec: str) -> ModelSpec:
    provider, _, model = spec.partition(":")
    if provider not in ("openai", "ollama") or not model:
        raise SystemExit(f"FAILED: wanted provider:model, got {spec!r}")
    return ModelSpec(provider, model)


def load_answers(run_dir: Path) -> tuple[list[dict], dict]:
    answers_path = run_dir / "answers.jsonl"
    selection_path = run_dir / "selection.json"
    for p in (answers_path, selection_path):
        if not p.exists():
            raise SystemExit(f"FAILED: {p} does not exist -- run Stage A first.")
    cells = [json.loads(l) for l in answers_path.open() if l.strip()]
    return cells, json.loads(selection_path.read_text())


def verify_render_caches(selection: dict, payloads: set[str]) -> None:
    """The bytes this stage feeds the judge must be the ones Stage A/R used."""
    pinned = {
        info["payload"]: (info.get("render_cache") or {}).get("fingerprint")
        for info in selection.get("retrieval_runs", {}).values()
    }
    for payload in payloads:
        module = format_cache if payload == "faceted" else unfaceted
        current = module._manifest_fingerprint()
        if pinned.get(payload) and current != pinned[payload]:
            raise SystemExit(
                f"FAILED: {payload} render cache fingerprint {current} does not "
                f"match the one the answers run pinned ({pinned[payload]}). "
                "The cache was rebuilt since; the contexts cannot be reproduced."
            )


def load_context(payload: str, fmt: str, cid: str) -> str:
    if payload == "faceted":
        return format_cache.load(fmt, cid)
    return unfaceted.load(fmt, cid)


def cell_key(row: dict) -> tuple:
    return (row["payload"], row["fmt"], row["query_id"],
            f"{row['provider']}:{row['model']}", row["judge"])


def load_done(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    done = set()
    with path.open() as fh:
        for line in fh:
            if line.strip():
                done.add(cell_key(json.loads(line)))
    return done


# --------------------------------------------------------------------------- #
# Judging.
# --------------------------------------------------------------------------- #


def judge_cells(args) -> int:
    run_dir = RUNS_DIR / args.answers
    cells, selection = load_answers(run_dir)

    metric_names = tuple(args.metrics or METRIC_NAMES)
    unknown = set(metric_names) - set(METRIC_NAMES)
    if unknown:
        raise SystemExit(f"FAILED: unknown metrics {sorted(unknown)}; "
                         f"choose from {list(METRIC_NAMES)}")

    judge_spec_ = parse_model(args.judge)
    usable, problems = resolve_models([judge_spec_])
    if not usable:
        raise SystemExit("FAILED: judge unavailable: " + "; ".join(problems))

    # Which answered cells still need this judge's verdict.
    todo = []
    for cell in cells:
        if not cell.get("answer") or cell.get("error"):
            continue
        if args.models and f"{cell['provider']}:{cell['model']}" not in args.models:
            continue
        if args.payloads and cell["payload"] not in args.payloads:
            continue
        if args.formats and cell["fmt"] not in args.formats:
            continue
        todo.append(cell)
    todo.sort(key=lambda c: (c["provider"] + ":" + c["model"], c["payload"],
                             c["fmt"], c["query_id"]))

    checkpoint = run_dir / "judgements.jsonl"
    done = load_done(checkpoint)
    todo = [c for c in todo
            if (c["payload"], c["fmt"], c["query_id"],
                f"{c['provider']}:{c['model']}", judge_spec_.key) not in done]
    if args.limit:
        todo = todo[: args.limit]

    payloads = {c["payload"] for c in todo}
    verify_render_caches(selection, payloads)

    queries = {q.query_id: q for q in load_queries(QUERIES_PATH)}
    missing = {c["query_id"] for c in todo} - set(queries)
    if missing:
        raise SystemExit(f"FAILED: {len(missing)} query ids not in {QUERIES_PATH}: "
                         f"{sorted(missing)[:5]} ...")

    answer_models = set(selection.get("models", []))
    self_judge = sorted(answer_models & {judge_spec_.key})
    print(f"answers run : {args.answers}")
    print(f"judge       : {judge_spec_.key}   metrics: {list(metric_names)}")
    if self_judge:
        print(f"  ! CAVEAT: judge is also under test in this run ({self_judge}); "
              "its own cells' scores are compromised for model-vs-model comparison.")
    print(f"cells       : {len(todo)} to judge ({len(done)} already checkpointed)")
    print(f"workers     : {args.workers}\n")
    if not todo:
        return write_summary(run_dir, metric_names)

    logger = RunLogger(run_id=args.answers)
    sink_lock = threading.Lock()
    local = threading.local()
    started = time.time()
    n_done = 0

    def worker_tools():
        # Metric objects and the judge's set_context are stateful; concurrent
        # cells sharing them would interleave scores and trace metadata. One
        # judge + metrics set per worker thread, one shared thread-safe logger.
        if not hasattr(local, "judge"):
            local.judge = _deepeval_judge(judge_spec_, logger)
            local.metrics = {
                name: metric
                for name, metric in build_metrics(local.judge).items()
                if name in metric_names
            }
        return local.judge, local.metrics

    def judge_one(cell: dict) -> dict:
        judge, metrics = worker_tools()
        query = queries[cell["query_id"]]
        contexts = [load_context(cell["payload"], cell["fmt"], cid)
                    for cid in cell["retrieved"]]

        # NB: key must not be "model" -- that collides with CallLogger.log()'s
        # own ``model`` argument (the judge's) and raises after the paid call.
        meta: dict = {"fmt": cell["fmt"], "query_id": cell["query_id"],
                      "payload": cell["payload"], "answer_model": cell["model"]}
        if judge_spec_.provider == "ollama":
            need = sum(len(c) for c in contexts) // 3 + OLLAMA_CTX_MARGIN
            meta["ollama_options"] = {"num_ctx": min(OLLAMA_CTX_MAX, max(8192, need))}

        t0 = time.time()
        result = score_with_deepeval(
            question=query.text,
            answer=cell["answer"],
            expected=expected_answer(query),
            contexts=contexts,
            judge=judge,
            metrics=metrics,
            logger=logger,
            **meta,
        )
        return {
            "run_id": cell["run_id"],
            "payload": cell["payload"],
            "fmt": cell["fmt"],
            "query_id": cell["query_id"],
            "source": cell["source"],
            "provider": cell["provider"],
            "model": cell["model"],
            "judge": judge_spec_.key,
            "metrics": list(metric_names),
            "scores": result.scores,
            "reasons": result.reasons,
            "errors": result.errors,
            "latency_s": round(time.time() - t0, 2),
        }

    with checkpoint.open("a") as sink, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(judge_one, cell): cell for cell in todo}
        for future in as_completed(futures):
            cell = futures[future]
            try:
                row = future.result()
            except (LLMError, format_cache.StaleCacheError) as exc:
                row = {
                    "run_id": cell["run_id"], "payload": cell["payload"],
                    "fmt": cell["fmt"], "query_id": cell["query_id"],
                    "source": cell["source"], "provider": cell["provider"],
                    "model": cell["model"], "judge": judge_spec_.key,
                    "metrics": list(metric_names), "scores": {}, "reasons": {},
                    "errors": {"cell": f"{type(exc).__name__}: {exc}"},
                    "latency_s": None,
                }
            with sink_lock:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
            n_done += 1
            if n_done % 10 == 0 or n_done == len(todo):
                print(f"  {n_done}/{len(todo)} cells  ({time.time() - started:.0f}s)")

    return write_summary(run_dir, metric_names)


# --------------------------------------------------------------------------- #
# Summary / report.
# --------------------------------------------------------------------------- #


def _mean(values):
    real = [v for v in values if v is not None]
    return round(statistics.fmean(real), 4) if real else None


def write_summary(run_dir: Path, metric_names=METRIC_NAMES) -> int:
    checkpoint = run_dir / "judgements.jsonl"
    rows = [json.loads(l) for l in checkpoint.open() if l.strip()] if checkpoint.exists() else []
    if not rows:
        print("no judgements yet.")
        return 0

    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        grouped[f"{row['judge']}|{row['payload']}|{row['fmt']}|"
                f"{row['provider']}:{row['model']}"].append(row)

    summary = {}
    for key, group in sorted(grouped.items()):
        entry = {"n": len(group),
                 "n_errored": sum(1 for r in group if r["errors"])}
        for name in METRIC_NAMES:
            entry[name] = _mean([r["scores"].get(name) for r in group])
        summary[key] = entry

    out = {
        **provenance(run_dir.name),
        "judges": sorted({r["judge"] for r in rows}),
        "cells_judged": len(rows),
        "summary": summary,
    }
    (run_dir / "judge_summary.json").write_text(json.dumps(out, indent=2) + "\n")

    print("\nmean judge scores per judge x payload x format x model:")
    header = (f"{'judge':<22}{'payload':<10}{'fmt':<8}{'model':<26}{'n':>4}{'err':>4}"
              + "".join(f"{n[:12]:>14}" for n in METRIC_NAMES))
    print(header)
    print("-" * len(header))
    for key, entry in summary.items():
        judge, payload, fmt, model = key.split("|")
        values = "".join(
            f"{entry[n]:>14.3f}" if entry[n] is not None else f"{'—':>14}"
            for n in METRIC_NAMES
        )
        print(f"{judge:<22}{payload:<10}{fmt:<8}{model:<26}{entry['n']:>4}"
              f"{entry['n_errored']:>4}{values}")
    print(f"\nwrote {run_dir / 'judge_summary.json'}")
    return 0


def report(args) -> int:
    """Compare judges cell-by-cell -- the calibration readout."""
    run_dir = RUNS_DIR / args.answers
    checkpoint = run_dir / "judgements.jsonl"
    rows = [json.loads(l) for l in checkpoint.open() if l.strip()] if checkpoint.exists() else []
    judges = sorted({r["judge"] for r in rows})
    if len(judges) < 2:
        print(f"{len(judges)} judge(s) in {checkpoint}; calibration needs two.")
        return write_summary(run_dir)

    by_cell: dict[tuple, dict[str, dict]] = collections.defaultdict(dict)
    for r in rows:
        by_cell[(r["payload"], r["fmt"], r["query_id"],
                 f"{r['provider']}:{r['model']}")][r["judge"]] = r

    shared = {k: v for k, v in by_cell.items() if len(v) == len(judges)}
    print(f"judges: {judges}   cells scored by all: {len(shared)}\n")
    a, b = judges[0], judges[1]
    print(f"{'metric':<22}{'n':>5}{'mean ' + a[:14]:>20}{'mean ' + b[:14]:>20}{'corr':>8}")
    print("-" * 75)
    for name in METRIC_NAMES:
        pairs = [(v[a]["scores"].get(name), v[b]["scores"].get(name))
                 for v in shared.values()]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if not pairs:
            print(f"{name:<22}{0:>5}{'—':>20}{'—':>20}{'—':>8}")
            continue
        xs, ys = zip(*pairs)
        corr = None
        if len(pairs) > 2 and statistics.pstdev(xs) > 0 and statistics.pstdev(ys) > 0:
            corr = statistics.correlation(xs, ys)
        print(f"{name:<22}{len(pairs):>5}{statistics.fmean(xs):>20.3f}"
              f"{statistics.fmean(ys):>20.3f}"
              f"{corr:>8.3f}" if corr is not None else
              f"{name:<22}{len(pairs):>5}{statistics.fmean(xs):>20.3f}"
              f"{statistics.fmean(ys):>20.3f}{'—':>8}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--answers", required=True,
                        help="Stage A run id under runs/ whose answers.jsonl to judge")
    parser.add_argument("--judge", default="openai:gpt-5.4-nano",
                        help="provider:model (default openai:gpt-5.4-nano)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="only judge these answer models' cells")
    parser.add_argument("--payloads", nargs="+", choices=["faceted", "unfaceted"],
                        default=None)
    parser.add_argument("--formats", nargs="+", choices=list(FORMATS), default=None)
    parser.add_argument("--metrics", nargs="+", default=None,
                        help=f"subset of {list(METRIC_NAMES)}")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap cells judged this invocation (calibration slices)")
    parser.add_argument("--report", action="store_true",
                        help="no judging: summarise, and compare judges if two exist")
    args = parser.parse_args(argv)
    if args.report:
        return report(args)
    return judge_cells(args)


if __name__ == "__main__":
    raise SystemExit(main())
