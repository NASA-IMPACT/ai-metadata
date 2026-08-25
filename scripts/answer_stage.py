#!/usr/bin/env python
"""Stage A -- LLM answers replayed from a Stage R retrieval log.

The pipeline is deliberately staged so each result is a frozen, auditable
artifact the next stage *reads* rather than recomputes:

    Stage R  scripts/unfaceted_chunked_eval.py eval
             -> runs/<rid>/chunked_retrieval.jsonl   (query, ranked ids, metrics)
    Stage A  this script
             -> runs/<rid>/answers.jsonl             (one cell per query x format x payload x model)
             -> runs/<rid>/llm/answer.jsonl          (every raw call: full prompt, full response)
    Stage J  judge stage, later, reading answers.jsonl

Nothing here retrieves. The contexts fed to each model are the concept-ids the
retrieval log recorded, rendered from the format caches -- and both cache
fingerprints are checked against the ones the retrieval run pinned, so "the
model saw what retrieval ranked" is a verified claim, not an assumption.

What this stage measures on its own, before any judge runs: the real prompt
token cost of each (payload x format) in use, per model -- the operational
price of a representation, measured rather than estimated.

## Usage

    # 50-query slice (all 11 SME + stratified synthetic), both payloads,
    # gpt-5.4-nano + muse-glimmer:
    uv run python scripts/answer_stage.py \
        --retrieval 20260814T205849Z 20260814T205912Z --n 50

    # Resume an interrupted run:
    uv run python scripts/answer_stage.py \
        --retrieval 20260814T205849Z 20260814T205912Z --n 50 --run-id <RUN_ID>

    --models     provider:model, repeatable (default gpt-5.4-nano + muse-glimmer)
    --k          contexts per prompt (default 10, the retrieval log's depth)
    --workers    concurrent OpenAI cells (default 1)
    --ollama-workers
                 concurrent Ollama cells (default 1; raise only against a
                 remote/multi-slot Ollama, see the dispatch comment below)
    --smoke      2 formats x first model x 5 queries, faceted retrieval only

    # Stage S only: a 300-query slice that nests the reference 50-query run
    uv run python scripts/answer_stage.py \
        --retrieval 20260814T205849Z 20260814T205912Z \
        --n 300 --superset-of 20260814T210927Z --select-only
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airm import format_cache, unfaceted  # noqa: E402
from airm.config import FORMATS, RUNS_DIR, ModelSpec, new_run_id  # noqa: E402
from airm.exp2 import ANSWER_SYSTEM, answer_prompt  # noqa: E402
from airm.llm import PURPOSE_ANSWER, CallLogger, LLMError, ask, resolve_models  # noqa: E402
from airm.provenance import provenance  # noqa: E402

#: The seed ``corpus.build``, ``synth`` and ``make_query_subset`` use.
RANDOM_SEED = 20260731

#: The two models this study's answer stage runs by default. Overridable with
#: ``--models``, and validated against the live provider listings either way.
DEFAULT_MODELS = ("openai:gpt-5.4-nano", "ollama:muse-glimmer:30b-mlx")

#: Ollama silently truncates prompts that overflow its default ~4k window.
#: The window is sized per call from the prompt instead: chars/3 is a safe
#: over-estimate of tokens for this content, plus room for the answer.
OLLAMA_CTX_MARGIN = 2048
OLLAMA_CTX_MAX = 131072


class RunLogger(CallLogger):
    """CallLogger whose files live under ``runs/<run_id>/llm/``.

    The default logger writes to ``logs/``; this stage keeps the raw calls
    beside the cells they produced so one directory is the whole audit trail.
    """

    @property
    def dir(self) -> Path:
        return RUNS_DIR / self.run_id / "llm"


# --------------------------------------------------------------------------- #
# Stage R artefacts.
# --------------------------------------------------------------------------- #


@dataclass
class RetrievalRun:
    run_id: str
    payload: str
    summary: dict
    rows: dict[tuple[str, str], dict] = field(default_factory=dict)  # (fmt, query_id) -> row


def load_retrieval_run(run_id: str) -> RetrievalRun:
    directory = RUNS_DIR / run_id
    summary_path = directory / "chunked_summary.json"
    detail_path = directory / "chunked_retrieval.jsonl"
    for p in (summary_path, detail_path):
        if not p.exists():
            raise SystemExit(f"FAILED: {p} does not exist -- run Stage R first.")
    summary = json.loads(summary_path.read_text())
    run = RetrievalRun(run_id=run_id, payload=summary["payload"], summary=summary)
    with detail_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            run.rows[(row["format"], row["query_id"])] = row
    return run


def verify_render_cache(run: RetrievalRun) -> None:
    """The renderings on disk must be the ones the retrieval run indexed.

    Retrieval pinned the cache manifest fingerprint; if the cache has been
    rebuilt from different code since, the answer stage would feed the models
    different bytes than the ones retrieval ranked -- silently. Fail instead.
    """
    pinned = run.summary.get("render_cache", {})
    module = format_cache if run.payload == "faceted" else unfaceted
    current = module._manifest_fingerprint()
    if not pinned.get("fingerprint"):
        raise SystemExit(
            f"FAILED: retrieval run {run.run_id} pinned no render-cache "
            "fingerprint; re-run Stage R with the current script."
        )
    if current != pinned["fingerprint"]:
        raise SystemExit(
            f"FAILED: {run.payload} render cache fingerprint {current} does not "
            f"match the one retrieval run {run.run_id} pinned "
            f"({pinned['fingerprint']}). The cache was rebuilt since retrieval; "
            "re-run Stage R against the current cache."
        )


def load_context(payload: str, fmt: str, cid: str) -> str:
    if payload == "faceted":
        return format_cache.load(fmt, cid)
    return unfaceted.load(fmt, cid)


# --------------------------------------------------------------------------- #
# Query selection (Stage S).
# --------------------------------------------------------------------------- #


def select_queries(runs: list[RetrievalRun], n: int, seed: int,
                   superset_of: list[str] | None = None) -> tuple[list[str], dict]:
    """All SME queries plus a stratified synthetic draw, ``n`` total.

    Mirrors ``make_query_subset``: synthetic quotas proportional to the full
    set's topic mix under largest-remainder rounding with an alphabetical
    tie-break, each topic's draw seeded so the selection rebuilds identically.

    The SME queries are all kept because they are the expert half of the ground
    truth; at n=50 that makes them 22% of the slice against 2.2% of the full
    set, so pooled numbers overweight the hard expert queries -- compare
    per-slice, not pooled.

    ``superset_of`` (optional): query ids of an earlier, smaller selection that
    the new one must contain. Those ids are taken first within each topic and
    the remainder of the topic's quota is drawn with the seeded sampler, so a
    larger slice nests the smaller one (its answered cells can be reused) while
    the topic quotas stay proportional to the full set.
    """
    # Any format's rows carry the full query list; formats share the query set.
    fmt = FORMATS[0]
    base = {qid: row for (f, qid), row in runs[0].rows.items() if f == fmt}
    for run in runs[1:]:
        ids = {qid for (f, qid) in run.rows if f == fmt}
        if ids != set(base):
            raise SystemExit(
                f"FAILED: retrieval runs disagree on the query set "
                f"({runs[0].run_id} vs {run.run_id}); they must come from the "
                "same Stage R query file."
            )

    sme = sorted(qid for qid, row in base.items() if row["source"] == "sme")
    synthetic = {qid: row for qid, row in base.items() if row["source"] == "synthetic"}
    want = n - len(sme)
    if want <= 0:
        raise SystemExit(f"FAILED: --n {n} leaves no room for synthetic queries "
                         f"beside the {len(sme)} SME ones.")

    by_topic: dict[str, list[str]] = collections.defaultdict(list)
    for qid, row in synthetic.items():
        by_topic[row["topic"] or "UNCLASSIFIED"].append(qid)

    total = len(synthetic)
    exact = {t: want * len(ids) / total for t, ids in by_topic.items()}
    quota = {t: int(v) for t, v in exact.items()}
    short = want - sum(quota.values())
    for t in sorted(exact, key=lambda t: (-(exact[t] - quota[t]), t))[:short]:
        quota[t] += 1

    keep = set(superset_of or [])
    unknown = keep - set(base)
    if unknown:
        raise SystemExit(f"FAILED: --superset-of ids not in the retrieval runs: {sorted(unknown)[:5]}")
    rng = random.Random(seed)
    chosen: list[str] = []
    for topic in sorted(by_topic):
        ids = sorted(by_topic[topic])
        kept = [q for q in ids if q in keep]
        if len(kept) > quota[topic]:
            raise SystemExit(f"FAILED: --superset-of holds {len(kept)} {topic} queries "
                             f"but the n={n} quota is {quota[topic]}; raise --n.")
        rest = [q for q in ids if q not in keep]
        chosen.extend(kept + rng.sample(rest, min(quota[topic] - len(kept), len(rest))))

    report = {
        "n": n,
        "seed": seed,
        "sme": len(sme),
        "synthetic": len(chosen),
        "quota": {t: quota[t] for t in sorted(quota) if quota[t]},
        "full_set": {"queries": len(base), "sme": len(sme), "synthetic": total},
        "superset_of": sorted(keep) if keep else None,
        "sme_share_note": (
            f"SME queries are {len(sme)}/{n} = {len(sme) / n:.0%} of this slice vs "
            f"{len(sme)}/{len(base)} = {len(sme) / len(base):.1%} of the full set; "
            "compare per-slice, not pooled."
        ),
    }
    return sme + sorted(chosen), report


# --------------------------------------------------------------------------- #
# Cells.
# --------------------------------------------------------------------------- #


def parse_model(spec: str) -> ModelSpec:
    provider, _, model = spec.partition(":")
    if provider not in ("openai", "ollama") or not model:
        raise SystemExit(f"FAILED: --models wants provider:model, got {spec!r}")
    return ModelSpec(provider, model)


def cell_key(row: dict) -> tuple[str, str, str, str]:
    return (row["payload"], row["fmt"], row["query_id"], f"{row['provider']}:{row['model']}")


def load_done(path: Path) -> dict[tuple, dict]:
    if not path.exists():
        return {}
    done = {}
    with path.open() as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                done[cell_key(row)] = row
    return done



def seed_cells_from(src_run: str, out: Path, retrieval_runs: list[RetrievalRun],
                    selected: list[str], model_keys: list[str], formats) -> dict[str, int]:
    """Copy frozen cells of an earlier run into this one.

    Rows of ``runs/<src>/answers.jsonl`` (and ``judgements.jsonl`` if present)
    whose cell falls inside this run's grid are appended to this run's files,
    stamped with ``seeded_from``. Refuses unless the source run is bound to the
    same Stage R runs and render-cache fingerprints, so the copied answers were
    produced from exactly the contexts this run would replay.
    """
    src = RUNS_DIR / src_run
    src_sel = json.loads((src / "selection.json").read_text())
    mine = {r.run_id: r.summary.get("render_cache", {}).get("fingerprint") for r in retrieval_runs}
    theirs = {rid: v.get("render_cache", {}).get("fingerprint")
              for rid, v in src_sel["retrieval_runs"].items()}
    if mine != theirs:
        raise SystemExit(f"FAILED: --seed-answers-from {src_run} is bound to retrieval runs "
                         f"{theirs}, this run to {mine}; cells are not replay-equivalent.")
    want_q, want_m, want_f = set(selected), set(model_keys), set(formats)
    want_p = {r.payload for r in retrieval_runs}

    def in_grid(row):
        return (row["query_id"] in want_q and row["payload"] in want_p
                and row["fmt"] in want_f and f"{row['provider']}:{row['model']}" in want_m)

    counts = {}
    for name, keyf in (("answers.jsonl", cell_key),
                       ("judgements.jsonl", lambda r: cell_key(r) + (r["judge"],))):
        src_path, dst_path = src / name, out / name
        if not src_path.exists():
            continue
        have = set()
        if dst_path.exists():
            with dst_path.open() as fh:
                have = {keyf(json.loads(l)) for l in fh if l.strip()}
        n = 0
        with src_path.open() as fh, dst_path.open("a") as sink:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not in_grid(row) or keyf(row) in have:
                    continue
                if name == "answers.jsonl" and row.get("error"):
                    continue
                row = {**row, "seeded_from": row.get("seeded_from") or src_run}
                sink.write(json.dumps(row) + "\n")
                have.add(keyf(row))
                n += 1
        counts[name] = n
    return counts

def run_stage(args) -> int:
    run_id = args.run_id or new_run_id()
    out = RUNS_DIR / run_id
    out.mkdir(parents=True, exist_ok=True)

    retrieval_runs = [load_retrieval_run(rid) for rid in args.retrieval]
    payloads = [r.payload for r in retrieval_runs]
    if len(set(payloads)) != len(payloads):
        raise SystemExit(f"FAILED: duplicate payloads in --retrieval: {payloads}")
    for run in retrieval_runs:
        verify_render_cache(run)

    superset_of = None
    if args.superset_of:
        base_sel = json.loads((RUNS_DIR / args.superset_of / "selection.json").read_text())
        superset_of = base_sel["query_ids"]
    selected, selection_report = select_queries(retrieval_runs, args.n, args.seed, superset_of)
    if args.superset_of:
        selection_report["superset_of_run"] = args.superset_of
    if args.limit:
        selected = selected[: args.limit]

    specs = [parse_model(m) for m in args.models]
    if args.select_only:
        usable, problems = specs, []
    else:
        usable, problems = resolve_models(specs)
    for problem in problems:
        print(f"  ! {problem}")
    if not usable:
        raise SystemExit("FAILED: no requested model is available.")
    if args.smoke:
        usable = usable[:1]

    formats = ("json", "mat") if args.smoke else FORMATS
    if args.smoke:
        retrieval_runs = [r for r in retrieval_runs if r.payload == "faceted"] or retrieval_runs
        selected = selected[:5]

    (out / "selection.json").write_text(json.dumps({
        **provenance(run_id),
        "retrieval_runs": {
            r.run_id: {"payload": r.payload, "render_cache": r.summary.get("render_cache")}
            for r in retrieval_runs
        },
        "models": [s.key for s in usable],
        "model_problems": problems,
        "k": args.k,
        "formats": list(formats),
        "query_ids": selected,
        **selection_report,
    }, indent=2) + "\n")

    if args.seed_answers_from:
        seeded = seed_cells_from(args.seed_answers_from, out, retrieval_runs, selected,
                                 [s.key for s in usable], formats)
        print(f"seeded   : " + ", ".join(f"{n} {f}" for f, n in seeded.items())
              + f"  from {args.seed_answers_from}")

    if args.select_only:
        n_sme = sum(1 for qid in selected if qid.startswith("sme-"))
        print(f"run      : {run_id}  (Stage S only; no answers produced)")
        print(f"queries  : {len(selected)} ({n_sme} SME + {len(selected) - n_sme} synthetic)")
        print(f"wrote    : {out / 'selection.json'}")
        return 0

    logger = RunLogger(run_id=run_id)
    checkpoint = out / "answers.jsonl"
    done = load_done(checkpoint)
    total = len(usable) * len(retrieval_runs) * len(formats) * len(selected)
    print(f"run      : {run_id}")
    print(f"models   : {', '.join(s.key for s in usable)}")
    print(f"payloads : {', '.join(r.payload for r in retrieval_runs)}")
    n_sme = sum(1 for qid in selected if qid.startswith("sme-"))
    print(f"queries  : {len(selected)} ({n_sme} SME + "
          f"{len(selected) - n_sme} synthetic)  formats: {len(formats)}")
    print(f"cells    : {total} ({len(done)} already checkpointed)")
    print(f"workers  : {args.workers} (openai), {args.ollama_workers} (ollama)\n")

    # Build the pending cells up front; answer them, OpenAI cells ``--workers``
    # at a time and Ollama cells ``--ollama-workers`` at a time.
    #
    # Ollama defaults to 1 because that is right for a single-GPU local box: on
    # an M4 Max a lone stream already saturates the hardware, and a measured
    # 2-way pass moved throughput by under 5% (136.9 vs 142.9 tok/s) while
    # doubling per-call latency. It is wrong for a rented multi-slot GPU, where
    # this workload -- ~96% prefill -- is exactly what batching accelerates.
    # Raise it there, and raise the server's OLLAMA_NUM_PARALLEL to match.
    pending: list[tuple[ModelSpec, RetrievalRun, str, dict]] = []
    for spec in usable:
        for retrieval in retrieval_runs:
            for fmt in formats:
                for qid in selected:
                    row = retrieval.rows[(fmt, qid)]
                    cell = {
                        "run_id": run_id,
                        "retrieval_run": retrieval.run_id,
                        "payload": retrieval.payload,
                        "fmt": fmt,
                        "query_id": qid,
                        "source": row["source"],
                        "topic": row["topic"],
                        "provider": spec.provider,
                        "model": spec.model,
                        "expected": row["expected"],
                        "retrieved": [h["concept_id"] for h in row["retrieved"][: args.k]],
                    }
                    if cell_key(cell) not in done:
                        pending.append((spec, retrieval, fmt, cell))

    def answer_cell(item) -> dict:
        spec, retrieval, fmt, cell = item
        row = retrieval.rows[(fmt, cell["query_id"])]
        contexts = [load_context(retrieval.payload, fmt, cid) for cid in cell["retrieved"]]
        prompt = answer_prompt(row["query_text"], contexts)
        cell["prompt_sha256"] = hashlib.sha256(
            (ANSWER_SYSTEM + "\x00" + prompt).encode()
        ).hexdigest()

        meta: dict = {"fmt": fmt, "query_id": cell["query_id"], "payload": retrieval.payload}
        if spec.provider == "ollama":
            need = len(ANSWER_SYSTEM + prompt) // 3 + OLLAMA_CTX_MARGIN
            meta["ollama_options"] = {"num_ctx": min(OLLAMA_CTX_MAX, max(8192, need))}

        try:
            response = ask(
                spec, prompt, system=ANSWER_SYSTEM,
                purpose=PURPOSE_ANSWER, logger=logger,
                temperature=0.0, **meta,
            )
            cell.update(
                answer=response.text,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                cost_usd=response.cost_usd,
                latency_s=response.latency_s,
                error=None,
            )
        except LLMError as exc:
            cell.update(answer="", prompt_tokens=None, completion_tokens=None,
                        cost_usd=None, latency_s=None, error=str(exc))
        return cell

    started = time.time()
    n_done = len(done)
    sink_lock = threading.Lock()

    def record(cell: dict) -> None:
        nonlocal n_done
        with sink_lock:
            done[cell_key(cell)] = cell
            sink.write(json.dumps(cell, ensure_ascii=False) + "\n")
            sink.flush()
            n_done += 1
            if n_done % 25 == 0 or n_done == total:
                elapsed = time.time() - started
                print(f"  {n_done}/{total} cells  ({elapsed:.0f}s)", flush=True)

    with checkpoint.open("a") as sink:
        # Group by provider so the pool size can differ; order within a group
        # is the same model > payload > format > query order as before.
        for provider in ("openai", "ollama"):
            items = [it for it in pending if it[0].provider == provider]
            if not items:
                continue
            workers = args.workers if provider == "openai" else args.ollama_workers
            if workers <= 1:
                for item in items:
                    record(answer_cell(item))
                continue
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for cell in pool.map(answer_cell, items):
                    record(cell)

    summary = summarise(list(done.values()))
    (out / "answers_summary.json").write_text(json.dumps({
        **provenance(run_id),
        "retrieval_runs": [r.run_id for r in retrieval_runs],
        "models": [s.key for s in usable],
        "k": args.k,
        "queries": len(selected),
        "cells": len(done),
        "log_counts": logger.counts(),
        "summary": summary,
    }, indent=2) + "\n")

    print("\nmean tokens per answer call (prompt / completion), by payload x format x model:")
    header = f"{'payload':<10}{'format':<9}{'model':<28}{'n':>4}{'ok':>4}{'prompt':>9}{'compl':>8}{'lat s':>8}"
    print(header)
    print("-" * len(header))
    for key in sorted(summary):
        s = summary[key]
        payload, fmt, model = key.split("|")
        pt = f"{s['prompt_tokens']:.0f}" if s.get("prompt_tokens") is not None else "—"
        ct = f"{s['completion_tokens']:.0f}" if s.get("completion_tokens") is not None else "—"
        lt = f"{s['latency_s']:.1f}" if s.get("latency_s") is not None else "—"
        print(f"{payload:<10}{fmt:<9}{model:<28}{s['n']:>4}{s['n_answered']:>4}{pt:>9}{ct:>8}{lt:>8}")
    print(f"\nwrote {checkpoint}")
    print(f"wrote {out / 'answers_summary.json'}")
    print(f"raw calls in {logger.dir}/")
    return 0


def summarise(cells: list[dict]) -> dict:
    import statistics

    def mean(values):
        real = [v for v in values if v is not None]
        return round(statistics.fmean(real), 4) if real else None

    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for cell in cells:
        grouped[f"{cell['payload']}|{cell['fmt']}|{cell['provider']}:{cell['model']}"].append(cell)
    return {
        key: {
            "n": len(rows),
            "n_answered": sum(1 for r in rows if r.get("answer") and not r.get("error")),
            "prompt_tokens": mean([r.get("prompt_tokens") for r in rows]),
            "completion_tokens": mean([r.get("completion_tokens") for r in rows]),
            "latency_s": mean([r.get("latency_s") for r in rows]),
            "cost_usd": mean([r.get("cost_usd") for r in rows]),
        }
        for key, rows in sorted(grouped.items())
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--retrieval", nargs="+", required=True,
                        help="Stage R run ids under runs/ (one per payload)")
    parser.add_argument("--n", type=int, default=50, help="query slice size")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--k", type=int, default=10, help="contexts per prompt")
    parser.add_argument("--workers", type=int, default=1,
                        help="concurrent OpenAI cells")
    parser.add_argument("--ollama-workers", type=int, default=1,
                        help="concurrent Ollama cells (default 1: correct for a local "
                             "single-GPU box, which one stream already saturates). Raise "
                             "to 8-16 against a rented multi-slot GPU, and set the "
                             "server's OLLAMA_NUM_PARALLEL to at least the same number.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                        help="provider:model, e.g. openai:gpt-5.4-nano")
    parser.add_argument("--run-id", default=None,
                        help="resume an interrupted run under runs/RUN_ID")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the selected queries, for a quick pass")
    parser.add_argument("--smoke", action="store_true",
                        help="2 formats x first model x 5 queries, faceted only")
    parser.add_argument("--superset-of", default=None, metavar="RUN_ID",
                        help="make the selection contain runs/RUN_ID/selection.json's queries")
    parser.add_argument("--select-only", action="store_true",
                        help="run Stage S only: write selection.json and stop")
    parser.add_argument("--seed-answers-from", default=None, metavar="RUN_ID",
                        help="copy runs/RUN_ID's answered (and judged) cells that fall in "
                             "this run's grid into it before answering; requires the same "
                             "Stage R runs and cache fingerprints")
    args = parser.parse_args(argv)
    return run_stage(args)


if __name__ == "__main__":
    raise SystemExit(main())
