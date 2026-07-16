"""Experiment 1 — Format ablation on real CMR records.

Tests Report.md §6 Experiment 1: how the four record representations affect
retrieval and end-to-end answer accuracy, across the configured models, to probe
whether the best format is model-dependent.

Pipeline (honest RAG), per representation:
  1. Retrieval — embed every record under that representation, score all queries.
  2. Answer   — for each model, show the retrieved top-k candidates (rendered in
                that representation, each tagged with its concept-id) and ask the
                model to pick the concept-id that answers the question.

Then the validate+improve loop:
  * Robustness — re-run retrieval across seeds with paraphrased queries; flag
                 format pairs whose CIs overlap as not distinguishable.
  * Auto-tune  — hill-climb a templated variant of metadata_as_text on Recall@k.

Run:
    uv run python -m experiments.exp1_format_ablation.run [--max-queries N ...]
"""

from __future__ import annotations

import argparse
import random
import re
import statistics
from pathlib import Path

from airm import metrics
from airm.cmr import load_corpus
from airm.embeddings import Index
from airm.improve import (
    auto_tune,
    cis_overlap,
    paraphrase_queries,
    robustness_check,
)
from airm.llm import ProviderUnavailable, available_models, complete
from airm.queries import Query, _gold_cluster, load_queries, split_by_cluster
from airm.representations import RENDERERS, RENDERERS_FAIR, facets
from airm.run import (
    check_relevance_coverage,
    evaluate_retrieval,
    new_run_dir,
    set_seed,
    summarize,
    write_jsonl,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CONCEPT_ID_RE = re.compile(r"C\d+-[A-Z0-9_]+")

# Default templated renderer for the auto-tune target. Placeholders are the keys
# of flatten_facets(); the tuner may rewrite the prose but must keep the {fields}.
DEFAULT_TEMPLATE = (
    "{title} is a NASA Earth-science dataset archived at {data_center}. "
    "Instruments: {instruments}. Variables measured: {variables}. "
    "Spatial coverage: {spatial}. Temporal coverage: {temporal}. "
    "Quality: {quality}. {summary}"
)


# Run-log file handle; set by ``open_run_log`` in main() so every ``log`` /
# ``vlog`` call is also persisted to a timestamped file under the run dir.
_LOG_FH = None


def open_run_log(path: Path) -> None:
    global _LOG_FH
    _LOG_FH = path.open("w")


def close_run_log() -> None:
    global _LOG_FH
    if _LOG_FH is not None:
        _LOG_FH.close()
        _LOG_FH = None


def log(msg: str) -> None:
    """Status line: printed and (if open) appended to the run log."""
    line = f"[exp1] {msg}"
    print(line, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()


def vlog(msg: str) -> None:
    """Verbose dump: full text streamed to the console *and* the run log.

    Used for per-query/per-model prompts and outputs. Each line is flushed as it
    is produced so progress (including reasoning models' full <think> blocks) is
    visible live on the CLI while the run proceeds. Silence it with --no-verbose.
    """
    print(msg, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(msg + "\n")
        _LOG_FH.flush()


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #


def flatten_facets(record: dict) -> dict[str, str]:
    """Facets as plain strings for str.format templating (auto-tune target)."""
    f = facets(record)
    instruments = sorted({i for p in f["platforms"] for i in p["instruments"]})
    platforms = sorted({p["platform"] for p in f["platforms"] if p["platform"]})
    bbox = f["bbox"]
    spatial = "not specified"
    if bbox and None not in bbox.values():
        spatial = (
            f"{bbox['west']}° to {bbox['east']}° lon, "
            f"{bbox['south']}° to {bbox['north']}° lat"
        )
    t = f["temporal"]
    temporal = "not specified"
    if t and t.get("begin"):
        temporal = f"{t['begin']} to {t.get('end') or 'present'}"
    q = f["quality"]
    quality = ", ".join(
        s
        for s in [
            f"level {q['processing_level']}" if q["processing_level"] else "",
            f"version {q['version']}" if q["version"] else "",
            (q["collection_progress"] or "").lower(),
        ]
        if s
    ) or "not specified"
    return {
        "title": f["title"],
        "data_center": f["data_center"] or "an unspecified data center",
        "instruments": ", ".join(instruments) or "not specified",
        "platforms": ", ".join(platforms) or "not specified",
        "variables": ", ".join(f["variables"]) or "not specified",
        "spatial": spatial,
        "temporal": temporal,
        "quality": quality,
        "summary": f["summary"],
    }


def templated_renderer(template: str):
    """Return a render_fn(record)->str using ``template`` over flatten_facets."""

    def render(record: dict) -> str:
        return template.format(**flatten_facets(record))

    return render


def clean_answer(text: str) -> str:
    """Drop a reasoning model's <think>...</think> preamble before parsing.

    deepseek-r1 / qwen3-style models emit chain-of-thought in <think> tags; the
    final answer follows the last </think>. Keeping only the tail avoids matching
    a concept-id the model merely *considered* while reasoning.
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text


def extract_concept_id(text: str) -> str | None:
    m = CONCEPT_ID_RE.search(clean_answer(text or ""))
    return m.group(0) if m else None


def sample_queries(queries: list[Query], n: int, seed: int) -> list[Query]:
    set_seed(seed)
    if n >= len(queries):
        return list(queries)
    return random.sample(queries, n)


def tokens_per_record(records: list[dict], render_fn, cap: int = 100) -> float:
    sample = records[:cap]
    return statistics.mean(metrics.count_tokens(render_fn(r)) for r in sample)


# --------------------------------------------------------------------------- #
# Answer stage.
# --------------------------------------------------------------------------- #


def build_answer_prompt(question: str, candidates: list[tuple[str, str]]) -> str:
    blocks = []
    for i, (cid, rendered) in enumerate(candidates, start=1):
        blocks.append(f"[{i}] concept_id={cid}\n{rendered}")
    body = "\n\n".join(blocks)
    return (
        f"Question: {question}\n\n"
        f"Candidate datasets:\n{body}\n\n"
        "Reply with ONLY the concept_id of the single dataset that best answers "
        "the question."
    )


ANSWER_SYSTEM = (
    "You are a NASA Earth-science metadata search assistant. From the candidate "
    "datasets, choose the one that best answers the question. Respond with only a "
    "concept_id (e.g. C12345-PROV)."
)


def build_shared_candidates(
    q: Query,
    ref_ranked: list[str],
    corpus_by_id: dict[str, dict],
    *,
    answer_k: int,
    seed: int = 0,
) -> list[str]:
    """A candidate id set that is identical across formats and contains the gold.

    To measure *reasoning over format* (not retrieval), every format must be
    shown the same candidates, differing only in how they are rendered. We take
    the gold id(s) plus distractors from a fixed reference retrieval, cap at
    ``answer_k``, and shuffle deterministically so gold position is not a
    giveaway. Returns ``[]`` when the gold is not in the corpus (unanswerable).
    """
    gold = [c for c in q.relevant if c in corpus_by_id]
    if not gold:
        return []
    cand = list(gold[:answer_k])
    for cid in ref_ranked:
        if len(cand) >= answer_k:
            break
        if cid not in set(q.relevant) and cid in corpus_by_id:
            cand.append(cid)
    random.Random(f"{q.id}:{seed}").shuffle(cand)
    return cand


def run_answer_stage(
    corpus_by_id: dict[str, dict],
    retrieved: dict[tuple[str, str], list[str]],
    sampled: list[Query],
    models: list[str],
    *,
    answer_k: int,
    renderers: dict = RENDERERS,
    shared_candidates: dict[str, list[str]] | None = None,
    max_tokens: int = 512,
    verbose: bool = True,
) -> list[dict]:
    """For each (model, representation, sampled query), grade the LLM's pick.

    Two modes:
      * coupled (default): candidates are that representation's own retrieved
        top-k — the end-to-end RAG measurement, bounded by retrieval recall.
      * decoupled (``shared_candidates`` given): every representation is shown the
        *same* candidate id set (gold + fixed distractors), rendered in its own
        format, so accuracy isolates reasoning-over-format from retrieval.

    ``max_tokens`` is generous so reasoning models can finish a <think> block and
    still emit the concept-id (which ``extract_concept_id`` parses post-think).
    When ``verbose``, every query and each model's raw output is written to the
    run log (full text) and previewed on the console.
    """
    rows: list[dict] = []
    for rep, render_fn in renderers.items():
        for q in sampled:
            if shared_candidates is not None:
                top_ids = shared_candidates.get(q.id, [])
            else:
                top_ids = retrieved.get((rep, q.id), [])[:answer_k]
            candidates = [
                (cid, render_fn(corpus_by_id[cid])) for cid in top_ids if cid in corpus_by_id
            ]
            if not candidates:
                continue
            prompt = build_answer_prompt(q.question, candidates)
            n_tokens = metrics.count_tokens(prompt)
            if verbose:
                cand_ids = [cid for cid, _ in candidates]
                vlog(
                    f"\n=== query {q.id} | representation={rep} | "
                    f"difficulty={q.difficulty or '?'} ===\n"
                    f"  question: {q.question}\n"
                    f"  gold={q.relevant} candidates={cand_ids}"
                )
            for model in models:
                if verbose:
                    vlog(f"  [{model}] querying…")
                try:
                    out = complete(model, prompt, system=ANSWER_SYSTEM, max_tokens=max_tokens)
                except ProviderUnavailable as exc:
                    log(f"skip {model}: {exc}")
                    continue
                pred = extract_concept_id(out.text)
                correct = pred in set(q.relevant)
                rows.append(
                    {
                        "representation": rep,
                        "model": model,
                        "query_id": q.id,
                        "difficulty": q.difficulty,
                        "predicted": pred,
                        "relevant": q.relevant,
                        "correct": bool(correct),
                        "prompt_tokens": n_tokens,
                        "output_tokens": out.output_tokens,
                        "output": out.text,
                        "stage": "decoupled" if shared_candidates is not None else "coupled",
                        "retrieved_in_candidates": any(
                            cid in set(q.relevant) for cid in top_ids
                        ),
                    }
                )
                if verbose:
                    mark = "✓" if correct else "✗"
                    vlog(
                        f"  [{model}] pred={pred} {mark} "
                        f"(out_tokens={out.output_tokens})\n"
                        f"    output: {out.text.strip()}"
                    )
    return rows


# --------------------------------------------------------------------------- #
# Validate + improve loop.
# --------------------------------------------------------------------------- #


def run_robustness(corpus, sampled, k, renderers=RENDERERS):
    """Per-representation robustness across seeds with paraphrased queries."""
    reports = {}
    for rep, render_fn in renderers.items():
        index = Index.build(corpus, render_fn)

        def scorer(seed, index=index):
            set_seed(seed)
            pqs = paraphrase_queries(sampled, seed=seed)
            return [
                metrics.recall_at_k(
                    [cid for cid, _ in index.search(q.question, k=k)], set(q.relevant), k
                )
                for q in pqs
            ]

        reports[rep] = robustness_check(rep, scorer, seeds=(0, 1, 2))
        log(
            f"robustness {rep}: recall={reports[rep].mean:.3f} "
            f"CI[{reports[rep].ci_lo:.3f},{reports[rep].ci_hi:.3f}]"
        )
    return reports


def score_template(corpus, queries, k, template: str) -> float:
    """Mean Recall@k of a templated renderer over ``queries`` (−inf if invalid)."""
    try:
        render_fn = templated_renderer(template)
        index = Index.build(corpus, render_fn)
    except Exception:  # invalid placeholder / format error -> reject
        return float("-inf")
    if not queries:
        return 0.0
    return statistics.mean(
        metrics.recall_at_k(
            [cid for cid, _ in index.search(q.question, k=k)], set(q.relevant), k
        )
        for q in queries
    )


def run_auto_tune(corpus, tune_queries, k, model, rounds):
    """Hill-climb the templated metadata_as_text renderer on Recall@k.

    ``tune_queries`` should be a tuning split *disjoint* from the held-out queries
    used to report the final number, so the optimizer cannot train on the test
    set. The caller re-scores the winner on the held-out split afterward.
    """

    def score_fn(template: str) -> float:
        return score_template(corpus, tune_queries, k, template)

    context = (
        "The template renders one CMR dataset record as a natural-language summary "
        "for embedding-based retrieval. It MUST keep these placeholders exactly: "
        "{title}, {data_center}, {instruments}, {platforms}, {variables}, "
        "{spatial}, {temporal}, {quality}, {summary}. Improve wording/order for "
        "better retrieval; do not add other placeholders."
    )
    return auto_tune(
        DEFAULT_TEMPLATE, score_fn, model=model, rounds=rounds, context=context
    )


# --------------------------------------------------------------------------- #
# Summary.
# --------------------------------------------------------------------------- #


def _answer_table(lines: list[str], rows: list[dict], cost: dict, renderers: dict) -> dict:
    """Render one format×model accuracy table; return {rep: [acc per model]}."""
    models = sorted({r["model"] for r in rows})
    lines.append("| representation | " + " | ".join(models) + " |")
    lines.append("|---|" + "---|" * len(models))
    acc: dict = {}
    for rep in renderers:
        cells = []
        for m in models:
            sel = [r for r in rows if r["representation"] == rep and r["model"] == m]
            a = statistics.mean(r["correct"] for r in sel) if sel else 0.0
            acc.setdefault(rep, []).append(a)
            cells.append(f"{a:.2f}" if sel else "—")
        lines.append(f"| {rep} | " + " | ".join(cells) + " |")
    return acc


def write_summary(
    retrieval_summary,
    robustness,
    answer_rows,
    cost,
    tune_result,
    tune_baseline,
    path: Path,
    *,
    renderers: dict = RENDERERS,
    fair: bool = False,
    tune_heldout: float | None = None,
    coverage=None,
):
    lines = ["# Experiment 1 — Format ablation results\n"]
    lines.append(
        "**Content-hold mode:** "
        + (
            "`fair` — all four renderers driven from the same `facets()` payload, "
            "so differences reflect *format only*."
            if fair
            else "`default` — `raw_umm_json`/`dot_breadcrumb` render the full UMM "
            "tree (a superset of the curated formats' content), so retrieval "
            "differences confound format with information content. Re-run with "
            "`--fair` to isolate format."
        )
        + "\n"
    )

    # Coverage of ground truth vs corpus (silent-failure guard).
    if coverage is not None:
        lines.append("## Ground-truth coverage\n")
        lines.append(
            f"- {coverage.n_ok}/{coverage.n_queries} queries have an in-corpus "
            f"relevant id (max achievable recall)."
        )
        if coverage.missing_relevant:
            lines.append(
                f"- {len(coverage.missing_relevant)} query(ies) reference relevant "
                "ids **absent from the corpus** (these can never be retrieved and "
                "score a hard 0)."
            )
        if coverage.empty_relevant:
            lines.append(
                f"- {len(coverage.empty_relevant)} query(ies) have **no** ground "
                "truth and were excluded from scoring."
            )
        if not coverage.missing_relevant and not coverage.empty_relevant:
            lines.append("- No dangling or empty ground truth. ✓")
        lines.append("")

    # Retrieval table.
    lines.append("## Retrieval (all queries)\n")
    lines.append("| representation | Recall@k | 95% CI (cluster) | MRR | nDCG | tokens/rec |")
    lines.append("|---|---|---|---|---|---|")
    for rep, s in sorted(
        retrieval_summary.items(), key=lambda kv: -kv[1]["recall_at_k_mean"]
    ):
        lines.append(
            f"| {rep} | {s['recall_at_k_mean']:.3f} | "
            f"[{s['recall_at_k_lo']:.3f}, {s['recall_at_k_hi']:.3f}] | "
            f"{s['mrr_mean']:.3f} | {s['ndcg_mean']:.3f} | {cost.get(rep, 0):.0f} |"
        )
    lines.append(
        "\n_CIs are cluster-bootstrapped by ground-truth dataset (not per query), "
        "since the query set is paraphrase-clustered — a per-query bootstrap would "
        "understate their width._"
    )

    # Robustness distinguishability.
    lines.append("\n## Robustness — are format differences distinguishable?\n")
    reps = list(robustness)
    lines.append("Format pairs whose Recall@k CIs **overlap** (difference not robust):\n")
    overlaps = []
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            a, b = robustness[reps[i]], robustness[reps[j]]
            if cis_overlap(a, b):
                overlaps.append(f"- {reps[i]} ≈ {reps[j]} (CIs overlap)")
    lines.extend(overlaps or ["- (none — all pairwise differences are distinguishable)"])

    # Answer stages: coupled (end-to-end RAG) and decoupled (reasoning-only).
    coupled = [r for r in answer_rows if r.get("stage") != "decoupled"]
    decoupled = [r for r in answer_rows if r.get("stage") == "decoupled"]

    lines.append("\n## End-to-end answer accuracy — coupled (RAG)\n")
    if not coupled:
        lines.append(
            "_No models configured (keyed cloud tiers + pulled Ollama models). "
            "Answer stage skipped; retrieval + robustness above stand on their own._"
        )
    else:
        lines.append(
            "Candidates are each format's *own* retrieved top-k, so accuracy here "
            "is bounded by retrieval recall — this measures the full pipeline, not "
            "format's effect on reasoning (see the decoupled table below).\n"
        )
        _answer_table(lines, coupled, cost, renderers)

    if decoupled:
        lines.append("\n## Answer accuracy — decoupled (reasoning over format)\n")
        lines.append(
            "Every format is shown the **same** candidate set (gold + fixed "
            "distractors), rendered in its own format. Gold is always present, so "
            "this isolates how format affects the model's *selection/reasoning*, "
            "independent of retrieval.\n"
        )
        acc = _answer_table(lines, decoupled, cost, renderers)

        lines.append("\n### Model-dependence verdict (decoupled)\n")
        models = sorted({r["model"] for r in decoupled})
        best_by_model = {}
        for mi, m in enumerate(models):
            best_rep = max(renderers, key=lambda rep: acc[rep][mi])
            best_by_model[m] = best_rep
        for m, rep in best_by_model.items():
            lines.append(f"- best format for **{m}**: `{rep}`")
        distinct = len(set(best_by_model.values()))
        lines.append(
            f"\n**{'Model-dependent' if distinct > 1 else 'Consistent'}**: "
            f"{distinct} distinct best-format(s) across {len(models)} models."
        )

        pts = [(rep, statistics.mean(acc[rep]), cost.get(rep, 0.0)) for rep in renderers]
        front = metrics.pareto_frontier(pts)
        lines.append("\n### Pareto frontier (decoupled accuracy vs token cost)\n")
        for rep, a, c in pts:
            mark = " ⬅ frontier" if rep in front else ""
            lines.append(f"- {rep}: acc={a:.2f}, tokens/rec={c:.0f}{mark}")

    # Auto-tune.
    lines.append("\n## Auto-tune (metadata_as_text template)\n")
    if tune_result is None:
        lines.append("_Skipped (no model available for the optimizer)._")
    else:
        heldout = (
            f"{tune_heldout:.3f}" if tune_heldout is not None else "n/a"
        )
        lines.append(
            f"- baseline templated Recall@k (tuning split): {tune_baseline:.3f}\n"
            f"- tuned Recall@k (tuning split): {tune_result.best.score:.3f} "
            f"(round {tune_result.best.round}, plateaued={tune_result.plateaued})\n"
            f"- **tuned Recall@k on held-out test split: {heldout}** ← the honest "
            "number; the tuning-split gain is optimistic (selection over rounds).\n"
            f"- winning template saved to `results/tuned_metadata_as_text.txt`"
        )

    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1 — format ablation.")
    parser.add_argument("--k", type=int, default=10, help="retrieval depth")
    parser.add_argument("--answer-k", type=int, default=5, help="candidates shown to the model")
    parser.add_argument("--max-queries", type=int, default=20, help="answer/tune query sample size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tune-rounds", type=int, default=3)
    parser.add_argument(
        "--models",
        default="",
        help="comma-separated model ids to use (default: all configured via "
        "available_models — every pulled Ollama model + keyed cloud tiers). "
        "Pin fast models here to bound cost, e.g. 'llama3.2:latest,gemma3:4b'.",
    )
    parser.add_argument("--answer-max-tokens", type=int, default=512)
    parser.add_argument("--no-answer", action="store_true")
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument(
        "--fair",
        action="store_true",
        help="use the content-held renderer registry (RENDERERS_FAIR) so all four "
        "formats see the same facet content — isolates format from information "
        "content in the retrieval comparison.",
    )
    parser.add_argument(
        "--no-decoupled",
        action="store_true",
        help="skip the decoupled answer stage (shared candidate set); run only the "
        "coupled end-to-end RAG answer stage.",
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="disable per-query/per-model output logging (still writes answers.jsonl)",
    )
    args = parser.parse_args()
    verbose = not args.no_verbose
    renderers = RENDERERS_FAIR if args.fair else RENDERERS

    run_dir = new_run_dir(RESULTS_DIR)
    log_path = run_dir / f"run-{run_dir.name}.log"
    open_run_log(log_path)
    log(f"run dir: {run_dir} (also linked as results/latest)")
    log(f"run log: {log_path}")
    corpus = load_corpus()
    queries = load_queries()
    corpus_by_id = {r["meta"]["concept-id"]: r for r in corpus}
    log(f"content-hold mode: {'fair (facet-only)' if args.fair else 'default (full-tree)'}")

    # --- Ground-truth coverage (silent-failure guard) ---
    coverage = check_relevance_coverage(corpus, queries)
    log(
        f"coverage: {coverage.n_ok}/{coverage.n_queries} queries have an in-corpus "
        f"relevant id; {len(coverage.missing_relevant)} with dangling ids, "
        f"{len(coverage.empty_relevant)} with empty ground truth"
    )
    if coverage.missing_relevant:
        log(f"  dangling relevant ids (sample): {dict(list(coverage.missing_relevant.items())[:5])}")

    # Score only queries whose ground truth is actually reachable, so empty /
    # dangling ground truth does not silently depress every representation.
    scored_queries = [
        q for q in queries
        if q.relevant and any(c in corpus_by_id for c in q.relevant)
    ]
    if len(scored_queries) != len(queries):
        log(f"scoring {len(scored_queries)}/{len(queries)} queries with reachable ground truth")

    # Cluster-aware train/val/test split (paraphrases of one gold stay together).
    splits = split_by_cluster(scored_queries, seed=args.seed)
    log(f"splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    # Cluster map (query_id -> gold dataset) for honest, cluster-bootstrapped CIs.
    clusters = {q.id: _gold_cluster(q) for q in scored_queries}
    sampled = sample_queries(scored_queries, args.max_queries, args.seed)
    log(f"corpus={len(corpus)} queries={len(queries)} scored={len(scored_queries)} sample={len(sampled)}")

    # --- Retrieval stage (all scored queries, all representations) ---
    all_retrieval = []
    retrieved: dict[tuple[str, str], list[str]] = {}
    for rep, fn in renderers.items():
        results = evaluate_retrieval(corpus, scored_queries, fn, rep, k=args.k)
        all_retrieval += results
        for r in results:
            retrieved[(rep, r.query_id)] = r.top_ids
        log(f"retrieval {rep}: built + scored {len(results)} queries")
    write_jsonl(all_retrieval, run_dir / "retrieval.jsonl")
    retrieval_summary = summarize(all_retrieval, clusters=clusters)
    cost = {rep: tokens_per_record(corpus, fn) for rep, fn in renderers.items()}

    # --- Robustness ---
    robustness = run_robustness(corpus, sampled, args.k, renderers)

    # --- Answer stage ---
    if args.models.strip():
        requested = [m.strip() for m in args.models.split(",") if m.strip()]
        configured = set(available_models(requested))
        models = [m for m in requested if m in configured]
        missing = [m for m in requested if m not in configured]
        if missing:
            log(f"requested but not configured (skipped): {missing}")
    else:
        models = available_models()
    answer_rows: list[dict] = []
    if args.no_answer:
        log("answer stage disabled (--no-answer)")
    elif not models:
        log("answer stage skipped: no models configured (need OPENAI_API_KEY and/or pulled Ollama models)")
    else:
        if len(models) < 3:
            log(f"WARNING: only {len(models)} model(s) available; report design wants ≥3: {models}")
        else:
            log(f"models: {models}")

        # Coupled (end-to-end RAG): each format answers over its own top-k.
        answer_rows = run_answer_stage(
            corpus_by_id,
            retrieved,
            sampled,
            models,
            answer_k=args.answer_k,
            renderers=renderers,
            max_tokens=args.answer_max_tokens,
            verbose=verbose,
        )

        # Decoupled (reasoning over format): every format sees the SAME candidate
        # set (gold + fixed distractors from the reference retriever), so accuracy
        # is not bounded by that format's own recall.
        if not args.no_decoupled:
            ref = "metadata_as_text" if "metadata_as_text" in renderers else next(iter(renderers))
            shared = {
                q.id: build_shared_candidates(
                    q, retrieved.get((ref, q.id), []), corpus_by_id,
                    answer_k=args.answer_k, seed=args.seed,
                )
                for q in sampled
            }
            n_answerable = sum(1 for v in shared.values() if v)
            log(f"decoupled stage: {n_answerable}/{len(sampled)} queries have gold in corpus; ref={ref}")
            answer_rows += run_answer_stage(
                corpus_by_id,
                retrieved,
                sampled,
                models,
                answer_k=args.answer_k,
                renderers=renderers,
                shared_candidates=shared,
                max_tokens=args.answer_max_tokens,
                verbose=verbose,
            )
        write_jsonl(answer_rows, run_dir / "answers.jsonl")

    # --- Auto-tune (train on the tuning split, report on held-out test) ---
    tune_result = None
    tune_baseline = 0.0
    tune_heldout = None
    if args.no_tune:
        log("auto-tune disabled (--no-tune)")
    elif not models:
        log("auto-tune skipped: no optimizer model available")
    else:
        tune_split = splits["train"] + splits["val"]
        test_split = splits["test"]
        tune_baseline = score_template(corpus, tune_split, args.k, DEFAULT_TEMPLATE)
        log(
            f"auto-tune baseline Recall@{args.k}={tune_baseline:.3f} on "
            f"{len(tune_split)} tuning queries; optimizing with {models[0]}"
        )
        tune_result = run_auto_tune(corpus, tune_split, args.k, models[0], args.tune_rounds)
        # The honest number: re-score the winning template on the held-out test
        # split it was never optimized against.
        tune_heldout = score_template(corpus, test_split, args.k, tune_result.best.text)
        (run_dir / "tuned_metadata_as_text.txt").write_text(tune_result.best.text + "\n")
        log(
            f"auto-tune tuning-split Recall@{args.k}={tune_result.best.score:.3f}; "
            f"held-out test Recall@{args.k}={tune_heldout:.3f}"
        )

    # --- Summary ---
    write_summary(
        retrieval_summary,
        robustness,
        answer_rows,
        cost,
        tune_result,
        tune_baseline,
        run_dir / "summary.md",
        renderers=renderers,
        fair=args.fair,
        tune_heldout=tune_heldout,
        coverage=coverage,
    )
    log(f"wrote {run_dir / 'summary.md'}")
    log(f"verbose run log saved to {log_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        close_run_log()
