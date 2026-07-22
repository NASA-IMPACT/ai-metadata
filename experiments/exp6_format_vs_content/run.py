"""Experiment 6 — Format vs. content for CMR ingestion & retrieval.

Decomposes the confound behind the folklore "flatten everything to Metadata-as-
Text and raw JSON loses": that comparison varies *format* and *content* at once.
Here they are separate independent variables.

Six conditions = format × content-scope, assembled from the existing renderer
registries (no new renderers, no changes to `airm/`):

    raw_json__full     breadcrumb__full
    raw_json__facet    breadcrumb__facet    jsonld__facet    mat__facet

Three axes, reported separately (never collapsed into one "winner"):
  1. Retrieval        — Recall@10 / MRR / nDCG, cluster-bootstrapped CIs.
  2. Reasoning (decoupled) — every condition shown the SAME gold-containing
                        candidate set, rendered in its own format; grade the pick.
  3. Cost             — tokens/record; Pareto over (reasoning accuracy, cost).

Run (three OpenAI models):
    uv run python -m experiments.exp6_format_vs_content.run \
      --models "gpt-5.4-nano,gpt-4.1-mini,gpt-4o-mini" --max-queries 12
"""

from __future__ import annotations

import argparse
import random
import re
import statistics
from pathlib import Path

from airm import metrics
from airm.cmr import load_corpus
from airm.llm import ProviderUnavailable, available_models, complete
from airm.queries import Query, load_queries
from airm.representations import RENDERERS, RENDERERS_FAIR
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

# The six conditions: format × content-scope. `full` renders the whole UMM tree;
# `facet` renders only the shared 9-field facets() payload (RENDERERS_FAIR).
# jsonld / mat are facet-only by construction, so they have no `full` variant.
CONDITIONS = {
    "raw_json__full": RENDERERS["raw_umm_json"],
    "raw_json__facet": RENDERERS_FAIR["raw_umm_json"],
    "breadcrumb__full": RENDERERS["dot_breadcrumb"],
    "breadcrumb__facet": RENDERERS_FAIR["dot_breadcrumb"],
    "jsonld__facet": RENDERERS_FAIR["flattened_jsonld"],
    "mat__facet": RENDERERS_FAIR["metadata_as_text"],
}
FACET_CONDITIONS = [c for c in CONDITIONS if c.endswith("__facet")]
REFERENCE_CONDITION = "mat__facet"  # source of decoupled-stage distractors

ANSWER_SYSTEM = (
    "You are a NASA Earth-science metadata search assistant. From the candidate "
    "datasets, choose the one that best answers the question. Respond with only a "
    "concept_id (e.g. C12345-PROV)."
)


# --------------------------------------------------------------------------- #
# Logging (also persisted to a per-run log file).
# --------------------------------------------------------------------------- #

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
    line = f"[exp6] {msg}"
    print(line, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()


def vlog(msg: str) -> None:
    print(msg, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(msg + "\n")
        _LOG_FH.flush()


# --------------------------------------------------------------------------- #
# Helpers (self-contained — no imports from exp1 so it stays untouched).
# --------------------------------------------------------------------------- #


def clean_answer(text: str) -> str:
    """Drop a reasoning model's <think>…</think> preamble before parsing."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text


def extract_concept_id(text: str) -> str | None:
    m = CONCEPT_ID_RE.search(clean_answer(text or ""))
    return m.group(0) if m else None


def sample_queries(queries: list[Query], n: int, seed: int) -> list[Query]:
    """Sample ``n`` queries spread across ground-truth clusters.

    The query set is paraphrase-clustered by gold dataset, so a uniform sample
    can land several paraphrases of the same dataset and leave the reasoning
    accuracy resting on very few *distinct* datasets. Drawing round-robin over
    shuffled clusters keeps the sample spanning as many datasets as possible,
    which is also what makes the cluster bootstrap on this axis meaningful.
    """
    set_seed(seed)
    if n >= len(queries):
        return list(queries)
    rng = random.Random(seed)
    by_cluster: dict[str, list[Query]] = {}
    for q in queries:
        by_cluster.setdefault(q.relevant[0], []).append(q)
    for group in by_cluster.values():
        rng.shuffle(group)
    order = sorted(by_cluster)
    rng.shuffle(order)
    picked: list[Query] = []
    depth = 0
    while len(picked) < n:
        added = False
        for key in order:
            group = by_cluster[key]
            if depth < len(group):
                picked.append(group[depth])
                added = True
                if len(picked) == n:
                    break
        if not added:
            break
        depth += 1
    return picked


def tokens_per_record(records: list[dict], render_fn, cap: int = 100) -> float:
    sample = records[:cap]
    return statistics.mean(metrics.count_tokens(render_fn(r)) for r in sample)


def build_shared_candidates(
    q: Query,
    ref_ranked: list[str],
    corpus_by_id: dict[str, dict],
    *,
    answer_k: int,
    seed: int = 0,
) -> list[str]:
    """A candidate id set identical across conditions and containing the gold.

    Isolates reasoning-over-format from retrieval: gold id(s) + distractors from a
    fixed reference retrieval, capped at answer_k, shuffled deterministically so
    gold position is not a giveaway. Empty when the gold is not in the corpus.
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


def build_answer_prompt(question: str, candidates: list[tuple[str, str]]) -> str:
    blocks = [f"[{i}] concept_id={cid}\n{rendered}" for i, (cid, rendered) in enumerate(candidates, 1)]
    return (
        f"Question: {question}\n\n"
        f"Candidate datasets:\n" + "\n\n".join(blocks) + "\n\n"
        "Reply with ONLY the concept_id of the single dataset that best answers "
        "the question."
    )


def run_reasoning_stage(
    corpus_by_id: dict[str, dict],
    shared: dict[str, list[str]],
    sampled: list[Query],
    models: list[str],
    *,
    max_tokens: int,
    verbose: bool,
) -> list[dict]:
    """Decoupled reasoning: same candidate set per query, rendered per condition."""
    rows: list[dict] = []
    for cond, render_fn in CONDITIONS.items():
        for q in sampled:
            cand_ids = shared.get(q.id, [])
            candidates = [(cid, render_fn(corpus_by_id[cid])) for cid in cand_ids if cid in corpus_by_id]
            if not candidates:
                continue
            prompt = build_answer_prompt(q.question, candidates)
            n_tokens = metrics.count_tokens(prompt)
            if verbose:
                vlog(f"\n=== {q.id} | condition={cond} | gold={q.relevant} ===")
            for model in models:
                try:
                    out = complete(model, prompt, system=ANSWER_SYSTEM, max_tokens=max_tokens)
                except ProviderUnavailable as exc:
                    log(f"skip {model}: {exc}")
                    continue
                pred = extract_concept_id(out.text)
                correct = pred in set(q.relevant)
                rows.append(
                    {
                        "condition": cond,
                        "model": model,
                        "query_id": q.id,
                        "predicted": pred,
                        "relevant": q.relevant,
                        "correct": bool(correct),
                        "prompt_tokens": n_tokens,
                        "output_tokens": out.output_tokens,
                    }
                )
                if verbose:
                    vlog(f"  [{model}] pred={pred} {'✓' if correct else '✗'}")
    return rows


# --------------------------------------------------------------------------- #
# Summary.
# --------------------------------------------------------------------------- #


def _reasoning_accuracy(rows: list[dict]) -> dict[str, dict[str, float]]:
    """{condition: {model: accuracy}} over decoupled rows."""
    models = sorted({r["model"] for r in rows})
    acc: dict[str, dict[str, float]] = {}
    for cond in CONDITIONS:
        acc[cond] = {}
        for m in models:
            sel = [r for r in rows if r["condition"] == cond and r["model"] == m]
            acc[cond][m] = statistics.mean(r["correct"] for r in sel) if sel else float("nan")
    return acc


def _reasoning_ci(rows: list[dict]) -> dict[str, tuple[float, float, float, int, int]]:
    """{condition: (mean, lo, hi, n_rows, n_clusters)} pooled over models.

    Without an interval on this axis a 0.97-vs-1.00 gap reads as a winner when it
    can be a single query. Clusters are the gold dataset, matching the retrieval
    axis, so correlated paraphrases don't shrink the interval.
    """
    out: dict[str, tuple[float, float, float, int, int]] = {}
    for cond in CONDITIONS:
        sel = [r for r in rows if r["condition"] == cond]
        if not sel:
            continue
        values = [float(r["correct"]) for r in sel]
        clusters = [r["relevant"][0] if r.get("relevant") else r["query_id"] for r in sel]
        mean, lo, hi = metrics.bootstrap_ci_clustered(values, clusters)
        out[cond] = (mean, lo, hi, len(sel), len(set(clusters)))
    return out


def _ci_overlap_pairs(ci: dict[str, tuple[float, float, float, int, int]]) -> list[str]:
    keys = list(ci)
    pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = ci[keys[i]], ci[keys[j]]
            if not (a[2] < b[1] or b[2] < a[1]):
                pairs.append(f"- {keys[i]} ≈ {keys[j]} (CIs overlap)")
    return pairs


def write_summary(retrieval_summary, cost, answer_rows, coverage, path: Path) -> None:
    lines = ["# Experiment 6 — Format vs. content: results\n"]

    if coverage is not None:
        lines.append(
            f"Ground-truth coverage: {coverage.n_ok}/{coverage.n_queries} queries have an "
            f"in-corpus relevant id; {len(coverage.missing_relevant)} dangling, "
            f"{len(coverage.empty_relevant)} empty.\n"
        )

    # --- Axis 1: retrieval + cost, all six conditions ---
    lines.append("## Axis 1 — Retrieval + cost (all conditions)\n")
    lines.append("| condition | Recall@10 | 95% CI (cluster) | MRR | nDCG | tokens/rec |")
    lines.append("|---|---|---|---|---|---|")
    for cond, s in sorted(retrieval_summary.items(), key=lambda kv: -kv[1]["recall_at_k_mean"]):
        lines.append(
            f"| {cond} | {s['recall_at_k_mean']:.3f} | "
            f"[{s['recall_at_k_lo']:.3f}, {s['recall_at_k_hi']:.3f}] | "
            f"{s['mrr_mean']:.3f} | {s['ndcg_mean']:.3f} | {cost.get(cond, 0):.0f} |"
        )

    # --- H1: content effect (facet vs full within a format) ---
    lines.append("\n## H1 — Content effect (facet vs. full, same format)\n")
    for fmt in ("raw_json", "breadcrumb"):
        f, u = f"{fmt}__facet", f"{fmt}__full"
        if f in retrieval_summary and u in retrieval_summary:
            rf = retrieval_summary[f]["recall_at_k_mean"]
            ru = retrieval_summary[u]["recall_at_k_mean"]
            lines.append(
                f"- **{fmt}**: facet={rf:.3f} vs full={ru:.3f} "
                f"(Δ={rf - ru:+.3f}); tokens {cost.get(f,0):.0f} vs {cost.get(u,0):.0f}"
            )
    lines.append(
        "\n_H1 holds if facet ≫ full — i.e. the full UMM tree hurts retrieval "
        "(content dilution), so the naive 'raw JSON is a bad format' claim is "
        "really about content, not format._"
    )

    # --- H2: format effect at facet scope (CI overlap) ---
    lines.append("\n## H2 — Format effect at facet scope (content held constant)\n")
    facet = {c: retrieval_summary[c] for c in FACET_CONDITIONS if c in retrieval_summary}
    if facet:
        recalls = [s["recall_at_k_mean"] for s in facet.values()]
        lines.append(
            f"Recall@10 spread across the 4 facet-scope formats: "
            f"{min(recalls):.3f}–{max(recalls):.3f} (Δ={max(recalls) - min(recalls):.3f}).\n"
        )
        keys = list(facet)
        overlaps = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = facet[keys[i]], facet[keys[j]]
                if not (a["recall_at_k_hi"] < b["recall_at_k_lo"] or b["recall_at_k_hi"] < a["recall_at_k_lo"]):
                    overlaps.append(f"- {keys[i]} ≈ {keys[j]} (CIs overlap)")
        lines.append("Format pairs whose CIs **overlap** (not distinguishable):\n")
        lines.extend(overlaps or ["- (none — all facet-scope formats are distinguishable)"])
        lines.append(
            "\n_H2 holds if the spread is small and most CIs overlap — i.e. with "
            "content held constant, format barely moves retrieval._"
        )

    # --- Axis 2: decoupled reasoning (H3/H4) ---
    lines.append("\n## Axis 2 — Decoupled reasoning (reasoning over format)\n")
    if not answer_rows:
        lines.append("_Reasoning stage skipped (no models configured)._")
        path.write_text("\n".join(lines) + "\n")
        return

    acc = _reasoning_accuracy(answer_rows)
    models = sorted({r["model"] for r in answer_rows})
    lines.append(
        "Every condition is shown the **same** gold-containing candidate set, so "
        "accuracy reflects reasoning over format, not retrieval.\n"
    )
    lines.append("| condition | " + " | ".join(models) + " | mean |")
    lines.append("|---|" + "---|" * (len(models) + 1))
    cond_mean = {}
    for cond in CONDITIONS:
        cells = [f"{acc[cond][m]:.2f}" if acc[cond][m] == acc[cond][m] else "—" for m in models]
        vals = [acc[cond][m] for m in models if acc[cond][m] == acc[cond][m]]
        cond_mean[cond] = statistics.mean(vals) if vals else float("nan")
        lines.append(f"| {cond} | " + " | ".join(cells) + f" | {cond_mean[cond]:.2f} |")

    # Pooled-over-models CI per condition, so the reasoning axis carries the same
    # uncertainty reporting the retrieval axis does.
    ci = _reasoning_ci(answer_rows)
    if ci:
        n_rows = next(iter(ci.values()))[3]
        n_clusters = next(iter(ci.values()))[4]
        lines.append("\n### Reasoning accuracy with CIs (pooled over models)\n")
        lines.append("| condition | accuracy | 95% CI (cluster) | n | distinct datasets |")
        lines.append("|---|---|---|---|---|")
        for cond, (mean, lo, hi, n, nc) in sorted(ci.items(), key=lambda kv: -kv[1][0]):
            lines.append(f"| {cond} | {mean:.3f} | [{lo:.3f}, {hi:.3f}] | {n} | {nc} |")
        overlaps = _ci_overlap_pairs(ci)
        lines.append("\nCondition pairs whose CIs **overlap** (not distinguishable):\n")
        lines.extend(overlaps or ["- (none — all conditions are distinguishable)"])
        if len(overlaps) == len(ci) * (len(ci) - 1) // 2:
            lines.append(
                "\n**No condition is statistically distinguishable on this axis.** Any "
                "'winner' named below is a point estimate only."
            )
        if n_clusters < 20:
            lines.append(
                f"\n⚠️ Only **{n_clusters} distinct gold datasets** in the reasoning sample "
                f"({n_rows} rows/condition). The cluster bootstrap resamples those "
                f"{n_clusters} units, so these intervals are wide and the per-model cells "
                "below are coarse. Raise `--max-queries` before quoting any of it."
            )

    # H4: best format per model + model-dependence
    lines.append("\n### H4 — Model-dependence (best condition per model)\n")
    best_by_model = {m: max(CONDITIONS, key=lambda c: (acc[c][m] if acc[c][m] == acc[c][m] else -1)) for m in models}
    for m, c in best_by_model.items():
        lines.append(f"- **{m}**: `{c}` ({acc[c][m]:.2f})")
    distinct = len(set(best_by_model.values()))
    lines.append(
        f"\n**{'Model-dependent' if distinct > 1 else 'Consistent'}**: "
        f"{distinct} distinct best-condition(s) across {len(models)} models."
    )
    # A per-model argmax over near-tied cells is not evidence either way; say so
    # rather than letting the verdict word stand alone.
    spread = [
        max(v for v in acc[c].values() if v == v) - min(v for v in acc[c].values() if v == v)
        for c in CONDITIONS
        if any(v == v for v in acc[c].values())
    ]
    if spread and max(spread) < 0.15:
        lines.append(
            "\n⚠️ Every condition's per-model spread is <0.15 — the models are near-tied "
            "(likely ceiling-saturated), so this verdict has little power to detect "
            "model-dependence in either direction. Test H4 on weaker/more varied model "
            "families where the task is not saturated."
        )

    # H3: axis disagreement
    best_retr = max(retrieval_summary, key=lambda c: retrieval_summary[c]["recall_at_k_mean"])
    best_reason = max(cond_mean, key=lambda c: (cond_mean[c] if cond_mean[c] == cond_mean[c] else -1))
    lines.append("\n### H3 — Do the axes agree?\n")
    lines.append(f"- best on **retrieval**: `{best_retr}`")
    lines.append(f"- best on **reasoning** (mean over models): `{best_reason}`")
    lines.append(
        f"\n**{'Axes DISAGREE' if best_retr != best_reason else 'Axes agree'}** — "
        + (
            "the retrieval winner is not the reasoning winner, so 'best format' "
            "depends on which axis you optimize."
            if best_retr != best_reason
            else "the same condition leads on both axes here."
        )
    )
    # H3 rests on the reasoning winner being a real winner; if its CI overlaps the
    # retrieval winner's, the disagreement is not established.
    if best_retr != best_reason and best_retr in ci and best_reason in ci:
        a, b = ci[best_reason], ci[best_retr]
        if not (a[2] < b[1] or b[2] < a[1]):
            lines.append(
                f"\n⚠️ **Not established.** `{best_reason}` ({a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]) "
                f"and `{best_retr}` ({b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}]) have overlapping "
                "reasoning CIs, so the axis disagreement rests on a point-estimate gap that "
                "the data cannot resolve."
            )

    # Pareto (reasoning accuracy vs token cost)
    pts = [(c, cond_mean[c], cost.get(c, 0.0)) for c in CONDITIONS if cond_mean[c] == cond_mean[c]]
    front = metrics.pareto_frontier(pts)
    lines.append("\n### Pareto frontier (reasoning accuracy vs token cost)\n")
    for c, a, k in sorted(pts, key=lambda t: -t[1]):
        lines.append(f"- {c}: acc={a:.2f}, tokens/rec={k:.0f}{' ⬅ frontier' if c in front else ''}")

    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 6 — format vs. content.")
    parser.add_argument("--k", type=int, default=10, help="retrieval depth")
    parser.add_argument("--answer-k", type=int, default=5, help="candidates shown to the model")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=150,
        help="reasoning query sample size, stratified across gold datasets. At 12 "
        "(the old default) each cell moved in 0.08 steps and no difference on this "
        "axis was resolvable; 150 spans ~46 clusters. Costs "
        "max_queries x 6 conditions x n_models completions — lower it for a smoke run.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--models",
        default="",
        help="comma-separated model ids (default: all configured via available_models). "
        "For three OpenAI tiers: 'gpt-5.4-nano,gpt-4.1-mini,gpt-4o-mini'.",
    )
    parser.add_argument("--answer-max-tokens", type=int, default=512)
    parser.add_argument("--no-answer", action="store_true", help="retrieval + cost only")
    parser.add_argument("--no-verbose", action="store_true")
    args = parser.parse_args()
    verbose = not args.no_verbose

    run_dir = new_run_dir(RESULTS_DIR)
    log_path = run_dir / f"run-{run_dir.name}.log"
    open_run_log(log_path)
    log(f"run dir: {run_dir} (also linked as results/latest)")

    corpus = load_corpus()
    queries = load_queries()
    corpus_by_id = {r["meta"]["concept-id"]: r for r in corpus}

    coverage = check_relevance_coverage(corpus, queries)
    log(
        f"coverage: {coverage.n_ok}/{coverage.n_queries} in-corpus; "
        f"{len(coverage.missing_relevant)} dangling, {len(coverage.empty_relevant)} empty"
    )
    scored = [q for q in queries if q.relevant and any(c in corpus_by_id for c in q.relevant)]
    clusters = {q.id: q.relevant[0] for q in scored}
    sampled = sample_queries(scored, args.max_queries, args.seed)
    log(f"corpus={len(corpus)} queries={len(queries)} scored={len(scored)} sample={len(sampled)}")

    # --- Axis 1: retrieval (per condition) + cost ---
    all_retrieval = []
    retrieved: dict[tuple[str, str], list[str]] = {}
    for cond, fn in CONDITIONS.items():
        results = evaluate_retrieval(corpus, scored, fn, cond, k=args.k)
        all_retrieval += results
        for r in results:
            retrieved[(cond, r.query_id)] = r.top_ids
        log(f"retrieval {cond}: scored {len(results)} queries")
    write_jsonl(all_retrieval, run_dir / "retrieval.jsonl")
    retrieval_summary = summarize(all_retrieval, clusters=clusters)
    cost = {cond: tokens_per_record(corpus, fn) for cond, fn in CONDITIONS.items()}

    # --- Axis 2: decoupled reasoning ---
    answer_rows: list[dict] = []
    if args.no_answer:
        log("reasoning stage disabled (--no-answer)")
    else:
        if args.models.strip():
            requested = [m.strip() for m in args.models.split(",") if m.strip()]
            configured = set(available_models(requested))
            models = [m for m in requested if m in configured]
            missing = [m for m in requested if m not in configured]
            if missing:
                log(f"requested but not configured (skipped): {missing}")
        else:
            models = available_models()
        if not models:
            log("reasoning skipped: no models configured (need OPENAI_API_KEY and/or Ollama)")
        else:
            if len(models) < 3:
                log(f"WARNING: only {len(models)} model(s); design wants ≥3: {models}")
            log(f"models: {models}")
            shared = {
                q.id: build_shared_candidates(
                    q, retrieved.get((REFERENCE_CONDITION, q.id), []), corpus_by_id,
                    answer_k=args.answer_k, seed=args.seed,
                )
                for q in sampled
            }
            log(f"decoupled: {sum(1 for v in shared.values() if v)}/{len(sampled)} queries have gold; ref={REFERENCE_CONDITION}")
            answer_rows = run_reasoning_stage(
                corpus_by_id, shared, sampled, models,
                max_tokens=args.answer_max_tokens, verbose=verbose,
            )
            write_jsonl(answer_rows, run_dir / "answers.jsonl")

    write_summary(retrieval_summary, cost, answer_rows, coverage, run_dir / "summary.md")
    log(f"wrote {run_dir / 'summary.md'}")


if __name__ == "__main__":
    try:
        main()
    finally:
        close_run_log()
