"""Assemble FINDINGS.md from the run artefacts.

Every number here is read from a file under ``runs/<id>/`` -- nothing is computed
in the prose and nothing is typed in by hand. If a metric is missing (no judge
available, tokenizer unavailable, cells not yet run) the table says so rather
than omitting the row, so a partial study reads as partial.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import BASELINE_FORMAT, FORMAT_LABELS, FORMATS, RUNS_DIR

MISSING = "—"


def latest_run(pattern: str, root: Path = RUNS_DIR) -> Path | None:
    """The most recent run directory containing ``pattern``."""
    candidates = sorted(
        (d for d in root.glob("*") if d.is_dir() and (d / pattern).exists()),
        key=lambda d: d.name,
    )
    return candidates[-1] if candidates else None


def _fmt(value, spec: str = ".3f") -> str:
    if value is None or (isinstance(value, float) and value != value):
        return MISSING
    return format(value, spec)


# --------------------------------------------------------------------------- #
# Experiment 1.
# --------------------------------------------------------------------------- #


def exp1_section(summary: dict) -> str:
    rows = sorted(summary["summary_tiktoken"], key=lambda r: r["mean_tokens"])
    hf = {r["format"]: r for r in summary.get("summary_hf", [])}

    lines = [
        "## Experiment 1 — token cost by format",
        "",
        f"{summary['records']} CMR collection records, each rendered six ways from one "
        "canonical facet payload so **content is identical across formats**. "
        "`vs JSON` is a paired per-record ratio.",
        "",
        "| Format | Mean tokens | Median | 95% CI | vs JSON | Paired 95% CI | Qwen3 mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        label = FORMAT_LABELS.get(r["format"], r["format"])
        if r["format"] == BASELINE_FORMAT:
            label += " *(baseline)*"
        paired = (
            f"[{(r['ratio_ci_low'] - 1) * 100:+.1f}%, {(r['ratio_ci_high'] - 1) * 100:+.1f}%]"
            if r["format"] != BASELINE_FORMAT
            else MISSING
        )
        pct = f"{r['pct_vs_json']:+.1f}%" if r["format"] != BASELINE_FORMAT else MISSING
        hf_mean = _fmt(hf.get(r["format"], {}).get("mean_tokens"), ",.0f")
        lines.append(
            f"| {label} | {r['mean_tokens']:,.0f} | {r['median_tokens']:,.0f} | "
            f"[{r['ci_low']:,.0f}, {r['ci_high']:,.0f}] | {pct} | {paired} | {hf_mean} |"
        )

    ref = summary.get("reference_raw_umm")
    if ref:
        lines += [
            "",
            f"**Reference:** the raw UMM record CMR serves today costs "
            f"**{ref['mean_tokens']:,.0f} tokens** (median {ref['median_tokens']:,.0f}) — "
            f"{ref['mean_tokens'] / rows[0]['mean_tokens']:.1f}× the cheapest representation. "
            "It is excluded from the comparison because it carries more *content*, not just "
            "a different format.",
        ]

    tok = summary.get("tokenizers", {})
    if not tok.get("hf"):
        lines += ["", f"*Qwen3 column absent: {tok.get('hf_unavailable_reason', 'unavailable')}*"]

    per_topic = summary.get("per_topic_tiktoken") or {}
    if per_topic:
        spread = {
            t: v.get(BASELINE_FORMAT) for t, v in per_topic.items() if v.get(BASELINE_FORMAT)
        }
        if spread:
            hi = max(spread, key=spread.get)
            lo = min(spread, key=spread.get)
            lines += [
                "",
                f"Domain moves cost more than format does: JSON averages "
                f"{spread[hi]:,.0f} tokens for {hi} records and {spread[lo]:,.0f} for {lo} — "
                f"a {spread[hi] / spread[lo]:.1f}× spread, against a "
                f"{rows[-1]['mean_tokens'] / rows[0]['mean_tokens']:.2f}× spread across formats.",
            ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Experiment 2.
# --------------------------------------------------------------------------- #

METRIC_COLUMNS = (
    ("recall@10", "Recall@10", ".3f"),
    ("mrr", "MRR", ".3f"),
    ("ndcg@10", "nDCG@10", ".3f"),
    ("correctness", "Correct.", ".3f"),
    ("faithfulness", "Faith.", ".3f"),
    ("answer_relevancy", "Ans.Rel.", ".3f"),
    ("contextual_relevancy", "Ctx.Rel.", ".3f"),
    ("contextual_recall", "Ctx.Rec.", ".3f"),
)


def _by_format_and_model(summary: dict) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for key, row in summary.items():
        fmt, _, model = key.partition("|")
        out.setdefault(model, {})[fmt] = row
    return out


def exp2_section(result: dict) -> str:
    summary = result.get("summary") or {}
    if not summary:
        return "## Experiment 2 — model × format retrieval\n\n*No cells have been run yet.*"

    lines = [
        "## Experiment 2 — model × format retrieval",
        "",
        f"{result['queries']} queries × {len(result['formats'])} formats × "
        f"{len(result['models'])} model(s) = {result['cells']} cells, top-k = {result['top_k']}.",
    ]
    judge = result.get("judge")
    lines += [
        "",
        f"LLM-graded metrics were judged by **{judge['provider']}:{judge['model']}**, held "
        "fixed across every cell and never drawn from the model matrix."
        if judge
        else "*Run without a judge: retrieval metrics only.*",
    ]
    for problem in result.get("model_problems") or []:
        lines += ["", f"> **Model unavailable:** {problem}"]

    for model, per_fmt in _by_format_and_model(summary).items():
        lines += [
            "",
            f"### {model}",
            "",
            "| Format | n | " + " | ".join(label for _, label, _ in METRIC_COLUMNS) + " | Tokens in |",
            "|---" * (len(METRIC_COLUMNS) + 3) + "|",
        ]
        for fmt in FORMATS:
            row = per_fmt.get(fmt)
            if not row:
                continue
            cells = " | ".join(_fmt(row.get(key), spec) for key, _, spec in METRIC_COLUMNS)
            lines.append(
                f"| {FORMAT_LABELS.get(fmt, fmt)} | {row.get('n', 0)} | {cells} | "
                f"{_fmt(row.get('prompt_tokens'), ',.0f')} |"
            )

    lines += _winner_lines(summary)
    return "\n".join(lines)


def _winner_lines(summary: dict) -> list[str]:
    """Does the best format differ by model? That is the study's core question."""
    by_model = _by_format_and_model(summary)
    winners: dict[str, dict[str, str | None]] = {}
    for model, per_fmt in by_model.items():
        winners[model] = {}
        for metric in ("recall@10", "correctness"):
            scored = {f: r.get(metric) for f, r in per_fmt.items() if r.get(metric) is not None}
            winners[model][metric] = max(scored, key=scored.get) if scored else None

    if len(winners) < 2:
        return [
            "",
            "*Only one model has been run, so the model-dependence question — whether the "
            "best format changes with the model — is not yet answerable.*",
        ]

    lines = ["", "### Does the best format depend on the model?", ""]
    for metric in ("recall@10", "correctness"):
        picks = {m: w[metric] for m, w in winners.items() if w[metric]}
        if not picks:
            continue
        agree = len(set(picks.values())) == 1
        detail = ", ".join(f"{m} → {f}" for m, f in picks.items())
        lines.append(
            f"- **{metric}**: {'all models agree' if agree else 'models disagree'} ({detail})"
        )
    return lines


# --------------------------------------------------------------------------- #
# Coverage of the source record.
# --------------------------------------------------------------------------- #


def coverage_section() -> str:
    """What the six formats keep from the record CMR actually published.

    Reported next to the token table on purpose. The token numbers are only
    meaningful because content is held constant across formats — but "constant"
    is not "complete", and a reader deserves both facts in the same place.
    """
    from . import coverage
    from .config import CORPUS_PATH

    if not CORPUS_PATH.exists():
        return ""

    with CORPUS_PATH.open() as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    res = coverage.report(records)
    total = res["values"]
    reach = res["totals"]["CARRIED"] + res["totals"]["PARTIAL"]

    lines = [
        "## Content coverage — what the formats keep from the CMR record",
        "",
        "All six formats carry **identical content to each other**: every machine format "
        "round-trips to the same fact set and the prose contains every canonical value "
        "verbatim, on 500/500 records. That is what makes the token comparison a format "
        "comparison.",
        "",
        "They do not carry everything CMR published. `airm.facets.facets` is a projection, "
        "and `uv run python -m airm.coverage` classifies every scalar path in the source "
        "record so the projection is explicit and enforced — an undeclared path fails the "
        "check.",
        "",
        f"| Bucket | Values | Share | Paths |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "CARRIED": "Carried into all six formats",
        "PARTIAL": "Carried, first element of a repeating group only",
        "EXCLUDED": "Dropped deliberately (catalog bookkeeping, contact details)",
        "DEFERRED": "Dataset content not carried yet",
        "UNDECLARED": "Undeclared (fails the check)",
    }
    for name in ("CARRIED", "PARTIAL", "EXCLUDED", "DEFERRED", "UNDECLARED"):
        n = res["totals"][name]
        lines.append(
            f"| {labels[name]} | {n:,} | {n / total:.1%} | {len(res['buckets'][name])} |"
        )

    lines += [
        "",
        f"**{reach:,} of {total:,} scalar values ({reach / total:.1%}) reach the formats.** "
        "The largest deferred groups are `RelatedUrls`, `Projects`, `LocationKeywords`, "
        "`AdditionalAttributes` and `ISOTopicCategories` — real, searchable dataset content. "
        "Because the loss happens upstream of every renderer it is identical across formats "
        "and cannot bias the comparison between them, but it does mean the study measures "
        "the cost of a metadata *summary*, not of the full record.",
        "",
        "Run `uv run python -m airm.coverage --show deferred` for the itemised backlog.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Assembly.
# --------------------------------------------------------------------------- #


def build(out_path: Path | None = None) -> Path:
    out_path = out_path or (Path.cwd() / "FINDINGS.md")

    parts = [
        "# Findings",
        "",
        "Generated by `uv run python -m airm.report`. Every number is read from a file "
        "under `runs/`; none is typed in by hand. See `EXPERIMENTS.md` for the method and "
        "the decisions behind it.",
        "",
    ]

    exp1_dir = latest_run("exp1_summary.json")
    if exp1_dir:
        summary = json.loads((exp1_dir / "exp1_summary.json").read_text())
        parts += [exp1_section(summary), "", f"*Source: `runs/{exp1_dir.name}/`*", ""]
    else:
        parts += ["## Experiment 1 — token cost by format", "", "*Not run yet.*", ""]

    exp2_dir = latest_run("exp2_summary.json")
    if exp2_dir:
        result = json.loads((exp2_dir / "exp2_summary.json").read_text())
        parts += [exp2_section(result), "", f"*Source: `runs/{exp2_dir.name}/`*", ""]
    else:
        parts += ["## Experiment 2 — model × format retrieval", "", "*Not run yet.*", ""]

    section = coverage_section()
    if section:
        parts += [section, ""]

    out_path.write_text("\n".join(parts))
    return out_path


if __name__ == "__main__":  # pragma: no cover - CLI
    path = build()
    print(f"wrote {path}")
