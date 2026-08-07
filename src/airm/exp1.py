"""Experiment 1 — which metadata format costs the fewest tokens?

Renders every corpus record in all six formats from one canonical facet payload,
counts tokens three ways, and reports the result both absolutely and as a
**paired** comparison against JSON.

Why paired: every format sees the identical record, so the per-record ratio
``tokens(fmt) / tokens(json)`` removes between-record variance entirely. Records
differ enormously in size -- a one-line abstract versus a thousand-word one -- and
an unpaired comparison of means drowns a real 20% format effect in that spread.
The paired confidence interval is the honest one; the unpaired means are reported
too because they are what a reader expects to see.

A ``json_umm`` **reference** measures the full raw UMM record CMR serves today.
It is deliberately *outside* the six-format comparison -- it carries far more
content, so comparing it to the others would be exactly the content/format
confound this study exists to remove. It is reported because "what does the
status quo cost?" is the question that motivates the whole exercise.

No LLM and no API key are required.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from . import cmr, format_cache, tokens
from .config import BASELINE_FORMAT, FORMATS, FORMAT_LABELS, RUNS_DIR, ensure_dirs, new_run_id
from .corpus import topic_of
from .facets import facets

#: Resamples for the bootstrap confidence intervals.
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20260731

#: The raw CMR record, measured for context but excluded from the comparison.
REFERENCE_KEY = "json_umm"

MEASURES = ("tiktoken", "hf_tokens", "chars", "bytes")


# --------------------------------------------------------------------------- #
# Statistics.
# --------------------------------------------------------------------------- #


def bootstrap_ci(
    values: list[float],
    *,
    n: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean. Seeded, so runs are reproducible."""
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    k = len(values)
    means = []
    for _ in range(n):
        means.append(sum(values[rng.randrange(k)] for _ in range(k)) / k)
    means.sort()
    lo = means[int((alpha / 2) * n)]
    hi = means[min(n - 1, int((1 - alpha / 2) * n))]
    return (lo, hi)


@dataclass
class FormatSummary:
    fmt: str
    n: int
    mean: float
    median: float
    ci_low: float
    ci_high: float
    #: Paired per-record ratio to the baseline format.
    ratio_mean: float
    ratio_ci_low: float
    ratio_ci_high: float

    @property
    def pct_vs_baseline(self) -> float:
        return (self.ratio_mean - 1.0) * 100.0

    def to_dict(self) -> dict:
        return {
            "format": self.fmt,
            "n": self.n,
            "mean_tokens": round(self.mean, 2),
            "median_tokens": self.median,
            "ci_low": round(self.ci_low, 2),
            "ci_high": round(self.ci_high, 2),
            "ratio_vs_json": round(self.ratio_mean, 4),
            "ratio_ci_low": round(self.ratio_ci_low, 4),
            "ratio_ci_high": round(self.ratio_ci_high, 4),
            "pct_vs_json": round(self.pct_vs_baseline, 2),
        }


def summarise(
    rows: list[dict], measure: str = "tiktoken", baseline: str = BASELINE_FORMAT
) -> list[FormatSummary]:
    """Per-format summary of ``measure``, with paired ratios against ``baseline``."""
    by_record: dict[str, dict[str, float]] = {}
    for row in rows:
        value = row.get(measure)
        if value is None:
            continue
        by_record.setdefault(row["concept_id"], {})[row["format"]] = float(value)

    out: list[FormatSummary] = []
    for fmt in FORMATS:
        values = [rec[fmt] for rec in by_record.values() if fmt in rec]
        ratios = [
            rec[fmt] / rec[baseline]
            for rec in by_record.values()
            if fmt in rec and rec.get(baseline)
        ]
        if not values:
            continue
        ci = bootstrap_ci(values)
        rci = bootstrap_ci(ratios) if ratios else (float("nan"), float("nan"))
        out.append(
            FormatSummary(
                fmt=fmt,
                n=len(values),
                mean=statistics.fmean(values),
                median=statistics.median(values),
                ci_low=ci[0],
                ci_high=ci[1],
                ratio_mean=statistics.fmean(ratios) if ratios else float("nan"),
                ratio_ci_low=rci[0],
                ratio_ci_high=rci[1],
            )
        )
    return out


def per_topic_means(rows: list[dict], measure: str = "tiktoken") -> dict[str, dict[str, float]]:
    """``{topic: {format: mean tokens}}`` -- nesting depth varies by domain."""
    buckets: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        value = row.get(measure)
        if value is None:
            continue
        buckets.setdefault(row["topic"], {}).setdefault(row["format"], []).append(float(value))
    return {
        topic: {fmt: statistics.fmean(v) for fmt, v in sorted(per_fmt.items())}
        for topic, per_fmt in sorted(buckets.items())
    }


# --------------------------------------------------------------------------- #
# Measurement.
# --------------------------------------------------------------------------- #


def measure_records(records: list[dict], *, hf: bool = True) -> list[dict]:
    """One row per ``(record, format)``, plus the raw-UMM reference row."""
    rows: list[dict] = []
    for record in records:
        cid = cmr.concept_id(record)
        topic = topic_of(record)
        payload = facets(record)

        for fmt, text in format_cache.render_all_cached(cid, payload).items():
            rows.append(
                {"concept_id": cid, "topic": topic, "format": fmt, **tokens.measure(text, hf=hf)}
            )

        raw = json.dumps(record.get("umm", record), ensure_ascii=False, separators=(",", ":"))
        rows.append(
            {
                "concept_id": cid,
                "topic": topic,
                "format": REFERENCE_KEY,
                **tokens.measure(raw, hf=hf),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Output.
# --------------------------------------------------------------------------- #


def write_rows(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["concept_id", "topic", "format", *MEASURES]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


#: Validated with the data-viz palette validator against the light surface:
#: worst adjacent CVD deltaE 24.7 (protan), normal-vision 33.6, both above the
#: gates; contrast >= 3:1 on the surface.
SERIES_TIKTOKEN = "#2a78d6"
SERIES_HF = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"


def plot(summaries: dict[str, list[FormatSummary]], path: Path) -> Path | None:
    """Grouped horizontal bars: mean tokens per format, one bar per tokenizer."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = summaries["tiktoken"]
    if not primary:
        return None
    order = [s.fmt for s in sorted(primary, key=lambda s: s.mean)]
    series = [("tiktoken o200k_base", "tiktoken", SERIES_TIKTOKEN)]
    if summaries.get("hf_tokens"):
        series.append(("Qwen3 tokenizer", "hf_tokens", SERIES_HF))

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    widest = max(
        s.ci_high for key in (k for _, k, _ in series) for s in summaries[key]
    )
    x_max = widest * 1.16
    label_gap = widest * 0.015

    height = 0.36 if len(series) > 1 else 0.55
    for i, (label, key, color) in enumerate(series):
        lookup = {s.fmt: s for s in summaries[key]}
        offset = (i - (len(series) - 1) / 2) * (height + 0.04)
        ys = [j + offset for j in range(len(order))]
        means = [lookup[f].mean for f in order]
        errs = [
            [lookup[f].mean - lookup[f].ci_low for f in order],
            [lookup[f].ci_high - lookup[f].mean for f in order],
        ]
        ax.barh(ys, means, height=height, color=color, label=label, zorder=3)
        ax.errorbar(
            means, ys, xerr=errs, fmt="none", ecolor=INK_MUTED, elinewidth=1.2, capsize=3, zorder=4
        )
        # Label past the CI whisker, not past the bar, or the two collide.
        for y, fmt_name, mean in zip(ys, order, means):
            ax.text(
                lookup[fmt_name].ci_high + label_gap,
                y,
                f"{mean:,.0f}",
                va="center",
                ha="left",
                fontsize=8,
                color=INK_MUTED,
            )

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([FORMAT_LABELS.get(f, f) for f in order], fontsize=10, color=INK)
    ax.set_xlabel("mean tokens per record  (95% bootstrap CI)", fontsize=9, color=INK_MUTED)
    ax.set_title(
        "Token cost of six metadata formats, content held constant",
        fontsize=12,
        color=INK,
        loc="left",
        pad=30,
    )
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d8d7d2")
    ax.grid(axis="x", color="#e7e6e2", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, x_max)
    if len(series) > 1:
        # Above the plot: inside it, the legend lands on the bottom bars.
        ax.legend(
            frameon=False,
            fontsize=9,
            labelcolor=INK_MUTED,
            loc="lower left",
            bbox_to_anchor=(0, 1.005),
            ncol=len(series),
            handlelength=1.1,
            columnspacing=1.6,
        )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def format_table(summaries: list[FormatSummary], reference: FormatSummary | None) -> str:
    lines = [
        f"{'format':<18}{'mean':>9}{'median':>9}{'95% CI':>19}{'vs JSON':>10}{'paired CI':>18}",
        "-" * 83,
    ]
    for s in sorted(summaries, key=lambda x: x.mean):
        ci = f"[{s.ci_low:,.0f}, {s.ci_high:,.0f}]"
        rci = f"[{(s.ratio_ci_low - 1) * 100:+.1f}%, {(s.ratio_ci_high - 1) * 100:+.1f}%]"
        lines.append(
            f"{FORMAT_LABELS.get(s.fmt, s.fmt):<18}{s.mean:>9,.0f}{s.median:>9,.0f}"
            f"{ci:>19}{s.pct_vs_baseline:>9.1f}%{rci:>18}"
        )
    if reference:
        lines.append("-" * 83)
        lines.append(
            f"{'raw UMM (ref)':<18}{reference.mean:>9,.0f}{reference.median:>9,.0f}"
            f"{'':>19}{'':>10}{'  not comparable: more content':>18}"
        )
    return "\n".join(lines)


def run(*, hf: bool = True, run_id: str | None = None, out_dir: Path | None = None) -> dict:
    """Run Experiment 1 end to end and write its artefacts."""
    ensure_dirs()
    run_id = run_id or new_run_id()
    out_dir = out_dir or (RUNS_DIR / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = cmr.load_corpus()
    rows = measure_records(records, hf=hf)
    write_rows(rows, out_dir / "exp1_tokens.csv")

    comparable = [r for r in rows if r["format"] != REFERENCE_KEY]
    summaries = {m: summarise(comparable, measure=m) for m in ("tiktoken", "hf_tokens")}
    reference = None
    ref_rows = [r for r in rows if r["format"] == REFERENCE_KEY and r.get("tiktoken") is not None]
    if ref_rows:
        values = [float(r["tiktoken"]) for r in ref_rows]
        ci = bootstrap_ci(values)
        reference = FormatSummary(
            REFERENCE_KEY,
            len(values),
            statistics.fmean(values),
            statistics.median(values),
            ci[0],
            ci[1],
            float("nan"),
            float("nan"),
            float("nan"),
        )

    hf_ok, hf_detail = (True, "") if hf else (False, "disabled via --no-hf")
    if hf:
        hf_ok, hf_detail = tokens.hf_available()

    result = {
        "run_id": run_id,
        "records": len(records),
        "rows": len(rows),
        "baseline": BASELINE_FORMAT,
        "tokenizers": {
            "tiktoken": "o200k_base",
            "hf": hf_detail if hf_ok else None,
            "hf_unavailable_reason": None if hf_ok else hf_detail,
        },
        "summary_tiktoken": [s.to_dict() for s in summaries["tiktoken"]],
        "summary_hf": [s.to_dict() for s in summaries["hf_tokens"]],
        "reference_raw_umm": reference.to_dict() if reference else None,
        "per_topic_tiktoken": per_topic_means(comparable),
    }
    (out_dir / "exp1_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(summaries, out_dir / "exp1_tokens.png")
    return result


if __name__ == "__main__":  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 1: format token economy.")
    parser.add_argument("--no-hf", action="store_true", help="skip the Hugging Face tokenizer")
    args = parser.parse_args()

    res = run(hf=not args.no_hf)
    out = RUNS_DIR / res["run_id"]

    print(f"records measured : {res['records']}  ({res['rows']} rows)")
    print(f"tokenizers       : tiktoken o200k_base"
          f"{', ' + res['tokenizers']['hf'] if res['tokenizers']['hf'] else ' (HF unavailable)'}")
    if res["tokenizers"]["hf_unavailable_reason"]:
        print(f"  HF column absent: {res['tokenizers']['hf_unavailable_reason'][:110]}")
    print()

    summaries = [FormatSummary(**{
        "fmt": d["format"], "n": d["n"], "mean": d["mean_tokens"], "median": d["median_tokens"],
        "ci_low": d["ci_low"], "ci_high": d["ci_high"], "ratio_mean": d["ratio_vs_json"],
        "ratio_ci_low": d["ratio_ci_low"], "ratio_ci_high": d["ratio_ci_high"],
    }) for d in res["summary_tiktoken"]]
    ref = res["reference_raw_umm"]
    ref_summary = (
        FormatSummary(REFERENCE_KEY, ref["n"], ref["mean_tokens"], ref["median_tokens"],
                      ref["ci_low"], ref["ci_high"], float("nan"), float("nan"), float("nan"))
        if ref else None
    )
    print(format_table(summaries, ref_summary))
    print(f"\nwrote {out}/exp1_tokens.csv")
    print(f"wrote {out}/exp1_summary.json")
    print(f"wrote {out}/exp1_tokens.png")
