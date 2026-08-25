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

Two payloads, six formats
-------------------------
Every record is measured twice over:

``faceted``
    The 32-field canonical projection (:mod:`airm.facets`) -- the payload both
    experiments actually run on.
``unfaceted``
    The whole raw UMM record CMR serves today (:mod:`airm.unfaceted`) -- the
    status quo, and the control that answers "does the format ranking survive
    without the projection?".

That makes the design 2 payloads x 6 formats, and the ``payload`` column is a
**factor, not a format**. Ratios are always taken against the JSON rendering of
*the same payload*: comparing unfaceted YAML to faceted JSON would mix a content
difference into a format difference, which is the exact confound this study
exists to remove. The two payloads are compared only through
:func:`faceting_savings`, which is explicitly a content comparison and labelled
as one.

The unfaceted JSON rendering is byte-identical to the old ``json_umm``
reference row, so that row is gone: it was measuring the same string twice.
``reference_raw_umm`` survives in the summary JSON, sourced from the unfaceted
JSON rows, because "what does the status quo cost?" is the question that
motivates the whole exercise.

No LLM and no API key are required.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from . import cmr, format_cache, tokens, unfaceted
from .config import BASELINE_FORMAT, FORMATS, FORMAT_LABELS, RUNS_DIR, ensure_dirs, new_run_id
from .provenance import provenance
from .corpus import topic_of
from .facets import facets

#: Resamples for the bootstrap confidence intervals.
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20260731

#: The two payloads under measurement. ``faceted`` leads because it is what the
#: rest of the study runs on; ``unfaceted`` is the control.
PAYLOADS: tuple[str, ...] = ("faceted", "unfaceted")

FACETED, UNFACETED = PAYLOADS

#: Retained for the summary JSON and the report: the raw UMM record CMR serves.
#: It is now sourced from the ``(unfaceted, json)`` rows rather than measured
#: separately, because those are the identical string.
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
    rows: list[dict],
    measure: str = "tiktoken",
    baseline: str = BASELINE_FORMAT,
    *,
    payload: str | None = None,
) -> list[FormatSummary]:
    """Per-format summary of ``measure``, with paired ratios against ``baseline``.

    ``payload`` restricts the rows to one payload. Leaving it ``None`` on a
    two-payload row set would average faceted and unfaceted renderings of the
    same format together and take the ratio against a baseline that is itself
    such an average -- a number with no interpretation. Callers holding mixed
    rows must pass it.
    """
    if payload is not None:
        rows = [r for r in rows if r.get("payload") == payload]

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


def per_topic_means(
    rows: list[dict], measure: str = "tiktoken", *, payload: str | None = None
) -> dict[str, dict[str, float]]:
    """``{topic: {format: mean tokens}}`` -- nesting depth varies by domain."""
    if payload is not None:
        rows = [r for r in rows if r.get("payload") == payload]
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
    """One row per ``(record, payload, format)``.

    Twelve rows per record: two payloads by six formats. Both go through their
    module's cache-first accessor, so the measured bytes are exactly the bytes on
    disk when a cache is current and identical live renderings when it is not.
    """
    rows: list[dict] = []
    for record in records:
        cid = cmr.concept_id(record)
        topic = topic_of(record)

        rendered = {
            FACETED: format_cache.render_all_cached(cid, facets(record)),
            UNFACETED: unfaceted.render_all_cached(cid, unfaceted.payload(record)),
        }
        for payload_name, by_format in rendered.items():
            for fmt, text in by_format.items():
                rows.append(
                    {
                        "concept_id": cid,
                        "topic": topic,
                        "payload": payload_name,
                        "format": fmt,
                        **tokens.measure(text, hf=hf),
                    }
                )
    return rows


@dataclass
class FacetingSaving:
    """What the projection costs or saves, per format. A *content* comparison."""

    fmt: str
    n: int
    faceted_mean: float
    unfaceted_mean: float
    #: Paired per-record ratio faceted/unfaceted, so between-record variance
    #: drops out exactly as it does for the format ratios.
    ratio_mean: float
    ratio_ci_low: float
    ratio_ci_high: float

    @property
    def pct_saved(self) -> float:
        """Percent of unfaceted tokens the projection removes."""
        return (1.0 - self.ratio_mean) * 100.0

    def to_dict(self) -> dict:
        return {
            "format": self.fmt,
            "n": self.n,
            "faceted_mean": round(self.faceted_mean, 2),
            "unfaceted_mean": round(self.unfaceted_mean, 2),
            "ratio_faceted_over_unfaceted": round(self.ratio_mean, 4),
            "ratio_ci_low": round(self.ratio_ci_low, 4),
            "ratio_ci_high": round(self.ratio_ci_high, 4),
            "pct_saved_by_faceting": round(self.pct_saved, 2),
        }


def faceting_savings(rows: list[dict], measure: str = "tiktoken") -> list[FacetingSaving]:
    """Per-format paired comparison of the two payloads.

    This is the one place the payloads are compared to each other, and it is
    **not** a format comparison: the two carry different content by construction,
    so the number answers "how much does the projection remove?" and nothing
    about which format is better.
    """
    paired: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        value = row.get(measure)
        if value is None:
            continue
        paired.setdefault(row["format"], {}).setdefault(row["concept_id"], {})[
            row["payload"]
        ] = float(value)

    out: list[FacetingSaving] = []
    for fmt in FORMATS:
        per_record = paired.get(fmt, {})
        both = [
            v for v in per_record.values() if v.get(FACETED) and v.get(UNFACETED)
        ]
        if not both:
            continue
        ratios = [v[FACETED] / v[UNFACETED] for v in both]
        ci = bootstrap_ci(ratios)
        out.append(
            FacetingSaving(
                fmt=fmt,
                n=len(both),
                faceted_mean=statistics.fmean(v[FACETED] for v in both),
                unfaceted_mean=statistics.fmean(v[UNFACETED] for v in both),
                ratio_mean=statistics.fmean(ratios),
                ratio_ci_low=ci[0],
                ratio_ci_high=ci[1],
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Output.
# --------------------------------------------------------------------------- #


def write_rows(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["concept_id", "topic", "payload", "format", *MEASURES]
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


def plot(panels: dict[str, dict[str, list[FormatSummary]]], path: Path) -> Path | None:
    """Two panels — faceted and unfaceted — each sorted by its own mean.

    Sorting each panel independently is the whole point: the rank order differs
    between payloads, and a shared order would hide the finding. The x-axis is
    shared so the magnitudes stay comparable by eye.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not panels.get(FACETED, {}).get("tiktoken"):
        return None

    series_keys = [("tiktoken o200k_base", "tiktoken", SERIES_TIKTOKEN)]
    if panels[FACETED].get("hf_tokens"):
        series_keys.append(("Qwen3 tokenizer", "hf_tokens", SERIES_HF))

    widest = max(
        s.ci_high
        for panel in panels.values()
        for _, key, _ in series_keys
        for s in panel.get(key, [])
    )
    x_max = widest * 1.18
    label_gap = widest * 0.012

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=200, sharex=True)
    fig.patch.set_facecolor(SURFACE)

    titles = {
        FACETED: "Faceted — 32-field projection",
        UNFACETED: "Unfaceted — full raw UMM record",
    }
    height = 0.36 if len(series_keys) > 1 else 0.55

    for ax, payload_name in zip(axes, PAYLOADS):
        panel = panels[payload_name]
        ax.set_facecolor(SURFACE)
        order = [s.fmt for s in sorted(panel["tiktoken"], key=lambda s: s.mean)]

        for i, (label, key, color) in enumerate(series_keys):
            lookup = {s.fmt: s for s in panel.get(key, [])}
            if not lookup:
                continue
            offset = (i - (len(series_keys) - 1) / 2) * (height + 0.04)
            ys = [j + offset for j in range(len(order))]
            means = [lookup[f].mean for f in order]
            errs = [
                [lookup[f].mean - lookup[f].ci_low for f in order],
                [lookup[f].ci_high - lookup[f].mean for f in order],
            ]
            ax.barh(ys, means, height=height, color=color, label=label, zorder=3)
            ax.errorbar(
                means, ys, xerr=errs, fmt="none", ecolor=INK_MUTED,
                elinewidth=1.2, capsize=3, zorder=4,
            )
            for y, fmt_name, mean in zip(ys, order, means):
                ax.text(
                    lookup[fmt_name].ci_high + label_gap, y, f"{mean:,.0f}",
                    va="center", ha="left", fontsize=8, color=INK_MUTED,
                )

        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([FORMAT_LABELS.get(f, f) for f in order], fontsize=9.5, color=INK)
        ax.set_title(titles[payload_name], fontsize=10, color=INK, loc="left", pad=8)
        ax.set_xlabel("mean tokens per record  (95% bootstrap CI)", fontsize=8.5, color=INK_MUTED)
        ax.tick_params(axis="x", colors=INK_MUTED, labelsize=8)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#d8d7d2")
        ax.grid(axis="x", color="#e7e6e2", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(0, x_max)

    fig.suptitle(
        "Token cost of six metadata formats, at two levels of content",
        fontsize=12.5, color=INK, x=0.007, ha="left", y=0.99,
    )
    if len(series_keys) > 1:
        axes[0].legend(
            frameon=False, fontsize=8.5, labelcolor=INK_MUTED, loc="lower left",
            bbox_to_anchor=(0, 1.12), ncol=len(series_keys),
            handlelength=1.1, columnspacing=1.6,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def format_table(summaries: list[FormatSummary], title: str = "") -> str:
    lines = []
    if title:
        lines.append(title)
    lines += [
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
    return "\n".join(lines)


def savings_table(savings: list[FacetingSaving]) -> str:
    lines = [
        f"{'format':<18}{'faceted':>10}{'unfaceted':>11}{'saved':>9}{'paired CI':>20}",
        "-" * 68,
    ]
    for s in sorted(savings, key=lambda x: -x.pct_saved):
        ci = f"[{(1 - s.ratio_ci_high) * 100:+.1f}%, {(1 - s.ratio_ci_low) * 100:+.1f}%]"
        lines.append(
            f"{FORMAT_LABELS.get(s.fmt, s.fmt):<18}{s.faceted_mean:>10,.0f}"
            f"{s.unfaceted_mean:>11,.0f}{s.pct_saved:>8.1f}%{ci:>20}"
        )
    return "\n".join(lines)


def _rank_order(summaries: list[FormatSummary]) -> list[str]:
    return [s.fmt for s in sorted(summaries, key=lambda s: s.mean)]


def run(*, hf: bool = True, run_id: str | None = None, out_dir: Path | None = None) -> dict:
    """Run Experiment 1 end to end and write its artefacts."""
    ensure_dirs()
    run_id = run_id or new_run_id()
    out_dir = out_dir or (RUNS_DIR / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = cmr.load_corpus()
    rows = measure_records(records, hf=hf)
    write_rows(rows, out_dir / "exp1_tokens.csv")

    panels = {
        name: {
            m: summarise(rows, measure=m, payload=name) for m in ("tiktoken", "hf_tokens")
        }
        for name in PAYLOADS
    }
    savings = faceting_savings(rows)

    # The raw-UMM reference is the unfaceted JSON rendering -- the same string
    # the retired json_umm row used to measure, not a second measurement.
    reference = None
    ref = [s for s in panels[UNFACETED]["tiktoken"] if s.fmt == BASELINE_FORMAT]
    if ref:
        r = ref[0]
        reference = FormatSummary(
            REFERENCE_KEY, r.n, r.mean, r.median, r.ci_low, r.ci_high,
            float("nan"), float("nan"), float("nan"),
        )

    hf_ok, hf_detail = (True, "") if hf else (False, "disabled via --no-hf")
    if hf:
        hf_ok, hf_detail = tokens.hf_available()

    result = {
        **provenance(run_id),
        "records": len(records),
        "rows": len(rows),
        "payloads": list(PAYLOADS),
        "baseline": BASELINE_FORMAT,
        "tokenizers": {
            "tiktoken": "o200k_base",
            "hf": hf_detail if hf_ok else None,
            "hf_unavailable_reason": None if hf_ok else hf_detail,
        },
        # Keys without a payload suffix are the faceted payload: it is what the
        # rest of the study runs on, and existing readers of this file expect it.
        "summary_tiktoken": [s.to_dict() for s in panels[FACETED]["tiktoken"]],
        "summary_hf": [s.to_dict() for s in panels[FACETED]["hf_tokens"]],
        "summary_tiktoken_unfaceted": [s.to_dict() for s in panels[UNFACETED]["tiktoken"]],
        "summary_hf_unfaceted": [s.to_dict() for s in panels[UNFACETED]["hf_tokens"]],
        "faceting_savings": [s.to_dict() for s in savings],
        "rank_order": {
            name: _rank_order(panels[name]["tiktoken"]) for name in PAYLOADS
        },
        "reference_raw_umm": reference.to_dict() if reference else None,
        "per_topic_tiktoken": per_topic_means(rows, payload=FACETED),
        "per_topic_tiktoken_unfaceted": per_topic_means(rows, payload=UNFACETED),
    }
    (out_dir / "exp1_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(panels, out_dir / "exp1_tokens.png")
    return result


if __name__ == "__main__":  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 1: format token economy.")
    parser.add_argument("--no-hf", action="store_true", help="skip the Hugging Face tokenizer")
    args = parser.parse_args()

    res = run(hf=not args.no_hf)
    out = RUNS_DIR / res["run_id"]

    print(f"records measured : {res['records']}  ({res['rows']} rows = "
          f"{res['records']} x {len(PAYLOADS)} payloads x {len(FORMATS)} formats)")
    print(f"tokenizers       : tiktoken o200k_base"
          f"{', ' + res['tokenizers']['hf'] if res['tokenizers']['hf'] else ' (HF unavailable)'}")
    if res["tokenizers"]["hf_unavailable_reason"]:
        print(f"  HF column absent: {res['tokenizers']['hf_unavailable_reason'][:110]}")
    print()

    def _revive(dicts: list[dict]) -> list[FormatSummary]:
        return [
            FormatSummary(
                fmt=d["format"], n=d["n"], mean=d["mean_tokens"], median=d["median_tokens"],
                ci_low=d["ci_low"], ci_high=d["ci_high"], ratio_mean=d["ratio_vs_json"],
                ratio_ci_low=d["ratio_ci_low"], ratio_ci_high=d["ratio_ci_high"],
            )
            for d in dicts
        ]

    print(format_table(_revive(res["summary_tiktoken"]),
                       "FACETED — 32-field projection (what the study runs on)"))
    print()
    print(format_table(_revive(res["summary_tiktoken_unfaceted"]),
                       "UNFACETED — full raw UMM record (the status quo)"))

    print("\nWhat the projection removes (a content comparison, not a format one):\n")
    print(savings_table([
        FacetingSaving(
            fmt=d["format"], n=d["n"], faceted_mean=d["faceted_mean"],
            unfaceted_mean=d["unfaceted_mean"],
            ratio_mean=d["ratio_faceted_over_unfaceted"],
            ratio_ci_low=d["ratio_ci_low"], ratio_ci_high=d["ratio_ci_high"],
        )
        for d in res["faceting_savings"]
    ]))

    faceted_order, unfaceted_order = res["rank_order"][FACETED], res["rank_order"][UNFACETED]
    print(f"\nrank order  faceted : {' < '.join(faceted_order)}")
    print(f"rank order  unfaceted: {' < '.join(unfaceted_order)}")
    if faceted_order != unfaceted_order:
        moved = [f for f in faceted_order
                 if faceted_order.index(f) != unfaceted_order.index(f)]
        print(f"  ORDER DIFFERS between payloads; moved: {', '.join(moved)}")

    print(f"\nwrote {out}/exp1_tokens.csv")
    print(f"wrote {out}/exp1_summary.json")
    print(f"wrote {out}/exp1_tokens.png")
