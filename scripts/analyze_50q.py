#!/usr/bin/env python
"""Analysis of the 50-query staged pipeline: retrieval x tokens x judged quality.

Joins the three frozen artifacts of run 20260814T210927Z --

    Stage R  chunked_retrieval.jsonl   exact retrieval metrics (restricted to the 50)
    Stage A  answers.jsonl             tokens, latency per cell
    Stage J  judgements.jsonl          five nano-judged quality metrics per cell

-- into per-arm tables and *paired* comparisons. Every comparison is paired on
(query, format, model) or (query, payload, model): the same query is asked in
every arm, so pairing removes between-query variance, which is large (SME
queries score far below synthetic ones).

Outputs:
    runs/<rid>/analysis.json     every table, machine-readable
    runs/<rid>/pareto.png        correctness vs prompt tokens, per model panel

Confidence intervals are t-based over per-query paired differences (df = n-1).
Cells with a judge error on a metric contribute nothing to that metric's mean
(None, not zero) and are dropped pairwise from paired comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airm.config import FORMATS, RUNS_DIR  # noqa: E402
from airm.provenance import provenance  # noqa: E402

RID = "20260814T210927Z"
RETRIEVAL_RUNS = {"faceted": "20260814T205849Z", "unfaceted": "20260814T205912Z"}
JUDGE = "openai:gpt-5.4-nano"
JUDGE_METRICS = ("correctness", "faithfulness", "answer_relevancy", "contextual_recall")
#: contextual_relevancy is loaded but reported separately: with 1-2 expected
#: records among 10 retrieved, it measures retrieval precision, which Stage R
#: already measures exactly.

#: USD per M tokens (input, output). nano verified 2026-08-15 (pricepertoken.com);
#: mini verified 2026-08-19 (developers.openai.com/api/docs/pricing, standard tier).
PRICES = {"gpt-5.4-nano": (0.20, 1.25), "gpt-5.4-mini": (0.75, 4.50)}
NANO_IN, NANO_OUT = PRICES["gpt-5.4-nano"]

#: Baseline for the model axis: every other model is differenced against it.
BASE_MODEL = "gpt-5.4-nano"
SHORT = {"gpt-5.4-nano": "nano", "gpt-5.4-mini": "mini", "muse-glimmer:30b-mlx": "glimmer"}

#: two-sided t critical values at 95%, by df; linear enough between entries.
T95 = {29: 2.045, 39: 2.023, 49: 2.010, 59: 2.001, 99: 1.984, 149: 1.976, 199: 1.972}


def t95(df: int) -> float:
    for k in sorted(T95):
        if df <= k:
            return T95[k]
    return 1.96


def mean_ci(diffs: list[float]) -> dict:
    """Mean of paired differences with a 95% t-interval."""
    n = len(diffs)
    if n < 2:
        return {"n": n, "mean": diffs[0] if diffs else None, "ci": None}
    m = statistics.fmean(diffs)
    se = statistics.stdev(diffs) / math.sqrt(n)
    h = t95(n - 1) * se
    return {"n": n, "mean": round(m, 4), "ci": [round(m - h, 4), round(m + h, 4)]}


def mean(vals) -> float | None:
    real = [v for v in vals if v is not None]
    return round(statistics.fmean(real), 4) if real else None


def load(rid: str = RID):
    run = RUNS_DIR / rid
    answers = {}
    for line in (run / "answers.jsonl").open():
        c = json.loads(line)
        answers[(c["payload"], c["fmt"], c["query_id"], c["model"])] = c
    judgements = {}
    for line in (run / "judgements.jsonl").open():
        j = json.loads(line)
        if j["judge"] == JUDGE:
            judgements[(j["payload"], j["fmt"], j["query_id"], j["model"])] = j
    selected = set(json.loads((run / "selection.json").read_text())["query_ids"])
    retrieval = {}
    for payload, rrid in RETRIEVAL_RUNS.items():
        for line in (RUNS_DIR / rrid / "chunked_retrieval.jsonl").open():
            r = json.loads(line)
            if r["query_id"] in selected:
                retrieval[(payload, r["format"], r["query_id"])] = r
    return answers, judgements, selected, retrieval


def cells_join(answers, judgements):
    """One joined row per judged cell."""
    rows = []
    for key, j in judgements.items():
        a = answers[key]
        payload, fmt, qid, model = key
        rows.append({
            "payload": payload, "fmt": fmt, "query_id": qid, "model": model,
            "source": j["source"],
            **{m: j["scores"].get(m) for m in
               (*JUDGE_METRICS, "contextual_relevancy")},
            "prompt_tokens": a["prompt_tokens"],
            "completion_tokens": a["completion_tokens"],
            "latency_s": a["latency_s"],
        })
    return rows


def arm_table(rows, retrieval) -> dict:
    """Per (payload x format x model): quality, cost, and slice means."""
    arms = defaultdict(list)
    for r in rows:
        arms[(r["payload"], r["fmt"], r["model"])].append(r)
    out = {}
    for (payload, fmt, model), group in sorted(arms.items()):
        entry = {"n": len(group)}
        for m in (*JUDGE_METRICS, "contextual_relevancy"):
            entry[m] = mean(r[m] for r in group)
            for src in ("sme", "synthetic"):
                entry[f"{m}__{src}"] = mean(r[m] for r in group if r["source"] == src)
        entry["prompt_tokens"] = mean(r["prompt_tokens"] for r in group)
        entry["completion_tokens"] = mean(r["completion_tokens"] for r in group)
        entry["latency_s"] = mean(r["latency_s"] for r in group)
        if model in PRICES and entry["prompt_tokens"] is not None:
            p_in, p_out = PRICES[model]
            entry["usd_per_1k_queries"] = round(
                (entry["prompt_tokens"] * p_in + entry["completion_tokens"] * p_out)
                / 1e6 * 1000, 2)
        rmetrics = [retrieval[(payload, fmt, r["query_id"])] for r in group
                    if r["model"] == model]
        entry["recall@10"] = mean(r["recall@10"] for r in rmetrics)
        entry["ndcg@10"] = mean(r["ndcg@10"] for r in rmetrics)
        out[f"{payload}|{fmt}|{model}"] = entry
    return out


def paired(rows, metric: str, axis: str) -> dict:
    """Paired differences along one axis, everything else held fixed.

    axis='payload' : faceted - unfaceted per (query, fmt, model)
    axis='model'   : glimmer - nano   per (query, fmt, payload)
    axis='format'  : fmt - json       per (query, payload, model), one entry per fmt
    """
    def val(r):
        return r[metric]

    index = {(r["payload"], r["fmt"], r["query_id"], r["model"]): r for r in rows}
    out = {}
    if axis == "payload":
        for scope in ("all", "per_model"):
            pass
        diffs_all, by_model = [], defaultdict(list)
        for (payload, fmt, qid, model), r in index.items():
            if payload != "faceted":
                continue
            other = index.get(("unfaceted", fmt, qid, model))
            if other and val(r) is not None and val(other) is not None:
                d = val(r) - val(other)
                diffs_all.append(d)
                by_model[model].append(d)
        out["faceted_minus_unfaceted"] = mean_ci(diffs_all)
        for model, ds in sorted(by_model.items()):
            out[f"faceted_minus_unfaceted__{model}"] = mean_ci(ds)
    elif axis == "model":
        # Every model against the baseline, plus every other pair, so a third
        # model slots in without new code. Names read "<a>_minus_<b>".
        models = sorted({m for (_, _, _, m) in index})
        pairs = [(m, BASE_MODEL) for m in models if m != BASE_MODEL]
        others = [m for m in models if m != BASE_MODEL]
        pairs += [(a, b) for i, a in enumerate(others) for b in others[i + 1:]]
        for a, b in pairs:
            diffs_all, by_payload = [], defaultdict(list)
            for (payload, fmt, qid, model), r in index.items():
                if model != a:
                    continue
                other = index.get((payload, fmt, qid, b))
                if other and val(r) is not None and val(other) is not None:
                    d = val(r) - val(other)
                    diffs_all.append(d)
                    by_payload[payload].append(d)
            if not diffs_all:
                continue
            name = f"{SHORT.get(a, a)}_minus_{SHORT.get(b, b)}"
            out[name] = mean_ci(diffs_all)
            for payload, ds in sorted(by_payload.items()):
                out[f"{name}__{payload}"] = mean_ci(ds)
    elif axis == "format":
        for fmt in FORMATS:
            if fmt == "json":
                continue
            diffs = []
            for (payload, f, qid, model), r in index.items():
                if f != fmt:
                    continue
                base = index.get((payload, "json", qid, model))
                if base and val(r) is not None and val(base) is not None:
                    diffs.append(val(r) - val(base))
            out[f"{fmt}_minus_json"] = mean_ci(diffs)
    return out


# --------------------------------------------------------------------------- #
# Chart.
# --------------------------------------------------------------------------- #

# dataviz reference palette (light mode): payload is the color axis (2 slots,
# adjacent pair, validated); format identity is carried by direct labels, not
# color -- a 6-color scatter would need all-pairs validation the palette
# deliberately caps at three slots.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
C_FACETED = "#2a78d6"   # slot 1 blue
C_UNFACETED = "#eb6834"  # slot 2 orange

FMT_LABEL = {"json": "JSON", "csv": "CSV", "yaml": "YAML", "toon": "TOON",
             "jsonld": "JSON-LD", "mat": "MaT"}
MODEL_LABEL = {"gpt-5.4-nano": "gpt-5.4-nano (cloud)",
               "gpt-5.4-mini": "gpt-5.4-mini (cloud)",
               "muse-glimmer:30b-mlx": "muse-glimmer 30B (local)"}


def pareto_plot(arms: dict, path: Path, metric: str = "correctness",
                metric_label: str = "correctness") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    models = sorted({k.split("|")[2] for k in arms})
    fig, axes = plt.subplots(1, len(models), figsize=(5.8 * len(models), 4.8), dpi=200,
                             sharey=True, sharex=True, squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor(SURFACE)

    ymax = max(e[metric] for e in arms.values() if e[metric] is not None) + 0.025
    ymin = min(e[metric] for e in arms.values() if e[metric] is not None) - 0.02

    for ax, model in zip(axes, models):
        ax.set_facecolor(SURFACE)
        for fmt in FORMATS:
            pts = {}
            for payload in ("faceted", "unfaceted"):
                e = arms.get(f"{payload}|{fmt}|{model}")
                if e and e[metric] is not None:
                    pts[payload] = (e["prompt_tokens"] / 1000, e[metric])
            if len(pts) == 2:
                (x1, y1), (x2, y2) = pts["faceted"], pts["unfaceted"]
                ax.plot([x1, x2], [y1, y2], color=GRID, lw=1.2, zorder=1)
        labels = []
        points = []
        for payload, color in (("faceted", C_FACETED), ("unfaceted", C_UNFACETED)):
            for fmt in FORMATS:
                e = arms.get(f"{payload}|{fmt}|{model}")
                if not e or e[metric] is None:
                    continue
                x = e["prompt_tokens"] / 1000
                y = e[metric]
                ax.scatter([x], [y], s=64, color=color, zorder=3,
                           edgecolors=SURFACE, linewidths=2)
                points.append((x, y))
                if payload == "faceted":
                    labels.append((x, y, FMT_LABEL[fmt]))
        # Only the faceted point of each pair is labeled -- the connector
        # carries the identity to its unfaceted partner. Labels sit left of
        # the marker, greedily pushed down where they'd collide with an
        # already-placed label OR with a data point in the text's footprint.
        placed: list[float] = []
        gap = (ymax - ymin) * 0.048  # ~one text-line in data units

        def collides(x, ly):
            if any(abs(ly - py) < gap for py in placed):
                return True
            # text ends at x-1.1 and extends ~2.5 data units left of that
            return any(x - 3.6 < ox < x - 0.7 and abs(oy - ly) < gap * 0.7
                       for ox, oy in points)

        for x, y, text in sorted(labels, key=lambda p: -p[1]):
            ly = y
            while collides(x, ly):
                ly -= gap
            placed.append(ly)
            ax.annotate(text, (x, y), xytext=(x - 1.1, ly), textcoords="data",
                        ha="right", va="center", fontsize=7.5, color=INK_SECONDARY)
        ax.set_ylim(ymin, ymax)
        ax.set_title(MODEL_LABEL.get(model, model), fontsize=10.5, color=INK, pad=8)
        ax.set_xlabel("mean prompt tokens per query (thousands)",
                      fontsize=8.5, color=INK_MUTED)
        ax.grid(color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)

    axes[0].set_ylabel(f"{metric_label} (nano-judged)", fontsize=8.5, color=INK_MUTED)
    handles = [
        Line2D([], [], marker="o", ls="", color=C_FACETED, markersize=8,
               label="faceted (32-field projection)"),
        Line2D([], [], marker="o", ls="", color=C_UNFACETED, markersize=8,
               label="unfaceted (raw UMM record)"),
    ]
    axes[-1].legend(handles=handles, frameon=False, fontsize=8.5,
                    loc="lower right", labelcolor=INK_SECONDARY)
    fig.suptitle(f"Answer {metric_label} vs context cost — 50 queries, top-10 records in context",
                 fontsize=12, color=INK, x=0.008, ha="left", y=0.99)
    fig.text(0.008, 0.012,
             "Labels name the faceted point; the gray connector leads to the same format's "
             "unfaceted point. Judge: gpt-5.4-nano; correctness pooled over 11 SME + 39 "
             "synthetic queries.".replace("correctness", metric_label),
             fontsize=6.8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", default=RID, metavar="RUN_ID",
                        help=f"Stage A/J run id under runs/ to analyse (default {RID})")
    args = parser.parse_args(argv)
    rid = args.run

    answers, judgements, selected, retrieval = load(rid)
    rows = cells_join(answers, judgements)
    arms = arm_table(rows, retrieval)

    comparisons = {
        metric: {
            "payload": paired(rows, metric, "payload"),
            "model": paired(rows, metric, "model"),
            "format_vs_json": paired(rows, metric, "format"),
        }
        for metric in ("correctness", "faithfulness")
    }
    comparisons["prompt_tokens"] = {"payload": paired(rows, "prompt_tokens", "payload")}

    out = RUNS_DIR / rid
    analysis = {
        **provenance(rid),
        "judge": JUDGE,
        "queries": len(selected),
        "cells": len(rows),
        "arms": arms,
        "paired_comparisons": comparisons,
        "prices_usd_per_mtok": {**{m: list(v) for m, v in PRICES.items()},
                                "as_of": "2026-08-19"},
    }
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    chart = pareto_plot(arms, out / "pareto.png")
    chart2 = pareto_plot(arms, out / "faithfulness.png", metric="faithfulness",
                         metric_label="faithfulness")

    def fmt_ci(c):
        if c["ci"] is None:
            return f"{c['mean']}"
        return f"{c['mean']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] (n={c['n']})"

    print(f"cells analysed: {len(rows)}  queries: {len(selected)}\n")
    print("== per-arm means (correctness / faithfulness / prompt tok / $ per 1k queries) ==")
    print(f"  {'payload':<10}{'fmt':<8}{'model':<22}{'n':>4}{'corr':>8}{'faith':>8}"
          f"{'ptok':>9}{'$/1k':>8}{'R@10':>7}")
    for key, e in arms.items():
        payload, fmt, model = key.split("|")
        usd = e.get("usd_per_1k_queries")
        print(f"  {payload:<10}{fmt:<8}{model:<22}{e['n']:>4}"
              f"{(e['correctness'] if e['correctness'] is not None else float('nan')):>8.3f}"
              f"{(e['faithfulness'] if e['faithfulness'] is not None else float('nan')):>8.3f}"
              f"{e['prompt_tokens']:>9.0f}{(usd if usd is not None else float('nan')):>8.2f}"
              f"{e['recall@10']:>7.3f}")
    print()
    print("== paired: correctness ==")
    for name, c in comparisons["correctness"]["payload"].items():
        print(f"  {name:<46} {fmt_ci(c)}")
    for name, c in comparisons["correctness"]["model"].items():
        print(f"  {name:<46} {fmt_ci(c)}")
    print("  -- formats vs JSON baseline (paired, both payloads x models) --")
    for name, c in comparisons["correctness"]["format_vs_json"].items():
        print(f"  {name:<46} {fmt_ci(c)}")
    print("\n== paired: faithfulness ==")
    for name, c in comparisons["faithfulness"]["payload"].items():
        print(f"  {name:<46} {fmt_ci(c)}")
    for name, c in comparisons["faithfulness"]["model"].items():
        print(f"  {name:<46} {fmt_ci(c)}")
    print(f"\nwrote {out / 'analysis.json'}")
    print(f"wrote {chart}")
    print(f"wrote {chart2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
