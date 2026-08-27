#!/usr/bin/env python
"""Color-per-format variant of the pareto and faithfulness charts.

Same panels, axes, connectors and captions as ``analyze_50q.py``'s charts, but
identity is carried by color instead of text labels: each format keeps one
color in every panel, a filled circle is the faceted payload and an open
circle of the same color is the unfaceted payload. Two legends — formats by
color, payload by fill.

Reads the frozen ``runs/<rid>/analysis.json`` (no recomputation) and writes to
NEW files next to the originals:

    runs/<rid>/pareto_by_format.png
    runs/<rid>/faithfulness_by_format.png

Usage:
    uv run python scripts/pareto_by_format.py                 # default run
    uv run python scripts/pareto_by_format.py --run RUN_ID
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airm.config import FORMATS, RUNS_DIR  # noqa: E402

# shared surface/ink tokens (as analyze_50q.py)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

#: One vivid hue per format — blue / orange / green / brown / purple / red;
#: validated (dataviz six-checks, light mode; orange<->green sits in the CVD
#: floor band, carried by the text legend + connectors).
FMT_COLOR = {
    "json": "#2A78D6",
    "csv": "#E8930C",
    "yaml": "#109648",
    "toon": "#8C510A",
    "jsonld": "#8E44AD",
    "mat": "#D62728",
}
FMT_LABEL = {"json": "JSON", "csv": "CSV", "yaml": "YAML", "toon": "TOON",
             "jsonld": "JSON-LD", "mat": "MaT"}
MODEL_LABEL = {"gpt-5.4-nano": "gpt-5.4-nano (cloud)",
               "gpt-5.4-mini": "gpt-5.4-mini (cloud)",
               "muse-glimmer:30b": "muse-glimmer 30B (local)"}


def chart(arms: dict, path: Path, metric: str, metric_label: str,
          n_queries: int, n_sme: int, n_syn: int) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    models = sorted({k.split("|")[2] for k in arms})
    fig, axes = plt.subplots(1, len(models), figsize=(5.8 * len(models), 6.6),
                             dpi=200, sharey=True, sharex=True, squeeze=False)
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
            color = FMT_COLOR[fmt]
            if "faceted" in pts:
                # the surface rim eats into the colored disk; oversize the
                # filled marker so its visible diameter matches the open one
                ax.scatter(*map(lambda v: [v], pts["faceted"]), s=110,
                           color=color, zorder=3, edgecolors=SURFACE,
                           linewidths=2)
            if "unfaceted" in pts:
                ax.scatter(*map(lambda v: [v], pts["unfaceted"]), s=64,
                           facecolors="none", zorder=3, edgecolors=color,
                           linewidths=1.8)
        ax.set_ylim(ymin, ymax)
        ax.set_title(MODEL_LABEL.get(model, model), fontsize=14, color=INK,
                     pad=10)
        ax.set_xlabel("mean prompt tokens per query (thousands)",
                      fontsize=12, color=INK_MUTED)
        ax.grid(color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_MUTED, labelsize=11)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)

    axes[0].set_ylabel(f"{metric_label} (nano-judged)", fontsize=12,
                       color=INK_MUTED)

    # legend 1: payload by fill (first panel)
    payload_handles = [
        Line2D([], [], marker="o", ls="", color=INK_SECONDARY, markersize=10,
               label="subsetted records (32-field projection)"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none",
               markeredgecolor=INK_SECONDARY, markeredgewidth=1.6,
               markersize=10, label="full records (raw UMM record)"),
    ]
    axes[0].legend(handles=payload_handles, frameon=False, fontsize=11.5,
                   loc="upper right", labelcolor=INK_SECONDARY)

    # legend 2: format by color (last panel)
    fmt_handles = [
        Line2D([], [], marker="o", ls="", color=FMT_COLOR[f], markersize=10,
               label=FMT_LABEL[f])
        for f in FORMATS
    ]
    axes[-1].legend(handles=fmt_handles, frameon=False, fontsize=11.5,
                    loc="lower right", labelcolor=INK_SECONDARY, ncols=2,
                    columnspacing=1.2, handletextpad=0.4)

    fig.suptitle(f"Answer {metric_label} vs context cost — {n_queries} "
                 "queries, top-10 records in context",
                 fontsize=16, color=INK, x=0.008, ha="left", y=0.99)
    fig.text(0.008, 0.012,
             "One color per format; the filled circle is the subsetted record, "
             "the open circle its full-record partner, joined by the gray "
             f"connector. Judge: gpt-5.4-nano; {metric_label} pooled over "
             f"{n_sme} SME + {n_syn} synthetic queries.",
             fontsize=9.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", default="20260819T183313Z", metavar="RUN_ID")
    args = parser.parse_args(argv)

    run_dir = RUNS_DIR / args.run
    analysis = json.loads((run_dir / "analysis.json").read_text())
    arms = analysis["arms"]
    selection = json.loads((run_dir / "selection.json").read_text())
    counts = {"n_queries": analysis["queries"],
              "n_sme": selection.get("sme", 0),
              "n_syn": analysis["queries"] - selection.get("sme", 0)}

    for metric in ("correctness", "faithfulness"):
        out = run_dir / (("pareto_by_format.png" if metric == "correctness"
                          else "faithfulness_by_format.png"))
        chart(arms, out, metric, metric, **counts)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
