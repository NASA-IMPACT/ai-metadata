#!/usr/bin/env python
"""How large a query set does the format ranking actually need?

:mod:`compare_query_sets` compares one 111-query draw against the 511-query run.
One draw answers "did *this* subset agree?", which is a coin flip dressed as
evidence. This answers the question that decides the Experiment 2 budget: **over
many draws of size n, how often is the ranking preserved, and how precisely is
each format measured?**

It needs no retrieval and no LLM. ``chunked_rows.csv`` already holds one row per
(query, format) with that row's Recall@10 and nDCG@10, so a draw of n queries is
a resample of rows that already exist -- exact, not simulated.

Two things it reports, and the second matters more:

* **Ranking stability** -- how often the top format, the bottom format and the
  full order survive a draw of size n.
* **The resolution floor** -- the standard error of each format's mean at size n.
  If two formats sit closer together than that, no query set of that size can
  order them, and a subset "failing" to reproduce their order is measuring noise
  rather than disagreeing.

## Usage

    uv run python scripts/subset_power.py
    uv run python scripts/subset_power.py --sizes 50,100,200,300 --draws 2000
"""

from __future__ import annotations

import argparse
import collections
import csv
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airm.config import FORMAT_LABELS, FORMATS  # noqa: E402
from airm.runs import latest  # noqa: E402

CONDITIONS = [("faceted", "max"), ("faceted", "mean"), ("unfaceted", "max"), ("unfaceted", "mean")]
METRIC = "recall@10"
DEFAULT_SIZES = (50, 100, 200, 300, 400)
DEFAULT_DRAWS = 2000
RANDOM_SEED = 20260731


def load_rows(payload: str, pool: str, queries_path: str,
              source: str = "synthetic") -> dict[str, dict[str, float]]:
    """``{query_id: {fmt: metric}}`` for one run, restricted to one query source.

    The run is located by what its summary says it was; the rows CSV sits beside
    that summary.
    """
    summary_path, _ = latest(
        "chunked_summary.json", payload=payload, pool=pool, queries_path=queries_path
    )
    path = summary_path.parent / "chunked_rows.csv"
    if not path.exists():
        raise SystemExit(f"FAILED: no rows at {path}")
    out: dict[str, dict[str, float]] = collections.defaultdict(dict)
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row["source"] != source:
                continue
            value = row[METRIC]
            # Unmeasurable is not zero: a NaN row is dropped, matching how every
            # other mean in this study is taken.
            if value == "" or value != value:
                continue
            try:
                out[row["query_id"]][row["format"]] = float(value)
            except ValueError:
                continue
    return dict(out)


def means(rows: dict[str, dict[str, float]], qids: list[str]) -> dict[str, float]:
    return {
        fmt: statistics.fmean([rows[q][fmt] for q in qids if fmt in rows[q]])
        for fmt in FORMATS
    }


def analyse(rows: dict[str, dict[str, float]], sizes, draws: int, seed: int) -> dict:
    qids = sorted(rows)
    truth = means(rows, qids)
    order = sorted(FORMATS, key=lambda f: -truth[f])
    top, bottom = order[0], order[-1]

    out = {"n_population": len(qids), "truth": truth, "order": order, "sizes": {}}
    for n in sizes:
        if n > len(qids):
            continue
        rng = random.Random(seed + n)
        keep_top = keep_bottom = keep_all = 0
        per_fmt: dict[str, list[float]] = {f: [] for f in FORMATS}
        for _ in range(draws):
            sample = rng.sample(qids, n)
            m = means(rows, sample)
            drawn = sorted(FORMATS, key=lambda f: -m[f])
            keep_top += drawn[0] == top
            keep_bottom += drawn[-1] == bottom
            keep_all += drawn == order
            for f in FORMATS:
                per_fmt[f].append(m[f])
        out["sizes"][n] = {
            "top": keep_top / draws,
            "bottom": keep_bottom / draws,
            "exact": keep_all / draws,
            "se": {f: statistics.pstdev(per_fmt[f]) for f in FORMATS},
        }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--queries", default="data/queries_full.jsonl",
                        help="the query set whose runs are resampled")
    parser.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES))
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    print(f"resampling the synthetic slice of the {args.queries} runs, "
          f"{args.draws:,} draws per size, metric {METRIC}\n")

    verdicts = []
    for payload, pool in CONDITIONS:
        rows = load_rows(payload, pool, args.queries)
        res = analyse(rows, sizes, args.draws, args.seed)
        print(f"== {payload} / {pool}  (population {res['n_population']} queries) "
              + "=" * max(0, 20 - len(payload) - len(pool)))
        print("  ranking : " + " > ".join(FORMAT_LABELS.get(f, f) for f in res["order"]))
        gaps = [
            (res["truth"][a] - res["truth"][b], a, b)
            for a, b in zip(res["order"], res["order"][1:])
        ]
        tightest = min(gaps)
        print(f"  tightest adjacent gap : {tightest[0]:.3f} "
              f"({FORMAT_LABELS.get(tightest[1], tightest[1])} vs "
              f"{FORMAT_LABELS.get(tightest[2], tightest[2])})")
        print(f"\n  {'n':>5}{'P(top held)':>14}{'P(bottom held)':>16}{'P(exact order)':>16}"
              f"{'typical SE':>13}{'resolves':>11}")
        print("  " + "-" * 73)
        for n, v in res["sizes"].items():
            se = statistics.fmean(v["se"].values())
            print(f"  {n:>5}{v['top']:>14.1%}{v['bottom']:>16.1%}{v['exact']:>16.1%}"
                  f"{se:>13.3f}{2 * se:>11.3f}")
        verdicts.append((payload, pool, res))
        print()

    print("=" * 76)
    n_target = 100 if 100 in sizes else sizes[0]
    print(f"At n = {n_target}:")
    for payload, pool, res in verdicts:
        v = res["sizes"].get(n_target)
        if not v:
            continue
        se = statistics.fmean(v["se"].values())
        print(f"  {payload:<10}{pool:<6} top {v['top']:>6.1%} · bottom {v['bottom']:>6.1%} · "
              f"exact {v['exact']:>6.1%} · resolves gaps > {2 * se:.3f}")
    print("\nA format pair separated by less than the 'resolves' figure cannot be ordered")
    print("by a query set of that size -- in either direction. Where the 511-query run")
    print("puts two formats closer than that, the subset is not disagreeing with it; both")
    print("are reporting noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
