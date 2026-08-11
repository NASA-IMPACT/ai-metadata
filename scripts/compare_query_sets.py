#!/usr/bin/env python
"""Does the 111-query subset reproduce the 511-query retrieval picture?

Experiment 2 costs ~21 judge calls per cell, so its price is set by the query
count. Running it on ~100 queries is a 5x saving, and this decides whether that
saving costs anything.

**It compares the ``synthetic`` slice, not ``all``.** The subset keeps all 11 SME
queries, so their share rises from 2.2% (11/511) to 9.9% (11/111) -- a 4.5x
reweighting toward queries that score 0.26-0.39 against the synthetic slice's
0.85+. The pooled ``all`` numbers are therefore *expected* to move for a reason
that has nothing to do with sample size, and reading them as a sampling error
would be wrong. ``all`` and ``sme`` are printed underneath for completeness.

## The bar

* **GREEN**  ranking identical in all four conditions and max |delta| <= 0.03.
* **AMBER**  top-2 and bottom-2 hold, middle swaps, or max |delta| <= 0.06.
* **RED**    the top or bottom format changes.

At n=100 a Recall@10 near 0.87 carries a binomial standard error of ~0.034, so
adjacent formats separated by less than ~0.07 are not resolvable at this sample
size. Three formats in the 511 run sit within 0.002 of each other, so *some*
middle reordering is expected and is not a failure. That is why the verdict keys
on the ends of the ranking rather than on exact rank equality.

## Usage

    uv run python scripts/compare_query_sets.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airm.config import FORMAT_LABELS, FORMATS  # noqa: E402
from airm.runs import latest  # noqa: E402

CONDITIONS = [
    ("faceted", "max"),
    ("faceted", "mean"),
    ("unfaceted", "max"),
    ("unfaceted", "mean"),
]
METRIC = "recall@10"
SECONDARY = "ndcg@10"

GREEN, AMBER = 0.03, 0.06


def load(payload: str, pool: str, queries_path: str) -> dict:
    """The newest run matching this condition.

    Run directories are timestamps, so a run is found by what its summary says it
    was -- not by a folder name that would have to be kept in sync by hand.
    """
    _, data = latest(
        "chunked_summary.json", payload=payload, pool=pool, queries_path=queries_path
    )
    return data


def kendall_tau(a: list[str], b: list[str]) -> float:
    """Rank correlation between two orderings of the same items.

    Written out rather than pulled from scipy: six items, and the study already
    avoids a dependency it would use once.
    """
    rank_b = {x: i for i, x in enumerate(b)}
    pairs = list(itertools.combinations(range(len(a)), 2))
    if not pairs:
        return float("nan")
    # ``a[i]`` precedes ``a[j]``, so the pair is concordant when ``b`` agrees --
    # that is, when a[i] also sits *earlier* in b, i.e. the rank difference is
    # negative. Getting this sign backwards silently reports every agreement as a
    # disagreement.
    concordant = sum(1 for i, j in pairs if (rank_b[a[i]] - rank_b[a[j]]) < 0)
    return (2 * concordant - len(pairs)) / len(pairs)


def compare(slice_name: str, full: dict, sub: dict) -> dict:
    f_cells, s_cells = full["by_source"][slice_name], sub["by_source"][slice_name]
    rows = []
    for fmt in FORMATS:
        a, b = f_cells[fmt][METRIC], s_cells[fmt][METRIC]
        rows.append({
            "fmt": fmt,
            "full": a, "sub": b, "delta": b - a,
            "full_ndcg": f_cells[fmt][SECONDARY], "sub_ndcg": s_cells[fmt][SECONDARY],
        })
    order_full = [r["fmt"] for r in sorted(rows, key=lambda r: -r["full"])]
    order_sub = [r["fmt"] for r in sorted(rows, key=lambda r: -r["sub"])]
    deltas = [abs(r["delta"]) for r in rows]
    return {
        "rows": sorted(rows, key=lambda r: -r["full"]),
        "order_full": order_full,
        "order_sub": order_sub,
        "identical": order_full == order_sub,
        "ends_hold": order_full[0] == order_sub[0] and order_full[-1] == order_sub[-1],
        "top2_hold": set(order_full[:2]) == set(order_sub[:2]),
        "bottom2_hold": set(order_full[-2:]) == set(order_sub[-2:]),
        "tau": kendall_tau(order_full, order_sub),
        "max_delta": max(deltas),
        "mean_delta": sum(deltas) / len(deltas),
        "n_full": f_cells[FORMATS[0]]["n"],
        "n_sub": s_cells[FORMATS[0]]["n"],
    }


def verdict(results: dict[tuple[str, str], dict]) -> tuple[str, str]:
    worst = max(r["max_delta"] for r in results.values())
    if all(r["identical"] for r in results.values()) and worst <= GREEN:
        return "GREEN", "ranking identical in all four conditions and every delta within the bar"
    if not all(r["ends_hold"] for r in results.values()):
        broken = [f"{p}/{q}" for (p, q), r in results.items() if not r["ends_hold"]]
        return "RED", f"the top or bottom format changes in {', '.join(broken)}"
    if all(r["top2_hold"] and r["bottom2_hold"] for r in results.values()) and worst <= AMBER:
        return "AMBER", "top-2 and bottom-2 hold everywhere; middle formats reorder within noise"
    return "AMBER", f"ends hold everywhere, but max delta {worst:.3f} exceeds {AMBER}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--full", default="data/queries_full.jsonl", help="the reference query set")
    parser.add_argument("--sub", default="data/queries_subset111.jsonl", help="the subset query set")
    parser.add_argument("--slice", default="synthetic", choices=["synthetic", "all", "sme"])
    args = parser.parse_args(argv)

    results: dict[tuple[str, str], dict] = {}
    for payload, pool in CONDITIONS:
        full = load(payload, pool, args.full)
        sub = load(payload, pool, args.sub)
        results[(payload, pool)] = compare(args.slice, full, sub)

    first = next(iter(results.values()))
    print(f"runs           : " + ", ".join(
        f"{p}/{q} {load(p, q, args.full)['run_id']} vs {load(p, q, args.sub)['run_id']}"
        for p, q in CONDITIONS[:1]) + " (and 3 more)")
    print(f"slice compared : {args.slice}   n = {first['n_full']} (full) vs {first['n_sub']} (subset)")
    print(f"metric         : {METRIC}\n")

    for (payload, pool), r in results.items():
        print(f"== {payload} / {pool} " + "=" * (58 - len(payload) - len(pool)))
        print(f"{'format':<18}{'511':>9}{'111':>9}{'delta':>9}{'  ':2}{'nDCG 511':>10}{'nDCG 111':>10}")
        print("-" * 68)
        for row in r["rows"]:
            print(f"{FORMAT_LABELS.get(row['fmt'], row['fmt']):<18}"
                  f"{row['full']:>9.3f}{row['sub']:>9.3f}{row['delta']:>+9.3f}  "
                  f"{row['full_ndcg']:>10.3f}{row['sub_ndcg']:>10.3f}")
        print(f"\n  511 order : {' > '.join(FORMAT_LABELS.get(f, f) for f in r['order_full'])}")
        print(f"  111 order : {' > '.join(FORMAT_LABELS.get(f, f) for f in r['order_sub'])}")
        print(f"  identical={r['identical']}  ends_hold={r['ends_hold']}  "
              f"tau={r['tau']:+.2f}  max|delta|={r['max_delta']:.3f}  mean|delta|={r['mean_delta']:.3f}\n")

    tag, why = verdict(results)
    print("=" * 68)
    print(f"VERDICT: {tag} — {why}")
    print(f"  worst max|delta| across conditions : {max(r['max_delta'] for r in results.values()):.3f}")
    print(f"  conditions with identical ranking  : "
          f"{sum(r['identical'] for r in results.values())} of {len(results)}")
    print(f"  conditions with ends intact        : "
          f"{sum(r['ends_hold'] for r in results.values())} of {len(results)}")

    # The other slices, for completeness -- and to show why `all` is not the
    # comparison: the SME share moves from 2.2% to 9.9% between the two sets.
    if args.slice == "synthetic":
        for other in ("all", "sme"):
            worst = max(
                compare(other, load(p, q, args.full), load(p, q, args.sub))["max_delta"]
                for p, q in CONDITIONS
            )
            print(f"  (slice '{other}': worst max|delta| {worst:.3f} — not the comparison; "
                  f"the SME share differs between the sets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
