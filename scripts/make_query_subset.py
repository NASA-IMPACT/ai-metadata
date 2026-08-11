#!/usr/bin/env python
"""A 111-query stratified subset of the 511-query evaluation set.

Experiment 2 costs ~21 DeepEval judge calls per cell, so its price scales with
the query count and nothing else. Running it on ~100 queries instead of 511 is a
5x saving -- but only legitimate if the smaller set reproduces the retrieval
picture the full set produces. This builds that subset so the claim can be
tested rather than assumed.

* **All 11 SME queries**, verbatim. They are the expert half of the ground truth
  and there is no sampling decision to make about eleven queries.
* **100 synthetic**, stratified so the topic mix matches the 500 they come from.
  Sampling uniformly at random would drift the mix, and Experiment 1 found
  per-topic token means vary more than formats do -- topic mix is not a
  negligible variable to leave to chance.

Quotas use largest-remainder rounding with an alphabetical tie-break, so the one
topic that rounds down is a stated choice rather than dictionary order. Within a
topic the draw is seeded with the study's own seed, so the file rebuilds
byte-identically.

**The SME share changes, and the comparison has to account for it.** 11 of 111 is
9.9% against 11 of 511 at 2.2% -- a 4.5x reweighting toward queries that score
0.26-0.39 where the synthetic slice scores 0.85+. The pooled ``all`` numbers will
therefore differ for a reason that has nothing to do with sample size. Compare the
``synthetic`` slice.

## Usage

    uv run python scripts/make_query_subset.py
    uv run python scripts/make_query_subset.py --n 100 --out data/queries_subset111.jsonl
"""

from __future__ import annotations

import argparse
import collections
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airm.queries import Query, load_queries  # noqa: E402

SOURCE = "data/queries_full.jsonl"
OUT = "data/queries_subset111.jsonl"

#: How many synthetic queries to keep. The SME half is taken whole.
SYNTHETIC_N = 100

#: The seed ``corpus.build`` and ``synth`` use. Same seed, same subset, forever.
RANDOM_SEED = 20260731


def quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    """Proportional allocation by largest remainder.

    Rounding down everywhere loses records and rounding to nearest overshoots, so
    the fractional parts decide who gets the leftovers. Ties break alphabetically
    rather than on dict order, which would silently depend on insertion order.
    """
    population = sum(counts.values())
    exact = {t: counts[t] * total / population for t in counts}
    base = {t: int(exact[t]) for t in counts}
    leftover = total - sum(base.values())
    order = sorted(counts, key=lambda t: (-(exact[t] - base[t]), t))
    for topic in order[:leftover]:
        base[topic] += 1
    return base


def build(source: Path, n_synthetic: int, seed: int) -> tuple[list[Query], dict]:
    queries = load_queries(source)
    sme = [q for q in queries if q.source == "sme"]
    synthetic = [q for q in queries if q.source == "synthetic"]
    if not sme or not synthetic:
        raise SystemExit(f"FAILED: {source} has {len(sme)} sme and {len(synthetic)} synthetic.")

    by_topic: dict[str, list[Query]] = collections.defaultdict(list)
    for q in synthetic:
        by_topic[q.topic or "UNCLASSIFIED"].append(q)

    counts = {t: len(v) for t, v in by_topic.items()}
    want = quotas(counts, n_synthetic)

    rng = random.Random(seed)
    picked: list[Query] = []
    for topic in sorted(by_topic):
        # Sort before sampling: the file order is the generator's, and seeding a
        # draw over an unsorted list makes the result depend on that order.
        pool = sorted(by_topic[topic], key=lambda q: q.query_id)
        picked.extend(rng.sample(pool, want[topic]))

    picked.sort(key=lambda q: q.query_id)
    report = {
        "source": str(source),
        "sme": len(sme),
        "synthetic": len(picked),
        "total": len(sme) + len(picked),
        "topics": {t: {"in_source": counts[t], "in_subset": want[t]} for t in sorted(counts)},
    }
    return sme + picked, report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--n", type=int, default=SYNTHETIC_N, help="synthetic queries to keep")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)

    subset, report = build(Path(args.source), args.n, args.seed)

    print(f"source            : {report['source']}")
    print(f"sme (all kept)    : {report['sme']}")
    print(f"synthetic sampled : {report['synthetic']}")
    print(f"total             : {report['total']}\n")
    print(f"{'topic':<28}{'in 500':>8}{'in subset':>11}{'share kept':>12}")
    print("-" * 59)
    for topic, v in report["topics"].items():
        share = v["in_subset"] / v["in_source"]
        print(f"{topic:<28}{v['in_source']:>8}{v['in_subset']:>11}{share:>12.1%}")

    # Loud rather than quiet: a subset that silently came up short would be
    # compared against the full set as though the only difference were sampling.
    assert report["synthetic"] == args.n, f"wanted {args.n} synthetic, got {report['synthetic']}"
    assert len(subset) == report["total"]
    assert len({q.query_id for q in subset}) == len(subset), "duplicate query_id in subset"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(q.to_json() for q in subset) + "\n")
    print(f"\nwrote {out}  ({len(subset)} queries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
