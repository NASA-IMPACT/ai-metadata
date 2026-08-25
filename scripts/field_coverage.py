#!/usr/bin/env python
"""Field-level coverage of the two payloads: present vs informative.

Two numbers per field, because CMR conflates them:

``present``
    The source value exists in the record CMR serves -- even when it is a
    placeholder ("Not provided", "NA", a DOI MissingReason stub).
``informative``
    The value survives the projection's own placeholder rule
    (:data:`airm.facets.PLACEHOLDERS`) -- it asserts an actual fact.

Faceted -- both numbers come from the projection's own extractors, run twice
per raw record: once normally (informative -- cross-checked against the cached
rendering) and once with the placeholder set emptied (present). No hand-built
field mapping, so the two columns cannot drift from what the formats carry.

Unfaceted -- per top-level UMM field: present = the field appears in the
rendering; informative = at least one leaf under it is informative, where a
leaf is NOT informative when its key is absence machinery (``MissingReason``,
``Explanation``) or its string value is empty / in ``PLACEHOLDERS``. Numbers
and booleans are always informative.

The parity gate makes every number here identical across all six formats of a
payload -- these are payload facts, not format facts.

Outputs (default ``runs/cache_coverage/``):
    field_coverage.json     per-field present/informative + distributions
    field_presence.csv      payload, field, n/share present, n/share informative
and prints the summary tables.

Usage:
    uv run python scripts/field_coverage.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import airm.facets as facets_mod  # noqa: E402
from airm.facets import FACET_KEYS, PLACEHOLDERS  # noqa: E402

CACHES = {
    "faceted": Path("data/format_cache/json"),
    "unfaceted": Path("data/format_cache_unfaceted/json"),
}
RAW = Path("data/cmr_cache")
#: keys that describe an absence rather than a fact (UMM's DOI stub machinery)
ABSENCE_KEYS = frozenset({"MissingReason", "Explanation"})


def informative_leaf(key: str, value) -> bool:
    if key in ABSENCE_KEYS:
        return False
    if isinstance(value, str):
        text = value.strip()
        return bool(text) and text.lower() not in PLACEHOLDERS
    return value is not None


def walk(obj, key=""):
    """Yield ``(top_is_informative,)`` leaves as (key, value) pairs."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, k)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v, key)
    else:
        yield key, obj


def dist(vals: list[float]) -> dict:
    vals = sorted(vals)
    return {"mean": round(statistics.fmean(vals), 3),
            "median": vals[len(vals) // 2],
            "min": vals[0], "max": vals[-1]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="runs/cache_coverage", type=Path)
    args = parser.parse_args(argv)

    ids = sorted(f.stem for f in CACHES["faceted"].glob("*.json"))

    # -------------------------------------------------- faceted, two passes
    fac_inf, fac_pres = Counter(), Counter()
    fac_inf_pr, fac_pres_pr = [], []
    mismatches = 0
    for rid in ids:
        raw = json.loads((RAW / f"{rid}.json").read_text())
        cached = set(json.loads(
            (CACHES["faceted"] / f"{rid}.json").read_text()))
        informative = set(facets_mod.facets(raw)) & set(FACET_KEYS)
        if informative != cached & set(FACET_KEYS):
            mismatches += 1
        facets_mod.PLACEHOLDERS = frozenset()
        try:
            present = set(facets_mod.facets(raw)) & set(FACET_KEYS)
        finally:
            facets_mod.PLACEHOLDERS = PLACEHOLDERS
        present |= informative  # cleaning can only remove
        fac_inf.update(informative)
        fac_pres.update(present)
        fac_inf_pr.append(len(informative))
        fac_pres_pr.append(len(present))
    n = len(ids)
    if mismatches:
        print(f"! {mismatches} records where re-extraction != cached rendering "
              "(cache may be stale)")

    # ------------------------------------------- unfaceted, leaf-level walk
    unf_pres, unf_inf = Counter(), Counter()
    unf_pres_pr, unf_inf_pr = [], []
    path_pres, path_inf = Counter(), Counter()
    for rid in ids:
        doc = json.loads((CACHES["unfaceted"] / f"{rid}.json").read_text())
        inf_fields = set()
        seen_paths, inf_paths = set(), set()

        def paths(obj, prefix, top, key=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    paths(v, f"{prefix}.{k}" if prefix else k, top or k, k)
            elif isinstance(obj, list):
                for v in obj:
                    paths(v, prefix + "[]", top, key)
            else:
                seen_paths.add(prefix)
                if informative_leaf(key, obj):
                    inf_paths.add(prefix)
                    inf_fields.add(top)

        paths(doc, "", None)
        unf_pres.update(doc.keys())
        unf_inf.update(inf_fields)
        path_pres.update(seen_paths)
        path_inf.update(inf_paths)
        unf_pres_pr.append(len(doc))
        unf_inf_pr.append(len(inf_fields))
    top_universe = sorted(unf_pres)

    # -------------------------------------------------------------- persist
    def table(universe, pres, inf):
        return {k: {"n_present": pres[k],
                    "share_present": round(pres[k] / n, 4),
                    "n_informative": inf[k],
                    "share_informative": round(inf[k] / n, 4)}
                for k in universe}

    out = {
        "records": n,
        "definitions": {
            "present": "source value exists, placeholders included "
                       "(faceted: extractors with the placeholder rule off)",
            "informative": "survives airm.facets.PLACEHOLDERS + the "
                           f"absence-machinery keys {sorted(ABSENCE_KEYS)}",
        },
        "faceted": {
            "universe": len(FACET_KEYS),
            "fields": table(FACET_KEYS, fac_pres, fac_inf),
            "per_record_present": dist(fac_pres_pr),
            "per_record_informative": dist(fac_inf_pr),
        },
        "unfaceted": {
            "universe_top_level": len(top_universe),
            "universe_paths": len(path_pres),
            "fields": table(top_universe, unf_pres, unf_inf),
            "per_record_present": dist(unf_pres_pr),
            "per_record_informative": dist(unf_inf_pr),
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "field_coverage.json").write_text(
        json.dumps(out, indent=2) + "\n")
    with (args.out / "field_presence.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["payload", "field", "n_present", "share_present",
                    "n_informative", "share_informative"])
        for k in FACET_KEYS:
            w.writerow(["faceted", k, fac_pres[k], round(fac_pres[k] / n, 4),
                        fac_inf[k], round(fac_inf[k] / n, 4)])
        for k in top_universe:
            w.writerow(["unfaceted", k, unf_pres[k], round(unf_pres[k] / n, 4),
                        unf_inf[k], round(unf_inf[k] / n, 4)])
        for k in sorted(path_pres):
            w.writerow(["unfaceted:path", k, path_pres[k],
                        round(path_pres[k] / n, 4),
                        path_inf[k], round(path_inf[k] / n, 4)])

    # ---------------------------------------------------------------- print
    def show(title, universe, pres, inf):
        print(f"\n== {title} ==")
        print(f"  {'field':<34}{'present':>9}{'%':>8}{'inform.':>9}{'%':>8}"
              f"{'gap':>6}")
        for k in sorted(universe, key=lambda k: (-inf[k], -pres[k])):
            gap = pres[k] - inf[k]
            print(f"  {k:<34}{pres[k]:>9}{pres[k] / n:>8.1%}"
                  f"{inf[k]:>9}{inf[k] / n:>8.1%}{gap:>6}")

    print(f"records: {n}")
    show(f"faceted: the {len(FACET_KEYS)} facets", FACET_KEYS,
         fac_pres, fac_inf)
    print(f"\n  per record: present mean {dist(fac_pres_pr)['mean']}/32, "
          f"informative mean {dist(fac_inf_pr)['mean']}/32")
    show(f"unfaceted: {len(top_universe)} top-level UMM fields",
         top_universe, unf_pres, unf_inf)
    print(f"\n  per record: present mean {dist(unf_pres_pr)['mean']}"
          f"/{len(top_universe)}, informative mean "
          f"{dist(unf_inf_pr)['mean']}/{len(top_universe)}")
    print(f"  distinct leaf paths: {len(path_pres)} present, "
          f"{len(path_inf)} ever informative")
    print(f"\nwrote {args.out / 'field_coverage.json'}")
    print(f"wrote {args.out / 'field_presence.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
