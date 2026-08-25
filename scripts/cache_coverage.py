#!/usr/bin/env python
"""Coverage sweep of the rendered format caches.

Walks every rendering in ``data/format_cache`` (faceted) and
``data/format_cache_unfaceted`` (raw UMM) -- 6 formats x 2 payloads over the
full record set -- and measures each file four ways:

    bytes / chars      tokenizer-independent size controls
    tiktoken           o200k_base, the OpenAI-side count (airm.tokens)
    hf_tokens          Qwen/Qwen3-8B, the open-weights cross-check (optional)

plus, per record and payload, two structure measures read from the JSON
rendering: top-level key count and leaf-value count.

Completeness is part of coverage: the script reports any record id missing
from any of the 12 directories (relative to the union of ids seen).

Outputs (default ``runs/cache_coverage/``):
    per_record.csv     one row per (payload, format, record)
    coverage.json      aggregates per payload x format + completeness
and prints the summary tables.

Usage:
    uv run python scripts/cache_coverage.py
    uv run python scripts/cache_coverage.py --no-hf        # skip HF tokenizer
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airm.config import FORMATS  # noqa: E402
from airm import tokens  # noqa: E402

CACHES = {
    "faceted": Path("data/format_cache"),
    "unfaceted": Path("data/format_cache_unfaceted"),
}
EXT = {"json": "json", "csv": "csv", "yaml": "yaml", "toon": "toon",
       "jsonld": "jsonld", "mat": "txt"}


def leaves(obj) -> int:
    if isinstance(obj, dict):
        return sum(leaves(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(leaves(v) for v in obj)
    return 1


def summarize(vals: list[int]) -> dict:
    vals = sorted(vals)
    return {
        "total": sum(vals),
        "mean": round(statistics.fmean(vals), 1),
        "median": vals[len(vals) // 2],
        "p95": vals[min(len(vals) - 1, round(0.95 * len(vals)) - 1)],
        "max": vals[-1],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-hf", action="store_true",
                        help="skip the Hugging Face tokenizer column")
    parser.add_argument("--out", default="runs/cache_coverage", type=Path)
    args = parser.parse_args(argv)

    use_hf = not args.no_hf
    if use_hf:
        ok, detail = tokens.hf_available()
        if not ok:
            print(f"hf tokenizer unavailable ({detail}); column will be empty")
            use_hf = False

    # ------------------------------------------------------------ file walk
    ids: dict[tuple[str, str], set[str]] = {}
    rows: list[dict] = []
    for payload, root in CACHES.items():
        for fmt in FORMATS:
            d = root / fmt
            files = sorted(d.glob(f"*.{EXT[fmt]}"))
            ids[(payload, fmt)] = {f.stem for f in files}
            for f in files:
                text = f.read_text()
                m = tokens.measure(text, hf=use_hf)
                rows.append({"payload": payload, "fmt": fmt,
                             "record": f.stem, **m})
            print(f"measured {payload:<10}{fmt:<8}{len(files)} files")

    all_ids = set().union(*ids.values())
    missing = {f"{p}|{f}": sorted(all_ids - s)
               for (p, f), s in ids.items() if all_ids - s}

    # ------------------------------------------- structure, from the json dir
    structure: dict[str, dict] = {}
    for payload, root in CACHES.items():
        top, leaf = [], []
        for f in sorted((root / "json").glob("*.json")):
            doc = json.loads(f.read_text())
            top.append(len(doc))
            leaf.append(leaves(doc))
        structure[payload] = {"top_level_keys": summarize(top),
                              "leaf_values": summarize(leaf)}

    # ------------------------------------------------------------ aggregate
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["payload"], r["fmt"])].append(r)

    measures = ["bytes", "chars", "tiktoken"] + (["hf_tokens"] if use_hf else [])
    summary: dict[str, dict] = {}
    for (payload, fmt), group in sorted(grouped.items()):
        entry: dict = {"n": len(group)}
        for m in measures:
            entry[m] = summarize([r[m] for r in group])
        summary[f"{payload}|{fmt}"] = entry

    ratios = {
        fmt: {m: round(summary[f"unfaceted|{fmt}"][m]["total"]
                       / summary[f"faceted|{fmt}"][m]["total"], 2)
              for m in measures}
        for fmt in FORMATS
        if f"faceted|{fmt}" in summary and f"unfaceted|{fmt}" in summary
    }

    # -------------------------------------------------------------- persist
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "per_record.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["payload", "fmt", "record",
                                           "chars", "bytes", "tiktoken",
                                           "hf_tokens"])
        w.writeheader()
        w.writerows(rows)
    (args.out / "coverage.json").write_text(json.dumps({
        "records": len(all_ids),
        "files_measured": len(rows),
        "tokenizers": {"tiktoken": "o200k_base",
                       "hf": tokens.HF_TOKENIZER if use_hf else None},
        "completeness": {"complete": not missing, "missing": missing},
        "structure_from_json": structure,
        "summary": summary,
        "unfaceted_over_faceted": ratios,
    }, indent=2) + "\n")

    # --------------------------------------------------------------- report
    print(f"\nrecords: {len(all_ids)}   files measured: {len(rows)}")
    if missing:
        for k, v in missing.items():
            print(f"  ! {k} missing {len(v)}: {v[:5]} ...")
    else:
        print("completeness: every record present in all 12 directories")

    for m in measures:
        print(f"\n== {m} per record "
              f"(total / mean / median / p95 / max) ==")
        print(f"  {'payload':<11}{'fmt':<8}{'total':>13}{'mean':>10}"
              f"{'median':>9}{'p95':>9}{'max':>9}")
        for key, e in summary.items():
            payload, fmt = key.split("|")
            s = e[m]
            print(f"  {payload:<11}{fmt:<8}{s['total']:>13,}{s['mean']:>10,.0f}"
                  f"{s['median']:>9,}{s['p95']:>9,}{s['max']:>9,}")

    print("\n== unfaceted / faceted, whole-corpus ratios ==")
    print("  " + "".join(f"{c:>10}" for c in ["fmt", *measures]))
    for fmt, r in ratios.items():
        print(f"  {fmt:>10}" + "".join(f"{r[m]:>10}" for m in measures))

    print("\n== structure (json rendering) ==")
    for payload, s in structure.items():
        t, l = s["top_level_keys"], s["leaf_values"]
        print(f"  {payload:<11}top-level keys mean {t['mean']:>6}  "
              f"median {t['median']:>4}  max {t['max']:>5}   |   "
              f"leaf values mean {l['mean']:>8}  median {l['median']:>6}  "
              f"max {l['max']:>7}")

    print(f"\nwrote {args.out / 'per_record.csv'}")
    print(f"wrote {args.out / 'coverage.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
