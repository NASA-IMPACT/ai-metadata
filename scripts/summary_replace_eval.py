#!/usr/bin/env python
"""Format caches with the LLM summary *replacing* the abstract: build, tokens, retrieval.

The companion to ``summary_append_eval.py``. That script concatenated the
gpt-5.4-nano summary onto each rendering; this one *substitutes* it for the
record's own abstract and re-renders every format from the modified payload:

    faceted    ``summary``  facet  <- summary from data/format_cache_summary
    unfaceted  ``Abstract`` field  <- summary from data/format_cache_unfaceted_summary

Everything else in the record is untouched, so the renderings remain valid
JSON/YAML/TOON/CSV/JSON-LD/MaT and parity across formats still holds (it is
checked for the faceted payload exactly as the original cache build does).

New caches (the originals are never touched):

    data/format_cache_replace_summary/<fmt>/<id>.<ext>
    data/format_cache_unfaceted_replace_summary/<fmt>/<id>.<ext>

New chroma databases, built with the identical stack as the format study
(bge-large-en-v1.5, 510-token chunks / 64 overlap, max & mean pooling,
511 queries):

    data/chroma/no_summary_faceted/      (faceted, ``summary`` <- AI summary)
    data/chroma/no_abstract_unfaceted/   (unfaceted, ``Abstract`` <- AI summary)

Why this matters next to the append result: appending adds ~400 tokens per
record and therefore chunks, and max-pooling rewards chunk count. Replacing
holds length roughly constant (a 120-180-word summary vs a median 942-char
abstract), so any retrieval delta here is closer to a pure content effect.

Caveat: the summaries were generated from the record's MaT rendering, which
included the abstract, so the replacement is a rewrite of the abstract plus
the record's other fields, not new information.

Subcommands (run in order):
    cache    re-render every record with the summary in place of the abstract
    tokens   token counts per new cache vs the originals
    build    embed + index into the new chroma dbs (hours — run in background)
    eval     score the 511 queries, print vs the frozen baselines

Usage:
    uv run python scripts/summary_replace_eval.py cache
    uv run python scripts/summary_replace_eval.py tokens
    uv run python scripts/summary_replace_eval.py build [--payloads faceted]
    uv run python scripts/summary_replace_eval.py eval  [--pools max mean]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from airm import cmr, formats, unfaceted  # noqa: E402
from airm.config import FORMATS  # noqa: E402
from airm.facets import facets  # noqa: E402
from airm.provenance import provenance  # noqa: E402

import summary_append_eval as sae  # noqa: E402  (shared tokens/build/eval)

REPLACE = {"faceted": Path("data/format_cache_replace_summary"),
           "unfaceted": Path("data/format_cache_unfaceted_replace_summary")}
DBS = {"faceted": Path("data/chroma/no_summary_faceted"),
       "unfaceted": Path("data/chroma/no_abstract_unfaceted")}
OUT_DIR = Path("runs/summary_replace")
RUN_TAG = "summary_replace"
CMR_CACHE = Path("data/cmr_cache")

# Point the shared machinery at this study's paths. The append script's
# tokens/build/eval read these module globals; nothing else differs.
sae.PLUS = REPLACE
sae.DBS = DBS
sae.OUT_DIR = OUT_DIR
sae.RUN_TAG = RUN_TAG
sae.LABEL = "~summary"


# --------------------------------------------------------------------------- #
def _render_faceted(record: dict, summary: str) -> tuple[dict[str, str], dict]:
    payload = facets(record)
    payload["summary"] = summary
    rendered = {fmt: formats.render(fmt, payload) for fmt in FORMATS}
    parity = formats.parity_report(payload)
    return rendered, {k: v for k, v in parity.items() if not v["ok"]}


def _render_unfaceted(record: dict, summary: str) -> tuple[dict[str, str], dict]:
    p = copy.deepcopy(unfaceted.payload(record))
    p["Abstract"] = summary
    rendered = {fmt: unfaceted.render(fmt, p) for fmt in FORMATS}
    return rendered, {}


RENDER = {"faceted": _render_faceted, "unfaceted": _render_unfaceted}


def cache(args) -> int:
    for payload in args.payloads:
        summaries = {p.stem: p.read_text().strip()
                     for p in sae.SUMMARY[payload].glob("*.txt")}
        root = REPLACE[payload]
        for fmt in FORMATS:
            (root / fmt).mkdir(parents=True, exist_ok=True)
        written = {fmt: 0 for fmt in FORMATS}
        errors: list[dict] = []
        parity_failures: list[dict] = []
        replaced_placeholder = 0
        cids = sorted(p.stem for p in CMR_CACHE.glob("*.json"))
        for cid in cids:
            s = summaries.get(cid)
            if not s:
                print(f"  ! no summary for {cid}, skipped")
                continue
            record = cmr.load_cached(cid)
            if record is None:
                errors.append({"concept_id": cid, "error": "raw record missing"})
                continue
            original = (unfaceted.payload(record).get("Abstract") or "").strip()
            if original.lower() in {"not provided", "not available.", ""}:
                replaced_placeholder += 1
            try:
                rendered, failed = RENDER[payload](record, s)
            except Exception as exc:  # noqa: BLE001 - one bad record must not stop the build
                errors.append({"concept_id": cid, "error": f"{type(exc).__name__}: {exc}"})
                continue
            if failed:
                parity_failures.append({"concept_id": cid, "formats": failed})
            for fmt, text in rendered.items():
                (root / fmt / f"{cid}.{sae.EXT[fmt]}").write_text(text)
                written[fmt] += 1
        for fmt in FORMATS:
            print(f"{payload:<10}{fmt:<8}{written[fmt]} files -> {root / fmt}")
        print(f"{payload:<10}render errors {len(errors)}, parity failures "
              f"{len(parity_failures)}, placeholder abstracts replaced "
              f"{replaced_placeholder}")
        (root / "manifest.json").write_text(json.dumps({
            **provenance(RUN_TAG),
            "payload": payload,
            "replaced_field": "summary" if payload == "faceted" else "Abstract",
            "source_records": str(CMR_CACHE),
            "summary_cache": str(sae.SUMMARY[payload]),
            "original_cache": str(sae.ORIG[payload]),
            "records": len(cids),
            "written": written,
            "render_errors": errors,
            "parity_failures": parity_failures,
            "placeholder_abstracts_replaced": replaced_placeholder,
        }, indent=2) + "\n")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in (("cache", cache), ("tokens", sae.tokens),
                     ("build", sae.build), ("eval", sae.evaluate)):
        p = sub.add_parser(name)
        p.add_argument("--payloads", nargs="+",
                       default=["faceted", "unfaceted"],
                       choices=["faceted", "unfaceted"])
        if name == "eval":
            p.add_argument("--pools", nargs="+", default=["max", "mean"],
                           choices=["max", "mean"])
        p.set_defaults(func=fn)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
