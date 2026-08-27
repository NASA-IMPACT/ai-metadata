#!/usr/bin/env python
"""Format caches with the abstract *removed*: build, tokens, retrieval.

The third arm of the summary study, next to ``summary_append_eval.py``
(abstract + AI summary) and ``summary_replace_eval.py`` (AI summary instead of
abstract). Here the field is simply dropped and every format re-rendered from
the reduced payload:

    faceted    ``summary``  facet  removed
    unfaceted  ``Abstract`` field  removed

This is the ablation that says how much retrieval the abstract itself was
carrying. Read the three arms together: baseline (abstract), dropped (none),
replaced (summary), appended (both).

New caches (the originals are never touched):

    data/format_cache_no_summary/<fmt>/<id>.<ext>
    data/format_cache_unfaceted_no_abstract/<fmt>/<id>.<ext>

New chroma databases, built with the identical stack as the format study
(bge-large-en-v1.5, 510-token chunks / 64 overlap, max & mean pooling,
511 queries):

    data/chroma/dropped_summary_faceted/     (faceted, ``summary`` removed)
    data/chroma/dropped_abstract_unfaceted/  (unfaceted, ``Abstract`` removed)

Subcommands (run in order):
    cache    re-render every record without the abstract
    tokens   token counts per new cache vs the originals
    build    embed + index into the new chroma dbs (hours — run in background)
    eval     score the 511 queries, print vs the frozen baselines

Usage:
    uv run python scripts/abstract_drop_eval.py cache
    uv run python scripts/abstract_drop_eval.py tokens
    uv run python scripts/abstract_drop_eval.py build [--payloads faceted]
    uv run python scripts/abstract_drop_eval.py eval  [--pools max mean]
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

DROPPED = {"faceted": Path("data/format_cache_no_summary"),
           "unfaceted": Path("data/format_cache_unfaceted_no_abstract")}
DBS = {"faceted": Path("data/chroma/dropped_summary_faceted"),
       "unfaceted": Path("data/chroma/dropped_abstract_unfaceted")}
OUT_DIR = Path("runs/abstract_drop")
RUN_TAG = "abstract_drop"
CMR_CACHE = Path("data/cmr_cache")

# Point the shared machinery at this study's paths.
sae.PLUS = DROPPED
sae.DBS = DBS
sae.OUT_DIR = OUT_DIR
sae.RUN_TAG = RUN_TAG
sae.LABEL = "-abstract"


# --------------------------------------------------------------------------- #
def _render_faceted(record: dict) -> tuple[dict[str, str], dict]:
    payload = facets(record)
    payload.pop("summary", None)
    rendered = {fmt: formats.render(fmt, payload) for fmt in FORMATS}
    parity = formats.parity_report(payload)
    return rendered, {k: v for k, v in parity.items() if not v["ok"]}


def _render_unfaceted(record: dict) -> tuple[dict[str, str], dict]:
    p = copy.deepcopy(unfaceted.payload(record))
    p.pop("Abstract", None)
    rendered = {fmt: unfaceted.render(fmt, p) for fmt in FORMATS}
    return rendered, {}


RENDER = {"faceted": _render_faceted, "unfaceted": _render_unfaceted}


def cache(args) -> int:
    for payload in args.payloads:
        root = DROPPED[payload]
        for fmt in FORMATS:
            (root / fmt).mkdir(parents=True, exist_ok=True)
        written = {fmt: 0 for fmt in FORMATS}
        errors: list[dict] = []
        parity_failures: list[dict] = []
        cids = sorted(p.stem for p in CMR_CACHE.glob("*.json"))
        for cid in cids:
            record = cmr.load_cached(cid)
            if record is None:
                errors.append({"concept_id": cid, "error": "raw record missing"})
                continue
            try:
                rendered, failed = RENDER[payload](record)
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
              f"{len(parity_failures)}")
        (root / "manifest.json").write_text(json.dumps({
            **provenance(RUN_TAG),
            "payload": payload,
            "dropped_field": "summary" if payload == "faceted" else "Abstract",
            "source_records": str(CMR_CACHE),
            "original_cache": str(sae.ORIG[payload]),
            "records": len(cids),
            "written": written,
            "render_errors": errors,
            "parity_failures": parity_failures,
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
