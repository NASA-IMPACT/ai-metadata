#!/usr/bin/env python
"""Format + LLM-summary appended caches: build, token counts, retrieval.

Takes the gpt-5.4-nano summaries in ``data/format_cache_summary`` (faceted
source) and ``data/format_cache_unfaceted_summary`` (raw-record source) and
appends each record's summary to each of its six format renderings, producing
two NEW caches (the originals are never touched):

    data/format_cache_plus_summary/<fmt>/<id>.<ext>
    data/format_cache_unfaceted_plus_summary/<fmt>/<id>.<ext>

The combined document is for *embedding*, not parsing — appending prose to a
JSON rendering deliberately breaks its syntax; retrieval only sees text.

Chunked retrieval runs with the identical stack as the format study
(bge-large-en-v1.5, 510-token chunks / 64 overlap, max & mean pooling, the
511-query set) against NEW chroma databases:

    data/chroma/faceted_plus_summary_chunked/
    data/chroma/unfaceted_plus_summary_chunked/

Caveat, stated up front: appending text adds chunks, and under max-pooling
more chunks means more independent chances to match — some of any gain is
bought by length. ``tokens`` measures exactly how much longer each format got;
read it next to the retrieval delta.

Subcommands (run in order):
    cache    build the appended caches (fast, pure file concatenation)
    tokens   token/byte counts per new cache vs the originals
    build    embed + index into the new chroma dbs (hours — run in background)
    eval     score the 511 queries, print vs the frozen baselines

Usage:
    uv run python scripts/summary_append_eval.py cache
    uv run python scripts/summary_append_eval.py tokens
    uv run python scripts/summary_append_eval.py build [--payloads faceted]
    uv run python scripts/summary_append_eval.py eval  [--pools max mean]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from airm.config import FORMATS  # noqa: E402
from airm.evaluate import retrieval_metrics  # noqa: E402
from airm.provenance import provenance  # noqa: E402
from airm.queries import load_queries  # noqa: E402
from airm.tokens import count_tiktoken  # noqa: E402

import unfaceted_chunked_eval as uce  # noqa: E402  (shared encoder/chunk/pool)

EXT = {"json": "json", "csv": "csv", "yaml": "yaml", "toon": "toon",
       "jsonld": "jsonld", "mat": "txt"}
ORIG = {"faceted": Path("data/format_cache"),
        "unfaceted": Path("data/format_cache_unfaceted")}
SUMMARY = {"faceted": Path("data/format_cache_summary"),
           "unfaceted": Path("data/format_cache_unfaceted_summary")}
PLUS = {"faceted": Path("data/format_cache_plus_summary"),
        "unfaceted": Path("data/format_cache_unfaceted_plus_summary")}
DBS = {"faceted": Path("data/chroma/faceted_plus_summary_chunked"),
       "unfaceted": Path("data/chroma/unfaceted_plus_summary_chunked")}
BASELINE_RUNS = {"faceted": "20260814T205849Z", "unfaceted": "20260814T205912Z"}
QUERIES_PATH = Path("data/queries_full.jsonl")
OUT_DIR = Path("runs/summary_append")
SEPARATOR = "\n\nSummary: "
RUN_TAG = "summary_append"   # provenance tag; overridden by summary_replace_eval
LABEL = "+summary"          # payload label in index reports


# --------------------------------------------------------------------------- #
def cache(args) -> int:
    for payload in args.payloads:
        summaries = {p.stem: p.read_text().strip()
                     for p in SUMMARY[payload].glob("*.txt")}
        for fmt in FORMATS:
            src = ORIG[payload] / fmt
            out = PLUS[payload] / fmt
            out.mkdir(parents=True, exist_ok=True)
            n = 0
            for f in sorted(src.glob(f"*.{EXT[fmt]}")):
                s = summaries.get(f.stem)
                if s is None:
                    print(f"  ! no summary for {f.stem}, skipped")
                    continue
                (out / f.name).write_text(
                    f.read_text().rstrip("\n") + SEPARATOR + s + "\n")
                n += 1
            print(f"{payload:<10}{fmt:<8}{n} files -> {out}")
        (PLUS[payload] / "manifest.json").write_text(json.dumps({
            **provenance(RUN_TAG),
            "source_cache": str(ORIG[payload]),
            "summary_cache": str(SUMMARY[payload]),
            "separator": SEPARATOR,
        }, indent=2) + "\n")
    return 0


# --------------------------------------------------------------------------- #
def tokens(args) -> int:
    report: dict[str, dict] = {}
    print(f"{'payload':<11}{'fmt':<8}{'orig mean':>11}{'plus mean':>11}"
          f"{'delta':>8}{'plus total':>13}")
    for payload in args.payloads:
        for fmt in FORMATS:
            orig = [count_tiktoken(p.read_text())
                    for p in sorted((ORIG[payload] / fmt).glob(f"*.{EXT[fmt]}"))]
            plus = [count_tiktoken(p.read_text())
                    for p in sorted((PLUS[payload] / fmt).glob(f"*.{EXT[fmt]}"))]
            entry = {
                "n": len(plus),
                "orig_mean": round(statistics.fmean(orig), 1),
                "plus_mean": round(statistics.fmean(plus), 1),
                "delta_mean": round(statistics.fmean(plus) - statistics.fmean(orig), 1),
                "plus_total": sum(plus),
                "orig_total": sum(orig),
            }
            report[f"{payload}|{fmt}"] = entry
            print(f"{payload:<11}{fmt:<8}{entry['orig_mean']:>11,.0f}"
                  f"{entry['plus_mean']:>11,.0f}{entry['delta_mean']:>+8,.0f}"
                  f"{entry['plus_total']:>13,}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "tokens.json").write_text(json.dumps(
        {**provenance(RUN_TAG), "tokenizer": "o200k_base",
         "cells": report}, indent=2) + "\n")
    print(f"\nwrote {OUT_DIR / 'tokens.json'}")
    return 0


# --------------------------------------------------------------------------- #
def build(args) -> int:
    import chromadb

    model = uce.embedder()
    tokenizer = model.tokenizer
    for payload in args.payloads:
        db = DBS[payload]
        client = chromadb.PersistentClient(path=str(db))
        stats: dict[str, dict] = {}
        started = time.time()
        print(f"building {db} ...")
        for fmt in FORMATS:
            name = f"cmr_{fmt}"
            try:
                client.delete_collection(name)
            except Exception:  # noqa: BLE001 - absent on first build
                pass
            collection = client.create_collection(
                name, metadata={"hnsw:space": "cosine"})
            files = sorted((PLUS[payload] / fmt).glob(f"*.{EXT[fmt]}"))
            ids, texts, metas, per_record = [], [], [], []
            for f in files:
                pieces = uce.chunk(f.read_text(), tokenizer,
                                   uce.CHUNK_TOKENS, uce.OVERLAP)
                per_record.append(len(pieces))
                for i, piece in enumerate(pieces):
                    ids.append(f"{f.stem}#{i}")
                    texts.append(piece)
                    metas.append({"concept_id": f.stem, "format": fmt,
                                  "chunk": i, "n_chunks": len(pieces)})
            fmt_started = time.time()
            for start in range(0, len(ids), 256):
                end = start + 256
                collection.add(ids=ids[start:end], documents=texts[start:end],
                               metadatas=metas[start:end],
                               embeddings=uce.embed(model, texts[start:end]))
            stats[fmt] = {
                "records": len(per_record), "chunks": len(ids),
                "chunks_per_record_mean": round(statistics.fmean(per_record), 2),
                "seconds": round(time.time() - fmt_started, 1),
            }
            print(f"  {fmt:<8}{len(ids):>7,} chunks  "
                  f"{stats[fmt]['chunks_per_record_mean']:>6.2f}/record  "
                  f"{stats[fmt]['seconds']:>7.1f}s", flush=True)
        (db / "chunked_index_report.json").write_text(json.dumps({
            **provenance(RUN_TAG),
            "payload": f"{payload}{LABEL}",
            "cache": str(PLUS[payload]),
            "embed_model": uce.EMBED_MODEL,
            "chunk_tokens": uce.CHUNK_TOKENS, "overlap": uce.OVERLAP,
            "stats": stats,
            "seconds": round(time.time() - started, 1),
        }, indent=2) + "\n")
        print(f"  done in {(time.time() - started) / 60:.1f} min")
    return 0


# --------------------------------------------------------------------------- #
def baseline(payload: str) -> dict:
    path = Path("runs") / BASELINE_RUNS[payload] / "chunked_summary.json"
    return json.loads(path.read_text())["by_source"]["all"] if path.exists() else {}


def evaluate(args) -> int:
    import chromadb

    queries = load_queries(QUERIES_PATH)
    model = uce.embedder()
    vectors = uce.embed(model, [q.text for q in queries])
    keys = ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]

    all_results: dict[str, dict] = {}
    rows: list[dict] = []
    for payload in args.payloads:
        db = DBS[payload]
        if not (db / "chunked_index_report.json").exists():
            print(f"skipping {payload}: no index at {db} (run `build`)")
            continue
        client = chromadb.PersistentClient(path=str(db))
        base = baseline(payload)
        print(f"\n== {payload} + summary — {len(queries)} queries ==")
        print(f"  {'fmt':<8}{'pool':<6}" + "".join(f"{k:>10}" for k in keys)
              + f"{'Δr@10 vs base':>15}")
        for fmt in FORMATS:
            collection = client.get_collection(f"cmr_{fmt}")
            for pool_rule in args.pools:
                cells = []
                for query, vector in zip(queries, vectors):
                    got = collection.query(query_embeddings=[vector],
                                           n_results=10 * uce.OVERSAMPLE,
                                           include=["distances", "metadatas"])
                    pooled = uce.pool(got["ids"][0], got["distances"][0],
                                      got["metadatas"][0], pool_rule)
                    ranked = [cid for cid, _ in pooled]
                    m = retrieval_metrics(ranked, query.expected_concept_ids,
                                          ndcg_k=uce.NDCG_K)
                    cells.append(m)
                    rows.append({"payload": payload, "format": fmt,
                                 "pool": pool_rule, "query_id": query.query_id,
                                 "source": query.source, **m})
                agg = {k: round(statistics.fmean(c[k] for c in cells), 4)
                       for k in cells[0]}
                all_results[f"{payload}|{fmt}|{pool_rule}"] = agg
                delta = (f"{agg['recall@10'] - base[fmt]['recall@10']:>+15.3f}"
                         if pool_rule == "max" and fmt in base else f"{'—':>15}")
                print(f"  {fmt:<8}{pool_rule:<6}"
                      + "".join(f"{agg[k]:>10.3f}" for k in keys) + delta)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "retrieval_eval.json").write_text(json.dumps({
        **provenance(RUN_TAG),
        "queries": len(queries),
        "results": all_results,
        "baselines_max": {p: baseline(p) for p in args.payloads},
    }, indent=2) + "\n")
    import csv as _csv
    with (OUT_DIR / "retrieval_rows.csv").open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT_DIR / 'retrieval_eval.json'}")
    print(f"wrote {OUT_DIR / 'retrieval_rows.csv'}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in (("cache", cache), ("tokens", tokens),
                     ("build", build), ("eval", evaluate)):
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
