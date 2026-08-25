#!/usr/bin/env python
"""LLM-written summaries as a seventh representation — generation + retrieval.

The study's ``mat`` rendering is deterministic template prose. This script
tests the next step: have ``gpt-5.4-nano`` *write* a dense, retrieval-oriented
summary of each record, for both payloads, and measure whether chunked
retrieval improves over the template prose.

Two subcommands:

``generate``
    For every record, summarise its cached MaT rendering
    (``data/format_cache{,_unfaceted}/mat/<id>.txt``) with gpt-5.4-nano at
    temperature 0 and write the result to NEW caches:

        data/format_cache_summary/<id>.txt              (from faceted MaT)
        data/format_cache_unfaceted_summary/<id>.txt    (from raw-record MaT)

    Resumable: existing files are skipped. Every call is audit-logged via
    ``airm.llm.CallLogger``; a ``manifest.json`` per cache records the model
    and prompt hash.

``eval``
    Chunked retrieval over the summaries with the *identical* stack the
    format study used — bge-large-en-v1.5, 510-token chunks / 64 overlap, max
    (and mean) pooling, the 511-query set — and prints the result next to the
    frozen ``mat`` baseline from runs 20260814T205849Z / 20260814T205912Z.
    Index: ``data/chroma/summary_chunked``. Results:
    ``runs/summary_retrieval/summary_eval.json``.

Usage:
    uv run python scripts/summary_retrieval.py generate               # both payloads
    uv run python scripts/summary_retrieval.py generate --limit 25    # smoke test
    uv run python scripts/summary_retrieval.py eval
    uv run python scripts/summary_retrieval.py eval --rebuild
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from airm.llm import CallLogger, LLMError, ModelSpec, complete, resolve_models  # noqa: E402
from airm.provenance import provenance  # noqa: E402
from airm.queries import load_queries  # noqa: E402

import unfaceted_chunked_eval as uce  # noqa: E402  (shared encoder/chunk/pool stack)

SOURCE_MAT = {
    "faceted": Path("data/format_cache/mat"),
    "unfaceted": Path("data/format_cache_unfaceted/mat"),
}
SUMMARY_DIRS = {
    "faceted": Path("data/format_cache_summary"),
    "unfaceted": Path("data/format_cache_unfaceted_summary"),
}
#: The frozen 511-query max-pool baselines to compare against.
BASELINE_RUNS = {"faceted": "20260814T205849Z", "unfaceted": "20260814T205912Z"}
QUERIES_PATH = Path("data/queries_full.jsonl")
DB_DIR = Path("data/chroma/summary_chunked")
OUT_DIR = Path("runs/summary_retrieval")

SYSTEM = ("You write dense, factual prose summaries of Earth-science dataset "
          "metadata records, optimized for semantic-search retrieval.")
PROMPT = """Summarize the following dataset metadata record as ONE dense prose paragraph of roughly 120-180 words.

Rules:
- Preserve exactly: the dataset title, short name, organisation and data-center names, platform and instrument names, variable and science-keyword terms, and spatial/temporal coverage values.
- Plain prose only: no markdown, no lists, no headings.
- Use only facts stated in the record; never add outside knowledge.
- Skip placeholders ("Not provided", "Unknown", absent DOIs) and say nothing about missing information.

RECORD:
{record}"""


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #


def generate(args) -> int:
    spec = ModelSpec(*args.model.split(":", 1))
    usable, problems = resolve_models([spec])
    if not usable:
        raise SystemExit("FAILED: model unavailable: " + "; ".join(problems))
    logger = CallLogger()
    prompt_sha = hashlib.sha256((SYSTEM + PROMPT).encode()).hexdigest()[:16]

    for payload in args.payloads:
        src, out = SOURCE_MAT[payload], SUMMARY_DIRS[payload]
        out.mkdir(parents=True, exist_ok=True)
        todo = [p for p in sorted(src.glob("*.txt"))
                if not (out / p.name).exists()]
        if args.limit:
            todo = todo[: args.limit]
        print(f"{payload}: {len(todo)} to summarise "
              f"({len(list(out.glob('*.txt')))} already present)")

        done = 0
        cost = 0.0
        started = time.time()

        def one(path: Path):
            text = path.read_text()
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user",
                         "content": PROMPT.format(record=text)}]
            resp = complete(spec, messages, purpose="summary", logger=logger,
                            temperature=0.0, payload=payload,
                            record=path.stem)
            summary = resp.text.strip()
            if not summary:
                raise LLMError(f"empty summary for {path.stem}")
            (out / path.name).write_text(summary + "\n")
            return resp

        with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            futures = {pool.submit(one, p): p for p in todo}
            for fut in concurrent.futures.as_completed(futures):
                p = futures[fut]
                try:
                    resp = fut.result()
                    cost += resp.cost_usd or 0.0
                except LLMError as exc:
                    print(f"  ! {p.stem}: {exc}")
                    continue
                done += 1
                if done % 100 == 0:
                    rate = done / (time.time() - started)
                    print(f"  {done}/{len(todo)}  ({rate:.1f}/s, "
                          f"eta {(len(todo) - done) / rate / 60:.0f}m)")

        (out / "manifest.json").write_text(json.dumps({
            **provenance(logger.run_id),
            "model": f"{spec.provider}:{spec.model}",
            "prompt_sha256_16": prompt_sha,
            "source": str(src),
            "records": len(list(out.glob("*.txt"))),
            "call_log": str(logger.path_for("summary")),
        }, indent=2) + "\n")
        print(f"{payload}: wrote {done} summaries "
              f"(cost this pass ${cost:.2f})  -> {out}")
    return 0


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #


def build_index(client, payload: str, model) -> dict:
    name = f"summary_{payload}"
    try:
        client.delete_collection(name)
    except Exception:  # noqa: BLE001 - absent on first build
        pass
    collection = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    tokenizer = model.tokenizer

    files = sorted(SUMMARY_DIRS[payload].glob("*.txt"))
    ids, texts, metas, per_record = [], [], [], []
    for f in files:
        pieces = uce.chunk(f.read_text(), tokenizer, uce.CHUNK_TOKENS,
                           uce.OVERLAP)
        per_record.append(len(pieces))
        for i, piece in enumerate(pieces):
            ids.append(f"{f.stem}#{i}")
            texts.append(piece)
            metas.append({"concept_id": f.stem, "format": "summary",
                          "chunk": i, "n_chunks": len(pieces)})
    for start in range(0, len(ids), 256):
        end = start + 256
        collection.add(ids=ids[start:end], documents=texts[start:end],
                       metadatas=metas[start:end],
                       embeddings=uce.embed(model, texts[start:end]))
    return {"records": len(files), "chunks": len(ids),
            "chunks_per_record_mean": round(statistics.fmean(per_record), 2),
            "chunks_per_record_max": max(per_record)}


def baseline_mat(payload: str) -> dict | None:
    path = Path("runs") / BASELINE_RUNS[payload] / "chunked_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["by_source"]["all"]["mat"]


def evaluate(args) -> int:
    import chromadb

    client = chromadb.PersistentClient(path=str(DB_DIR))
    model = uce.embedder()

    stats = {}
    for payload in ("faceted", "unfaceted"):
        if not SUMMARY_DIRS[payload].exists():
            raise SystemExit(f"FAILED: no summaries at {SUMMARY_DIRS[payload]}"
                             " — run `generate` first")
        need = args.rebuild
        if not need:
            try:
                need = client.get_collection(f"summary_{payload}").count() == 0
            except Exception:  # noqa: BLE001 - collection absent
                need = True
        if need:
            print(f"building index: summary_{payload} ...")
            stats[payload] = build_index(client, payload, model)
            print(f"  {stats[payload]}")

    queries = load_queries(QUERIES_PATH)
    vectors = uce.embed(model, [q.text for q in queries])

    from airm.evaluate import retrieval_metrics

    results: dict[str, dict] = {}
    rows: list[dict] = []
    for payload in ("faceted", "unfaceted"):
        collection = client.get_collection(f"summary_{payload}")
        for pool_rule in args.pools:
            cells = []
            for query, vector in zip(queries, vectors):
                got = collection.query(query_embeddings=[vector],
                                       n_results=10 * uce.OVERSAMPLE,
                                       include=["distances", "metadatas"])
                pooled = uce.pool(got["ids"][0], got["distances"][0],
                                  got["metadatas"][0], pool_rule)
                ranked = [cid for cid, _ in pooled]
                metrics = retrieval_metrics(ranked, query.expected_concept_ids,
                                            ndcg_k=uce.NDCG_K)
                cells.append(metrics)
                rows.append({"payload": payload, "pool": pool_rule,
                             "query_id": query.query_id,
                             "source": query.source, **metrics,
                             "retrieved": "|".join(ranked[:10])})
            results[f"{payload}|{pool_rule}"] = {
                k: round(statistics.fmean(c[k] for c in cells), 4)
                for k in cells[0]}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary_eval.json").write_text(json.dumps({
        **provenance("summary_retrieval"),
        "queries": len(queries),
        "embed_model": uce.EMBED_MODEL,
        "chunk_tokens": uce.CHUNK_TOKENS, "overlap": uce.OVERLAP,
        "index_stats": stats,
        "results": results,
        "baseline_mat_max": {p: baseline_mat(p)
                             for p in ("faceted", "unfaceted")},
    }, indent=2) + "\n")
    import csv as _csv
    with (OUT_DIR / "summary_rows.csv").open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    keys = ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]
    print(f"\n== LLM summary vs template MaT — {len(queries)} queries ==")
    print(f"  {'arm':<38}" + "".join(f"{k:>10}" for k in keys))
    for payload in ("faceted", "unfaceted"):
        base = baseline_mat(payload)
        if base:
            print(f"  {payload + ' mat (template, max)':<38}"
                  + "".join(f"{base[k]:>10.3f}" for k in keys))
        for pool_rule in args.pools:
            r = results[f"{payload}|{pool_rule}"]
            print(f"  {payload + f' summary (nano, {pool_rule})':<38}"
                  + "".join(f"{r[k]:>10.3f}" for k in keys))
        if base:
            r = results[f"{payload}|max"]
            print(f"  {'  delta (summary max - mat max)':<38}"
                  + "".join(f"{r[k] - base[k]:>+10.3f}" for k in keys))
    print(f"\nwrote {OUT_DIR / 'summary_eval.json'}")
    print(f"wrote {OUT_DIR / 'summary_rows.csv'}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="summarise records with gpt-5.4-nano")
    g.add_argument("--model", default="openai:gpt-5.4-nano")
    g.add_argument("--payloads", nargs="+", default=["faceted", "unfaceted"],
                   choices=["faceted", "unfaceted"])
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--limit", type=int, default=None,
                   help="cap records per payload (smoke test)")
    g.set_defaults(func=generate)

    e = sub.add_parser("eval", help="chunked retrieval over the summaries")
    e.add_argument("--rebuild", action="store_true",
                   help="rebuild the chroma collections")
    e.add_argument("--pools", nargs="+", default=["max", "mean"],
                   choices=["max", "mean"])
    e.set_defaults(func=evaluate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
