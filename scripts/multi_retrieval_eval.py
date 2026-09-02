#!/usr/bin/env python
"""Retrieval-only format metrics for the multi-target query set.

Runs the ``airm.remote_eval`` machinery -- gates, search, Recall@k / MRR /
nDCG@10 scoring, per-format summary with bootstrap CIs, chart -- but against the
**local** full-cache index (``data/chroma/faceted_full``, 2,584 records,
gte-modernbert-base) instead of a deployed HTTP endpoint, and with the
multi-target query set (``data/queries_multi.jsonl``) instead of the 131/511
single-target sets.

Every query in that set expects 2-3 collections, so Recall@k here is genuine
*set* recall on every row -- the property the single-target synthetic sets only
had on their 11-query SME slice.

The index and the embedding model are exactly what ``airm.synth_multi`` used
for the retrievability gate, so "every expected id is retrievable in some
format's top-10" holds by construction; what this measures is *which* formats
surface them, how completely, and how early.

Usage:
    uv run python scripts/multi_retrieval_eval.py
    uv run python scripts/multi_retrieval_eval.py --limit 20        # smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airm import index, remote_eval  # noqa: E402
from airm.config import CHROMA_COLLECTION_PREFIX, EMBED_MODEL, RUNS_DIR  # noqa: E402
from airm.queries import load_queries  # noqa: E402

DEFAULT_DB = Path("data/chroma/faceted_full")
DEFAULT_QUERIES = Path("data/queries_multi.jsonl")
DEFAULT_RUN_ID = "MULTI500_FULL2584"


def local_backend(db_path: Path, embed_model: str = EMBED_MODEL) -> remote_eval.Backend:
    """A ``remote_eval.Backend`` over a local PersistentClient.

    Same encoder the collections were built with; the module's embedding-parity
    gate proves that rather than trusting it.
    """
    import chromadb

    if not db_path.exists():
        raise SystemExit(f"FAILED: no Chroma database at {db_path}")

    return remote_eval.Backend(
        client=chromadb.PersistentClient(path=str(db_path)),
        embed=lambda texts: index.embed(texts, embed_model, batch_size=16),
        count_tokens=lambda texts: index.count_wordpieces(texts, embed_model),
        window=index.max_sequence_tokens(embed_model),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--limit", type=int, default=None, help="first N queries (smoke)")
    parser.add_argument("--no-chart", action="store_true")
    args = parser.parse_args(argv)

    queries = load_queries(args.queries)
    if args.limit:
        queries = queries[: args.limit]

    sizes = sorted({len(q.expected_concept_ids) for q in queries})
    print(f"queries       : {len(queries)} from {args.queries} (expected ids per query: {sizes})")

    # ``host``/``port`` never see a socket -- the backend is injected below --
    # but the Endpoint still names the collections and labels the summary.
    endpoint = remote_eval.Endpoint(
        host=str(args.db), port=0, collection_prefix=CHROMA_COLLECTION_PREFIX, ssl=False
    )
    summary = remote_eval.run(
        endpoint=endpoint,
        backend=local_backend(args.db),
        queries=queries,
        embed_model=EMBED_MODEL,
        run_id=args.run_id,
        chart=not args.no_chart,
    )

    print(f"\nrun           : {summary['run_id']}")
    print(f"collections   : {summary['collection_counts']}")
    keys = [k for k in ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10")]
    print(f"\n  {'format':<10}" + "".join(f"{k:>10}" for k in keys))
    for cell_key, cell in summary["by_source"]["all"].items():
        fmt = cell_key.split("|")[0]
        print(f"  {fmt:<10}" + "".join(f"{cell[k]:>10.3f}" for k in keys))
    print(f"\nwrote {RUNS_DIR / summary['run_id']}/remote_retrieval_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
