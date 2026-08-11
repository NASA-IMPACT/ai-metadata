#!/usr/bin/env python
"""Chunked retrieval over the **unfaceted** renderings, with the deployment's vectoriser.

Independent of the study's main line. `airm.index` builds one document per record
into `data/chroma/{faceted,unfaceted}/` with `Alibaba-NLP/gte-modernbert-base`
(8,192-token window), because that window was chosen precisely so nothing is ever
clipped and nothing ever needs chunking.

This script does the opposite, deliberately:

* **Payload** — the unfaceted renderings (`data/format_cache_unfaceted/`), the raw
  UMM record. These run to 55,000 tokens and genuinely do not fit any embedding
  window.
* **Vectoriser** — `BAAI/bge-large-en-v1.5`, 1024 dimensions, **512-token window**,
  matching what `usage.md` documents for the deployed endpoint.
* **Chunking** — fixed-size token windows with overlap, one vector per chunk,
  pooled back to a record score at query time.

## The bias this introduces, stated up front

Chunking is not neutral in a *format* comparison. The six renderings carry
identical facts at different lengths (unfaceted: prose 1,532 tokens, JSON-LD
4,421). Chunk at a fixed size and each format gets a different **number** of
chunks for the same content, so a verbose format gets more independent chances to
match the query. Under max-pooling that is a direct advantage bought by verbosity
-- the opposite of what the study is trying to price.

A second, subtler bias: chunk boundaries fall differently by format. JSON cut
mid-object is syntactically meaningless; prose cut mid-paragraph still reads.

Neither is fixable by tuning. So this script **measures and reports** both --
`chunks_per_record` per format sits next to every score, and the summary carries
the caveat. Read the numbers as "how does chunked retrieval behave over full
records", not as a clean format ranking.

## Usage

    # 1. Build. Downloads bge-large-en-v1.5 (~1.34 GB) on first run.
    uv run python scripts/unfaceted_chunked_eval.py build

    # Smoke test first -- 50 records, a couple of minutes:
    uv run python scripts/unfaceted_chunked_eval.py build --limit 50

    # 2. Evaluate against the 131-query set. Writes rows, summary and a chart.
    uv run python scripts/unfaceted_chunked_eval.py eval

    # 3. Redraw from summaries already on disk -- no retrieval, no model load.
    #    Several runs become one figure, a panel each, for comparing pooling.
    uv run python scripts/unfaceted_chunked_eval.py plot chunked_max chunked_mean

    # Options
    --chunk-tokens 510      content tokens per chunk (512 minus [CLS]/[SEP])
    --overlap 64            token overlap between consecutive chunks
    --db data/chroma/unfaceted_chunked
    --pool max|mean         chunk -> record score (default max)
    --query-instruction     prepend bge's retrieval instruction to queries
    --no-chart              skip the figure
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airm import cmr, index, unfaceted  # noqa: E402
from airm.config import FORMATS, FORMAT_LABELS, RECALL_AT, RUNS_DIR, TOP_K, new_run_id  # noqa: E402
from airm.evaluate import retrieval_metrics  # noqa: E402
from airm.facets import facets  # noqa: E402
from airm.provenance import provenance  # noqa: E402
from airm.queries import ground_truth_ids, load_eval_queries, load_queries  # noqa: E402


def queries_from(path: str | None):
    """The evaluation set, from ``--queries`` when given.

    Kept as one helper so ``build``'s ground-truth check and ``eval``'s scoring
    can never end up looking at different query sets.
    """
    return load_queries(Path(path)) if path else load_eval_queries()

#: What ``usage.md`` says the deployed collections were built with.
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
WINDOW = 512

#: 512 minus [CLS] and [SEP].
CHUNK_TOKENS = 510
OVERLAP = 64

#: One database per payload, so a faceted build can never overwrite an unfaceted
#: one and ``eval`` cannot silently score the wrong corpus. Collection names stay
#: ``cmr_<fmt>`` in both, matching ``index.chroma_dir``'s reasoning.
def default_db(payload: str) -> str:
    return f"data/chroma/{payload}_chunked"


COLLECTION_PREFIX = "cmr_"

#: bge prefixes *queries* only; documents are embedded bare.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

EMBED_BATCH = 32
NDCG_K = 10

#: Chunks fetched per query before pooling. A record can occupy many chunks, so
#: fetching only ``k`` chunks could return ``k`` chunks of one record and leave
#: the top-k *records* undetermined.
OVERSAMPLE = 12


# --------------------------------------------------------------------------- #
# Model.
# --------------------------------------------------------------------------- #


def embedder():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    model.max_seq_length = WINDOW
    return model


def embed(model, texts: list[str]) -> list[list[float]]:
    vectors = model.encode(
        texts, batch_size=EMBED_BATCH, normalize_embeddings=True, show_progress_bar=False
    )
    return [v.tolist() for v in vectors]


# --------------------------------------------------------------------------- #
# Chunking.
# --------------------------------------------------------------------------- #


def chunk(text: str, tokenizer, size: int, overlap: int) -> list[str]:
    """Fixed-size token windows with overlap, returned as text.

    Overlap exists so a fact straddling a boundary survives in at least one
    chunk whole. Without it, a title split across two chunks is in neither.

    Chunks are cut on token ids and decoded back to text because Chroma stores
    documents as text and the encoder re-tokenises anyway; the round trip can
    shift the count by a token or two, which the model's own truncation absorbs.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= size:
        return [text]
    stride = max(1, size - overlap)
    out: list[str] = []
    for start in range(0, len(ids), stride):
        piece = ids[start : start + size]
        if not piece:
            break
        out.append(tokenizer.decode(piece, skip_special_tokens=True))
        if start + size >= len(ids):
            break
    return out


# --------------------------------------------------------------------------- #
# Build.
# --------------------------------------------------------------------------- #


def select_records(limit: int | None) -> tuple[list[dict], dict]:
    """The same 2,584 records the faceted full-cache index uses.

    ``index.all_cached_records`` drops one record for content parity and five for
    exceeding the *gte* 8,192-token window. That second exclusion is irrelevant
    here -- chunking removes any window constraint -- but the set is kept
    identical so this index is comparable with the faceted one record for record.
    Excluding them costs 5 of 2,589 and buys a like-for-like corpus.
    """
    records, report = index.all_cached_records()
    if limit:
        records = records[:limit]
        report["limited_to"] = limit
    return records, report


def build(args) -> int:
    import chromadb

    records, selection = select_records(args.limit)
    print(f"payload: {args.payload}")
    print(f"records: {len(records)} (examined {selection['examined']}, "
          f"parity failures {len(selection['parity_failures'])}, "
          f"over gte window {len(selection['over_window'])})")

    model = embedder()
    tokenizer = model.tokenizer
    print(f"vectoriser: {EMBED_MODEL}  window={WINDOW}  "
          f"chunk={args.chunk_tokens} overlap={args.overlap}\n")

    db = Path(args.db or default_db(args.payload))
    db.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db))

    # The same projection ``index.build`` applies, so the only difference between
    # this index and the study's is the chunking and the vectoriser.
    project = facets if args.payload == index.FACETED else unfaceted.payload
    payloads = [(cmr.concept_id(r), project(r)) for r in records]
    stats: dict[str, dict] = {}
    started = time.time()

    for fmt in FORMATS:
        name = f"{COLLECTION_PREFIX}{fmt}"
        try:
            client.delete_collection(name)
        except Exception:  # noqa: BLE001 - absent on a first build
            pass
        collection = client.create_collection(name, metadata={"hnsw:space": "cosine"})

        docs = index._documents(args.payload, fmt, payloads)
        ids: list[str] = []
        texts: list[str] = []
        metas: list[dict] = []
        per_record: list[int] = []

        for (cid, _), doc in zip(payloads, docs):
            pieces = chunk(doc, tokenizer, args.chunk_tokens, args.overlap)
            per_record.append(len(pieces))
            for i, piece in enumerate(pieces):
                ids.append(f"{cid}#{i}")
                texts.append(piece)
                metas.append({"concept_id": cid, "format": fmt, "chunk": i,
                              "n_chunks": len(pieces)})

        fmt_started = time.time()
        for start in range(0, len(ids), 256):
            end = start + 256
            collection.add(
                ids=ids[start:end],
                documents=texts[start:end],
                metadatas=metas[start:end],
                embeddings=embed(model, texts[start:end]),
            )
            index._release_accelerator_memory()

        stats[fmt] = {
            "records": len(per_record),
            "chunks": len(ids),
            "chunks_per_record_mean": round(statistics.fmean(per_record), 2),
            "chunks_per_record_median": statistics.median(per_record),
            "chunks_per_record_max": max(per_record),
            "seconds": round(time.time() - fmt_started, 1),
        }
        print(f"  {FORMAT_LABELS.get(fmt, fmt):<18} {len(ids):>7,} chunks  "
              f"{stats[fmt]['chunks_per_record_mean']:>6.2f}/record  "
              f"{stats[fmt]['seconds']:>7.1f}s")

    gt = ground_truth_ids(queries_from(args.queries))
    missing = {}
    for fmt in FORMATS:
        col = client.get_collection(f"{COLLECTION_PREFIX}{fmt}")
        present = set()
        for start in range(0, len(gt), 200):
            got = col.get(where={"concept_id": {"$in": gt[start:start + 200]}}, include=["metadatas"])
            present |= {m["concept_id"] for m in got["metadatas"]}
        missing[fmt] = sorted(set(gt) - present)

    report = {
        **provenance(),
        "payload": args.payload,
        "db": str(db),
        "embed_model": EMBED_MODEL,
        "window": WINDOW,
        "chunk_tokens": args.chunk_tokens,
        "overlap": args.overlap,
        "records": len(records),
        "by_format": stats,
        "ground_truth_ids": len(gt),
        "ground_truth_missing": {f: m for f, m in missing.items() if m},
        "total_seconds": round(time.time() - started, 1),
        "selection": {k: v for k, v in selection.items() if k != "parity_failures"},
        "caveat": chunk_caveat(stats),
    }
    (db / "chunked_index_report.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"\ntotal {report['total_seconds']:.0f}s")
    if not report["ground_truth_missing"]:
        print(f"ground truth present in every collection: yes ({len(gt)} ids)")
    else:
        # A --limit build is missing most of the ground truth by construction, and
        # printing 147 ids per format buries the rest of the summary. The full list
        # stays in the report either way.
        detail = ", ".join(
            f"{f} {len(ids)}" for f, ids in sorted(report["ground_truth_missing"].items())
        )
        print(f"ground truth MISSING of {len(gt)}: {detail}  (ids in the report)")
    print(f"\n! {report['caveat']}")
    print(f"\nwrote {db}/chunked_index_report.json")
    return 0


def chunk_caveat(stats: dict) -> str:
    if not stats:
        return ""
    order = sorted(stats.items(), key=lambda kv: -kv[1]["chunks_per_record_mean"])
    most, least = order[0], order[-1]
    ratio = most[1]["chunks_per_record_mean"] / max(least[1]["chunks_per_record_mean"], 1e-9)
    detail = ", ".join(
        f"{FORMAT_LABELS.get(f, f)} {v['chunks_per_record_mean']:.1f}" for f, v in order
    )
    return (
        f"{FORMAT_LABELS.get(most[0], most[0])} produces {ratio:.1f}x more chunks per record "
        f"than {FORMAT_LABELS.get(least[0], least[0])} for identical content ({detail}). "
        "Under max-pooling more chunks means more independent chances to match a query, so "
        "these scores reward verbosity. Read them as chunked retrieval over full records, "
        "not as a format ranking."
    )


# --------------------------------------------------------------------------- #
# Chart.
# --------------------------------------------------------------------------- #


def plot(panels: dict[str, dict], path: Path) -> Path | None:
    """One panel per pooling rule, each sorted by its **own** Recall@10.

    Sorting each panel independently is deliberate, the same choice ``exp1.plot``
    makes: a shared order would draw the eye to bar lengths, when the thing worth
    seeing is whether the rank order itself survives the change of pooling rule.
    Two panels in the same order *is* the finding here, and it should be legible
    as one.

    Chunks-per-record rides in the tick label rather than a separate axis. It is
    the number a reader needs to discount the verbosity caveat, and a format's
    name is useless without it -- JSON-LD at 15.8 chunks and prose at 3.6 are not
    comparable rows even though they carry identical facts.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from airm.exp1 import INK, INK_MUTED, SERIES_HF, SERIES_TIKTOKEN, SURFACE

    panels = {k: v for k, v in panels.items() if v.get("by_source", {}).get("all")}
    if not panels:
        return None

    recall_key, ndcg_key = f"recall@{TOP_K}", f"ndcg@{NDCG_K}"
    series = ((f"Recall@{TOP_K}", recall_key, SERIES_TIKTOKEN),
              (f"nDCG@{NDCG_K}", ndcg_key, SERIES_HF))

    fig, axes = plt.subplots(
        1, len(panels), figsize=(6.4 * len(panels) + 0.8, 5.0), dpi=200, squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    height = 0.36

    # A shared x-limit across panels: per-panel autoscaling would make a 0.02
    # difference look like a large one when the panels sit side by side.
    ceiling = max(
        cell[k] or 0.0
        for summary in panels.values()
        for cell in summary["by_source"]["all"].values()
        for _, k, _ in series
    )

    for ax, (label, summary) in zip(axes[0], panels.items()):
        cells = summary["by_source"]["all"]
        per_record = summary.get("chunks_per_record", {})
        order = sorted(FORMATS, key=lambda f: cells[f][recall_key] or 0.0)

        ax.set_facecolor(SURFACE)
        for i, (name, key, color) in enumerate(series):
            offset = (i - (len(series) - 1) / 2) * (height + 0.04)
            ys = [j + offset for j in range(len(order))]
            values = [cells[f][key] or 0.0 for f in order]
            ax.barh(ys, values, height=height, color=color, label=name, zorder=3)
            for y, value in zip(ys, values):
                ax.text(value + 0.008, y, f"{value:.3f}", va="center", ha="left",
                        fontsize=7.5, color=INK_MUTED)

        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(
            [f"{FORMAT_LABELS.get(f, f)}  ({per_record.get(f, float('nan')):.1f}/rec)"
             for f in order],
            fontsize=9, color=INK,
        )
        ax.set_title(f"pool = {label}", fontsize=10, color=INK, pad=8)
        ax.set_xlabel(f"score  ({summary['queries']['total']} queries, higher is better)",
                      fontsize=8.5, color=INK_MUTED)
        ax.set_xlim(0, ceiling * 1.22)
        ax.tick_params(axis="x", colors=INK_MUTED, labelsize=8)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#d8d7d2")
        ax.grid(axis="x", color="#e7e6e2", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    axes[0][-1].legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK_MUTED)

    first = next(iter(panels.values()))
    fig.suptitle(
        f"Chunked retrieval over {first.get('payload', 'unfaceted')} records — "
        f"{EMBED_MODEL}, {first['chunk_tokens']}-token chunks",
        fontsize=12.5, color=INK, x=0.007, ha="left", y=0.99,
    )
    caveat = first.get("caveat", "")
    if caveat:
        fig.text(0.007, 0.015, _wrap(caveat, 150), fontsize=6.5, color=INK_MUTED,
                 ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.11 if caveat else 0.0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def _wrap(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width))


def plot_only(args) -> int:
    """Redraw from summaries already on disk, without re-running any retrieval.

    The summary carries every number the figure needs, so regenerating a chart
    -- or building the max-vs-mean comparison after the fact -- should never
    cost another pass over the index.
    """
    panels: dict[str, dict] = {}
    for spec in args.runs:
        directory = Path(spec) if Path(spec).exists() else RUNS_DIR / spec
        summary_path = directory / "chunked_summary.json"
        if not summary_path.exists():
            print(f"FAILED: no chunked_summary.json in {directory}", file=sys.stderr)
            return 1
        summary = json.loads(summary_path.read_text())
        panels[summary.get("pool", directory.name)] = summary

    out = Path(args.out) if args.out else RUNS_DIR / "chunked_pooling.png"
    written = plot(panels, out)
    if not written:
        print("FAILED: nothing to plot.", file=sys.stderr)
        return 1
    print(f"wrote {written}")
    return 0


# --------------------------------------------------------------------------- #
# Evaluate.
# --------------------------------------------------------------------------- #


def pool(ids: list[str], distances: list[float], metas: list[dict], how: str) -> list[str]:
    """Chunk hits -> ranked concept-ids.

    ``max`` takes each record's best chunk, which is the standard rule and the
    one most sensitive to chunk count. ``mean`` averages a record's retrieved
    chunks, which penalises a record whose other chunks are irrelevant -- a
    different bias, not a smaller one.
    """
    by_record: dict[str, list[float]] = defaultdict(list)
    for meta, distance in zip(metas, distances):
        by_record[meta["concept_id"]].append(distance)
    if how == "mean":
        scored = {cid: statistics.fmean(ds) for cid, ds in by_record.items()}
    else:
        scored = {cid: min(ds) for cid, ds in by_record.items()}
    return [cid for cid, _ in sorted(scored.items(), key=lambda kv: kv[1])]


def evaluate(args) -> int:
    import chromadb

    db = Path(args.db or default_db(args.payload))
    report_path = db / "chunked_index_report.json"
    if not report_path.exists():
        print(f"FAILED: no index at {db}. Run `build` first.", file=sys.stderr)
        return 1
    built = json.loads(report_path.read_text())

    # The report is the authority on what was indexed. Trusting --payload here
    # would let a mistyped flag label a faceted index as unfaceted in the summary.
    payload = built.get("payload", "unfaceted")
    # Resolved before the summary is built, not after, so the id written into
    # the artefact is the same one that names the directory holding it.
    # A plain timestamp, like every other experiment in the study. What the run
    # *was* -- payload, pooling rule, query set -- is recorded in the summary,
    # not encoded in the directory name, so a run is identified by reading its
    # artefact rather than by parsing its folder.
    run_id = args.run_id or new_run_id()

    client = chromadb.PersistentClient(path=str(db))
    collections = {f: client.get_collection(f"{COLLECTION_PREFIX}{f}") for f in FORMATS}
    counts = {f: c.count() for f, c in collections.items()}

    queries = queries_from(args.queries)
    model = embedder()
    prefix = BGE_QUERY_INSTRUCTION if args.query_instruction else ""
    vectors = embed(model, [prefix + q.text for q in queries])

    rows: list[dict] = []
    for fmt, collection in collections.items():
        for query, vector in zip(queries, vectors):
            got = collection.query(
                query_embeddings=[vector],
                n_results=args.top_k * OVERSAMPLE,
                include=["distances", "metadatas"],
            )
            ranked = pool(got["ids"][0], got["distances"][0], got["metadatas"][0], args.pool)
            metrics = retrieval_metrics(ranked, query.expected_concept_ids, ndcg_k=NDCG_K)
            rows.append({
                "query_id": query.query_id, "source": query.source,
                "topic": query.topic or "", "format": fmt,
                "n_expected": len(query.expected_concept_ids),
                "retrieved": "|".join(ranked[:args.top_k]),
                **metrics,
            })

    metric_keys = [f"recall@{k}" for k in RECALL_AT] + ["mrr", f"ndcg@{NDCG_K}"]

    def mean(values):
        real = [v for v in values if v is not None and v == v]
        return sum(real) / len(real) if real else None

    summary = {
        **provenance(run_id),
        "payload": payload,
        "db": str(db),
        "embed_model": EMBED_MODEL,
        "window": WINDOW,
        "chunk_tokens": built["chunk_tokens"],
        "overlap": built["overlap"],
        "pool": args.pool,
        "top_k": args.top_k,
        "oversample": OVERSAMPLE,
        "query_instruction": prefix or None,
        "queries_path": args.queries or "data/queries.jsonl",
        "index_run_id": built.get("run_id"),
        "index_built": built.get("created"),
        "index_seconds": built.get("total_seconds"),
        "records": built["records"],
        "chunk_counts": counts,
        "chunks_per_record": {f: built["by_format"][f]["chunks_per_record_mean"] for f in FORMATS},
        "queries": {"total": len(queries),
                    "sme": sum(1 for q in queries if q.source == "sme"),
                    "synthetic": sum(1 for q in queries if q.source == "synthetic")},
        "caveat": built["caveat"],
        "by_source": {},
    }
    for slice_name in ("all", "sme", "synthetic"):
        subset = [r for r in rows if slice_name == "all" or r["source"] == slice_name]
        cells = {}
        for fmt in FORMATS:
            fmt_rows = [r for r in subset if r["format"] == fmt]
            cell = {"n": len(fmt_rows)}
            for key in metric_keys:
                m = mean(r[key] for r in fmt_rows)
                cell[key] = round(m, 4) if m is not None else None
            cells[fmt] = cell
        summary["by_source"][slice_name] = cells

    ranking = sorted(FORMATS, key=lambda f: -(summary["by_source"]["all"][f]["recall@10"] or 0.0))
    summary["ranking"] = ranking

    out = RUNS_DIR / run_id
    out.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    fields = ["query_id", "source", "topic", "format", "n_expected", "retrieved", *metric_keys]
    with (out / "chunked_rows.csv").open("w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "chunked_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    chart = None
    if not args.no_chart:
        chart = plot({args.pool: summary}, out / "chunked_retrieval.png")

    print(f"vectoriser  : {EMBED_MODEL} (window {WINDOW})")
    print(f"chunking    : {built['chunk_tokens']} tokens, {built['overlap']} overlap, pool={args.pool}")
    print(f"records     : {built['records']}   chunks: {counts}")
    print(f"queries     : {summary['queries']}")
    print(f"\n! {summary['caveat']}")
    for slice_name in ("all", "sme", "synthetic"):
        print(f"\n== {slice_name} ==")
        header = f"{'format':<18}{'chunks/rec':>11}{'n':>5}" + "".join(f"{k:>12}" for k in metric_keys)
        print(header)
        print("-" * len(header))
        for fmt in ranking:
            cell = summary["by_source"][slice_name][fmt]
            values = "".join(
                f"{cell[k]:>12.3f}" if cell.get(k) is not None else f"{'—':>12}" for k in metric_keys
            )
            print(f"{FORMAT_LABELS.get(fmt, fmt):<18}"
                  f"{summary['chunks_per_record'][fmt]:>11.2f}{cell['n']:>5}{values}")
    print(f"\nwrote {out}/chunked_summary.json")
    if chart:
        print(f"wrote {chart}")
    return 0


# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="chunk and vectorise the renderings")
    b.add_argument("--payload", choices=list(index.PAYLOADS), default=index.UNFACETED)
    b.add_argument("--db", default=None, help="default data/chroma/<payload>_chunked")
    b.add_argument("--chunk-tokens", type=int, default=CHUNK_TOKENS)
    b.add_argument("--overlap", type=int, default=OVERLAP)
    b.add_argument("--limit", type=int, default=None, help="first N records, for a smoke build")
    b.add_argument("--queries", default=None, help="query set for the ground-truth check")
    b.set_defaults(func=build)

    e = sub.add_parser("eval", help="score the 131-query set against the chunked index")
    e.add_argument("--payload", choices=list(index.PAYLOADS), default=index.UNFACETED)
    e.add_argument("--db", default=None, help="default data/chroma/<payload>_chunked")
    e.add_argument("--pool", choices=["max", "mean"], default="max")
    e.add_argument("-k", "--top-k", type=int, default=TOP_K)
    e.add_argument("--query-instruction", action="store_true",
                   help="prepend bge's retrieval instruction to each query")
    e.add_argument("--run-id", default=None)
    e.add_argument("--queries", default=None, help="default data/queries.jsonl")
    e.add_argument("--no-chart", action="store_true")
    e.set_defaults(func=evaluate)

    p = sub.add_parser("plot", help="redraw from summaries on disk, no retrieval")
    p.add_argument("runs", nargs="+", help="run ids under runs/, or paths to run dirs")
    p.add_argument("--out", default=None, help="default runs/chunked_pooling.png")
    p.set_defaults(func=plot_only)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
