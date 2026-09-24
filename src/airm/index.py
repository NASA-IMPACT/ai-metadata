"""ChromaDB indexes — one collection per format, everything else held fixed.

Six collections over the *same* 500 records, the *same* embedding model, and the
*same* one-document-per-record chunking. The only thing that varies is how the
record is written down, which is the point.

Two things this module refuses to do quietly:

* **Truncate.** Formats differ in length, so a model with a fixed input window
  would clip the verbose ones more than the terse ones -- silently converting a
  format comparison into a truncation comparison. Any record whose rendering
  exceeds the embedding model's window is counted and reported per format.
* **Lose ground truth.** A collection missing a ground-truth record makes
  Recall@k unmeasurable for every query that expects it, so presence is asserted
  per collection rather than assumed.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path

from . import cmr, format_cache, unfaceted
from .config import (
    CHROMA_DIR,
    EMBED_MODEL,
    FORMATS,
    MAX_EMBED_TRUNCATIONS,
    TOP_K,
    collection_name,
    ensure_dirs,
)
from .corpus import topic_of
from .facets import facets

#: The two payloads, each in its own Chroma database. ``faceted`` is what the
#: study runs on; ``unfaceted`` is the control (see :mod:`airm.unfaceted`).
PAYLOADS: tuple[str, ...] = ("faceted", "unfaceted")

FACETED, UNFACETED = PAYLOADS


def chroma_dir(payload: str = FACETED) -> Path:
    """Separate databases rather than separate collection names.

    Collection names stay ``cmr_<fmt>`` in both, so a caller that has one
    database open cannot accidentally read the other's vectors by naming the
    wrong collection -- it would have to open the wrong path, which is explicit.
    """
    if payload not in PAYLOADS:
        raise ValueError(f"unknown payload {payload!r}; known: {', '.join(PAYLOADS)}")
    return CHROMA_DIR / payload


def _documents(payload: str, fmt: str, items: list[tuple[str, dict]]) -> list[str]:
    """Renderings for ``items`` from whichever cache the payload belongs to."""
    if payload == FACETED:
        return format_cache.render_many(fmt, items)
    return [unfaceted.render_all_cached(cid, p)[fmt] for cid, p in items]


@dataclass
class IndexReport:
    """What was indexed, and what was clipped doing it."""

    payload: str = FACETED
    embed_model: str = EMBED_MODEL
    max_sequence_tokens: int | None = None
    counts: dict[str, int] = field(default_factory=dict)
    truncated: dict[str, int] = field(default_factory=dict)
    truncated_examples: dict[str, list[str]] = field(default_factory=dict)
    ground_truth_missing: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "payload": self.payload,
            "embed_model": self.embed_model,
            "max_sequence_tokens": self.max_sequence_tokens,
            "counts": self.counts,
            "truncated": self.truncated,
            "truncated_examples": self.truncated_examples,
            "ground_truth_missing": self.ground_truth_missing,
        }


# --------------------------------------------------------------------------- #
# Embedding.
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=2)
def embedder(name: str = EMBED_MODEL):
    """The single embedding model, shared by all six collections."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def max_sequence_tokens(name: str = EMBED_MODEL) -> int:
    return int(embedder(name).max_seq_length)


#: Documents run to 6,300 tokens, and attention is quadratic in that. A batch of
#: 32 (sentence-transformers' default) asks MPS for a >10 GiB allocation and
#: dies; 4 keeps the peak small. Throughput barely suffers because the library
#: sorts by length internally, so short documents still batch densely.
EMBED_BATCH = 4


def embed(
    texts: list[str], name: str = EMBED_MODEL, *, batch_size: int = EMBED_BATCH
) -> list[list[float]]:
    """Embed ``texts``, re-encoding any batch-poisoned rows individually.

    On MPS, a batch of several near-window-length documents (7,500+ tokens,
    still inside the 8,192 limit) comes back NaN while each of those same
    documents embeds clean alone -- the instability is in the batched, padded
    attention, not the input. sentence-transformers sorts by length, which
    *guarantees* the long documents share a batch. The faceted corpus tops out
    near 6,300 tokens and never trips this; the unfaceted control does, on its
    first collection. A NaN that slipped through would be rejected by Chroma at
    ``add`` time and abort a multi-hour build.
    """
    import numpy as np

    model = embedder(name)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    for i in range(len(texts)):
        if np.isfinite(vectors[i]).all():
            continue
        vectors[i] = model.encode(
            [texts[i]], normalize_embeddings=True, show_progress_bar=False
        )[0]
        if not np.isfinite(vectors[i]).all():
            raise RuntimeError(
                f"embedding of document {i} (of {len(texts)}, "
                f"{len(texts[i])} chars) is non-finite even when encoded alone"
            )
    return [v.tolist() for v in vectors]


def _release_accelerator_memory() -> None:
    """Drop cached device memory between batches.

    Long documents leave a large MPS/CUDA cache behind; over 3,000 documents it
    accumulates until the next allocation fails.
    """
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - a cache hint must never break the build
        pass


def count_wordpieces(texts: list[str], name: str = EMBED_MODEL) -> list[int]:
    """Token lengths under the *embedding* model's tokenizer, not tiktoken."""
    tok = embedder(name).tokenizer
    return [len(tok.encode(t, add_special_tokens=True)) for t in texts]


def truncate_to_window(texts: list[str], name: str = EMBED_MODEL) -> list[str]:
    """Cut each text to the embedding model's context window.

    Necessary, not defensive. ``gte-modernbert-base`` does **not** silently
    truncate an over-length input -- it runs it through position embeddings it
    does not have and returns ``NaN``, which Chroma then rejects outright
    ("Embeddings must not contain NaN or Infinity values"). Before this, the
    build counted truncations without ever performing one, so the counter meant
    "would have been clipped" while the actual clipping never happened. The
    faceted index never hit it because its gate forbids over-length documents;
    the unfaceted control hits it on the first collection.

    Truncating here makes ``IndexReport.truncated`` mean what it says: this many
    documents reached the model with their tail removed.
    """
    tok = embedder(name).tokenizer
    limit = max_sequence_tokens(name)
    out: list[str] = []
    for text in texts:
        ids = tok.encode(text, add_special_tokens=True)
        if len(ids) <= limit:
            out.append(text)
            continue
        kept = tok(text, truncation=True, max_length=limit, add_special_tokens=True)
        out.append(tok.decode(kept["input_ids"], skip_special_tokens=True))
    return out


# --------------------------------------------------------------------------- #
# Chroma.
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=4)
def client(path: str | None = None):
    import chromadb

    ensure_dirs()
    target = Path(path) if path else chroma_dir()
    target.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(target))


def get_collection(fmt: str, *, path: str | None = None, payload: str = FACETED):
    return client(path or str(chroma_dir(payload))).get_collection(collection_name(fmt))


def all_cached_records(
    *,
    window: int | None = None,
    check_parity: bool = True,
    embed_model: str = EMBED_MODEL,
) -> tuple[list[dict], dict]:
    """Every record in ``cmr_cache``, minus those that would break a hard gate.

    The study's corpus is 500 stratified, parity-gated records. This is the other
    population: everything the pipeline ever *examined*, including sampling
    candidates that were never selected. Indexing it answers a different question
    -- how the format ranking behaves in a 5x larger haystack -- so it gets its
    own database rather than replacing the corpus index.

    Two exclusions, both of which the existing gates would otherwise fail on, and
    both **reported rather than silently applied**:

    ``over_window``
        A record whose rendering in *any* format exceeds the embedding window.
        ``MAX_EMBED_TRUNCATIONS = 0`` forbids these, and rightly: the clipping
        falls unevenly across the formats under test (CSV and JSON-LD overflow
        where prose does not), so including them would put a truncation effect
        inside the format comparison. A record is dropped for *every* format if
        it overflows in one, because dropping it per-format would leave the six
        collections searching different haystacks.

    ``parity``
        A record whose six renderings do not carry an identical fact set. Same
        rule the corpus builder applies (``corpus.build``), for the same reason.

    Returns ``(records, report)``.
    """
    from . import formats

    window = window or max_sequence_tokens(embed_model)
    ids = format_cache.cached_concept_ids()
    # Reading cached bytes directly would bypass the staleness guard and measure
    # renderings produced by older code. Check once, then fall back to live
    # rendering for the whole pass rather than per record.
    cache_usable = format_cache.is_current()

    records: list[dict] = []
    report: dict = {"examined": len(ids), "over_window": [], "parity_failures": [], "unreadable": []}

    for cid in ids:
        record = cmr.load_cached(cid)
        if record is None:
            report["unreadable"].append(cid)
            continue

        if check_parity:
            ok, detail = formats.check_record(record)
            if not ok:
                report["parity_failures"].append(
                    {"concept_id": cid, "formats": {k: v for k, v in detail.items() if not v["ok"]}}
                )
                continue

        # A token can never be shorter than one byte, so a rendering smaller than
        # the window in *bytes* cannot exceed it in tokens. That bound turns a
        # 15,540-document tokenisation into a few dozen.
        oversized: dict[str, int] = {}
        rendered: dict[str, str] | None = None
        for fmt in FORMATS:
            text: str | None = None
            if cache_usable:
                path = format_cache.path_for(fmt, cid)
                try:
                    if path.stat().st_size <= window:
                        continue  # cannot exceed the window in tokens
                    text = path.read_text()
                except OSError:
                    text = None
            if text is None:
                if rendered is None:
                    rendered = format_cache.render_all_cached(cid, facets(record))
                text = rendered[fmt]
                if len(text.encode()) <= window:
                    continue
            n = count_wordpieces([text], embed_model)[0]
            if n > window:
                oversized[fmt] = n
        if oversized:
            report["over_window"].append({"concept_id": cid, "tokens": oversized})
            continue

        records.append(record)

    report["kept"] = len(records)
    return records, report


def build(
    records: list[dict] | None = None,
    *,
    payload: str = FACETED,
    ground_truth: list[str] | None = None,
    path: str | None = None,
    batch_size: int = 128,
    embed_model: str = EMBED_MODEL,
    embed_batch: int = EMBED_BATCH,
) -> IndexReport:
    """(Re)build all six collections from the corpus, for one payload.

    Both payloads index the *same* 500 corpus records with the same model and
    chunking; only the rendering differs. They land in separate databases under
    ``data/chroma/<payload>/`` so neither can be read by accident.
    """
    if records is None:
        records = cmr.load_corpus()
    if ground_truth is None:
        from .queries import ground_truth_ids, load_queries

        ground_truth = ground_truth_ids(load_queries())

    chroma = client(path or str(chroma_dir(payload)))
    report = IndexReport(
        payload=payload,
        embed_model=embed_model,
        max_sequence_tokens=max_sequence_tokens(embed_model),
    )

    project = facets if payload == FACETED else unfaceted.payload
    payloads = [(cmr.concept_id(r), topic_of(r), project(r)) for r in records]

    for fmt in FORMATS:
        name = collection_name(fmt)
        try:
            chroma.delete_collection(name)
        except Exception:  # noqa: BLE001 - absent on a first build
            pass
        # Cosine matches the normalised embeddings produced above; Chroma's
        # default is L2, which would rank differently for the same vectors.
        collection = chroma.create_collection(name, metadata={"hnsw:space": "cosine"})

        ids = [cid for cid, _, _ in payloads]
        docs = _documents(payload, fmt, [(cid, p) for cid, _, p in payloads])
        metas = [
            {"concept_id": cid, "topic": topic, "format": fmt}
            for cid, topic, _ in payloads
        ]

        lengths = count_wordpieces(docs, embed_model)
        limit = report.max_sequence_tokens or 10**9
        clipped = [ids[i] for i, n in enumerate(lengths) if n > limit]
        report.truncated[fmt] = len(clipped)
        report.truncated_examples[fmt] = clipped[:5]

        # The document stored in Chroma stays whole; only the text handed to the
        # embedder is cut, so `--probe` and any inspection still show the real
        # rendering rather than a silently shortened one.
        embeddable = truncate_to_window(docs, embed_model)

        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            collection.add(
                ids=ids[start:end],
                documents=docs[start:end],
                metadatas=metas[start:end],
                embeddings=embed(
                    embeddable[start:end], embed_model, batch_size=embed_batch
                ),
            )
            _release_accelerator_memory()

        report.counts[fmt] = collection.count()
        # Chroma raises on an empty id list rather than returning nothing, so the
        # lookup is guarded: a caller with no ground truth to check is asking for
        # no check, not for an error.
        present = (
            set(collection.get(ids=list(ground_truth), include=[])["ids"])
            if ground_truth
            else set()
        )
        report.ground_truth_missing[fmt] = [c for c in ground_truth if c not in present]

    return report


def query(
    fmt: str,
    text: str,
    *,
    k: int = TOP_K,
    path: str | None = None,
    payload: str = FACETED,
    embed_model: str = EMBED_MODEL,
) -> list[dict]:
    """Top-``k`` records for ``text`` from the ``fmt`` collection."""
    collection = get_collection(fmt, path=path, payload=payload)
    result = collection.query(
        query_embeddings=embed([text], embed_model),
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "rank": i + 1,
            "concept_id": cid,
            "document": result["documents"][0][i],
            "topic": (result["metadatas"][0][i] or {}).get("topic"),
            "distance": result["distances"][0][i],
        }
        for i, cid in enumerate(result["ids"][0])
    ]


def verify(report: IndexReport, expected_size: int) -> list[str]:
    """Hard-gate violations; empty when the indexes are sound.

    Truncation is fatal for the **faceted** index and merely reported for the
    control, for the same reason parity is: the faceted index is what
    Experiment 2 measures a format effect on, and uneven clipping there would
    turn that effect into a truncation artefact. The control exists to show what
    the raw record costs, and the fact that it *cannot* be embedded whole at 8k
    is one of the things it is there to show -- see :func:`truncation_note`.
    """
    problems: list[str] = []
    for fmt in FORMATS:
        n = report.counts.get(fmt)
        if n != expected_size:
            problems.append(f"{fmt}: indexed {n} documents, expected {expected_size}")
        missing = report.ground_truth_missing.get(fmt) or []
        if missing:
            problems.append(
                f"{fmt}: {len(missing)} ground-truth record(s) not indexed: {missing[:5]}"
            )
        clipped = report.truncated.get(fmt, 0)
        if clipped > MAX_EMBED_TRUNCATIONS and report.payload == FACETED:
            # Uneven clipping across formats turns the format axis into a
            # truncation axis, which would invalidate Experiment 2 outright.
            problems.append(
                f"{fmt}: {clipped} document(s) exceed the "
                f"{report.max_sequence_tokens}-token embedding window"
            )
    return problems


def truncation_note(report: IndexReport) -> str:
    """Why the control's index must not be used for a format comparison.

    Returns an empty string when nothing was clipped.
    """
    clipped = {f: n for f, n in report.truncated.items() if n}
    if not clipped:
        return ""
    worst = max(clipped.values())
    return (
        f"{sum(clipped.values())} document(s) over the {report.max_sequence_tokens}-token "
        f"window, unevenly: " + ", ".join(f"{f} {n}" for f, n in sorted(clipped.items(), key=lambda kv: -kv[1]))
        + f". The verbose formats are clipped up to {worst}x more than the terse ones, so "
        "retrieval scores from this index measure truncation as much as format. It is built "
        "for inspection and completeness, not for a format comparison."
    )


if __name__ == "__main__":  # pragma: no cover - CLI
    import argparse
    import json

    from .config import CORPUS_SIZE

    parser = argparse.ArgumentParser(description="Build the six ChromaDB collections.")
    parser.add_argument("--build", action="store_true")
    parser.add_argument(
        "--payload",
        choices=[*PAYLOADS, "both"],
        default=FACETED,
        help="which rendering to index (default: faceted, what the study runs on)",
    )
    parser.add_argument("--probe", help="run a query against every collection")
    parser.add_argument(
        "--records",
        choices=["corpus", "all"],
        default="corpus",
        help="corpus (500, the study's stratified sample) or all cached records "
        "minus those failing the parity or embedding-window gates",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="database directory; defaults to data/chroma/<payload>. Give an "
        "explicit path with --records all so the corpus index survives.",
    )
    parser.add_argument(
        "--embed-model",
        default=EMBED_MODEL,
        help="sentence-transformers checkpoint to embed with (default: %(default)s). "
        "Swapping this changes the vectors, so it needs its own --path: a "
        "collection built by one encoder cannot be queried by another.",
    )
    parser.add_argument(
        "--embed-batch",
        type=int,
        default=EMBED_BATCH,
        help="documents per forward pass (default: %(default)s, tuned for MPS at "
        "an 8k window). Raise it on a CUDA box with headroom.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="index only the first N records -- a smoke test, not a study run. "
        "The corpus-size gate is relaxed accordingly and the report says so.",
    )
    args = parser.parse_args()

    if args.probe:
        for name in (PAYLOADS if args.payload == "both" else [args.payload]):
            print(f"\n=== {name} ===")
            for fmt in FORMATS:
                hits = query(
                    fmt,
                    args.probe,
                    k=3,
                    payload=name,
                    path=args.path,
                    embed_model=args.embed_model,
                )
                print(f"  {fmt}:")
                for h in hits:
                    print(f"    {h['rank']}. {h['concept_id']:<24} d={h['distance']:.4f}")
        raise SystemExit(0)

    if not args.build:
        parser.error("pass --build or --probe")

    selection_report: dict | None = None
    if args.records == "all":
        print("selecting from the full record cache ...")
        corpus_records, selection_report = all_cached_records(
            embed_model=args.embed_model
        )
        print(f"  examined         : {selection_report['examined']}")
        print(f"  parity failures  : {len(selection_report['parity_failures'])}")
        for f in selection_report["parity_failures"][:5]:
            print(f"      {f['concept_id']}  ({', '.join(f['formats'])})")
        print(f"  over the window  : {len(selection_report['over_window'])}")
        for f in selection_report["over_window"][:5]:
            detail = ", ".join(f"{k} {v:,}" for k, v in f["tokens"].items())
            print(f"      {f['concept_id']}  ({detail})")
        if selection_report["unreadable"]:
            print(f"  unreadable       : {len(selection_report['unreadable'])}")
        print(f"  kept             : {selection_report['kept']}")
    else:
        corpus_records = cmr.load_corpus()

    if args.limit:
        corpus_records = corpus_records[: args.limit]
        print(f"\nSMOKE TEST: {len(corpus_records)} records only -- not a study run.")

    # The evaluation set, not just the SME half -- a ground-truth id absent from
    # a collection makes Recall@k unmeasurable for every query naming it.
    from .queries import ground_truth_ids, load_eval_queries

    ground_truth = ground_truth_ids(load_eval_queries())
    if args.limit:
        # A truncated corpus is missing most ground-truth records by
        # construction; gating on their presence would fail every smoke test
        # for the one reason that is not a defect.
        indexed = {cmr.concept_id(r) for r in corpus_records}
        ground_truth = [c for c in ground_truth if c in indexed]
    failed = False

    for name in (PAYLOADS if args.payload == "both" else [args.payload]):
        out = Path(args.path) if args.path else chroma_dir(name)
        rep = build(
            corpus_records,
            payload=name,
            ground_truth=ground_truth,
            path=str(out),
            embed_model=args.embed_model,
            embed_batch=args.embed_batch,
        )
        out.mkdir(parents=True, exist_ok=True)
        if selection_report is not None:
            (out / "selection_report.json").write_text(
                json.dumps(selection_report, indent=2) + "\n"
            )
        (out / "index_report.json").write_text(json.dumps(rep.to_dict(), indent=2) + "\n")

        print(f"\n=== {name} ===")
        print(f"embedding model  : {rep.embed_model}")
        print(f"max seq (tokens) : {rep.max_sequence_tokens}")
        print("documents per collection:")
        for fmt in FORMATS:
            note = f"  ({rep.truncated[fmt]} over the window)" if rep.truncated.get(fmt) else ""
            print(f"  {collection_name(fmt):<12} {rep.counts.get(fmt):>4}{note}")
        if (note := truncation_note(rep)):
            print(f"\n  NOTE: {note}")

        issues = verify(rep, len(corpus_records))
        if issues:
            failed = True
            print(f"\nHARD GATE FAILURES ({name}):")
            for issue in issues:
                print(f"  - {issue}")
        else:
            expected = len(corpus_records) if args.limit or args.records == "all" else CORPUS_SIZE
            print(f"\nall hard gates passed ({len(corpus_records)} records, "
                  f"expected {expected})")
        print(f"wrote {out}/index_report.json")

    raise SystemExit(1 if failed else 0)
