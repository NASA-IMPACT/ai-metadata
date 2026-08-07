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

from . import cmr, format_cache
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


@dataclass
class IndexReport:
    """What was indexed, and what was clipped doing it."""

    embed_model: str = EMBED_MODEL
    max_sequence_tokens: int | None = None
    counts: dict[str, int] = field(default_factory=dict)
    truncated: dict[str, int] = field(default_factory=dict)
    truncated_examples: dict[str, list[str]] = field(default_factory=dict)
    ground_truth_missing: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
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
    model = embedder(name)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
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


# --------------------------------------------------------------------------- #
# Chroma.
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=2)
def client(path: str | None = None):
    import chromadb

    ensure_dirs()
    return chromadb.PersistentClient(path=str(path or CHROMA_DIR))


def get_collection(fmt: str, *, path: str | None = None):
    return client(path).get_collection(collection_name(fmt))


def build(
    records: list[dict] | None = None,
    *,
    ground_truth: list[str] | None = None,
    path: str | None = None,
    batch_size: int = 128,
) -> IndexReport:
    """(Re)build all six collections from the corpus."""
    if records is None:
        records = cmr.load_corpus()
    if ground_truth is None:
        from .queries import ground_truth_ids, load_queries

        ground_truth = ground_truth_ids(load_queries())

    chroma = client(path)
    report = IndexReport(max_sequence_tokens=max_sequence_tokens())

    payloads = [(cmr.concept_id(r), topic_of(r), facets(r)) for r in records]

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
        docs = format_cache.render_many(fmt, [(cid, p) for cid, _, p in payloads])
        metas = [
            {"concept_id": cid, "topic": topic, "format": fmt}
            for cid, topic, _ in payloads
        ]

        lengths = count_wordpieces(docs)
        limit = report.max_sequence_tokens or 10**9
        clipped = [ids[i] for i, n in enumerate(lengths) if n > limit]
        report.truncated[fmt] = len(clipped)
        report.truncated_examples[fmt] = clipped[:5]

        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            chunk = docs[start:end]
            collection.add(
                ids=ids[start:end],
                documents=chunk,
                metadatas=metas[start:end],
                embeddings=embed(chunk),
            )
            _release_accelerator_memory()

        report.counts[fmt] = collection.count()
        present = set(collection.get(ids=list(ground_truth), include=[])["ids"])
        report.ground_truth_missing[fmt] = [c for c in ground_truth if c not in present]

    return report


def query(
    fmt: str,
    text: str,
    *,
    k: int = TOP_K,
    path: str | None = None,
) -> list[dict]:
    """Top-``k`` records for ``text`` from the ``fmt`` collection."""
    collection = get_collection(fmt, path=path)
    result = collection.query(
        query_embeddings=embed([text]),
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
    """Hard-gate violations; empty when the indexes are sound."""
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
        if clipped > MAX_EMBED_TRUNCATIONS:
            # Uneven clipping across formats turns the format axis into a
            # truncation axis, which would invalidate Experiment 2 outright.
            problems.append(
                f"{fmt}: {clipped} document(s) exceed the "
                f"{report.max_sequence_tokens}-token embedding window"
            )
    return problems


if __name__ == "__main__":  # pragma: no cover - CLI
    import argparse
    import json

    from .config import CORPUS_SIZE

    parser = argparse.ArgumentParser(description="Build the six ChromaDB collections.")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--probe", help="run a query against every collection")
    args = parser.parse_args()

    if args.probe:
        for fmt in FORMATS:
            hits = query(fmt, args.probe, k=3)
            print(f"\n{fmt}:")
            for h in hits:
                print(f"  {h['rank']}. {h['concept_id']:<24} d={h['distance']:.4f}")
        raise SystemExit(0)

    if not args.build:
        parser.error("pass --build or --probe")

    corpus_records = cmr.load_corpus()
    rep = build(corpus_records)

    print(f"embedding model  : {rep.embed_model}")
    print(f"max seq (tokens) : {rep.max_sequence_tokens}")
    print("documents per collection:")
    for fmt in FORMATS:
        note = f"  ({rep.truncated[fmt]} over the window)" if rep.truncated.get(fmt) else ""
        print(f"  {collection_name(fmt):<12} {rep.counts.get(fmt):>4}{note}")

    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    (CHROMA_DIR / "index_report.json").write_text(json.dumps(rep.to_dict(), indent=2) + "\n")

    issues = verify(rep, len(corpus_records))
    if issues:
        print("\nHARD GATE FAILURES:")
        for issue in issues:
            print(f"  - {issue}")
        raise SystemExit(1)
    print(f"\nall hard gates passed ({len(corpus_records)} records, expected {CORPUS_SIZE})")
