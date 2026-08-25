"""Retrieval-only metrics against the **deployed** CMR vector database.

This is not Experiment 2. Experiment 2 varies ``format x model``, and for each
cell it retrieves, prompts a model, takes an answer, and has DeepEval grade that
answer on five LLM-judged metrics. It answers *did the model use what was
surfaced?* and needs an API key, a judge, checkpointing and hours.

This module strips everything downstream of the vector search. The only
independent variable is the **format**, the only metrics are Recall@k, MRR and
nDCG@10, no model is involved anywhere, and the whole run is six HTTP round
trips. It answers the other half of the question -- *does the representation
surface the right record at all?* -- exactly and repeatably.

Experiment 2 already computes these same numbers (``exp2.py`` calls
``retrieval_metrics`` for every cell) but computes them redundantly across the
model axis and cannot be reached without paying for the LLM half. There was no
way to get at them alone: ``index --probe`` prints hits with no ground truth,
and ``exp2 --no-judge`` still makes one answer call per cell.

**The target is a different index from the study's.** ``usage.md`` documents a
deployed ChromaDB holding 2,590 records per collection under ``nasa_cmr_*``
names, embedded with ``BAAI/bge-large-en-v1.5``. The study's own index holds 500
records per collection under ``cmr_*`` names, embedded with
``Alibaba-NLP/gte-modernbert-base``, in a local ``PersistentClient``. Nothing
this module reports is comparable with the local figures in ``EXPERIMENTS.md``
Step 7 -- the haystack is 5.2x larger and the encoder is a different model.

**And that encoder has a 512-token window**, which is precisely the confound
Step 7 rejected ``bge-small-en-v1.5`` for: it clipped JSON-LD 444/500 against
prose 205/500, unevenly and aimed straight at the formats under test. This
module therefore *measures and reports* truncation next to every score rather
than pretending the comparison is clean. See :func:`truncation_audit`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .config import (
    BASELINE_FORMAT,
    DATA_DIR,
    FORMAT_LABELS,
    FORMATS,
    RECALL_AT,
    RUNS_DIR,
    TOP_K,
)
from .evaluate import retrieval_metrics
from .provenance import provenance
from .queries import Query, ground_truth_ids, load_eval_queries

# --------------------------------------------------------------------------- #
# The endpoint, as documented in usage.md.
# --------------------------------------------------------------------------- #

DEFAULT_HOST = "metadata4ai.duckdns.org"
DEFAULT_PORT = 443

#: The production endpoint is HTTPS on 443. ``usage.md`` also lists a legacy
#: plain-HTTP endpoint; point ``--host/--port/--no-ssl`` at it if it comes back.
DEFAULT_SSL = True

#: The deployed collections are ``nasa_cmr_<fmt>``. The study's local ones are
#: ``cmr_<fmt>`` (``config.CHROMA_COLLECTION_PREFIX``). Keeping the prefixes
#: distinct is what stops a mistyped flag from silently scoring the wrong index.
DEFAULT_COLLECTION_PREFIX = "nasa_cmr_"

DEFAULT_EMBED_MODEL = "BAAI/bge-large-en-v1.5"

#: ``bge-*-en-v1.5`` is trained for asymmetric retrieval with this prefix on the
#: *query* only -- documents are embedded bare. ``usage.md`` shows a plain
#: ``embedder.encode(query)``, so the default here is no instruction, matching
#: how the index is documented to be used. ``--query-instruction`` opts in. It
#: moves scores, so whichever was used is recorded in the summary.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

#: Query embeddings per ``collection.query`` call. The queries are short, so this
#: is about keeping one HTTP body reasonable, not about memory.
QUERY_BATCH = 32

#: Records sampled for the truncation audit. All 2,590 is ~79 MB of document
#: text over the wire, so a seeded paired sample is the default.
DEFAULT_TRUNCATION_SAMPLE = 500

#: Same seed as the corpus and the synthetic query set, so the audit samples the
#: same records on every run.
RANDOM_SEED = 20260731

NDCG_K = 10

#: An embedding is judged to come from the same model as the stored one at or
#: above this cosine similarity. Identical models on different hardware differ in
#: the last few float bits, not in the third decimal place.
EMBED_PARITY_MIN = 0.999

METRIC_KEYS: tuple[str, ...] = (
    *(f"recall@{k}" for k in RECALL_AT),
    "mrr",
    f"ndcg@{NDCG_K}",
)

CSV_FIELDS: tuple[str, ...] = (
    "query_id",
    "source",
    "topic",
    "format",
    "n_expected",
    "n_retrieved",
    "retrieved",
    *METRIC_KEYS,
)


class RemoteEvalError(RuntimeError):
    """A gate failed. Every one of these means the numbers would be wrong."""


@dataclass(frozen=True)
class Endpoint:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    collection_prefix: str = DEFAULT_COLLECTION_PREFIX
    ssl: bool = DEFAULT_SSL

    @property
    def url(self) -> str:
        scheme = "https" if self.ssl else "http"
        # 443/80 are implicit in the scheme, and naming them explicitly is what
        # some proxies reject on the Host header.
        if (self.ssl and self.port == 443) or (not self.ssl and self.port == 80):
            return f"{scheme}://{self.host}"
        return f"{scheme}://{self.host}:{self.port}"

    def collection(self, fmt: str) -> str:
        return f"{self.collection_prefix}{fmt}"


# --------------------------------------------------------------------------- #
# Backend -- the three things this module needs from the outside world.
# --------------------------------------------------------------------------- #


@dataclass
class Backend:
    """The live dependencies, bundled so tests can replace them wholesale.

    ``client`` is a ChromaDB client, ``embed`` turns query texts into vectors,
    ``count_tokens`` measures text against the *embedding* model's tokenizer
    (not tiktoken -- the window is the model's, so the count has to be too), and
    ``window`` is that model's maximum sequence length.
    """

    client: object
    embed: Callable[[list[str]], list[list[float]]]
    count_tokens: Callable[[list[str]], list[int]]
    window: int


def connect(endpoint: Endpoint, *, timeout: float = 10.0) -> object:
    """Heartbeat first, then a client.

    ``chromadb.HttpClient`` raises its own connection error, but it arrives
    wrapped in enough layers that the actual cause -- nobody is listening on that
    port -- is not the first thing you read. Probing the documented heartbeat
    route directly buys a message that names the endpoint and the command to
    re-check it.
    """
    import httpx

    try:
        response = httpx.get(f"{endpoint.url}/api/v2/heartbeat", timeout=timeout)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - every failure mode is the same failure
        raise RemoteEvalError(
            f"No response from {endpoint.url} ({exc.__class__.__name__}: {exc}).\n"
            f"Re-check with: curl {endpoint.url}/api/v2/heartbeat"
        ) from exc

    import chromadb

    return chromadb.HttpClient(host=endpoint.host, port=endpoint.port, ssl=endpoint.ssl)


def live_backend(endpoint: Endpoint, embed_model: str) -> Backend:
    """The real backend: an HTTP client and a locally-loaded sentence encoder.

    The encoder must be the model the collections were built with -- a query
    vector from a different model is a random direction in the index's space, and
    every metric below it is noise. :func:`check_embedding_parity` proves it
    rather than trusting this argument.

    ``embed`` is deliberately **bare**: any query instruction is applied by
    :func:`search`, not here. bge prefixes queries and never documents, and the
    parity gate re-embeds a *document* through this same callable -- prefixing it
    there would make the gate compare a query vector against a document vector
    and fail for a reason that has nothing to do with the model.
    """
    from . import index

    def embed(texts: list[str]) -> list[list[float]]:
        return index.embed(texts, embed_model, batch_size=16)

    return Backend(
        client=connect(endpoint),
        embed=embed,
        count_tokens=lambda texts: index.count_wordpieces(texts, embed_model),
        window=index.max_sequence_tokens(embed_model),
    )


# --------------------------------------------------------------------------- #
# Preflight gates. Each one exists because the alternative is a plausible-looking
# table of wrong numbers, which is worse than a crash.
# --------------------------------------------------------------------------- #


def open_collections(backend: Backend, endpoint: Endpoint, formats: Sequence[str]) -> dict:
    """Resolve every collection up front, so a missing one fails before any work."""
    collections: dict[str, object] = {}
    missing: list[str] = []
    for fmt in formats:
        name = endpoint.collection(fmt)
        try:
            collections[fmt] = backend.client.get_collection(name)
        except Exception:  # noqa: BLE001 - chroma raises several types here
            missing.append(name)
    if missing:
        raise RemoteEvalError(
            f"Collections not found at {endpoint.url}: {', '.join(missing)}. "
            f"Check --collection-prefix (currently {endpoint.collection_prefix!r})."
        )
    return collections


def check_counts(collections: dict) -> dict[str, int]:
    """Every format must search the same haystack.

    A format whose collection holds fewer documents scores better for free --
    fewer distractors, same targets -- so an inequality here is not a warning.
    """
    counts = {fmt: int(col.count()) for fmt, col in collections.items()}
    empty = [fmt for fmt, n in counts.items() if n == 0]
    if empty:
        raise RemoteEvalError(f"Empty collections: {', '.join(sorted(empty))}")
    if len(set(counts.values())) > 1:
        raise RemoteEvalError(
            "Collections hold different numbers of documents, so the formats are "
            f"not searching the same haystack: {counts}"
        )
    return counts


def _as_vector(embeddings) -> list[float] | None:
    """Chroma returns embeddings as a numpy array in some versions, a list in others."""
    if embeddings is None:
        return None
    if len(embeddings) == 0:
        return None
    return [float(x) for x in embeddings[0]]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else float("nan")


def check_embedding_parity(
    backend: Backend,
    collection,
    concept_id: str,
    *,
    minimum: float = EMBED_PARITY_MIN,
    strict: bool = True,
) -> dict:
    """Prove the query encoder matches the one the index was built with.

    ``usage.md`` *documents* the model; it does not prove the deployment used it.
    So pull one stored document together with its stored vector, re-embed that
    document text locally, and compare. Below ``minimum`` the two live in
    different spaces and every score in this module is noise.

    The document is re-embedded **bare** -- no query instruction -- because bge
    prefixes queries only, never documents.

    The stored vector's norm and the collection's configured distance space are
    recorded at the same time: cosine is scale-invariant but L2 is not, so
    normalised-vs-raw matters if the deployment ranks by L2.
    """
    got = collection.get(ids=[concept_id], include=["embeddings", "documents"])
    stored = _as_vector(got.get("embeddings"))
    documents = got.get("documents") or []
    if stored is None or not documents:
        raise RemoteEvalError(
            f"Cannot read document {concept_id} back from the index, so the query "
            "encoder cannot be checked against the stored vectors."
        )

    local = backend.embed([documents[0]])[0]
    similarity = _cosine(stored, local)
    report = {
        "probe_concept_id": concept_id,
        "cosine_similarity": round(similarity, 6),
        "stored_vector_norm": round(math.sqrt(sum(x * x for x in stored)), 6),
        "stored_dimensions": len(stored),
        "local_dimensions": len(local),
        "hnsw_space": (getattr(collection, "metadata", None) or {}).get("hnsw:space"),
        "threshold": minimum,
        "passed": bool(similarity >= minimum),
    }
    if not report["passed"]:
        message = (
            f"Query encoder does not match the indexed vectors: cosine similarity "
            f"{similarity:.4f} < {minimum} on {concept_id} "
            f"(stored dim {len(stored)}, local dim {len(local)}). "
            "Check --embed-model against the model the deployment was built with."
        )
        if strict:
            raise RemoteEvalError(message)
        report["override"] = message
    return report


def check_ground_truth(
    collections: dict, expected: Sequence[str], *, strict: bool = True
) -> dict[str, list[str]]:
    """Every expected concept-id must exist in every collection.

    An id the index does not hold makes Recall@k unmeasurable for each query that
    expects it -- not zero, unmeasurable. Scoring it as a miss would quietly
    convert a coverage gap into a format finding.
    """
    wanted = list(expected)
    missing: dict[str, list[str]] = {}
    for fmt, collection in collections.items():
        # One request per collection, not one per id -- the lookup has to be
        # hoisted out of the comprehension or this is 147 round trips per format.
        present = set(collection.get(ids=wanted, include=[])["ids"])
        missing[fmt] = [c for c in wanted if c not in present]
    broken = {fmt: ids for fmt, ids in missing.items() if ids}
    if broken and strict:
        detail = "; ".join(f"{fmt}: {len(ids)} missing (e.g. {ids[0]})" for fmt, ids in broken.items())
        raise RemoteEvalError(
            f"Expected concept-ids absent from the index, so Recall@k is unmeasurable "
            f"for the queries naming them -- {detail}. "
            "Pass --allow-missing-ground-truth to record this and continue."
        )
    return missing


# --------------------------------------------------------------------------- #
# Which renderings is the deployment actually holding?
# --------------------------------------------------------------------------- #


def identify_payload(document: str, concept_id: str, fmt: str) -> str:
    """Compare a fetched document against the local rendering caches, byte for byte.

    The faceted cache is the 32-field projection; the unfaceted one is the raw
    UMM record. They are not comparable content, so knowing which the deployment
    holds is the difference between reading these scores as a format comparison
    and reading them as a format-plus-truncation comparison (§7 of
    ``DataGeneration.md``). An exact byte match is a much stronger answer than
    guessing from document length.
    """
    suffix = {"mat": ".txt"}.get(fmt, f".{fmt}")
    candidates = {
        "faceted": DATA_DIR / "format_cache" / fmt / f"{concept_id}{suffix}",
        "unfaceted": DATA_DIR / "format_cache_unfaceted" / fmt / f"{concept_id}{suffix}",
    }
    for label, path in candidates.items():
        try:
            if path.read_text() == document:
                return label
        except OSError:
            continue
    return "unknown"


# --------------------------------------------------------------------------- #
# The measurement.
# --------------------------------------------------------------------------- #


def search(
    backend: Backend,
    collections: dict,
    queries: Sequence[Query],
    *,
    k: int = TOP_K,
    instruction: str = "",
) -> dict[tuple[str, str], list[str]]:
    """Ranked concept-ids for every ``(query, format)``.

    Queries are embedded **once** and reused across all six collections -- the
    vector does not depend on the format -- and each collection is searched in
    batches rather than one query at a time, which is ``usage.md``'s own advice
    and turns 786 round trips into a couple of dozen.

    The bge retrieval instruction, if any, is applied *here* and nowhere else:
    it belongs to queries only, and the encoder itself must stay bare so the
    parity gate can re-embed a stored document through it.
    """
    texts = [instruction + q.text for q in queries]
    vectors = backend.embed(texts) if texts else []

    ranked: dict[tuple[str, str], list[str]] = {}
    for fmt, collection in collections.items():
        for start in range(0, len(vectors), QUERY_BATCH):
            chunk = vectors[start : start + QUERY_BATCH]
            result = collection.query(query_embeddings=chunk, n_results=k, include=[])
            ids = result["ids"]
            if len(ids) != len(chunk):
                raise RemoteEvalError(
                    f"{fmt}: asked for {len(chunk)} rankings, got {len(ids)}."
                )
            for offset, row in enumerate(ids):
                ranked[(queries[start + offset].query_id, fmt)] = list(row)
    return ranked


def score(
    queries: Sequence[Query],
    ranked: dict[tuple[str, str], list[str]],
    *,
    formats: Sequence[str] = FORMATS,
) -> list[dict]:
    """One row per ``(query, format)``, with the metrics ``evaluate`` computes."""
    rows: list[dict] = []
    for query in queries:
        for fmt in formats:
            key = (query.query_id, fmt)
            if key not in ranked:
                raise RemoteEvalError(f"No ranking recorded for {query.query_id} / {fmt}.")
            retrieved = ranked[key]
            metrics = retrieval_metrics(retrieved, query.expected_concept_ids, ndcg_k=NDCG_K)
            rows.append(
                {
                    "query_id": query.query_id,
                    "source": query.source,
                    "topic": query.topic or "",
                    "format": fmt,
                    "n_expected": len(query.expected_concept_ids),
                    "n_retrieved": len(retrieved),
                    "retrieved": "|".join(retrieved),
                    **{k: metrics[k] for k in METRIC_KEYS},
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# The truncation audit -- reported, never hidden.
# --------------------------------------------------------------------------- #


def truncation_audit(
    backend: Backend,
    collections: dict,
    *,
    sample: int | None = DEFAULT_TRUNCATION_SAMPLE,
    seed: int = RANDOM_SEED,
) -> dict:
    """How much of each format the encoder actually sees.

    ``bge-large-en-v1.5`` has a 512-token window. Step 7 of ``EXPERIMENTS.md``
    rejected the small sibling for exactly this: the clipping is *uneven across
    the formats under test*, so a measured "format effect" is partly a
    truncation effect. Every collection still reports its full document count and
    still produces plausible Recall@k while this is happening, which is why it
    has to be measured rather than assumed away.

    The sample is **paired** -- the same concept-ids in every format -- so the
    per-format overflow counts are comparable rather than six independent draws.
    ``sample=None`` audits every record; ``sample=0`` skips the audit.
    """
    if sample == 0:
        return {"skipped": True, "window": backend.window}

    baseline = collections.get(BASELINE_FORMAT) or next(iter(collections.values()))
    all_ids = list(baseline.get(include=[])["ids"])
    if sample is not None and sample < len(all_ids):
        rng = random.Random(seed)
        ids = sorted(rng.sample(all_ids, sample))
    else:
        ids = sorted(all_ids)

    per_format: dict[str, dict] = {}
    payload_guess: dict[str, str] = {}
    for fmt, collection in collections.items():
        got = collection.get(ids=ids, include=["documents"])
        documents = [d for d in (got.get("documents") or []) if d]
        if not documents:
            continue
        lengths = backend.count_tokens(documents)
        over = [n for n in lengths if n > backend.window]
        per_format[fmt] = {
            "sampled": len(lengths),
            "mean": round(statistics.fmean(lengths), 1),
            "median": round(statistics.median(lengths), 1),
            "max": max(lengths),
            "over_window": len(over),
            "over_window_share": round(len(over) / len(lengths), 4),
        }
        payload_guess[fmt] = identify_payload(documents[0], got["ids"][0], fmt)

    worst = max((v["over_window_share"] for v in per_format.values()), default=0.0)
    least = min((v["over_window_share"] for v in per_format.values()), default=0.0)
    return {
        "skipped": False,
        "window": backend.window,
        "sampled_records": len(ids),
        "by_format": per_format,
        "payload_looks_like": payload_guess,
        "caveat": _truncation_caveat(per_format, backend.window, worst, least),
    }


def _truncation_caveat(per_format: dict, window: int, worst: float, least: float) -> str:
    if not per_format or worst == 0:
        return (
            f"No sampled document exceeds the {window}-token embedding window, so the "
            "scores below are a format comparison and not a truncation comparison."
        )
    ordered = sorted(per_format.items(), key=lambda kv: -kv[1]["over_window_share"])
    detail = ", ".join(
        f"{FORMAT_LABELS.get(f, f)} {v['over_window']}/{v['sampled']}" for f, v in ordered
    )
    return (
        f"{worst:.0%} of the worst format's documents and {least:.0%} of the best format's "
        f"exceed the {window}-token embedding window ({detail}). The clipping is uneven "
        "across the formats under test, so these scores measure truncation as well as "
        "format and must not be read as a clean format comparison. This is the confound "
        "EXPERIMENTS.md Step 7 rejected bge-small-en-v1.5 for."
    )


# --------------------------------------------------------------------------- #
# Aggregation.
# --------------------------------------------------------------------------- #


def _mean(values: Iterable[float]) -> float | None:
    """Mean over real numbers only.

    ``recall`` and ``ndcg`` return ``nan`` when a query has no expected records:
    unmeasurable, which is a different event from a miss. ``v == v`` is the
    ``nan`` test, and this is the same predicate ``exp2.summarise`` uses.
    """
    real = [v for v in values if v is not None and v == v]
    return sum(real) / len(real) if real else None


def _paired_delta(
    rows: Sequence[dict], fmt: str, baseline: str, metric: str
) -> tuple[float | None, tuple[float, float]]:
    """Mean per-query difference against the baseline format, with a bootstrap CI.

    Paired, because every format sees the identical query: pairing removes
    between-query variance, which dwarfs the format effect. This mirrors how
    Experiment 1 reports its ratios.
    """
    from .exp1 import bootstrap_ci

    by_query = {(r["query_id"], r["format"]): r[metric] for r in rows}
    query_ids = sorted({r["query_id"] for r in rows})
    diffs = []
    for qid in query_ids:
        a, b = by_query.get((qid, fmt)), by_query.get((qid, baseline))
        if a is None or b is None or a != a or b != b:
            continue
        diffs.append(a - b)
    if not diffs:
        return None, (float("nan"), float("nan"))
    return sum(diffs) / len(diffs), bootstrap_ci(diffs)


def summarise(
    rows: Sequence[dict],
    *,
    embed_model: str,
    formats: Sequence[str] = FORMATS,
    baseline: str = BASELINE_FORMAT,
) -> dict:
    """Per-format means, split by query source, plus paired deltas vs the baseline.

    The split matters. The 11 SME queries expect 1-9 collections each, so their
    Recall@k is genuine *set* recall; the 120 synthetic queries expect exactly
    one, so theirs is a hit-rate. Pooling them would let the synthetic slice
    drown the expert one and would report two different metric semantics under a
    single heading.
    """
    from .exp1 import bootstrap_ci

    model_key = f"remote:{embed_model}"
    by_source: dict[str, dict] = {}
    for slice_name in ("all", "sme", "synthetic"):
        subset = [r for r in rows if slice_name == "all" or r["source"] == slice_name]
        if not subset:
            continue
        cells: dict[str, dict] = {}
        for fmt in formats:
            fmt_rows = [r for r in subset if r["format"] == fmt]
            if not fmt_rows:
                continue
            cell: dict = {"n": len(fmt_rows)}
            for metric in METRIC_KEYS:
                mean = _mean(r[metric] for r in fmt_rows)
                cell[metric] = round(mean, 4) if mean is not None else None
                lo, hi = bootstrap_ci([r[metric] for r in fmt_rows if r[metric] == r[metric]])
                cell[f"{metric}_ci"] = [round(lo, 4), round(hi, 4)]
            cells[f"{fmt}|{model_key}"] = cell
        by_source[slice_name] = cells

    deltas: dict[str, dict] = {}
    for fmt in formats:
        if fmt == baseline:
            continue
        entry: dict = {}
        for metric in (f"recall@{TOP_K}", f"ndcg@{NDCG_K}", "mrr"):
            if metric not in METRIC_KEYS:
                continue
            mean, (lo, hi) = _paired_delta(rows, fmt, baseline, metric)
            entry[metric] = {
                "mean_delta": round(mean, 4) if mean is not None else None,
                "ci": [round(lo, 4), round(hi, 4)],
            }
        deltas[fmt] = entry

    ranking_metric = f"recall@{TOP_K}"
    ranked = sorted(
        (f for f in formats if f"{f}|{model_key}" in by_source.get("all", {})),
        key=lambda f: -(by_source["all"][f"{f}|{model_key}"][ranking_metric] or 0.0),
    )
    return {
        "summary": by_source.get("all", {}),
        "by_source": by_source,
        "paired_vs_baseline": {"baseline": baseline, "formats": deltas},
        "ranking": {"metric": ranking_metric, "order": ranked},
    }


# --------------------------------------------------------------------------- #
# Output.
# --------------------------------------------------------------------------- #


def write_rows(rows: Sequence[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_summary(summary: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return path


def plot(summary: dict, path: Path, *, embed_model: str, caveat: str = "") -> Path | None:
    """Recall@10 and nDCG@10 per format, sorted by Recall@10."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .exp1 import INK, INK_MUTED, SERIES_HF, SERIES_TIKTOKEN, SURFACE

    cells = summary.get("by_source", {}).get("all", {})
    if not cells:
        return None

    model_key = f"remote:{embed_model}"
    recall_key, ndcg_key = f"recall@{TOP_K}", f"ndcg@{NDCG_K}"
    order = sorted(
        (f for f in FORMATS if f"{f}|{model_key}" in cells),
        key=lambda f: cells[f"{f}|{model_key}"][recall_key] or 0.0,
    )
    if not order:
        return None

    series = [(f"Recall@{TOP_K}", recall_key, SERIES_TIKTOKEN), (f"nDCG@{NDCG_K}", ndcg_key, SERIES_HF)]
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    height = 0.36

    for i, (label, key, color) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * (height + 0.04)
        ys = [j + offset for j in range(len(order))]
        means = [cells[f"{f}|{model_key}"][key] or 0.0 for f in order]
        ax.barh(ys, means, height=height, color=color, label=label, zorder=3)
        for y, mean in zip(ys, means):
            ax.text(mean + 0.008, y, f"{mean:.3f}", va="center", ha="left", fontsize=8, color=INK_MUTED)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([FORMAT_LABELS.get(f, f) for f in order], fontsize=9.5, color=INK)
    ax.set_xlabel("score  (131 queries, higher is better)", fontsize=8.5, color=INK_MUTED)
    ax.set_xlim(0, max(1.0, max(cells[f"{f}|{model_key}"][recall_key] or 0.0 for f in order) * 1.25))
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d8d7d2")
    ax.grid(axis="x", color="#e7e6e2", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK_MUTED)

    fig.suptitle(
        f"Retrieval by format — deployed index, {embed_model}",
        fontsize=12.5, color=INK, x=0.007, ha="left", y=0.99,
    )
    if caveat:
        fig.text(0.007, 0.015, _wrap(caveat, 120), fontsize=7, color=INK_MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.08 if caveat else 0.0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def _wrap(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width))


def format_table(summary: dict, embed_model: str, *, source: str = "all") -> str:
    """One slice as a table.

    Every slice is printed in the **same** format order -- the one the pooled
    slice ranks -- so the SME and synthetic tables can be read against each other
    line by line. A per-slice sort would look tidier and would hide exactly the
    thing worth seeing, which is where the two slices disagree.
    """
    cells = summary.get("by_source", {}).get(source, {})
    if not cells:
        return f"({source}: no rows)"
    model_key = f"remote:{embed_model}"
    header = f"{'format':<18}{'n':>5}" + "".join(f"{k:>12}" for k in METRIC_KEYS)
    lines = [header, "-" * len(header)]
    for fmt in summary["ranking"]["order"]:
        cell = cells.get(f"{fmt}|{model_key}")
        if not cell:
            continue
        values = "".join(
            f"{cell[k]:>12.3f}" if cell.get(k) is not None else f"{'—':>12}" for k in METRIC_KEYS
        )
        lines.append(f"{FORMAT_LABELS.get(fmt, fmt):<18}{cell['n']:>5}{values}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The run.
# --------------------------------------------------------------------------- #


def run(
    *,
    endpoint: Endpoint | None = None,
    backend: Backend | None = None,
    queries: Sequence[Query] | None = None,
    formats: Sequence[str] = FORMATS,
    embed_model: str = DEFAULT_EMBED_MODEL,
    query_instruction: str = "",
    k: int = TOP_K,
    truncation_sample: int | None = DEFAULT_TRUNCATION_SAMPLE,
    run_id: str | None = None,
    out_dir: Path | None = None,
    chart: bool = True,
    allow_embedding_mismatch: bool = False,
    allow_missing_ground_truth: bool = False,
) -> dict:
    """Gate, search, score, write. Returns the summary dict that was written."""
    endpoint = endpoint or Endpoint()
    queries = list(queries) if queries is not None else load_eval_queries()
    if not queries:
        raise RemoteEvalError("No evaluation queries. Run `uv run python -m airm.synth` first.")

    backend = backend or live_backend(endpoint, embed_model)
    collections = open_collections(backend, endpoint, formats)
    counts = check_counts(collections)

    expected = ground_truth_ids(queries)
    probe_format = BASELINE_FORMAT if BASELINE_FORMAT in collections else formats[0]
    parity = check_embedding_parity(
        backend,
        collections[probe_format],
        expected[0],
        strict=not allow_embedding_mismatch,
    )
    missing = check_ground_truth(collections, expected, strict=not allow_missing_ground_truth)

    truncation = truncation_audit(backend, collections, sample=truncation_sample)

    ranked = search(backend, collections, queries, k=k, instruction=query_instruction)
    rows = score(queries, ranked, formats=formats)

    summary = summarise(rows, embed_model=embed_model, formats=formats)
    summary = {
        **provenance(run_id),
        "endpoint": endpoint.url,
        "collections": {fmt: endpoint.collection(fmt) for fmt in formats},
        "collection_counts": counts,
        "embed_model": embed_model,
        "embed_window": backend.window,
        "query_instruction": query_instruction or None,
        "top_k": k,
        "queries": {
            "total": len(queries),
            "sme": sum(1 for q in queries if q.source == "sme"),
            "synthetic": sum(1 for q in queries if q.source == "synthetic"),
            "expected_concept_ids": len(expected),
        },
        "gates": {
            "embedding_parity": parity,
            "ground_truth_missing": {f: ids for f, ids in missing.items() if ids},
        },
        "truncation": truncation,
        "caveats": _caveats(truncation, counts),
        **summary,
    }

    out_dir = out_dir or (RUNS_DIR / summary["run_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary["rows_csv"] = str(write_rows(rows, out_dir / "remote_retrieval_rows.csv"))
    if chart:
        chart_path = plot(
            summary, out_dir / "remote_retrieval.png",
            embed_model=embed_model, caveat=truncation.get("caveat", ""),
        )
        if chart_path:
            summary["chart"] = str(chart_path)
    write_summary(summary, out_dir / "remote_retrieval_summary.json")
    return summary


def _caveats(truncation: dict, counts: dict[str, int]) -> list[str]:
    """The things a reader of these numbers has to know, carried in the artefact."""
    size = next(iter(counts.values()), 0)
    notes = [
        f"This index holds {size:,} documents per collection against the study's local 500, "
        "and uses a different embedding model, so absolute scores are not comparable with "
        "EXPERIMENTS.md Step 7.",
        "The 120 synthetic queries were accepted only if their source record was "
        "retrievable in the *local* index (500 documents, gte-modernbert-base). That gate "
        "does not carry over here; queries no format can answer on this index are a "
        "measured property of the query set, not a defect to filter out.",
        "Recall@k over the synthetic slice is effectively a hit-rate — every synthetic "
        "query expects exactly one record — while the SME slice expects 1-9. They are "
        "reported separately for that reason.",
    ]
    if not truncation.get("skipped") and truncation.get("caveat"):
        notes.insert(0, truncation["caveat"])
    return notes


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #


def _sample_arg(value: str) -> int | None:
    if value.lower() == "all":
        return None
    return int(value)


def main(argv: Sequence[str] | None = None) -> int:
    import os

    parser = argparse.ArgumentParser(
        description="Retrieval-only metrics (Recall@k, MRR, nDCG) for each format "
        "collection on the deployed CMR vector database.",
    )
    parser.add_argument("--host", default=os.getenv("AIRM_REMOTE_CHROMA_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("AIRM_REMOTE_CHROMA_PORT", DEFAULT_PORT)))
    parser.add_argument("--collection-prefix", default=DEFAULT_COLLECTION_PREFIX)
    parser.add_argument(
        "--no-ssl", dest="ssl", action="store_false", default=DEFAULT_SSL,
        help="talk plain HTTP (the legacy endpoint) instead of HTTPS",
    )
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--formats", default=",".join(FORMATS), help="comma-separated")
    parser.add_argument("-k", "--top-k", type=int, default=TOP_K)
    parser.add_argument("--limit", type=int, default=None, help="first N queries, for a smoke run")
    parser.add_argument(
        "--query-instruction",
        nargs="?",
        const=BGE_QUERY_INSTRUCTION,
        default="",
        help="prepend the bge retrieval instruction to each query "
        f"(bare flag uses {BGE_QUERY_INSTRUCTION!r})",
    )
    parser.add_argument(
        "--truncation-sample", type=_sample_arg, default=DEFAULT_TRUNCATION_SAMPLE,
        help="records to audit against the embedding window: N, 'all', or 0 to skip",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-chart", action="store_true")
    parser.add_argument("--allow-embedding-mismatch", action="store_true")
    parser.add_argument("--allow-missing-ground-truth", action="store_true")
    args = parser.parse_args(argv)

    endpoint = Endpoint(args.host, args.port, args.collection_prefix, args.ssl)
    queries = load_eval_queries()
    if args.limit:
        queries = queries[: args.limit]

    try:
        summary = run(
            endpoint=endpoint,
            queries=queries,
            formats=tuple(f.strip() for f in args.formats.split(",") if f.strip()),
            embed_model=args.embed_model,
            query_instruction=args.query_instruction,
            k=args.top_k,
            truncation_sample=args.truncation_sample,
            run_id=args.run_id,
            chart=not args.no_chart,
            allow_embedding_mismatch=args.allow_embedding_mismatch,
            allow_missing_ground_truth=args.allow_missing_ground_truth,
        )
    except RemoteEvalError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"endpoint          : {summary['endpoint']}")
    print(f"collections       : {summary['collection_counts']}")
    print(f"embedding         : {summary['embed_model']} (window {summary['embed_window']})")
    print(f"encoder parity    : cosine {summary['gates']['embedding_parity']['cosine_similarity']}")
    print(f"queries           : {summary['queries']}")
    trunc = summary["truncation"]
    if not trunc.get("skipped"):
        print(f"\npayload looks like: {trunc.get('payload_looks_like')}")
        print(f"{'format':<18}{'mean':>8}{'median':>8}{'max':>8}{'over window':>14}")
        for fmt, v in trunc.get("by_format", {}).items():
            over = f"{v['over_window']}/{v['sampled']}"
            print(
                f"{FORMAT_LABELS.get(fmt, fmt):<18}{v['mean']:>8.0f}{v['median']:>8.0f}"
                f"{v['max']:>8}{over:>14}"
            )
    for note in summary["caveats"]:
        print(f"\n! {note}")
    for slice_name in ("all", "sme", "synthetic"):
        print(f"\n== {slice_name} ==")
        print(format_table(summary, summary["embed_model"], source=slice_name))
    print(f"\nwrote {summary['rows_csv']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
