"""Generate synthetic evaluation queries whose ground truth spans several datasets.

``airm.synth`` writes one query per record, so every synthetic query has exactly
one expected concept-id; only a handful of SME queries exercise the multi-target
scoring path. This module produces a set where *every* query expects two or
three collections, the way sme-002 does ("sea-ice concentration and thickness"
needs both a concentration product and a thickness product).

Groups are built from the index, not sampled at random: a seed record's nearest
same-topic neighbours are the datasets a joint question can plausibly need
together. Two random ATMOSPHERE records (say, lightning flashes and stratospheric
aerosol) share a topic but no research question; forcing a query over them would
manufacture ground truth no single question honestly has.

Each candidate must clear the same three checks as the single-target set --
it parses, it leaks no identifier from *any* grouped record, and it is
retrievable -- with retrievability tightened to every expected id: a
multi-target query whose second dataset no format can surface in the top-k
carries ground truth that only ever scores as a miss, which penalises every
format equally and measures nothing.
"""

from __future__ import annotations

import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import cmr, index
from .config import (
    DATA_DIR,
    FORMATS,
    MAX_QUERY_REGEN_ATTEMPTS,
    TOP_K,
    ModelSpec,
    synthetic_quota,
)
from .corpus import topic_of
from .facets import facets
from .llm import PURPOSE_QUERY_GEN, CallLogger, LLMError, ask
from .queries import (
    Query,
    _generation_prompt,
    _leaked_identifiers,
    parse_generated,
    write_queries,
)

RANDOM_SEED = 20260831

MULTI_QUERIES_PATH = DATA_DIR / "queries_multi.jsonl"

#: Share of groups that take a third dataset. The rest are pairs. Mixed rather
#: than fixed so the set exercises both |expected| = 2 and |expected| = 3.
TRIPLE_SHARE = 0.4

#: Neighbours fetched per seed before topic filtering. Generous because the
#: nearest hits in a full-cache index are often the seed's own version siblings
#: from other providers plus off-topic lookalikes.
NEIGHBOUR_POOL_K = 30

#: Serialises every embedding + Chroma call. The generation workers parallelise
#: the *network* wait on the LLM; the local embedder is not a resource to
#: contend over, and concurrent inference through one model on MPS is the kind
#: of thing that fails rarely enough to be trusted and often enough to corrupt
#: a run.
_INDEX_LOCK = threading.Lock()

GENERATION_SYSTEM_MULTI = """\
You write realistic Earth-science data-discovery questions for evaluating \
metadata search systems.

You are shown several NASA CMR collections. Write ONE question a domain \
scientist would plausibly ask that requires ALL of these collections together \
to answer -- because it compares their variables, combines their measurements, \
or spans what each one covers.

Rules:
- Ask a research question, not a lookup. "How did sea-ice concentration and \
thickness in the Kara Sea change each March from 2020 to 2025?" is good; \
"Find sea ice products" is not.
- The question must genuinely need every collection shown. Do not write a \
question that any one of them could answer alone.
- NEVER name a dataset, its short name, its DOI, its concept id, its version, \
or its data centre. The question must be answerable only by understanding what \
the data measures.
- Ground it in the science: name the phenomena, a plausible region and a \
plausible time period.
- One or two sentences.

Return JSON: {"query": "<the question>"}"""


@dataclass
class MultiSynthReport:
    generator: str = ""
    requested: dict[str, int] = field(default_factory=dict)
    produced: dict[str, int] = field(default_factory=dict)
    group_sizes: dict[str, int] = field(default_factory=dict)  # "2"/"3" -> count
    dropped: list[dict] = field(default_factory=list)
    retries: int = 0
    seeds_without_neighbours: int = 0
    reused_calls: int = 0
    llm_calls: int = 0

    @property
    def total(self) -> int:
        return sum(self.produced.values())

    def to_dict(self) -> dict:
        return {
            "generator": self.generator,
            "requested": self.requested,
            "produced": self.produced,
            "total": self.total,
            "group_sizes": self.group_sizes,
            "retries": self.retries,
            "seeds_without_neighbours": self.seeds_without_neighbours,
            "reused_calls": self.reused_calls,
            "llm_calls": self.llm_calls,
            "dropped": self.dropped,
        }


# --------------------------------------------------------------------------- #
# Replay of already-paid generation calls.
# --------------------------------------------------------------------------- #


class ResponseReplay:
    """Responses from earlier (aborted) runs, keyed by the exact prompt.

    Group construction is deterministic given the seed, so a rerun asks the
    generator the same questions in the same order; an interrupted run's logged
    responses are therefore reusable verbatim instead of being paid for twice.
    Keys hold a FIFO of responses because a retried prompt is byte-identical to
    its first attempt -- replay must hand attempts back in the order they were
    originally answered or attempt 2 would replay as attempt 1.
    """

    def __init__(self) -> None:
        self._responses: dict[tuple[str, ...], list[str]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(messages: list[dict]) -> tuple[str, ...]:
        return tuple(str(m.get("content", "")) for m in messages)

    @classmethod
    def from_logs(cls, spec: ModelSpec, paths: list[Path]) -> "ResponseReplay":
        replay = cls()
        entries: list[dict] = []
        for path in paths:
            try:
                with path.open() as fh:
                    for line in fh:
                        if line.strip():
                            entries.append(json.loads(line))
            except OSError:
                continue
        entries.sort(key=lambda e: e.get("ts", ""))
        for entry in entries:
            if entry.get("provider") != spec.provider or entry.get("model") != spec.model:
                continue
            if entry.get("error") or not entry.get("response"):
                continue  # a failed call is not worth replaying
            key = cls._key(entry.get("messages") or [])
            replay._responses.setdefault(key, []).append(entry["response"])
        return replay

    def __len__(self) -> int:
        return sum(len(v) for v in self._responses.values())

    def pop(self, system: str, prompt: str) -> str | None:
        key = (system, prompt)
        with self._lock:
            queue = self._responses.get(key)
            return queue.pop(0) if queue else None


def _neighbour_probe(payload: dict) -> str:
    """What the seed record is *about*, for finding related collections.

    The full generation prompt would work, but the abstract dominates it; title
    plus keywords plus variables is the science content without 900 characters
    of boilerplate about file formats and versioning.
    """
    parts = [payload.get("title", "")]
    if payload.get("science_keywords"):
        parts.append("; ".join(payload["science_keywords"][:6]))
    if payload.get("variables"):
        parts.append(", ".join(payload["variables"][:8]))
    return "\n".join(p for p in parts if p)


def _neighbours(
    payload: dict,
    seed_cid: str,
    topic: str,
    cid_topics: dict[str, str],
    *,
    want: int,
    index_path: str | None,
) -> list[str]:
    """The ``want`` nearest same-topic collections to the seed, nearest first."""
    with _INDEX_LOCK:
        hits = index.query("mat", _neighbour_probe(payload), k=NEIGHBOUR_POOL_K, path=index_path)
    out: list[str] = []
    for hit in hits:
        cid = hit["concept_id"]
        if cid == seed_cid or cid in out:
            continue
        if cid_topics.get(cid) != topic:
            continue
        out.append(cid)
        if len(out) == want:
            break
    return out


def _multi_prompt(payloads: list[dict]) -> str:
    blocks = []
    for i, payload in enumerate(payloads, 1):
        blocks.append(f"Collection {i}:\n{_generation_prompt(payload)}")
    return "\n\n".join(blocks)


def _unretrievable(
    text: str, concept_ids: list[str], *, k: int, index_path: str | None
) -> list[str]:
    """Expected ids that no format surfaces in its top-``k`` for ``text``."""
    missing = set(concept_ids)
    for fmt in FORMATS:
        if not missing:
            break
        with _INDEX_LOCK:
            hits = {h["concept_id"] for h in index.query(fmt, text, k=k, path=index_path)}
        missing -= hits
    return sorted(missing)


@dataclass
class _CandidateOutcome:
    """What one seed produced: an accepted query, a drop record, or a skip."""

    query: Query | None = None
    drop: dict | None = None
    no_neighbour: bool = False
    retries: int = 0
    reused_calls: int = 0
    llm_calls: int = 0


def _candidate(
    spec: ModelSpec,
    record: dict,
    extra: int,
    topic: str,
    by_cid: dict[str, dict],
    cid_topics: dict[str, str],
    *,
    k: int,
    index_path: str | None,
    logger: CallLogger | None,
    temperature: float | None,
    replay: ResponseReplay | None,
) -> _CandidateOutcome:
    """Run one seed record through grouping, generation and the three gates."""
    out = _CandidateOutcome()
    seed_cid = cmr.concept_id(record)
    seed_payload = facets(record)
    neighbour_cids = _neighbours(
        seed_payload, seed_cid, topic, cid_topics, want=extra, index_path=index_path
    )
    if not neighbour_cids:
        out.no_neighbour = True
        return out

    group_cids = [seed_cid, *neighbour_cids]
    payloads = [seed_payload] + [facets(by_cid[c]) for c in neighbour_cids]
    prompt = _multi_prompt(payloads)

    for attempt in range(1, MAX_QUERY_REGEN_ATTEMPTS + 1):
        if attempt > 1:
            out.retries += 1

        raw = replay.pop(GENERATION_SYSTEM_MULTI, prompt) if replay else None
        if raw is not None:
            out.reused_calls += 1
        else:
            out.llm_calls += 1
            try:
                response = ask(
                    spec,
                    prompt,
                    system=GENERATION_SYSTEM_MULTI,
                    purpose=PURPOSE_QUERY_GEN,
                    logger=logger,
                    json_mode=True,
                    temperature=temperature,
                    topic=topic,
                    concept_id=seed_cid,
                    attempt=attempt,
                )
            except LLMError as exc:
                out.drop = {
                    "concept_ids": group_cids,
                    "topic": topic,
                    "reason": f"llm error: {exc}",
                }
                return out
            raw = response.text

        text = parse_generated(raw)
        if not text:
            continue

        leaks = [leak for payload in payloads for leak in _leaked_identifiers(text, payload)]
        if leaks:
            if attempt == MAX_QUERY_REGEN_ATTEMPTS:
                out.drop = {
                    "concept_ids": group_cids,
                    "topic": topic,
                    "reason": "leaked identifiers",
                    "detail": leaks,
                    "query": text,
                }
            continue

        missing = _unretrievable(text, group_cids, k=k, index_path=index_path)
        if missing:
            if attempt == MAX_QUERY_REGEN_ATTEMPTS:
                out.drop = {
                    "concept_ids": group_cids,
                    "topic": topic,
                    "reason": "expected ids not retrievable by any format",
                    "detail": missing,
                    "query": text,
                }
            continue

        out.query = Query(
            query_id="",  # assigned in accepted order by generate()
            text=text,
            expected_concept_ids=group_cids,
            source="synthetic",
            topic=topic,
            expected_titles=[p.get("title", "") for p in payloads],
        )
        return out

    return out


def generate(
    spec: ModelSpec,
    *,
    records: list[dict],
    quota: dict[str, int] | None = None,
    logger: CallLogger | None = None,
    seed: int = RANDOM_SEED,
    k: int = TOP_K,
    index_path: str | None = None,
    temperature: float | None = None,
    workers: int = 1,
    replay: ResponseReplay | None = None,
) -> tuple[list[Query], MultiSynthReport]:
    """Generate one multi-target query per seed record until each quota is met.

    ``workers`` parallelises the LLM wait; index access stays serialised behind
    ``_INDEX_LOCK``. Seeds are dispatched in waves of exactly the remaining
    quota and their randomness (pool shuffle, group-size draw) is consumed at
    *scheduling* time in pool order, so the set of seeds visited -- and hence
    every prompt -- is identical to a sequential run with the same seed. That
    identity is what lets ``replay`` reuse an aborted run's paid responses.
    """
    quota = quota or synthetic_quota()
    report = MultiSynthReport(generator=spec.key, requested=dict(quota))
    rng = random.Random(seed)

    by_cid = {cmr.concept_id(r): r for r in records}
    cid_topics = {cid: topic_of(r) for cid, r in by_cid.items()}
    by_topic: dict[str, list[dict]] = {}
    for record in records:
        by_topic.setdefault(topic_of(record), []).append(record)

    queries: list[Query] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for topic in sorted(quota):
            want = quota[topic]
            pool = list(by_topic.get(topic, []))
            rng.shuffle(pool)
            produced = 0
            next_seed = 0

            if len(pool) < want:
                report.dropped.append(
                    {
                        "topic": topic,
                        "reason": "too few corpus records to draw from",
                        "available": len(pool),
                        "requested": want,
                    }
                )

            while produced < want and next_seed < len(pool):
                batch = []
                while len(batch) < want - produced and next_seed < len(pool):
                    record = pool[next_seed]
                    next_seed += 1
                    extra = 2 if rng.random() < TRIPLE_SHARE else 1
                    batch.append(
                        executor.submit(
                            _candidate,
                            spec,
                            record,
                            extra,
                            topic,
                            by_cid,
                            cid_topics,
                            k=k,
                            index_path=index_path,
                            logger=logger,
                            temperature=temperature,
                            replay=replay,
                        )
                    )
                # Collected in submission order so accepted queries keep the
                # pool's deterministic ordering regardless of completion order.
                for future in batch:
                    out = future.result()
                    report.retries += out.retries
                    report.reused_calls += out.reused_calls
                    report.llm_calls += out.llm_calls
                    if out.no_neighbour:
                        report.seeds_without_neighbours += 1
                    if out.drop:
                        report.dropped.append(out.drop)
                    if out.query and produced < want:
                        out.query.query_id = f"mds-{len(queries) + 1:03d}"
                        queries.append(out.query)
                        produced += 1
                        size = str(len(out.query.expected_concept_ids))
                        report.group_sizes[size] = report.group_sizes.get(size, 0) + 1

            report.produced[topic] = produced

    return queries, report


def build(
    spec: ModelSpec,
    *,
    logger: CallLogger | None = None,
    out_path: Path = MULTI_QUERIES_PATH,
    report_path: Path | None = None,
    **kwargs,
) -> tuple[list[Query], MultiSynthReport]:
    """Generate the multi-target set and write it with its provenance report.

    Unlike ``synth.build`` this does *not* prepend the SME queries: most of them
    expect a single collection, and the point of this set is that every query
    exercises multi-target scoring.
    """
    queries, report = generate(spec, logger=logger, **kwargs)
    write_queries(queries, out_path)
    if report_path is None:
        report_path = out_path.with_name(f"{out_path.stem}_report.json")
    report_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n")
    return queries, report


def verify(report: MultiSynthReport) -> list[str]:
    """Hard-gate violations for the multi-target set."""
    problems: list[str] = []
    for topic, want in report.requested.items():
        got = report.produced.get(topic, 0)
        if got < want:
            problems.append(f"{topic}: produced {got} of {want} requested queries")
    if report.group_sizes.get("2", 0) + report.group_sizes.get("3", 0) != report.total:
        problems.append("some queries expect fewer than 2 or more than 3 collections")
    return problems


if __name__ == "__main__":  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a query set where every query expects 2-3 collections."
    )
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument("--limit", type=int, help="cap per-topic quota (smoke run)")
    parser.add_argument(
        "--total",
        type=int,
        default=500,
        help="queries to generate; the per-topic shape is scaled, not reshaped",
    )
    parser.add_argument(
        "--records",
        choices=("corpus", "full"),
        default="full",
        help="draw from the 500 stratified corpus or from every gate-passing cached record",
    )
    parser.add_argument(
        "--index-path",
        default=None,
        help="Chroma database for neighbour search and the retrievability check; "
        "must hold the same records as --records",
    )
    parser.add_argument("--out", type=Path, default=MULTI_QUERIES_PATH)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent generation calls; the produced set is identical at any width",
    )
    parser.add_argument(
        "--reuse-logs",
        nargs="*",
        type=Path,
        default=None,
        help="query_gen.jsonl files from aborted runs whose responses should be "
        "replayed instead of paid for again; pass nothing to disable reuse",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=-1,
        help="omitted by default; pass a value for models that accept one",
    )
    args = parser.parse_args()

    quota = synthetic_quota(args.total)
    if args.limit:
        quota = {t: min(n, args.limit) for t, n in quota.items()}

    spec = ModelSpec(args.provider, args.model)
    from .llm import resolve_models

    usable, issues = resolve_models([spec])
    for issue in issues:
        print(f"! {issue}")
    if not usable:
        raise SystemExit(f"{spec.key} is not available; nothing generated.")

    if args.records == "full":
        records, selection = index.all_cached_records()
        print(
            f"records       : {selection['kept']} of {selection['examined']} cached "
            f"({len(selection['parity_failures'])} parity, {len(selection['over_window'])} over window)"
        )
        if args.index_path is None:
            raise SystemExit(
                "--records full needs --index-path pointing at the index built over those "
                "records (e.g. data/chroma/faceted_full); the default 500-record index would "
                "mark every candidate unretrievable."
            )
    else:
        records = cmr.load_corpus()

    replay = None
    if args.reuse_logs:
        replay = ResponseReplay.from_logs(spec, args.reuse_logs)
        print(f"replayable    : {len(replay)} responses from {len(args.reuse_logs)} log file(s)")

    log = CallLogger()
    qs, rep = build(
        spec,
        logger=log,
        quota=quota,
        records=records,
        out_path=args.out,
        report_path=args.report,
        index_path=args.index_path,
        seed=args.seed,
        temperature=None if args.temperature < 0 else args.temperature,
        workers=args.workers,
        replay=replay,
    )

    print(f"generator     : {rep.generator}")
    print(f"queries total : {len(qs)} (all multi-target)")
    print(f"group sizes   : {rep.group_sizes}")
    print(f"calls         : {rep.llm_calls} paid, {rep.reused_calls} replayed")
    print(f"retries       : {rep.retries}")
    print(f"seeds skipped : {rep.seeds_without_neighbours} (no same-topic neighbour)")
    print("per topic     :")
    for topic in sorted(rep.requested):
        print(f"  {topic:26s} {rep.produced.get(topic, 0):>3} / {rep.requested[topic]}")
    if rep.dropped:
        print(f"dropped       : {len(rep.dropped)}")
        for d in rep.dropped[:6]:
            ids = ",".join(d.get("concept_ids", ["-"]))
            print(f"  {ids[:48]:<48} {d['reason']}")

    issues = verify(rep)
    if issues:
        print("\nQUOTA SHORTFALLS:")
        for issue in issues:
            print(f"  - {issue}")
    print(f"\nwrote {args.out}")
    print(f"log records: {log.counts()}")
