"""Generate the 120 synthetic evaluation queries.

Generated *from indexed records*, so ground truth is in-corpus by construction --
the failure mode where a query has no findable answer cannot occur here, unlike
the expert slice where 7 targets had already been retired from CMR.

Each candidate must clear three checks before it joins the query set:

1. **It parses.** A model that returns prose instead of JSON is retried.
2. **It does not leak identifiers.** A query naming the short name, DOI, concept
   id, or a long verbatim slice of the title turns retrieval into string
   matching and would inflate every format equally, hiding real differences.
3. **It is retrievable.** The source record must appear in the top-k for at
   least one format. A query no representation can answer measures nothing;
   keeping it would just add noise to every cell equally.

Candidates that fail three times are dropped and reported -- never silently
replaced, since a quietly rebalanced quota is a quietly biased corpus.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from . import cmr, index
from .config import (
    FORMATS,
    MAX_QUERY_REGEN_ATTEMPTS,
    QUERIES_PATH,
    TOP_K,
    ModelSpec,
    synthetic_quota,
)
from .corpus import topic_of
from .facets import facets
from .llm import PURPOSE_QUERY_GEN, CallLogger, LLMError, ask
from .queries import (
    GENERATION_SYSTEM,
    Query,
    _generation_prompt,
    _leaked_identifiers,
    load_queries,
    parse_generated,
    write_queries,
)

RANDOM_SEED = 20260731


@dataclass
class SynthReport:
    generator: str = ""
    requested: dict[str, int] = field(default_factory=dict)
    produced: dict[str, int] = field(default_factory=dict)
    dropped: list[dict] = field(default_factory=list)
    retries: int = 0

    @property
    def total(self) -> int:
        return sum(self.produced.values())

    def to_dict(self) -> dict:
        return {
            "generator": self.generator,
            "requested": self.requested,
            "produced": self.produced,
            "total": self.total,
            "retries": self.retries,
            "dropped": self.dropped,
        }


def _retrievable(
    text: str, concept_id: str, *, k: int = TOP_K, index_path: str | None = None
) -> list[str]:
    """Formats whose top-``k`` contains ``concept_id``.

    ``index_path`` must name the database holding the *same* records the queries
    are being generated from. Checking a full-cache candidate against the 500
    corpus index would ask whether a record that is not in that haystack can be
    found in it, and drop every query as unretrievable.
    """
    found = []
    for fmt in FORMATS:
        hits = {h["concept_id"] for h in index.query(fmt, text, k=k, path=index_path)}
        if concept_id in hits:
            found.append(fmt)
    return found


def generate(
    spec: ModelSpec,
    *,
    records: list[dict] | None = None,
    quota: dict[str, int] | None = None,
    logger: CallLogger | None = None,
    seed: int = RANDOM_SEED,
    k: int = TOP_K,
    check_retrievable: bool = True,
    index_path: str | None = None,
    temperature: float | None = 0.7,
) -> tuple[list[Query], SynthReport]:
    """Generate one query per sampled record until each topic quota is met."""
    records = records if records is not None else cmr.load_corpus()
    quota = quota or synthetic_quota()
    report = SynthReport(generator=spec.key, requested=dict(quota))
    rng = random.Random(seed)

    by_topic: dict[str, list[dict]] = {}
    for record in records:
        by_topic.setdefault(topic_of(record), []).append(record)

    queries: list[Query] = []
    for topic in sorted(quota):
        want = quota[topic]
        pool = list(by_topic.get(topic, []))
        rng.shuffle(pool)
        produced = 0

        if len(pool) < want:
            report.dropped.append(
                {
                    "topic": topic,
                    "reason": "too few corpus records to draw from",
                    "available": len(pool),
                    "requested": want,
                }
            )

        for record in pool:
            if produced >= want:
                break
            cid = cmr.concept_id(record)
            payload = facets(record)
            prompt = _generation_prompt(payload)
            accepted: Query | None = None

            for attempt in range(1, MAX_QUERY_REGEN_ATTEMPTS + 1):
                if attempt > 1:
                    report.retries += 1
                try:
                    response = ask(
                        spec,
                        prompt,
                        system=GENERATION_SYSTEM,
                        purpose=PURPOSE_QUERY_GEN,
                        logger=logger,
                        json_mode=True,
                        temperature=temperature,
                        topic=topic,
                        concept_id=cid,
                        attempt=attempt,
                    )
                except LLMError as exc:
                    report.dropped.append(
                        {"concept_id": cid, "topic": topic, "reason": f"llm error: {exc}"}
                    )
                    break

                text = parse_generated(response.text)
                if not text:
                    continue

                leaks = _leaked_identifiers(text, payload)
                if leaks:
                    if attempt == MAX_QUERY_REGEN_ATTEMPTS:
                        report.dropped.append(
                            {
                                "concept_id": cid,
                                "topic": topic,
                                "reason": "leaked identifiers",
                                "detail": leaks,
                                "query": text,
                            }
                        )
                    continue

                formats_hit = (
                    _retrievable(text, cid, k=k, index_path=index_path)
                    if check_retrievable
                    else list(FORMATS)
                )
                if not formats_hit:
                    if attempt == MAX_QUERY_REGEN_ATTEMPTS:
                        report.dropped.append(
                            {
                                "concept_id": cid,
                                "topic": topic,
                                "reason": "not retrievable by any format",
                                "query": text,
                            }
                        )
                    continue

                accepted = Query(
                    query_id=f"syn-{len(queries) + 1:03d}",
                    text=text,
                    expected_concept_ids=[cid],
                    source="synthetic",
                    topic=topic,
                    expected_titles=[payload.get("title", "")],
                )
                break

            if accepted:
                queries.append(accepted)
                produced += 1

        report.produced[topic] = produced

    return queries, report


def build(
    spec: ModelSpec,
    *,
    logger: CallLogger | None = None,
    out_path: Path = QUERIES_PATH,
    report_path: Path | None = None,
    **kwargs,
) -> tuple[list[Query], SynthReport]:
    """Generate synthetic queries and write the combined SME + synthetic set.

    ``report_path`` defaults to ``synthetic_report.json`` beside ``out_path``,
    which is right for the canonical set and wrong for any other: a second run
    writing to a different ``out_path`` would otherwise overwrite the first run's
    report and leave a query set whose provenance describes a different one.
    """
    synthetic, report = generate(spec, logger=logger, **kwargs)
    combined = load_queries() + synthetic
    write_queries(combined, out_path)
    if report_path is None:
        report_path = (
            out_path.with_name("synthetic_report.json")
            if out_path == QUERIES_PATH
            else out_path.with_name(f"{out_path.stem}_report.json")
        )
    report_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n")
    return combined, report


def verify(report: SynthReport) -> list[str]:
    """Hard-gate violations for the synthetic half."""
    problems: list[str] = []
    for topic, want in report.requested.items():
        got = report.produced.get(topic, 0)
        if got < want:
            problems.append(f"{topic}: produced {got} of {want} requested queries")
    return problems


if __name__ == "__main__":  # pragma: no cover - CLI
    import argparse

    from .config import ModelSpec as Spec

    parser = argparse.ArgumentParser(description="Generate the synthetic query set.")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="qwen3.6:latest")
    parser.add_argument("--limit", type=int, help="cap per-topic quota (smoke run)")
    parser.add_argument(
        "--total",
        type=int,
        default=None,
        help="synthetic queries to generate; the per-topic shape is scaled, not reshaped",
    )
    parser.add_argument(
        "--records",
        choices=("corpus", "full"),
        default="corpus",
        help="draw from the 500 stratified corpus or from every gate-passing cached record",
    )
    parser.add_argument(
        "--index-path",
        default=None,
        help="Chroma database for the retrievability check; must hold the same records as --records",
    )
    parser.add_argument("--out", type=Path, default=QUERIES_PATH)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="pass -1 to omit the parameter for models that reject it",
    )
    args = parser.parse_args()

    quota = synthetic_quota(args.total) if args.total else synthetic_quota()
    if args.limit:
        quota = {t: min(n, args.limit) for t, n in quota.items()}

    spec = Spec(args.provider, args.model)
    # Checked before anything expensive: a retired model name is a 404 on every
    # one of several hundred calls, and the run would burn the whole record pool
    # discovering it.
    from .llm import resolve_models

    usable, issues = resolve_models([spec])
    for issue in issues:
        print(f"! {issue}")
    if not usable:
        raise SystemExit(f"{spec.key} is not available; nothing generated.")

    records = None
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
    )

    print(f"generator     : {rep.generator}")
    print(f"queries total : {len(qs)} ({rep.total} synthetic + {len(qs) - rep.total} SME)")
    print(f"retries       : {rep.retries}")
    print("per topic     :")
    for topic in sorted(rep.requested):
        print(f"  {topic:26s} {rep.produced.get(topic, 0):>3} / {rep.requested[topic]}")
    if rep.dropped:
        print(f"dropped       : {len(rep.dropped)}")
        for d in rep.dropped[:6]:
            print(f"  {d.get('concept_id', '-'):<24} {d['reason']}")

    issues = verify(rep)
    if issues:
        print("\nQUOTA SHORTFALLS:")
        for issue in issues:
            print(f"  - {issue}")
    print(f"\nwrote {args.out}")
    print(f"log records: {log.counts()}")
