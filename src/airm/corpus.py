"""Build the indexed corpus: ~500 real CMR records, ground-truth-seeded and stratified.

Ordering matters and is not negotiable:

1. **Seed with ground truth.** Every concept-id the SME queries expect goes in
   first. A corpus that does not contain the answer makes Recall@k unmeasurable,
   so this is asserted at the end rather than assumed.
2. **Stratify the remainder** across all 13 CMR science-keyword topics, with
   Land / Ocean / Atmosphere weighted equally as the study requires and every
   other topic guaranteed a floor.
3. **Validate content parity** on every record and exclude -- loudly -- any that
   fails, since a record that renders differently across formats would confound
   the very comparison the corpus exists to support.

Sampling is seeded, so a rebuild against an unchanged CMR gives the same corpus.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import cmr, formats
from .config import (
    CMR_TOPICS,
    CORPUS_PATH,
    CORPUS_SIZE,
    MIN_RECORDS_PER_TOPIC,
    PRIMARY_TOPICS,
    SECONDARY_TOPICS,
    UNCLASSIFIED_TOPIC,
)
from .facets import _umm

#: Fixed so a rebuild is reproducible.
RANDOM_SEED = 20260731

#: How many candidates to pull per topic before sampling. Sampling from a pool
#: several times the quota avoids just taking whatever CMR happens to rank first,
#: which would correlate the corpus with CMR's own relevance ordering.
POOL_PER_TOPIC = 250


@dataclass
class CorpusReport:
    """Everything a reader needs to judge whether the corpus is sound."""

    size: int = 0
    seeded_from_ground_truth: int = 0
    ground_truth_requested: int = 0
    ground_truth_missing: list[str] = field(default_factory=list)
    per_topic: dict[str, int] = field(default_factory=dict)
    parity_failures: list[dict] = field(default_factory=list)
    pool_sizes: dict[str, int] = field(default_factory=dict)
    quota_shortfalls: dict[str, int] = field(default_factory=dict)
    seed: int = RANDOM_SEED

    def to_dict(self) -> dict:
        return {
            "size": self.size,
            "seeded_from_ground_truth": self.seeded_from_ground_truth,
            "ground_truth_requested": self.ground_truth_requested,
            "ground_truth_missing": self.ground_truth_missing,
            "per_topic": dict(sorted(self.per_topic.items(), key=lambda kv: -kv[1])),
            "parity_failures": self.parity_failures,
            "pool_sizes": self.pool_sizes,
            "quota_shortfalls": self.quota_shortfalls,
            "seed": self.seed,
        }


_CANONICAL = set(CMR_TOPICS)


def topic_of(record: dict) -> str:
    """The stratification key for a record.

    Not simply ``ScienceKeywords[0].Topic``. CMR matches a topic search against
    *any* of a record's keywords, so the first one is often a different domain --
    and its casing is whatever the provider typed. Three real failure modes this
    has to absorb, all observed in a live build:

    * **Case drift.** ``Oceans`` and ``OCEANS`` are the same domain; treating
      them as two split the corpus and pushed topics under the floor.
    * **A later keyword holds the real topic.** Scanning all keywords for a
      canonical one recovers the domain instead of bucketing the record as junk.
    * **Junk in the Topic field.** One record carries a NOAA department name
      there. Values matching no canonical topic become ``UNCLASSIFIED`` rather
      than inventing a domain of one record.
    """
    topics = [
        (kw.get("Topic") or "").strip().upper()
        for kw in _umm(record).get("ScienceKeywords") or []
    ]
    for topic in topics:
        if topic in _CANONICAL:
            return topic
    return UNCLASSIFIED_TOPIC


def _quotas(total: int) -> dict[str, int]:
    """Split ``total`` corpus slots across topics.

    Land, Ocean and Atmosphere share half the budget equally -- the study's
    stated requirement -- and the other eleven topics share the rest equally,
    which lands each of them well above :data:`MIN_RECORDS_PER_TOPIC`.
    Representation of the smaller domains is the point; sampling proportional to
    CMR would bury Sun-Earth Interactions (407 collections) under Oceans
    (20,443).

    ``total`` is the whole corpus, not the unfilled remainder: ground-truth
    records are counted against these quotas rather than added on top, so the
    corpus lands on ``size`` instead of ``size`` minus the seed twice over.
    """
    primary_budget = total // 2
    per_primary = primary_budget // len(PRIMARY_TOPICS)
    quotas = {t: per_primary for t in PRIMARY_TOPICS}

    secondary_budget = total - per_primary * len(PRIMARY_TOPICS)
    per_secondary, extra = divmod(secondary_budget, len(SECONDARY_TOPICS))
    for i, topic in enumerate(SECONDARY_TOPICS):
        quotas[topic] = per_secondary + (1 if i < extra else 0)
    return quotas


def build(
    *,
    size: int = CORPUS_SIZE,
    ground_truth: list[str] | None = None,
    out_path: Path | None = CORPUS_PATH,
    seed: int = RANDOM_SEED,
    validate: bool = True,
) -> tuple[list[dict], CorpusReport]:
    """Build the corpus and a report describing exactly what it contains."""
    from .queries import ground_truth_ids, load_queries

    if ground_truth is None:
        ground_truth = ground_truth_ids(load_queries())

    report = CorpusReport(seed=seed)
    rng = random.Random(seed)

    # 1. Ground truth first -- these are non-negotiable members of the corpus.
    report.ground_truth_requested = len(ground_truth)
    seeded = cmr.fetch_by_concept_ids(ground_truth)
    got = {cmr.concept_id(r) for r in seeded}
    report.ground_truth_missing = [c for c in ground_truth if c not in got]
    report.seeded_from_ground_truth = len(seeded)

    selected: dict[str, dict] = {cmr.concept_id(r): r for r in seeded}

    # 2. Stratified fill. Records already seeded count toward their topic quota,
    #    so a domain over-represented in ground truth is not topped up twice.
    already = Counter(topic_of(r) for r in selected.values())
    quotas = _quotas(size)
    shortfalls: dict[str, int] = {}

    for topic in (*PRIMARY_TOPICS, *SECONDARY_TOPICS):
        need = quotas[topic] - already.get(topic, 0)
        if need <= 0:
            report.pool_sizes[topic] = 0
            continue
        pool = [
            r
            for r in cmr.search_by_topic(topic, count=POOL_PER_TOPIC)
            if cmr.concept_id(r) not in selected
        ]
        report.pool_sizes[topic] = len(pool)
        rng.shuffle(pool)
        # A record can match a topic search while its canonical topic is another
        # domain. Take only those that actually land in this bucket, so the
        # quota means what it says.
        taken = 0
        for record in pool:
            if taken >= need:
                break
            if topic_of(record) != topic:
                continue
            selected[cmr.concept_id(record)] = record
            taken += 1
        if taken < need:
            shortfalls[topic] = need - taken
    report.quota_shortfalls = shortfalls

    # 3. Parity gate. A record that renders differently across formats would
    #    confound the comparison, so it leaves the corpus -- and is reported.
    records = list(selected.values())
    if validate:
        kept: list[dict] = []
        gt_set = set(ground_truth)
        for record in records:
            ok, detail = formats.check_record(record)
            if ok:
                kept.append(record)
                continue
            cid = cmr.concept_id(record)
            report.parity_failures.append(
                {
                    "concept_id": cid,
                    "topic": topic_of(record),
                    "is_ground_truth": cid in gt_set,
                    "formats": {k: v for k, v in detail.items() if not v["ok"]},
                }
            )
        records = kept

    # Deterministic order: ground truth first, then by concept-id.
    gt_order = {cid: i for i, cid in enumerate(ground_truth)}
    records.sort(key=lambda r: (gt_order.get(cmr.concept_id(r), 10**9), cmr.concept_id(r)))

    report.size = len(records)
    report.per_topic = dict(Counter(topic_of(r) for r in records))

    if out_path is not None:
        cmr.write_corpus(records, out_path)
        report_path = out_path.with_name("corpus_report.json")
        report_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")

    return records, report


def verify(
    records: list[dict],
    report: CorpusReport,
    ground_truth: list[str],
    *,
    expected_size: int = CORPUS_SIZE,
) -> list[str]:
    """Return the hard-gate violations, empty when the corpus is sound."""
    problems: list[str] = []
    present = {cmr.concept_id(r) for r in records}

    # Quota shortfalls can leave the corpus short while every topic still
    # clears its floor -- without this gate that build reports success.
    if len(records) != expected_size:
        problems.append(
            f"corpus holds {len(records)} records, expected {expected_size}"
            + (f" (quota shortfalls: {report.quota_shortfalls})" if report.quota_shortfalls else "")
        )

    missing_gt = [c for c in ground_truth if c not in present]
    if missing_gt:
        problems.append(
            f"{len(missing_gt)} ground-truth concept-id(s) absent from the corpus: "
            f"{missing_gt[:5]}"
        )

    thin = {t: n for t, n in report.per_topic.items() if n < MIN_RECORDS_PER_TOPIC}
    expected_topics = set(PRIMARY_TOPICS) | set(SECONDARY_TOPICS)
    absent = sorted(expected_topics - set(report.per_topic))
    if absent:
        problems.append(f"topic(s) with no records at all: {absent}")
    if thin:
        problems.append(f"topic(s) below the floor of {MIN_RECORDS_PER_TOPIC}: {thin}")

    gt_parity = [f["concept_id"] for f in report.parity_failures if f["is_ground_truth"]]
    if gt_parity:
        problems.append(f"ground-truth record(s) failed content parity: {gt_parity}")

    return problems


if __name__ == "__main__":  # pragma: no cover - CLI
    import argparse

    from .queries import ground_truth_ids, load_queries

    parser = argparse.ArgumentParser(description="Build the stratified CMR corpus.")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--size", type=int, default=CORPUS_SIZE)
    args = parser.parse_args()

    gt = ground_truth_ids(load_queries())
    recs, rep = build(size=args.size, ground_truth=gt, out_path=CORPUS_PATH if args.build else None)

    print(f"corpus size              : {rep.size}")
    print(f"seeded from ground truth : {rep.seeded_from_ground_truth} / {rep.ground_truth_requested}")
    if rep.ground_truth_missing:
        print(f"  UNRESOLVED             : {rep.ground_truth_missing}")
    print(f"parity failures excluded : {len(rep.parity_failures)}")
    for failure in rep.parity_failures[:5]:
        print(f"  {failure['concept_id']} [{failure['topic']}] {list(failure['formats'])}")
    if rep.quota_shortfalls:
        print(f"quota shortfalls         : {rep.quota_shortfalls}")
    print("records per topic:")
    for topic, n in sorted(rep.per_topic.items(), key=lambda kv: -kv[1]):
        print(f"  {topic:26s} {n:4d}")

    issues = verify(recs, rep, gt, expected_size=args.size)
    if issues:
        print("\nHARD GATE FAILURES:")
        for issue in issues:
            print(f"  - {issue}")
        raise SystemExit(1)
    print("\nall hard gates passed")
    if args.build:
        print(f"wrote {CORPUS_PATH}")
