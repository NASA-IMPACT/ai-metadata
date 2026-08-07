"""Tests for the corpus build and its hard gates.

The gates matter more than the build: a corpus missing a ground-truth record
makes Recall@k unmeasurable, and a corpus whose topic buckets have silently split
on capitalisation makes the per-domain breakdown a fiction. Both happened during
development and both are pinned here.
"""

from __future__ import annotations

import json

import pytest

from airm import corpus
from airm.config import (
    CMR_TOPICS,
    MIN_RECORDS_PER_TOPIC,
    PRIMARY_TOPICS,
    SECONDARY_TOPICS,
    UNCLASSIFIED_TOPIC,
    synthetic_quota,
)


def _record(cid: str, *topics: str, title: str | None = None) -> dict:
    return {
        "meta": {"concept-id": cid},
        "umm": {
            "EntryTitle": title or f"Collection {cid}",
            "ScienceKeywords": [{"Category": "EARTH SCIENCE", "Topic": t} for t in topics],
        },
    }


# --------------------------------------------------------------------------- #
# Topic normalisation.
# --------------------------------------------------------------------------- #


def test_topic_is_upper_cased():
    """`Oceans` and `OCEANS` are one domain; treating them as two split the corpus."""
    assert corpus.topic_of(_record("C1-A", "Oceans")) == "OCEANS"
    assert corpus.topic_of(_record("C2-A", "OCEANS")) == "OCEANS"
    assert corpus.topic_of(_record("C3-A", "Sun-Earth Interactions")) == "SUN-EARTH INTERACTIONS"


def test_topic_scans_past_a_non_canonical_first_keyword():
    """CMR matches a topic search on any keyword, so the first is often junk."""
    junk = "National Centers for Coastal Ocean Science, NOAA"
    assert corpus.topic_of(_record("C4-A", junk, "OCEANS")) == "OCEANS"


def test_unrecognised_topics_do_not_become_a_domain_of_one():
    assert corpus.topic_of(_record("C5-A", "-Na-")) == UNCLASSIFIED_TOPIC
    assert corpus.topic_of(_record("C6-A")) == UNCLASSIFIED_TOPIC


def test_biological_classification_is_a_recognised_topic():
    """4,731 collections; omitting it bucketed them all as unclassified."""
    assert "BIOLOGICAL CLASSIFICATION" in CMR_TOPICS
    assert corpus.topic_of(_record("C7-A", "Biological Classification")) == (
        "BIOLOGICAL CLASSIFICATION"
    )


# --------------------------------------------------------------------------- #
# Quotas.
# --------------------------------------------------------------------------- #


def test_quotas_cover_every_topic_and_sum_to_the_corpus_size():
    quotas = corpus._quotas(500)
    assert set(quotas) == set(CMR_TOPICS)
    assert sum(quotas.values()) == 500


def test_primary_topics_get_equal_shares():
    quotas = corpus._quotas(500)
    shares = {quotas[t] for t in PRIMARY_TOPICS}
    assert len(shares) == 1, "Land/Ocean/Atmosphere must be equally represented"
    assert sum(quotas[t] for t in PRIMARY_TOPICS) == pytest.approx(250, abs=3)


def test_every_secondary_topic_clears_the_floor():
    quotas = corpus._quotas(500)
    assert all(quotas[t] >= MIN_RECORDS_PER_TOPIC for t in SECONDARY_TOPICS)


def test_quotas_are_over_the_whole_corpus_not_the_unfilled_remainder():
    """Subtracting the ground-truth seed twice produced 468 records, not 500."""
    assert sum(corpus._quotas(500).values()) == 500


def test_synthetic_quota_matches_the_declared_total():
    quota = synthetic_quota()
    assert set(quota) == set(CMR_TOPICS)
    assert sum(quota.values()) == 120
    assert len({quota[t] for t in PRIMARY_TOPICS}) == 1
    assert all(quota[t] >= 2 for t in SECONDARY_TOPICS)


# --------------------------------------------------------------------------- #
# Build, with CMR stubbed out.
# --------------------------------------------------------------------------- #


@pytest.fixture
def stub_cmr(monkeypatch):
    """A fake CMR with 40 well-formed records per topic."""
    pools = {
        topic: [_record(f"C{i:03d}{n}-PROV", topic) for i in range(40)]
        for n, topic in enumerate(CMR_TOPICS)
    }
    gt = {cid: _record(cid, "OCEANS") for cid in ("C900-GT", "C901-GT", "C902-GT")}

    monkeypatch.setattr(
        corpus.cmr,
        "fetch_by_concept_ids",
        lambda cids, **kw: [gt[c] for c in cids if c in gt],
    )
    monkeypatch.setattr(
        corpus.cmr, "search_by_topic", lambda topic, count=0, **kw: list(pools[topic])
    )
    return list(gt)


def test_build_seeds_every_ground_truth_record(stub_cmr, tmp_path):
    records, report = corpus.build(
        size=100, ground_truth=stub_cmr, out_path=None, validate=False
    )
    present = {r["meta"]["concept-id"] for r in records}
    assert set(stub_cmr) <= present
    assert report.seeded_from_ground_truth == len(stub_cmr)
    assert report.ground_truth_missing == []


def test_build_puts_ground_truth_first_for_a_stable_ordering(stub_cmr):
    records, _ = corpus.build(size=100, ground_truth=stub_cmr, out_path=None, validate=False)
    head = [r["meta"]["concept-id"] for r in records[: len(stub_cmr)]]
    assert head == stub_cmr


def test_build_is_deterministic_under_a_fixed_seed(stub_cmr):
    a, _ = corpus.build(size=100, ground_truth=stub_cmr, out_path=None, validate=False)
    b, _ = corpus.build(size=100, ground_truth=stub_cmr, out_path=None, validate=False)
    assert [r["meta"]["concept-id"] for r in a] == [r["meta"]["concept-id"] for r in b]


def test_build_reports_unresolvable_ground_truth_rather_than_ignoring_it(stub_cmr):
    _, report = corpus.build(
        size=50, ground_truth=[*stub_cmr, "C999-GONE"], out_path=None, validate=False
    )
    assert report.ground_truth_missing == ["C999-GONE"]


def test_build_writes_corpus_and_report(stub_cmr, tmp_path):
    out = tmp_path / "corpus.jsonl"
    records, _ = corpus.build(size=60, ground_truth=stub_cmr, out_path=out, validate=False)
    assert len(out.read_text().strip().splitlines()) == len(records)
    written = json.loads((tmp_path / "corpus_report.json").read_text())
    assert written["size"] == len(records)
    assert written["seed"] == corpus.RANDOM_SEED


def test_parity_failures_are_excluded_and_recorded(stub_cmr, monkeypatch):
    """A record that renders differently across formats must leave the corpus."""
    clean, _ = corpus.build(size=100, ground_truth=stub_cmr, out_path=None, validate=False)
    victim = next(
        r["meta"]["concept-id"] for r in clean if r["meta"]["concept-id"] not in stub_cmr
    )

    def check(record):
        if record["meta"]["concept-id"] == victim:
            return False, {"toon": {"ok": False, "missing": [("title", "x")], "extra": []}}
        return True, {}

    monkeypatch.setattr(corpus.formats, "check_record", check)
    records, report = corpus.build(size=100, ground_truth=stub_cmr, out_path=None)

    assert victim not in {r["meta"]["concept-id"] for r in records}
    assert [f["concept_id"] for f in report.parity_failures] == [victim]
    assert report.parity_failures[0]["is_ground_truth"] is False
    assert "toon" in report.parity_failures[0]["formats"]


# --------------------------------------------------------------------------- #
# Hard gates.
# --------------------------------------------------------------------------- #


def test_verify_passes_a_sound_corpus(stub_cmr):
    # 240 puts every topic quota within the 40-record stub pools; 500 would not.
    records, report = corpus.build(
        size=240, ground_truth=stub_cmr, out_path=None, validate=False
    )
    assert corpus.verify(records, report, stub_cmr, expected_size=240) == []


def test_verify_fails_when_a_ground_truth_record_is_absent(stub_cmr):
    records, report = corpus.build(
        size=100, ground_truth=stub_cmr, out_path=None, validate=False
    )
    problems = corpus.verify(records, report, [*stub_cmr, "C999-GONE"], expected_size=100)
    assert any("ground-truth" in p for p in problems)


def test_verify_fails_when_the_corpus_is_short(stub_cmr):
    """Quota shortfalls above the topic floor must not read as success."""
    records, report = corpus.build(
        size=100, ground_truth=stub_cmr, out_path=None, validate=False
    )
    problems = corpus.verify(records, report, stub_cmr, expected_size=500)
    assert any("expected 500" in p for p in problems)
    assert corpus.verify(records, report, stub_cmr, expected_size=100) == []


def test_verify_fails_when_a_topic_is_missing_entirely():
    report = corpus.CorpusReport(size=2, per_topic={"OCEANS": 2})
    problems = corpus.verify([], report, [], expected_size=0)
    assert any("no records at all" in p for p in problems)


def test_verify_fails_when_a_topic_is_below_the_floor():
    report = corpus.CorpusReport(
        size=100, per_topic={t: 10 for t in CMR_TOPICS} | {"PALEOCLIMATE": 1}
    )
    problems = corpus.verify([], report, [], expected_size=0)
    assert any("below the floor" in p for p in problems)


def test_verify_fails_when_a_ground_truth_record_failed_parity():
    report = corpus.CorpusReport(
        size=1,
        per_topic={t: 10 for t in CMR_TOPICS},
        parity_failures=[{"concept_id": "C900-GT", "topic": "OCEANS", "is_ground_truth": True}],
    )
    problems = corpus.verify([], report, [], expected_size=0)
    assert any("failed content parity" in p for p in problems)


# --------------------------------------------------------------------------- #
# Live: the built corpus on disk.
# --------------------------------------------------------------------------- #


@pytest.mark.live
def test_live_built_corpus_satisfies_every_gate():
    from airm.cmr import load_corpus
    from airm.queries import ground_truth_ids, load_queries

    records = load_corpus()
    report = corpus.CorpusReport(
        size=len(records),
        per_topic=dict(__import__("collections").Counter(corpus.topic_of(r) for r in records)),
    )
    gt = ground_truth_ids(load_queries())

    assert len(records) == 500
    assert corpus.verify(records, report, gt) == []
    assert len({r["meta"]["concept-id"] for r in records}) == 500
    # The three required domains must be equally represented.
    assert len({report.per_topic[t] for t in PRIMARY_TOPICS}) == 1
