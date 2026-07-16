"""Unit tests for the audit-driven correctness/faithfulness fixes.

Covers the metric edge cases, the content-held renderer registry, the
cluster-aware bootstrap + split, and the ground-truth coverage guard.
"""

from __future__ import annotations

import pytest

from airm import metrics, run
from airm.queries import Query, split_by_cluster
from airm.representations import (
    RENDERERS_FAIR,
    facets,
    metadata_as_text,
    umm_facet_subset,
)

from tests.test_harness import DISTRACTOR, RECORD


# --------------------------------------------------------------------------- #
# Metric edge cases.
# --------------------------------------------------------------------------- #


def test_recall_and_ndcg_dedup_duplicate_ids():
    # A ranked list with a duplicated relevant id must not score above 1.0.
    ranked = ["c", "c", "c"]
    rel = {"c"}
    assert metrics.recall_at_k(ranked, rel, 3) == pytest.approx(1.0)
    assert metrics.ndcg_at_k(ranked, rel, 3) == pytest.approx(1.0)


def test_recall_multi_relevant_counts_distinct():
    ranked = ["a", "a", "b"]  # 'a' duplicated
    rel = {"a", "b", "z"}
    # two distinct relevant ids hit out of three -> 2/3, not 3/3
    assert metrics.recall_at_k(ranked, rel, 3) == pytest.approx(2 / 3)


def test_temporal_validator_accepts_real_cmr_forms():
    v = metrics.value_format_valid
    assert v("temporal", "2018-10-14T00:00:00.000Z")  # fractional seconds
    assert v("temporal", "2020-01-01T00:00:00+00:00")  # numeric offset
    assert v("temporal", "2020-01-01T00:00:00Z,2020-12-31T23:59:59Z")  # range
    assert v("temporal", "2020-01-01T00:00:00Z,")  # open-ended range


def test_temporal_validator_rejects_bad_values():
    v = metrics.value_format_valid
    assert not v("temporal", "2020-13-40T25:99:99Z")  # out-of-range fields
    assert not v("temporal", "Dec 2024")
    assert not v("temporal", "2020-12-31T00:00:00Z,2020-01-01T00:00:00Z")  # end < start


# --------------------------------------------------------------------------- #
# Cluster bootstrap.
# --------------------------------------------------------------------------- #


def test_clustered_bootstrap_wider_than_naive_for_correlated_data():
    # 40 identical-within-cluster observations across 2 clusters. Treating them
    # as independent gives a tight CI; clustering by their 2 groups is wider.
    values = [1.0] * 20 + [0.0] * 20
    clusters = ["g1"] * 20 + ["g2"] * 20
    _, lo_n, hi_n = metrics.bootstrap_ci(values, seed=0)
    _, lo_c, hi_c = metrics.bootstrap_ci_clustered(values, clusters, seed=0)
    assert (hi_c - lo_c) > (hi_n - lo_n)


def test_clustered_bootstrap_single_cluster_falls_back():
    values = [1.0, 0.0, 1.0]
    clusters = ["only"] * 3
    a = metrics.bootstrap_ci_clustered(values, clusters, seed=0)
    b = metrics.bootstrap_ci(values, seed=0)
    assert a == b


# --------------------------------------------------------------------------- #
# Content-held renderers.
# --------------------------------------------------------------------------- #


def test_umm_facet_subset_holds_content_constant():
    # The fair raw-JSON renderer must not carry fields outside the facet set.
    subset = umm_facet_subset(RECORD)["umm"]
    assert set(subset) <= {
        "EntryTitle", "Abstract", "DataCenters", "Platforms", "ScienceKeywords",
        "SpatialExtent", "TemporalExtents", "ProcessingLevel", "CollectionProgress",
        "Version",
    }
    # facets recovered from the subset match the originals (content preserved).
    assert facets({"meta": RECORD["meta"], "umm": subset})["title"] == facets(RECORD)["title"]


def test_fair_registry_curated_renderers_unchanged():
    # flattened_jsonld / metadata_as_text already use facets, so the fair
    # variants are identical to the defaults.
    assert RENDERERS_FAIR["metadata_as_text"](RECORD) == metadata_as_text(RECORD)


def test_fair_registry_all_nonempty_and_distinct():
    outs = {name: fn(RECORD) for name, fn in RENDERERS_FAIR.items()}
    assert all(o.strip() for o in outs.values())
    assert len(set(outs.values())) == len(outs)


# --------------------------------------------------------------------------- #
# Split + coverage.
# --------------------------------------------------------------------------- #


def _q(i, gold):
    return Query(id=f"q{i}", question=f"question {i}", relevant=[gold])


def test_split_keeps_clusters_together():
    # Two gold datasets, several paraphrases each. No gold should span splits.
    queries = [_q(i, "C1") for i in range(6)] + [_q(i, "C2") for i in range(6, 12)]
    splits = split_by_cluster(queries, ratios=(0.5, 0.0, 0.5), seed=1)
    where = {}
    for name, qs in splits.items():
        for q in qs:
            where.setdefault(q.relevant[0], set()).add(name)
    assert all(len(s) == 1 for s in where.values())  # each gold in exactly one split


def test_coverage_flags_dangling_and_empty():
    corpus = [RECORD, DISTRACTOR]  # ids C123-TEST, C999-TEST
    queries = [
        Query(id="ok", question="x", relevant=["C123-TEST"]),
        Query(id="dangling", question="y", relevant=["C-DOES-NOT-EXIST"]),
        Query(id="empty", question="z", relevant=[]),
    ]
    cov = run.check_relevance_coverage(corpus, queries)
    assert cov.n_ok == 1
    assert "dangling" in cov.missing_relevant
    assert cov.empty_relevant == ["empty"]


def test_summarize_accepts_clusters():
    corpus = [RECORD, DISTRACTOR]
    queries = [Query(id="q0", question="vegetation canopy height", relevant=["C123-TEST"])]
    from airm.representations import RENDERERS
    results = []
    for name, fn in RENDERERS.items():
        results += run.evaluate_retrieval(corpus, queries, fn, name, k=2)
    clusters = {"q0": "C123-TEST"}
    summary = run.summarize(results, clusters=clusters)
    assert set(summary) == set(RENDERERS)
