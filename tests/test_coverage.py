"""Tests for the raw-CMR coverage declaration.

The formats carry identical content *to each other* — ``test_formats.py`` proves
that. These tests pin the separate, weaker claim: whatever the canonical payload
drops from the record CMR actually published is declared and deliberate, so a new
UMM field cannot silently vanish from the study.
"""

from __future__ import annotations

import json

import pytest

from airm import coverage
from airm.config import CORPUS_PATH, SAMPLE_DATA_DIR

SAMPLE_UMM = json.loads((SAMPLE_DATA_DIR / "umm_json.json").read_text())


# --------------------------------------------------------------------------- #
# Path walking.
# --------------------------------------------------------------------------- #


def test_scalar_paths_collapses_indices():
    paths = coverage.scalar_paths({"a": [{"b": 1}, {"b": 2}], "c": "x"})
    assert sorted(paths) == ["a[].b", "a[].b", "c"]


def test_generalise():
    assert coverage.generalise("p[0].i[12].n") == "p[].i[].n"


def test_empty_containers_contribute_no_paths():
    assert coverage.scalar_paths({"a": [], "b": {}}) == []


# --------------------------------------------------------------------------- #
# Prefix matching.
# --------------------------------------------------------------------------- #


def test_covers_matches_only_at_a_structural_boundary():
    # The bug this guards: a bare `startswith` lets umm.Version swallow
    # umm.VersionDescription, silently marking deferred content as carried.
    assert coverage._covers("umm.Version", "umm.Version")
    assert not coverage._covers("umm.Version", "umm.VersionDescription")
    assert coverage._covers("umm.RelatedUrls", "umm.RelatedUrls[].URL")
    assert coverage._covers("umm.DOI", "umm.DOI.DOI")


def test_longest_prefix_wins():
    # meta.concept-id is carried even though the rest of meta is excluded.
    assert coverage._classify("meta.concept-id")[0] == "CARRIED"
    assert coverage._classify("meta.revision-id")[0] == "EXCLUDED"
    # DataCenters[].ShortName is partial; the wider subtree is deferred.
    assert coverage._classify("umm.DataCenters[].ShortName")[0] == "PARTIAL"
    assert coverage._classify("umm.DataCenters[].LongName")[0] == "DEFERRED"


def test_unknown_path_is_undeclared():
    assert coverage._classify("umm.SomethingCMRAddedYesterday")[0] == "UNDECLARED"


# --------------------------------------------------------------------------- #
# The declaration itself.
# --------------------------------------------------------------------------- #


def test_every_declared_prefix_is_unique_across_buckets():
    tables = [coverage.CARRIED, coverage.PARTIAL, coverage.EXCLUDED, coverage.DEFERRED]
    seen: dict[str, int] = {}
    for i, table in enumerate(tables):
        for prefix in table:
            assert prefix not in seen, f"{prefix} declared in two buckets"
            seen[prefix] = i


def test_every_declaration_has_a_reason():
    for table in (coverage.EXCLUDED, coverage.DEFERRED):
        for prefix, note in table.items():
            assert note.strip(), f"{prefix} declared without a reason"


def test_sample_record_is_fully_declared():
    coverage.verify([SAMPLE_UMM])


def test_verify_rejects_an_undeclared_path():
    record = {"umm": {"AFieldNobodyDeclared": "surprise"}}
    with pytest.raises(coverage.CoverageError, match="AFieldNobodyDeclared"):
        coverage.verify([record])


def test_report_buckets_partition_the_corpus():
    res = coverage.report([SAMPLE_UMM])
    assert sum(res["totals"].values()) == res["values"]


# --------------------------------------------------------------------------- #
# Whole corpus.
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not CORPUS_PATH.exists(), reason="corpus not built")
def test_corpus_has_no_undeclared_paths():
    with CORPUS_PATH.open() as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    coverage.verify(records)
