"""Tests for the SME query set and its ground truth.

The spreadsheet is small but messy: duplicate query rows, Unicode dashes inside
concept-ids, free-text notes in an id column, and retired collections. Every one
of those is a way to silently corrupt ground truth, so each gets a test.
"""

from __future__ import annotations

import pytest

from airm import queries
from airm.config import SME_XLSX_PATH

pytestmark = pytest.mark.skipif(
    not SME_XLSX_PATH.exists(), reason="sample_sme_queries.xlsx not present"
)


@pytest.fixture(scope="module")
def offline():
    """The SME set built without touching CMR -- retired ids still included."""
    return queries.build_sme_queries(resolve=False)


# --------------------------------------------------------------------------- #
# Normalisation.
# --------------------------------------------------------------------------- #


def test_non_breaking_hyphen_is_folded_to_ascii():
    """Three cells use U+2011, which looks right and compares wrong."""
    assert queries.normalise_concept_id("C2531308461‑NSIDC_ECS") == "C2531308461-NSIDC_ECS"


@pytest.mark.parametrize("dash", list(queries.DASHES))
def test_every_unicode_dash_folds(dash):
    assert queries.normalise_concept_id(f"C1{dash}PROV") == "C1-PROV"


def test_surrounding_and_internal_whitespace_is_stripped():
    assert queries.normalise_concept_id("  C1235316218-GES_DISC ") == "C1235316218-GES_DISC"
    assert queries.normalise_concept_id("C123 -GES_DISC") == "C123-GES_DISC"


def test_split_keeps_the_original_cell_text_for_reporting():
    pairs = queries.split_concept_ids("C1-A, Not in CMR")
    assert pairs == [("C1-A", "C1-A"), ("Not in CMR", "NotinCMR")]


def test_concept_id_pattern_accepts_real_ids_and_rejects_notes():
    assert queries.CONCEPT_ID_RE.match("C3205181648-NSIDC_CPRD")
    assert not queries.CONCEPT_ID_RE.match("SENTINEL-1A_SP_GRD_FULL")
    assert not queries.CONCEPT_ID_RE.match("NotinCMR")


# --------------------------------------------------------------------------- #
# Sheet parsing and merging.
# --------------------------------------------------------------------------- #


def test_reads_all_data_rows():
    assert len(queries.read_sme_rows()) == 25


def test_duplicate_rows_merge_into_one_query_per_research_question(offline):
    """25 rows carry 11 questions; scoring rows would count some 3-4 times."""
    qs, report = offline
    assert report["rows_read"] == 25
    assert report["distinct_queries"] == 11
    assert len(qs) == 11
    assert len({q.text for q in qs}) == 11


def test_merged_query_unions_the_expected_ids_across_its_rows(offline):
    qs, _ = offline
    fires = next(q for q in qs if "savanna" in q.text)
    # Three rows in the sheet, three ids each.
    assert len(fires.expected_concept_ids) == 9
    assert len(set(fires.expected_concept_ids)) == 9


def test_non_concept_id_tokens_are_dropped_and_reported(offline):
    _, report = offline
    dropped = {t for tokens in report["dropped"]["not_a_concept_id"].values() for t in tokens}
    assert dropped == {"Not in CMR", "SENTINEL-1A_SP_GRD_FULL"}


def test_dropped_report_shows_the_cell_text_a_reader_would_recognise(offline):
    _, report = offline
    dropped = {t for tokens in report["dropped"]["not_a_concept_id"].values() for t in tokens}
    assert "Not in CMR" in dropped  # not the stripped "NotinCMR"


def test_well_formed_id_count(offline):
    _, report = offline
    assert report["concept_ids_well_formed"] == 41
    assert report["concept_id_tokens_seen"] == 43


def test_query_ids_are_unique_and_stable(offline):
    qs, _ = offline
    ids = [q.query_id for q in qs]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
    assert all(q.source == "sme" for q in qs)


def test_ground_truth_ids_are_deduplicated(offline):
    qs, _ = offline
    unique = queries.ground_truth_ids(qs)
    total_slots = sum(len(q.expected_concept_ids) for q in qs)
    assert len(unique) < total_slots  # some collections answer several questions
    assert unique == sorted(set(unique))


# --------------------------------------------------------------------------- #
# Persistence.
# --------------------------------------------------------------------------- #


def test_queries_round_trip_through_jsonl(tmp_path, offline):
    qs, _ = offline
    path = tmp_path / "gt.jsonl"
    queries.write_queries(qs, path)
    assert queries.load_queries(path) == qs


def test_load_queries_missing_file_names_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="airm.queries"):
        queries.load_queries(tmp_path / "nope.jsonl")


def test_eval_queries_prefer_the_combined_set(tmp_path, monkeypatch, offline):
    """Experiment 2 must see SME + synthetic, not silently just the 11 SME."""
    from airm import config

    qs, _ = offline
    gt_path = tmp_path / "gt.jsonl"
    combined_path = tmp_path / "queries.jsonl"
    queries.write_queries(qs, gt_path)
    monkeypatch.setattr(queries, "GT_PATH", gt_path)
    monkeypatch.setattr(config, "QUERIES_PATH", combined_path)

    # Before synth has run: falls back to the SME ground truth.
    assert len(queries.load_eval_queries()) == len(qs)

    synthetic = queries.Query("syn-001", "q?", ["C1-A"], "synthetic", "OCEANS")
    queries.write_queries([*qs, synthetic], combined_path)
    combined = queries.load_eval_queries()
    assert len(combined) == len(qs) + 1
    assert combined[-1].source == "synthetic"


# --------------------------------------------------------------------------- #
# Live: the retired-id decision.
# --------------------------------------------------------------------------- #

#: Confirmed retired against the live CMR API. All NSIDC_ECS; the study drops
#: them rather than remapping to successor collections, so ground truth only
#: contains collections CMR returns today.
RETIRED = {
    "C1542606326-NSIDC_ECS",
    "C2399557265-NSIDC_ECS",
    "C2531308461-NSIDC_ECS",
    "C2537927247-NSIDC_ECS",
    "C2776463935-NSIDC_ECS",
    "C2776463943-NSIDC_ECS",
    "C3383993430-NSIDC_ECS",
}


@pytest.mark.live
def test_live_retired_ids_are_exactly_the_expected_seven():
    _, report = queries.build_sme_queries()
    assert set(report["dropped"]["retired"]) == RETIRED


@pytest.mark.live
def test_live_every_surviving_id_resolves_in_cmr():
    from airm.cmr import missing_concept_ids

    qs, report = queries.build_sme_queries()
    kept = queries.ground_truth_ids(qs)
    assert len(kept) == 34
    assert report["concept_ids_kept"] == 34
    assert missing_concept_ids(kept) == []


@pytest.mark.live
def test_live_no_query_is_lost_to_the_retired_drop():
    """Dropping 7 ids must not empty out any of the 11 research questions."""
    qs, report = queries.build_sme_queries()
    assert len(qs) == 11
    assert report["queries_dropped_for_having_no_resolvable_ids"] == []
    assert all(q.expected_concept_ids for q in qs)
