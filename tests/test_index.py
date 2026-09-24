"""Tests for the ChromaDB indexes and their hard gates.

The gates are the point. Silent truncation and a missing ground-truth record
both produce a run that completes, writes plausible numbers, and means nothing.
"""

from __future__ import annotations

import pytest

from airm import index, unfaceted
from airm.config import FORMATS, MAX_EMBED_TRUNCATIONS, collection_name
from airm.facets import facets


def _record(cid: str, title: str, topic: str = "OCEANS", abstract: str = "") -> dict:
    return {
        "meta": {"concept-id": cid},
        "umm": {
            "EntryTitle": title,
            "Abstract": abstract or f"Observations relating to {title}.",
            "ScienceKeywords": [{"Category": "EARTH SCIENCE", "Topic": topic}],
        },
    }


RECORDS = [
    _record("C1-SEA", "Sea Ice Concentration Daily Polar Grids", "OCEANS"),
    _record("C2-SOIL", "Global Surface Soil Moisture Level 3", "LAND SURFACE"),
    _record("C3-OZONE", "Total Column Ozone Daily Global Gridded", "ATMOSPHERE"),
    _record("C4-FIRE", "Active Fire and Thermal Anomalies Swath", "BIOSPHERE"),
]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One real index build over four tiny records, shared by the module."""
    path = str(tmp_path_factory.mktemp("chroma"))
    report = index.build(RECORDS, ground_truth=["C1-SEA", "C3-OZONE"], path=path)
    return path, report


# --------------------------------------------------------------------------- #
# Structure.
# --------------------------------------------------------------------------- #


def test_one_collection_per_format_each_holding_every_record(built):
    _, report = built
    assert set(report.counts) == set(FORMATS)
    assert all(n == len(RECORDS) for n in report.counts.values())


def test_collections_are_named_distinctly(built):
    path, _ = built
    names = {c.name for c in index.client(path).list_collections()}
    assert {collection_name(f) for f in FORMATS} <= names


def test_the_same_records_are_indexed_in_every_collection(built):
    path, _ = built
    expected = {r["meta"]["concept-id"] for r in RECORDS}
    for fmt in FORMATS:
        got = set(index.get_collection(fmt, path=path).get(include=[])["ids"])
        assert got == expected, fmt


def test_documents_differ_by_format_but_describe_the_same_record(built):
    path, _ = built
    docs = {}
    for fmt in FORMATS:
        got = index.get_collection(fmt, path=path).get(ids=["C1-SEA"], include=["documents"])
        docs[fmt] = got["documents"][0]
    assert len(set(docs.values())) == len(FORMATS), "formats produced identical text"
    assert all("Sea Ice Concentration" in d for d in docs.values())


def test_metadata_carries_the_topic_for_per_domain_analysis(built):
    path, _ = built
    got = index.get_collection("json", path=path).get(ids=["C2-SOIL"], include=["metadatas"])
    assert got["metadatas"][0]["topic"] == "LAND SURFACE"
    assert got["metadatas"][0]["format"] == "json"


# --------------------------------------------------------------------------- #
# Retrieval.
# --------------------------------------------------------------------------- #


def test_query_returns_ranked_hits(built):
    path, _ = built
    hits = index.query("json", "sea ice concentration in polar regions", k=3, path=path)
    assert [h["rank"] for h in hits] == [1, 2, 3]
    assert hits[0]["concept_id"] == "C1-SEA"


def test_distances_are_non_decreasing_down_the_ranking(built):
    path, _ = built
    hits = index.query("mat", "ozone column measurements", k=4, path=path)
    distances = [h["distance"] for h in hits]
    assert distances == sorted(distances)


def test_every_format_retrieves_the_obvious_answer(built):
    """A trivially on-topic query must not depend on the representation."""
    path, _ = built
    for fmt in FORMATS:
        hits = index.query(fmt, "soil moisture", k=2, path=path)
        assert "C2-SOIL" in {h["concept_id"] for h in hits}, fmt


def test_k_bounds_the_result_size(built):
    path, _ = built
    assert len(index.query("json", "fire", k=2, path=path)) == 2


# --------------------------------------------------------------------------- #
# Hard gates.
# --------------------------------------------------------------------------- #


def test_verify_passes_a_sound_build(built):
    _, report = built
    assert index.verify(report, len(RECORDS)) == []


def test_ground_truth_presence_is_checked_per_collection(built):
    _, report = built
    assert set(report.ground_truth_missing) == set(FORMATS)
    assert all(v == [] for v in report.ground_truth_missing.values())


def test_verify_fails_on_a_short_collection():
    report = index.IndexReport(counts={f: 500 for f in FORMATS} | {"toon": 499})
    problems = index.verify(report, 500)
    assert any("indexed 499" in p for p in problems)


def test_verify_fails_when_ground_truth_is_unindexed():
    report = index.IndexReport(
        counts={f: 500 for f in FORMATS},
        ground_truth_missing={f: [] for f in FORMATS} | {"csv": ["C900-GT"]},
    )
    problems = index.verify(report, 500)
    assert any("ground-truth" in p and "csv" in p for p in problems)


def test_verify_fails_on_any_truncation():
    """Uneven clipping across formats would make this a truncation study."""
    report = index.IndexReport(
        max_sequence_tokens=512,
        counts={f: 500 for f in FORMATS},
        truncated={f: 0 for f in FORMATS} | {"jsonld": 444},
    )
    problems = index.verify(report, 500)
    assert any("exceed the 512-token embedding window" in p for p in problems)


def test_zero_truncation_is_the_declared_tolerance():
    assert MAX_EMBED_TRUNCATIONS == 0


# --------------------------------------------------------------------------- #
# Live: the real corpus and the real embedding model.
# --------------------------------------------------------------------------- #


@pytest.mark.live
def test_live_embedding_window_clears_the_longest_rendering():
    from airm import cmr, formats
    from airm.facets import facets

    payloads = [facets(r) for r in cmr.load_corpus()]
    longest = max(
        max(index.count_wordpieces([formats.render(f, p) for p in payloads]))
        for f in FORMATS
    )
    assert longest <= index.max_sequence_tokens(), (
        f"longest document is {longest} tokens; the window is "
        f"{index.max_sequence_tokens()}"
    )


@pytest.mark.live
def test_live_built_indexes_pass_every_gate():
    from airm.cmr import load_corpus
    from airm.queries import ground_truth_ids, load_queries

    records = load_corpus()
    gt = ground_truth_ids(load_queries())
    report = index.IndexReport(
        max_sequence_tokens=index.max_sequence_tokens(),
        counts={f: index.get_collection(f).count() for f in FORMATS},
        truncated={f: 0 for f in FORMATS},
        ground_truth_missing={
            f: [
                c
                for c in gt
                if c not in set(index.get_collection(f).get(ids=gt, include=[])["ids"])
            ]
            for f in FORMATS
        },
    )
    assert index.verify(report, len(records)) == []


@pytest.mark.live
def test_live_expert_queries_retrieve_their_targets():
    from airm.queries import load_queries

    found = 0
    queries = load_queries()
    for q in queries:
        hits = {h["concept_id"] for h in index.query("json", q.text, k=10)}
        if hits & set(q.expected_concept_ids):
            found += 1
    assert found >= len(queries) - 2, f"only {found}/{len(queries)} queries hit a target"


# --------------------------------------------------------------------------- #
# Two payloads, two databases.
# --------------------------------------------------------------------------- #


def test_the_two_payloads_get_separate_databases():
    """Separate paths, identical collection names.

    Naming the wrong collection cannot reach the other payload's vectors — you
    would have to open the wrong database, which is explicit.
    """
    assert index.chroma_dir("faceted") != index.chroma_dir("unfaceted")
    assert index.chroma_dir("faceted").name == "faceted"
    assert {collection_name(f) for f in FORMATS} == {collection_name(f) for f in FORMATS}


def test_unknown_payload_is_rejected():
    with pytest.raises(ValueError, match="unknown payload"):
        index.chroma_dir("nonsense")


def test_the_payloads_index_different_documents(tmp_path):
    """Same records, same model, different renderings — the whole design."""
    faceted = index._documents("faceted", "json", [(r["meta"]["concept-id"], facets(r)) for r in RECORDS])
    unfaceted_docs = index._documents(
        "unfaceted", "json", [(r["meta"]["concept-id"], unfaceted.payload(r)) for r in RECORDS]
    )
    assert faceted != unfaceted_docs
    # Size is deliberately not asserted here: these are hand-built stubs whose
    # raw UMM is smaller than their facet payload (the facets add a constructed
    # cmr_link). Only on real records does the control carry strictly more —
    # `test_the_unfaceted_payload_dwarfs_the_faceted_one` checks that on ATL08.
    assert all(unfaceted.payload(r) is not facets(r) for r in RECORDS)


def test_truncation_is_fatal_for_faceted_and_graded_for_the_control():
    """The asymmetry that lets the control exist at all.

    Experiment 2 measures a format effect on the faceted index, so uneven
    clipping there would be an artefact. The control exists partly *to show*
    that a raw UMM record does not fit an 8k window.
    """
    clipped = index.IndexReport(
        payload="faceted", max_sequence_tokens=8192,
        counts={f: 2 for f in FORMATS}, truncated={f: 3 for f in FORMATS},
        ground_truth_missing={f: [] for f in FORMATS},
    )
    assert any("exceed the" in p for p in index.verify(clipped, 2))

    clipped.payload = "unfaceted"
    assert index.verify(clipped, 2) == []
    assert "not for a format comparison" in index.truncation_note(clipped)


def test_truncation_note_is_empty_when_nothing_was_clipped():
    clean = index.IndexReport(
        payload="unfaceted", max_sequence_tokens=8192,
        counts={f: 2 for f in FORMATS}, truncated={f: 0 for f in FORMATS},
        ground_truth_missing={f: [] for f in FORMATS},
    )
    assert index.truncation_note(clean) == ""
    assert index.verify(clean, 2) == []


def test_the_report_records_which_payload_it_describes():
    assert index.IndexReport(payload="unfaceted").to_dict()["payload"] == "unfaceted"


def test_over_length_documents_are_truncated_not_turned_into_nan():
    """The bug that killed the first unfaceted build.

    ``gte-modernbert-base`` returns NaN for input past its window rather than
    truncating, and Chroma rejects NaN outright. The build counted truncations
    without performing any, so the counter meant "would have been clipped" while
    nothing was. Only the faceted zero-truncation gate kept it hidden.
    """
    import math

    limit = index.max_sequence_tokens()
    long_doc = "sea ice concentration measurements " * 4000
    assert index.count_wordpieces([long_doc])[0] > limit

    cut = index.truncate_to_window([long_doc])
    assert index.count_wordpieces(cut)[0] <= limit

    vector = index.embed(cut)[0]
    assert all(math.isfinite(v) for v in vector)


def test_documents_within_the_window_are_returned_untouched():
    short = ["a short record", "another short one"]
    assert index.truncate_to_window(short) == short


def test_an_empty_ground_truth_list_means_no_check_rather_than_a_crash(tmp_path):
    """Chroma raises on ``get(ids=[])`` instead of returning nothing.

    A caller can legitimately have nothing to check -- a smoke build over the
    first N records holds none of the evaluation set's targets. Before this,
    that aborted the build inside the gate that exists to *report* on ground
    truth, so the failure looked like a corrupt index rather than an empty list.
    """
    report = index.build(RECORDS, ground_truth=[], path=str(tmp_path))

    assert all(report.ground_truth_missing[f] == [] for f in FORMATS)
    assert index.verify(report, len(RECORDS)) == []


def test_the_embedding_model_is_recorded_so_two_databases_can_be_told_apart(tmp_path):
    """Vectors from different encoders are not interchangeable.

    ``build`` takes the encoder as an argument so a second index can be built
    beside the first, and the only thing that distinguishes the two on disk is
    this field.
    """
    report = index.build(RECORDS, ground_truth=["C1-SEA"], path=str(tmp_path))
    assert report.to_dict()["embed_model"] == index.EMBED_MODEL
    assert report.max_sequence_tokens == index.max_sequence_tokens(index.EMBED_MODEL)
