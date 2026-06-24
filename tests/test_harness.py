"""Harness unit + smoke tests.

Network-free: a small synthetic CMR record stands in for the live API so tests
are fast and deterministic. The embedder downloads a small model on first run.
"""

from __future__ import annotations

import pytest

from airm import metrics, representations, schema
from airm.embeddings import DualIndex, Index
from airm.queries import Query
from airm.representations import RENDERERS


# A minimal but realistic UMM item.
RECORD = {
    "meta": {"concept-id": "C123-TEST", "provider-id": "TEST_PROV"},
    "umm": {
        "EntryTitle": "Test Canopy Height V1",
        "ShortName": "TCH",
        "Abstract": "Estimates of vegetation canopy height from lidar.",
        "Version": "1",
        "ProcessingLevel": {"Id": "3"},
        "CollectionProgress": "ACTIVE",
        "ScienceKeywords": [
            {"Category": "EARTH SCIENCE", "Topic": "BIOSPHERE", "Term": "VEGETATION",
             "VariableLevel1": "CANOPY CHARACTERISTICS", "VariableLevel2": "VEGETATION HEIGHT"},
        ],
        "Platforms": [
            {"ShortName": "ICESat-2", "Instruments": [{"ShortName": "ATLAS"}]},
        ],
        "TemporalExtents": [
            {"EndsAtPresentFlag": True,
             "RangeDateTimes": [{"BeginningDateTime": "2018-10-14T00:00:00.000Z"}]},
        ],
        "SpatialExtent": {"HorizontalSpatialDomain": {"Geometry": {"BoundingRectangles": [
            {"WestBoundingCoordinate": -180.0, "SouthBoundingCoordinate": -88.0,
             "EastBoundingCoordinate": 180.0, "NorthBoundingCoordinate": 88.0}]}}},
        "DataCenters": [{"Roles": ["ARCHIVER"], "ShortName": "NASA NSIDC DAAC"}],
    },
}

DISTRACTOR = {
    "meta": {"concept-id": "C999-TEST", "provider-id": "TEST_PROV"},
    "umm": {
        "EntryTitle": "Sea Surface Temperature Daily V2",
        "Abstract": "Daily global sea surface temperature from infrared radiometry.",
        "ScienceKeywords": [
            {"Category": "EARTH SCIENCE", "Topic": "OCEANS", "Term": "OCEAN TEMPERATURE",
             "VariableLevel1": "SEA SURFACE TEMPERATURE"}],
        "Platforms": [{"ShortName": "Aqua", "Instruments": [{"ShortName": "MODIS"}]}],
    },
}


# --------------------------------------------------------------------------- #
# Representations.
# --------------------------------------------------------------------------- #


def test_all_renderers_nonempty_and_distinct():
    outputs = {name: fn(RECORD) for name, fn in RENDERERS.items()}
    assert all(out.strip() for out in outputs.values())
    assert len(set(outputs.values())) == len(outputs)  # all distinct


def test_facets_extract_expected_content():
    f = representations.facets(RECORD)
    assert f["title"] == "Test Canopy Height V1"
    assert "Vegetation Height" in f["variables"]
    assert f["platforms"][0]["instruments"] == ["ATLAS"]
    assert f["bbox"] == {"west": -180.0, "south": -88.0, "east": 180.0, "north": 88.0}
    assert f["temporal"]["begin"].startswith("2018-10-14")


def test_metadata_as_text_mentions_key_facets():
    txt = representations.metadata_as_text(RECORD)
    for needle in ["Canopy Height", "ATLAS", "Vegetation Height", "2018-10-14"]:
        assert needle in txt


# --------------------------------------------------------------------------- #
# Schema conditions.
# --------------------------------------------------------------------------- #


def test_schema_conditions_increase_in_detail():
    bare = schema.render_fields("bare")
    described = schema.render_fields("described")
    rich = schema.render_fields("rich")
    assert "description" not in bare
    assert "description" in described
    assert "examples" in rich
    assert len(rich) > len(described) > len(bare)


# --------------------------------------------------------------------------- #
# Metrics.
# --------------------------------------------------------------------------- #


def test_retrieval_metrics_basic():
    ranked = ["a", "b", "c", "d"]
    rel = {"c"}
    assert metrics.recall_at_k(ranked, rel, 4) == 1.0
    assert metrics.recall_at_k(ranked, rel, 2) == 0.0
    assert metrics.mrr(ranked, rel) == pytest.approx(1 / 3)
    assert 0 < metrics.ndcg_at_k(ranked, rel, 4) < 1


def test_value_format_valid():
    assert metrics.value_format_valid("temporal", "2024-12-01T00:00:00Z,2024-12-31T23:59:59Z")
    assert not metrics.value_format_valid("temporal", "Dec 2024")
    assert metrics.value_format_valid("bounding_box", "-75,-15,-45,5")
    assert not metrics.value_format_valid("bounding_box", "-200,0,0,0")


def test_field_selection_f1():
    assert metrics.field_selection_f1({"a", "b"}, {"a", "b"}) == 1.0
    assert metrics.field_selection_f1({"a"}, {"b"}) == 0.0


def test_pareto_frontier():
    pts = [("hi_acc_hi_cost", 0.9, 100), ("mid", 0.7, 50), ("dominated", 0.6, 100)]
    front = metrics.pareto_frontier(pts)
    assert "dominated" not in front
    assert "mid" in front and "hi_acc_hi_cost" in front


def test_bootstrap_ci_bounds_mean():
    mean, lo, hi = metrics.bootstrap_ci([1.0, 0.0, 1.0, 1.0, 0.0], seed=0)
    assert lo <= mean <= hi


def test_faithfulness_tally():
    t = metrics.FaithfulnessTally()
    t.add("correct"); t.add("fabrication"); t.add("correct")
    assert t.total == 3
    assert t.rate("correct") == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- #
# Embeddings — plant a relevant record and confirm it's retrieved.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("render_name", ["metadata_as_text", "flattened_jsonld"])
def test_index_retrieves_planted_record(render_name):
    corpus = [RECORD, DISTRACTOR]
    index = Index.build(corpus, RENDERERS[render_name])
    top = index.search("vegetation canopy height from lidar", k=1)
    assert top[0][0] == "C123-TEST"


def test_dual_index_retrieves_and_reembeds():
    corpus = [RECORD, DISTRACTOR]
    di = DualIndex.build(
        corpus,
        content_fn=representations.extract_summary,
        metadata_fn=lambda r: " ".join(representations.extract_variables(r)),
    )
    top = di.search("canopy height", k=1)
    assert top[0][0] == "C123-TEST"
    # re-embedding the metadata side should not raise and keeps dimensions
    di.reembed_metadata(corpus, lambda r: representations.extract_title(r))
    assert di.metadata_matrix.shape[0] == 2


# --------------------------------------------------------------------------- #
# LLM client degrades gracefully without credentials.
# --------------------------------------------------------------------------- #


def test_llm_provider_routing():
    from airm import llm
    assert llm.provider_for("claude-opus-4-8") == "anthropic"
    assert llm.provider_for("gpt-4.1") == "openai"
    assert llm.provider_for("llama3.1") == "ollama"


def test_llm_missing_key_raises_provider_unavailable(monkeypatch):
    from airm import llm
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(llm.ProviderUnavailable):
        llm.complete("claude-opus-4-8", "hello")


# --------------------------------------------------------------------------- #
# Smoke run — full retrieval pipeline end-to-end.
# --------------------------------------------------------------------------- #


def test_smoke_retrieval_pipeline(tmp_path):
    from airm import run
    corpus = [RECORD, DISTRACTOR]
    queries = [Query(id="q0", question="vegetation canopy height", relevant=["C123-TEST"])]

    all_results = []
    for name, fn in RENDERERS.items():
        all_results += run.evaluate_retrieval(corpus, queries, fn, name, k=2)

    summary = run.summarize(all_results)
    assert set(summary) == set(RENDERERS)
    # the planted record should be retrievable under at least one representation
    assert any(s["recall_at_k_mean"] > 0 for s in summary.values())

    out = tmp_path / "results.jsonl"
    run.write_jsonl(all_results, out)
    assert out.exists() and out.read_text().count("\n") == len(all_results)
