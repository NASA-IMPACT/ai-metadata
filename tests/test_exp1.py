"""Experiment 1 runner tests — network-free via a stubbed llm.complete."""

from __future__ import annotations

import pytest

from airm.queries import Query
from experiments.exp1_format_ablation import run as exp1

RECORD = {
    "meta": {"concept-id": "C123-TEST"},
    "umm": {
        "EntryTitle": "Test Canopy Height V1",
        "Abstract": "Vegetation canopy height from lidar.",
        "ProcessingLevel": {"Id": "3"},
        "CollectionProgress": "ACTIVE",
        "Version": "1",
        "ScienceKeywords": [
            {"Category": "EARTH SCIENCE", "Topic": "BIOSPHERE", "Term": "VEGETATION",
             "VariableLevel2": "VEGETATION HEIGHT"}],
        "Platforms": [{"ShortName": "ICESat-2", "Instruments": [{"ShortName": "ATLAS"}]}],
        "TemporalExtents": [{"EndsAtPresentFlag": True,
                             "RangeDateTimes": [{"BeginningDateTime": "2018-10-14T00:00:00.000Z"}]}],
        "SpatialExtent": {"HorizontalSpatialDomain": {"Geometry": {"BoundingRectangles": [
            {"WestBoundingCoordinate": -180.0, "SouthBoundingCoordinate": -88.0,
             "EastBoundingCoordinate": 180.0, "NorthBoundingCoordinate": 88.0}]}}},
        "DataCenters": [{"Roles": ["ARCHIVER"], "ShortName": "NASA NSIDC DAAC"}],
    },
}
DISTRACTOR = {
    "meta": {"concept-id": "C999-TEST"},
    "umm": {"EntryTitle": "Sea Surface Temperature", "Abstract": "Daily SST.",
            "ScienceKeywords": [{"Category": "EARTH SCIENCE", "Topic": "OCEANS",
                                 "Term": "OCEAN TEMPERATURE"}]},
}


def test_extract_concept_id():
    assert exp1.extract_concept_id("The answer is C123-NSIDC_CPRD.") == "C123-NSIDC_CPRD"
    assert exp1.extract_concept_id("none here") is None


def test_flatten_facets_strings_only():
    flat = exp1.flatten_facets(RECORD)
    assert flat["instruments"] == "ATLAS"
    assert "Vegetation Height" in flat["variables"]
    assert all(isinstance(v, str) for v in flat.values())


def test_templated_renderer_default_formats():
    out = exp1.templated_renderer(exp1.DEFAULT_TEMPLATE)(RECORD)
    assert "ATLAS" in out and "Test Canopy Height" in out


def test_build_answer_prompt_tags_concept_ids():
    prompt = exp1.build_answer_prompt("which dataset?", [("C123-TEST", "body-a"), ("C999-TEST", "body-b")])
    assert "concept_id=C123-TEST" in prompt and "concept_id=C999-TEST" in prompt


def test_answer_stage_end_to_end_stubbed(monkeypatch):
    """Full answer stage with a fake model that echoes the first candidate."""
    corpus = [RECORD, DISTRACTOR]
    corpus_by_id = {r["meta"]["concept-id"]: r for r in corpus}
    sampled = [Query(id="q0", question="vegetation canopy height", relevant=["C123-TEST"])]
    # retrieval places the correct record first
    retrieved = {(rep, "q0"): ["C123-TEST", "C999-TEST"] for rep in exp1.RENDERERS}

    class FakeCompletion:
        def __init__(self, text):
            self.text = text

    def fake_complete(model, prompt, **kw):
        # pick the first concept_id present in the prompt (the top candidate)
        return FakeCompletion(exp1.extract_concept_id(prompt))

    monkeypatch.setattr(exp1, "complete", fake_complete)

    rows = exp1.run_answer_stage(corpus_by_id, retrieved, sampled, ["fake-model"], answer_k=2)
    assert len(rows) == len(exp1.RENDERERS)  # one row per representation
    assert all(r["correct"] for r in rows)   # top candidate is the relevant one
    assert all(r["prompt_tokens"] > 0 for r in rows)


def test_answer_stage_skips_unavailable_provider(monkeypatch):
    corpus_by_id = {RECORD["meta"]["concept-id"]: RECORD}
    sampled = [Query(id="q0", question="canopy", relevant=["C123-TEST"])]
    retrieved = {(rep, "q0"): ["C123-TEST"] for rep in exp1.RENDERERS}

    from airm.llm import ProviderUnavailable

    def boom(model, prompt, **kw):
        raise ProviderUnavailable("no key")

    monkeypatch.setattr(exp1, "complete", boom)
    rows = exp1.run_answer_stage(corpus_by_id, retrieved, sampled, ["x"], answer_k=1)
    assert rows == []  # gracefully skipped, no crash
