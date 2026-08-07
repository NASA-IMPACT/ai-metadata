"""Tests for synthetic query generation.

Three guards decide whether a generated query is usable, and each of them exists
because the alternative silently corrupts Experiment 2:

* parse failures would drop queries below quota without saying so;
* leaked identifiers turn retrieval into string matching, inflating every format
  equally and hiding the effect being measured;
* unretrievable queries add noise to every cell.
"""

from __future__ import annotations

import json

import pytest

from airm import queries as q
from airm import synth
from airm.config import ModelSpec

SPEC = ModelSpec("ollama", "test-model")


def _record(cid: str, topic: str, title: str) -> dict:
    return {
        "meta": {"concept-id": cid},
        "umm": {
            "EntryTitle": title,
            "ShortName": cid.split("-")[0],
            "Abstract": f"Measurements of {title.lower()} from orbit.",
            "ScienceKeywords": [{"Category": "EARTH SCIENCE", "Topic": topic}],
        },
    }


RECORDS = [
    _record("SEA1-A", "OCEANS", "Sea Ice Concentration Daily"),
    _record("SEA2-A", "OCEANS", "Sea Surface Temperature Daily"),
    _record("LAND1-A", "LAND SURFACE", "Surface Soil Moisture Level 3"),
]


# --------------------------------------------------------------------------- #
# Response parsing.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        '{"query": "How did sea ice change?"}',
        '```json\n{"query": "How did sea ice change?"}\n```',
        'Here you go: {"query": "How did sea ice change?"} — enjoy!',
    ],
)
def test_generated_json_is_parsed_through_common_wrappers(text):
    assert q.parse_generated(text) == "How did sea ice change?"


@pytest.mark.parametrize("text", ["not json at all", "{}", '{"query": ""}', '{"query": 42}'])
def test_unusable_responses_parse_to_none(text):
    assert q.parse_generated(text) is None


# --------------------------------------------------------------------------- #
# Identifier leakage.
# --------------------------------------------------------------------------- #


def test_a_query_naming_the_short_name_is_a_leak():
    payload = {"title": "Sea Ice Concentration Daily", "short_name": "ATL08", "concept_id": "C1-A"}
    assert q._leaked_identifiers("Where can I find ATL08 data?", payload)


def test_a_query_naming_the_concept_id_or_doi_is_a_leak():
    payload = {"title": "T", "concept_id": "C1-A", "doi": "10.5067/XYZ"}
    assert q._leaked_identifiers("Use C1-A for this", payload)
    assert q._leaked_identifiers("See 10.5067/XYZ", payload)


def test_a_long_verbatim_title_slice_is_the_same_leak_in_disguise():
    payload = {"title": "ATLAS ICESat Land and Vegetation Height Product"}
    leaks = q._leaked_identifiers(
        "I need ATLAS ICESat Land and Vegetation Height data", payload
    )
    assert any("title-phrase" in leak for leak in leaks)


def test_a_title_quoted_with_its_stopwords_is_still_caught():
    """Comparing a stopword-filtered title against raw text never matched."""
    payload = {"title": "Sea Ice Concentration from Passive Microwave Data"}
    assert q._leaked_identifiers(
        "Show me Sea Ice Concentration from Passive Microwave Data please", payload
    )


def test_punctuation_and_case_do_not_defeat_the_title_check():
    payload = {"title": "MERRA-2 Surface Flux Diagnostics Hourly"}
    assert q._leaked_identifiers("merra 2 surface flux diagnostics hourly?", payload)


def test_a_short_title_cannot_trigger_the_ngram_check():
    payload = {"title": "Ozone Daily"}
    assert q._leaked_identifiers("How much ozone daily?", payload) == []


def test_a_properly_science_framed_question_is_not_a_leak():
    payload = {
        "title": "ATLAS ICESat Land and Vegetation Height Product",
        "short_name": "ATL08",
        "concept_id": "C1-A",
    }
    assert (
        q._leaked_identifiers(
            "How has forest canopy height in the Amazon changed since 2019?", payload
        )
        == []
    )


# --------------------------------------------------------------------------- #
# Generation.
# --------------------------------------------------------------------------- #


@pytest.fixture
def stub_llm(monkeypatch):
    """Return a canned query and make everything retrievable."""
    from airm import llm

    def fake_ask(spec, prompt, **kwargs):
        return llm.LLMResponse(
            text=json.dumps({"query": "How did the measured phenomenon change over time?"}),
            provider=spec.provider,
            model=spec.model,
        )

    monkeypatch.setattr(synth, "ask", fake_ask)
    monkeypatch.setattr(synth, "_retrievable", lambda text, cid, k=10: ["json"])


def test_generation_meets_the_quota_per_topic(stub_llm):
    quota = {"OCEANS": 2, "LAND SURFACE": 1}
    qs, report = synth.generate(SPEC, records=RECORDS, quota=quota)
    assert len(qs) == 3
    assert report.produced == {"OCEANS": 2, "LAND SURFACE": 1}
    assert synth.verify(report) == []


def test_each_query_is_tied_to_the_record_it_was_generated_from(stub_llm):
    qs, _ = synth.generate(SPEC, records=RECORDS, quota={"OCEANS": 2})
    assert all(len(query.expected_concept_ids) == 1 for query in qs)
    assert {query.expected_concept_ids[0] for query in qs} <= {"SEA1-A", "SEA2-A"}
    assert all(query.source == "synthetic" for query in qs)
    assert all(query.topic == "OCEANS" for query in qs)


def test_ground_truth_is_in_corpus_by_construction(stub_llm):
    """Unlike the expert slice, a synthetic query cannot target a retired record."""
    corpus_ids = {r["meta"]["concept-id"] for r in RECORDS}
    qs, _ = synth.generate(SPEC, records=RECORDS, quota={"OCEANS": 2, "LAND SURFACE": 1})
    assert all(set(query.expected_concept_ids) <= corpus_ids for query in qs)


def test_query_ids_are_unique(stub_llm):
    qs, _ = synth.generate(SPEC, records=RECORDS, quota={"OCEANS": 2, "LAND SURFACE": 1})
    assert len({query.query_id for query in qs}) == len(qs)


def test_generation_is_deterministic_under_a_fixed_seed(stub_llm):
    a, _ = synth.generate(SPEC, records=RECORDS, quota={"OCEANS": 1})
    b, _ = synth.generate(SPEC, records=RECORDS, quota={"OCEANS": 1})
    assert [x.expected_concept_ids for x in a] == [x.expected_concept_ids for x in b]


# --------------------------------------------------------------------------- #
# The three rejection paths.
# --------------------------------------------------------------------------- #


def test_a_leaking_query_is_retried_then_dropped_with_a_reason(monkeypatch):
    from airm import llm

    monkeypatch.setattr(
        synth,
        "ask",
        lambda spec, prompt, **kw: llm.LLMResponse(
            text=json.dumps({"query": "Give me SEA1 data"}), provider="ollama", model="m"
        ),
    )
    monkeypatch.setattr(synth, "_retrievable", lambda text, cid, k=10: ["json"])

    qs, report = synth.generate(SPEC, records=RECORDS[:1], quota={"OCEANS": 1})
    assert qs == []
    assert report.dropped[0]["reason"] == "leaked identifiers"
    assert report.retries == 2  # attempts 2 and 3 after the first


def test_an_unretrievable_query_is_dropped_with_a_reason(monkeypatch):
    from airm import llm

    monkeypatch.setattr(
        synth,
        "ask",
        lambda spec, prompt, **kw: llm.LLMResponse(
            text=json.dumps({"query": "An entirely unrelated question about tax policy"}),
            provider="ollama",
            model="m",
        ),
    )
    monkeypatch.setattr(synth, "_retrievable", lambda text, cid, k=10: [])

    qs, report = synth.generate(SPEC, records=RECORDS[:1], quota={"OCEANS": 1})
    assert qs == []
    assert report.dropped[0]["reason"] == "not retrievable by any format"


def test_an_llm_failure_is_recorded_rather_than_swallowed(monkeypatch):
    from airm.llm import LLMError

    def boom(*args, **kwargs):
        raise LLMError("provider down")

    monkeypatch.setattr(synth, "ask", boom)
    qs, report = synth.generate(SPEC, records=RECORDS[:1], quota={"OCEANS": 1})
    assert qs == []
    assert "llm error" in report.dropped[0]["reason"]


def test_a_quota_shortfall_fails_verification(stub_llm):
    """A quietly rebalanced quota is a quietly biased query set."""
    qs, report = synth.generate(SPEC, records=RECORDS, quota={"OCEANS": 5})
    assert len(qs) == 2  # only two OCEANS records exist
    problems = synth.verify(report)
    assert any("produced 2 of 5" in p for p in problems)
    assert report.dropped[0]["reason"] == "too few corpus records to draw from"


def test_generation_calls_are_logged(monkeypatch, tmp_path):
    from airm import llm

    logger = llm.CallLogger(run_id="testrun", root=tmp_path)
    monkeypatch.setattr(
        llm,
        "_call_ollama",
        lambda spec, msgs, **kw: llm.LLMResponse(
            text=json.dumps({"query": "How did the phenomenon change?"}),
            provider="ollama",
            model="m",
            prompt_tokens=20,
            completion_tokens=8,
        ),
    )
    monkeypatch.setattr(synth, "_retrievable", lambda text, cid, k=10: ["json"])

    synth.generate(SPEC, records=RECORDS[:1], quota={"OCEANS": 1}, logger=logger)
    records = logger.read(llm.PURPOSE_QUERY_GEN)
    assert len(records) == 1
    assert records[0]["concept_id"] == "SEA1-A"
    assert records[0]["topic"] == "OCEANS"
