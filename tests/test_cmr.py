"""Tests for the CMR client.

Network-touching tests are marked ``live`` so the suite stays fast and offline by
default; run them with ``uv run pytest -m live``.
"""

from __future__ import annotations

import json

import pytest

from airm import cmr


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    d = tmp_path / "cmr_cache"
    d.mkdir()
    monkeypatch.setattr(cmr, "CMR_CACHE_DIR", d)
    return d


def _record(cid: str, title: str = "Test Collection") -> dict:
    return {"meta": {"concept-id": cid}, "umm": {"EntryTitle": title}}


def test_cache_roundtrip(cache_dir):
    rec = _record("C1-TEST")
    cmr._write_cache("C1-TEST", rec)
    assert cmr.is_cached("C1-TEST")
    assert cmr.load_cached("C1-TEST") == rec


def test_corrupt_cache_reads_as_a_miss_not_a_crash(cache_dir):
    """A half-written cache file must not take down the next run."""
    (cache_dir / "C2-TEST.json").write_text('{"meta": {"concept-id"')
    assert cmr.load_cached("C2-TEST") is None


def test_absent_cache_entry_is_none(cache_dir):
    assert cmr.load_cached("C404-TEST") is None


def test_fetch_by_concept_ids_is_served_entirely_from_cache(cache_dir, monkeypatch):
    """With every id cached, no HTTP request is made at all."""
    for cid in ("C1-TEST", "C2-TEST"):
        cmr._write_cache(cid, _record(cid))

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("network was hit despite a full cache")

    monkeypatch.setattr(cmr, "_get", explode)
    got = cmr.fetch_by_concept_ids(["C1-TEST", "C2-TEST"])
    assert [cmr.concept_id(r) for r in got] == ["C1-TEST", "C2-TEST"]


def test_fetch_by_concept_ids_dedups_and_preserves_order(cache_dir, monkeypatch):
    for cid in ("C1-TEST", "C2-TEST"):
        cmr._write_cache(cid, _record(cid))
    monkeypatch.setattr(cmr, "_get", lambda *a, **k: {"items": []})
    got = cmr.fetch_by_concept_ids(["C2-TEST", "C1-TEST", "C2-TEST"])
    assert [cmr.concept_id(r) for r in got] == ["C2-TEST", "C1-TEST"]


def test_unresolvable_ids_are_reported_not_silently_dropped(cache_dir, monkeypatch):
    cmr._write_cache("C1-TEST", _record("C1-TEST"))
    monkeypatch.setattr(cmr, "_get", lambda *a, **k: {"items": []})
    assert cmr.missing_concept_ids(["C1-TEST", "C-GONE"]) == ["C-GONE"]


def test_id_batches_are_chunked(cache_dir, monkeypatch):
    """Long id lists must be split, or the request URL overflows."""
    calls: list[list] = []

    def fake_get(url, params):
        calls.append(params)
        return {"items": []}

    monkeypatch.setattr(cmr, "_get", fake_get)
    cmr.fetch_by_concept_ids([f"C{i}-TEST" for i in range(120)])
    assert len(calls) == 3  # 120 ids / chunk size 50
    for params in calls:
        ids = [v for k, v in params if k == "concept_id"]
        assert len(ids) <= cmr._ID_CHUNK


def test_corpus_write_then_load(tmp_path):
    path = tmp_path / "corpus.jsonl"
    records = [_record("C1-TEST"), _record("C2-TEST")]
    cmr.write_corpus(records, path)
    assert cmr.load_corpus(path) == records
    # Written as one JSON object per line.
    assert len(path.read_text().strip().splitlines()) == 2


def test_load_corpus_missing_file_names_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="airm.corpus"):
        cmr.load_corpus(tmp_path / "nope.jsonl")


@pytest.mark.live
def test_live_fetch_and_cache_hit(cache_dir, monkeypatch):
    """Real CMR call, then assert the second call never touches the network."""
    got = cmr.fetch_by_concept_ids(["C3550186689-ESDIS"])
    assert len(got) == 1
    assert cmr.concept_id(got[0]) == "C3550186689-ESDIS"
    assert "IPCC" in cmr.entry_title(got[0])
    assert cmr.is_cached("C3550186689-ESDIS")

    def explode(*args, **kwargs):  # pragma: no cover
        raise AssertionError("second call hit the network instead of the cache")

    monkeypatch.setattr(cmr, "_get", explode)
    again = cmr.fetch_by_concept_ids(["C3550186689-ESDIS"])
    assert cmr.concept_id(again[0]) == "C3550186689-ESDIS"


@pytest.mark.live
def test_live_topic_search_is_stratifiable():
    recs = cmr.search_by_topic("SUN-EARTH INTERACTIONS", count=5, use_cache=False)
    assert len(recs) == 5
    topics = {
        kw.get("Topic")
        for r in recs
        for kw in r.get("umm", {}).get("ScienceKeywords", [])
    }
    assert "SUN-EARTH INTERACTIONS" in topics
