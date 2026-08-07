"""Tests for the per-format rendering cache.

The cache's whole value rests on two guarantees: the mapping to ``cmr_cache`` is
one-to-one, and the bytes on disk were produced by the code currently importable.
Both are tested here, along with the fallback that keeps a missing or stale cache
from ever changing a result -- only how long it takes to get one.
"""

from __future__ import annotations

import json

import pytest

from airm import format_cache, formats
from airm.config import FORMATS
from airm.facets import facets


def _record(cid: str, title: str = "Test Collection") -> dict:
    return {
        "meta": {"concept-id": cid},
        "umm": {
            "EntryTitle": title,
            "ShortName": "TEST",
            "Platforms": [{"ShortName": "Terra", "Instruments": [{"ShortName": "MODIS"}]}],
            "ScienceKeywords": [
                {"Category": "EARTH SCIENCE", "Topic": "OCEANS", "Term": "SEA ICE"}
            ],
        },
    }


@pytest.fixture
def caches(tmp_path, monkeypatch):
    """A raw cache with three records, and an empty format-cache root."""
    raw = tmp_path / "cmr_cache"
    raw.mkdir()
    for i in (1, 2, 3):
        cid = f"C{i}-TEST"
        (raw / f"{cid}.json").write_text(json.dumps(_record(cid, f"Collection {i}")))
    monkeypatch.setattr(format_cache, "CMR_CACHE_DIR", raw)
    return raw, tmp_path / "format_cache"


# --------------------------------------------------------------------------- #
# Build.
# --------------------------------------------------------------------------- #


def test_build_writes_one_file_per_record_per_format(caches):
    raw, root = caches
    report = format_cache.build(root=root, cache_dir=raw)

    assert report.records == 3
    assert report.written == {fmt: 3 for fmt in FORMATS}
    for fmt in FORMATS:
        files = sorted(p.name for p in (root / fmt).glob(f"*{format_cache.EXTENSIONS[fmt]}"))
        assert len(files) == 3, f"{fmt} holds {files}"


def test_cached_bytes_are_what_the_renderer_produces(caches):
    """The cache must be a cache, not a second implementation."""
    raw, root = caches
    format_cache.build(root=root, cache_dir=raw)

    payload = facets(json.loads((raw / "C2-TEST.json").read_text()))
    for fmt in FORMATS:
        assert format_cache.load(fmt, "C2-TEST", root=root) == formats.render(fmt, payload)


def test_rebuild_skips_records_already_current(caches):
    raw, root = caches
    format_cache.build(root=root, cache_dir=raw)
    second = format_cache.build(root=root, cache_dir=raw)

    assert second.skipped_unchanged == 3
    assert second.written == {fmt: 0 for fmt in FORMATS}


def test_force_rerenders_everything(caches):
    raw, root = caches
    format_cache.build(root=root, cache_dir=raw)
    forced = format_cache.build(root=root, cache_dir=raw, force=True)

    assert forced.skipped_unchanged == 0
    assert forced.written == {fmt: 3 for fmt in FORMATS}


def test_a_missing_raw_record_is_reported_not_raised(caches):
    raw, root = caches
    report = format_cache.build(["C9-ABSENT"], root=root, cache_dir=raw)

    assert report.render_errors == [{"concept_id": "C9-ABSENT", "error": "raw record missing"}]


# --------------------------------------------------------------------------- #
# Staleness.
# --------------------------------------------------------------------------- #


def test_fingerprint_tracks_the_rendering_code(monkeypatch, tmp_path):
    """Editing formats.py or facets.py must invalidate the cache."""
    before = format_cache.fingerprint()

    edited = tmp_path / "src"
    edited.mkdir()
    (edited / "facets.py").write_bytes(b"# facets")
    (edited / "formats.py").write_bytes(b"# formats")
    monkeypatch.setattr(format_cache, "__file__", str(edited / "format_cache.py"))

    assert format_cache.fingerprint() != before


def test_load_refuses_to_serve_bytes_from_older_code(caches, monkeypatch):
    raw, root = caches
    format_cache.build(root=root, cache_dir=raw)

    monkeypatch.setattr(format_cache, "fingerprint", lambda: "0000000000000000")
    with pytest.raises(format_cache.StaleCacheError):
        format_cache.load("json", "C1-TEST", root=root)


def test_a_stale_cache_is_rebuilt_wholesale_not_incrementally(caches, monkeypatch):
    """A code change invalidates every file, so nothing may be skipped."""
    raw, root = caches
    format_cache.build(root=root, cache_dir=raw)

    monkeypatch.setattr(format_cache, "fingerprint", lambda: "0000000000000000")
    report = format_cache.build(root=root, cache_dir=raw)

    assert report.skipped_unchanged == 0
    assert report.written == {fmt: 3 for fmt in FORMATS}


# --------------------------------------------------------------------------- #
# Fallback.
# --------------------------------------------------------------------------- #


def test_render_many_falls_back_when_the_cache_is_absent(caches):
    """No cache must cost time, never correctness."""
    raw, root = caches  # never built
    payload = facets(json.loads((raw / "C1-TEST.json").read_text()))

    got = format_cache.render_many("toon", [("C1-TEST", payload)], root=root)
    assert got == [formats.render("toon", payload)]


def test_render_many_falls_back_when_the_cache_is_stale(caches, monkeypatch):
    raw, root = caches
    format_cache.build(root=root, cache_dir=raw)
    payload = facets(json.loads((raw / "C1-TEST.json").read_text()))

    monkeypatch.setattr(format_cache, "fingerprint", lambda: "0000000000000000")
    got = format_cache.render_many("csv", [("C1-TEST", payload)], root=root)
    assert got == [formats.render("csv", payload)]


def test_render_many_falls_back_for_a_record_the_cache_lacks(caches):
    """A partially-built cache must still render the records it is missing."""
    raw, root = caches
    format_cache.build(["C1-TEST"], root=root, cache_dir=raw)
    payload = facets(json.loads((raw / "C3-TEST.json").read_text()))

    got = format_cache.render_many("yaml", [("C3-TEST", payload)], root=root)
    assert got == [formats.render("yaml", payload)]


def test_render_all_cached_covers_every_format(caches):
    raw, root = caches
    format_cache.build(root=root, cache_dir=raw)
    payload = facets(json.loads((raw / "C1-TEST.json").read_text()))

    got = format_cache.render_all_cached("C1-TEST", payload, root=root)
    assert got == {fmt: formats.render(fmt, payload) for fmt in FORMATS}


# --------------------------------------------------------------------------- #
# Verification.
# --------------------------------------------------------------------------- #


def test_verify_passes_on_a_complete_cache(caches):
    raw, root = caches
    format_cache.build(root=root, cache_dir=raw)
    assert format_cache.verify(root=root, cache_dir=raw) == []


def test_verify_reports_a_never_built_cache(caches):
    raw, root = caches
    problems = format_cache.verify(root=root, cache_dir=raw)
    assert len(problems) == 1 and "never been built" in problems[0]


def test_verify_catches_a_record_with_no_rendering(caches):
    """The failure mode that would silently shrink an experiment."""
    raw, root = caches
    format_cache.build(root=root, cache_dir=raw)
    (root / "toon" / "C2-TEST.toon").unlink()

    problems = format_cache.verify(root=root, cache_dir=raw)
    assert any("toon" in p and "no rendering" in p for p in problems)


def test_verify_catches_a_rendering_with_no_record(caches):
    """A leftover rendering could let a dropped record back into a run."""
    raw, root = caches
    format_cache.build(root=root, cache_dir=raw)
    (root / "json" / "C99-GHOST.json").write_text("{}")

    problems = format_cache.verify(root=root, cache_dir=raw)
    assert any("json" in p and "no raw record" in p for p in problems)


def test_verify_catches_staleness(caches, monkeypatch):
    raw, root = caches
    format_cache.build(root=root, cache_dir=raw)

    monkeypatch.setattr(format_cache, "fingerprint", lambda: "0000000000000000")
    problems = format_cache.verify(root=root, cache_dir=raw)
    assert any("stale" in p for p in problems)


def test_every_format_has_a_distinct_extension():
    """Two formats sharing an extension would overwrite each other."""
    assert set(format_cache.EXTENSIONS) == set(FORMATS)
    assert len(set(format_cache.EXTENSIONS.values())) == len(FORMATS)
