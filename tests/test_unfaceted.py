"""The unfaceted control — payload, generic renderers, and its build.

The control only earns its place if it is held to the same parity standard as
the faceted cache. These tests pin that, and pin the two properties that make it
a *control* rather than a second experiment: it reads no facet code, and its
JSON rendering is byte-identical to the ``json_umm`` reference Experiment 1
computes independently.
"""

from __future__ import annotations

import json

import pytest

from airm import formats, unfaceted
from airm.facets import fact_set, flatten

RECORD = {
    "meta": {"concept-id": "C1-TEST", "revision-id": 7},
    "umm": {
        "EntryTitle": "A Test Collection",
        "ShortName": "TEST",
        "Version": "2",
        "Abstract": "An abstract, with a comma and a \"quote\".",
        "DOI": {"DOI": "10.5067/TEST"},
        "ProcessingLevel": {"Id": "3"},
        "Platforms": [{"ShortName": "Terra", "Instruments": [{"ShortName": "MODIS"}]}],
        "SpatialExtent": {
            "HorizontalSpatialDomain": {
                "Geometry": {
                    "CoordinateSystem": "CARTESIAN",
                    "BoundingRectangles": [
                        {
                            "WestBoundingCoordinate": -180,
                            "SouthBoundingCoordinate": -90,
                            "EastBoundingCoordinate": 180,
                            "NorthBoundingCoordinate": 90,
                        }
                    ],
                }
            }
        },
        # Present in the raw record and deliberately absent from the 32 facets:
        # the control must carry it, which is the point of the control.
        "ContactPersons": [{"LastName": "Doe", "Roles": ["TECHNICAL CONTACT"]}],
        "MetadataDates": [{"Type": "CREATE", "Date": "2020-01-01T00:00:00.000Z"}],
    },
}


# --------------------------------------------------------------------------- #
# Payload.
# --------------------------------------------------------------------------- #


def test_payload_is_the_raw_umm_untouched():
    assert unfaceted.payload(RECORD) is RECORD["umm"]


def test_payload_excludes_meta():
    """``meta`` is CMR bookkeeping, not metadata about the dataset.

    Including it would put the concept-id in the payload of some formats and not
    others depending on how each renderer treats it.
    """
    assert "concept-id" not in unfaceted.payload(RECORD)
    assert "revision-id" not in unfaceted.payload(RECORD)


def test_payload_accepts_a_bare_umm_object():
    bare = {"EntryTitle": "T"}
    assert unfaceted.payload(bare) == bare


def test_control_carries_fields_the_facets_drop():
    """The whole reason the control exists."""
    from airm.facets import facets

    paths = {p for p, _ in flatten(unfaceted.payload(RECORD))}
    facet_paths = {p for p, _ in flatten(facets(RECORD))}
    assert any(p.startswith("ContactPersons") for p in paths)
    assert not any(p.startswith("ContactPersons") for p in facet_paths)


# --------------------------------------------------------------------------- #
# Parity — the same standard the faceted cache is held to.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fmt", ["json", "csv", "yaml", "jsonld"])
def test_schema_agnostic_formats_round_trip(fmt):
    p = unfaceted.payload(RECORD)
    got = fact_set(unfaceted.PARSERS[fmt](unfaceted.render(fmt, p)))
    assert got == fact_set(p)


def test_prose_covers_every_value():
    p = unfaceted.payload(RECORD)
    report = unfaceted.parity_report(p)
    assert report["mat"]["ok"], report["mat"]["missing"][:5]


def test_parity_report_can_fail(monkeypatch):
    """The validator must be able to fail, not just always pass."""
    monkeypatch.setitem(unfaceted.RENDERERS, "mat", lambda p: "nothing here")
    report = unfaceted.parity_report(unfaceted.payload(RECORD))
    assert report["mat"]["ok"] is False


# --------------------------------------------------------------------------- #
# The two renderers that need a schema and do not have one.
# --------------------------------------------------------------------------- #


def test_jsonld_models_five_fields_and_dumps_the_rest_into_property_values():
    doc = json.loads(unfaceted.render_jsonld(unfaceted.payload(RECORD)))
    assert doc["name"] == "A Test Collection"
    assert doc["identifier"] == "10.5067/TEST"
    names = {p["name"] for p in doc["additionalProperty"]}
    # Nothing schema.org models is duplicated into additionalProperty...
    assert "EntryTitle" not in names and "DOI.DOI" not in names
    # ...and everything it does not model lands there, keyed by path.
    assert "ContactPersons[0].LastName" in names


def test_mat_is_prose_not_a_breadcrumb_dump():
    """The check that was missing when ``mat`` was silently a path dump.

    A renderer is not verified by its name. Prose carries several values per
    sentence, so its clause count must be far below its leaf count; a breadcrumb
    dump emits exactly one clause per leaf and spells a path in every one.
    """
    p = unfaceted.payload(RECORD)
    prose = unfaceted.render_mat(p)
    crumbs = unfaceted.render_breadcrumb(p)
    leaves = len(flatten(p))

    assert crumbs.count(" is ") == leaves          # breadcrumb: one clause per leaf
    assert prose.count(" is ") < leaves / 3        # prose: several values per sentence
    assert unfaceted.PATH_SEPARATOR not in prose.split("Additional metadata:")[0]
    assert prose.startswith("A Test Collection")   # a sentence, not a path


def test_prose_is_substantially_cheaper_than_the_breadcrumb_dump():
    """The mislabel cost Experiment 1 roughly a factor of two."""
    p = unfaceted.payload(RECORD)
    assert len(unfaceted.render_mat(p)) < len(unfaceted.render_breadcrumb(p))


def test_prose_carries_almost_everything_before_the_tail():
    """The tail is a backstop, not the renderer.

    Without this a renderer that routed every leaf through the tail would pass
    parity while being a breadcrumb dump again — exactly the failure this module
    already made once.
    """
    coverage = unfaceted.prose_coverage(unfaceted.payload(RECORD))
    assert coverage["prose_share"] > 0.9


def test_values_appear_verbatim_not_prettified():
    """``full_mat.txt`` prettifies (``ACTIVE`` -> ``active``) and so covers 78%.

    Parity is defined on the canonical string, so prettifying drops the value.
    """
    record = {"umm": {"EntryTitle": "T", "CollectionProgress": "ACTIVE"}}
    assert "ACTIVE" in unfaceted.render_mat(unfaceted.payload(record))


def test_data_centres_are_merged_by_name():
    """CMR lists one entry per role; repeating the name states no extra fact."""
    record = {"umm": {"EntryTitle": "T", "DataCenters": [
        {"ShortName": "NSIDC", "Roles": ["ARCHIVER"]},
        {"ShortName": "NSIDC", "Roles": ["DISTRIBUTOR"]},
    ]}}
    text = unfaceted.render_mat(unfaceted.payload(record))
    assert "NSIDC (ARCHIVER, DISTRIBUTOR)" in text
    assert text.count("NSIDC") == 1


def test_json_rendering_matches_exp1_raw_umm_reference():
    """The control's JSON must be the same string Experiment 1 measures.

    ``exp1`` computes its ``json_umm`` reference independently; if these two ever
    diverge, the reference and the control stop describing the same thing.
    """
    expected = json.dumps(RECORD["umm"], ensure_ascii=False, separators=(",", ":"))
    assert unfaceted.render("json", unfaceted.payload(RECORD)) == expected


# --------------------------------------------------------------------------- #
# Build.
# --------------------------------------------------------------------------- #


@pytest.fixture
def raw_cache(tmp_path):
    d = tmp_path / "cmr_cache"
    d.mkdir()
    (d / "C1-TEST.json").write_text(json.dumps(RECORD))
    return d


def test_build_writes_every_format_and_a_manifest(tmp_path, raw_cache):
    root = tmp_path / "unfaceted"
    report = unfaceted.build(root=root, cache_dir=raw_cache)
    assert report.records == 1
    for fmt in unfaceted.EXTENSIONS:
        assert unfaceted.path_for(fmt, "C1-TEST", root=root).exists()
    assert unfaceted.load_manifest(root)["payload"] == "raw UMM (unfaceted)"


def test_verify_is_clean_after_a_build(tmp_path, raw_cache):
    root = tmp_path / "unfaceted"
    unfaceted.build(root=root, cache_dir=raw_cache)
    assert unfaceted.verify(root=root, cache_dir=raw_cache) == []


def test_verify_flags_a_missing_rendering(tmp_path, raw_cache):
    root = tmp_path / "unfaceted"
    unfaceted.build(root=root, cache_dir=raw_cache)
    unfaceted.path_for("yaml", "C1-TEST", root=root).unlink()
    assert any("yaml" in p for p in unfaceted.verify(root=root, cache_dir=raw_cache))


def test_verify_flags_a_rendering_with_no_raw_record(tmp_path, raw_cache):
    root = tmp_path / "unfaceted"
    unfaceted.build(root=root, cache_dir=raw_cache)
    unfaceted.path_for("json", "C9-GHOST", root=root).write_text("{}")
    assert any("C9-GHOST" in p for p in unfaceted.verify(root=root, cache_dir=raw_cache))


def test_a_toon_round_trip_failure_is_recorded_not_fatal(tmp_path, raw_cache, monkeypatch):
    """TOON cannot round-trip every raw record; that is measured, not a crash.

    Unlike the faceted cache -- where a parity failure costs a record its place
    in the corpus -- the control reports the rate and carries on.
    """
    monkeypatch.setitem(unfaceted.PARSERS, "toon", lambda t: {})
    root = tmp_path / "unfaceted"
    report = unfaceted.build(root=root, cache_dir=raw_cache)
    assert report.parity_by_format["toon"] == 1
    assert unfaceted.verify(root=root, cache_dir=raw_cache) == []


def test_fingerprint_covers_formats_and_unfaceted_only():
    """The control must not invalidate when the *projection* changes.

    It reads no facet code, so an edit to ``facets.py`` has no bearing on its
    bytes — whereas the faceted cache must invalidate on exactly that. Asserted
    by recomputing the digest from the two files it is supposed to cover.
    """
    import hashlib
    from pathlib import Path

    here = Path(unfaceted.__file__).parent
    digest = hashlib.sha256()
    for name in ("formats.py", "unfaceted.py"):
        digest.update((here / name).read_bytes())
    assert unfaceted.fingerprint() == digest.hexdigest()[:16]

    # And it is genuinely a different guard from the faceted one.
    from airm import format_cache

    assert unfaceted.fingerprint() != format_cache.fingerprint()


def test_stale_cache_raises_rather_than_serving_old_bytes(tmp_path, raw_cache, monkeypatch):
    root = tmp_path / "unfaceted"
    unfaceted.build(root=root, cache_dir=raw_cache)
    monkeypatch.setattr(unfaceted, "fingerprint", lambda: "0000000000000000")
    with pytest.raises(unfaceted.StaleCacheError):
        unfaceted.load("json", "C1-TEST", root=root)


def test_unknown_format_names_the_known_ones():
    with pytest.raises(KeyError, match="unknown format"):
        unfaceted.render("xml", {})
