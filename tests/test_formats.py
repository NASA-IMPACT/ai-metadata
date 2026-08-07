"""Content-parity tests — the methodological core of both experiments.

If these fail, every number the study produces is uninterpretable: a format
comparison only measures format when the formats carry identical content.
"""

from __future__ import annotations

import json
import re

import pytest

from airm import cmr, facets, formats
from airm.config import SAMPLE_DATA_DIR

SAMPLE_UMM = json.loads((SAMPLE_DATA_DIR / "umm_json.json").read_text())


@pytest.fixture
def sample() -> dict:
    return facets.facets(SAMPLE_UMM)


# --------------------------------------------------------------------------- #
# Edge-case payloads: each one broke, or could plausibly break, a renderer.
# --------------------------------------------------------------------------- #

EDGE_CASES: dict[str, dict] = {
    "minimal": {"title": "Bare Record"},
    "no_bbox_no_temporal": {
        "concept_id": "C1-TEST",
        "title": "Sparse",
        "platform": ["Aqua"],
    },
    "commas_and_quotes": {
        "concept_id": "C2-TEST",
        "title": 'A dataset, with "quotes", commas and a > sign',
        "summary": 'He said "hello, world" >> loudly',
    },
    "newline_in_summary": {
        "concept_id": "C3-TEST",
        "title": "Multiline",
        "summary": "First line.\nSecond line.\n\nFourth line.",
    },
    "numeric_looking_strings": {
        "concept_id": "C4-TEST",
        "title": "Version Trap",
        # "007" and "3" must survive as strings, not become 7 and 3.
        "processing_level": "3",
        "version": "007",
        "collection_progress": "ACTIVE",
    },
    "negative_and_zero_coords": {
        "concept_id": "C5-TEST",
        "title": "Coordinates",
        # 0 was the case python-toon encoded as `0` regardless of float-ness.
        "bbox": {"west": -180, "south": 0, "east": 0.5, "north": 88.25},
    },
    "unicode": {
        "concept_id": "C6-TEST",
        "title": "Températures de surface de la mer — °C",
        "summary": "Coverage spans 45°N–60°N; naïve résumé.",
    },
    "many_platforms": {
        "concept_id": "C7-TEST",
        "title": "Multi-platform",
        # Platform and instrument are separate flat facets: an instrument is a
        # field CMR records, not a child of a platform node.
        "platform": ["Terra", "Aqua", "Suomi-NPP"],
        "instrument": ["MODIS", "ASTER", "CERES"],
    },
    "gcmd_paths": {
        "concept_id": "C8-TEST",
        "title": "Keywords",
        "science_keywords": [
            "EARTH SCIENCE > OCEANS > SEA ICE > SEA ICE CONCENTRATION",
            "EARTH SCIENCE > CRYOSPHERE",
        ],
    },
    "colon_in_title": {
        "concept_id": "C9-TEST",
        # YAML and TOON both treat ": " specially; this must still round-trip.
        "title": "MERRA-2 tavg1_2d_flx_Nx: 2d,1-Hourly,Time-Averaged",
    },
}


# --------------------------------------------------------------------------- #
# Rendering.
# --------------------------------------------------------------------------- #


def test_all_six_formats_render(sample):
    rendered = {fmt: formats.render(fmt, sample) for fmt in formats.FORMATS}
    assert set(rendered) == set(formats.FORMATS)
    assert all(text.strip() for text in rendered.values())


def test_render_all_drives_every_format_from_one_payload():
    rendered = formats.render_all(SAMPLE_UMM)
    assert set(rendered) == set(formats.FORMATS)
    # The title is content, so it must appear in every single representation.
    title = facets.facets(SAMPLE_UMM)["title"]
    for fmt, text in rendered.items():
        assert title in text, f"{fmt} dropped the title"


def test_unknown_format_names_the_known_ones(sample):
    with pytest.raises(KeyError, match="unknown format"):
        formats.render("xml", sample)


# --------------------------------------------------------------------------- #
# Parity — the load-bearing tests.
# --------------------------------------------------------------------------- #


def test_sample_record_has_full_parity_across_all_six_formats(sample):
    formats.validate_parity(sample)


@pytest.mark.parametrize("name", sorted(EDGE_CASES))
def test_edge_case_payloads_keep_parity(name):
    formats.validate_parity(EDGE_CASES[name])


@pytest.mark.parametrize("name", sorted(EDGE_CASES))
def test_edge_case_payloads_survive_prose_value_coverage(name):
    payload = EDGE_CASES[name]
    assert formats.missing_values_in_prose(payload, formats.render_mat(payload)) == []


def test_parity_report_flags_a_format_that_loses_content(sample, monkeypatch):
    """The validator must actually be able to fail, not just always pass."""

    def lossy(f: dict) -> str:
        return json.dumps({k: v for k, v in f.items() if k != "title"})

    monkeypatch.setitem(formats.RENDERERS, "json", lossy)
    report = formats.parity_report(sample)
    assert report["json"]["ok"] is False
    assert any(path == "title" for path, _ in report["json"]["missing"])
    with pytest.raises(formats.ParityError, match="parity failed"):
        formats.validate_parity(sample)


def test_parity_report_flags_a_format_that_invents_content(sample, monkeypatch):
    def inventive(f: dict) -> str:
        return json.dumps({**f, "title": "Something Else Entirely"})

    monkeypatch.setitem(formats.RENDERERS, "json", inventive)
    report = formats.parity_report(sample)
    assert report["json"]["ok"] is False


def test_prose_parity_fails_when_a_value_is_missing(sample, monkeypatch):
    monkeypatch.setitem(formats.RENDERERS, "mat", lambda f: "Nothing useful here.")
    report = formats.parity_report(sample)
    assert report["mat"]["ok"] is False


# --------------------------------------------------------------------------- #
# Per-format specifics.
# --------------------------------------------------------------------------- #


def test_toon_tabular_headers_follow_the_spec(sample):
    """Headers must read `[N]{...}`, not `[N,]{...}`.

    The retired `python-toon` emitted the delimiter inside the header, which
    charged TOON one spurious token per tabular array. `toon_format` does not,
    but the assertion stays: it is the property Experiment 1's token counts
    depend on, and it should fail loudly if the dependency ever regresses.
    """
    text = formats.render_toon(sample)
    assert not re.search(r"\[\d+,\]\{", text), "TOON header leaked a spurious delimiter token"


def test_toon_round_trips_multiline_string_values():
    """A value containing a newline must survive encode/decode.

    Real UMM instrument descriptions carry embedded newlines (see
    C2105107978-NOAA_NCEI). `python-toon` decoded such a list item back into a
    bare string, silently failing the parity gate and costing the record its
    place in the corpus.
    """
    payload = {
        "title": "T",
        "instrument": ["Line one.\n     Line two."],
    }
    assert formats.parse_toon(formats.render_toon(payload)) == payload


def test_csv_is_a_two_column_long_form_with_a_header(sample):
    text = formats.render_csv(sample)
    lines = text.splitlines()
    assert lines[0] == "path,value"
    assert len(lines) > 5


def test_csv_path_tokeniser_handles_nesting_and_indices():
    assert formats._path_tokens("bbox.west") == ["bbox", "west"]
    assert formats._path_tokens("platforms[0].instruments[1]") == [
        "platforms",
        0,
        "instruments",
        1,
    ]
    assert formats._path_tokens("science_keywords[10]") == ["science_keywords", 10]


def test_jsonld_uses_schema_org_property_names(sample):
    doc = json.loads(formats.render_jsonld(sample))
    assert doc["@context"] == formats.JSONLD_CONTEXT
    assert doc["@type"] == "Dataset"
    # The renaming is exactly what JSON-LD costs; assert it actually happens.
    assert "name" in doc and "title" not in doc
    assert "description" in doc and "summary" not in doc


def test_jsonld_bbox_uses_schema_org_box_ordering(sample):
    doc = json.loads(formats.render_jsonld(sample))
    south, west, north, east = doc["spatialCoverage"]["geo"]["box"].split()
    b = sample["bbox"]
    assert (float(south), float(west), float(north), float(east)) == (
        b["south"],
        b["west"],
        b["north"],
        b["east"],
    )


def test_mat_covers_the_facets_the_old_renderer_dropped(sample):
    """The recovered prose renderer omitted these, which was a real confound."""
    text = formats.render_mat(sample)
    for key in ("short_name", "doi", "topic"):
        assert sample[key] in text
    for keyword in sample["science_keywords"]:
        assert keyword in text


def test_mat_is_deterministic(sample):
    assert formats.render_mat(sample) == formats.render_mat(sample)


# --------------------------------------------------------------------------- #
# Facet extraction.
# --------------------------------------------------------------------------- #


def test_facets_omit_absent_fields_rather_than_emitting_null():
    """`"bbox": null` and an absent key carry the same fact but not the same cost."""
    f = facets.facets({"meta": {"concept-id": "C1-TEST"}, "umm": {"EntryTitle": "T"}})
    assert "bbox" not in f
    assert "platforms" not in f
    assert f["title"] == "T"


def test_facets_key_order_is_canonical(sample):
    assert list(sample) == [k for k in facets.FACET_KEYS if k in sample]


def test_integral_coordinates_are_canonicalised_to_int():
    """`-180.0` and `-180` state the same fact; the `.0` is a pure token tax."""
    record = {
        "umm": {
            "EntryTitle": "T",
            "SpatialExtent": {
                "HorizontalSpatialDomain": {
                    "Geometry": {
                        "BoundingRectangles": [
                            {
                                "WestBoundingCoordinate": -180.0,
                                "SouthBoundingCoordinate": 0.0,
                                "EastBoundingCoordinate": 180.0,
                                "NorthBoundingCoordinate": 88.25,
                            }
                        ]
                    }
                }
            },
        }
    }
    bbox = facets.facets(record)["bbox"]
    assert bbox == {"west": -180, "south": 0, "east": 180, "north": 88.25}
    assert all(isinstance(bbox[k], int) for k in ("west", "south", "east"))
    assert isinstance(bbox["north"], float)


def test_integral_coordinates_cost_the_same_in_every_format():
    payload = {"title": "T", "bbox": {"west": -180, "south": 0, "east": 180, "north": 88}}
    for fmt in ("json", "yaml", "toon", "csv"):
        text = formats.render(fmt, payload)
        assert "-180.0" not in text and "88.0" not in text, f"{fmt} re-introduced a .0"


def test_partial_bounding_box_is_dropped_whole():
    record = {
        "umm": {
            "EntryTitle": "T",
            "SpatialExtent": {
                "HorizontalSpatialDomain": {
                    "Geometry": {"BoundingRectangles": [{"WestBoundingCoordinate": -180.0}]}
                }
            },
        }
    }
    assert "bbox" not in facets.facets(record)


def test_booleans_do_not_print_as_integers():
    assert facets.num_str(True) == "true"
    assert facets.num_str(False) == "false"
    assert facets.num_str(1) == "1"


def test_science_keywords_are_gcmd_paths():
    record = {
        "umm": {
            "EntryTitle": "T",
            "ScienceKeywords": [
                {"Category": "EARTH SCIENCE", "Topic": "OCEANS", "Term": "SEA ICE"},
                {"Category": "EARTH SCIENCE", "Topic": "OCEANS", "Term": "SEA ICE"},
            ],
        }
    }
    kws = facets.facets(record)["science_keywords"]
    assert kws == ["EARTH SCIENCE > OCEANS > SEA ICE"]  # deduplicated


def test_topic_is_the_first_science_keyword_topic():
    record = {
        "umm": {
            "EntryTitle": "T",
            "ScienceKeywords": [{"Topic": "CRYOSPHERE"}, {"Topic": "OCEANS"}],
        }
    }
    assert facets.facets(record)["topic"] == "CRYOSPHERE"


# --------------------------------------------------------------------------- #
# Empty containers must not reach a renderer.
# --------------------------------------------------------------------------- #


def test_platform_without_instruments_omits_the_key():
    """An empty facet asserts no fact, so it must not reach a renderer.

    ``flatten`` yields nothing for ``[]``, so the parity check is blind to it —
    but JSON, YAML and TOON still print the brackets while CSV, JSON-LD and prose
    do not. That is token cost with no content behind it, which is exactly the
    confound the canonical payload exists to prevent.
    """
    record = {"umm": {"EntryTitle": "T", "Platforms": [{"ShortName": "Suomi-NPP"}]}}
    payload = facets.facets(record)
    assert payload["platform"] == ["Suomi-NPP"]
    assert "instrument" not in payload


def test_no_format_serialises_an_empty_container():
    """No rendering may contain an empty array.

    Asserted on the empty-array literal rather than on the facet name: JSON-LD
    maps ``platform`` onto schema.org's ``instrument`` property, so a name check
    would fire on a facet that is legitimately present.
    """
    record = {"umm": {"EntryTitle": "T", "Platforms": [{"ShortName": "Suomi-NPP"}]}}
    payload = facets.facets(record)
    assert "instrument" not in payload
    for fmt in formats.FORMATS:
        text = formats.render(fmt, payload)
        assert "[]" not in text, f"{fmt} serialised an empty container"


def test_instruments_survive_when_present():
    """Instruments are their own facet, flattened across every platform."""
    record = {
        "umm": {
            "EntryTitle": "T",
            "Platforms": [
                {"ShortName": "Terra", "Instruments": [{"ShortName": "MODIS"}]},
                {"ShortName": "Aqua", "Instruments": [{"ShortName": "AIRS"}]},
            ],
        }
    }
    payload = facets.facets(record)
    assert payload["platform"] == ["Terra", "Aqua"]
    assert payload["instrument"] == ["MODIS", "AIRS"]


# --------------------------------------------------------------------------- #
# Live: parity must hold on real records, not just curated ones.
# --------------------------------------------------------------------------- #


@pytest.mark.live
def test_parity_holds_across_real_records_from_every_topic():
    """The corpus build in Step 5 depends on this holding broadly."""
    from airm.config import CMR_TOPICS

    failures: list[str] = []
    checked = 0
    for topic in CMR_TOPICS:
        for record in cmr.search_by_topic(topic, count=3, use_cache=False):
            checked += 1
            ok, report = formats.check_record(record)
            if not ok:
                bad = {k: v for k, v in report.items() if not v["ok"]}
                failures.append(f"{cmr.concept_id(record)} ({topic}): {bad}")

    assert checked >= 30
    assert not failures, f"{len(failures)}/{checked} real records failed parity:\n" + "\n".join(
        failures[:5]
    )
