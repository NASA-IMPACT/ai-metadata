"""Tests for token counting and the Experiment 1 analysis."""

from __future__ import annotations

import csv

import pytest

from airm import exp1, tokens
from airm.config import FORMATS


def _record(cid: str, topic: str = "OCEANS", title: str = "A dataset") -> dict:
    return {
        "meta": {"concept-id": cid},
        "umm": {
            "EntryTitle": title,
            "Abstract": "Some description of the collection.",
            "ScienceKeywords": [{"Category": "EARTH SCIENCE", "Topic": topic}],
        },
    }


# --------------------------------------------------------------------------- #
# Tokenizers.
# --------------------------------------------------------------------------- #


def test_measure_reports_all_four_columns():
    m = tokens.measure("hello world", hf=False)
    assert m["chars"] == 11
    assert m["bytes"] == 11
    assert m["tiktoken"] > 0
    assert m["hf_tokens"] is None


def test_bytes_and_chars_diverge_on_non_ascii():
    """The character control must not quietly become a byte count."""
    m = tokens.measure("Températures 45°N", hf=False)
    assert m["bytes"] > m["chars"]


def test_unavailable_hf_tokenizer_yields_none_not_an_estimate(monkeypatch):
    def boom(*args, **kwargs):
        raise tokens.TokenizerUnavailable("no network")

    monkeypatch.setattr(tokens, "count_hf", boom)
    assert tokens.measure("hello", hf=True)["hf_tokens"] is None


def test_tiktoken_counts_are_stable_for_identical_text():
    assert tokens.count_tiktoken("a b c") == tokens.count_tiktoken("a b c")


# --------------------------------------------------------------------------- #
# Bootstrap.
# --------------------------------------------------------------------------- #


def test_bootstrap_ci_brackets_the_mean():
    values = [float(x) for x in range(1, 101)]
    lo, hi = exp1.bootstrap_ci(values, n=2000)
    assert lo < 50.5 < hi


def test_bootstrap_ci_is_deterministic_under_the_fixed_seed():
    values = [1.0, 5.0, 9.0, 12.0, 30.0]
    assert exp1.bootstrap_ci(values, n=500) == exp1.bootstrap_ci(values, n=500)


def test_bootstrap_ci_handles_degenerate_inputs():
    assert exp1.bootstrap_ci([7.0]) == (7.0, 7.0)
    lo, hi = exp1.bootstrap_ci([])
    assert lo != lo and hi != hi  # NaN


def test_a_tighter_sample_gives_a_tighter_interval():
    wide = exp1.bootstrap_ci([1.0, 100.0] * 25, n=2000)
    tight = exp1.bootstrap_ci([50.0, 51.0] * 25, n=2000)
    assert (tight[1] - tight[0]) < (wide[1] - wide[0])


# --------------------------------------------------------------------------- #
# Measurement.
# --------------------------------------------------------------------------- #


def test_every_record_yields_one_row_per_payload_and_format():
    rows = exp1.measure_records([_record("C1-A"), _record("C2-A")], hf=False)
    assert len(rows) == 2 * len(exp1.PAYLOADS) * len(FORMATS)
    assert {r["format"] for r in rows} == set(FORMATS)
    assert {r["payload"] for r in rows} == set(exp1.PAYLOADS)
    # Every (payload, format) cell is filled exactly once per record.
    cells = [(r["concept_id"], r["payload"], r["format"]) for r in rows]
    assert len(cells) == len(set(cells))


def test_the_unfaceted_payload_dwarfs_the_faceted_one():
    """The projection is what makes the study's payload small.

    Measured on the real ATL08 record -- a hand-built stub has so few fields that
    its raw UMM is *smaller* than the facet renderings, which says nothing about
    what CMR actually serves.
    """
    import json

    from airm.config import SAMPLE_DATA_DIR

    record = {
        "meta": {"concept-id": "C3565574177-NSIDC_CPRD"},
        "umm": json.loads((SAMPLE_DATA_DIR / "umm_json.json").read_text()),
    }
    rows = exp1.measure_records([record], hf=False)
    by = {(r["payload"], r["format"]): r["tiktoken"] for r in rows}
    # Compared per format, not cheapest-against-dearest: prose over the raw
    # record is cheaper than JSON-LD over the projection, so a global min/max
    # comparison says nothing about content. Like for like is the invariant.
    for fmt in FORMATS:
        assert by[(exp1.UNFACETED, fmt)] > by[(exp1.FACETED, fmt)], fmt


def test_the_reference_is_the_unfaceted_json_row_not_a_second_measurement():
    """``json_umm`` was measuring the same string as unfaceted JSON.

    If these ever diverge, the reference has stopped describing the raw record.
    """
    import json

    record = _record("C1-A")
    rows = exp1.measure_records([record], hf=False)
    unfaceted_json = next(
        r for r in rows if r["payload"] == exp1.UNFACETED and r["format"] == "json"
    )
    raw = json.dumps(record["umm"], ensure_ascii=False, separators=(",", ":"))
    from airm import tokens

    assert unfaceted_json["tiktoken"] == tokens.measure(raw, hf=False)["tiktoken"]


def test_topic_is_carried_onto_every_row():
    rows = exp1.measure_records([_record("C1-A", topic="Cryosphere")], hf=False)
    assert {r["topic"] for r in rows} == {"CRYOSPHERE"}


# --------------------------------------------------------------------------- #
# Summarising.
# --------------------------------------------------------------------------- #


@pytest.fixture
def rows():
    return exp1.measure_records(
        [_record(f"C{i}-A", topic="OCEANS" if i % 2 else "ATMOSPHERE") for i in range(12)],
        hf=False,
    )


def test_summary_covers_every_format(rows):
    summaries = exp1.summarise(rows, payload=exp1.FACETED)
    assert {s.fmt for s in summaries} == set(FORMATS)


def test_the_baseline_is_exactly_one_relative_to_itself(rows):
    summaries = {s.fmt: s for s in exp1.summarise(rows, payload=exp1.FACETED)}
    base = summaries["json"]
    assert base.ratio_mean == pytest.approx(1.0)
    assert base.pct_vs_baseline == pytest.approx(0.0)


def test_ratios_are_paired_per_record_not_a_ratio_of_means(rows):
    """Records vary hugely in size; unpaired means bury a real format effect."""
    summaries = {s.fmt: s for s in exp1.summarise(rows, payload=exp1.FACETED)}
    per_record = {}
    for r in rows:
        # Filtering by payload is load-bearing: without it the unfaceted rows
        # overwrite the faceted ones and the expectation silently changes.
        if r["payload"] != exp1.FACETED:
            continue
        per_record.setdefault(r["concept_id"], {})[r["format"]] = r["tiktoken"]
    expected = sum(v["jsonld"] / v["json"] for v in per_record.values()) / len(per_record)
    assert summaries["jsonld"].ratio_mean == pytest.approx(expected)


def test_summary_counts_every_record(rows):
    for s in exp1.summarise(rows, payload=exp1.FACETED):
        assert s.n == 12


def test_a_measure_with_no_data_is_skipped_rather_than_zeroed(rows):
    assert exp1.summarise(rows, measure="hf_tokens", payload=exp1.FACETED) == []


def test_per_topic_means_split_by_domain(rows):
    means = exp1.per_topic_means(rows, payload=exp1.FACETED)
    assert set(means) == {"OCEANS", "ATMOSPHERE"}
    assert set(means["OCEANS"]) == set(FORMATS)


# --------------------------------------------------------------------------- #
# Output.
# --------------------------------------------------------------------------- #


def test_rows_write_to_csv_with_a_stable_schema(tmp_path, rows):
    path = exp1.write_rows(rows, tmp_path / "exp1_tokens.csv")
    read = list(csv.DictReader(path.open()))
    assert len(read) == len(rows)
    assert list(read[0]) == ["concept_id", "topic", "payload", "format", *exp1.MEASURES]


def test_run_writes_every_artefact(tmp_path, monkeypatch):
    monkeypatch.setattr(exp1.cmr, "load_corpus", lambda *a, **k: [_record(f"C{i}-A") for i in range(6)])
    result = exp1.run(hf=False, run_id="testrun", out_dir=tmp_path)

    assert result["records"] == 6
    assert result["rows"] == 6 * len(exp1.PAYLOADS) * len(FORMATS)
    assert result["reference_raw_umm"]["n"] == 6
    assert result["tokenizers"]["hf"] is None
    for name in ("exp1_tokens.csv", "exp1_summary.json", "exp1_tokens.png"):
        assert (tmp_path / name).exists(), name


# --------------------------------------------------------------------------- #
# Live: the built corpus.
# --------------------------------------------------------------------------- #


@pytest.mark.live
def test_live_toon_wins_on_the_shape_it_was_designed_for():
    """TOON loses on CMR records; confirm that is the data shape, not a bug."""
    from airm import formats

    tabular = {"rows": [{"id": i, "name": f"item{i}", "role": "admin"} for i in range(20)]}
    toon = tokens.count_tiktoken(formats.render("toon", tabular))
    js = tokens.count_tiktoken(formats.render("json", tabular))
    assert toon < js * 0.8, "TOON should be much cheaper on a uniform tabular array"
