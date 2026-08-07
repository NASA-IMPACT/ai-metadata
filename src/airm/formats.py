"""The six representations under test — the independent variable of both experiments.

Every renderer takes the canonical :func:`airm.facets.facets` payload and returns
a string. None of them may touch the raw CMR record: that is what makes the
comparison a *format* comparison rather than a content comparison.

Each machine-readable format also has an inverse parser, so
:func:`validate_parity` can prove that a rendering carries exactly the facts it
started with. Prose cannot round-trip, so ``mat`` is instead held to
value-coverage: every canonical value must appear verbatim in the text.

TOON note
---------
TOON comes from ``toon_format``, the official implementation maintained by the
TOON authors at ``github.com/toon-format/toon-python``, pinned to a commit in
``pyproject.toml``. It is installed from git rather than PyPI because the PyPI
``toon-format`` release is a namespace reservation whose ``encode`` raises
``NotImplementedError``.

This replaces the third-party ``python-toon`` 0.1.3 the study previously used,
which needed a post-hoc ``[N,]{`` -> ``[N]{`` fix to stop it charging TOON one
spurious token per tabular array. The official encoder is spec-conformant
unpatched: its output is byte-identical to the patched ``python-toon`` output on
all 2,590 cached records, so the swap changes no token count. Its decoder is
strictly better -- ``python-toon`` fails to round-trip one corpus record whose
instrument descriptions contain embedded newlines, silently decoding a list item
into a bare string, which cost that record a place in the corpus at the parity
gate.
"""

from __future__ import annotations

import csv
import io
import json
import re
from typing import Any, Callable

import toon_format
import yaml

from .facets import (
    FACET_KEYS,
    canonical_coord,
    fact_set,
    fact_values,
    facets,
    flatten,
    num_str,
    scalar_str,
)

FORMATS: tuple[str, ...] = ("json", "csv", "yaml", "toon", "jsonld", "mat")


class ParityError(AssertionError):
    """Raised when a rendering does not carry the facts it was given."""


# --------------------------------------------------------------------------- #
# json
# --------------------------------------------------------------------------- #


def render_json(f: dict) -> str:
    """Compact JSON. The baseline every other format is reported against."""
    return json.dumps(f, ensure_ascii=False, separators=(",", ":"))


def parse_json(text: str) -> dict:
    return json.loads(text)


# --------------------------------------------------------------------------- #
# yaml
# --------------------------------------------------------------------------- #


def render_yaml(f: dict) -> str:
    return yaml.safe_dump(f, sort_keys=False, allow_unicode=True, default_flow_style=False).rstrip(
        "\n"
    )


def parse_yaml(text: str) -> dict:
    return yaml.safe_load(text)


# --------------------------------------------------------------------------- #
# toon
# --------------------------------------------------------------------------- #

def render_toon(f: dict) -> str:
    return toon_format.encode(f)


def parse_toon(text: str) -> dict:
    return toon_format.decode(text)


# --------------------------------------------------------------------------- #
# csv
# --------------------------------------------------------------------------- #

CSV_HEADER = ("path", "value")


def render_csv(f: dict) -> str:
    """Two-column ``path,value`` long form over the flattened facet tree.

    A nested record has no faithful *wide* CSV: a single row would either drop
    the repeated groups (platforms, keywords) or explode into hundreds of sparse
    columns whose width is set by the worst record in the corpus. The long form
    is lossless, is what a spreadsheet export of nested metadata actually looks
    like, and keeps the format's real cost -- repeating the full path on every
    row -- visible rather than hidden.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(CSV_HEADER)
    for path, value in flatten(f):
        writer.writerow([path, scalar_str(value)])
    return buf.getvalue().rstrip("\n")


#: One path token: either a key (``platforms``) or an index (``[0]``).
_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def _path_tokens(path: str) -> list[str | int]:
    """``"platforms[0].instruments[1]"`` -> ``["platforms", 0, "instruments", 1]``."""
    tokens: list[str | int] = []
    for m in _PATH_TOKEN.finditer(path):
        key, index = m.group(1), m.group(2)
        tokens.append(key if key is not None else int(index))
    return tokens


def parse_csv(text: str) -> dict:
    """Rebuild the nested payload from ``path,value`` rows."""
    rows = list(csv.reader(io.StringIO(text)))
    if rows and tuple(rows[0]) == CSV_HEADER:
        rows = rows[1:]

    root: dict = {}
    for path, value in rows:
        _assign(root, _path_tokens(path), value)
    return root


def _assign(root: dict, tokens: list[str | int], value: str) -> None:
    """Set ``root`` at ``tokens``, creating dicts and lists along the way."""
    cursor: Any = root
    for i, token in enumerate(tokens):
        last = i == len(tokens) - 1
        if last:
            child: Any = value
        else:
            child = [] if isinstance(tokens[i + 1], int) else {}

        if isinstance(token, int):
            while len(cursor) <= token:
                cursor.append(None)
            if last or cursor[token] is None:
                cursor[token] = child
            cursor = cursor[token]
        else:
            if last or cursor.get(token) is None:
                cursor[token] = child
            cursor = cursor[token]


# --------------------------------------------------------------------------- #
# jsonld
# --------------------------------------------------------------------------- #

JSONLD_CONTEXT = "https://schema.org/"


#: Facets with no schema.org property. JSON-LD's own escape hatch for
#: out-of-vocabulary data is ``additionalProperty``/``PropertyValue``, so they go
#: there rather than being invented as bare keys -- and the wrapper node is a
#: real part of what JSON-LD costs for metadata schema.org does not model.
#: The ``name`` is the facet key verbatim, which is what makes inversion exact.
JSONLD_EXTRA_KEYS: tuple[str, ...] = (
    "spectral_bands",
    "coordinate_system",
    "spatial_resolution",
    "temporal_resolution",
    "processing_level_description",
    "urls",
)


def render_jsonld(f: dict) -> str:
    """schema.org ``Dataset``, the shape a linked-data consumer would expect.

    Property names are schema.org's, not ours -- that renaming, plus the
    ``@context``/``@type`` scaffolding and the typed sub-nodes, *is* what
    JSON-LD costs. Follows the projection in ``sample_data/record_jsonld.json``.

    Six facets have no schema.org property and go through
    :data:`JSONLD_EXTRA_KEYS`. That is not a shortcut: schema.org genuinely does
    not model spectral bands or coordinate systems, and ``PropertyValue`` is what
    a real linked-data publisher would reach for.
    """
    doc: dict[str, Any] = {"@context": JSONLD_CONTEXT, "@type": "Dataset"}
    if "concept_id" in f:
        doc["identifier"] = f["concept_id"]
    if "cmr_link" in f:
        doc["url"] = f["cmr_link"]
    if "title" in f:
        doc["name"] = f["title"]
    if "short_name" in f:
        doc["alternateName"] = f["short_name"]
    if "doi" in f:
        doc["sameAs"] = f["doi"]
    if "agency" in f:
        doc["publisher"] = {"@type": "Organization", "name": f["agency"]}
    if "data_center" in f:
        doc["creator"] = {"@type": "Organization", "name": f["data_center"]}
    if "topic" in f:
        doc["about"] = {"@type": "DefinedTerm", "name": f["topic"]}
    if "summary" in f:
        doc["description"] = f["summary"]
    if "science_keywords" in f:
        doc["keywords"] = [
            {"@type": "DefinedTerm", "inDefinedTermSet": "GCMD", "termCode": kw}
            for kw in f["science_keywords"]
        ]
    if "variables" in f:
        doc["variableMeasured"] = f["variables"]
    if "platform" in f:
        doc["instrument"] = [{"@type": "Thing", "name": p} for p in f["platform"]]
    if "instrument" in f:
        doc["hasPart"] = [{"@type": "Thing", "name": i} for i in f["instrument"]]
    if "techniques" in f:
        doc["measurementTechnique"] = f["techniques"]
    if "bbox" in f:
        b = f["bbox"]
        box = " ".join(num_str(b[k]) for k in ("south", "west", "north", "east"))
        doc["spatialCoverage"] = {
            "@type": "Place",
            "geo": {"@type": "GeoShape", "box": box},
        }
    if "temporal" in f:
        t = f["temporal"]
        doc["temporalCoverage"] = f"{t['begin']}/{t['end']}" if "end" in t else t["begin"]
    if "data_format" in f:
        doc["encodingFormat"] = f["data_format"]
    if "data_volume" in f:
        doc["contentSize"] = f["data_volume"]
    if "processing_level" in f:
        doc["processingLevel"] = f["processing_level"]
    if "version" in f:
        doc["version"] = f["version"]
    if "collection_progress" in f:
        doc["creativeWorkStatus"] = f["collection_progress"]
    if "access_constraints" in f:
        doc["conditionsOfAccess"] = f["access_constraints"]
    if "use_constraints" in f:
        doc["usageInfo"] = f["use_constraints"]
    if "urls_publication" in f:
        doc["citation"] = f["urls_publication"]
    if "urls_repository" in f:
        doc["distribution"] = [
            {"@type": "DataDownload", "contentUrl": u} for u in f["urls_repository"]
        ]
    if "urls_additional" in f:
        doc["subjectOf"] = [{"@type": "CreativeWork", "url": u} for u in f["urls_additional"]]
    extras = [
        {"@type": "PropertyValue", "name": key, "value": f[key]}
        for key in JSONLD_EXTRA_KEYS
        if key in f
    ]
    if extras:
        doc["additionalProperty"] = extras
    return json.dumps(doc, ensure_ascii=False, separators=(",", ":"))


def parse_jsonld(text: str) -> dict:
    """Invert :func:`render_jsonld` back onto the canonical facet payload."""
    doc = json.loads(text)
    out: dict[str, Any] = {}
    if "identifier" in doc:
        out["concept_id"] = doc["identifier"]
    if "url" in doc:
        out["cmr_link"] = doc["url"]
    if "name" in doc:
        out["title"] = doc["name"]
    if "alternateName" in doc:
        out["short_name"] = doc["alternateName"]
    if "sameAs" in doc:
        out["doi"] = doc["sameAs"]
    if "publisher" in doc:
        out["agency"] = doc["publisher"]["name"]
    if "creator" in doc:
        out["data_center"] = doc["creator"]["name"]
    if "about" in doc:
        out["topic"] = doc["about"]["name"]
    if "description" in doc:
        out["summary"] = doc["description"]
    if "keywords" in doc:
        out["science_keywords"] = [k["termCode"] for k in doc["keywords"]]
    if "variableMeasured" in doc:
        out["variables"] = doc["variableMeasured"]
    if "instrument" in doc:
        out["platform"] = [node["name"] for node in doc["instrument"]]
    if "hasPart" in doc:
        out["instrument"] = [node["name"] for node in doc["hasPart"]]
    if "measurementTechnique" in doc:
        out["techniques"] = doc["measurementTechnique"]
    if "spatialCoverage" in doc:
        south, west, north, east = doc["spatialCoverage"]["geo"]["box"].split()
        out["bbox"] = {
            "west": canonical_coord(west),
            "south": canonical_coord(south),
            "east": canonical_coord(east),
            "north": canonical_coord(north),
        }
    if "temporalCoverage" in doc:
        value = doc["temporalCoverage"]
        begin, sep, end = value.partition("/")
        out["temporal"] = {"begin": begin, "end": end} if sep else {"begin": begin}
    if "encodingFormat" in doc:
        out["data_format"] = doc["encodingFormat"]
    if "contentSize" in doc:
        out["data_volume"] = doc["contentSize"]
    if "processingLevel" in doc:
        out["processing_level"] = doc["processingLevel"]
    if "version" in doc:
        out["version"] = doc["version"]
    if "creativeWorkStatus" in doc:
        out["collection_progress"] = doc["creativeWorkStatus"]
    if "conditionsOfAccess" in doc:
        out["access_constraints"] = doc["conditionsOfAccess"]
    if "usageInfo" in doc:
        out["use_constraints"] = doc["usageInfo"]
    if "citation" in doc:
        out["urls_publication"] = doc["citation"]
    if "distribution" in doc:
        out["urls_repository"] = [node["contentUrl"] for node in doc["distribution"]]
    if "subjectOf" in doc:
        out["urls_additional"] = [node["url"] for node in doc["subjectOf"]]
    for prop in doc.get("additionalProperty") or []:
        out[prop["name"]] = prop["value"]
    # Re-impose canonical key order. ``additionalProperty`` entries are read last
    # regardless of where their facets belong, so without this the round-trip
    # would return the right facts in the wrong order.
    return {k: out[k] for k in FACET_KEYS if k in out}


# --------------------------------------------------------------------------- #
# mat -- natural-language Metadata-as-Text
# --------------------------------------------------------------------------- #


def render_mat(f: dict) -> str:
    """Deterministic prose covering every canonical facet.

    Recovered from ``git show fe00bb6^:airm/representations.py`` and extended to
    the full facet set: the original omitted concept id, DOI, short name, topic
    and the keyword hierarchy, which meant prose silently carried *less* content
    than the structured formats -- exactly the confound this study exists to
    remove. Every facet value now appears verbatim, which
    :func:`missing_values_in_prose` enforces on every record.

    Structured facets are spelled out rather than joined arbitrarily: a
    resolution renders as "30 by 30 Meters" because ``x``, ``y`` and ``unit`` are
    three separate values the prose has to carry, not one composed string.

    No LLM is involved, so the representation is reproducible run to run.
    """
    parts: list[str] = []

    opening = f"{f['title']} is a NASA Earth-science dataset"
    if "short_name" in f:
        opening += f" (short name {f['short_name']})"
    if "concept_id" in f:
        opening += f" with CMR concept id {f['concept_id']}"
    if "data_center" in f:
        opening += f", archived at {f['data_center']}"
    if "agency" in f:
        opening += f" of {f['agency']}"
    parts.append(opening + ".")

    if "doi" in f:
        parts.append(f"Its DOI is {f['doi']}.")
    if "cmr_link" in f:
        parts.append(f"Its CMR record is at {f['cmr_link']}.")

    if "platform" in f:
        parts.append(f"Data were acquired by {', '.join(f['platform'])}.")
    if "instrument" in f:
        parts.append(f"Instruments: {', '.join(f['instrument'])}.")
    if "techniques" in f:
        parts.append(f"Measurement technique: {'; '.join(f['techniques'])}.")
    if "spectral_bands" in f:
        parts.append(f"Spectral bands and modes: {'; '.join(f['spectral_bands'])}.")

    if "topic" in f:
        parts.append(f"Its primary science topic is {f['topic']}.")
    if "science_keywords" in f:
        parts.append("Science keywords: " + "; ".join(f["science_keywords"]) + ".")
    if "variables" in f:
        parts.append(f"It measures {', '.join(f['variables'])}.")

    if "bbox" in f:
        b = f["bbox"]
        parts.append(
            f"Spatial coverage spans {num_str(b['west'])}° to {num_str(b['east'])}° "
            f"longitude and {num_str(b['south'])}° to {num_str(b['north'])}° latitude."
        )
    if "coordinate_system" in f:
        parts.append(f"Spatial bounds are given in {f['coordinate_system']}.")
    if "spatial_resolution" in f:
        parts.append("Spatial resolution: " + "; ".join(
            _dimensions_to_prose(r) for r in f["spatial_resolution"]
        ) + ".")

    if "temporal" in f:
        t = f["temporal"]
        if "end" in t:
            parts.append(f"Temporal coverage runs from {t['begin']} to {t['end']}.")
        else:
            parts.append(f"Temporal coverage begins {t['begin']}.")
    if "temporal_resolution" in f:
        parts.append("Temporal resolution: " + "; ".join(
            " ".join(scalar_str(r[k]) for k in ("value", "unit") if k in r)
            for r in f["temporal_resolution"]
        ) + ".")

    if "data_format" in f:
        parts.append(f"Distributed in {', '.join(f['data_format'])}.")
    if "data_volume" in f:
        v = f["data_volume"]
        parts.append("Data volume: " + " ".join(
            scalar_str(v[k]) for k in ("size", "unit", "basis") if k in v
        ) + ".")

    bits = []
    if "processing_level" in f:
        bits.append(f"processing level {f['processing_level']}")
    if "version" in f:
        bits.append(f"version {f['version']}")
    if "collection_progress" in f:
        bits.append(f"status {f['collection_progress']}")
    if bits:
        parts.append("Quality: " + ", ".join(bits) + ".")
    if "processing_level_description" in f:
        parts.append(f"Processing level description: {f['processing_level_description']}")

    if "access_constraints" in f:
        parts.append(f"Access constraints: {f['access_constraints']}")
    if "use_constraints" in f:
        parts.append(f"Use constraints: {f['use_constraints']}")

    # The three kinds are listed and `urls` is not: every value in `urls` also
    # appears in exactly one kind, so listing both would repeat every URL twice
    # in prose while the structured formats repeat it only as data. Prose parity
    # is value coverage, and each value is covered.
    for key, label in (
        ("urls_publication", "Publications"),
        ("urls_repository", "Data access"),
        ("urls_additional", "Related links"),
    ):
        if key in f:
            parts.append(f"{label}: {', '.join(f[key])}.")

    if "summary" in f:
        parts.append(f["summary"])

    return " ".join(parts)


def _dimensions_to_prose(entry: dict) -> str:
    """``{x, y, unit}`` -> ``"30 by 30 Meters"``; partial entries degrade cleanly."""
    x, y = entry.get("x"), entry.get("y")
    unit = entry.get("unit")
    if x is not None and y is not None:
        text = f"{scalar_str(x)} by {scalar_str(y)}"
    elif x is not None or y is not None:
        text = scalar_str(x if x is not None else y)
    else:
        return scalar_str(unit) if unit is not None else ""
    return f"{text} {scalar_str(unit)}" if unit is not None else text


# --------------------------------------------------------------------------- #
# Registries.
# --------------------------------------------------------------------------- #

RENDERERS: dict[str, Callable[[dict], str]] = {
    "json": render_json,
    "csv": render_csv,
    "yaml": render_yaml,
    "toon": render_toon,
    "jsonld": render_jsonld,
    "mat": render_mat,
}

#: Formats that can be read back into the canonical payload. ``mat`` is absent
#: by construction: prose is lossy about structure, which is the point of it.
PARSERS: dict[str, Callable[[str], dict]] = {
    "json": parse_json,
    "csv": parse_csv,
    "yaml": parse_yaml,
    "toon": parse_toon,
    "jsonld": parse_jsonld,
}

PROSE_FORMATS: tuple[str, ...] = ("mat",)


def render(fmt: str, f: dict) -> str:
    """Render the canonical payload ``f`` in format ``fmt``."""
    try:
        return RENDERERS[fmt](f)
    except KeyError:
        raise KeyError(f"unknown format {fmt!r}; known: {', '.join(RENDERERS)}") from None


def render_all(record: dict) -> dict[str, str]:
    """Render one CMR record in every format, from a single facet payload."""
    f = facets(record)
    return {fmt: render(fmt, f) for fmt in FORMATS}


# --------------------------------------------------------------------------- #
# Parity.
# --------------------------------------------------------------------------- #


def missing_values_in_prose(f: dict, text: str) -> list[str]:
    """Canonical values absent from ``text`` -- the prose parity check.

    Prose cannot be parsed back into structure, so its parity obligation is
    weaker but still checkable: every fact the payload asserts must be *present*
    in the text, so the model reading the prose has access to the same
    information as one reading the JSON.
    """
    return [value for value in fact_values(f) if value not in text]


def parity_report(f: dict) -> dict[str, dict]:
    """Per-format parity result for one payload.

    Each entry is ``{"ok": bool, "missing": [...], "extra": [...]}`` where
    ``missing``/``extra`` are facts the rendering lost or invented.
    """
    expected = fact_set(f)
    report: dict[str, dict] = {}

    for fmt, parser in PARSERS.items():
        try:
            got = fact_set(parser(render(fmt, f)))
        except Exception as exc:  # noqa: BLE001 - any failure is a parity failure
            report[fmt] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            continue
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        report[fmt] = {"ok": not missing and not extra, "missing": missing, "extra": extra}

    for fmt in PROSE_FORMATS:
        missing = missing_values_in_prose(f, render(fmt, f))
        report[fmt] = {
            "ok": not missing,
            "missing": [("<value>", v) for v in missing],
            "extra": [],
        }

    return report


def validate_parity(f: dict) -> None:
    """Raise :class:`ParityError` unless every format carries identical content."""
    report = parity_report(f)
    failures = {fmt: r for fmt, r in report.items() if not r["ok"]}
    if failures:
        lines = [f"content parity failed for {len(failures)} format(s):"]
        for fmt, r in failures.items():
            if "error" in r:
                lines.append(f"  {fmt}: {r['error']}")
                continue
            if r["missing"]:
                lines.append(f"  {fmt}: lost {r['missing'][:5]}")
            if r["extra"]:
                lines.append(f"  {fmt}: invented {r['extra'][:5]}")
        raise ParityError("\n".join(lines))


def check_record(record: dict) -> tuple[bool, dict[str, dict]]:
    """``(passes, report)`` for one raw CMR record."""
    report = parity_report(facets(record))
    return all(r["ok"] for r in report.values()), report


if __name__ == "__main__":  # pragma: no cover - manual inspection
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Render one record in every format.")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(Path(__file__).resolve().parents[2] / "sample_data" / "umm_json.json"),
    )
    args = parser.parse_args()

    record = json.loads(Path(args.path).read_text())
    payload = facets(record)
    for name, text in render_all(record).items():
        print(f"\n{'=' * 72}\n{name}  ({len(text)} chars)\n{'=' * 72}")
        print(text)
    print(f"\n{'=' * 72}\nparity\n{'=' * 72}")
    for name, result in parity_report(payload).items():
        print(f"  {name:8s} {'OK' if result['ok'] else result}")
