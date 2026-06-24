"""Record renderers — the independent variable of Experiment 1 (format ablation).

Each renderer takes a CMR UMM-JSON item (``{"meta": ..., "umm": ...}``) and
returns a single string. The four representations correspond to Report.md §6
Experiment 1:

    (a) raw_umm_json        -- the precise canonical record
    (b) flattened_jsonld    -- Schema.org / STAC-style flattened projection
    (c) dot_breadcrumb      -- dot-notation key = value breadcrumbs
    (d) metadata_as_text    -- natural-language "Metadata-as-Text" summary

``metadata_as_text`` is the renderer the Experiment 1 auto-tune loop edits, so
it is kept deterministic and self-contained (no LLM call) for reproducibility.

The small ``_extract_*`` helpers pull the common facets (instrument, variables,
spatial/temporal coverage, quality) once; renderers compose them differently so
the *content* is held roughly constant while the *format* varies.
"""

from __future__ import annotations

import json
from typing import Callable

# --------------------------------------------------------------------------- #
# Facet extractors — shared, format-agnostic content.
# --------------------------------------------------------------------------- #


def _umm(record: dict) -> dict:
    return record.get("umm", record)


def extract_title(record: dict) -> str:
    u = _umm(record)
    return u.get("EntryTitle") or u.get("ShortName") or "(untitled)"


def extract_summary(record: dict) -> str:
    return (_umm(record).get("Abstract") or "").strip()


def extract_platforms(record: dict) -> list[dict]:
    """Return [{platform, instruments:[...]}] from UMM Platforms."""
    out = []
    for p in _umm(record).get("Platforms", []) or []:
        instruments = [i.get("ShortName") for i in p.get("Instruments", []) or [] if i.get("ShortName")]
        out.append({"platform": p.get("ShortName"), "instruments": instruments})
    return out


def extract_variables(record: dict) -> list[str]:
    """Most specific science-keyword term per entry (the 'variables')."""
    variables = []
    for sk in _umm(record).get("ScienceKeywords", []) or []:
        for level in ("VariableLevel3", "VariableLevel2", "VariableLevel1", "Term", "Topic"):
            if sk.get(level):
                variables.append(sk[level].title())
                break
    # preserve order, dedupe
    seen: set[str] = set()
    return [v for v in variables if not (v in seen or seen.add(v))]


def extract_bbox(record: dict) -> dict | None:
    """Return {west, south, east, north} of the first bounding rectangle."""
    geom = (
        _umm(record)
        .get("SpatialExtent", {})
        .get("HorizontalSpatialDomain", {})
        .get("Geometry", {})
    )
    rects = geom.get("BoundingRectangles") or []
    if not rects:
        return None
    r = rects[0]
    return {
        "west": r.get("WestBoundingCoordinate"),
        "south": r.get("SouthBoundingCoordinate"),
        "east": r.get("EastBoundingCoordinate"),
        "north": r.get("NorthBoundingCoordinate"),
    }


def extract_temporal(record: dict) -> dict | None:
    """Return {begin, end} ISO strings (end may be None when ongoing)."""
    for te in _umm(record).get("TemporalExtents", []) or []:
        for rng in te.get("RangeDateTimes", []) or []:
            return {
                "begin": rng.get("BeginningDateTime"),
                "end": rng.get("EndingDateTime")
                or ("present" if te.get("EndsAtPresentFlag") else None),
            }
    return None


def extract_quality(record: dict) -> dict:
    u = _umm(record)
    return {
        "processing_level": (u.get("ProcessingLevel") or {}).get("Id"),
        "collection_progress": u.get("CollectionProgress"),
        "version": u.get("Version"),
    }


def extract_data_center(record: dict) -> str | None:
    for dc in _umm(record).get("DataCenters", []) or []:
        if dc.get("ShortName"):
            return dc["ShortName"]
    return None


def facets(record: dict) -> dict:
    """All extracted facets in one dict — the common content payload."""
    return {
        "concept_id": record.get("meta", {}).get("concept-id"),
        "title": extract_title(record),
        "summary": extract_summary(record),
        "platforms": extract_platforms(record),
        "variables": extract_variables(record),
        "bbox": extract_bbox(record),
        "temporal": extract_temporal(record),
        "quality": extract_quality(record),
        "data_center": extract_data_center(record),
    }


# --------------------------------------------------------------------------- #
# Renderers.
# --------------------------------------------------------------------------- #


def raw_umm_json(record: dict) -> str:
    """(a) The precise canonical UMM record, pretty-printed."""
    return json.dumps(_umm(record), indent=2, ensure_ascii=False)


def flattened_jsonld(record: dict) -> str:
    """(b) Schema.org ``Dataset`` / STAC-style flattened JSON-LD projection."""
    f = facets(record)
    bbox = f["bbox"]
    spatial = None
    if bbox and None not in bbox.values():
        # GeoJSON-style bbox order: west south east north
        spatial = {
            "@type": "Place",
            "geo": {
                "@type": "GeoShape",
                "box": f"{bbox['south']} {bbox['west']} {bbox['north']} {bbox['east']}",
            },
        }
    temporal = f["temporal"]
    doc = {
        "@context": "https://schema.org/",
        "@type": "Dataset",
        "identifier": f["concept_id"],
        "name": f["title"],
        "description": f["summary"],
        "variableMeasured": f["variables"],
        "instrument": [
            i for p in f["platforms"] for i in p["instruments"]
        ],
        "platform": [p["platform"] for p in f["platforms"] if p["platform"]],
        "spatialCoverage": spatial,
        "temporalCoverage": (
            f"{temporal['begin']}/{temporal['end']}" if temporal else None
        ),
        "version": f["quality"]["version"],
        "creator": {"@type": "Organization", "name": f["data_center"]},
        "processingLevel": f["quality"]["processing_level"],
        "creativeWorkStatus": f["quality"]["collection_progress"],
    }
    # drop null/empty values for a tight projection
    doc = {k: v for k, v in doc.items() if v not in (None, [], "")}
    return json.dumps(doc, indent=2, ensure_ascii=False)


def dot_breadcrumb(record: dict) -> str:
    """(c) Flattened ``a.b.c = value`` breadcrumbs over the UMM tree."""
    lines: list[str] = []

    def walk(node, prefix: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(node, list):
            for idx, v in enumerate(node):
                walk(v, f"{prefix}[{idx}]")
        else:
            lines.append(f"{prefix} = {node}")

    walk(_umm(record), "")
    return "\n".join(lines)


def metadata_as_text(record: dict) -> str:
    """(d) Natural-language Metadata-as-Text summary (deterministic template).

    This renderer is the auto-tune loop's editable target for Experiment 1.
    """
    f = facets(record)
    parts: list[str] = []
    parts.append(f"{f['title']} is a NASA Earth-science dataset")
    if f["data_center"]:
        parts[-1] += f" archived at {f['data_center']}"
    parts[-1] += "."

    if f["platforms"]:
        plat_strs = []
        for p in f["platforms"]:
            if p["instruments"]:
                plat_strs.append(
                    f"the {', '.join(p['instruments'])} instrument(s) aboard {p['platform']}"
                )
            elif p["platform"]:
                plat_strs.append(p["platform"])
        if plat_strs:
            parts.append(f"Data were acquired by {'; '.join(plat_strs)}.")

    if f["variables"]:
        parts.append(f"It measures {', '.join(f['variables'])}.")

    bbox = f["bbox"]
    if bbox and None not in bbox.values():
        parts.append(
            "Spatial coverage spans "
            f"{bbox['west']}° to {bbox['east']}° longitude and "
            f"{bbox['south']}° to {bbox['north']}° latitude."
        )

    t = f["temporal"]
    if t and t.get("begin"):
        end = t.get("end") or "present"
        parts.append(f"Temporal coverage runs from {t['begin']} to {end}.")

    q = f["quality"]
    qual_bits = []
    if q["processing_level"]:
        qual_bits.append(f"processing level {q['processing_level']}")
    if q["version"]:
        qual_bits.append(f"version {q['version']}")
    if q["collection_progress"]:
        qual_bits.append(f"status {q['collection_progress'].lower()}")
    if qual_bits:
        parts.append("Quality: " + ", ".join(qual_bits) + ".")

    if f["summary"]:
        parts.append(f["summary"])

    return " ".join(parts)


# Registry used by the runner and experiments. Order is the canonical
# Experiment 1 condition order.
RENDERERS: dict[str, Callable[[dict], str]] = {
    "raw_umm_json": raw_umm_json,
    "flattened_jsonld": flattened_jsonld,
    "dot_breadcrumb": dot_breadcrumb,
    "metadata_as_text": metadata_as_text,
}


if __name__ == "__main__":  # pragma: no cover - demo entrypoint
    import argparse

    from .cmr import load_cached, search_collections

    parser = argparse.ArgumentParser(description="Render all representations of one record.")
    parser.add_argument("--demo", metavar="CONCEPT_ID", help="render a cached concept-id")
    parser.add_argument("--keyword", help="fetch one record by keyword instead")
    args = parser.parse_args()

    rec = None
    if args.demo:
        rec = load_cached(args.demo)
    if rec is None:
        items = search_collections(args.keyword or "vegetation height", count=1)
        rec = items[0]

    for name, fn in RENDERERS.items():
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        print(fn(rec))
