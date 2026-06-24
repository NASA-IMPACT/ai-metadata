"""Field-description conditions — the independent variable of Experiment 2.

Report.md §6 Experiment 2 ("description-quality ablation") compares three
schema presentations of CMR's searchable parameters, treated as tool/function
parameters:

    (a) bare         -- field names only
    (b) described    -- names + one-line inline descriptions
    (c) rich         -- descriptions + enums + example values

The function-calling literature (Gorilla, ToolAlpaca, Databricks) predicts a
large (a)->(c) jump in correct field selection and value formatting. The field
text in CONDITION_RICH is the auto-tune loop's editable target for Experiment 2.
"""

from __future__ import annotations

import json

# A curated subset of CMR collection search parameters, modelled as tool params.
# https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html
FIELDS: dict[str, dict] = {
    "keyword": {
        "type": "string",
        "description": "Free-text terms matched across title, summary, science keywords, instrument and platform.",
        "example": "vegetation canopy height",
    },
    "platform": {
        "type": "string",
        "description": "Name of the observing platform (satellite, aircraft, or field station).",
        "enum_hint": "GCMD platform keywords, e.g. ICESat-2, Terra, Aqua, Sentinel-1A",
        "example": "ICESat-2",
    },
    "instrument": {
        "type": "string",
        "description": "Name of the sensor/instrument that acquired the data.",
        "enum_hint": "GCMD instrument keywords, e.g. ATLAS, MODIS, VIIRS",
        "example": "ATLAS",
    },
    "science_keywords": {
        "type": "string",
        "description": "Controlled Earth-science variable from the GCMD hierarchy (Category > Topic > Term > VariableLevel).",
        "enum_hint": "GCMD science keywords, e.g. BIOSPHERE > VEGETATION > CANOPY CHARACTERISTICS > VEGETATION HEIGHT",
        "example": "VEGETATION HEIGHT",
    },
    "bounding_box": {
        "type": "string",
        "description": "Spatial filter as a lon/lat rectangle, lower-left then upper-right.",
        "enum_hint": "format: west,south,east,north in decimal degrees, WGS84",
        "example": "-75,-15,-45,5",
    },
    "temporal": {
        "type": "string",
        "description": "Temporal filter as an ISO-8601 datetime range.",
        "enum_hint": "format: <start>,<end> each YYYY-MM-DDThh:mm:ssZ; open-ended sides may be empty",
        "example": "2024-12-01T00:00:00Z,2024-12-31T23:59:59Z",
    },
    "processing_level_id": {
        "type": "string",
        "description": "Data product processing level.",
        "enum_hint": "one of: 0, 1, 1A, 1B, 2, 3, 4",
        "example": "3",
    },
    "collection_data_type": {
        "type": "string",
        "description": "Maturity / production status of the collection.",
        "enum_hint": "one of: SCIENCE_QUALITY, NEAR_REAL_TIME, OTHER",
        "example": "SCIENCE_QUALITY",
    },
}


def render_fields(condition: str) -> str:
    """Render the search-parameter schema under one of the three conditions.

    condition in {"bare", "described", "rich"}.
    Returns a JSON-Schema-like string suitable for a system prompt / tool spec.
    """
    if condition not in {"bare", "described", "rich"}:
        raise ValueError(f"unknown condition: {condition!r}")

    props: dict[str, dict] = {}
    for name, spec in FIELDS.items():
        entry: dict = {"type": spec["type"]}
        if condition in {"described", "rich"}:
            entry["description"] = spec["description"]
        if condition == "rich":
            if spec.get("enum_hint"):
                # carried as a description suffix; CMR values are open vocab so
                # we hint rather than hard-enumerate.
                entry["description"] += f" ({spec['enum_hint']})"
            if spec.get("example"):
                entry["examples"] = [spec["example"]]
        props[name] = entry

    return json.dumps({"type": "object", "properties": props}, indent=2)


CONDITIONS = ["bare", "described", "rich"]
