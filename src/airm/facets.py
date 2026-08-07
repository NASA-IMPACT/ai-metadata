"""The canonical content payload — the thing every format must carry identically.

The whole point of Experiment 1 is that a format comparison is meaningless unless
content is held constant. A full nested UMM tree versus a short prose summary
differ in *which facts are present*, not just in shape, so any measured
difference between them is uninterpretable.

So there is exactly one content source: :func:`facets`, which projects a CMR
UMM-JSON record onto a fixed set of fields. Every renderer in
:mod:`airm.formats` consumes that dict and nothing else. Nothing may read the
raw record directly.

One facet, one UMM field
------------------------
Thirty-two flat facets, each standing for a field CMR actually records. Two
rules follow from that and are worth stating because the previous payload broke
both:

* **Nothing is aggregated.** ``platform`` and ``instrument`` are separate
  facets, not instruments nested inside platforms; the four quality fields are
  four facets, not one ``quality`` dict. A grouping is a modelling choice, and a
  modelling choice baked into the content is exactly the kind of thing that
  makes a format comparison mean something other than it claims.
* **Nothing is rendered.** Where UMM records structure, the facet keeps it:
  ``spatial_resolution`` is ``{x, y, unit}``, not the string ``"30 x 30
  Meters"``; ``data_volume`` is ``{size, unit}``. ``bbox`` was always the
  precedent -- four coordinate fields, not a composed string -- and composing
  text here would push a rendering decision into the payload every format then
  has to carry identically.

``cmr_link`` is the single deliberate exception: it is constructed from the
concept-id because CMR does not store it, and it is constructed identically for
every format.

Placeholders are absences
-------------------------
CMR fills required fields it has no value for with literal placeholder strings --
``"Not provided"``, ``"Not applicable"``, ``"NA"``. Two thirds of
``ProcessingLevel.Id`` values are ``"Not provided"``. These assert no fact, so
:func:`_clean` drops them and the facet is omitted like any other empty value.
Carrying them would put the same content-free string into every representation
and into the embedding of every record that has one.

Number canonicalisation
-----------------------
Numbers are canonicalised at extraction time (:func:`canonical_coord`) and
printed through one shared helper (:func:`num_str`), so a coordinate reads
identically in JSON, YAML, TOON, CSV and prose. Without this, Python's float
``repr`` would charge JSON, YAML and TOON three characters for the ``.0`` in
``-180.0`` while prose printed ``-180`` -- a pure serialisation artefact showing
up as a token-cost difference in an experiment whose entire subject is token
cost.

Extractors are recovered from the implementation deleted in ``fe00bb6``
(``git show fe00bb6^:airm/representations.py``), extended with the fields this
study needs.
"""

from __future__ import annotations

import re
from typing import Any

#: Facet keys in canonical order. Renderers that impose an ordering (JSON, YAML,
#: TOON, CSV) all use this one, so ordering never becomes a hidden variable.
#: Grouped for readability only -- every one of these is a flat top-level key.
FACET_KEYS: tuple[str, ...] = (
    # identity
    "concept_id",
    "cmr_link",
    "title",
    "short_name",
    "doi",
    # provenance
    "agency",
    "data_center",
    # science
    "topic",
    "summary",
    "science_keywords",
    "variables",
    # instrumentation
    "platform",
    "instrument",
    "techniques",
    "spectral_bands",
    # space
    "bbox",
    "coordinate_system",
    "spatial_resolution",
    # time
    "temporal",
    "temporal_resolution",
    # distribution
    "data_format",
    "data_volume",
    # quality
    "processing_level",
    "processing_level_description",
    "version",
    "collection_progress",
    # rights
    "access_constraints",
    "use_constraints",
    # links
    "urls",
    "urls_publication",
    "urls_repository",
    "urls_additional",
)

#: Values CMR writes when it has nothing to write. They assert no fact, so they
#: are treated as absent rather than carried into every representation.
PLACEHOLDERS: frozenset[str] = frozenset(
    {
        "not provided",
        "not applicable",
        "notprovided",
        "not available",
        "not specified",
        "none",
        "n/a",
        "na",
        "unknown",
        "tbd",
        "-",
    }
)

#: Where a CMR collection can always be resolved, given its concept-id.
CMR_CONCEPT_URL = "https://cmr.earthdata.nasa.gov/search/concepts/"


def _umm(record: dict) -> dict:
    """Accept either a full ``{"meta", "umm"}`` item or a bare UMM payload."""
    return record.get("umm", record)


def _meta(record: dict) -> dict:
    meta = record.get("meta")
    return meta if isinstance(meta, dict) else {}


def _clean(value: Any) -> str | None:
    """A usable string, or ``None`` for empty and placeholder values."""
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text or text.lower() in PLACEHOLDERS:
        return None
    return text


def _dedup(values: list[str]) -> list[str]:
    """Preserve order, drop repeats -- lists are unioned from several UMM paths."""
    seen: set[str] = set()
    return [v for v in values if not (v in seen or seen.add(v))]


def _dedup_dicts(values: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for v in values:
        key = tuple(sorted(v.items()))
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def _list(node: Any, key: str) -> list:
    value = node.get(key) if isinstance(node, dict) else None
    return value if isinstance(value, list) else []


def num_str(value: float | int | bool) -> str:
    """Canonical text for a number, shared by every renderer.

    ``True``/``False`` are handled before numbers because ``bool`` is a subclass
    of ``int`` in Python and would otherwise print as ``1``/``0``.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return repr(float(value))


def canonical_coord(value: Any) -> float | int | None:
    """Canonicalise a numeric field value, or ``None`` if unusable.

    An integral value is stored as an ``int``. Python's float ``repr`` would
    otherwise write ``-180.0`` where ``-180`` states the identical fact, which
    charges JSON, YAML and TOON three characters of pure serialisation artefact.
    Used for every number that reaches the payload -- coordinates, resolutions,
    file sizes -- so the canonicalisation is uniform.

    It also used to sidestep a ``python-toon`` quirk that encoded ``0.0`` as
    ``0`` (and only ``0.0`` -- ``-180.0`` survived), which the parity validator
    flagged. The official ``toon_format`` encoder does not have that bug, so
    canonicalisation now rests solely on the token-artefact argument above --
    which is reason enough on its own.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


def _number_or_text(value: Any) -> Any:
    """A canonical number where the field holds one, else a usable string."""
    number = canonical_coord(value)
    return number if number is not None else _clean(value)


# --------------------------------------------------------------------------- #
# Identity.
# --------------------------------------------------------------------------- #


def extract_concept_id(record: dict) -> str | None:
    return _clean(_meta(record).get("concept-id"))


def extract_cmr_link(record: dict) -> str | None:
    """The collection's canonical CMR URL.

    The one constructed facet. CMR does not store it -- only 140 of 2,590
    records carry a cmr.earthdata URL in ``RelatedUrls`` -- but every collection
    resolves at this address by concept-id, so it states a true fact about the
    record and is derived identically for every format.
    """
    cid = extract_concept_id(record)
    return f"{CMR_CONCEPT_URL}{cid}" if cid else None


def extract_title(record: dict) -> str:
    u = _umm(record)
    return _clean(u.get("EntryTitle")) or _clean(u.get("ShortName")) or "(untitled)"


def extract_short_name(record: dict) -> str | None:
    return _clean(_umm(record).get("ShortName"))


def extract_doi(record: dict) -> str | None:
    return _clean((_umm(record).get("DOI") or {}).get("DOI"))


# --------------------------------------------------------------------------- #
# Provenance.
# --------------------------------------------------------------------------- #

#: Roles that make a data centre the dataset's repository rather than merely a
#: contributor.
REPOSITORY_ROLES = ("ARCHIVER", "DISTRIBUTOR")


def extract_agency(record: dict) -> str | None:
    """``DataCenters[].LongName`` -- the agency the data centre belongs to.

    Falls back to the CMR provider id, which is the only agency-level identifier
    a record without a long name carries.
    """
    for dc in _list(_umm(record), "DataCenters"):
        name = _clean(dc.get("LongName"))
        if name:
            return name
    return _clean(_meta(record).get("provider-id"))


def extract_data_center(record: dict) -> str | None:
    """The repository: ``DataCenters[].ShortName`` for an archiving centre.

    Prefers a centre carrying an archiving role over whichever happens to be
    first, since ``DataCenters`` also lists originators and processors.
    """
    centres = _list(_umm(record), "DataCenters")
    for dc in centres:
        roles = [str(r).upper() for r in _list(dc, "Roles")]
        if any(role in roles for role in REPOSITORY_ROLES):
            name = _clean(dc.get("ShortName"))
            if name:
                return name
    for dc in centres:
        name = _clean(dc.get("ShortName"))
        if name:
            return name
    return _clean(_meta(record).get("provider-id"))


# --------------------------------------------------------------------------- #
# Science.
# --------------------------------------------------------------------------- #


def extract_summary(record: dict) -> str:
    return (_umm(record).get("Abstract") or "").strip()


#: How GCMD keyword paths are written, both by NASA's own documentation and by
#: the prose renderer.
GCMD_SEPARATOR = " > "


def extract_science_keywords(record: dict) -> list[str]:
    """The GCMD keyword hierarchy as ``"EARTH SCIENCE > LAND SURFACE > ..."`` paths.

    Deliberately a list of *strings* rather than a list of ``{category, topic,
    term}`` objects. A structured form has to survive JSON-LD's flattening into
    a ``DefinedTerm.termCode``, and a record that omits a middle level (category
    and term but no topic) then cannot be reassembled unambiguously -- so the
    parity check would fail on a rendering artefact rather than on a real loss of
    content. The joined path is also GCMD's own canonical notation, so every
    format carries the identical string and inversion is exact.
    """
    out: list[str] = []
    for sk in _list(_umm(record), "ScienceKeywords"):
        levels = [
            sk.get("Category"),
            sk.get("Topic"),
            sk.get("Term"),
            sk.get("VariableLevel1"),
            sk.get("VariableLevel2"),
            sk.get("VariableLevel3"),
        ]
        path = GCMD_SEPARATOR.join(level for level in levels if level)
        if path and path not in out:
            out.append(path)
    return out


def extract_topic(record: dict) -> str | None:
    """``ScienceKeywords[].Topic`` — also the corpus stratification key."""
    for sk in _list(_umm(record), "ScienceKeywords"):
        topic = _clean(sk.get("Topic"))
        if topic:
            return topic
    return None


def extract_variables(record: dict) -> list[str]:
    """The most specific science-keyword term per entry, deduplicated."""
    variables: list[str] = []
    for sk in _list(_umm(record), "ScienceKeywords"):
        for level in ("VariableLevel3", "VariableLevel2", "VariableLevel1", "Term", "Topic"):
            value = _clean(sk.get(level))
            if value:
                variables.append(value.title())
                break
    return _dedup(variables)


# --------------------------------------------------------------------------- #
# Instrumentation.
# --------------------------------------------------------------------------- #


def _instruments(record: dict) -> list[dict]:
    return [
        inst
        for p in _list(_umm(record), "Platforms")
        for inst in _list(p, "Instruments")
        if isinstance(inst, dict)
    ]


def extract_platform(record: dict) -> list[str]:
    """``Platforms[].ShortName``.

    A flat list of the field's values. Instruments are their own facet rather
    than nested here: the nesting was a modelling choice, and this payload
    carries fields, not a model of them.
    """
    return _dedup(
        [n for p in _list(_umm(record), "Platforms") if (n := _clean(p.get("ShortName")))]
    )


def extract_instrument(record: dict) -> list[str]:
    """``Platforms[].Instruments[].ShortName``, flattened across platforms."""
    return _dedup([n for i in _instruments(record) if (n := _clean(i.get("ShortName")))])


def extract_techniques(record: dict) -> list[str]:
    """``Instruments[].Technique`` — the measurement method.

    The sparsest facet in the payload (2.4% of records). Kept because where it
    exists it is the only statement of *how* a measurement was made, which no
    other field implies.
    """
    techniques: list[str] = []
    for inst in _instruments(record):
        if value := _clean(inst.get("Technique")):
            techniques.append(value)
        for part in _list(inst, "ComposedOf"):
            if isinstance(part, dict) and (value := _clean(part.get("Technique"))):
                techniques.append(value)
    return _dedup(techniques)


#: Vocabulary that marks a field value as naming a spectral band or wavelength.
SPECTRAL_RE = re.compile(
    r"wavelength|spectral|infrared|microwave|ultraviolet|visible|radiance|"
    r"band|frequency|channel|gigahertz|nanometer|micron",
    re.I,
)


def extract_spectral_bands(record: dict) -> list[str]:
    """Operating wavelength / spectral band.

    UMM-C has no field for this, so the values are taken verbatim from the four
    places the information actually lives: instrument sub-components
    (``ComposedOf[].ShortName``, which is how channel sets are recorded),
    ``OperationalModes``, ``Characteristics[].Name`` where the name is spectral,
    and science-keyword levels naming a band. Nothing is composed -- each entry
    is a field value as CMR wrote it. This resolves for 13.7% of records, and it
    is a scrape across fields rather than one field being read.
    """
    bands: list[str] = []
    umm = _umm(record)

    for inst in _instruments(record):
        for part in _list(inst, "ComposedOf"):
            if isinstance(part, dict) and (name := _clean(part.get("ShortName"))):
                bands.append(name)
        for mode in _list(inst, "OperationalModes"):
            if value := _clean(mode):
                bands.append(value)

    holders = _instruments(record) + [p for p in _list(umm, "Platforms") if isinstance(p, dict)]
    for holder in holders:
        for ch in _list(holder, "Characteristics"):
            if isinstance(ch, dict):
                name = _clean(ch.get("Name"))
                if name and SPECTRAL_RE.search(name):
                    bands.append(name)

    for sk in _list(umm, "ScienceKeywords"):
        for level in ("Term", "VariableLevel1", "VariableLevel2", "VariableLevel3"):
            value = _clean(sk.get(level))
            if value and SPECTRAL_RE.search(value):
                bands.append(value)

    for aa in _list(umm, "AdditionalAttributes"):
        name = _clean(aa.get("Name"))
        if name and SPECTRAL_RE.search(name):
            bands.append(name)

    return _dedup(bands)


# --------------------------------------------------------------------------- #
# Space.
# --------------------------------------------------------------------------- #


def _horizontal_domain(record: dict) -> dict:
    spatial = _umm(record).get("SpatialExtent")
    if not isinstance(spatial, dict):
        return {}
    domain = spatial.get("HorizontalSpatialDomain")
    return domain if isinstance(domain, dict) else {}


def extract_bbox(record: dict) -> dict | None:
    """``{west, south, east, north}`` of the first bounding rectangle."""
    geom = _horizontal_domain(record).get("Geometry") or {}
    rects = geom.get("BoundingRectangles") or []
    if not rects:
        return None
    r = rects[0]
    box = {
        "west": canonical_coord(r.get("WestBoundingCoordinate")),
        "south": canonical_coord(r.get("SouthBoundingCoordinate")),
        "east": canonical_coord(r.get("EastBoundingCoordinate")),
        "north": canonical_coord(r.get("NorthBoundingCoordinate")),
    }
    # A partial box is not a box; drop it rather than emit a half-fact that some
    # formats would render and others would not.
    return box if all(v is not None for v in box.values()) else None


def extract_coordinate_system(record: dict) -> str | None:
    """``Geometry.CoordinateSystem``, else ``GranuleSpatialRepresentation``.

    The fallback is a real field, not a guess: where a record has no geometry,
    ``GranuleSpatialRepresentation`` is its only statement of spatial framing --
    including ``NO_SPATIAL``, which is a fact about the dataset rather than a
    missing value.
    """
    geom = _horizontal_domain(record).get("Geometry") or {}
    if direct := _clean(geom.get("CoordinateSystem")):
        return direct
    spatial = _umm(record).get("SpatialExtent")
    return _clean(spatial.get("GranuleSpatialRepresentation")) if isinstance(spatial, dict) else None


def extract_spatial_resolution(record: dict) -> list[dict]:
    """``[{x, y, unit}]`` from ``HorizontalDataResolution``.

    Structured rather than composed into ``"30 x 30 Meters"``, for the same
    reason ``bbox`` is four coordinates and not a string: the dimensions and the
    unit are three separate fields, and joining them is a rendering decision
    every format would then be forced to carry identically.
    """
    resolution = (_horizontal_domain(record).get("ResolutionAndCoordinateSystem") or {}).get(
        "HorizontalDataResolution"
    )
    if not isinstance(resolution, dict):
        return []

    out: list[dict] = []
    for key in ("GriddedResolutions", "NonGriddedResolutions", "GenericResolutions"):
        for entry in _list(resolution, key):
            if not isinstance(entry, dict):
                continue
            item = {
                "x": _number_or_text(entry.get("XDimension")),
                "y": _number_or_text(entry.get("YDimension")),
                "unit": _clean(entry.get("Unit")),
            }
            item = {k: v for k, v in item.items() if v is not None}
            if item:
                out.append(item)
    if varies := _clean(resolution.get("VariesResolution")):
        out.append({"unit": varies})
    return _dedup_dicts(out)


# --------------------------------------------------------------------------- #
# Time.
# --------------------------------------------------------------------------- #


def extract_temporal(record: dict) -> dict | None:
    """``{begin, end}`` ISO strings; ``end`` is ``"present"`` when ongoing."""
    for te in _list(_umm(record), "TemporalExtents"):
        for rng in _list(te, "RangeDateTimes"):
            begin = rng.get("BeginningDateTime")
            if not begin:
                continue
            end = rng.get("EndingDateTime") or (
                "present" if te.get("EndsAtPresentFlag") else None
            )
            return {"begin": begin, "end": end} if end else {"begin": begin}
    return None


def extract_temporal_resolution(record: dict) -> list[dict]:
    """``[{value, unit}]`` from ``TemporalExtents[].TemporalResolution``."""
    out: list[dict] = []
    for te in _list(_umm(record), "TemporalExtents"):
        tr = te.get("TemporalResolution")
        if not isinstance(tr, dict):
            continue
        item = {
            "value": _number_or_text(tr.get("Value")),
            "unit": _clean(tr.get("Unit")),
        }
        item = {k: v for k, v in item.items() if v is not None}
        if item:
            out.append(item)
    return _dedup_dicts(out)


# --------------------------------------------------------------------------- #
# Distribution.
# --------------------------------------------------------------------------- #


def _archive_info(record: dict) -> dict:
    info = _umm(record).get("ArchiveAndDistributionInformation")
    return info if isinstance(info, dict) else {}


def extract_data_format(record: dict) -> list[str]:
    """File formats, unioned across distribution, archive and access-URL records."""
    info = _archive_info(record)
    formats: list[str] = []
    for key in ("FileDistributionInformation", "FileArchiveInformation"):
        for entry in _list(info, key):
            if isinstance(entry, dict) and (value := _clean(entry.get("Format"))):
                formats.append(value)
    for url in _list(_umm(record), "RelatedUrls"):
        get_data = url.get("GetData") if isinstance(url, dict) else None
        if isinstance(get_data, dict) and (value := _clean(get_data.get("Format"))):
            formats.append(value)
    return _dedup(formats)


def extract_data_volume(record: dict) -> dict | None:
    """``{size, unit, basis}`` from the archive/distribution size fields.

    ``basis`` distinguishes a collection total from a per-file average, which
    are different facts; emitting whichever exists without saying which would
    make the number uninterpretable.
    """
    info = _archive_info(record)
    entries = [
        e
        for key in ("FileDistributionInformation", "FileArchiveInformation")
        for e in _list(info, key)
        if isinstance(e, dict)
    ]
    for size_key, unit_key, basis in (
        ("TotalCollectionFileSize", "TotalCollectionFileSizeUnit", "total"),
        ("AverageFileSize", "AverageFileSizeUnit", "average per file"),
    ):
        for entry in entries:
            size = _number_or_text(entry.get(size_key))
            if size is None:
                continue
            item = {"size": size, "unit": _clean(entry.get(unit_key)), "basis": basis}
            return {k: v for k, v in item.items() if v is not None}

    for url in _list(_umm(record), "RelatedUrls"):
        get_data = url.get("GetData") if isinstance(url, dict) else None
        if isinstance(get_data, dict):
            size = _number_or_text(get_data.get("Size"))
            if size is not None:
                item = {"size": size, "unit": _clean(get_data.get("Unit")), "basis": "total"}
                return {k: v for k, v in item.items() if v is not None}
    return None


# --------------------------------------------------------------------------- #
# Quality. Four separate fields, not one grouped facet.
# --------------------------------------------------------------------------- #


def extract_processing_level(record: dict) -> str | None:
    return _clean((_umm(record).get("ProcessingLevel") or {}).get("Id"))


def extract_processing_level_description(record: dict) -> str | None:
    return _clean(
        (_umm(record).get("ProcessingLevel") or {}).get("ProcessingLevelDescription")
    )


def extract_version(record: dict) -> str | None:
    return _clean(_umm(record).get("Version"))


def extract_collection_progress(record: dict) -> str | None:
    return _clean(_umm(record).get("CollectionProgress"))


# --------------------------------------------------------------------------- #
# Rights.
# --------------------------------------------------------------------------- #


def extract_access_constraints(record: dict) -> str | None:
    node = _umm(record).get("AccessConstraints")
    if not isinstance(node, dict):
        return _clean(node)
    if description := _clean(node.get("Description")):
        return description
    value = canonical_coord(node.get("Value"))
    return num_str(value) if value is not None else None


def extract_use_constraints(record: dict) -> str | None:
    """Licence or usage statement, from whichever sub-field carries it."""
    node = _umm(record).get("UseConstraints")
    if not isinstance(node, dict):
        return _clean(node)
    for key in ("Description", "LicenseText", "LicenseURL", "FreeAndOpenData"):
        value = node.get(key)
        if isinstance(value, dict):
            value = value.get("Linkage") or value.get("URL")
        if isinstance(value, bool):
            value = "free and open" if value else None
        if cleaned := _clean(value):
            return cleaned
    return None


# --------------------------------------------------------------------------- #
# Links.
# --------------------------------------------------------------------------- #

_PUBLICATION_CONTENT_TYPE = "PublicationURL"
_REPOSITORY_CONTENT_TYPE = "DistributionURL"
_REPOSITORY_TYPE = "DATA SET LANDING PAGE"


def _url_kind(entry: dict) -> str:
    """``publication`` | ``repository`` | ``additional``.

    The three kinds partition ``RelatedUrls`` exactly -- verified across all 62
    ``(URLContentType, Type, Subtype)`` combinations in the cache, with no
    residue -- so no URL is counted twice and none is lost.
    """
    if entry.get("URLContentType") == _PUBLICATION_CONTENT_TYPE:
        return "publication"
    if entry.get("URLContentType") == _REPOSITORY_CONTENT_TYPE or entry.get("Type") == _REPOSITORY_TYPE:
        return "repository"
    return "additional"


def extract_urls(record: dict) -> dict[str, list[str]]:
    """``RelatedUrls[].URL`` -- the whole list and each of the three kinds.

    ``all`` is the field as CMR records it; the three kinds are that same field
    partitioned by ``URLContentType``/``Type``. Every URL therefore appears
    twice in the payload, once in ``urls`` and once in its kind. That is
    redundancy the caller asked for, and it is identical in every format, so it
    costs tokens without confounding the comparison.

    Where a record has no publication URL, the citation's online resource is the
    same fact recorded elsewhere, so it stands in.
    """
    kinds: dict[str, list[str]] = {
        "all": [],
        "publication": [],
        "repository": [],
        "additional": [],
    }
    for entry in _list(_umm(record), "RelatedUrls"):
        if not isinstance(entry, dict):
            continue
        if url := _clean(entry.get("URL")):
            kinds["all"].append(url)
            kinds[_url_kind(entry)].append(url)

    if not kinds["publication"]:
        for citation in _list(_umm(record), "CollectionCitations"):
            resource = citation.get("OnlineResource") if isinstance(citation, dict) else None
            if isinstance(resource, dict) and (url := _clean(resource.get("Linkage"))):
                kinds["publication"].append(url)

    return {kind: _dedup(urls) for kind, urls in kinds.items()}


# --------------------------------------------------------------------------- #
# The canonical payload.
# --------------------------------------------------------------------------- #


def facets(record: dict) -> dict:
    """Project a CMR record onto the canonical facet payload.

    Empty facets are omitted entirely rather than emitted as ``null``: a format
    that renders ``"bbox": null`` and one that omits the key carry the same
    information but very different token counts, so absent facts are absent
    everywhere. Placeholder strings are empty by the same argument -- see
    :data:`PLACEHOLDERS`.
    """
    urls = extract_urls(record)
    payload: dict[str, Any] = {
        "concept_id": extract_concept_id(record),
        "cmr_link": extract_cmr_link(record),
        "title": extract_title(record),
        "short_name": extract_short_name(record),
        "doi": extract_doi(record),
        "agency": extract_agency(record),
        "data_center": extract_data_center(record),
        "topic": extract_topic(record),
        "summary": extract_summary(record),
        "science_keywords": extract_science_keywords(record),
        "variables": extract_variables(record),
        "platform": extract_platform(record),
        "instrument": extract_instrument(record),
        "techniques": extract_techniques(record),
        "spectral_bands": extract_spectral_bands(record),
        "bbox": extract_bbox(record),
        "coordinate_system": extract_coordinate_system(record),
        "spatial_resolution": extract_spatial_resolution(record),
        "temporal": extract_temporal(record),
        "temporal_resolution": extract_temporal_resolution(record),
        "data_format": extract_data_format(record),
        "data_volume": extract_data_volume(record),
        "processing_level": extract_processing_level(record),
        "processing_level_description": extract_processing_level_description(record),
        "version": extract_version(record),
        "collection_progress": extract_collection_progress(record),
        "access_constraints": extract_access_constraints(record),
        "use_constraints": extract_use_constraints(record),
        "urls": urls["all"],
        "urls_publication": urls["publication"],
        "urls_repository": urls["repository"],
        "urls_additional": urls["additional"],
    }
    return {k: payload[k] for k in FACET_KEYS if payload.get(k)}


# --------------------------------------------------------------------------- #
# Fact sets — the parity currency.
# --------------------------------------------------------------------------- #


def flatten(payload: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten a facet payload to ``(path, scalar)`` pairs in document order.

    Paths look like ``spatial_resolution[0].unit`` or ``bbox.west``. Used by the
    CSV renderer and by :func:`fact_set`.
    """
    out: list[tuple[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            out.extend(flatten(value, f"{prefix}.{key}" if prefix else key))
    elif isinstance(payload, list):
        for i, value in enumerate(payload):
            out.extend(flatten(value, f"{prefix}[{i}]"))
    else:
        out.append((prefix, payload))
    return out


def scalar_str(value: Any) -> str:
    """Canonical text for one scalar, used on both sides of every comparison."""
    if isinstance(value, (bool, int, float)):
        return num_str(value)
    return str(value).strip()


def fact_set(payload: dict) -> set[tuple[str, str]]:
    """The ``(path, canonical value)`` pairs a payload asserts.

    Parity is defined as equality of these sets. Comparing canonical *strings*
    rather than raw Python values is deliberate: CSV cannot distinguish the
    string ``"3"`` from the integer ``3``, but it carries the same fact, and the
    experiment is about information content, not Python types.
    """
    return {(path, scalar_str(value)) for path, value in flatten(payload)}


def fact_values(payload: dict) -> list[str]:
    """Just the canonical values, for the prose value-coverage check."""
    return [scalar_str(value) for _, value in flatten(payload)]
