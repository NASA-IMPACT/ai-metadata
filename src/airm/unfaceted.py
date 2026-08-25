"""The unfaceted control — the same six formats over the *whole* raw UMM record.

:mod:`airm.facets` projects a CMR record onto 32 fields and every renderer reads
that projection. This module renders the record CMR actually serves: all 45
top-level UMM fields, 398 distinct paths, contacts and addresses and metadata
dates included. Same six formats, same parity discipline, no projection.

Why it exists
-------------
Experiment 1 reports a ``json_umm`` reference — what the status quo costs — but
only in JSON, so it answers "how much does faceting save?" and not "does the
format ranking survive without faceting?". Those are different questions, and
the second one is the one a reader who distrusts the projection will ask. This
cache answers it by holding content constant at the *other* extreme.

It is a control, not a competitor. Nothing in the study's main line reads it:
the corpus, the index and both experiments go through
:mod:`airm.format_cache`. Faceted and unfaceted renderings are **not**
comparable to each other record-for-record, because they carry different content
— which is the whole reason the faceted payload exists.

What a schema buys, and what it does not
----------------------------------------
Four of the six formats are schema-agnostic: JSON, YAML, TOON and CSV serialise
an arbitrary tree and read it back. The other two need to know what a field
*means*, and UMM-C 1.18.5 tells them — so both are written against the schema
rather than against the tree:

* **Metadata-as-Text is real prose** (:func:`render_mat`). An earlier version of
  this module emitted one ``path is value`` clause per leaf and called it ``mat``,
  which was simply wrong: that is the old study's ``dot_breadcrumb``
  representation, preserved here as :func:`render_breadcrumb`. The mislabel made
  Experiment 1 report a mechanical path dump as prose and cost it a factor of
  two — 4,911 tokens against 2,101 on the same record. UMM is a *published*
  schema, so prose over the whole record is a matter of writing one template per
  field, not an impossibility.
* **JSON-LD genuinely cannot.** schema.org has properties for five UMM fields and
  nothing for the other forty. The rest can only travel as ``PropertyValue``
  entries keyed by path — legal JSON-LD, semantically inert, and the ``@context``
  buys nothing when every term is a literal path string. That is a limit of the
  *vocabulary*, not of the tree, which is why prose escapes it and JSON-LD does
  not.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import cmr, formats
from .config import CMR_CACHE_DIR, DATA_DIR, FORMATS
from .facets import GCMD_SEPARATOR, fact_set, fact_values, flatten, scalar_str

#: Sibling of ``data/format_cache``, same layout, different payload.
UNFACETED_CACHE_DIR = DATA_DIR / "format_cache_unfaceted"

EXTENSIONS: dict[str, str] = {
    "json": ".json",
    "csv": ".csv",
    "yaml": ".yaml",
    "toon": ".toon",
    "jsonld": ".jsonld",
    "mat": ".txt",
}

MANIFEST_NAME = "manifest.json"


class StaleCacheError(RuntimeError):
    """Raised when the cache was written by different rendering code."""


# --------------------------------------------------------------------------- #
# Payload.
# --------------------------------------------------------------------------- #


def payload(record: dict) -> dict:
    """The unfaceted payload: the record's whole ``umm`` object, untouched.

    The analogue of :func:`airm.facets.facets`, and deliberately trivial. No
    projection, no placeholder filtering, no canonicalisation — the point of the
    control is that nothing has been decided on the reader's behalf.

    ``meta`` is excluded: it is CMR's record-keeping (revision id, ingest dates)
    rather than metadata about the dataset, and including it would put the
    concept-id in some formats and not others.
    """
    return record.get("umm", record)


# --------------------------------------------------------------------------- #
# JSON-LD over an arbitrary tree.
# --------------------------------------------------------------------------- #

#: The five UMM fields schema.org actually models. Everything else in the record
#: has no schema.org property and can only travel as a keyed literal.
JSONLD_MAPPED: tuple[tuple[str, str], ...] = (
    ("EntryTitle", "name"),
    ("Abstract", "description"),
    ("ShortName", "alternateName"),
    ("Version", "version"),
    ("DOI.DOI", "identifier"),
)


def render_jsonld(p: dict) -> str:
    """schema.org ``Dataset`` over a full UMM record.

    Five fields get real properties; every other leaf becomes a
    ``PropertyValue`` keyed by its flattened path. That is what JSON-LD without
    a schema degrades to, and the ratio — 5 modelled against ~400 paths — is the
    measurement worth taking.
    """
    doc: dict[str, Any] = {"@context": formats.JSONLD_CONTEXT, "@type": "Dataset"}
    leaves = flatten(p)
    mapped = dict(JSONLD_MAPPED)

    for path, value in leaves:
        if path in mapped:
            doc[mapped[path]] = value

    doc["additionalProperty"] = [
        {"@type": "PropertyValue", "name": path, "value": value}
        for path, value in leaves
        if path not in mapped
    ]
    return json.dumps(doc, ensure_ascii=False, separators=(",", ":"))


def parse_jsonld(text: str) -> dict:
    """Invert :func:`render_jsonld` by reassembling paths into a tree."""
    doc = json.loads(text)
    root: dict = {}
    for umm_path, schema_key in JSONLD_MAPPED:
        if schema_key in doc:
            formats._assign(root, formats._path_tokens(umm_path), doc[schema_key])
    for prop in doc.get("additionalProperty") or []:
        formats._assign(root, formats._path_tokens(prop["name"]), prop["value"])
    return root


# --------------------------------------------------------------------------- #
# Prose over an arbitrary tree.
# --------------------------------------------------------------------------- #

#: How a flattened path is spoken. ``SpatialExtent.HorizontalSpatialDomain`` ->
#: ``SpatialExtent > HorizontalSpatialDomain``, matching GCMD's own notation for
#: hierarchy so the breadcrumb reads consistently with the keyword paths.
PATH_SEPARATOR = " > "


def _speak_path(path: str) -> str:
    return path.replace(".", PATH_SEPARATOR)


def render_breadcrumb(p: dict) -> str:
    """One ``path is value`` clause per leaf.

    **This is not prose.** It is the old study's ``dot_breadcrumb``
    representation (``git show 92553ba:airm/representations.py:245``) with
    different separators, and it is kept under its real name because it was
    briefly mislabelled ``mat`` -- which made Experiment 1 report a mechanical
    path dump as Metadata-as-Text. It is not one of the six formats; it exists so
    that correction stays auditable and so the two can be compared directly.
    """
    return " ".join(
        f"{_speak_path(path)} is {scalar_str(value)}." for path, value in flatten(p)
    )


# --------------------------------------------------------------------------- #
# Prose. One writer per UMM field that is worth a sentence.
# --------------------------------------------------------------------------- #


def _s(value: Any) -> str:
    return scalar_str(value)


def _list(node: Any, key: str) -> list:
    """The list at ``key``, or empty. UMM omits keys freely and varies types."""
    value = node.get(key) if isinstance(node, dict) else None
    return value if isinstance(value, list) else []


def _vals(node: Any, *keys: str) -> list[str]:
    """Non-empty scalar values at ``keys``, in order, canonicalised."""
    if not isinstance(node, dict):
        return []
    return [_s(node[k]) for k in keys if node.get(k) not in (None, "", [], {})]


def _label(node: Any, *keys: str) -> str:
    """A name built from ``keys``, without repeating an identical value.

    Many records set ``LongName`` equal to ``ShortName``, which would otherwise
    read "Sentinel-1A Sentinel-1A". Deduplication loses no fact: parity compares
    the *set* of values, and the value is still present once.
    """
    seen: list[str] = []
    for value in _vals(node, *keys):
        if value not in seen:
            seen.append(value)
    return " ".join(seen)


def _person(person: dict) -> str:
    name = " ".join(_vals(person, "FirstName", "MiddleName", "LastName"))
    roles = ", ".join(_s(r) for r in _list(person, "Roles"))
    detail = _contact_detail(person.get("ContactInformation"))
    out = name or roles or "unnamed contact"
    if roles and name:
        out += f" ({roles})"
    if detail:
        out += f" -- {detail}"
    return out


def _contact_detail(info: Any) -> str:
    """Mechanisms, addresses and URLs of one ``ContactInformation`` block."""
    if not isinstance(info, dict):
        return ""
    bits: list[str] = []
    for m in _list(info, "ContactMechanisms"):
        if isinstance(m, dict):
            pair = _vals(m, "Type", "Value")
            if pair:
                bits.append(" ".join(pair))
    for a in _list(info, "Addresses"):
        if isinstance(a, dict):
            street = [_s(x) for x in _list(a, "StreetAddresses")]
            rest = _vals(a, "City", "StateProvince", "PostalCode", "Country")
            if street or rest:
                bits.append(", ".join(street + rest))
    for u in _list(info, "RelatedUrls"):
        if isinstance(u, dict):
            bits.append(" ".join(
                _vals(u, "URLContentType", "Type", "Subtype", "URL", "Description")
            ).strip())
    for key in ("ServiceHours", "ContactInstruction"):
        if info.get(key):
            bits.append(_s(info[key]))
    return "; ".join(bits)


def _identity(u: dict, parts: list[str]) -> None:
    title = _s(u.get("EntryTitle") or u.get("ShortName") or "(untitled)")
    opening = title
    tags = _vals(u, "ShortName", "Version")
    if tags:
        opening += f" (short name {tags[0]}"
        opening += f", version {tags[1]})" if len(tags) > 1 else ")"
    opening += " is a NASA Earth-science dataset"

    # Merge centres by name and union their roles. CMR lists the same centre
    # once per role, so ATL08 would otherwise read "NASA NSIDC DAAC (ARCHIVER);
    # NASA NSIDC DAAC (DISTRIBUTOR)" -- the name repeated for no added fact.
    centres: dict[str, list[str]] = {}
    for dc in _list(u, "DataCenters"):
        if not isinstance(dc, dict):
            continue
        name = _label(dc, "ShortName", "LongName")
        if not name:
            continue
        roles = centres.setdefault(name, [])
        for role in _list(dc, "Roles"):
            if (text := _s(role)) not in roles:
                roles.append(text)
    if centres:
        opening += " managed by " + "; ".join(
            f"{name} ({', '.join(roles)})" if roles else name
            for name, roles in centres.items()
        )
    parts.append(opening + ".")

    doi = u.get("DOI")
    if isinstance(doi, dict):
        if doi.get("DOI"):
            authority = _s(doi["Authority"]) if doi.get("Authority") else ""
            parts.append(
                f"Its DOI is {_s(doi['DOI'])}"
                + (f", issued by {authority}" if authority else "")
                + "."
            )
        elif doi.get("MissingReason"):
            reason = ", ".join(_vals(doi, "MissingReason", "Explanation"))
            parts.append(f"It has no DOI ({reason.rstrip('.')}).")

    for cit in _list(u, "CollectionCitations"):
        if not isinstance(cit, dict):
            continue
        bits = []
        for key, label in (
            ("Title", ""), ("Creator", "by {}"), ("SeriesName", "series {}"),
            ("Version", "version {}"), ("ReleaseDate", "released {}"),
            ("Publisher", "published by {}"), ("DataPresentationForm", "as {}"),
            ("OtherCitationDetails", "{}"),
        ):
            if cit.get(key) not in (None, "", [], {}):
                value = _s(cit[key])
                bits.append(label.format(value) if label else value)
        link = cit.get("OnlineResource")
        if isinstance(link, dict):
            bits += [_s(v) for _, v in flatten(link)]
        if bits:
            parts.append("Cite it as " + ", ".join(bits) + ".")

    for key, template in (
        ("DataLanguage", "The data language is {}."),
        ("MetadataLanguage", "The metadata language is {}."),
        ("Purpose", "Purpose: {}."),
        ("CollectionDataType", "Collection data type: {}."),
        ("DataMaturity", "Data maturity: {}."),
        ("Quality", "Quality notes: {}."),
        ("StandardProduct", "NASA standard product: {}."),
    ):
        if u.get(key) not in (None, "", [], {}):
            parts.append(template.format(_s(u[key])))


def _instrumentation(u: dict, parts: list[str]) -> None:
    described = []
    for plat in _list(u, "Platforms"):
        if not isinstance(plat, dict):
            continue
        name = _label(plat, "ShortName", "LongName")
        if not name:
            continue
        entry = name
        if plat.get("Type"):
            kind = _s(plat["Type"])
            article = "an" if kind[:1].upper() in "AEIOU" else "a"
            entry += f", {article} {kind} class platform"
        instruments = []
        for inst in _list(plat, "Instruments"):
            if not isinstance(inst, dict):
                continue
            label = _label(inst, "ShortName", "LongName")
            extra = _vals(inst, "Technique", "NumberOfInstruments")
            for part in _list(inst, "ComposedOf"):
                if isinstance(part, dict):
                    extra += _vals(part, "ShortName", "LongName", "Technique")
            for mode in _list(inst, "OperationalModes"):
                extra.append(_s(mode))
            for ch in _list(inst, "Characteristics"):
                if isinstance(ch, dict):
                    extra.append(" ".join(_vals(ch, "Name", "Value", "Unit", "Description")))
            if label:
                instruments.append(label + (f" [{', '.join(x for x in extra if x)}]" if extra else ""))
        if instruments:
            entry += ", carrying " + ", ".join(instruments)
        for ch in _list(plat, "Characteristics"):
            if isinstance(ch, dict):
                entry += " (" + " ".join(_vals(ch, "Name", "Value", "Unit", "Description")) + ")"
        described.append(entry)
    if described:
        parts.append("Data were acquired by " + "; ".join(described) + ".")

    projects = []
    for pr in _list(u, "Projects"):
        if isinstance(pr, dict):
            bits = _vals(pr, "ShortName", "LongName", "StartDate", "EndDate")
            bits += [_s(c) for c in _list(pr, "Campaigns")]
            if bits:
                projects.append(" ".join(bits))
    if projects:
        parts.append("Collected under " + "; ".join(projects) + ".")


def _keywords(u: dict, parts: list[str]) -> None:
    paths = []
    for sk in _list(u, "ScienceKeywords"):
        if isinstance(sk, dict):
            levels = _vals(sk, "Category", "Topic", "Term",
                           "VariableLevel1", "VariableLevel2", "VariableLevel3",
                           "DetailedVariable")
            if levels:
                paths.append(GCMD_SEPARATOR.join(levels))
    if paths:
        parts.append("Science keywords: " + "; ".join(paths) + ".")

    for key, label in (("ISOTopicCategories", "ISO topic categories"),
                       ("AncillaryKeywords", "Ancillary keywords"),
                       ("TemporalKeywords", "Temporal keywords")):
        values = [_s(v) for v in _list(u, key)]
        if values:
            parts.append(f"{label}: " + "; ".join(values) + ".")

    places = []
    for lk in _list(u, "LocationKeywords"):
        if isinstance(lk, dict):
            levels = _vals(lk, "Category", "Type", "Subregion1", "Subregion2",
                           "Subregion3", "DetailedLocation")
            if levels:
                places.append(GCMD_SEPARATOR.join(levels))
    if places:
        parts.append("Location keywords: " + "; ".join(places) + ".")

    names = []
    for dn in _list(u, "DirectoryNames"):
        if isinstance(dn, dict):
            if (name := _label(dn, "ShortName", "LongName")):
                names.append(name)
    if names:
        parts.append("Directory names: " + "; ".join(names) + ".")


def _space(u: dict, parts: list[str]) -> None:
    spatial = u.get("SpatialExtent")
    if not isinstance(spatial, dict):
        return
    hsd = spatial.get("HorizontalSpatialDomain")
    geom = hsd.get("Geometry") if isinstance(hsd, dict) else None

    if isinstance(geom, dict):
        for rect in _list(geom, "BoundingRectangles"):
            if isinstance(rect, dict):
                w, s, e, north = (
                    rect.get("WestBoundingCoordinate"), rect.get("SouthBoundingCoordinate"),
                    rect.get("EastBoundingCoordinate"), rect.get("NorthBoundingCoordinate"),
                )
                if None not in (w, s, e, north):
                    parts.append(
                        f"Spatial coverage spans {_s(w)}° to {_s(e)}° longitude and "
                        f"{_s(s)}° to {_s(north)}° latitude."
                    )
        for key, label in (("GPolygons", "Bounding polygon"), ("Points", "Point location"),
                           ("Lines", "Bounding line")):
            for shape in _list(geom, key):
                coords = [_s(v) for _, v in flatten(shape)]
                if coords:
                    parts.append(f"{label}: " + ", ".join(coords) + ".")
        if geom.get("CoordinateSystem"):
            parts.append(f"Coordinates are {_s(geom['CoordinateSystem'])}.")

    framing = []
    if isinstance(hsd, dict) and hsd.get("ZoneIdentifier"):
        framing.append(f"zone {_s(hsd['ZoneIdentifier'])}")
    for key in ("SpatialCoverageType", "GranuleSpatialRepresentation"):
        if spatial.get(key):
            framing.append(f"{key} {_s(spatial[key])}")
    if framing:
        parts.append("Spatial framing: " + ", ".join(framing) + ".")

    rcs = hsd.get("ResolutionAndCoordinateSystem") if isinstance(hsd, dict) else None
    if isinstance(rcs, dict):
        res = rcs.get("HorizontalDataResolution")
        if isinstance(res, dict):
            for key in ("GriddedResolutions", "NonGriddedResolutions", "GenericResolutions",
                        "GriddedRangeResolutions", "NonGriddedRangeResolutions"):
                for entry in _list(res, key):
                    bits = [_s(v) for _, v in flatten(entry)]
                    if bits:
                        parts.append("Horizontal resolution: " + " ".join(bits) + ".")
            if res.get("VariesResolution"):
                parts.append(f"Horizontal resolution {_s(res['VariesResolution'])}.")
            if res.get("PointResolution"):
                parts.append(f"Point resolution {_s(res['PointResolution'])}.")
        for key in ("Description", "GeodeticModel", "LocalCoordinateSystem"):
            node = rcs.get(key)
            if node:
                bits = [_s(v) for _, v in flatten(node)]
                parts.append(f"Coordinate system {key}: " + " ".join(bits) + ".")

    for vsd in _list(spatial, "VerticalSpatialDomains"):
        bits = [_s(v) for _, v in flatten(vsd)]
        if bits:
            parts.append("Vertical extent: " + " ".join(bits) + ".")

    for tile in _list(u, "TilingIdentificationSystems"):
        bits = [_s(v) for _, v in flatten(tile)]
        if bits:
            parts.append("Tiling system: " + " ".join(bits) + ".")


def _time(u: dict, parts: list[str]) -> None:
    for te in _list(u, "TemporalExtents"):
        if not isinstance(te, dict):
            continue
        for rng in _list(te, "RangeDateTimes"):
            if isinstance(rng, dict) and rng.get("BeginningDateTime"):
                end = _s(rng["EndingDateTime"]) if rng.get("EndingDateTime") else "present"
                parts.append(
                    f"Temporal coverage runs from {_s(rng['BeginningDateTime'])} to {end}."
                )
        singles = [_s(d) for d in _list(te, "SingleDateTimes")]
        if singles:
            parts.append("Observed on " + ", ".join(singles) + ".")
        for per in _list(te, "PeriodicDateTimes"):
            bits = [_s(v) for _, v in flatten(per)]
            if bits:
                parts.append("Periodic coverage: " + " ".join(bits) + ".")
        res = te.get("TemporalResolution")
        if isinstance(res, dict):
            bits = _vals(res, "Value", "Unit")
            if bits:
                parts.append("Temporal resolution is " + " ".join(bits) + ".")
        extras = _vals(te, "EndsAtPresentFlag", "TemporalRangeType", "PrecisionOfSeconds")
        if extras:
            parts.append("Temporal flags: " + ", ".join(extras) + ".")

    for pt in _list(u, "PaleoTemporalCoverages"):
        bits = [_s(v) for _, v in flatten(pt)]
        if bits:
            parts.append("Paleo temporal coverage: " + " ".join(bits) + ".")


def _quality_and_content(u: dict, parts: list[str]) -> None:
    level = u.get("ProcessingLevel")
    bits = []
    if isinstance(level, dict):
        if level.get("Id"):
            bits.append(f"processing level {_s(level['Id'])}")
        if level.get("ProcessingLevelDescription"):
            bits.append(_s(level["ProcessingLevelDescription"]))
    if u.get("Version"):
        bits.append(f"version {_s(u['Version'])}")
    if u.get("CollectionProgress"):
        bits.append(f"status {_s(u['CollectionProgress'])}")
    if bits:
        parts.append("Quality: " + ", ".join(bits) + ".")

    if u.get("VersionDescription"):
        parts.append(f"Changes in this version: {_s(u['VersionDescription'])}")
    if u.get("Abstract"):
        parts.append(_s(u["Abstract"]))

    for attr in _list(u, "AdditionalAttributes"):
        if isinstance(attr, dict):
            bits = _vals(attr, "Name", "DataType", "Value", "Description",
                         "Measurement Resolution", "ParameterUnitsOfMeasure")
            if bits:
                parts.append("Additional attribute " + ", ".join(bits) + ".")


def _distribution(u: dict, parts: list[str]) -> None:
    adi = u.get("ArchiveAndDistributionInformation")
    if isinstance(adi, dict):
        for key, label in (("FileDistributionInformation", "Distributed as"),
                           ("FileArchiveInformation", "Archived as")):
            for entry in _list(adi, key):
                bits = [_s(v) for _, v in flatten(entry)]
                if bits:
                    parts.append(f"{label} " + ", ".join(bits) + ".")

    ddi = u.get("DirectDistributionInformation")
    if isinstance(ddi, dict):
        bits = [_s(v) for _, v in flatten(ddi)]
        if bits:
            parts.append("Direct S3 access: " + ", ".join(bits) + ".")

    if u.get("FileNamingConvention"):
        bits = [_s(v) for _, v in flatten(u["FileNamingConvention"])]
        parts.append("File naming convention: " + " ".join(bits) + ".")

    for key, label in (("AccessConstraints", "Access constraints"),
                       ("UseConstraints", "Use constraints")):
        node = u.get(key)
        if node:
            bits = [_s(v) for _, v in flatten(node)]
            parts.append(f"{label}: " + " ".join(bits))

    for url in _list(u, "RelatedUrls"):
        if not isinstance(url, dict):
            continue
        label = " ".join(_vals(url, "URLContentType", "Type", "Subtype"))
        target = _s(url["URL"]) if url.get("URL") else ""
        detail = _s(url["Description"]) if url.get("Description") else ""
        extra: list[str] = []
        for k in ("GetData", "GetService"):
            if isinstance(url.get(k), dict):
                extra += [_s(v) for _, v in flatten(url[k])]
        if target or label:
            sentence = f"{label} URL: {target}".strip()
            if detail:
                sentence += f" -- {detail}"
            if extra:
                sentence += " (" + ", ".join(extra) + ")"
            parts.append(sentence.rstrip(".") + ".")


def _people(u: dict, parts: list[str]) -> None:
    people = [_person(p) for p in _list(u, "ContactPersons") if isinstance(p, dict)]
    if people:
        parts.append("Contacts: " + "; ".join(people) + ".")

    groups = []
    for g in _list(u, "ContactGroups"):
        if isinstance(g, dict):
            name = _s(g["GroupName"]) if g.get("GroupName") else ""
            roles = ", ".join(_s(r) for r in _list(g, "Roles"))
            detail = _contact_detail(g.get("ContactInformation"))
            groups.append("; ".join(x for x in (name, roles, detail) if x))
    if groups:
        parts.append("Contact groups: " + " | ".join(groups) + ".")

    for dc in _list(u, "DataCenters"):
        if not isinstance(dc, dict):
            continue
        name = _s(dc["ShortName"]) if dc.get("ShortName") else "the data centre"
        detail = _contact_detail(dc.get("ContactInformation"))
        nested = [_person(p) for p in _list(dc, "ContactPersons") if isinstance(p, dict)]
        for g in _list(dc, "ContactGroups"):
            if isinstance(g, dict):
                nested.append("; ".join(
                    x for x in (
                        _s(g["GroupName"]) if g.get("GroupName") else "",
                        ", ".join(_s(r) for r in _list(g, "Roles")),
                        _contact_detail(g.get("ContactInformation")),
                    ) if x
                ))
        if detail or nested:
            parts.append(
                f"Reach {name} via " + "; ".join(x for x in [detail, *nested] if x) + "."
            )


def _provenance(u: dict, parts: list[str]) -> None:
    for key, label in (("DataDates", "Data record"), ("MetadataDates", "Metadata")):
        for d in _list(u, key):
            if isinstance(d, dict):
                bits = _vals(d, "Type", "Date")
                if bits:
                    parts.append(f"{label} {bits[0]}" + (f" on {bits[1]}." if len(bits) > 1 else "."))

    for ref in _list(u, "PublicationReferences"):
        bits = [_s(v) for _, v in flatten(ref)]
        if bits:
            parts.append("Publication reference: " + ", ".join(bits) + ".")

    for assoc in _list(u, "MetadataAssociations"):
        bits = [_s(v) for _, v in flatten(assoc)]
        if bits:
            parts.append("Associated metadata: " + ", ".join(bits) + ".")

    for doi in _list(u, "AssociatedDOIs"):
        bits = [_s(v) for _, v in flatten(doi)]
        if bits:
            parts.append("Associated DOI: " + ", ".join(bits) + ".")

    spec = u.get("MetadataSpecification")
    if isinstance(spec, dict):
        bits = _vals(spec, "Name", "Version", "URL")
        if bits:
            parts.append("Metadata follows " + " ".join(bits) + ".")


#: Section writers, in reading order. Each appends sentences to ``parts``.
_SECTIONS = (
    _identity,
    _instrumentation,
    _keywords,
    _space,
    _time,
    _quality_and_content,
    _distribution,
    _people,
    _provenance,
)


def render_mat(p: dict) -> str:
    """Natural-language prose over a full UMM record.

    UMM-C is a *published schema*, not an arbitrary tree, so prose over the whole
    record is a matter of writing one template per field. :data:`_SECTIONS`
    covers the fields worth a sentence; whatever they do not carry is appended by
    :func:`_remainder` in breadcrumb form.

    That tail is what makes parity achievable without hand-writing a template for
    all 45 top-level fields *and* their long tail of sub-fields -- and it is
    self-correcting: a value is appended only if the prose does not already
    contain it, which is the same predicate :func:`parity_report` checks. Prose
    therefore satisfies value coverage by construction rather than by inspection.

    Values appear **verbatim**, which is why this reads ``status ACTIVE`` rather
    than ``status active``: parity is defined on the canonical string, and
    prettifying a value would drop it. ``sample_data/full_mat.txt`` -- the
    hand-authored target this follows -- prettifies, which is exactly why it
    covers only 78% of the record's values.
    """
    parts: list[str] = []
    for section in _SECTIONS:
        section(p, parts)
    body = " ".join(parts)

    tail = _remainder(p, body)
    return f"{body} {tail}".strip() if tail else body


def _remainder(p: dict, body: str) -> str:
    """Breadcrumb clauses for values the prose did not already carry."""
    missing = [
        (path, value) for path, value in flatten(p) if scalar_str(value) not in body
    ]
    if not missing:
        return ""
    return "Additional metadata: " + " ".join(
        f"{_speak_path(path)} is {scalar_str(value)}." for path, value in missing
    )


def prose_coverage(p: dict) -> dict:
    """How much of the record the prose carries before the tail takes over.

    Reported rather than assumed: a renderer that silently routed everything
    through the tail would pass parity while being a breadcrumb dump again.
    """
    parts: list[str] = []
    for section in _SECTIONS:
        section(p, parts)
    body = " ".join(parts)
    leaves = flatten(p)
    tailed = [1 for path, value in leaves if scalar_str(value) not in body]
    return {
        "leaves": len(leaves),
        "in_prose": len(leaves) - len(tailed),
        "in_tail": len(tailed),
        "prose_share": (len(leaves) - len(tailed)) / len(leaves) if leaves else 1.0,
    }


# --------------------------------------------------------------------------- #
# Registries and parity.
# --------------------------------------------------------------------------- #

RENDERERS = {
    "json": formats.render_json,
    "csv": formats.render_csv,
    "yaml": formats.render_yaml,
    "toon": formats.render_toon,
    "jsonld": render_jsonld,
    "mat": render_mat,
}

PARSERS = {
    "json": formats.parse_json,
    "csv": formats.parse_csv,
    "yaml": formats.parse_yaml,
    "toon": formats.parse_toon,
    "jsonld": parse_jsonld,
}

PROSE_FORMATS = ("mat",)


def render(fmt: str, p: dict) -> str:
    try:
        return RENDERERS[fmt](p)
    except KeyError:
        raise KeyError(f"unknown format {fmt!r}; known: {', '.join(RENDERERS)}") from None


def render_all(record: dict) -> dict[str, str]:
    p = payload(record)
    return {fmt: render(fmt, p) for fmt in FORMATS}


def parity_report(p: dict) -> dict[str, dict]:
    """Per-format parity for one unfaceted payload.

    Identical in shape to :func:`airm.formats.parity_report`, so the two caches
    are judged by the same standard rather than the control being graded gently.
    """
    expected = fact_set(p)
    report: dict[str, dict] = {}

    for fmt, parser in PARSERS.items():
        try:
            got = fact_set(parser(render(fmt, p)))
        except Exception as exc:  # noqa: BLE001 - any failure is a parity failure
            report[fmt] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            continue
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        report[fmt] = {"ok": not missing and not extra, "missing": missing, "extra": extra}

    # Dispatch through :func:`render` rather than calling ``render_mat``
    # directly, so ``RENDERERS`` is the single point of dispatch and a swapped
    # renderer is actually seen by the validator.
    for fmt in PROSE_FORMATS:
        text = render(fmt, p)
        missing_values = [v for v in fact_values(p) if v not in text]
        report[fmt] = {
            "ok": not missing_values,
            "missing": [("<value>", v) for v in missing_values],
            "extra": [],
        }
    return report


def check_record(record: dict) -> tuple[bool, dict[str, dict]]:
    report = parity_report(payload(record))
    return all(r["ok"] for r in report.values()), report


# --------------------------------------------------------------------------- #
# Paths and fingerprint.
# --------------------------------------------------------------------------- #


def fingerprint() -> str:
    """Hash over every module that can change an unfaceted rendering.

    ``formats.py`` supplies four of the six renderers and this module supplies
    the other two, so both are hashed. ``facets.py`` is deliberately *not*: the
    unfaceted payload does not go through it, which is the point, and hashing it
    would invalidate this cache every time the projection changed.
    """
    digest = hashlib.sha256()
    here = Path(__file__).parent
    for name in ("formats.py", "unfaceted.py"):
        digest.update((here / name).read_bytes())
    return digest.hexdigest()[:16]


def format_dir(fmt: str, *, root: Path | None = None) -> Path:
    return (root or UNFACETED_CACHE_DIR) / fmt


def path_for(fmt: str, cid: str, *, root: Path | None = None) -> Path:
    try:
        ext = EXTENSIONS[fmt]
    except KeyError:
        raise KeyError(f"unknown format {fmt!r}; known: {', '.join(EXTENSIONS)}") from None
    return format_dir(fmt, root=root) / f"{cid}{ext}"


def cached_concept_ids(*, cache_dir: Path | None = None) -> list[str]:
    return sorted(p.stem for p in (cache_dir or CMR_CACHE_DIR).glob("C*.json"))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Build.
# --------------------------------------------------------------------------- #


@dataclass
class CacheReport:
    fingerprint: str = ""
    records: int = 0
    written: dict[str, int] = field(default_factory=dict)
    skipped_unchanged: int = 0
    render_errors: list[dict] = field(default_factory=list)
    parity_failures: list[dict] = field(default_factory=list)
    #: Per-format count of records that failed to round-trip, so the TOON
    #: shortfall is a headline number rather than something to derive.
    parity_by_format: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint,
            "payload": "raw UMM (unfaceted)",
            "formats": list(FORMATS),
            "extensions": EXTENSIONS,
            "records": self.records,
            "written": self.written,
            "skipped_unchanged": self.skipped_unchanged,
            "render_errors": self.render_errors,
            "parity_by_format": self.parity_by_format,
            "parity_failures": self.parity_failures,
        }


def build(
    concept_ids: list[str] | None = None,
    *,
    root: Path | None = None,
    cache_dir: Path | None = None,
    force: bool = False,
    check_parity: bool = True,
) -> CacheReport:
    """Render every cached record, unfaceted, in every format."""
    root = root or UNFACETED_CACHE_DIR
    fp = fingerprint()
    report = CacheReport(fingerprint=fp)

    if force or _manifest_fingerprint(root) != fp:
        force = True

    if concept_ids is None:
        concept_ids = cached_concept_ids(cache_dir=cache_dir)
    report.records = len(concept_ids)
    report.written = {fmt: 0 for fmt in FORMATS}
    report.parity_by_format = {fmt: 0 for fmt in FORMATS}

    for cid in concept_ids:
        paths = {fmt: path_for(fmt, cid, root=root) for fmt in FORMATS}
        if not force and all(p.exists() for p in paths.values()):
            report.skipped_unchanged += 1
            continue

        record = cmr.load_cached(cid) if cache_dir is None else _load_from(cache_dir, cid)
        if record is None:
            report.render_errors.append({"concept_id": cid, "error": "raw record missing"})
            continue

        p = payload(record)
        try:
            rendered = {fmt: render(fmt, p) for fmt in FORMATS}
        except Exception as exc:  # noqa: BLE001 - one bad record must not stop the build
            report.render_errors.append(
                {"concept_id": cid, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue

        for fmt, text in rendered.items():
            _write(paths[fmt], text)
            report.written[fmt] += 1

        if check_parity:
            detail = parity_report(p)
            failed = {k: v for k, v in detail.items() if not v["ok"]}
            for fmt in failed:
                report.parity_by_format[fmt] += 1
            if failed:
                report.parity_failures.append(
                    {
                        "concept_id": cid,
                        # Full missing/extra lists run to thousands of entries on
                        # a raw record; the format and a sample is what a reader
                        # can act on.
                        "formats": {
                            k: v.get("error") or {
                                "missing": v["missing"][:3],
                                "extra": v["extra"][:3],
                                "n_missing": len(v["missing"]),
                                "n_extra": len(v["extra"]),
                            }
                            for k, v in failed.items()
                        },
                    }
                )

    _write_manifest(root, report)
    return report


def _load_from(cache_dir: Path, cid: str) -> dict | None:
    path = cache_dir / f"{cid}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


# --------------------------------------------------------------------------- #
# Manifest and read.
# --------------------------------------------------------------------------- #


def manifest_path(root: Path | None = None) -> Path:
    return (root or UNFACETED_CACHE_DIR) / MANIFEST_NAME


def load_manifest(root: Path | None = None) -> dict | None:
    path = manifest_path(root)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _manifest_fingerprint(root: Path | None = None) -> str | None:
    manifest = load_manifest(root)
    return manifest.get("fingerprint") if manifest else None


def _write_manifest(root: Path, report: CacheReport) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path(root).write_text(json.dumps(report.to_dict(), indent=2) + "\n")


def is_current(root: Path | None = None) -> bool:
    return _manifest_fingerprint(root) == fingerprint()


def load(fmt: str, cid: str, *, root: Path | None = None, strict: bool = True) -> str:
    if strict and not is_current(root):
        raise StaleCacheError(
            f"unfaceted cache was written by rendering code {_manifest_fingerprint(root)!r}, "
            f"current is {fingerprint()!r}. Rebuild with "
            f"`uv run python -m airm.unfaceted --build`."
        )
    return path_for(fmt, cid, root=root).read_text(encoding="utf-8")


def render_all_cached(cid: str | None, p: dict, *, root: Path | None = None) -> dict[str, str]:
    """Every format's unfaceted rendering of one record — cache first, live otherwise.

    The mirror of :func:`airm.format_cache.render_all_cached`, and for the same
    reason: a caller gets the speed of the cache without depending on it. A
    missing, stale or partial cache costs time, never correctness, because the
    fallback renders the same payload through the same code that filled it.
    """
    usable = is_current(root)
    out: dict[str, str] = {}
    for fmt in FORMATS:
        if usable and cid is not None:
            path = path_for(fmt, cid, root=root)
            if path.exists():
                out[fmt] = path.read_text(encoding="utf-8")
                continue
        out[fmt] = render(fmt, p)
    return out


def load_all(fmt: str, *, root: Path | None = None, strict: bool = True) -> dict[str, str]:
    if strict and not is_current(root):
        raise StaleCacheError("unfaceted cache is stale; rebuild it")
    ext = EXTENSIONS[fmt]
    return {
        p.name[: -len(ext)]: p.read_text(encoding="utf-8")
        for p in sorted(format_dir(fmt, root=root).glob(f"*{ext}"))
    }


# --------------------------------------------------------------------------- #
# Verification.
# --------------------------------------------------------------------------- #


def verify(*, root: Path | None = None, cache_dir: Path | None = None) -> list[str]:
    """Hard-gate violations; empty when the mapping is one-to-one and current.

    Parity failures are *not* a violation here. TOON cannot round-trip every raw
    UMM record, and that is a measured property of the control rather than a
    broken build -- unlike the faceted cache, where a parity failure means a
    record must leave the corpus.
    """
    problems: list[str] = []
    expected = set(cached_concept_ids(cache_dir=cache_dir))

    manifest = load_manifest(root)
    if manifest is None:
        return [f"no manifest at {manifest_path(root)}; the cache has never been built"]
    if manifest.get("fingerprint") != fingerprint():
        problems.append(
            f"stale: written by rendering code {manifest.get('fingerprint')!r}, "
            f"current is {fingerprint()!r}"
        )

    errored = {e["concept_id"] for e in manifest.get("render_errors") or []}
    renderable = expected - errored
    if errored:
        problems.append(f"{len(errored)} record(s) failed to render: {sorted(errored)[:5]}")

    for fmt in FORMATS:
        present = set(load_all(fmt, root=root, strict=False))
        missing = renderable - present
        extra = present - expected
        if missing:
            problems.append(f"{fmt}: {len(missing)} record(s) with no rendering: {sorted(missing)[:5]}")
        if extra:
            problems.append(f"{fmt}: {len(extra)} rendering(s) with no raw record: {sorted(extra)[:5]}")
    return problems


if __name__ == "__main__":  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(
        description="Render every cached CMR record, unfaceted, in every format."
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--show", metavar="CONCEPT_ID")
    args = parser.parse_args()

    if args.show:
        for fmt in FORMATS:
            text = load(fmt, args.show, strict=False)
            print(f"\n{'=' * 72}\n{fmt}  ({len(text):,} chars)\n{'=' * 72}")
            print(text[:2000] + ("\n... [truncated]" if len(text) > 2000 else ""))
        raise SystemExit(0)

    if args.build:
        rep = build(force=args.force)
        print(f"fingerprint      : {rep.fingerprint}")
        print(f"records          : {rep.records}")
        print(f"skipped (current): {rep.skipped_unchanged}")
        print("written per format:")
        for fmt in FORMATS:
            print(f"  {fmt:<8} {rep.written.get(fmt, 0):>6}")
        if rep.render_errors:
            print(f"render errors    : {len(rep.render_errors)}")
            for err in rep.render_errors[:5]:
                print(f"  {err['concept_id']}: {err['error'][:100]}")
        print("round-trip failures per format:")
        for fmt in FORMATS:
            n = rep.parity_by_format.get(fmt, 0)
            pct = 100 * n / rep.records if rep.records else 0
            print(f"  {fmt:<8} {n:>6}  ({pct:.1f}%)")
        print(f"wrote {manifest_path()}")

    if args.verify or args.build:
        issues = verify()
        if issues:
            print("\nHARD GATE FAILURES:")
            for issue in issues:
                print(f"  - {issue}")
            raise SystemExit(1)
        print("\ncache is one-to-one with cmr_cache and current with the rendering code")

    if not (args.build or args.verify or args.show):
        parser.error("pass --build, --verify or --show")
