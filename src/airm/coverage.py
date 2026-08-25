"""What the canonical payload keeps from the raw CMR record — and what it drops.

:mod:`airm.formats` guarantees that all six representations carry *the same*
content as each other: every machine format round-trips to an identical fact set
and prose contains every canonical value verbatim. That is the guarantee
Experiment 1 needs, and it holds.

It is not the same as saying the formats carry *everything CMR published*. They
do not. :func:`airm.facets.facets` is a projection, and a projection discards.
This module makes that discard explicit, measured and enforced, so it can never
again be a silent property of the code.

Every scalar path in a CMR item is classified into exactly one bucket:

``CARRIED``
    Read by :func:`~airm.facets.facets` and therefore present in all six formats.
``EXCLUDED``
    Deliberately dropped, with a stated reason. These describe the *catalog
    entry* or the *publisher*, not the dataset — CMR bookkeeping, UMM schema
    identifiers, and personal contact details.
``DEFERRED``
    Real dataset content that is not carried yet. This is a backlog, not a
    justification. Each entry says what it would cost to include.

:func:`verify` fails on any path in none of the three — so when CMR adds a field
or an extractor stops reading one, the study finds out instead of quietly
shrinking. Run it with ``uv run python -m airm.coverage``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .config import CORPUS_PATH

#: ``platforms[0].instruments[1]`` -> ``platforms[].instruments[]``. Indices are
#: irrelevant to whether a *field* is carried, so they are collapsed.
_INDEX = re.compile(r"\[\d+\]")


class CoverageError(AssertionError):
    """Raised when a CMR path is neither carried nor declared."""


# --------------------------------------------------------------------------- #
# The declaration.
# --------------------------------------------------------------------------- #

#: UMM path prefix -> the facet key that carries it. Anything matching one of
#: these reaches all six formats.
CARRIED: dict[str, str] = {
    "meta.concept-id": "concept_id",
    "umm.EntryTitle": "title",
    "umm.ShortName": "short_name",
    "umm.DOI.DOI": "doi",
    "umm.Abstract": "summary",
    "umm.Version": "quality.version",
    "umm.CollectionProgress": "quality.collection_progress",
    "umm.ProcessingLevel.Id": "quality.processing_level",
    "umm.ScienceKeywords[]": "science_keywords / topic / variables",
    "umm.Platforms[].ShortName": "platforms[].platform",
    "umm.Platforms[].Instruments[].ShortName": "platforms[].instruments",
}

#: Carried, but only the *first* element of a repeating group. The rest of the
#: group is real content that no format sees, so each is also a ``DEFERRED``
#: entry describing what is lost.
PARTIAL: dict[str, str] = {
    "umm.DataCenters[].ShortName": "data_center — first centre only",
    "umm.SpatialExtent.HorizontalSpatialDomain.Geometry.BoundingRectangles[]": (
        "bbox — first rectangle only"
    ),
    "umm.TemporalExtents[].RangeDateTimes[]": "temporal — first range only",
    "umm.TemporalExtents[].EndsAtPresentFlag": "temporal.end = 'present'",
}

#: Deliberately dropped, with the reason. Describes the catalog record or the
#: publisher rather than the dataset.
EXCLUDED: dict[str, str] = {
    "meta.associations": "CMR graph edges to other concepts; catalog plumbing",
    "meta.association-details": "CMR graph edges to other concepts; catalog plumbing",
    "meta.concept-type": "always 'collection' in this corpus",
    "meta.deleted": "tombstone flag for the catalog record",
    "meta.format": "MIME type of the stored metadata document",
    "meta.native-id": "provider's internal filename for the metadata record",
    "meta.provider-id": "duplicate of DataCenters ShortName",
    "meta.revision-id": "version of the metadata record, not of the data",
    "meta.revision-date": "when the metadata record was last edited",
    "meta.user-id": "which operator last touched the catalog record",
    "meta.has-combine": "CMR service capability flag, not dataset content",
    "meta.has-formats": "CMR service capability flag, not dataset content",
    "meta.has-spatial-subsetting": "CMR service capability flag, not dataset content",
    "meta.has-temporal-subsetting": "CMR service capability flag, not dataset content",
    "meta.has-transforms": "CMR service capability flag, not dataset content",
    "meta.has-variables": "CMR service capability flag, not dataset content",
    "meta.s3-links": "storage location, duplicated in DirectDistributionInformation",
    "umm.MetadataSpecification": "UMM schema name/version/URL; identical for every record",
    "umm.MetadataDates": "when the metadata was authored, not what the data covers",
    "umm.MetadataLanguage": "language of the metadata document itself",
    "umm.ContactPersons[].ContactInformation": "personal addresses, phone numbers, email",
    "umm.ContactGroups[].ContactInformation": "postal addresses, phone numbers, email",
    "umm.DataCenters[].ContactPersons": "personal addresses, phone numbers, email",
    "umm.DataCenters[].ContactGroups": "postal addresses, phone numbers, email",
    "umm.DataCenters[].ContactInformation": "postal addresses, phone numbers, email",
}

#: Substantive dataset content that no format currently carries. A backlog.
DEFERRED: dict[str, str] = {
    "umm.Purpose": "prose statement of intended use",
    "umm.Quality": "prose statement of known quality issues",
    "umm.VersionDescription": "what changed in this version",
    "umm.CollectionDataType": "SCIENCE_QUALITY / NEAR_REAL_TIME — retrieval-relevant",
    "umm.DataMaturity": "Validated / Provisional — retrieval-relevant",
    "umm.StandardProduct": "flags a NASA standard product",
    "umm.FileNamingConvention": "how granule filenames are constructed",
    "umm.RelatedUrls": "data-access, documentation and thumbnail URLs",
    "umm.Projects": "mission/campaign names — highly searchable",
    "umm.LocationKeywords": "GCMD place names — the spatial equivalent of ScienceKeywords",
    "umm.ISOTopicCategories": "ISO 19115 topic codes",
    "umm.AncillaryKeywords": "free-text keywords chosen by the provider",
    "umm.TemporalKeywords": "temporal resolution vocabulary",
    "umm.DirectoryNames": "GCMD directory / data-set naming",
    "umm.AdditionalAttributes": "provider-defined attributes and their descriptions",
    "umm.ArchiveAndDistributionInformation": "file formats, sizes, average granule size",
    "umm.DirectDistributionInformation": "S3 bucket and credentials endpoints",
    "umm.CollectionCitations": "how to cite the dataset",
    "umm.PublicationReferences": "papers describing the dataset",
    "umm.UseConstraints": "licence and use restrictions",
    "umm.AccessConstraints": "access restrictions",
    "umm.DataLanguage": "language of the data itself",
    "umm.DataDates": "creation / last-revision dates of the data",
    "umm.PaleoTemporalCoverages": "geologic-time coverage for paleo datasets",
    "umm.TilingIdentificationSystems": "tiling grid the granules are cut on",
    "umm.MetadataAssociations": "links to related collections",
    "umm.ContactPersons[].Roles": "investigator / author role",
    "umm.ContactPersons[].FirstName": "investigator name",
    "umm.ContactPersons[].MiddleName": "investigator name",
    "umm.ContactPersons[].LastName": "investigator name",
    "umm.ContactPersons[].NonDataCenterAffiliation": "investigator affiliation",
    "umm.ContactGroups[].Roles": "contact group role",
    "umm.ContactGroups[].GroupName": "contact group name",
    "umm.ContactGroups[].NonDataCenterAffiliation": "contact group affiliation",
    "umm.DataCenters[].Roles": "ARCHIVER / DISTRIBUTOR / PROCESSOR",
    "umm.DataCenters[].LongName": "full name of the archiving organisation",
    "umm.DataCenters[]": "second and subsequent data centres",
    "umm.DOI.Authority": "DOI registration authority",
    "umm.DOI.MissingReason": "why a DOI is absent",
    "umm.DOI.Explanation": "narrative on DOI status",
    "umm.ProcessingLevel.ProcessingLevelDescription": "what the level means for this product",
    "umm.Platforms[].LongName": "full platform name",
    "umm.Platforms[].Type": "Earth Observation Satellite / Aircraft / etc.",
    "umm.Platforms[].Characteristics": "platform characteristics",
    "umm.Platforms[].Instruments[].LongName": "full instrument name",
    "umm.Platforms[].Instruments[].Characteristics": "instrument characteristics",
    "umm.Platforms[].Instruments[].ComposedOf": "sub-instruments / channels",
    "umm.Platforms[].Instruments[].Technique": "measurement technique",
    "umm.Platforms[].Instruments[].NumberOfInstruments": "instrument count",
    "umm.Platforms[].Instruments[].OperationalModes": "instrument operating modes",
    "umm.SpatialExtent.GranuleSpatialRepresentation": "CARTESIAN / GEODETIC / NO_SPATIAL",
    "umm.SpatialExtent.SpatialCoverageType": "HORIZONTAL / VERTICAL / ORBITAL",
    "umm.SpatialExtent.HorizontalSpatialDomain.Geometry.CoordinateSystem": "geometry CRS",
    "umm.SpatialExtent.HorizontalSpatialDomain.Geometry.GPolygons": "polygon footprints",
    "umm.SpatialExtent.HorizontalSpatialDomain.Geometry.Points": "point footprints",
    "umm.SpatialExtent.HorizontalSpatialDomain.Geometry.Lines": "line footprints",
    "umm.SpatialExtent.HorizontalSpatialDomain.ResolutionAndCoordinateSystem": (
        "horizontal resolution"
    ),
    "umm.SpatialExtent.HorizontalSpatialDomain.ZoneIdentifier": "grid zone",
    "umm.SpatialExtent.VerticalSpatialDomains": "depth / altitude coverage",
    "umm.SpatialExtent.OrbitParameters": "orbital geometry",
    "umm.TemporalExtents[]": "second and subsequent temporal extents",
    "umm.TemporalExtents[].SingleDateTimes": "discrete observation times",
    "umm.TemporalExtents[].PeriodicDateTimes": "repeating campaign windows",
    "umm.TemporalExtents[].PrecisionOfSeconds": "timestamp precision",
    "umm.TemporalExtents[].TemporalRangeType": "range type",
    "umm.SpatialInformation": "spatial reference details",
    "umm.LocationKeywords[]": "GCMD place names",
    "umm.lastUpdate": "provider-supplied update timestamp",
}


# --------------------------------------------------------------------------- #
# Census.
# --------------------------------------------------------------------------- #


def generalise(path: str) -> str:
    """``a[0].b[3]`` -> ``a[].b[]``."""
    return _INDEX.sub("[]", path)


def scalar_paths(value: Any, prefix: str = "") -> list[str]:
    """Every scalar path in a CMR item, indices collapsed."""
    out: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            out.extend(scalar_paths(child, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        for child in value:
            out.extend(scalar_paths(child, f"{prefix}[]"))
    elif prefix:
        out.append(prefix)
    return out


def _covers(prefix: str, path: str) -> bool:
    """Does ``prefix`` name ``path`` or one of its ancestors?

    A prefix matches at a structural boundary only, so ``umm.Version`` does not
    swallow ``umm.VersionDescription`` — the next character must start a new key
    (``.``) or an array step (``[``).
    """
    return path == prefix or path.startswith((prefix + ".", prefix + "["))


def _classify(path: str) -> tuple[str, str]:
    """``(bucket, note)`` for one path, by longest matching prefix.

    Longest-match matters: ``umm.DataCenters[].ShortName`` is carried while the
    broader ``umm.DataCenters[]`` is deferred, and ``meta.concept-id`` is carried
    while the rest of ``meta`` is excluded.
    """
    best: tuple[int, str, str] = (-1, "UNDECLARED", "")
    for table, bucket in (
        (CARRIED, "CARRIED"),
        (PARTIAL, "PARTIAL"),
        (EXCLUDED, "EXCLUDED"),
        (DEFERRED, "DEFERRED"),
    ):
        for prefix, note in table.items():
            if _covers(prefix, path) and len(prefix) > best[0]:
                best = (len(prefix), bucket, note)
    return best[1], best[2]


def as_item(record: dict) -> dict:
    """Normalise to the ``{"meta", "umm"}`` item shape the declaration is written against.

    :func:`airm.facets.facets` accepts either a full CMR item or a bare UMM
    payload, so this module has to as well — otherwise a bare payload's paths all
    read ``ScienceKeywords[]`` instead of ``umm.ScienceKeywords[]`` and every one
    of them looks undeclared.
    """
    return record if "umm" in record else {"umm": record}


def census(records: Iterable[dict]) -> Counter:
    """How many scalar values each generalised path contributes, corpus-wide."""
    counts: Counter = Counter()
    for record in records:
        for path in scalar_paths(as_item(record)):
            counts[generalise(path)] += 1
    return counts


def report(records: Iterable[dict]) -> dict:
    """Classify the whole corpus and total the values in each bucket."""
    counts = census(records)
    buckets: dict[str, list[tuple[str, int, str]]] = {
        "CARRIED": [],
        "PARTIAL": [],
        "EXCLUDED": [],
        "DEFERRED": [],
        "UNDECLARED": [],
    }
    for path, n in counts.items():
        bucket, note = _classify(path)
        buckets[bucket].append((path, n, note))
    for rows in buckets.values():
        rows.sort(key=lambda row: -row[1])
    totals = {name: sum(n for _, n, _ in rows) for name, rows in buckets.items()}
    return {"buckets": buckets, "totals": totals, "values": sum(totals.values())}


def verify(records: Iterable[dict]) -> None:
    """Raise :class:`CoverageError` if any CMR path is undeclared."""
    res = report(records)
    unknown = res["buckets"]["UNDECLARED"]
    if unknown:
        lines = [
            f"{len(unknown)} CMR path(s) are neither carried nor declared. "
            f"Add each to CARRIED, EXCLUDED or DEFERRED in airm/coverage.py:"
        ]
        lines += [f"  {path}  ({n} values)" for path, n, _ in unknown[:20]]
        raise CoverageError("\n".join(lines))


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #


def _print(res: dict, *, show: str | None) -> None:
    total = res["values"]
    print(f"{total:,} scalar values across the corpus\n")
    order = ("CARRIED", "PARTIAL", "EXCLUDED", "DEFERRED", "UNDECLARED")
    print(f"{'bucket':12s} {'values':>10s} {'share':>7s}  {'paths':>6s}")
    for name in order:
        n = res["totals"][name]
        print(f"{name:12s} {n:10,d} {n / total:6.1%}  {len(res['buckets'][name]):6d}")

    reach = res["totals"]["CARRIED"] + res["totals"]["PARTIAL"]
    print(f"\nreaches all six formats: {reach:,} / {total:,} = {reach / total:.1%}")

    if show:
        rows = res["buckets"][show.upper()]
        print(f"\n{show.upper()} ({len(rows)} paths)")
        for path, n, note in rows:
            print(f"  {n:7,d}  {path}\n           {note}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument(
        "--show",
        choices=["carried", "partial", "excluded", "deferred", "undeclared"],
        help="list the paths in one bucket",
    )
    args = parser.parse_args()

    with args.corpus.open() as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    res = report(records)
    _print(res, show=args.show)

    unknown = res["buckets"]["UNDECLARED"]
    if unknown:
        print(f"\nFAIL: {len(unknown)} undeclared path(s)")
        for path, n, _ in unknown[:20]:
            print(f"  {path}  ({n:,} values)")
        return 1
    print("\nOK: every CMR path is declared.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
