# Field inventory — faceted projection vs. unfaceted control

Sources: `src/airm/facets.py` (the 32-facet projection consumed by every renderer)
and `src/airm/unfaceted.py` (the control, which renders the whole `umm` object —
45 top-level fields, ~398 distinct paths).

Coverage is over all 2,590 records in `data/cmr_cache/`.

## Faceted fields — the 32 in `FACET_KEYS`

| # | Facet | UMM source | Coverage |
|---|---|---|---|
| | **identity** | | |
| 1 | `concept_id` | `meta.concept-id` | 100.0% |
| 2 | `cmr_link` | *constructed* from concept-id (only derived facet) | 100.0% |
| 3 | `title` | `EntryTitle` → `ShortName` fallback | 100.0% |
| 4 | `short_name` | `ShortName` | 100.0% |
| 5 | `doi` | `DOI.DOI` | 43.7% |
| | **provenance** | | |
| 6 | `agency` | `DataCenters[].LongName` → `meta.provider-id` | 100.0% |
| 7 | `data_center` | `DataCenters[].ShortName` (ARCHIVER/DISTRIBUTOR preferred) | 100.0% |
| | **science** | | |
| 8 | `topic` | `ScienceKeywords[].Topic` | 100.0% |
| 9 | `summary` | `Abstract` | 100.0% |
| 10 | `science_keywords` | `ScienceKeywords[]` joined as GCMD `A > B > C` paths | 100.0% |
| 11 | `variables` | most specific `ScienceKeywords[]` level per entry | 100.0% |
| | **instrumentation** | | |
| 12 | `platform` | `Platforms[].ShortName` | 64.0% |
| 13 | `instrument` | `Platforms[].Instruments[].ShortName` | 57.2% |
| 14 | `techniques` | `Instruments[].Technique`, `ComposedOf[].Technique` | 2.4% |
| 15 | `spectral_bands` | scrape: `ComposedOf[].ShortName`, `OperationalModes`, `Characteristics[].Name`, science-keyword levels, `AdditionalAttributes[].Name` (regex-matched) | 13.7% |
| | **space** | | |
| 16 | `bbox` | `SpatialExtent…Geometry.BoundingRectangles[0]` `{west,south,east,north}` | 86.8% |
| 17 | `coordinate_system` | `Geometry.CoordinateSystem` → `GranuleSpatialRepresentation` | 100.0% |
| 18 | `spatial_resolution` | `HorizontalDataResolution.{Gridded,NonGridded,Generic}Resolutions` `{x,y,unit}` | 4.7% |
| | **time** | | |
| 19 | `temporal` | `TemporalExtents[].RangeDateTimes[]` `{begin,end}` | 96.4% |
| 20 | `temporal_resolution` | `TemporalExtents[].TemporalResolution` `{value,unit}` | 2.5% |
| | **distribution** | | |
| 21 | `data_format` | `File{Distribution,Archive}Information[].Format` ∪ `RelatedUrls[].GetData.Format` | 35.3% |
| 22 | `data_volume` | `TotalCollectionFileSize` / `AverageFileSize` (+unit, +basis) | 30.3% |
| | **quality** | | |
| 23 | `processing_level` | `ProcessingLevel.Id` | 25.6% |
| 24 | `processing_level_description` | `ProcessingLevel.ProcessingLevelDescription` | 12.3% |
| 25 | `version` | `Version` | 54.8% |
| 26 | `collection_progress` | `CollectionProgress` | 76.3% |
| | **rights** | | |
| 27 | `access_constraints` | `AccessConstraints.Description` → `.Value` | 35.9% |
| 28 | `use_constraints` | `UseConstraints.{Description,LicenseText,LicenseURL,FreeAndOpenData}` | 59.4% |
| | **links** | | |
| 29 | `urls` | `RelatedUrls[].URL` (all) | 86.8% |
| 30 | `urls_publication` | `URLContentType == PublicationURL` (+ `CollectionCitations[].OnlineResource.Linkage` fallback) | 68.4% |
| 31 | `urls_repository` | `URLContentType == DistributionURL` or `Type == DATA SET LANDING PAGE` | 68.1% |
| 32 | `urls_additional` | remainder of `RelatedUrls` | 45.1% |

The projection reads **18 of the 45 top-level UMM fields** — two of them only
partially: `CollectionCitations` just for the publication-URL fallback,
`AdditionalAttributes` just for spectral-sounding names — plus two `meta` keys.

## Unfaceted-only — the top-level UMM fields the projection drops

Present only in `data/format_cache_unfaceted*/`, which renders `umm` untouched
(`meta` excluded).

| # | UMM field | Records carrying it |
|---|---|---|
| 1 | `MetadataSpecification` | 2,590 (100%) |
| 2 | `MetadataDates` | 2,140 (82.6%) |
| 3 | `ISOTopicCategories` | 2,107 (81.4%) |
| 4 | `LocationKeywords` | 2,106 (81.3%) |
| 5 | `ContactPersons` | 1,778 (68.6%) |
| 6 | `CollectionCitations` † | 1,737 (67.1%) |
| 7 | `AncillaryKeywords` | 1,570 (60.6%) |
| 8 | `DataLanguage` | 1,441 (55.6%) |
| 9 | `DirectoryNames` | 1,383 (53.4%) |
| 10 | `Projects` | 1,273 (49.2%) |
| 11 | `AdditionalAttributes` † | 1,108 (42.8%) |
| 12 | `DataDates` | 1,048 (40.5%) |
| 13 | `Quality` | 580 (22.4%) |
| 14 | `StandardProduct` | 511 (19.7%) |
| 15 | `DirectDistributionInformation` | 503 (19.4%) |
| 16 | `Purpose` | 449 (17.3%) |
| 17 | `ContactGroups` | 300 (11.6%) |
| 18 | `MetadataAssociations` | 286 (11.0%) |
| 19 | `PublicationReferences` | 237 (9.2%) |
| 20 | `CollectionDataType` | 167 (6.4%) |
| 21 | `TemporalKeywords` | 108 (4.2%) |
| 22 | `VersionDescription` | 95 (3.7%) |
| 23 | `PaleoTemporalCoverages` | 90 (3.5%) |
| 24 | `MetadataLanguage` | 44 (1.7%) |
| 25 | `FileNamingConvention` | 16 (0.6%) |
| 26 | `SpatialInformation` | 15 (0.6%) |
| 27 | `DataMaturity` | 11 (0.4%) |
| 28 | `TilingIdentificationSystems` | 8 (0.3%) |
| 29 | `AssociatedDOIs` | 2 (0.1%) |

† `CollectionCitations` and `AdditionalAttributes` appear here because the
projection touches only one narrow sub-path of each; the field as a whole
survives only in the unfaceted control. So: 27 fully dropped, 2 partially.

The asymmetry runs the other way too — `concept_id` and `cmr_link` exist **only**
in the faceted payload. `unfaceted.payload()` deliberately excludes `meta`, and
`concept-id` lives in `meta`, not `umm`, so no unfaceted rendering carries it as
a *field*. It still leaks in incidentally for **546 of 2,590 records (21.1%)** as
a substring of a `cmr.earthdata…/concepts/C…` URL — always inside `RelatedUrls`,
the only top-level field it ever appears in.

Record identity in the unfaceted payload therefore rests on UMM's own keys:
`ShortName` and `Version` (100% each, CMR's native collection key), `EntryTitle`
(100%) and `DOI.DOI` (43.7%). Nothing in the study depends on this — every
rendering is filed under its concept-id (`format_cache_unfaceted/<fmt>/<id>.<ext>`,
a mapping `verify` asserts in both directions), so retrieval and scoring key on
the filename, and no query in either set (0 of 511, 0 of 500) asks for a concept
id or DOI in its text.
