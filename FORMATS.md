# Formats — what each one carries, and what never reaches them

This is the content audit behind Experiment 1. It answers two different questions that
are easy to conflate:

1. **Do the six formats carry the same content as each other?** Yes — exactly, on
   500/500 records. That is what makes the token comparison a *format* comparison.
2. **Do they carry everything CMR published?** No — 18.4% of the source record's scalar
   values reach them. The loss happens upstream of every renderer, so it cannot bias the
   comparison, but it does mean the study prices a metadata *summary*, not the full record.

Numbers here come from the 500-record corpus (`data/corpus.jsonl`). Regenerate the
coverage figures with `uv run python -m airm.coverage`.

---

## 1. The pipeline

```
CMR UMM-JSON item          facets(record)              render(fmt, payload)
{meta, umm}        ──►     13-key canonical      ──►   json · csv · yaml
321 distinct paths         payload                     toon · jsonld · mat
97,312 scalar values       17,942 values reach here
```

There is exactly one content source. Every renderer consumes `facets()` and nothing
else — no renderer may touch the raw record. That single chokepoint is what makes the
formats comparable, and it is also where all the content loss happens.

---

## 2. What the canonical payload carries

Thirteen keys, in a fixed order shared by every format that imposes one, so key ordering
never becomes a hidden variable.

| Facet key | UMM source | Present in |
|---|---|---:|
| `concept_id` | `meta.concept-id` | 100% |
| `title` | `umm.EntryTitle` (falls back to `ShortName`) | 100% |
| `short_name` | `umm.ShortName` | 100% |
| `doi` | `umm.DOI.DOI` | 49.6% |
| `data_center` | `umm.DataCenters[0].ShortName` | 100% |
| `topic` | first `umm.ScienceKeywords[].Topic` | 100% |
| `summary` | `umm.Abstract` | 100% |
| `platforms` | `umm.Platforms[].ShortName` + `Instruments[].ShortName` | 100% |
| `science_keywords` | `umm.ScienceKeywords[]`, joined `A > B > C` | 100% |
| `variables` | most specific level per science keyword, deduplicated | 100% |
| `bbox` | `umm.SpatialExtent…BoundingRectangles[0]` | 88.6% |
| `temporal` | `umm.TemporalExtents[0].RangeDateTimes[0]` | 96.4% |
| `quality` | `ProcessingLevel.Id`, `Version`, `CollectionProgress` | 100% |

Two rules make the payload safe to compare across formats:

- **Empty facets are omitted, never emitted as `null` or `[]`.** A format that prints
  `"bbox": null` and one that omits the key carry the same information at very different
  token cost. This applies to nested values too — a platform with no instruments has no
  `instruments` key at all. (That nested case was a live bug until recently; see §6.)
- **Numbers are canonicalised once, at extraction.** An integral coordinate is stored as
  an `int`, so a bounding box reads `-180` in every format instead of `-180.0` in JSON,
  YAML and TOON and `-180` in prose — a pure serialisation artifact that would otherwise
  show up as a token difference in an experiment about token cost.

---

## 3. The six formats, side by side

Same record (`C1214305730-AU_AADC`), rendered from one payload. Token counts are
`tiktoken` `o200k_base`.

### `json` — 282 tokens *(baseline)*

```json
{"concept_id":"C1214305730-AU_AADC","title":"Amery Ice Shelf - hot water drill borehole, 2001-02 - AM01 drilling parameters data","short_name":"ASAC_1164_AM01_Drilling","doi":"doi:10.4225/15/5271A0B888E35","data_center":"AU/AADC","topic":"CRYOSPHERE","summary":"AM01 borehole drilled January 2002.\nData collected in series of files over a period of 2 days during production of borehole.\n\nConsult Readme file for detail of data files and formats.","platforms":[{"platform":"Not provided","instruments":["CORING DEVICES"]},{"platform":"GROUND-BASED OBSERVATIONS"},{"platform":"FIELD INVESTIGATION"}],"science_keywords":["EARTH SCIENCE > CRYOSPHERE > SNOW/ICE"],"variables":["Snow/Ice"],"bbox":{"west":71.42,"south":-69.44,"east":71.42,"north":-69.44},"temporal":{"begin":"2002-01-08T00:00:00.000Z","end":"2002-01-09T23:59:59.999Z"},"quality":{"processing_level":"Not provided","version":"1","collection_progress":"COMPLETE"}}
```

Compact separators, `ensure_ascii=False`. Carries the payload verbatim — every key, every
nesting level. It is the baseline because it is what the payload *is*.

### `mat` — 272 tokens (−1.9%)

```
Amery Ice Shelf - hot water drill borehole, 2001-02 - AM01 drilling parameters data is a
NASA Earth-science dataset (short name ASAC_1164_AM01_Drilling) with CMR concept id
C1214305730-AU_AADC, archived at AU/AADC. Its DOI is doi:10.4225/15/5271A0B888E35. Data
were acquired by the CORING DEVICES instrument(s) aboard Not provided; GROUND-BASED
OBSERVATIONS; FIELD INVESTIGATION. Its primary science topic is CRYOSPHERE. Science
keywords: EARTH SCIENCE > CRYOSPHERE > SNOW/ICE. It measures Snow/Ice. Spatial coverage
spans 71.42° to 71.42° longitude and -69.44° to -69.44° latitude. Temporal coverage runs
from 2002-01-08T00:00:00.000Z to 2002-01-09T23:59:59.999Z. Quality: processing level Not
provided, version 1, status COMPLETE. AM01 borehole drilled January 2002. …
```

Deterministic prose — no LLM, so it reproduces run to run. **Carries every value, drops
all structure.** You cannot tell from the text which number was `bbox.west` versus a
figure quoted in the abstract; the model has to infer roles from the sentence. That is
the interesting property, not a defect.

### `toon` — 305 tokens (+3.2%)

```
concept_id: C1214305730-AU_AADC
title: "Amery Ice Shelf - hot water drill borehole, 2001-02 - AM01 drilling parameters data"
doi: "doi:10.4225/15/5271A0B888E35"
platforms[3]:
  - platform: Not provided
    instruments[1]: CORING DEVICES
  - platform: GROUND-BASED OBSERVATIONS
  - platform: FIELD INVESTIGATION
science_keywords[1]: EARTH SCIENCE > CRYOSPHERE > SNOW/ICE
bbox:
  west: 71.42
  …
```

Indentation-based, with array lengths declared up front (`platforms[3]`) and a tabular
form (`keywords[2]{Type,Value}:`) for uniform object lists. Carries the full structure.
Its compression advantage needs uniform repeated records; CMR collection metadata has
little of that, which is why it lands near JSON rather than well below it.

### `csv` — 302 tokens (+6.5%)

```
path,value
concept_id,C1214305730-AU_AADC
title,"Amery Ice Shelf - hot water drill borehole, 2001-02 - AM01 drilling parameters data"
platforms[0].platform,Not provided
platforms[0].instruments[0],CORING DEVICES
platforms[1].platform,GROUND-BASED OBSERVATIONS
bbox.west,71.42
…
```

Two-column `path,value` long form over the flattened tree. **This is the only lossless
CSV for a nested record.** A wide single row would either drop the repeated groups
(platforms, keywords) or explode into hundreds of sparse columns whose width is set by
the worst record in the corpus. The long form keeps the format's real cost — repeating
the full path on every row — visible instead of hidden.

### `yaml` — 297 tokens (+8.0%)

Block style, `sort_keys=False`. Carries the full structure. Note it folds long lines at
80 columns, so an abstract is split across several — semantically identical after
parsing, but it means a naive substring search for a long value fails against YAML.

### `jsonld` — 357 tokens (+21.5%)

```json
{"@context":"https://schema.org/","@type":"Dataset","identifier":"C1214305730-AU_AADC","name":"Amery Ice Shelf …","alternateName":"ASAC_1164_AM01_Drilling","sameAs":"doi:10.4225/15/5271A0B888E35","creator":{"@type":"Organization","name":"AU/AADC"},"about":{"@type":"DefinedTerm","name":"CRYOSPHERE"},"instrument":[{"@type":"Thing","name":"Not provided","hasPart":[{"@type":"Thing","name":"CORING DEVICES"}]}],"keywords":[{"@type":"DefinedTerm","inDefinedTermSet":"GCMD","termCode":"EARTH SCIENCE > CRYOSPHERE > SNOW/ICE"}],"spatialCoverage":{"@type":"Place","geo":{"@type":"GeoShape","box":"-69.44 71.42 -69.44 71.42"}},"temporalCoverage":"2002-01-08T00:00:00.000Z/2002-01-09T23:59:59.999Z", …}
```

The same facts under schema.org's vocabulary. Every facet survives, but three things are
*re-shaped* rather than dropped, and the mapping is inverted exactly by `parse_jsonld`:

| Facet | Becomes |
|---|---|
| `bbox` (4 keys) | one `GeoShape.box` string, `"south west north east"` |
| `temporal` (2 keys) | one `temporalCoverage` string, `"begin/end"` |
| `platforms` | `instrument[]` with nested `hasPart[]` typed nodes |

The renaming plus the `@context`/`@type` scaffolding and typed sub-nodes **is** what
JSON-LD costs. It is the most expensive format here by a wide margin.

---

## 4. Parity between formats — the guarantee that holds

| Format | Round-trips to an identical fact set | Notes |
|---|---|---|
| `json` | 500/500 | — |
| `csv` | 500/500 | path tokens rebuild the nesting |
| `yaml` | 500/500 | — |
| `toon` | 500/500 | official `toon_format`; byte-identical to the `@toon-format/toon` reference, 500/500 |
| `jsonld` | 500/500 | schema.org mapping inverted exactly |
| `mat` | n/a — prose cannot round-trip | 0 missing values across the corpus |

Parity is defined as equality of `(path, canonical value)` sets, comparing canonical
*strings* rather than Python types: CSV cannot distinguish `"3"` from `3`, but it carries
the same fact, and this experiment is about information content, not types. Prose is held
to the weaker but still checkable obligation of value-coverage — every canonical value
must appear verbatim in the text, so a model reading the prose has access to the same
facts as one reading the JSON.

**A caveat on how you measure this.** A naive "is every value a substring of the
rendering?" check reports differences (YAML −1,072, JSON/TOON/JSON-LD −296, CSV −79).
Those are **serialisation artifacts, not content loss**: JSON escapes `\n` inside
abstracts, YAML folds at 80 columns, CSV doubles embedded quotes. The round-trip test is
the authoritative one, and it is exact.

---

## 5. What never reaches any format

`facets()` is a projection. `airm/coverage.py` classifies all 321 distinct paths in the
corpus so the projection is explicit and enforced — a path in no bucket fails the check.

| Bucket | Values | Share | Paths |
|---|---:|---:|---:|
| **Carried** into all six formats | 13,913 | 14.3% | 17 |
| **Partial** — first element of a repeating group only | 4,029 | 4.1% | 8 |
| **Excluded** — declared, with a reason | 38,058 | 39.1% | 94 |
| **Deferred** — dataset content not carried yet | 41,312 | 42.5% | 202 |
| **Undeclared** | 0 | 0.0% | 0 |

**17,942 of 97,312 scalar values (18.4%) reach the formats.**

### Partial — the silent truncations

These look carried but keep only the first element of a repeating group:

- `bbox` — first bounding rectangle only. Polygons, points, lines, vertical domains and
  resolution are dropped entirely.
- `temporal` — first range of the first extent only. Single dates, periodic campaign
  windows and precision are dropped.
- `data_center` — first centre's `ShortName` only. Roles, long name and every subsequent
  centre are dropped.

### Excluded — deliberate, with reasons

Describes the *catalog entry* or the *publisher*, not the dataset.

| Group | Values | Why |
|---|---:|---|
| `meta.*` except `concept-id` | 20,515 | CMR bookkeeping: revision ids, native id, service capability flags, association graph edges |
| `DataCenters[].Contact*` | 9,066 | postal addresses, phone numbers, email |
| `ContactPersons[].ContactInformation` | 5,103 | personal addresses, phone numbers, email |
| `MetadataDates` | 1,553 | when the metadata was authored, not what the data covers |
| `MetadataSpecification` | 1,500 | UMM schema name/version/URL, identical for every record |

### Deferred — real content, not carried

This is a backlog, not a justification. Several of these are exactly what a search query
would hit.

| UMM field | Values | Records | What is lost |
|---|---:|---:|---|
| `RelatedUrls` | 10,804 | 448 | data-access, documentation and thumbnail URLs |
| `LocationKeywords` | 5,375 | 391 | GCMD place names — the spatial twin of `ScienceKeywords` |
| `AdditionalAttributes` | 3,146 | 212 | provider-defined attributes and descriptions |
| `ContactPersons` (names/roles) | 2,874 | 319 | investigator names and roles |
| `Projects` | 2,445 | 271 | mission and campaign names |
| `SpatialExtent` (rest) | 2,249 | 500 | granule representation, CRS, polygons, resolution |
| `AncillaryKeywords` | 1,755 | 269 | provider free-text keywords |
| `CollectionCitations` | 1,715 | 332 | how to cite the dataset |
| `DataCenters` (rest) | 1,690 | 500 | roles, long names, additional centres |
| `Platforms` (rest) | 1,583 | 500 | platform/instrument long names, type, technique |
| `PublicationReferences` | 1,198 | 44 | papers describing the dataset |
| `ArchiveAndDistributionInformation` | 1,009 | 208 | file formats, sizes, average granule size |
| `ISOTopicCategories` | 657 | 374 | ISO 19115 topic codes |
| `UseConstraints` / `AccessConstraints` | 854 | 335 / 219 | licence and access restrictions |
| `DataDates` | 862 | 237 | creation and last-revision dates |

Whole fields that reach **no** format at all, despite being plain descriptive prose:

`Purpose` (94 records) · `Quality` (96) · `CollectionDataType` (43) ·
`VersionDescription` (35) · `TemporalKeywords` (37) · `MetadataAssociations` (35) ·
`StandardProduct` (22) · `FileNamingConvention` (12) · `DataMaturity` (10) ·
`TilingIdentificationSystems` (8)

Run `uv run python -m airm.coverage --show deferred` for the itemised list.

---

## 6. Why the payload is this narrow

Two reasons, one fixed and one still open.

**Fixed — empty containers were costing tokens for nothing.** `extract_platforms` used to
emit `instruments: []` for a platform with no instruments. An empty list asserts no fact,
so `flatten()` yields nothing for it and the parity check could not see it — yet JSON,
YAML and TOON serialise the brackets while CSV, JSON-LD and prose do not. Across 318 of
747 platform entries that charged JSON 0.43%, YAML 0.40% and TOON 0.56% of their total
tokens for information-free punctuation. Fixing it also made `python-toon` agree with the
reference implementation on 500/500 records instead of 283/500.

**Narrowed — the TOON library no longer blocks a wider payload outright.** This was
recorded as a hard blocker against `python-toon` 0.1.3, whose encoder wrote a bare `-`
marker where the spec puts the item's first field on the marker line, producing output
**neither its own decoder nor the reference implementation could parse**: 0/500 full-UMM
payloads survived a round-trip.

The study now uses `toon_format` 0.9.0-beta.1, the official implementation. Full raw-UMM
documents round-trip **2,385/2,590 (92.1%)** — 169 mismatch, 36 raise `ToonDecodeError`,
with differing leaves concentrated in `SpatialExtent` (1,246), `ContactPersons` (743) and
`DataCenters` (687). So a richer content model is now a 7.9% gap to close rather than a
wall, though it is still not free. Facet-payload rendering — what this study actually
measures — is unaffected: 2,590/2,590 round-trip exactly.

So widening the content model means adopting the Node reference encoder as a runtime
dependency, and changing what the study measures. That is a design decision, which is why
the deferred content is tracked here rather than quietly folded in.

---

## 7. Cost summary

500 records, `tiktoken` `o200k_base`, paired per-record ratios against JSON.

| Format | Mean | Median | vs JSON | Paired 95% CI |
|---|---:|---:|---:|---:|
| Metadata-as-Text | 574 | 503 | −1.9% | [−2.0%, −1.8%] |
| JSON *(baseline)* | 584 | 514 | — | — |
| TOON | 599 | 527 | +3.2% | [+3.1%, +3.4%] |
| CSV | 625 | 550 | +6.5% | [+6.0%, +7.0%] |
| YAML | 635 | 546 | +8.0% | [+7.8%, +8.2%] |
| JSON-LD | 705 | 622 | +21.5% | [+20.7%, +22.4%] |

The full raw UMM record costs **1,991 tokens** (median 1,770) — 3.5× the cheapest
representation, and it is outside the comparison precisely because it carries the other
81.6% of the content documented above.

For context on how small the format effect is: moving between science domains costs more
than moving between formats. JSON averages 984 tokens for AGRICULTURE records and 434 for
CLIMATE INDICATORS — a 2.3× spread, against 1.23× across all six formats.

---

## 8. Reproducing this

```bash
uv run python -m airm.formats            # render one record in every format + parity
uv run python -m airm.coverage           # bucket every CMR path; fails on undeclared
uv run python -m airm.coverage --show deferred
uv run python -m airm.exp1               # token measurement
uv run pytest tests/test_formats.py tests/test_coverage.py
```

See `EXPERIMENTS.md` Step 0, Step 3 and Step 3a for the decisions behind all of this.
