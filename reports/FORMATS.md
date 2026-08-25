# The six formats: derivation, examples, and corpus coverage

What each of the study's six representations is, how it is derived from the raw
CMR record, what it looks like on a real record, and what it costs across the
full corpus. The running example throughout is
`C1214584243-SCIOPS` — *Large Model Outputs from the CEDAR Data Base at
NCAR/HAO* — chosen because it is a typical, compact record.

Companion documents: [`STAGED_PIPELINE.md`](STAGED_PIPELINE.md) (how the
evaluation runs), [`ANALYSIS_300Q.md`](ANALYSIS_300Q.md) (results),
[`CHUNKED_RETRIEVAL_511Q.md`](CHUNKED_RETRIEVAL_511Q.md) (retrieval study).
Coverage numbers in §6–§7 come from `runs/cache_coverage/`
(`scripts/cache_coverage.py` for sizes/tokens, `scripts/field_coverage.py`
for field-level presence).

---

## 1. The pipeline in one picture

```
data/cmr_cache/<id>.json              the record CMR actually serves ({meta, umm})
        │
        ├── airm.facets.facets() ──────────► faceted payload   (32 flat facets)
        │                                          │
        │                                          ├─ render_json    data/format_cache/json/
        │                                          ├─ render_yaml    data/format_cache/yaml/
        │                                          ├─ render_toon    data/format_cache/toon/
        │                                          ├─ render_csv     data/format_cache/csv/
        │                                          ├─ render_jsonld  data/format_cache/jsonld/
        │                                          └─ render_mat     data/format_cache/mat/
        │
        └── airm.unfaceted (whole umm tree) ─► unfaceted payload (45 top-level fields,
                                                   │              398 distinct paths)
                                                   └─ same six renderers, no projection
                                                      data/format_cache_unfaceted/*/
```

Two payloads × six formats = **12 renderings per record × 2,590 records =
31,080 cached files**, each content-fingerprinted. Content parity inside a
payload is hard-gated at cache build (§3), so any measured difference between
formats is representation, not content.

---

## 2. What a raw CMR record looks like

`data/cmr_cache/C1214584243-SCIOPS.json` is the UMM-JSON document the CMR API
returns: a `meta` envelope (CMR bookkeeping) plus the `umm` payload
(UMM-C 1.18.5).

```json
{
  "meta": {
    "revision-id": 5,
    "deleted": false,
    "format": "application/dif10+xml",
    "provider-id": "SCIOPS",
    "user-id": "mmorahan",
    "native-id": "CEDAR_LARGE_MODEL_OUTPUT",
    "concept-id": "C1214584243-SCIOPS",
    "revision-date": "2018-11-01T19:33:05.334Z", ...
  },
  "umm": {
    "ShortName": "CEDAR_LARGE_MODEL_OUTPUT",
    "EntryTitle": "Large Model Outputs from the CEDAR Data Base at NCAR/HAO",
    "TemporalExtents": [
      { "EndsAtPresentFlag": false,
        "RangeDateTimes": [ { "BeginningDateTime": "1970-01-01T00:00:00.000Z" } ] }
    ],
    "SpatialExtent": {
      "SpatialCoverageType": "HORIZONTAL_VERTICAL",
      "HorizontalSpatialDomain": {
        "Geometry": {
          "CoordinateSystem": "CARTESIAN",
          "BoundingRectangles": [
            { "WestBoundingCoordinate": -180.0, "NorthBoundingCoordinate": 90.0,
              "EastBoundingCoordinate": 180.0,  "SouthBoundingCoordinate": -90.0 } ] } },
      "VerticalSpatialDomains": [
        { "Type": "Minimum Altitude", "Value": "100 KM" }, ... ]
    },
    "ScienceKeywords": [...], "DataCenters": [...], "ContactPersons": [...],
    "MetadataDates": [...],  "RelatedUrls": [...],  "Abstract": "...", ...
  }
}
```

This record has 29 top-level UMM fields; across the corpus the raw payload
spans **45 top-level fields and 398 distinct paths**, including contact
persons, postal addresses, metadata dates, and schema bookkeeping. Depth,
single-element arrays, and CMR placeholder strings (`"Not provided"`,
`"NA"`) are all characteristic. The worst record renders to 74,528 bge
tokens — no format fits a 512-token embedding window even at the median.

---

## 3. The two payloads

### Faceted — the curated 32-field projection (`airm.facets`)

One content source for every format: `facets()` projects the UMM record onto
**32 flat facets**, each standing for one field CMR actually records:

```
concept_id, cmr_link, title, short_name, doi, agency, data_center, topic,
summary, science_keywords, variables, platform, instrument, techniques,
spectral_bands, bbox, coordinate_system, spatial_resolution, temporal,
temporal_resolution, data_format, data_volume, processing_level,
processing_level_description, version, collection_progress,
access_constraints, use_constraints, urls, urls_publication,
urls_repository, urls_additional
```

Design rules that make the downstream comparison meaningful:

- **Nothing is aggregated** — `platform` and `instrument` are separate facets,
  not nested; grouping is a modelling choice that would leak into every format.
- **Nothing is pre-rendered** — `spatial_resolution` stays `{x, y, unit}`,
  `bbox` stays four coordinates; composing strings would bake a rendering
  decision into the content.
- **Placeholders are absences** — CMR filler like `"Not provided"` (two thirds
  of all `ProcessingLevel.Id` values) asserts no fact and is dropped.
- **Numbers are canonicalised once** — `-180` prints identically in every
  format, so no format is charged tokens for a float-repr artefact.
- `cmr_link` is the one constructed field (CMR does not store it), built
  identically for every format.

Facets with no value are omitted, which is why the SCIOPS record renders ~21
of the 32.

### Unfaceted — the raw-record control (`airm.unfaceted`)

The same six renderers over the **whole** `umm` tree — every field, contacts
and addresses and metadata dates included, no projection. It exists to answer
the question a reader who distrusts the projection will ask: *does the format
ranking survive without faceting?* It is a control, not a competitor: faceted
and unfaceted renderings of the same record carry different content and are
never compared cell-for-cell inside a payload experiment.

What the projection keeps and drops is itself enforced: `airm.coverage`
classifies every scalar path in a CMR record as `CARRIED` / `EXCLUDED` (with a
stated reason) / `DEFERRED`, and fails on anything unclassified — so field
drift in CMR breaks the build instead of silently shrinking the study.

### The parity gate

Every machine-readable format has an inverse parser, and
`formats.validate_parity()` proves each rendering round-trips to exactly the
fact set it was given. Prose cannot round-trip, so `mat` is held to
**value-coverage**: every canonical value must appear verbatim in the text.
The same discipline applies to the unfaceted cache.

---

## 4. The six formats

Corpus-wide numbers below are per record, from the coverage sweep in §6
(tiktoken `o200k_base`).

### 4.1 JSON — the baseline

**Derivation.** `render_json`: the payload dict serialised compactly
(no whitespace padding), keys in canonical facet order.

```json
{"concept_id":"C1214584243-SCIOPS","cmr_link":"https://cmr.earthdata.nasa.gov/search/concepts/C1214584243-SCIOPS",
 "title":"Large Model Outputs from the CEDAR Data Base at NCAR/HAO",
 "short_name":"CEDAR_LARGE_MODEL_OUTPUT",
 "agency":"Coupling, Energetics and Dynamics of Atmospheric Regions, High Altitude Observatory, ...",
 "data_center":"UCAR/NCAR/HAO/CEDAR","topic":"ATMOSPHERE",
 "summary":"The Coupling, Energetics, and Dynamics of Atmospheric Regions (CEDAR)\nData Base holds output..."}
```

The unfaceted variant is the raw `umm` tree, same serialisation.

**Cost.** 913 tokens/record faceted, 1,918 unfaceted (2.1×).
**Standing in the experiments.** The reference every format is paired against.
Mid-pack at retrieval; under the raw payload it is the **reasoning winner for
all three models** — explicit structure appears to help when there are 398
paths to navigate.

### 4.2 YAML

**Derivation.** `render_yaml`: the same dict through a YAML dumper —
block style, indentation instead of braces, long strings wrapped.

```yaml
concept_id: C1214584243-SCIOPS
cmr_link: https://cmr.earthdata.nasa.gov/search/concepts/C1214584243-SCIOPS
title: Large Model Outputs from the CEDAR Data Base at NCAR/HAO
short_name: CEDAR_LARGE_MODEL_OUTPUT
agency: Coupling, Energetics and Dynamics of Atmospheric Regions, High Altitude Observatory,
  National Center for Atmospheric Research, UCAR
topic: ATMOSPHERE
summary: 'The Coupling, Energetics, and Dynamics of Atmospheric Regions (CEDAR)

  Data Base holds output from various large models. ...'
```

**Cost.** 971 faceted, 2,100 unfaceted (2.2×).
**Standing.** Strong faceted retriever (2nd–3rd); `gpt-5.4-nano`'s faceted
reasoning winner (0.850). Pooled reasoning effect vs JSON −0.011, on the edge
of its interval.

### 4.3 TOON — token-oriented object notation

**Derivation.** `render_toon` via the official `toon_format` encoder (pinned
to the TOON authors' repo — the PyPI name is an empty namespace reservation).
YAML-like scalars, but uniform arrays collapse into a header row plus tabular
rows, declared once: `AdditionalAttributes[2]{Value,Name,...}:`. Designed
specifically to cut LLM token cost on repetitive structures.

```toon
concept_id: C1214584243-SCIOPS
title: Large Model Outputs from the CEDAR Data Base at NCAR/HAO
summary: "The Coupling, Energetics, and Dynamics of Atmospheric Regions (CEDAR)\nData Base holds..."
```

Unfaceted, the tabular arrays earn their keep:

```toon
AncillaryKeywords[14]: AMIE,AURORAL ENERGY FLUX,CEDAR,GSWM,HAO,ION TEMPERATURE,...
CollectionCitations[1]{SeriesName,Creator,ReleasePlace,Title,Publisher}:
  CEDAR Data Base,CEDAR,"Boulder, CO",Large Model Output from CEDAR,NCAR/HAO
AdditionalAttributes[2]{Value,Name,Description,DataType}:
  "2015-12-02 12:09:57",metadata.extraction_date,Not provided,STRING
```

**Cost.** 931 faceted, 2,026 unfaceted (2.2×) — second-cheapest after prose.
**Standing.** `muse-glimmer:30b`'s faceted reasoning winner (0.864); pooled
effect vs JSON −0.008, inside the interval. Mid retrieval.

### 4.4 CSV — flattened path/value rows

**Derivation.** `render_csv`: the payload is flattened to dot-notation leaf
paths (`flatten()`, list indices as `[i]`), emitted as a two-column
`path,value` CSV. The parser reassembles the tree from the paths, so parity
still round-trips.

```csv
path,value
concept_id,C1214584243-SCIOPS
title,Large Model Outputs from the CEDAR Data Base at NCAR/HAO
topic,ATMOSPHERE
summary,"The Coupling, Energetics, and Dynamics of Atmospheric Regions (CEDAR)
Data Base holds output from various large models. ..."
```

Unfaceted, every leaf of 398 possible paths becomes a row
(`AncillaryKeywords[0],AMIE` … `CollectionCitations[0].SeriesName,…`) — the
path prefixes repeat, which is why CSV is the second-most expensive raw
rendering.

**Cost.** 984 faceted, 2,780 unfaceted (2.8× — the structure tax is in the
repeated paths).
**Standing.** The clearest format loser at reasoning: −0.027 vs JSON, the only
effect besides JSON-LD to clear its interval. Retrieval is payload-dependent —
4th faceted, last unfaceted.

### 4.5 JSON-LD — schema.org `Dataset`

**Derivation.** `render_jsonld`: the facets mapped onto schema.org vocabulary —
`identifier`, `name`, `publisher{Organization}`, `keywords[DefinedTerm]`,
`spatialCoverage{Place/GeoShape}`, `temporalCoverage`, `distribution
[DataDownload]` — with facets schema.org has no property for carried as
`additionalProperty[PropertyValue]` entries.

```json
{"@context":"https://schema.org/","@type":"Dataset",
 "identifier":"C1214584243-SCIOPS",
 "name":"Large Model Outputs from the CEDAR Data Base at NCAR/HAO",
 "publisher":{"@type":"Organization","name":"Coupling, Energetics and Dynamics of ..."},
 "about":{"@type":"DefinedTerm","name":"ATMOSPHERE"},
 "keywords":[{"@type":"DefinedTerm","inDefinedTermSet":"GCMD",
              "termCode":"EARTH SCIENCE > ATMOSPHERE > ..."}], ...}
```

Unfaceted, the vocabulary runs out almost immediately: schema.org covers about
five UMM fields (`name`, `alternateName`, `description`, `version`), and the
other forty travel as semantically inert `PropertyValue` entries keyed by path
— legal JSON-LD whose `@context` buys nothing. That is a limit of the
vocabulary, not the tree, and it is why the unfaceted JSON-LD is the largest
rendering in the corpus (the AADC record from the analysis: 24 top-level
fields faceted vs 7 top-level / 712 keys unfaceted).

**Cost.** 1,110 faceted, 4,295 unfaceted (3.9× — the wrapping tax). The most
expensive format in both payloads.
**Standing.** Worst retriever in all four conditions and −0.015 reasoning vs
JSON. Its value is web interoperability and linking, not LLM context — the
study's clearest negative result.

### 4.6 MaT — Metadata-as-Text (prose)

**Derivation.** `render_mat`: real prose written by one template per facet —
not a path dump. (An earlier iteration emitted `path is value` clauses and
mislabelled it prose; that mechanical form is preserved separately as
`render_breadcrumb` and cost the old study a factor of two.) Held to
value-coverage rather than round-trip parity: every canonical value must
appear verbatim.

```
Large Model Outputs from the CEDAR Data Base at NCAR/HAO is a NASA
Earth-science dataset (short name CEDAR_LARGE_MODEL_OUTPUT) with CMR concept
id C1214584243-SCIOPS, archived at UCAR/NCAR/HAO/CEDAR of Coupling, Energetics
and Dynamics of Atmospheric Regions, High Altitude Observatory, National
Center for Atmospheric Research, UCAR. Its CMR record is at
https://cmr.earthdata.nasa.gov/search/concepts/C1214584243-SCIOPS. Its primary
science topic is ATMOSPHERE. Science keywords: EARTH SCIENCE > ATMOSPHERE >
ATMOSPHERIC PRESSURE > ATMOSPHERIC PRESSURE MEASUREMENTS; ...
```

The unfaceted variant is prose over the *whole* record, one template per
UMM-C 1.18.5 field — possible precisely because UMM is a published schema:

```
Large Model Outputs from the CEDAR Data Base at NCAR/HAO (short name
CEDAR_LARGE_MODEL_OUTPUT, version Not provided) is a NASA Earth-science
dataset managed by UCAR/NCAR/HAO/CEDAR ... (ARCHIVER, DISTRIBUTOR). It has no
DOI (Unknown...). Cite it as Large Model Output from CEDAR, by CEDAR, series
CEDAR Data Base, published by NCAR/HAO. The data language is eng. ...
```

**Cost.** 805 faceted, 1,476 unfaceted (1.8×) — **the cheapest format in both
payloads**, because prose pays no bracket/quote/path tax.
**Standing.** First at retrieval in all four chunked conditions on all five
metrics; +0.009 vs JSON at reasoning (best pooled format effect);
`gpt-5.4-mini`'s faceted winner. Cheapest and near-best — the study's best
Pareto point.

---

## 5. Same record, twelve ways (bytes)

`C1214584243-SCIOPS` — raw record 10,319 B:

| Format | Faceted | Unfaceted | ratio |
|---|---:|---:|---:|
| MaT | 3,247 | 6,386 | 2.0× |
| JSON | 3,302 | 9,370 | 2.8× |
| TOON | 3,247 | 9,328 | 2.9× |
| YAML | 3,329 | 9,632 | 2.9× |
| CSV | 3,363 | 15,891 | 4.7× |
| JSON-LD | 3,979 | 26,514 | 6.7× |

Faceted renderings cluster within 20% of each other — the projection equalises
content, so format overhead is all that separates them. Unfaceted, the
structural taxes diverge: CSV pays in repeated paths, JSON-LD in
`PropertyValue` wrapping.

---

## 6. Corpus coverage report

From `scripts/cache_coverage.py` over both caches
(`runs/cache_coverage/coverage.json`; per-file rows in `per_record.csv`).

**Completeness.** All **2,590 records present in all 12 payload × format
directories** — 31,080 files, no gaps. (The retrieval studies index 2,584 of
these — the all-formats parity rule at index build excludes six.)

### File sizes (bytes per record)

| payload | fmt | total MB | mean | median | p95 | max |
|---|---|---:|---:|---:|---:|---:|
| faceted | mat | 8.0 | 3,084 | 2,769 | 6,430 | 31,353 |
| faceted | toon | 8.6 | 3,306 | 2,984 | 6,858 | 60,416 |
| faceted | json | 8.7 | 3,364 | 3,042 | 6,958 | 60,483 |
| faceted | yaml | 8.7 | 3,364 | 3,030 | 7,069 | 60,443 |
| faceted | csv | 9.1 | 3,504 | 3,134 | 7,299 | 67,281 |
| faceted | jsonld | 10.6 | 4,100 | 3,704 | 8,077 | 61,393 |
| unfaceted | mat | 14.3 | 5,540 | 4,849 | 11,035 | 78,244 |
| unfaceted | toon | 19.0 | 7,327 | 6,537 | 13,806 | 95,603 |
| unfaceted | json | 19.1 | 7,385 | 6,613 | 13,985 | 93,881 |
| unfaceted | yaml | 19.3 | 7,445 | 6,671 | 14,138 | 93,582 |
| unfaceted | csv | 26.6 | 10,266 | 9,151 | 18,860 | 111,137 |
| unfaceted | jsonld | 43.6 | 16,852 | 14,801 | 31,206 | 171,237 |

Faceted corpus: **53.7 MB**. Unfaceted: **141.9 MB** (2.6×).

### Tokens per record — tiktoken `o200k_base`

| payload | fmt | total | mean | median | p95 | max |
|---|---|---:|---:|---:|---:|---:|
| faceted | mat | 2,085,589 | 805 | 719 | 1,632 | 12,141 |
| faceted | json | 2,364,550 | 913 | 827 | 1,806 | 23,588 |
| faceted | toon | 2,411,283 | 931 | 846 | 1,829 | 23,616 |
| faceted | yaml | 2,514,072 | 971 | 872 | 1,948 | 24,131 |
| faceted | csv | 2,548,837 | 984 | 876 | 1,968 | 26,126 |
| faceted | jsonld | 2,876,023 | 1,110 | 996 | 2,119 | 23,832 |
| unfaceted | mat | 3,823,657 | 1,476 | 1,305 | 2,813 | 24,172 |
| unfaceted | json | 4,966,401 | 1,918 | 1,722 | 3,574 | 27,757 |
| unfaceted | toon | 5,247,824 | 2,026 | 1,810 | 3,789 | 30,088 |
| unfaceted | yaml | 5,439,509 | 2,100 | 1,878 | 3,933 | 30,088 |
| unfaceted | csv | 7,199,620 | 2,780 | 2,493 | 5,117 | 34,459 |
| unfaceted | jsonld | 11,124,819 | 4,295 | 3,787 | 7,941 | 47,947 |

The Qwen3-8B cross-check (`hf_tokens` in `coverage.json`) lands within 5–8%
of tiktoken with the identical ranking, and both track raw characters — the
differences are **length, not tokenizer vocabulary**. (The bge wordpiece
numbers in `CHUNKED_RETRIEVAL_511Q.md` §2 run higher — 786 mat faceted, 7,012
jsonld unfaceted — but rank the formats the same way.)

### Unfaceted / faceted, whole-corpus ratios

| fmt | bytes | chars | tiktoken | hf_tokens |
|---|---:|---:|---:|---:|
| mat | 1.80 | 1.80 | 1.83 | 1.86 |
| json | 2.20 | 2.20 | 2.10 | 2.06 |
| yaml | 2.21 | 2.21 | 2.16 | 2.14 |
| toon | 2.22 | 2.22 | 2.18 | 2.16 |
| csv | 2.93 | 2.93 | 2.82 | 2.78 |
| jsonld | 4.11 | 4.11 | 3.87 | 3.70 |

Prose absorbs the raw record most gracefully; JSON-LD amplifies it worst.

### Structure (measured on the JSON renderings)

| payload | top-level keys (mean / median / max) | leaf values (mean / median / max) |
|---|---|---|
| faceted | 20.7 / 21 / 31 | 39.6 / 34 / 540 |
| unfaceted | 23.5 / 23 / 35 | 149.1 / 130 / 1,818 |

The projection keeps most top-level *sections* but cuts ~73% of the *data
points* — that reduction is what the +0.045 correctness gain and the 60%
prompt-token saving ride on
([`ANALYSIS_300Q.md`](ANALYSIS_300Q.md) §3).

### The heavy tail

In every payload × format cell, p95 ≈ 2× the mean and the maximum is 20–30×
the median: a small set of giant records dominates worst-case context budgets
(the corpus worst case renders at 74,528 bge tokens). Whole-record embedding
is infeasible at the 512-token window for **every** format and payload — the
justification for the chunked index. Outliers are traceable per record in
`runs/cache_coverage/per_record.csv`.

---

## 7. Field-level coverage — present vs informative

From `scripts/field_coverage.py` (`runs/cache_coverage/field_coverage.json`;
per-field rows, including all 311 collapsed leaf paths, in
`field_presence.csv`). Every field gets **two** numbers, because CMR
conflates them:

- **present** — the source value exists in the record CMR serves, even when
  it is a placeholder (`"Not provided"`, `"NA"`, a DOI `MissingReason` stub).
- **informative** — the value survives the projection's own placeholder rule
  (`airm.facets.PLACEHOLDERS`, plus the absence-machinery keys
  `MissingReason`/`Explanation`): it asserts an actual fact.

Both columns are mechanical, not hand-classified. Faceted: the projection's
extractors run twice per raw record — once normally (informative,
cross-checked against the cached renderings) and once with the placeholder
set emptied (present) — so the columns cannot drift from what the formats
carry. Unfaceted: a field is informative when at least one leaf under it is.
The parity gate makes every number identical across all six formats of a
payload — these are payload facts, not format facts.

### Faceted — the 32 facets (2,590 records)

| Facet | present | informative | gap (records) |
|---|---:|---:|---:|
| concept_id | 100% | 100% | 0 |
| cmr_link | 100% | 100% | 0 |
| title | 100% | 100% | 0 |
| short_name | 100% | 100% | 0 |
| agency | 100% | 100% | 0 |
| data_center | 100% | 100% | 0 |
| topic | 100% | 100% | 0 |
| summary | 100% | 100% | 0 |
| science_keywords | 100% | 100% | 0 |
| variables | 100% | 100% | 0 |
| coordinate_system | 100% | 100% | 0 |
| temporal | 96.4% | 96.4% | 0 |
| urls | 86.8% | 86.8% | 0 |
| bbox | 86.8% | 86.8% | 0 |
| collection_progress | 100% | 76.3% | 613 |
| urls_publication | 68.4% | 68.4% | 0 |
| urls_repository | 68.1% | 68.1% | 0 |
| platform | 100% | 64.0% | 932 |
| use_constraints | 62.6% | 59.4% | 83 |
| instrument | 64.2% | 57.2% | 180 |
| version | 100% | 54.8% | 1,170 |
| urls_additional | 45.1% | 45.1% | 0 |
| doi | 43.7% | 43.7% | 0 |
| access_constraints | 43.6% | 35.9% | 198 |
| data_format | 48.2% | 35.3% | 334 |
| data_volume | 30.3% | 30.3% | 0 |
| processing_level | 100% | 25.6% | 1,927 |
| spectral_bands | 13.7% | 13.7% | 0 |
| processing_level_description | 12.3% | 12.3% | 0 |
| spatial_resolution | 4.7% | 4.7% | 0 |
| temporal_resolution | 2.5% | 2.5% | 0 |
| techniques | 2.4% | 2.4% | 0 |

**Per record: 22.8 of 32 facets present (71.3%) vs 20.7 informative
(64.7%).** Two facets a record, on average, are placeholder noise the
projection removes. The biggest gaps are all *required-adjacent* fields
providers must fill: `processing_level` (1,927 records of `"Not provided"`),
`version` (1,170), `platform` (932 placeholder platform names),
`collection_progress` (613), `data_format` (334). Note `doi` shows no gap
*here* because the DOI stub is structural — `{"MissingReason": …}` has no
`DOI` key for the extractor to find — the stub shows up in the unfaceted
table below instead.

### Unfaceted — the 45 top-level UMM fields

| Field | present | informative | gap | | Field | present | informative | gap |
|---|---:|---:|---:|---|---|---:|---:|---:|
| DataCenters | 100% | 100% | 0 | | DataLanguage | 55.6% | 55.6% | 0 |
| EntryTitle | 100% | 100% | 0 | | Version | 100% | 54.8% | 1,170 |
| MetadataSpecification | 100% | 100% | 0 | | DirectoryNames | 53.4% | 53.4% | 0 |
| ScienceKeywords | 100% | 100% | 0 | | Projects | 49.2% | 48.8% | 9 |
| ShortName | 100% | 100% | 0 | | DOI | 100% | 43.7% | 1,458 |
| SpatialExtent | 100% | 100% | 0 | | AdditionalAttributes | 42.8% | 42.8% | 0 |
| TemporalExtents | 100% | 100% | 0 | | DataDates | 40.5% | 40.5% | 0 |
| Abstract | 100% | 99.7% | 9 | | ArchiveAndDistributionInfo. | 39.3% | 39.0% | 9 |
| RelatedUrls | 86.8% | 86.8% | 0 | | AccessConstraints | 43.6% | 35.9% | 198 |
| MetadataDates | 82.6% | 82.6% | 0 | | ProcessingLevel | 100% | 25.6% | 1,927 |
| ISOTopicCategories | 81.4% | 81.4% | 0 | | Quality | 22.4% | 22.2% | 6 |
| LocationKeywords | 81.3% | 81.3% | 0 | | StandardProduct | 19.7% | 19.7% | 0 |
| CollectionProgress | 100% | 76.3% | 613 | | DirectDistributionInfo. | 19.4% | 19.4% | 0 |
| ContactPersons | 68.6% | 68.6% | 0 | | Purpose | 17.3% | 17.3% | 0 |
| Platforms | 100% | 68.3% | 821 | | ContactGroups | 11.6% | 11.6% | 0 |
| CollectionCitations | 67.1% | 67.1% | 0 | | MetadataAssociations | 11.0% | 11.0% | 0 |
| AncillaryKeywords | 60.6% | 60.6% | 0 | | PublicationReferences | 9.2% | 9.2% | 0 |
| UseConstraints | 62.6% | 59.4% | 83 | | CollectionDataType | 6.4% | 6.4% | 0 |
| | | | | | TemporalKeywords | 4.2% | 4.2% | 0 |
| | | | | | VersionDescription | 3.7% | 3.6% | 3 |
| | | | | | PaleoTemporalCoverages | 3.5% | 3.5% | 0 |
| | | | | | MetadataLanguage | 1.7% | 1.7% | 0 |
| | | | | | FileNamingConvention | 0.6% | 0.6% | 0 |
| | | | | | SpatialInformation | 0.6% | 0.6% | 0 |
| | | | | | DataMaturity | 0.4% | 0.4% | 0 |
| | | | | | TilingIdentificationSystems | 0.3% | 0.3% | 0 |
| | | | | | AssociatedDOIs | 0.1% | 0.1% | 0 |

**Per record: 23.5 of 45 fields present (52.2%) vs 21.0 informative
(46.8%), median 23, range 14–35 present.** The thirteen 100%-present fields
are exactly UMM-C's required set — but only seven of them are also 100%
informative. For the other six, schema compliance and information part ways:
`DOI` 100% → 43.7%, `ProcessingLevel` 100% → 25.6%, `Version` 100% → 54.8%,
`Platforms` 100% → 68.3%, `CollectionProgress` 100% → 76.3%, `Abstract`
100% → 99.7% (nine records with a placeholder abstract). **Required means
present; it does not mean informative.**

Collapsing array indices, the corpus spans **311 distinct leaf paths** within
`umm` (337 including the `meta` envelope's 26 — the 398 quoted in older
documents predates the current corpus snapshot and does not reproduce). Of
the 311, **six are never informative in any record**: `DOI.MissingReason`
(1,458 records), `DOI.Explanation` (1,432), and four
`RelatedUrls[].GetService.*` placeholders in a single record. Those first two
are pure absence machinery — ~2,900 field instances across the corpus whose
entire content is "we don't know", rendered and embedded in every unfaceted
representation.

### Two mechanisms behind the gaps

**Placeholders** account for almost everything above: required fields filled
with `"Not provided"`-class strings (`Version`, `ProcessingLevel.Id`,
platform names) or the structured DOI shrug
`{"MissingReason": "Unknown", "Explanation": "It is unknown if this record
has a DOI."}`. The projection drops them; the raw payload spends tokens on
them and embeds them into every affected vector — identical content-free
strings that make unrelated records look alike. This is why "faceted is
cheaper *and* more correct" is not a paradox: part of what it removes is
anti-signal.

**Projection scope** is the one gap running the *other* way: `temporal` is
96.4% in both columns, yet raw `TemporalExtents` is 100% informative — of
the 93 records without the facet, 92 record only `SingleDateTimes` (discrete
observation times, a `DEFERRED` path in `airm.coverage`'s audit: real content
the projection does not carry yet) and 1 has no usable range. That is a
backlog item, not a placeholder.

---

## 8. Reproducing

```bash
# rebuild / verify the caches (parity-gated)
uv run python -m airm.format_cache          # faceted
uv run python -m airm.unfaceted             # unfaceted control
uv run python -m airm.coverage              # carried/excluded/deferred audit

# the coverage sweeps behind §6 and §7
uv run python scripts/cache_coverage.py     # sizes/tokens; --no-hf skips Qwen
uv run python scripts/field_coverage.py     # field-level present vs informative
```
