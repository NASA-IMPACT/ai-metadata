# Data Generation

How every artefact under `data/` comes into existence: where the evaluation queries come
from, how NASA CMR records are fetched and cached, why `data/cmr_cache/` holds 2,590 files
when the corpus is 500, how the six format renderings are derived, and what keeps all of it
honest.

This document covers the **data-production chain only** — it stops at the point where
Experiment 1 and Experiment 2 begin consuming the artefacts. For the renderers themselves
see [`FORMATS.md`](FORMATS.md); for the chronological build log and per-step results see
[`EXPERIMENTS.md`](EXPERIMENTS.md); for the study's conclusions see
[`FINDINGS.md`](FINDINGS.md).

---

## 0. Overview

### The chain

```
sample_sme_queries.xlsx
        │
        │  airm.queries          parse · merge rows · normalise ids · drop retired
        ▼
   data/gt.jsonl ──────────────────────────────┐  (11 SME queries, 34 concept-ids)
   data/gt_dropped.json                        │
                                               │
        ┌──────────────────────────────────────┘
        │  the 34 ground-truth ids seed the corpus
        ▼
   CMR search API  ──  airm.cmr  ──►  data/cmr_cache/<concept-id>.json
   (no auth)                          2,590 raw UMM-JSON items, one per file
        │                                      │
        │  airm.corpus                         │  airm.format_cache
        │  seed · stratify · parity-gate       │  facets() → 6 renderers
        ▼                                      ▼
   data/corpus.jsonl                      data/format_cache/<fmt>/<concept-id>.<ext>
   data/corpus_report.json                data/format_cache/manifest.json
   (500 selected records)                 (2,590 × 6 = 15,540 files)
        │                                      │
        │                                      │  airm.unfaceted  (control, §6a)
        │                                      └► data/format_cache_unfaceted/<fmt>/…
        │                                         raw UMM, no projection — read by
        │                                         nothing in the main line
        │                                      │
        └──────────────┬───────────────────────┘
                       │  airm.index — one ChromaDB collection per format
                       ▼
                  data/chroma/               6 collections × 500 documents
                  data/chroma/index_report.json
                       │
                       │  airm.synth — generate, then gate on retrievability
                       ▼
                  data/queries.jsonl          SME (11) + synthetic (120) = 131
                  data/synthetic_report.json
```

### Artefacts

| Artefact | Produced by | Command | Tracked? | Current state |
|---|---|---|---|---|
| `data/gt.jsonl` | `airm.queries` | `--build` | no | 11 queries, 34 unique ids |
| `data/gt_dropped.json` | `airm.queries` | `--build` | no | 9 dropped tokens with reasons |
| `data/cmr_cache/*.json` | `airm.cmr` | side effect of any fetch | no | 2,590 files, 26 MB |
| `data/corpus.jsonl` | `airm.corpus` | `--build` | no | 500 records, 4.7 MB |
| `data/corpus_report.json` | `airm.corpus` | `--build` | no | per-topic counts, pool sizes, seed |
| `data/format_cache/**` | `airm.format_cache` | `--build` | no | 15,540 files, 69 MB |
| `data/format_cache/manifest.json` | `airm.format_cache` | `--build` | no | fingerprint `e99575eaf32a4fa5` |
| `data/format_cache_unfaceted/**` | `airm.unfaceted` | `--build` | no | 15,540 files, 179 MB — control, see §6a |
| `data/chroma/**` | `airm.index` | `--build` | no | 6 collections × 500 docs |
| `data/queries.jsonl` | `airm.synth` | (module run) | no | **not yet built** — see §8 |
| `data/synthetic_report.json` | `airm.synth` | (module run) | no | **not yet built** |

Everything above is gitignored (`.gitignore:224-240`) and derived. Only
`sample_sme_queries.xlsx` — the expert-authored input — is tracked. `data/cmr_cache.zip`
(8.2 MB) exists as a point-in-time snapshot of the raw cache.

### Why the build order is forced

Each stage is a hard dependency of the next, and the ordering is not a convenience:

1. **`airm.queries` must run first** — it needs a live CMR call to discover which
   expected concept-ids have been retired, and its surviving ids are the corpus seed.
2. **`airm.corpus` needs those ids** — ground truth is seeded before stratified sampling,
   because a corpus that does not contain the answer makes Recall@k unmeasurable.
3. **`airm.format_cache` needs `cmr_cache`** — it renders whatever raw records are on
   disk, which is a superset of the corpus.
4. **`airm.index` needs both** — the corpus decides *which* 500 records are indexed, the
   format cache supplies the exact bytes.
5. **`airm.synth` needs the index** — one of its three acceptance gates is that a
   generated query actually retrieves its source record, which cannot be checked before
   the collections exist.

---

## 1. Stage 1 — SME queries and ground truth

**Module:** `src/airm/queries.py` · **Input:** `sample_sme_queries.xlsx` ·
**Output:** `data/gt.jsonl`, `data/gt_dropped.json` ·
**Command:** `uv run python -m airm.queries --build`

The expert slice: 11 real research questions written by subject-matter experts, each
annotated with the CMR collections that should answer it. This is the held-out
human-authored half of the evaluation set.

### Reading the sheet

`read_sme_rows` (`src/airm/queries.py:75`) opens the workbook with openpyxl in read-only,
`data_only` mode and locates columns by header rather than position: exact matches on
`query` and `expected dataset`, and a substring match on `concept id`. A header it cannot
resolve raises rather than silently returning empty rows.

Each row yields `{text, dataset, ids_cell}`; rows with an empty query cell are skipped.

### Merging rows into queries

The sheet has **25 rows but only 11 distinct research questions** — it splits one question
across several rows, one per dataset family. `build_sme_queries`
(`src/airm/queries.py:118`) groups rows by query string and unions their expected
concept-ids.

This is not cosmetic. Scoring rows as queries would have counted the fire-regime question
three times and the Cyclone Mocha question four times, weighting the final metrics toward
whichever questions the expert happened to annotate most thoroughly.

### Normalising concept-id tokens

A CMR collection concept-id matches `^C\d+-[A-Z0-9_]+$` (`CONCEPT_ID_RE`,
`src/airm/queries.py:26`). Three cells in the sheet write the separator as **U+2011
NON-BREAKING HYPHEN** instead of ASCII hyphen-minus — ids that look correct to a human and
fail every string comparison.

`normalise_concept_id` (`src/airm/queries.py:54`) applies NFKC normalisation, folds all six
Unicode dash variants in `DASHES` (`src/airm/queries.py:31`) down to `-`, and strips
internal whitespace. `split_concept_ids` (`src/airm/queries.py:62`) keeps the **original
cell text alongside the normalised token**, so the dropped-token report shows what a reader
would actually find in the spreadsheet (`Not in CMR`, not `NotinCMR`).

### Two classes of drop, both recorded

| Class | What it catches | Actual values |
|---|---|---|
| `not_a_concept_id` | cells that are not ids at all | `SENTINEL-1A_SP_GRD_FULL` (a granule short-name), `Not in CMR` (a literal note) |
| `retired` | well-formed ids CMR no longer resolves | 7 ids, all `-NSIDC_ECS` |

Retirement is discovered live: `missing_concept_ids` (`src/airm/cmr.py:211`) batches every
well-formed id through CMR and returns those that come back unresolved. Per the study's
decision these are **dropped rather than remapped** to a successor collection, so ground
truth only ever names collections CMR returns today. Pass `resolve=False` to skip the
network check in offline tests.

### The actual numbers

```
rows read                : 25
distinct queries         : 11
queries kept             : 11
concept-id tokens seen   : 43 unique
  well-formed            : 41
  kept (resolve in CMR)  : 34 unique  (38 slots across queries)
dropped (not an id)      : ['Not in CMR', 'SENTINEL-1A_SP_GRD_FULL']
dropped (retired in CMR) : 7 x -NSIDC_ECS
```

**Unique ids (34) and slots across queries (38) are reported separately** because three
collections legitimately answer more than one research question. An unqualified sum would
overstate coverage by three.

No query was emptied by the retired-id drop — all 11 survive with at least one live target,
though `sme-001` and `sme-007` are down to a single id each, which is worth remembering
when reading their per-query scores. Had any query lost all its ids it would appear in
`queries_dropped_for_having_no_resolvable_ids`, not vanish.

### The loader trap

`load_queries()` (`src/airm/queries.py:231`) reads `data/gt.jsonl` — the SME half only.
`load_eval_queries()` (`src/airm/queries.py:240`) reads `data/queries.jsonl` when it exists
and falls back to `gt.jsonl` when it does not.

**Experiment 2 must go through `load_eval_queries`.** Calling `load_queries` directly would
silently run the 11 expert queries and skip the 120 synthetic ones — and every hard gate in
the study would still pass, because none of them assert a query count.

---

## 2. Stage 2 — Fetching records from CMR

**Module:** `src/airm/cmr.py` · **Output:** `data/cmr_cache/` (as a side effect of every
fetch) · **Probe:** `uv run python -m airm.cmr --concept-id C179001730-ASF`

### The API

Everything goes through the public CMR search REST API. No authentication, no key:

```
GET https://cmr.earthdata.nasa.gov/search/collections.umm_json?keyword=...
```

A **record** in this codebase is one item from the `items` array of that response: a dict
with two keys — `meta` (CMR-managed fields: `concept-id`, `provider-id`, `native-id`,
`revision-id`, `revision-date`) and `umm` (the Unified Metadata Model payload: `EntryTitle`,
`Abstract`, `ScienceKeywords`, `Platforms`, `SpatialExtent`, …).

**The whole item is cached, not a projection.** Both halves are carried through the harness
so renderers can use either; `facets()` reads `meta["concept-id"]` and everything else from
`umm`.

### Transport behaviour

`_get` (`src/airm/cmr.py:106`) wraps every request:

- **Timeout** 60 s (`_TIMEOUT`).
- **3 attempts** (`_RETRIES`), with linear backoff of 2 s × attempt (`_BACKOFF_SECONDS`).
- **Retries** on 5xx, transport errors, and undecodable JSON.
- **Re-raises 4xx immediately.** A 4xx is a request we got wrong; retrying it would only
  waste time and hide the bug.
- After exhausting attempts, raises `CMRError` carrying the last failure.

### The two write paths

Every item either function receives is written to the cache. There is no third path.

**Search** — `search_collections` (`src/airm/cmr.py:136`) takes an optional keyword plus
arbitrary extra parameters, capping `page_size` at CMR's own hard limit of 2000
(`_MAX_PAGE_SIZE`).

`search_by_topic` (`src/airm/cmr.py:161`) is the wrapper the corpus builder uses:

```python
extra_params={"science_keywords[0][topic]": topic}
```

CMR indexes the GCMD science-keyword hierarchy, so filtering on the topic field gives a
clean per-domain pool to sample from. This is *the* stratification primitive — verified
live against all 14 topics in `config.CMR_TOPICS`.

**Exact-id fetch** — `fetch_by_concept_ids` (`src/airm/cmr.py:181`) is **cache-first**:

1. Deduplicate the input while preserving order.
2. Serve every id already on disk from `load_cached`.
3. Batch the misses into chunks of `_ID_CHUNK = 50` (`src/airm/cmr.py:38`) — concept-ids
   are sent as repeated query parameters, so a 500-id request would overflow the server's
   URL length limit. 50 keeps requests short while cutting round-trips by an order of
   magnitude.
4. Cache every returned item.
5. Return results in the caller's original order.

**Ids CMR does not return are simply absent from the result.** They are not `None`, not an
error, not a log line. `missing_concept_ids` (`src/airm/cmr.py:211`) exists so callers can
*report* them explicitly — which Stage 1 does — rather than silently losing them from a
result list.

### Corpus JSONL helpers

`write_corpus` / `load_corpus` (`src/airm/cmr.py:223`, `:234`) serialise the selected
records as one JSON object per line, written atomically via a `.tmp` file. `load_corpus`
raises a message naming the build command rather than a bare `FileNotFoundError`.

---

## 3. Stage 3 — `data/cmr_cache/`

### What a file is

`data/cmr_cache/C179001730-ASF.json` is **one verbatim CMR item** — nothing derived,
nothing reshaped. The filename is the concept-id, which is simultaneously:

- the **cache key** (`_cache_path`, `src/airm/cmr.py:67`),
- the **ground-truth key** (what SME queries name and what Recall@k is measured against),
- the **format-cache key** (`format_cache.path_for`),
- the **ChromaDB document id**.

One identifier threads the entire pipeline, which is why nothing needs a mapping table.

### Atomic writes

`_write_cache` (`src/airm/cmr.py:71`) writes to `<cid>.json.tmp` and then `replace()`s it
into position. A plain `write_text` interrupted mid-write would leave a truncated file that
crashes the next `load_cached`; the rename makes publication atomic, so readers never see a
partial record.

On the read side, `load_cached` (`src/airm/cmr.py:85`) catches `JSONDecodeError` and
`OSError` and returns `None` — a corrupt file is treated as a **cache miss**, so the next
fetch simply re-downloads it. Corruption self-heals rather than propagating.

### Why 2,590 files but a 500-record corpus

This is the question the directory listing provokes, and the answer is that the two counts
measure different things:

- **`cmr_cache` holds everything the pipeline ever *examined*.**
- **`corpus.jsonl` holds only what was *selected*.**

The corpus builder pulls a candidate pool of up to 250 records per topic (§4) and samples
from it. Every candidate in every pool is written to the cache as a side effect of
`search_by_topic`, whether or not it survives sampling. With 14 topics and the real pool
sizes recorded in `data/corpus_report.json`:

```
LAND SURFACE 250 · SUN-EARTH INTERACTIONS 250 · AGRICULTURE 248 · OCEANS 245
ATMOSPHERE 243 · PALEOCLIMATE 243 · CRYOSPHERE 241 · BIOLOGICAL CLASSIFICATION 240
SOLID EARTH 239 · TERRESTRIAL HYDROSPHERE 237 · HUMAN DIMENSIONS 236
CLIMATE INDICATORS 234 · SPECTRAL/ENGINEERING 232 · BIOSPHERE 230
```

plus the 34 ground-truth seeds and overlap across pools, which lands at 2,590 distinct
records on disk. The discarded candidates cost nothing to keep and make a rebuild against
an unchanged CMR entirely offline.

### Consequence: the format cache is a superset too

Because `format_cache.build()` renders *every* concept-id in `cmr_cache` rather than only
the corpus, the rendering cache is also 2,590 wide. That is deliberate — it means parity
failures and rendering errors surface for records the corpus has not sampled yet, so
re-running the corpus build with a different seed cannot introduce a surprise.

### Caveat: revision pinning

A cache hit is keyed on **concept-id alone**. `revision-id` is stored in the file (the
opened example is at revision 2925) but never compared. A record whose revision has
advanced upstream will keep serving the old revision until the file is deleted.

This is what makes the study reproducible — the corpus does not shift under you between
runs — but it means the cache pins you to a point-in-time CMR snapshot rather than to live
metadata. To refresh a record, delete its file and re-fetch.

### Inspecting

```bash
uv run python -m airm.cmr --concept-id C179001730-ASF   # title + presence
uv run python -m airm.cmr --topic OCEANS -n 5           # live topic probe
```

---

## 4. Stage 4 — The corpus

**Module:** `src/airm/corpus.py` · **Output:** `data/corpus.jsonl`,
`data/corpus_report.json` · **Command:** `uv run python -m airm.corpus --build`

500 real CMR records, ground-truth-seeded and stratified across all 14 science-keyword
topics. `build()` (`src/airm/corpus.py:129`) runs three phases whose **ordering is not
negotiable**.

### Phase 1 — Seed with ground truth

Every concept-id the SME queries expect goes in first, via `fetch_by_concept_ids`. A corpus
that does not contain the answer makes Recall@k unmeasurable for that query, so this is
**asserted in `verify()` rather than assumed**. Current build: 34 of 34 seeded, none
missing.

### Phase 2 — Stratified fill

`_quotas()` (`src/airm/corpus.py:104`) splits the 500 slots:

- **Land Surface / Oceans / Atmosphere share half the budget equally** — the study's stated
  requirement — giving 83 each.
- The other 11 topics share the remainder equally, landing each at 22–23, well above the
  floor of `MIN_RECORDS_PER_TOPIC = 3`.

Representation of the smaller domains is the point. Sampling proportional to CMR's actual
collection counts would bury Sun-Earth Interactions (407 collections) under Oceans (20,443).

**Seeded records count against quotas, not on top of them.** `_quotas` takes the whole
corpus size, not the unfilled remainder, so a domain over-represented in ground truth is not
topped up twice and the corpus lands on exactly 500.

For each topic with an unmet quota:

1. Pull `POOL_PER_TOPIC = 250` candidates via `search_by_topic`, excluding anything already
   selected.
2. Shuffle under `RANDOM_SEED = 20260731` (`src/airm/corpus.py:39`). Sampling from a pool
   several times the quota avoids just taking whatever CMR ranked first, which would
   correlate the corpus with CMR's own relevance ordering.
3. Take candidates **only if `topic_of(record) == topic`** — CMR matches a topic search
   against *any* of a record's keywords, so a hit is not proof of domain. Filtering makes
   the quota mean what it says.

### `topic_of` — the stratification key

`topic_of` (`src/airm/corpus.py:78`) is deliberately **not** `ScienceKeywords[0].Topic`. It
upper-cases and scans *all* keywords for a canonical topic, falling back to `UNCLASSIFIED`.
Three real failure modes it absorbs, all observed in a live build:

- **Case drift.** CMR records carry both `Oceans` and `OCEANS`. As distinct keys they split
  the corpus and pushed `SPECTRAL/ENGINEERING` (1 record) and `CLIMATE INDICATORS` (2)
  below the floor, failing the hard gate for a reason that was pure formatting.
- **A later keyword holds the real topic.** Scanning all keywords recovers the domain
  instead of bucketing the record as junk.
- **Junk in the Topic field.** One record carries a NOAA department name there. Values
  matching no canonical topic become `UNCLASSIFIED` rather than inventing a domain of one
  record. After the fix, no record in the corpus is unclassified.

Two further bugs the first build exposed (it produced 468 records across 23 buckets):
quota arithmetic subtracted the ground-truth seed twice, so the build could never reach
500; and checking CMR's own `include_facets=v2` listing rather than trusting the planned
topic list caught a **missing domain — `BIOLOGICAL CLASSIFICATION`, 4,731 collections**.
The same listing shows a long tail of malformed topic values (`-Na-`, `Atmospheric`,
`Hydrosphere`) which are data-entry artefacts, not domains, and stay excluded.

### Phase 3 — The parity gate

Every candidate is run through `formats.check_record` (`src/airm/formats.py:474`). A record
that cannot be represented **identically in all six formats** would confound the very
comparison the corpus exists to support, so it leaves the corpus — and is recorded in
`report.parity_failures` with its topic, its ground-truth status, and the per-format
detail. It is excluded *loudly*, never dropped.

Records are then sorted deterministically: ground truth first in query order, then by
concept-id.

### Hard gates

`verify()` (`src/airm/corpus.py:225`) returns violations; a non-empty list exits 1:

- corpus size is exactly the expected size — **this gate exists because quota shortfalls
  can leave the corpus short while every topic still clears its floor**, and without it
  that build would report success;
- every ground-truth id is present;
- no expected topic is absent entirely;
- no topic falls below `MIN_RECORDS_PER_TOPIC = 3`;
- no ground-truth record failed content parity.

### The current build

```
corpus size              : 500
seeded from ground truth : 34 / 34
parity failures excluded : 0
quota shortfalls         : none

ATMOSPHERE 83 · OCEANS 83 · LAND SURFACE 83
AGRICULTURE 23 · BIOSPHERE 23 · CRYOSPHERE 23 · CLIMATE INDICATORS 23
HUMAN DIMENSIONS 23 · BIOLOGICAL CLASSIFICATION 23 · SUN-EARTH INTERACTIONS 23
SOLID EARTH 23 · TERRESTRIAL HYDROSPHERE 23 · PALEOCLIMATE 22 · SPECTRAL/ENGINEERING 22
```

The three required domains are exactly equal, every other domain clears the floor, and
**zero of the 500 records failed content parity**.

---

## 5. Stage 5 — The canonical payload

**Module:** `src/airm/facets.py`

This is the hinge of the whole study, so it belongs in the data-generation story even
though it produces no file of its own: it is what the format cache renders.

### One content source

`facets(record)` projects a CMR record onto a fixed set of **32 flat fields** in the order
given by `FACET_KEYS` (`src/airm/facets.py:67`):

| Group | Facets |
|---|---|
| identity | `concept_id` · `cmr_link` · `title` · `short_name` · `doi` |
| provenance | `agency` · `data_center` |
| science | `topic` · `summary` · `science_keywords` · `variables` |
| instrumentation | `platform` · `instrument` · `techniques` · `spectral_bands` |
| space | `bbox` · `coordinate_system` · `spatial_resolution` |
| time | `temporal` · `temporal_resolution` |
| distribution | `data_format` · `data_volume` |
| quality | `processing_level` · `processing_level_description` · `version` · `collection_progress` |
| rights | `access_constraints` · `use_constraints` |
| links | `urls` · `urls_publication` · `urls_repository` · `urls_additional` |

**Every renderer consumes that dict and nothing else. Nothing may read the raw record.**
That is what makes the comparison a *format* comparison rather than a content comparison —
a full nested UMM tree versus a short prose summary differ in *which facts are present*, so
any measured difference between them would be uninterpretable.

### One facet, one UMM field

Two rules keep the payload a set of *fields* rather than a data model:

- **Nothing is aggregated.** `platform` and `instrument` are separate facets rather than
  instruments nested inside platforms, and the four quality fields are four facets rather
  than one `quality` dict. A grouping is a modelling choice, and a modelling choice baked
  into the content makes the comparison measure something other than format.
- **Nothing is rendered.** Where UMM records structure, the facet keeps it:
  `spatial_resolution` is `{x, y, unit}`, not the string `"30 x 30 Meters"`; `data_volume`
  is `{size, unit, basis}`. `bbox` was always the precedent — four coordinate fields, not a
  composed string.

`cmr_link` is the single deliberate exception: constructed from the concept-id, because CMR
does not store it (only 140 of 2,590 records carry a cmr.earthdata URL), and constructed
identically for every format.

### Placeholders are absences

CMR fills required fields it has no value for with literal placeholders — `"Not provided"`,
`"Not applicable"`, `"NA"`. Two thirds of `ProcessingLevel.Id` values are `"Not provided"`.
`_clean` (`src/airm/facets.py:144`) drops them, so the facet is omitted like any other
empty value. Carrying them would put the same content-free string into every
representation and into the embedding of every record that has one — which is why
`processing_level` populates 25.6% of records rather than the 100% a presence check
reports.

### Facet population

Share of the 2,590 cached records carrying each facet:

| ≥99% | 57–96% | 25–45% | <14% |
|---|---|---|---|
| `concept_id` `cmr_link` `title` `short_name` `agency` `data_center` `topic` `summary` `science_keywords` `variables` `coordinate_system` | `temporal` 96.4 · `bbox` 86.8 · `urls` 86.8 · `collection_progress` 76.3 · `urls_publication` 68.4 · `urls_repository` 68.1 · `platform` 64.0 · `use_constraints` 59.4 · `instrument` 57.2 | `version` 54.8 · `urls_additional` 45.1 · `doi` 43.7 · `access_constraints` 35.9 · `data_format` 35.3 · `data_volume` 30.3 · `processing_level` 25.6 | `spectral_bands` 13.7 · `processing_level_description` 12.3 · `spatial_resolution` 4.7 · `temporal_resolution` 2.5 · `techniques` 2.4 |

Because empty facets are omitted, the sparse ones cost nothing on records that lack them.

**`spectral_bands` is the one heuristic facet.** UMM-C has no wavelength field, so values
are taken verbatim from `ComposedOf[].ShortName`, `OperationalModes`, spectral-named
`Characteristics`, and matching science-keyword levels. The top values are genuine
(`VISIBLE WAVELENGTHS` ×127, `INFRARED WAVELENGTHS` ×88, `MICROWAVE` ×58) but the tail
carries noise: `Arctic`/`Antarctic` ×14 each are AIRSAR operational modes, and
`CLOUD FREQUENCY` ×31 matches on "frequency". Roughly four fifths signal. Every other facet
reads a field.

### Decisions that exist to stop format artefacts masquerading as findings

- **Empty facets are omitted, not emitted as `null`.** A format that renders `"bbox":null`
  and one that omits the key carry the same information and very different token counts.
- **Numbers are canonicalised at extraction time.** `canonical_coord`
  (`src/airm/facets.py:189`) stores an integral coordinate as an `int`, and `num_str`
  (`src/airm/facets.py:176`) is the single printing helper. Without this, Python's float
  `repr` would charge JSON, YAML and TOON three characters for the `.0` in `-180.0` while
  prose printed `-180` — a pure serialisation artefact appearing as a token-cost difference
  in a study about token cost. (`num_str` handles `bool` before `int` because `bool`
  subclasses `int` and would otherwise print as `1`/`0`.)
- **A partial bounding box is dropped entirely** (`extract_bbox`,
  `src/airm/facets.py:470`) rather than emitted as a half-fact some formats would render
  and others would not.
- **Empty lists never reach a renderer.** `platform` and `instrument`
  (`src/airm/facets.py:371`, `:383`) are omitted rather than emitted as `[]`. An empty list
  asserts no fact, so `flatten` yields nothing for it and the parity check cannot see it —
  but JSON, YAML and TOON still *serialise* it. Under the previous nested shape, 318 of 747
  platform entries charged those three formats 0.4–0.6% of their total tokens for
  information-free punctuation.
- **Science keywords are joined strings, not objects** (`extract_science_keywords`,
  `src/airm/facets.py:309`). A structured form has to survive JSON-LD flattening into a
  `DefinedTerm.termCode`, and a record omitting a middle level cannot then be reassembled
  unambiguously — so parity would fail on a rendering artefact rather than a real loss of
  content. `"EARTH SCIENCE > LAND SURFACE > ..."` is also GCMD's own canonical notation.

### The parity currency

- `flatten` (`src/airm/facets.py:799`) reduces a payload to `(path, scalar)` pairs in
  document order — `platforms[0].instruments[1]`, `bbox.west`.
- `fact_set` (`src/airm/facets.py:824`) turns those into a set of `(path, canonical
  string)` pairs. **Parity is defined as equality of these sets.** Comparing canonical
  *strings* rather than raw Python values is deliberate: CSV cannot distinguish the string
  `"3"` from the integer `3`, but it carries the same fact, and the experiment is about
  information content, not Python types.
- `fact_values` (`src/airm/facets.py:835`) drops the paths, for the prose value-coverage
  check — prose cannot round-trip, so `mat` is instead required to contain every canonical
  value verbatim (`missing_values_in_prose`, `src/airm/formats.py:416`).

See [`FORMATS.md`](FORMATS.md) for what each of the six renderers does with this payload.

---

## 6. Stage 6 — `data/format_cache/`

**Module:** `src/airm/format_cache.py` · **Output:** `data/format_cache/` ·
**Commands:** `--build`, `--build --force`, `--verify`, `--show <CONCEPT_ID>`

Where `cmr_cache` holds the raw record, this holds **one file per (record, format)**:

```
data/format_cache/json/C1234-PROV.json
data/format_cache/csv/C1234-PROV.csv
data/format_cache/yaml/C1234-PROV.yaml
data/format_cache/toon/C1234-PROV.toon
data/format_cache/jsonld/C1234-PROV.jsonld
data/format_cache/mat/C1234-PROV.txt
```

Extensions come from `EXTENSIONS` (`src/airm/format_cache.py:51`). `mat` gets `.txt`
deliberately — it is prose, and a suffix implying a parseable structure it does not have
would be a lie to every tool that opens it.

### How a build runs

`build()` (`src/airm/format_cache.py:150`):

1. Compute the current `fingerprint()` (below). If it differs from the manifest's, set
   `force = True` — every file on disk came from different code and none may be trusted.
2. Enumerate concept-ids via `cached_concept_ids()` (`src/airm/format_cache.py:104`), a
   glob of `C*.json` over `cmr_cache`. **This is the source of truth for what gets
   rendered** — the corpus does not enter into it.
3. For each id: skip if not forcing and all six files exist; otherwise load the raw record,
   call `facets()` **once**, render all six formats from that single payload, and write each
   atomically (`_write`, `src/airm/format_cache.py:109`, same temp-file-plus-rename
   discipline as `cmr_cache` — a truncated CSV or prose file parses as valid-but-wrong,
   which is worse than a crash).
4. A record that raises is appended to `render_errors` and the build continues. One bad
   record must not stop 2,589 good ones.
5. Optionally check parity and record failures.
6. Write `manifest.json`.

### The staleness guard

A rendering cache's characteristic failure is outliving the code that produced it. This is
closed off, not documented away.

`fingerprint()` (`src/airm/format_cache.py:72`) is a SHA-256 over the **source bytes** of
`facets.py` and `formats.py`, truncated to 16 hex characters. Those two modules are the
complete set of code that can change a rendering — `facets` decides *what* is rendered,
`formats` decides *how*, and nothing else touches the output. The digest is stored in the
manifest (currently `e99575eaf32a4fa5`).

Any edit to either module changes the fingerprint, and then:

- `load()` / `load_guard()` (`src/airm/format_cache.py:261`, `:291`) raise
  `StaleCacheError` naming the rebuild command, rather than serving bytes from older code;
- `build()` treats every file as suspect, so `"file exists"` can never mean `"exists but
  was rendered by older code"`;
- `verify()` reports the cache as stale.

`load(..., strict=False)` exists only to inspect an old cache deliberately. **An experiment
must never pass it** — the entire guarantee of the cache is that its bytes match its code.

### Nothing depends on it

`render_many` (`src/airm/format_cache.py:307`) and `render_all_cached`
(`src/airm/format_cache.py:329`) check `is_current()` **once per batch** and fall back to
live rendering per record when the cache is absent, stale, or missing that id. The fallback
renders the same payload through the same code that filled the cache, so:

> A missing, stale or partial cache costs time, never correctness.

This was verified over 200 records: Experiment 1's 1,400 measured rows and all six of the
index's document lists are byte-identical either way. A cache that could change a result
would be a liability, not an optimisation.

### Verification checks both directions

`verify()` (`src/airm/format_cache.py:348`) compares each format directory against
`cmr_cache`:

- **A missing rendering** would silently shrink an experiment.
- **An extra rendering** means a concept-id was dropped from `cmr_cache` and left its
  rendering behind, which would let a stale record back into a run.

Records listed in the manifest's `render_errors` are excluded from the "missing" set — they
are expected to be absent, and are reported separately.

```bash
uv run python -m airm.format_cache --verify
# → cache is one-to-one with cmr_cache and current with the rendering code
```

### Cost and payoff

2,590 records × 6 formats = **15,540 files, 69 MB**. A full build takes ~14 s; an unchanged
rebuild ~1.5 s.

Speed is the *weakest* of the three reasons to have this cache. Cached reads beat live
rendering only ~2.5× (0.44 s → 0.17 s for 500 records × 6 formats), which alone would not
justify the module. The other two matter more:

- **Inspectability.** The renderings are `diff`-able artefacts, so *why* a format costs what
  it costs is legible without running Python. `--show <CONCEPT_ID>` prints all six.
- **A stable referent.** Every token count, retrieval hit and LLM answer traces back to
  exact bytes on disk rather than to a function call that has to be re-run to be inspected.

### Caveat: manifest parity only reflects re-rendered records

Parity is evaluated inside the per-record loop, *after* the skip check. A record skipped as
unchanged contributes no parity entry.

So an incremental no-op rebuild rewrites the manifest with `skipped_unchanged: 2590` and
`parity_failures: []` — which is exactly the current on-disk state — even though the true
full-render result is **2,589 / 2,590**, the single failure being `yaml` on
`C1214595237-SCIOPS`. (That record carries mojibake in its abstract — double-encoded UTF-8
for "Åkerman" — containing U+0085 NEL, which YAML treats as a line break and normalises to
a space on load. The parity gate catching it and excluding the record is correct
behaviour.)

**Reading parity out of the manifest requires `--build --force`.** An empty
`parity_failures` after an incremental build means "nothing was checked", not "nothing
failed".

---

## 6a. The unfaceted control — `data/format_cache_unfaceted/`

**Module:** `src/airm/unfaceted.py` · **Output:** `data/format_cache_unfaceted/` ·
**Commands:** `--build`, `--build --force`, `--verify`, `--show <CONCEPT_ID>`

Stage 6 renders the 32-facet projection. This renders **the record CMR actually serves** —
all 45 top-level UMM fields, contacts, addresses, metadata dates, the lot — in the same six
formats, judged by the same parity standard.

It is a **control, not a competitor.** Nothing in the study's main line reads it: the
corpus, the index and both experiments go through `format_cache`. Faceted and unfaceted
renderings are *not* comparable record-for-record, because they carry different content —
which is precisely why the faceted payload exists.

### Why it exists

Experiment 1 reports a `json_umm` reference — what the status quo costs — but only in JSON.
That answers "how much does faceting save?" and not "does the format ranking survive
without faceting?". The second is the question a reader who distrusts the projection will
ask, and it needs all six formats over the raw record. `unfaceted.payload()` is deliberately
trivial: it returns `record["umm"]` untouched. No projection, no placeholder filtering, no
number canonicalisation — nothing decided on the reader's behalf.

`meta` is excluded: it is CMR's bookkeeping (revision id, ingest dates) rather than metadata
about the dataset, and including it would put the concept-id into some renderings and not
others.

### What the control contains

Per record, against the faceted payload:

| | mean leaves | median | max | distinct paths |
|---|---:|---:|---:|---:|
| **Unfaceted** | 149 | 130 | 1,818 | 311 |
| **Faceted** | 40 | 34 | 540 | 41 |

**Faceting keeps 26.6% of the leaf values and 13.2% of the distinct paths.** The other
three quarters are overwhelmingly contact blocks, postal addresses, metadata dates and
citation apparatus — real metadata, but not about what the data measures.

Top-level UMM field coverage across the 2,590 records, for the fields the facets do *not*
carry:

```
MetadataSpecification 100%  ·  MetadataDates 82.6%  ·  ISOTopicCategories 81.4%
LocationKeywords 81.3%  ·  ContactPersons 68.7%  ·  CollectionCitations 67.1%
AncillaryKeywords 60.6%  ·  DataLanguage 55.6%  ·  DirectoryNames 53.4%
Projects 49.2%  ·  AdditionalAttributes 42.8%  ·  DataDates 40.5%
Quality 22.4%  ·  StandardProduct 19.7%  ·  DirectDistributionInformation 19.4%
Purpose 17.3%  ·  ContactGroups 11.6%  ·  MetadataAssociations 11.0%
PublicationReferences 9.2%  ·  … 26 more below 7%
```

### Two formats need a schema and do not have one

Four of the six are schema-agnostic: JSON, YAML, TOON and CSV serialise an arbitrary tree
and read it back. The other two cannot, and their degradation is the most informative thing
the control produces.

- **JSON-LD** has schema.org properties for **five** UMM fields (`EntryTitle`→`name`,
  `Abstract`→`description`, `ShortName`→`alternateName`, `Version`→`version`,
  `DOI.DOI`→`identifier`) and nothing for the other forty. Everything else can only travel
  as `PropertyValue` entries keyed by flattened path — legal JSON-LD, and semantically
  inert. The `@context` buys nothing when every term is a literal path string.
- **Metadata-as-Text cannot be prose.** Writing a sentence about a field requires knowing
  what the field *means*; over 311 arbitrary paths there is no such knowledge, so the only
  faithful rendering is one `path is value` clause per leaf. It passes the value-coverage
  parity check and reads like a database dump, because that is what it is.

Both are rendered anyway and honestly labelled. "This format needs a schema, and that schema
is exactly what faceting provides" is a finding, not an obstacle.

### The headline result: the ranking inverts

Same 500 corpus records, tiktoken `o200k_base`, each payload compared against **its own**
JSON baseline:

| Format | faceted | vs JSON | unfaceted | vs JSON | rank |
|---|---:|---:|---:|---:|:--|
| Metadata-as-Text | 782 | **−11.1%** | 3,115 | **+58.0%** | ↓ 4 places |
| JSON | 897 | +0.0% | 1,991 | +0.0% | ↑ 1 |
| TOON | 917 | +2.5% | 2,098 | +5.6% | ↑ 1 |
| YAML | 955 | +6.0% | 2,181 | +9.3% | ↑ 1 |
| CSV | 962 | +6.1% | 2,873 | +45.3% | ↑ 1 |
| JSON-LD | 1,078 | +21.7% | 4,421 | **+124.2%** | — |

```
faceted   : mat < json < toon < yaml < csv < jsonld
unfaceted : json < toon < yaml < csv < mat < jsonld
```

**Prose goes from cheapest to second-dearest.** Its advantage is not a property of prose —
it is a property of prose *over a known schema*. Given 13 fields it amortises field names
into words it would use anyway; given 311 arbitrary paths it must name every one of them
explicitly, and it pays the full path string on every leaf.

Two corollaries worth stating:

- **JSON-LD's penalty scales with unmodelled fields**, from +21.7% to +124.2%. Its overhead
  is per-node scaffolding, so it grows with the number of nodes — and when schema.org models
  5 of 45 fields, almost every node is a `PropertyValue` wrapper.
- **CSV degrades the same way and for the same reason** (+6.1% → +45.3%): its cost is the
  full path repeated on every row, and unfaceted paths are far longer
  (`SpatialExtent.HorizontalSpatialDomain.Geometry.BoundingRectangles[0].WestBoundingCoordinate`).
- **TOON and YAML barely move** (+2.5%→+5.6%, +6.0%→+9.3%). Their overhead is genuinely
  structural, not per-field, so it survives a 3.7× increase in content.

The unfaceted JSON figure of **1,991 tokens matches Experiment 1's independently-computed
`json_umm` reference exactly**, which is the cross-check that the control and the reference
describe the same object. A test pins it (`test_json_rendering_matches_exp1_raw_umm_reference`).

### Round-trip parity: the control is graded, not gated

`parity_report` (`src/airm/unfaceted.py`) is identical in shape to the faceted one, so both
caches are held to the same standard. Full-record round-trip over 2,590 records:

| Format | failures | rate |
|---|---:|---:|
| JSON, CSV, JSON-LD, MAT | 0 | 0.0% |
| YAML | 1 | 0.0% |
| **TOON** | **206** | **8.0%** |

TOON's 206 breaks down as 170 value mismatches and 36 `ToonDecodeError`s, concentrated in
`SpatialExtent`, `ContactPersons` and `DataCenters` — corroborating the 7.9% figure recorded
independently in `EXPERIMENTS.md` Step 0.

**Parity failure is not a build failure here.** `verify()` deliberately does not gate on it:
TOON's inability to round-trip every raw record is a *measured property of the control*,
whereas in the faceted cache a parity failure costs a record its place in the corpus. The
rate is reported per format in the manifest as `parity_by_format`.

### Fingerprint: a different guard

`fingerprint()` hashes `formats.py` + `unfaceted.py` — **not `facets.py`**. The control reads
no facet code, so an edit to the projection has no bearing on its bytes and must not
invalidate it. That is the opposite of `format_cache`, which must invalidate on exactly that
edit. Current digest: `46c7eefcea76406e`.

### Cost

2,590 records × 6 formats = 15,540 files, **179 MB** (against 79 MB faceted). A full build
takes ~38 s. Per format: JSON-LD 47 MB, MAT 33 MB, CSV 31 MB, JSON/YAML/TOON 23 MB each —
the disk footprint tells the same story as the token counts.

```bash
uv run python -m airm.unfaceted --build      # render all six, unfaceted
uv run python -m airm.unfaceted --verify     # one-to-one with cmr_cache
uv run python -m airm.unfaceted --show C179001730-ASF
```

---

## 7. Stage 7 — The index

**Module:** `src/airm/index.py` · **Output:** `data/chroma/`,
`data/chroma/index_report.json` · **Command:** `uv run python -m airm.index --build`

Included here because Stage 8 cannot run without it.

Six ChromaDB collections (`cmr_json`, `cmr_csv`, …) over the **same** 500 corpus records,
the **same** embedding model, the **same** one-document-per-record chunking. The only thing
that varies is how the record is written down — which is the point.

Documents come through `format_cache.render_many`, so the index reads the same bytes the
token counts were computed from. Collections are created with `{"hnsw:space": "cosine"}` to
match the normalised embeddings; Chroma's default L2 would rank the same vectors
differently.

### The truncation confound

The planned embedding model, `BAAI/bge-small-en-v1.5`, has a **512-token window**, and the
rendered records overflow it *unevenly*:

| Format | Documents over the 512-token window |
|---|---:|
| JSON-LD | 444 / 500 (89%) |
| JSON | 326 / 500 (65%) |
| CSV | 253 / 500 (51%) |
| TOON | 219 / 500 (44%) |
| YAML | 213 / 500 (43%) |
| Metadata-as-Text | 205 / 500 (41%) |

Every collection would still have reported 500 documents and produced plausible Recall@k —
while quietly clipping the verbose formats roughly twice as often as the terse one. The
measured "format effect" would have been substantially a truncation effect, aimed at exactly
the formats under test.

The longest rendering in the corpus is 6,325 tokens, so the embedder needs an 8k window.
`Alibaba-NLP/gte-modernbert-base` (8192 tokens, 768 dims) was chosen over `nomic-embed-text-v1.5`
and `jina-embeddings-v2-small-en` because it is the only one of the three that loads
**without `trust_remote_code`** — no remote code execution for a model pulled at build time.

`MAX_EMBED_TRUNCATIONS = 0` makes this a hard gate: `verify()` (`src/airm/index.py:224`)
fails the build on a **single** truncated document, measured with the embedding model's own
tokenizer (`count_wordpieces`, `src/airm/index.py:114`), not tiktoken.

### Other gates and current state

`verify()` also requires each collection to hold exactly the corpus size and every
ground-truth record to be present in **every** collection — a collection missing one makes
Recall@k unmeasurable for each query that expects it.

```json
{ "embed_model": "Alibaba-NLP/gte-modernbert-base",
  "max_sequence_tokens": 8192,
  "counts":    { "json": 500, "csv": 500, "yaml": 500, "toon": 500, "jsonld": 500, "mat": 500 },
  "truncated": { "json": 0,   "csv": 0,   "yaml": 0,   "toon": 0,   "jsonld": 0,   "mat": 0 },
  "ground_truth_missing": { "…": [] } }
```

`EMBED_BATCH = 4` rather than sentence-transformers' default of 32: documents run to ~6,300
tokens and attention is quadratic in that, so a batch of 32 asks MPS for a >10 GiB
allocation and dies. Throughput barely suffers because the library sorts by length
internally. `_release_accelerator_memory()` drops cached device memory between batches,
which otherwise accumulates over 3,000 documents until an allocation fails.

Probe a live query across all six collections:

```bash
uv run python -m airm.index --probe "sea ice thickness in the Kara Sea"
```

---

## 8. Stage 8 — Synthetic queries

**Module:** `src/airm/synth.py` · **Output:** `data/queries.jsonl`,
`data/synthetic_report.json` · **Command:** `uv run python -m airm.synth`

> **Status: built and unit-tested (24 tests against a stubbed provider), not yet run.**
> `data/queries.jsonl` does not currently exist, so `load_eval_queries()` presently returns
> the 11 SME queries alone.

120 queries generated **from indexed records**, so their ground truth is in-corpus by
construction — the failure mode that cost the expert slice 7 targets cannot occur here.

### Quota and sampling

`config.synthetic_quota()` gives 30 each to Land Surface, Oceans and Atmosphere (the
equal-representation requirement) and spreads the remaining 30 over the other 11 topics at
2–3 each, totalling exactly 120.

Corpus records are grouped by `topic_of` and shuffled under `RANDOM_SEED = 20260731`
(`src/airm/synth.py:50`). One query is attempted per sampled record until the topic quota is
met. If a topic's pool is smaller than its quota, that is recorded as a drop with
`available` and `requested` counts before generation even starts.

### The prompt

`GENERATION_SYSTEM` (`src/airm/queries.py:279`) instructs the model to write a question a
domain scientist would plausibly ask, with two hard constraints:

- **A research question, not a lookup.** *"How did sea-ice thickness in the Kara Sea change
  each March from 2020 to 2025?"* is good; *"Find the CryoSat-2 sea ice product"* is not.
- **Never name the dataset**, its short name, DOI, concept id, version, or data centre. The
  question must be answerable only by understanding what the data measures.

`_generation_prompt` (`src/airm/queries.py:300`) renders the *facet payload* — not the raw
record — as title, topic, keywords, measures, instruments, spatial and temporal extent, and
the first 900 characters of the abstract. Calls go through `airm.llm.ask` at
`temperature=0.7` with `json_mode=True`, logged to `logs/llm/<run_id>/query_gen.jsonl` under
`PURPOSE_QUERY_GEN`.

### Three acceptance gates

Each candidate gets up to `MAX_QUERY_REGEN_ATTEMPTS = 3` attempts, and must clear all
three. What each one prevents:

**1. It parses.** `parse_generated` (`src/airm/queries.py:375`) tolerates fenced blocks and
chatty preambles, falling back to a brace scan. Local models wrap JSON, and that is not a
generation failure. Unparseable responses are retried.

**2. It leaks no identifiers.** `_leaked_identifiers` (`src/airm/queries.py:339`) rejects a
query containing the concept id, short name or DOI as a substring, **or** sharing a
5-consecutive-word run (`TITLE_NGRAM = 5`) with the title. A query naming an identifier
turns retrieval into string matching: every format contains that literal, so every format
scores alike and the effect under measurement disappears.

> The leak detector had a real bug the tests caught. It built title n-grams from
> *stopword-filtered* words but matched them against the *raw* query, so
> `"ATLAS ICESat Land and Vegetation Height"` never matched the filtered
> `"ATLAS ICESat Land Vegetation Height"` — meaning a title quoted **with** its stopwords,
> the likeliest way a model quotes one, sailed straight through. Both sides are now
> tokenised identically by `_words`.

**3. It is retrievable.** `_retrievable` (`src/airm/synth.py:76`) requires the source record
to appear in the top-`TOP_K` (10) for **at least one** of the six formats. A query no
representation can answer measures nothing; keeping it would add noise to every cell
equally. This is the gate that forces Stage 7 to run first.

### Failure handling

Candidates failing three times are **dropped and reported** with a reason
(`leaked identifiers` with the detail, `not retrievable by any format`, or `llm error`) —
never silently replaced with another record, because a quietly rebalanced quota is a quietly
biased query set. `verify()` (`src/airm/synth.py:220`) returns a violation for any topic
that produced fewer queries than requested.

### Output

`build()` (`src/airm/synth.py:203`) writes the **combined** set — `load_queries()` (SME) +
synthetic — to `data/queries.jsonl`, plus `data/synthetic_report.json` with per-topic
requested/produced counts, retry count, and every drop. Query ids are `sme-NNN` and
`syn-NNN`; each `Query` carries `source`, `topic`, `expected_concept_ids` and
`expected_titles`.

A smoke run with a reduced quota:

```bash
uv run python -m airm.synth --provider ollama --model qwen3.6:latest --limit 2
```

---

## 9. Reproducing everything from scratch

```bash
# 0. dependencies (installs ai-metadata editable, src layout)
uv sync

# 1. SME queries + ground truth               [needs network: retirement check]
uv run python -m airm.queries --build
#    → data/gt.jsonl (11), data/gt_dropped.json
#    check: "queries kept : 11", "kept (resolve in CMR) : 34 unique"

# 2. Corpus                                   [needs network: topic pools]
uv run python -m airm.corpus --build
#    → data/corpus.jsonl (500), data/corpus_report.json
#    → also populates data/cmr_cache/ as a side effect
#    check: "all hard gates passed"; 83/83/83 on the primary topics

# 3. Format cache                             [offline]
uv run python -m airm.format_cache --build
#    → data/format_cache/{json,csv,yaml,toon,jsonld,mat}/, manifest.json
#    check: "cache is one-to-one with cmr_cache and current with the rendering code"
#    note: use --build --force if you need trustworthy parity_failures in the manifest

# 3a. Unfaceted control                      [offline, optional — read by nothing]
uv run python -m airm.unfaceted --build
#    → data/format_cache_unfaceted/, manifest.json
#    check: TOON round-trip failures ~8%; every other format 0

# 4. Indexes                                  [offline, downloads the embedder once]
uv run python -m airm.index --build
#    → data/chroma/, data/chroma/index_report.json
#    check: 500 documents per collection, 0 over the window

# 5. Synthetic queries                        [needs an LLM provider]
uv run python -m airm.synth
#    → data/queries.jsonl (131), data/synthetic_report.json
#    check: 120 synthetic, quota met per topic, no shortfalls
```

### Inspection and verification

```bash
uv run python -m airm.cmr --concept-id C179001730-ASF      # one raw record
uv run python -m airm.cmr --topic OCEANS -n 5              # live topic probe
uv run python -m airm.format_cache --show C179001730-ASF   # all six renderings
uv run python -m airm.format_cache --verify                # mapping + freshness
uv run python -m airm.unfaceted --show C179001730-ASF      # the same record, unfaceted
uv run python -m airm.formats sample_data/umm_json.json    # render + parity one file
uv run python -m airm.index --probe "<a query>"            # top-3 per collection
```

### Tests

The everyday suite is offline and fast — `pyproject.toml` sets
`addopts = "-m 'not live'"`, so network tests are deselected by default:

```bash
uv run pytest              # offline suite
uv run pytest -m live      # network tests (real CMR calls)
```

---

## 10. Determinism, provenance and known caveats

### What is fixed

| Mechanism | Where | What it pins |
|---|---|---|
| `RANDOM_SEED = 20260731` | `corpus.py:39`, `synth.py:50` | which records are sampled per topic, in what order |
| `POOL_PER_TOPIC = 250` | `corpus.py:44` | how large the candidate pool is before sampling |
| Concept-id keyed cache | `cmr.py:67` | the exact CMR revision of every record |
| `fingerprint()` | `format_cache.py:72` | that renderings match the code that produced them |
| `fingerprint()` | `unfaceted.py` | control renderings; hashes `formats.py` only, **not** `facets.py` (§6a) |
| `EMBED_MODEL`, `EMBED_BATCH` | `config.py:163`, `index.py:81` | the embedding is never an independent variable |
| Pinned `toon-format` commit | `pyproject.toml:51` | TOON encoding, byte-for-byte |
| `FACET_KEYS` order | `facets.py:67` | key ordering never becomes a hidden variable |

### What is asserted rather than hoped for

Each build stage exits non-zero on its own hard gates: corpus size / ground-truth presence /
topic floor (`corpus.verify`), one-to-one mapping and freshness (`format_cache.verify`),
document counts / ground-truth presence / zero truncation (`index.verify`), and quota
fulfilment (`synth.verify`).

### Three honest caveats

1. **`cmr_cache` pins revisions.** A cache hit is keyed on concept-id alone, so a record
   updated upstream keeps serving its cached revision until the file is deleted. Good for
   reproducibility; it means the corpus reflects a point-in-time CMR snapshot, not live
   metadata. (§3)

2. **Manifest parity only covers re-rendered records.** After an incremental build,
   `parity_failures: []` means "nothing was checked". The true full-render figure is
   2,589/2,590. Use `--build --force` when you need the real number. (§6)

3. **Synthetic queries are not byte-reproducible.** Record *selection* is seeded, but
   generation runs an LLM at `temperature=0.7`. A rerun produces different question text
   for the same records. The `logs/llm/<run_id>/query_gen.jsonl` record of every call is
   what makes a specific run auditable, not re-execution. (§8)

### What is not tracked in git

All of `data/` except user-authored config: `cmr_cache/`, `format_cache/`,
`format_cache_unfaceted/`, `chroma/`,
`corpus.jsonl`, `corpus_report.json`, `gt.jsonl`, `gt_dropped.json`, `queries.jsonl`,
`synthetic_report.json` (`.gitignore:224-239`). Every one of them is derived and rebuildable
from `sample_sme_queries.xlsx` plus CMR, by the commands in §9.
