# Schema vs. format — what is subsetted, what is full, and how the six renderings are made

Two axes run through this study and are routinely conflated in the metadata
literature. Keeping them apart is what makes either measurement mean anything.

| | **Schema** (content) | **Format** (syntax) |
|---|---|---|
| Answers | "Which fields exist and what do they mean?" | "How do I write this down and read it back?" |
| Here | 32-facet projection vs. full UMM-C 1.18.5 | JSON, CSV, YAML, TOON, JSON-LD, MaT |
| Validated by | field coverage — are the facts there? | a parser — does it round-trip? |
| Broken by | renaming `Abstract` → `Description` | a stray brace |
| Determines | which facts are present at all | token cost for fixed content |
| Varied in | the faceted / unfaceted axis | the six-format axis |

**Disambiguating test.** Change the thing. If a generic parser still reads the
bytes but a consumer no longer knows what a field *means*, you changed the
schema. If the parser breaks while every field means exactly what it did, you
changed the format.

They are not interchangeable, and the study is designed as a grid over both:
**2 payloads × 6 formats**, every cell carrying identical facts within its
payload.

---

## Axis 1 — content: subsetted vs. full

Both payloads start from the same source: `data/cmr_cache/<concept-id>.json`,
2,590 raw CMR UMM-JSON items, one file per collection. What differs is the
projection applied before rendering.

### Subsetted — the faceted payload

`airm.facets.facets(record)` → `data/format_cache/`

A projection onto **32 flat top-level fields** (`FACET_KEYS`), drawn from 18 of
the record's 45 top-level UMM fields plus two `meta` keys. Per-field sources and
coverage are in [`FIELD_INVENTORY.md`](FIELD_INVENTORY.md).

Three rules govern it, and each is there to stop a content decision leaking into
the format comparison:

* **Nothing is aggregated.** `platform` and `instrument` are separate facets,
  not instruments nested inside platforms; the four quality fields are four
  facets, not one `quality` dict. A grouping is a modelling choice.
* **Nothing is rendered.** `spatial_resolution` stays `{x, y, unit}`, not the
  string `"30 x 30 Meters"`; `data_volume` stays `{size, unit, basis}`.
  Composing text in the payload would force a rendering decision on every format
  identically.
* **Absences are absent.** Empty facets are omitted rather than emitted as
  `null`, and CMR's placeholder strings (`"Not provided"`, `"N/A"`, …) count as
  empty. A format that writes `"bbox": null` and one that omits the key carry
  the same information at very different token counts.

`cmr_link` is the single deliberate exception — constructed from the concept-id,
identically for every format, because CMR does not store it.

### Full — the unfaceted control

`airm.unfaceted.payload(record)` → `data/format_cache_unfaceted/`

The record's whole `umm` object, untouched: **45 top-level fields, ~398 distinct
paths**, contacts and addresses and metadata dates included. No projection, no
placeholder filtering, no canonicalisation — the point of a control is that
nothing has been decided on the reader's behalf.

`meta` is excluded (CMR's own record-keeping), which is why `concept_id` and
`cmr_link` exist **only** in the faceted payload — `concept-id` lives in `meta`,
not `umm`. It survives incidentally in 21.1% of unfaceted records as a substring
of a `cmr.earthdata` URL inside `RelatedUrls`, but never as a field. Identity in
the full payload rests on UMM's own keys instead (`ShortName` + `Version`, 100%),
and at the system level on the filename, which is the concept-id in every cache.

### What the projection costs and buys

| | faceted | unfaceted | ratio |
|---|---:|---:|---:|
| top-level keys (mean / median) | 20.7 / 21 | 23.5 / 23 | 1.1× |
| leaf values (mean / median) | 39.6 / 34 | 149.1 / 130 | 3.8× |
| corpus size, JSON | — | — | 53.7 MB vs 141.9 MB (2.6×) |
| tokens/record, JSON (mean) | 913 | 1,918 | 2.1× |

The projection keeps most top-level *sections* but cuts **~73% of the data
points**. That reduction is what the +0.045 correctness gain and the 60%
prompt-token saving ride on ([`ANALYSIS_300Q.md`](ANALYSIS_300Q.md) §3).

**The two payloads are not comparable record-for-record.** They carry different
content by construction — that is the whole reason the faceted payload exists.
The unfaceted cache is a *control*, not a competitor: nothing in the study's main
line reads it. It answers a question Experiment 1 cannot, namely "does the format
ranking survive without faceting?", by holding content constant at the other
extreme.

---

## Axis 2 — how the six formats are made

`airm.formats.render_all(record)` calls `facets(record)` **once** and hands that
one dict to all six renderers:

```
cmr_cache/<id>.json  →  facets()  →  one payload dict  →  6 renderers  →  format_cache/<fmt>/<id>.<ext>
```

No renderer may touch the raw CMR record. That single-source rule is what makes
this a format comparison rather than a content comparison.

| Format | How it is produced | Inverse | Schema-aware? |
|---|---|---|---|
| **JSON** | `json.dumps`, compact separators, `ensure_ascii=False`. The baseline every other format is reported against. | `json.loads` | no |
| **YAML** | `yaml.safe_dump`, `sort_keys=False` (so `FACET_KEYS` order holds), block style. | `yaml.safe_load` | no |
| **TOON** | `toon_format.encode` — the official encoder from the TOON authors, pinned by commit in `pyproject.toml`. | `toon_format.decode` | no |
| **CSV** | Two-column `path,value` long form over `flatten(payload)`. A nested record has no faithful *wide* CSV; long form is lossless and keeps the format's real cost — the full path repeated on every row — visible. | rebuild nested payload from path tokens | no |
| **JSON-LD** | schema.org `Dataset`. Property names are schema.org's, not ours; that renaming plus `@context`/`@type` scaffolding and typed sub-nodes *is* what JSON-LD costs. Six facets have no schema.org property and go through `PropertyValue`. | parse back through the mapping | **yes** |
| **MaT** (Metadata-as-Text) | Deterministic prose, one template per facet. No LLM, so it is reproducible run to run. Structured facets are spelled out — a resolution reads "30 by 30 Meters" because `x`, `y` and `unit` are three values the prose must carry. | none — prose is lossy about structure | **yes** |

Ordering is fixed by `FACET_KEYS` for every format that imposes one (JSON, YAML,
TOON, CSV), so ordering never becomes a hidden variable.

### The parity gate — why the comparison is honest

`validate_parity` proves each rendering carries exactly the facts it started
with, in two flavours:

* **Round-trip** (the five parseable formats): parse the rendering back and
  compare fact sets. Anything `missing` or `extra` fails the record.
* **Value coverage** (`mat`): prose cannot be parsed back, so its obligation is
  weaker but still checkable — every canonical value must appear *verbatim* in
  the text, so a model reading the prose has access to the same information as
  one reading the JSON.

This gate is load-bearing. An earlier prose renderer silently omitted concept id,
DOI, short name, topic and the keyword hierarchy, which meant prose carried less
content than the structured formats — exactly the confound the study exists to
remove.

### The unfaceted renderings use the same six formats — but two are rewritten

Four formats are **schema-agnostic**: JSON, YAML, TOON and CSV serialise an
arbitrary tree and read it back, so they run over the full record unchanged. The
other two need to know what a field *means*, and UMM-C 1.18.5 tells them, so both
are written against the schema rather than against the tree:

* **Prose is real prose.** An earlier version emitted one `path is value` clause
  per leaf and called it `mat`. That is a mechanical path dump (`dot_breadcrumb`),
  not prose, and the mislabel cost a factor of two — 4,911 tokens against 2,101 on
  the same record. UMM is a *published* schema, so prose over the whole record is
  one template per field, not an impossibility.
* **JSON-LD genuinely cannot.** schema.org has properties for five UMM fields
  (`EntryTitle`, `Abstract`, `ShortName`, `Version`, `DOI`) and nothing for the
  other forty. The rest travel as `PropertyValue` entries keyed by flattened path
  — legal JSON-LD, semantically inert, and the `@context` buys nothing when every
  term is a literal path string. **That is a limit of the vocabulary, not of the
  tree**, which is why prose escapes it and JSON-LD does not, and why JSON-LD is
  the only format whose cost more than doubles across the payload axis (1,110 →
  4,295 mean tokens, 3.9×).

### Cost, both payloads — tiktoken `o200k_base`, mean tokens/record

| fmt | faceted | unfaceted | ratio |
|---|---:|---:|---:|
| mat | 805 | 1,476 | 1.8× |
| json | 913 | 1,918 | 2.1× |
| toon | 931 | 2,026 | 2.2× |
| yaml | 971 | 2,100 | 2.2× |
| csv | 984 | 2,780 | 2.8× |
| jsonld | 1,110 | 4,295 | 3.9× |

The format *ranking* is identical under both payloads — the projection changes
the magnitude, not the order.

---

## The caching layer

Renderings are deterministic, so they are written to disk once and reused by
Experiment 1, Experiment 2 and the index build. The mapping is one-to-one and
total: every concept-id has exactly one file in every format directory, and
`verify` asserts both directions — a *partial* cache is the failure mode that
would silently drop records from an experiment.

Staleness is closed off rather than documented away: the manifest records a
fingerprint over the source of `airm.facets` and `airm.formats`, so any edit to
either module makes `load` refuse to serve stale bytes.

### The ten cache directories

| Directory | Payload | Content variant |
|---|---|---|
| `format_cache/` | faceted | baseline (record's own abstract) |
| `format_cache_unfaceted/` | full | baseline |
| `format_cache_summary/` | faceted | *the gpt-5.4-nano summaries themselves* (`.txt`, not renderings) |
| `format_cache_unfaceted_summary/` | full | ditto |
| `format_cache_plus_summary/` | faceted | abstract **and** AI summary appended |
| `format_cache_unfaceted_plus_summary/` | full | ditto |
| `format_cache_replace_summary/` | faceted | AI summary **instead of** the abstract |
| `format_cache_unfaceted_replace_summary/` | full | ditto |
| `format_cache_no_summary/` | faceted | abstract **removed** |
| `format_cache_unfaceted_no_abstract/` | full | ditto |

The last six are a third axis — the summary ablation — read as four arms:
baseline (abstract), dropped (none), replaced (summary), appended (both).
Replace and drop re-render every format from a modified payload, so parity still
holds; **append does not** — concatenating prose onto a JSON rendering
deliberately breaks its syntax, because those caches exist only for embedding,
and retrieval only ever sees text.

---

## Summary

* **Subsetted** = `facets()`, 32 flat fields from 18 UMM top-level fields →
  `format_cache/`.
* **Full** = the raw `umm` object, 45 fields / ~398 paths → `format_cache_unfaceted/`.
* **Formats** are six renderings of *one* payload dict, never of the raw record,
  each gated by round-trip or value-coverage parity.
* Schema is the axis that decides **which facts exist**; format is the axis that
  decides **what those facts cost**. The study varies one at a time, which is the
  only reason either number is interpretable.
