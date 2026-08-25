# Experiments: format token economy & model × format retrieval

Living tracker for the two experiments described below. **A box is ticked only after its
`Verify` command has actually run and passed**, and the observed result is recorded
inline underneath it. Nothing is marked done on the strength of "the code looks right".

- Design source: [`Report.md`](Report.md), [`STUDY_SUMMARY.md`](STUDY_SUMMARY.md)
- The previous implementation was deleted in `fe00bb6` ("start from scratch"). Where a
  piece of it is worth recovering, the tracker cites `git show fe00bb6^:<path>`.

---

## The two experiments

### Experiment 1 — Which metadata format costs the fewest tokens?

For one canonical CMR record, rendered six ways with **content held constant**:

`json` · `csv` · `yaml` · `toon` · `jsonld` · `mat` (metadata-as-text)

Measured over ~500 real CMR collection records. No LLM calls required.

### Experiment 2 — How does model variation affect LLM retrieval across formats?

Index each format into its own ChromaDB collection, run the same query set through each
`(format × model)` cell, and score with DeepEval on **Correctness, Faithfulness, Answer
Relevancy, Contextual Relevancy, Contextual Recall** — plus non-LLM retrieval metrics
(Recall@k, MRR, nDCG) and token/dollar cost.

---

## Design decisions

| Decision | Value | Why |
|---|---|---|
| Model matrix | OpenAI GPT tier + local Ollama (`gpt-oss:20b`, `qwen3:8b`) | Two independent model families; the "format winner is model-dependent" hypothesis needs at least two. No Anthropic. |
| Retired SME concept IDs | **Dropped**, not remapped | Ground truth contains only IDs CMR returns today. |
| Corpus size | ~500 records | Matches the ~437 of the prior study; keeps six-collection indexing tractable. |
| Query set | 11 SME (verbatim) + 120 synthetic = **131** | 11 alone is far too few for stable per-cell metrics. |
| Embedding model | `BAAI/bge-small-en-v1.5`, fixed | Held constant — it is *not* an independent variable. |
| DeepEval judge | Fixed OpenAI model, separate config knob from the system under test | A judge that varies with the tested model is a confounder in the very axis being measured. |

### Content parity — the core methodological guard

A format comparison only means something if every format carries **the same facts**. A
full nested UMM tree versus a short prose summary differ in *content*, not just shape, so
any measured difference is uninterpretable.

So all six renderers are driven from one `facets(record) -> dict` payload, and
`formats.validate_parity()` asserts:

- the five machine formats (`json`, `csv`, `yaml`, `toon`, `jsonld`) decode back to an
  identical normalised fact set;
- `mat` (prose, which cannot round-trip) contains every canonical **value** verbatim —
  parity asserted as value-coverage = 1.0.

Any record failing parity is **excluded from the corpus and reported**, never silently
dropped.

### Format rendering choices

| Format | Rendering | Note |
|---|---|---|
| `json` | `json.dumps` of the UMM-shaped facet subset | The baseline all percentages are relative to |
| `yaml` | `yaml.safe_dump`, `sort_keys=False`, block style | |
| `jsonld` | schema.org `Dataset` (pattern from `sample_data/record_jsonld.json`) | |
| `toon` | `toon_format.encode` — the official implementation, unpatched (see Step 0) | |
| `csv` | Two-column `path,value` long form over the flattened facet tree | The only *lossless* CSV for a nested record. A wide single row would either drop the repeated groups or explode into hundreds of sparse columns. Documented as a deliberate choice. |
| `mat` | Natural-language prose (`git show fe00bb6^:airm/representations.py`, `metadata_as_text`) | |

### Logging

Every LLM call — query generation, answers, **and DeepEval judge calls** — is written to
`logs/llm/<run_id>/{purpose}.jsonl` with timestamp, provider, model, purpose, format,
query id, full prompt, full response, token counts, latency, cost and retry count. Judge
calls are routed through the same wrapper via a `DeepEvalBaseLLM` subclass so nothing
bypasses the log.

### Hard gates (must fail loudly, never warn)

- Every ground-truth concept ID is present in the corpus **and** in all six collections.
- Content parity holds across all six renderings of every corpus record.
- Every LLM call, judge calls included, appears in `logs/llm/<run_id>/`.
- Dropped concept IDs, dropped synthetic queries and truncated documents are counted and
  reported — never silently discarded.

---

## Steps

### [x] Step 0 — Dependency install & TOON spike

Confirm a usable TOON encoder exists before building anything on it.

**Verify:** round-trip a facet-shaped payload through `encode`/`decode`; cross-check the
output against the reference implementation.

**Result — passed.**

- `uv add chromadb deepeval python-toon pyarrow openpyxl` succeeded
  (`chromadb 1.5.9`, `deepeval 4.1.5`, `python-toon 0.1.3`, `tiktoken 0.13.0`).
- PyPI `toon` is an unrelated neuroscience package and `toon-format` is an empty
  namespace reservation — neither is usable.
- `toon-py` **rejected**: encode-only (no decoder), and it mis-encodes inline arrays
  nested in list items, emitting `instruments:` / `[1]: ATLAS` on two lines where the
  reference emits `instruments[1]: ATLAS`.
- `python-toon` **selected**. Its encoder is spec-correct except that it always emits the
  delimiter inside tabular headers — `science_keywords[2,]{…}` where the spec and the
  reference emit `science_keywords[2]{…}` (deliberate, `primitives.py:192-194`). Left
  alone this adds one spurious token per tabular array and would bias Experiment 1
  against TOON, so a `[N,]{` → `[N]{` fix is applied on output. Its decoder accepts the
  corrected form.
- Cross-checked against the reference JS implementation `@toon-format/toon` under Node
  v25.1.0: corrected `python-toon` output is **byte-identical** to the reference on
  **500/500** corpus records, after normalising the one JS artifact (JS has no int/float
  distinction, so it prints `-180` where Python prints `-180.0`).

  *Re-verified at full scale in Step 3a.* The original claim rested on a spot check and
  was in fact true for only **283/500** records: `python-toon` writes an empty array as
  `instruments[0]:` where the reference writes `instruments: []`. Removing empty lists
  from the payload (Step 3a) made the two agree everywhere, so the claim now holds as
  stated — but it was not established by the check that was run at the time.

- **`python-toon` 0.1.3 is only trustworthy for the shapes we render.** Two further
  defects, both found in Step 3a and both outside the facet payload, are recorded here so
  nobody rediscovers them when extending the content model:
  1. *Decoder, inline arrays.* `decoder.py:521` locates the header colon with
     `rfind(COLON)` — the **last** colon on the line rather than the first *unquoted* one.
     A quoted value containing a colon (`"s3://…"`) therefore truncates the array and
     raises `Expected N values, but got 1`.
  2. *Encoder, list-form items.* It emits a bare `-` marker on its own line with the
     item's fields indented beneath; the spec and the reference put the first field on the
     marker line (`- Roles[1]: Investigator`). The output is **not valid TOON** — neither
     `python-toon`'s own decoder nor the reference decoder can read it.

  Consequence, measured on full raw-UMM payloads: **0/500** survive a reference round-trip
  as emitted. A textual fix for (2) lifts that to 500/500 decodable, but `python-toon`
  still declines tabular form where the reference uses it (345/500 differ), which would
  *understate* TOON's efficiency — tabular compression is the format's whole point. So a
  richer content model would need the Node reference encoder, not `python-toon`. This is
  the binding constraint on closing the coverage gap; see Step 3a.
- Float handling: bbox coordinates are normalised identically across **all** formats, so
  the JS/Python float difference cannot leak into the token comparison.

**Superseded — the official implementation replaced `python-toon`.**

Everything above describes `python-toon` 0.1.3, which the study no longer uses. TOON now
comes from **`toon_format` 0.9.0-beta.1**, the implementation maintained by the TOON
authors at `github.com/toon-format/toon-python`, installed from git at pinned commit
`e475c82` and *not* from PyPI — the PyPI `toon-format` 0.1.0 release is a namespace
reservation whose `encode()` raises `NotImplementedError`.

- **No token counts change.** `toon_format.encode` output is **byte-identical** to the
  patched `python-toon` output on all **2,590** cached records. The `[N,]{` → `[N]{` fix
  was emulating spec behaviour correctly, and `_toon_spec_fix` is now deleted rather than
  carried as an unexplained patch on a third-party library.
- **One record is recovered at the parity gate.** `python-toon`'s decoder fails to
  round-trip `C2105107978-NOAA_NCEI`, whose DMSP instrument descriptions contain embedded
  newlines: it decodes a list item back into a bare string. `toon_format` round-trips it.
  Full-cache parity is now **2,589/2,590**, and the single remaining failure is `yaml`,
  not `toon` (`C1214595237-SCIOPS` — see below).
- **The `0.0` → `0` encoder quirk is gone.** `canonical_coord` now rests solely on the
  token-artefact argument, which was always the stronger of its two justifications.
- **The wider-payload blocker is reduced, not cleared.** Full raw-UMM documents now
  round-trip **2,385/2,590 (92.1%)** against **0/500** with `python-toon`: 169 mismatch
  and 36 raise `ToonDecodeError`. Differing leaves concentrate in `SpatialExtent` (1,246),
  `ContactPersons` (743) and `DataCenters` (687). Untested territory for this study, which
  only renders facet payloads — but the constraint on a richer content model is now a
  gap of 7.9%, not a wall.
- **Unrelated to TOON: one YAML parity failure survives.** `C1214595237-SCIOPS` carries
  mojibake in its abstract (double-encoded UTF-8 for "Åkerman") containing U+0085 NEL,
  which YAML treats as a line break and normalises to a space on load. The parity gate
  catches it and excludes the record, which is the correct behaviour.

### [x] Step 1 — Tracker & scaffold

This file, then `src/airm/` skeleton, `config.py`, `pyproject.toml` (`pythonpath = ["src"]`),
`.gitignore` for `runs/`, `logs/`, `data/chroma/`.

**Verify:** `uv run pytest` collects; `uv run python -c "import airm"`.

**Result — passed.** `uv sync` installs `ai-metadata==0.1.0` editable (a hatchling
`src`-layout build backend was added; the stale `metadata-eval` editable install from the
deleted implementation was removed). `import airm` resolves and `uv run pytest` reports
**7 passed**. `tests/test_config.py` pins the invariants that are easy to break later: six
distinct formats with JSON as baseline, primary/secondary topics partitioning the 13 CMR
topics, the synthetic-query quota summing to exactly 120, and the judge being configured
outside the system-under-test matrix.

### [x] Step 2 — `cmr.py`: fetch + disk cache

Port `search_collections`, `fetch_by_concept_ids` and the atomic per-concept-id cache
from `git show fe00bb6^:airm/cmr.py`.

**Verify:** fetch `C3550186689-ESDIS`; assert the second call is served from cache.

**Result — passed.** `uv run pytest` → **16 passed**; `uv run pytest -m live tests/test_cmr.py`
→ **2 passed**. The live test fetches `C3550186689-ESDIS` (IPCC Socio-Economic Baseline
Dataset), then monkeypatches `_get` to raise and re-fetches — proving the second call is
served entirely from disk. Network tests are marked `live` and deselected by default
(`addopts = "-m 'not live'"`), so the everyday suite stays offline and fast.

Three additions beyond the ported original, each needed later:
- `search_by_topic()` filters on `science_keywords[0][topic]` — the stratification
  primitive Step 5 samples from.
- Concept-id batches are chunked at 50; a single 500-id request would overflow the
  request URL.
- `_get()` retries 5xx/transport errors with backoff but re-raises 4xx immediately —
  retrying a malformed request only hides the bug.

`missing_concept_ids()` exists so unresolvable ids are *reported* rather than quietly
vanishing from a result list; Step 4 depends on it.

### [x] Step 2a — `format_cache.py`: the renderings on disk

`data/cmr_cache/<concept-id>.json` holds the raw CMR item. This gives every format the
same treatment, one file per (record, format):

```
data/format_cache/{json,csv,yaml,toon,jsonld,mat}/<concept-id>.{json,csv,yaml,toon,jsonld,txt}
```

**Verify:** the mapping is one-to-one and total in both directions, and the bytes match
what the renderers produce today.

**Result — passed.** `uv run python -m airm.format_cache --build` writes **15,540 files**
(2,590 records × 6 formats, 36.7 MB) in ~14s; an unchanged rebuild takes 1.5s. `--verify`
asserts both directions of the mapping. `uv run pytest` → **263 passed**, 18 of them new.

- **Speed is the smallest of the three reasons.** Cached reads beat live rendering only
  2.5× (0.44s → 0.17s for 500 records × 6 formats), which alone would not justify the
  module. The other two matter more: the renderings become `diff`-able artefacts, so *why*
  a format costs what it costs is legible without running Python; and every token count,
  retrieval hit and LLM answer now traces back to exact bytes on disk rather than to a
  function call that has to be re-run to be inspected.
- **Staleness is closed off, not documented away.** The manifest stores a fingerprint over
  the source of `facets.py` and `formats.py` — every line that can change a rendering. Any
  edit to either invalidates the whole cache at once, `load()` raises `StaleCacheError`
  rather than serving bytes from older code, and `build()` stops treating "file exists" as
  "file is current".
- **Nothing depends on it.** `index.build` and `exp1.measure_records` read through
  `render_many` / `render_all_cached`, which fall back to live rendering when the cache is
  absent, stale, or missing that record. Verified over 200 records: exp1's 1,400 measured
  rows and all six of index's document lists are byte-identical either way. A cache that
  could change a result would be a liability, not an optimisation.
- **Parity is recorded at cache time**, so a record that cannot be represented identically
  in all six formats is visible in the manifest from the moment it is rendered, not only
  when Step 5 happens to sample it. Full cache: **2,589/2,590 pass**, the one failure being
  `yaml` on `C1214595237-SCIOPS` (see Step 0).
- Not tracked in git — it is derived, and rebuilds deterministically in 14s.

### [x] Step 3 — `facets.py`, `formats.py`, parity validator

Recover the facet extractors and `metadata_as_text` from git history; add the YAML, CSV
and TOON renderers.

**Verify:** all six renderings of `sample_data/umm_json.json` carry identical facts;
snapshot-diff `mat` against `sample_data/record_mat.txt`.

**Result — passed.** `uv run pytest` → **59 passed**. The live sweep checked **520 real
CMR records across all 13 topics with zero parity failures**, which is what Step 5's
corpus build depends on.

Parity is defined as equality of `(path, canonical value)` fact sets: the five machine
formats are rendered, parsed back, and compared to the payload they came from; `mat` is
held to value-coverage instead, since prose cannot round-trip. Two tests deliberately
break a renderer (one that drops the title, one that invents a title) to prove the
validator can actually fail rather than always passing.

**The validator caught two real problems, which is the reason it exists:**

1. `python-toon` encodes `0.0` as `0` — and *only* `0.0`; `-180.0` survives intact. A
   bounding box touching the equator or prime meridian would therefore have decoded back
   as an integer and failed round-trip.
2. Following that thread exposed a larger measurement artefact: Python's float `repr`
   writes `-180.0` where `-180` states the identical fact, charging JSON, YAML and TOON
   three characters of pure serialisation noise in an experiment whose entire subject is
   token cost.

Both are fixed at the source — `canonical_coord()` stores integral coordinates as `int`,
so every format now emits `-180` and the numbers are genuinely comparable.

Two further design choices worth recording, both made to keep parity honest rather than
to make it pass:

- **Science keywords are stored as joined GCMD paths** (`"EARTH SCIENCE > OCEANS > SEA
  ICE"`), not as `{category, topic, term}` objects. JSON-LD has to flatten them into a
  `DefinedTerm.termCode`, and a record missing a middle level cannot then be reassembled
  unambiguously — parity would fail on a rendering artefact rather than on real content
  loss. The joined path is also GCMD's own notation.
- **CSV is two-column `path,value` long form.** A nested record has no faithful *wide*
  CSV: one row would either drop the repeated groups or explode into hundreds of sparse
  columns sized by the worst record in the corpus. The long form is lossless and keeps
  CSV's real cost — repeating the full path on every row — visible instead of hidden.

The recovered prose renderer needed extending: the deleted original omitted concept id,
DOI, short name, topic and the keyword hierarchy, so prose silently carried *less*
content than the structured formats — precisely the confound this study exists to
remove. Against `sample_data/record_mat.txt` the new renderer adds those five facts
(682 → 939 chars) and stops lower-casing `CollectionProgress`, which would otherwise have
failed value coverage.

### [x] Step 3a — `coverage.py`: content parity *against the source record*

Steps 0–3 establish that the six formats carry the same content **as each other**. They
never established what that content is relative to the record CMR actually publishes.
This step answers that, because "the formats are equal" and "nothing was discarded" are
different claims and only the first had ever been checked.

**Verify:** classify every scalar path in all 500 corpus items; no path may be
unaccounted for.

**Result — the format comparison is sound; the projection is much narrower than the
prose around it implied.**

- **Between formats: no loss.** All five machine formats round-trip to an identical fact
  set on **500/500** records, and prose contains **every** canonical value verbatim — 0
  missing literals across the corpus. Per-format literal-containment differences (YAML
  −1,072, JSON/TOON/JSON-LD −296, CSV −79) are **serialisation artifacts, not loss**: JSON
  escapes `\n` in abstracts, YAML folds lines at 80 columns, CSV doubles quotes. The
  round-trip test is the authoritative one and it is exact.
- **Against the source: 18.4% survives.** Of 97,312 scalar values in the corpus items,
  17,942 reach the formats. `airm/coverage.py` sorts every path into `CARRIED` (14.3%),
  `PARTIAL` (4.1%, first element of a repeating group only — first bounding rectangle,
  first temporal range, first data centre), `EXCLUDED` (39.1%, declared with a reason:
  CMR catalog bookkeeping, UMM schema identifiers, contact addresses and phone numbers)
  and `DEFERRED` (42.5%, real dataset content not carried yet). `verify()` fails on any
  path in none of them, so a new UMM field cannot silently vanish. `tests/test_coverage.py`
  pins this; the whole-corpus case runs in the default suite.
- The largest deferred groups are `RelatedUrls`, `Projects`, `LocationKeywords`,
  `AdditionalAttributes`, `ISOTopicCategories`, `Purpose` and `Quality` — searchable
  content, not plumbing. Because the loss is upstream of every renderer it is identical
  across formats and cannot bias the comparison, but the study measures the cost of a
  metadata *summary*, not of the full record. `FINDINGS.md` now says so.

**One real confound found and fixed.** `extract_platforms` emitted `instruments: []` for
a platform with no instruments. An empty list asserts no fact — `flatten` yields nothing
for it, so the parity check was blind to it — yet JSON, YAML and TOON serialise the
brackets while CSV, JSON-LD and prose do not. Across 318 of 747 platform entries that
charged JSON 0.43%, YAML 0.40% and TOON 0.56% of their total tokens for information-free
punctuation. `facets()` already documented the rule ("empty facets are omitted entirely");
the nested case simply escaped it. Fixing it also made `python-toon` agree with the
reference implementation on 500/500 records instead of 283/500.

Effect on Experiment 1 — rankings unchanged, every effect slightly smaller or larger by
under a point:

| Format | vs JSON before | vs JSON after |
|---|---:|---:|
| Metadata-as-Text | −2.5% | −1.9% |
| TOON | +3.4% | +3.2% |
| CSV | +5.9% | +6.5% |
| YAML | +8.0% | +8.0% |
| JSON-LD | +20.8% | +21.5% |

**Why the gap is not simply closed.** Carrying the full record is blocked by the TOON
arm, not by the extractors: `python-toon` emits invalid TOON for deeply-nested UMM
structures and declines tabular form where the reference uses it (see Step 0). Widening
the payload therefore means adopting the Node reference encoder — a runtime dependency
and a change to what the study measures. That is a design decision, so the deferred
content is listed and tracked rather than quietly folded in.

### [x] Step 4 — `queries.py` (SME half) & ground truth

Parse `sample_sme_queries.xlsx`, merge duplicate query rows, drop unusable IDs, write
`data/gt.jsonl` and `data/gt_dropped.json`.

Known quirks in the sheet, confirmed during planning: 25 rows but only **11 distinct
queries**; 43 unique concept-id tokens; three cells use **U+2011 non-breaking hyphens**;
`SENTINEL-1A_SP_GRD_FULL` and `Not in CMR` are not concept IDs; and 7 of the 41
well-formed IDs are retired (all `-NSIDC_ECS`): `C2776463935`, `C2531308461`,
`C2537927247`, `C2399557265`, `C1542606326`, `C3383993430`, `C2776463943`.

**Verify:** exactly 11 queries; all 34 surviving IDs resolve in a live CMR batch call;
the dropped list matches the 7 + 2 above.

**Result — passed.** `uv run pytest` → **79 passed**; `-m live` → **6 passed**.
`uv run python -m airm.queries --build` reports:

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

Every planned figure matched. Crucially, **no query was emptied by the retired-id drop** —
all 11 research questions survive with at least one live target (`sme-001` and `sme-007`
are down to a single id each, which is worth remembering when reading their per-query
scores).

Two things the sheet does that would have quietly corrupted ground truth:

- **25 rows, 11 questions.** The sheet splits one research question across several rows,
  one per dataset family. Scoring rows as queries would have counted the fire-regime
  question three times and the Cyclone Mocha question four. Rows sharing a query string
  are merged and their expected sets unioned.
- **Ids that look right and compare wrong.** Three cells use U+2011 NON-BREAKING HYPHEN
  instead of ASCII `-`. Normalisation folds all six Unicode dash variants.

`data/gt_dropped.json` records every removed token with its reason and the original cell
text. Counts are reported as **unique** ids (34) alongside slots across queries (38) —
three collections legitimately answer more than one question, and an unqualified sum
overstates coverage.

### [x] Step 5 — `corpus.py`: 500 stratified records

Seed with every ground-truth ID, stratified-sample the remainder across all
science-keyword topics (Land / Ocean / Atmosphere weighted equally), assign domains, run
parity validation, write `data/corpus.jsonl`.

**Verify:** length is 500; all 34 GT IDs present; ≥3 records per topic; parity-failure
count printed.

**Result — passed.** `uv run pytest` → **101 passed**; `-m live` → **7 passed**.
`uv run python -m airm.corpus --build`:

```
corpus size              : 500
seeded from ground truth : 34 / 34
parity failures excluded : 0
ATMOSPHERE 83 · OCEANS 83 · LAND SURFACE 83
AGRICULTURE / BIOSPHERE / CRYOSPHERE / CLIMATE INDICATORS / HUMAN DIMENSIONS /
BIOLOGICAL CLASSIFICATION / SUN-EARTH INTERACTIONS / SOLID EARTH /
TERRESTRIAL HYDROSPHERE  23 each · PALEOCLIMATE 22 · SPECTRAL/ENGINEERING 22

all hard gates passed
```

The three required domains are exactly equal at 83, every other domain clears the floor,
and **zero of the 500 records failed content parity** — the Step 3 sweep generalised.

The first build produced **468 records with topics split across 23 buckets**, which
exposed three genuine bugs:

1. **Quota arithmetic subtracted the ground-truth seed twice** — once from the corpus
   target and again per topic — so the build could never reach 500. Quotas are now
   computed over the whole corpus, with seeded records counted against them.
2. **Topic buckets split on capitalisation.** CMR records carry `Oceans` and `OCEANS`,
   `Atmosphere` and `ATMOSPHERE`; as distinct keys they pushed `Spectral/Engineering`
   (1) and `Climate Indicators` (2) below the floor and failed the gate for a reason that
   was pure formatting.
3. **`ScienceKeywords[0].Topic` is not the record's domain.** CMR matches a topic search
   against *any* keyword, so the first one often names a different domain — and one
   record carries a NOAA department name there. `topic_of()` now scans all keywords for a
   canonical topic and only falls back to `UNCLASSIFIED` when none matches; after the
   fix, no record in the corpus is unclassified.

Checking CMR's own `include_facets=v2` listing rather than trusting the planned topic
list also caught a **missing domain: `BIOLOGICAL CLASSIFICATION`, 4,731 collections**.
It is now one of 14 topics. The same listing shows a long tail of malformed topic values
(`-Na-`, `Atmospheric`, `Hydrosphere`) which are data-entry artefacts, not domains, and
stay excluded.

Sampling draws from a 250-record pool per topic under a fixed seed (`20260731`), so a
rebuild against an unchanged CMR reproduces the corpus; taking CMR's first N would have
correlated the corpus with CMR's own relevance ranking.

### [x] Step 6 — `tokens.py` + `exp1.py`: token economy

Count with `tiktoken` `o200k_base` (OpenAI side), a HF Qwen3 tokenizer (open-weights
cross-check) and raw chars/bytes (tokenizer-independent control).

**Verify:** `uv run python -m airm.exp1` writes `runs/<id>/exp1_tokens.csv` plus a chart;
sanity-check that TOON and MAT both come in under JSON.

**Result — passed, with one predicted finding refuted.** `uv run pytest` → **120 passed**;
`-m live` → **8 passed**. 500 records × 6 formats + a raw-UMM reference row = **3,500 rows**.

## Experiment 1 result

Mean tokens per record, `tiktoken o200k_base`, content held constant, n = 500.
"vs JSON" is a **paired** per-record ratio — every format sees the identical record, so
pairing removes between-record variance, which is enormous (median 516 vs mean 587).

| Format | Mean | Median | 95% CI | vs JSON | Paired 95% CI |
|---|---:|---:|---:|---:|---:|
| **Metadata-as-Text** | **574** | 503 | [545, 604] | **−2.5%** | [−2.7%, −2.3%] |
| JSON *(baseline)* | 587 | 516 | [558, 617] | — | — |
| TOON | 603 | 532 | [574, 634] | +3.4% | [+3.2%, +3.6%] |
| CSV | 625 | 550 | [593, 661] | +5.9% | [+5.4%, +6.3%] |
| YAML | 637 | 548 | [605, 671] | +8.0% | [+7.8%, +8.2%] |
| JSON-LD | 705 | 622 | [669, 745] | +20.8% | [+20.0%, +21.6%] |
| *raw UMM (reference)* | *1,991* | *1,770* | — | — | *not comparable — more content* |

The Qwen3 tokenizer reproduces the **identical ranking** (mat −0.8%, toon +4.1%, csv
+6.9%, yaml +8.7%, jsonld +19.2%), so this is a property of the formats, not of one
vocabulary.

**The planned sanity check "TOON < JSON" is false, and that is a real result.** TOON's
saving comes from collapsing uniform arrays of objects into a tabular header, and CMR
collection metadata has almost none: the facet payload is six scalars, two lists of
plain strings, three small dicts, and one list of objects with a *nested* list inside
(platforms → instruments) which cannot be tabularised. Verified directly — on a uniform
20-row tabular array TOON costs **211 tokens against JSON's 345, a 39% saving**, exactly
as advertised. On CMR records it pays its quoting overhead and earns none of that back.
A pinned live test guards this so the claim cannot silently rot.

Other observations worth carrying into Experiment 2:

- The spread is narrow. Excluding JSON-LD, the six formats sit within ~6% of each other,
  which is consistent with the study's own hypothesis that format effects are small once
  content is held constant. JSON-LD's +20.8% is the one large effect, and it buys
  `@context`/`@type` scaffolding, typed sub-nodes and longer schema.org property names.
- **The raw UMM record CMR serves today costs 1,991 tokens — 3.4× the cheapest
  representation.** It is reported as a reference, never in the comparison, because it
  carries far more content; treating it as a seventh condition would reintroduce exactly
  the content/format confound this study exists to remove. But it is the number that
  motivates the exercise.
- Per-topic means vary more than formats do (Agriculture 989 tokens vs Biological
  Classification 525 for JSON), so the domain mix of a corpus moves token cost more than
  the format choice does.

The chart (`runs/<id>/exp1_tokens.png`) was validated with the data-viz palette checker
(worst adjacent CVD ΔE 24.7, normal-vision 33.6, contrast ≥3:1) and redrawn once after
inspection: value labels were colliding with the CI whiskers and the legend was sitting
on top of the bottom bars.

### [x] Step 7 — `index.py`: six ChromaDB collections

One collection per format, same 500 records, fixed embedding function, one document per
record; log any truncation per format since formats differ in length.

**Verify:** each collection holds 500 documents; all 34 GT IDs indexed in **every**
collection; a known query returns its GT record in the top 10 for at least one format.

**Result — passed, after replacing the embedding model.** `uv run pytest` → **135 passed**;
`-m live` → **11 passed**. All six collections hold 500 documents, all 34 ground-truth
records are present in every one, and **zero documents are truncated**.

**The truncation counter caught a confound that would have invalidated Experiment 2.**
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

Every collection would still have reported 500 documents and produced plausible Recall@k
— while quietly clipping the verbose formats roughly twice as often as the terse one.
The measured "format effect" would have been substantially a truncation effect, aimed at
exactly the formats under test.

Longest rendering in the corpus is **6,325 tokens** (a JSON-LD record), so the embedder
needs an 8k window. `Alibaba-NLP/gte-modernbert-base` (8192 tokens, 768 dims) was chosen
over `nomic-embed-text-v1.5` and `jina-embeddings-v2-small-en` because it is the only one
of the three that loads **without `trust_remote_code`** — no remote code execution for a
model pulled at build time. A live test now asserts the longest rendering fits the window,
and `verify()` fails the build on a single truncated document (`MAX_EMBED_TRUNCATIONS = 0`).

Two smaller corrections along the way:

- 8k-token attention at the default batch of 32 asked MPS for a >10 GiB allocation and
  died. The encode batch is 4, with the device cache released between batches; the
  library sorts by length internally so short documents still batch densely. Full build:
  **4m07s** for 3,000 embeddings.
- Chroma's default space is L2. The embeddings are normalised, so the collections are
  created with `hnsw:space: cosine` — L2 over normalised vectors ranks differently.

Retrieval sanity check on the expert slice (Recall@10 against the 500-record corpus,
38 expected-id slots across 11 queries):

```
json 55.3% · yaml 55.3% · toon 55.3% · jsonld 55.3% · mat 50.0% · csv 47.4%
```

Ten of the eleven queries surface at least one expected record. `sme-001` returns none in
any format — a single-target question ("agricultural drought severity in California")
whose only surviving id is an AMSR2 soil-moisture collection. These are research
questions, not dataset names, so this difficulty is expected and is the point of keeping
the expert slice separate from the synthetic one.

### [x] Step 8 — `llm.py` + JSONL call logging

OpenAI and Ollama behind one interface; the logger described above; a cost table.

**Verify:** one call per provider; `logs/llm/<run_id>/` contains a well-formed record
with non-zero token counts.

**Result — Ollama verified, OpenAI blocked.** `uv run pytest tests/test_llm.py` →
**18 passed**. `uv run python -m airm.llm` produced real logged calls against Ollama
(`gpt-oss:20b` 5.39 s, in=67 out=48; `qwen3:8b` 2.93 s, in=21 out=2) with complete records:
timestamp, provider, model, purpose, full prompt, full response, token counts, latency,
attempts, finish reason and arbitrary trace metadata.

> **Blocker — OpenAI key rejected.** `OPENAI_API_KEY` is present in the shell environment
> but the API returns `401 Incorrect API key provided: sk-proj-…zuYA`. The cloud arm of
> Experiment 2 and the DeepEval judge both depend on it. Per the user's direction, Steps
> 9–12 are **built but not executed**; all their logic is unit-tested against stubbed
> providers instead. `resolve_models()` surfaces this at startup rather than 404-ing
> partway through the matrix.

Tests pin the properties that make the log an audit trail rather than decoration: a
**failed** call is logged too (a silent failure is the one thing an audit trail must never
allow), judge traffic lands in its own `judge.jsonl`, trace metadata (`fmt`, `query_id`)
is written through so a results row can be traced back to the call that produced it, and
16 concurrent writers produce 16 complete JSON lines — Experiment 2 fans out, and a torn
line destroys the record it was written for.

**Costs are `None` unless a price is configured** (`data/prices.json`). A fabricated zero
would let local models win every cost-per-accuracy comparison by default. Token counts
are exact and always recorded, and carry the cost comparison on their own.

### [~] Step 9 — Synthetic query generation (120) — *built, not run*

`airm/synth.py`. Generated *from indexed records*, so ground truth is in-corpus by
construction — the failure mode that cost the expert slice 7 targets cannot occur here.
Quota: 30 Land / 30 Ocean / 30 Atmosphere, and the remaining 30 spread over the other 11
topics (2–3 each, `config.synthetic_quota()`).

**Verify (pending a working key):** `uv run python -m airm.synth` → 120 queries; quota met
per domain; every target retrievable; generation calls present in the log.

**Built and unit-tested — 24 tests passing against a stubbed provider.** Three guards
decide whether a candidate joins the set, each because the alternative silently corrupts
Experiment 2:

1. **It parses.** Fenced JSON and chatty preambles are tolerated; unparseable responses
   are retried.
2. **It leaks no identifiers.** A query naming the short name, DOI, concept id — or
   quoting five consecutive words of the title — turns retrieval into string matching,
   which inflates *every* format equally and hides the effect being measured.
3. **It is retrievable.** The source record must appear in the top-k for at least one
   format, or the query measures nothing.

Failures are dropped with a recorded reason and a quota shortfall fails `verify()`; a
quietly rebalanced quota is a quietly biased query set.

The leak detector had a real bug the tests caught: it built title n-grams from
*stopword-filtered* words but matched them against the *raw* query, so `"ATLAS ICESat Land
and Vegetation Height"` never matched the filtered `"ATLAS ICESat Land Vegetation
Height"` — meaning a title quoted with its stopwords, the likeliest way a model quotes
one, sailed straight through. Both sides are now tokenised identically.

### [~] Step 10 — `evaluate.py`: DeepEval + retrieval metrics — *built, not run*

Recall@{1,5,10}, MRR, nDCG@10, plus `GEval` Correctness, `FaithfulnessMetric`,
`AnswerRelevancyMetric`, `ContextualRelevancyMetric`, `ContextualRecallMetric`.

**Verify (pending a working key):** full suite on 3 queries × 1 format × 1 model; all five
metrics return scores; judge calls appear in `logs/llm/<run_id>/judge.jsonl`.

**Built and unit-tested — 20 tests passing.** The retrieval metrics involve no model, so
they are exact and pinned against hand-computed values rather than against themselves.
Design points worth stating:

- **Recall is over the expected set, not a hit-rate.** Several SME queries expect up to
  nine collections; a hit-rate would score "found one of nine" as perfect.
- **nDCG@10 is reported alongside Recall@10** because Recall cannot tell rank 1 from rank
  10, and a format that ranks the answer first is meaningfully better than one that
  buries it ninth.
- **No ground truth yields `NaN`, not `0`.** Unmeasurable and failed are different.
- **A judge failure is recorded as an error with a `None` score, never as a zero.** A
  broken judge and a bad answer are different events, and averaging a failure in as zero
  quietly drags a cell down.
- **`LoggedJudge` routes every DeepEval call through `airm.llm`.** Left to itself DeepEval
  talks to OpenAI directly, which would make the judge calls — the ones that actually
  produce the reported scores — the only LLM traffic in the study with no record of what
  was asked or answered.
- The judge is parsed through the same fence/preamble tolerance as generation, since local
  models wrap JSON and that is not a grading failure.

### [~] Step 11 — `exp2.py`: the full matrix — *built, not run*

6 formats × N models × 131 queries, with per-cell checkpointing and resume.

**Verify (pending a working key):** smoke run (2 formats × 1 model × 5 queries) writes
`runs/<id>/exp2_cells.jsonl`; then the full run completes.

**Built and unit-tested — 20 tests passing against stubbed retrieval and providers.**
The bookkeeping is what is under test, and two properties matter most:

- **Retrieval runs once per `(format, query)`, shared across models.** It depends only on
  the format, so re-running it per model would duplicate identical vector searches N
  times. A test caught the runner doing 12 searches where 4 were needed — `retrieve_for`
  was iterating all six formats regardless of which the run had requested.
- **Every cell is checkpointed to JSONL as it completes, and resume skips completed
  cells.** A matrix this size *will* be interrupted, and a run that restarts from zero is
  a run that never finishes.

A failed answer is written as an error cell that still carries its retrieval metrics —
those do not depend on the model, so losing them to a provider outage would throw away
good data. Unavailable models are reported in the summary rather than silently reducing
the matrix.

### [x] Step 12 — `report.py` + `FINDINGS.md`

Tables, per-domain breakdown, accuracy-vs-token-cost Pareto plot.

**Verify:** every number in `FINDINGS.md` traces to a row under `runs/<id>/`.

**Result — passed for Experiment 1.** `uv run python -m airm.report` writes `FINDINGS.md`
with the full Experiment 1 table sourced from `runs/20260731T204026Z/exp1_summary.json`,
and marks Experiment 2 *"Not run yet"* rather than omitting it — a partial study should
read as partial.

The report answers the study's core question explicitly when the data allows: it compares
the best-scoring format per model and states **"models disagree (gpt-5 → json, qwen3.6 →
mat)"** or *"all models agree"*. With one model it says the question is **"not yet
answerable"** rather than implying a finding from a single data point. Missing metrics
render as `—`, never as `0`.

The Pareto plot is deferred until Experiment 2 has cells to plot against Experiment 1's
token costs.

---

### [x] Review pass over the uncommitted code

A fresh-eyes review of Steps 0–12 before first commit found six defects; all are fixed
and covered by new tests (**230 offline + 11 live passing**):

1. **Exp 2 ignored the synthetic queries.** `exp2` loaded `data/gt.jsonl` (11 SME
   queries) while `airm.synth` wrote the combined 131-query set to `data/queries.jsonl`
   — the full matrix would have silently run 8% of its query set. `load_eval_queries()`
   now prefers the combined set and falls back to SME-only before synth has run.
2. **Resume was unreachable from the CLI.** Every invocation minted a fresh run id, so
   the checkpoint machinery could never engage. Added `exp2 --resume RUN_ID`, and a
   resumed run now also skips the vector searches for fully-checkpointed cells.
3. **Non-retryable errors were retried.** `llm.complete()` retried 401/404s four times
   with backoff; now only 408/429/5xx/transport errors retry. And the DeepEval judge —
   which bypasses `resolve_models` — is probed at `exp2.run()` startup, so a dead judge
   key fails fast with a `--no-judge` hint instead of burning hours of 401s per metric
   per cell and "completing" with every judge score `None`.
4. **`corpus.verify()` never checked total size.** Quota shortfalls above the per-topic
   floor read as success; the gate now fails on any size mismatch and names the
   shortfalls.
5. **`LoggedJudge` skipped `DeepEvalBaseLLM.__init__`,** leaving `model.name` unset —
   read directly by DeepEval's support checks. Now set explicitly.
6. **Judge JSON parse failures pre-empted DeepEval's own parser.** `generate` now
   returns the raw text on parse failure so `trimAndLoadJson` gets its shot, instead of
   turning a parseable response into a metric error.

Minor: dead line in `exp1.format_table`; unused import in `report.py`; Ollama `think`
parameter now falls back for models without the capability; `summarise()` reports
`n_answered` alongside `n` so retrieval and judge denominators are distinguishable.
Hygiene: generated `data/*` artifacts are gitignored (all seeded-deterministic rebuilds);
`STUDY_SUMMARY.md` deletion confirmed intentional.

## Status

| Steps | State |
|---|---|
| 0–7, 12 | **Done and verified** — 217 offline tests, 11 live tests |
| **Experiment 1** | **Complete and answered** (table above, and in `FINDINGS.md`) |
| 9–11 | **Built and unit-tested, not executed** — blocked on a valid `OPENAI_API_KEY` |

## Running it

```bash
uv sync
uv run pytest                          # 263 offline tests
uv run pytest -m live                  # 11 network tests (CMR, Chroma, embeddings)

uv run python -m airm.queries --build       # 11 SME queries, ground truth
uv run python -m airm.corpus --build        # 500 records, GT-seeded, parity-checked
uv run python -m airm.format_cache --build  # optional: 2,590 records × 6 renderings
uv run python -m airm.exp1                  # Experiment 1 — needs no API key
uv run python -m airm.index --build         # six Chroma collections

# Experiment 2 — needs a working OPENAI_API_KEY and a running Ollama
uv run python -m airm.synth            # 120 synthetic queries
uv run python -m airm.exp2 --smoke     # 2 formats × 1 model × 5 queries
uv run python -m airm.exp2             # full matrix
uv run python -m airm.report           # FINDINGS.md
```

Experiment 1 requires no credentials and is fully reproducible today. Experiment 2 needs
`OPENAI_API_KEY` in `.env` (the key currently in the environment returns 401) and a
running Ollama with `qwen3.6` pulled.

To exercise the Experiment 2 pipeline without a cloud key, point both the model matrix and
the judge at Ollama — but note in any write-up that the LLM-graded metrics were then
judged locally, since the judge is a real caveat on those numbers:

```bash
AIRM_OPENAI_MODELS= AIRM_JUDGE_PROVIDER=ollama AIRM_JUDGE_MODEL=qwen3.6:latest \
  uv run python -m airm.exp2 --smoke
```
