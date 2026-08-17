# Next experiments: closing the gap against Plan.md

Roadmap for adding the experiments from `~/my_bbenson/cmr-ai-metadata/Plan.md` that this
repo does not yet cover, ordered by implementation cost given the infrastructure that
already exists. Written in the same spirit as `EXPERIMENTS.md`: each step has a concrete
**Verify** command, and nothing counts as done until it has run and passed.

- Plan source: `~/my_bbenson/cmr-ai-metadata/Plan.md` (13-experiment program)
- Current coverage: this repo implements the plan's experiments 2 (token efficiency),
  3 (semantic fidelity), 9 (retrieval), plus partial 4 (queries), 10 (RAG — built, not
  run), 11 (chunking — generic only), 12 (UMM-AI ≈ `mat`)
- Missing entirely: 1 (profiling), 5 (direct comprehension), 6 (schema context),
  7 (structured extraction), 8 (complexity stress), 13 (agent tasks)

## Why this ordering

Two facts about the existing codebase drive the ranking:

1. **`facets.py` already extracts programmatic ground truth.** The plan's question
   dataset (Exp 4) and extraction targets (Exp 7) ask for exactly the fields
   `facets(record)` already returns: short name, title, DOI, platforms, instruments,
   science keywords, temporal range, spatial bbox. Expected answers are therefore
   computable, not authored.
2. **The blocked OpenAI key does not block the highest-value work.** Direct comprehension
   (Exp 5) and structured extraction (Exp 7) score against programmatic ground truth with
   deterministic matching — no DeepEval judge needed — and the Ollama arm
   (`gpt-oss:20b`, `qwen3:8b`) is verified working (`EXPERIMENTS.md` Step 8).

Recommended sequence:

```
A. Exp 1  dataset profiling            (no LLM, pure Python)
B. Exp 8  complexity × tokens          (join A onto existing exp1 results)
C. Exp 4  question generation          (templates over facets, no LLM)
D. Exp 5  direct LLM comprehension     (Ollama-runnable today)
E. Exp 8  complexity × accuracy        (join A onto D's results)
F. Exp 7  structured extraction        (same loop as D, JSON output)
G. cheap extras: cost-at-scale projection, UMM-aware semantic chunking
H. later: Exp 6 schema context, Exp 13 agent tasks, STAC renderer
```

Everything through step F runs without the blocked OpenAI key.

---

## A. Experiment 1 — Dataset profiling (`airm/profile.py`)

**Effort: smallest. No LLM, no network — the raw records are already on disk.**

`data/cmr_cache/<concept-id>.json` holds all 2,590 raw CMR items. Profiling is a pure
function over each one.

### Metrics per record (from Plan.md)

| Metric | Source |
|---|---|
| file size (bytes) | `os.stat` on the cache file |
| character count | `len(json.dumps(umm))` |
| number of fields | count of scalar paths — reuse `coverage.flatten()` |
| number of arrays | recursive walk counting `list` nodes |
| maximum nesting depth | recursive walk |
| null fields / empty arrays | recursive walk |
| number of URLs | count of `RelatedUrls[]` + any value matching `^https?://` |
| ScienceKeywords / Platforms / Instruments counts | direct len() on UMM paths |
| temporal extent complexity | number of `TemporalExtents[].RangeDateTimes` + flags (`EndsAtPresentFlag`, single vs multiple ranges) |
| spatial extent complexity | number of bounding rectangles + polygons + points; presence of vertical/orbital extent |

### Complexity score and buckets

Composite score = weighted sum of z-scored components (depth, field count, array count,
temporal, spatial). Weights live in `config.py`, not hard-coded in the analysis
(mirrors the plan's AI-readiness-score rule). Bucket by quartile into
`low / medium / high / extreme`, so the buckets are equal-sized by construction and the
cut points are recorded in the output.

### Output

```
runs/<id>/dataset_profile.csv    # one row per record, all metrics + score + bucket
runs/<id>/profile_summary.json   # per-bucket stats, cut points, weights used
```

Also profile the **faceted payload** alongside the raw record (two column groups), since
this repo's formats render the projection — both complexities are relevant and the
raw-vs-faceted contrast is itself informative.

### Design notes

- Reuse `coverage.flatten()` for path enumeration rather than writing a second walker;
  one traversal implementation means one set of bugs.
- Deterministic — belongs in the default offline test suite with a pinned fixture record
  whose depth/counts are hand-computed.
- Do **not** stratify anything on the score yet; just emit it. Re-sampling the corpus by
  complexity would invalidate comparability with existing runs.

**Verify:** `uv run python -m airm.profile` writes 2,590 rows; hand-computed metrics for
`sample_data/C1206485320-ASF.json` match; bucket sizes are 25% ± rounding.

---

## B. Experiment 8, token half — complexity stress on token cost

**Effort: a merge and two plots once A exists.**

`runs/20260731T204026Z/exp1_tokens.csv` already has per-record token counts for all six
formats, keyed by concept-id. Join against `dataset_profile.csv` and produce, per format:

- tokens vs complexity score (scatter + per-bucket means)
- tokens vs nesting depth (raw record depth and faceted depth)
- tokens vs field count
- **token-reduction-vs-JSON by bucket** — the plan's real question: does a format that
  looks fine on small records degrade on extreme ones?

Statistical treatment mirrors `exp1.py`: paired per-record ratios within each bucket,
with 95% CIs. Report per-bucket tables, not just the plot.

**Verify:** `uv run python -m airm.exp8 --tokens` writes `runs/<id>/exp8_tokens_complexity.csv`
+ plots; row count equals (records in both inputs) × 6 formats.

---

## C. Experiment 4 — Ground-truth question dataset (`airm/questions.py`)

**Effort: moderate, no LLM. Templates over `facets()`, answers programmatic by
construction.** This is *not* `synth.py` — synthetic retrieval queries find a record in a
corpus; these are per-record Q&A with field-derived expected answers.

### Categories, in build order

| Cat | Type | Template source | Expected answer |
|---|---|---|---|
| A | Simple lookup | "What is the short name / title / DOI / version of this collection?" | the facet value, verbatim |
| B | Nested metadata | "Which instruments / platforms / science keywords…" | the facet list; scored as set F1 |
| C | Temporal | "When does temporal coverage begin?" / "Does this dataset contain observations during {year}?" | ISO date string / boolean from range comparison; pick years both inside and outside the range |
| F | Negative | "Does this collection include {platform} observations?" where {platform} is drawn from *other* corpus records and verified absent from this record's full raw UMM (not just facets) | `false` / NOT FOUND — the hallucination probe |
| D | Spatial | "Could this dataset contain observations over {region}?" only when the record has a bounding rectangle; {region} from a small fixed table of named regions with bboxes (North America, Greenland, Gulf of Mexico, …); answer = bbox intersection test | boolean |
| E | Multi-field | "Does this collection provide {instrument} observations related to {keyword-topic}?" — requires two facets to co-occur | boolean |

Ship A, B, C, F first; D and E in a second pass (D needs the region table, E needs care
that the two fields genuinely both matter to the answer).

### Critical guards

- **Negative questions must be verified against the raw UMM record, not the facets.**
  A platform mentioned only in a `DEFERRED` path (e.g. inside `Projects` or a
  `RelatedUrl` label) would make "false" wrong even though the facets don't carry it.
  Check absence with a case-insensitive scan of the full raw JSON.
- **Skip, don't guess.** A record with no temporal extent generates no Category C
  question; the plan's rule ("only when the metadata provides enough information") is a
  hard gate, and skipped templates are counted in the build report.
- **Answer-type metadata drives scoring.** Each row records
  `answer_type ∈ {string, date, boolean, set}` so Exp 5 scoring is table-driven rather
  than per-question special cases.

### Output — plan's schema, verbatim

`data/questions.jsonl`, one row:

```json
{"question_id": "", "record_id": "", "category": "", "question": "",
 "expected_answer": "", "source_fields": [], "answer_type": "", "difficulty": ""}
```

Target 5–10 questions per record over a subset of the corpus (see Exp 5 sampling).
`source_fields` holds the UMM paths used — this doubles as the `ai_field → umm_path`
traceability mapping the plan asks Exp 12 for.

**Verify:** `uv run python -m airm.questions --build` reports per-category counts; a
sampled 20 questions manually spot-checked; every expected answer re-derivable by
re-running the generator (bit-identical rebuild under the same seed).

---

## D. Experiment 5 — Direct LLM comprehension (`airm/exp5.py`)

**Effort: moderate. All the hard parts already exist; runnable today on Ollama.**

The plan's core baseline, explicitly required *before* RAG: full record in context, no
embeddings, no retrieval.

### The loop

```
for record in sample:
  for fmt in six formats:                      # bytes from format_cache
    for question in questions[record]:
      for model in matrix:                     # llm.complete(), temperature 0
        prompt = PLAN_TEMPLATE(record_rendering, question)
        answer → score against expected_answer
```

Prompt is the plan's template verbatim (analyzing NASA metadata / use only the metadata
below / answer "NOT FOUND" if absent / do not infer). Version the prompt string
(`prompt_version` in every row) from day one.

### What gets reused

- `format_cache.load()` — exact bytes per (record, format), staleness-guarded
- `llm.complete()` — provider abstraction, retry policy, full audit logging to
  `logs/llm/<run_id>/`
- `exp2.py`'s per-cell JSONL checkpoint + `--resume` pattern — this matrix is bigger
  than exp2's and *will* be interrupted
- `runs.py` provenance stamping

### Scoring (deterministic — no judge, no OpenAI dependency)

By `answer_type`:

- `string`: exact match + normalized match (casefold, strip, collapse whitespace,
  fold unicode dashes — reuse the normalizer from `queries.py`)
- `date`: parse both sides, compare instants; accept date-only vs datetime
- `boolean`: normalized yes/no/true/false mapping; anything else = wrong
- `set`: precision / recall / F1 over normalized elements
- **NOT FOUND accuracy**: for questions whose answer is absent from the rendering,
  did the model say NOT FOUND? (This is where Category F earns its keep.)
- **Hallucination rate**: Category F questions answered "yes", plus non-NOT-FOUND
  answers to unanswerable questions

Also record per cell: latency, input tokens, output tokens (all already captured by the
`llm.py` logger — the trace metadata mechanism carries `question_id`, `fmt`).

### Sampling

Full corpus × 6 formats × ~8 questions × 2 models ≈ 48k calls — too many for a first
run. Stratified subsample of **100 records** (seeded, stored as
`data/exp5_sample.json`, stratified by topic *and* — once A lands — complexity bucket),
giving ≈ 9.6k calls per model. The sample file is committed so every later experiment
(Exp 6, Exp 7) uses the identical sample, per the plan's sampling rule.

### Output

```
runs/<id>/exp5_cells.jsonl        # checkpointed raw cells
runs/<id>/exp5_summary.json       # per (format × model × category) accuracy table
raw responses: already in logs/llm/<run_id>/ — no second copy
```

**Verify:** smoke run `--smoke` (2 formats × 1 Ollama model × 5 records) completes and
every cell traces to a logged call; then full run; summary accuracies for Category A on
`json` should be near-ceiling for any competent model — if they are not, suspect the
harness before the model.

---

## E. Experiment 8, accuracy half — complexity stress on comprehension

**Effort: another join.** Merge `exp5_cells.jsonl` with `dataset_profile.csv`:
accuracy vs token count, vs nesting depth, vs field count, vs bucket — per format, per
model. This completes the plan's Experiment 8. Watch for the confound the plan implies:
complexity correlates with record length, so report accuracy-vs-complexity *within*
token-count bands as well as overall.

**Verify:** `uv run python -m airm.exp8 --accuracy` produces per-format curves; n per
bucket reported (buckets with <10 records rendered as insufficient, not plotted as
trends).

---

## F. Experiment 7 — Structured extraction (`airm/exp7.py`)

**Effort: moderate — same skeleton as D with a different prompt and scorer.**

Ask each model to emit a JSON object with the plan's field list (title, short_name,
concept_id, version, science_keywords, platforms, instruments, temporal_start,
temporal_end, spatial_extent, processing_level, data_centers, download_urls — trimmed to
the fields the faceted payload actually carries, with the trim documented).

- **Ground truth is `facets(record)`** — already computed, already canonical.
- Parse with the fence/preamble-tolerant JSON reader from `synth.py`.
- Metrics per the plan: valid-JSON %, per-field extraction accuracy, missing-field rate,
  incorrect-value rate, **hallucinated-field rate** (fields present in output but absent
  from the rendering), array completeness (recall on list fields), schema compliance
  (validate against a small JSON Schema for the target object).
- Same 100-record sample as Exp 5, same checkpointing, same models.

**Verify:** smoke run; valid-JSON % > 0 for both providers; a deliberately corrupted
stub response is scored as invalid-JSON, not as zero-accuracy (broken ≠ wrong — the
`evaluate.py` principle).

---

## G. Cheap extras (each < half a day)

### G1. Cost-at-scale projection

Pure arithmetic over existing data: mean input/output tokens per query per
(format × model) from exp2/exp5 cells × `data/prices.json` × {1k, 10k, 1M} queries.
Emit `runs/<id>/cost_projection.csv`. Keeps the `llm.py` rule: no configured price →
`None`, never a fabricated zero.

### G2. UMM-aware semantic chunking (plan Exp 11, Strategy C)

The plan's semantic groups (IDENTITY / SCIENCE / OBSERVATION / TEMPORAL / SPATIAL /
PROCESSING / ACCESS) map ≈1:1 onto facet keys. Add a `chunks_semantic(record, fmt)`
that renders one chunk per facet group, and run it through the existing
`scripts/unfaceted_chunked_eval.py` harness as a third arm beside whole-record and
510-token windows. Same encoder, same 511 queries, same pooling. Answers the plan's
question "UMM-aware chunking > generic chunking?" with zero new evaluation code.

### G3. AI-readiness composite score

A `config`-driven weighted score over the component metrics as they land
(token efficiency + fidelity + retrieval exist today; comprehension, extraction,
hallucination arrive with D/F). Weights in YAML, raw components always emitted
alongside, no winner declared from the composite alone — all three rules straight from
the plan. Wire into `report.py`.

---

## H. Deferred (real design work — do after D/F produce data)

- **Exp 6 — schema context.** Straightforward *harness* (three prompt conditions reusing
  Exp 5's loop) but needs curated UMM schema documentation snippets per field, and its
  motivating question ("format or domain-model confusion?") is only worth asking once
  Exp 5 shows where models actually fail.
- **Exp 13 — agent tasks.** Genuinely new: task authoring, CMR query-validity checking
  (though `cmr.py` gives a live validator for free — execute the generated query and
  check the expected concept-ids appear), and multi-criterion scoring. Highest realism,
  highest cost.
- **STAC renderer.** A seventh format arm: renderer + parity rules + re-index. Mechanical
  but touches the parity gate, `format_cache` fingerprint, and every downstream matrix —
  do it only if the STAC comparison is actually wanted, and note it invalidates the
  format cache by design.

---

## Standing constraints (apply to every step above)

- **Temperature 0, seeds fixed, every run under `runs/<timestamp>/`** — never overwrite.
- **The faceted-projection caveat travels with every result.** All comprehension and
  extraction numbers measure the 32-field projection (~18% of source scalars), not the
  full UMM record. Say so wherever the numbers appear, exactly as `FINDINGS.md` does for
  Exp 1.
- **Content parity remains the gate.** Any new rendering path (semantic chunks, STAC)
  goes through `formats.validate_parity()` or an explicit documented exemption.
- **Failures are recorded, never averaged in as zeros.** Provider errors, parse
  failures and skipped templates each get their own counter.
- **Every LLM call is logged** through `airm.llm` — no exceptions, including any future
  judge.
