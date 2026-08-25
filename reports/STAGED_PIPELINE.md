# The staged evaluation pipeline: context, workflow, stages, artifacts

A standalone reference for how the faceted-vs-unfaceted RAG evaluation is built
and run. It explains *why* the work is split into stages, *what each stage does*,
and *what every artifact and field means*, so that a reader can reproduce a run,
audit any single number, or extend the pipeline without reading the code first.

Companion documents:

- [`ANALYSIS_50Q.md`](ANALYSIS_50Q.md) — the results and findings from run
  `20260814T210927Z` (what the numbers say; three models as of 2026-08-19).
- [`PIPELINE_50Q.md`](PIPELINE_50Q.md) — the status tracker for that run
  (what has been run, when, at what cost).
- [`CHUNKED_RETRIEVAL_511Q.md`](CHUNKED_RETRIEVAL_511Q.md) — the full
  511-query retrieval study Stage R produces.
- [`EXPERIMENTS.md`](EXPERIMENTS.md) / [`Report.md`](Report.md) — the study
  design and the literature it rests on.

Contents: [1. Context](#1-context) · [2. Design principles](#2-design-principles) ·
[3. The workflow](#3-the-workflow-at-a-glance) · [4. Stages](#4-the-stages) ·
[5. Artifacts and schemas](#5-artifacts-and-their-schemas) · [6. Audit chain](#6-the-audit-chain) ·
[7. Running and resuming](#7-running-and-resuming) · [8. Extending](#8-extending-the-pipeline)

---

## 1. Context

### The question

How should a metadata provider like NASA's Common Metadata Repository (CMR)
represent its collection records so that a large language model can **find** the
right dataset and **reason** about it well — and at what token cost?
[`Report.md`](Report.md) argues there is no a-priori best representation: it is
model-dependent and has to be measured. This pipeline is the measurement.

### What varies, what is held fixed

The experiment is a grid over three independent variables:

| Axis | Levels | What it tests |
|---|---|---|
| **Payload** | `faceted`, `unfaceted` | *What* metadata goes into context: a curated 32-field projection of the UMM record, or the raw record CMR actually serves (45 top-level fields, 398 paths). The projection is the "derived AI representation" a provider would publish; the raw record is the status-quo control. |
| **Format** | `json`, `yaml`, `toon`, `csv`, `jsonld`, `mat` | *How* that payload is serialised. All six renderings of a record carry identical facts (parity is hard-gated at cache build); `mat` is Metadata-as-Text, real prose. |
| **Model** | `gpt-5.4-nano` (OpenAI, cloud), `gpt-5.4-mini` (OpenAI, cloud, added 2026-08-19), `muse-glimmer:30b-mlx` (30B, local via Ollama) | Two independent model families plus a size step inside one of them, because the hypothesis is that the best format is model-dependent. The mini arm separates "cloud vs local" from "bigger vs smaller". |

Everything else is fixed so that differences are attributable to those axes:

| Held constant | Value |
|---|---|
| Corpus | 2,584 real CMR collection records, each rendered 6 formats × 2 payloads = 12 renderings, cached with a content fingerprint |
| Queries | 511 in `data/queries_full.jsonl` — 11 SME (expert-written research questions) + 500 synthetic (generated *from* indexed records, filtered for no identifier leakage and retrievability). The LLM stages use a seeded 50-query slice (all 11 SME + 39 synthetic); a 300-query slice that nests it (all 11 SME + 289 synthetic, same topic quotas) is selected in run `20260819T183313Z` and awaits answering. |
| Embedding | `BAAI/bge-large-en-v1.5`, 1024-dim, 512-token window |
| Chunking | 510-token windows, 64-token overlap, one vector per chunk; chunk hits pooled to records by `max` |
| Contexts per prompt | k = 10 records |
| Answer prompt | `airm.exp2.ANSWER_SYSTEM` — *"name the candidates that genuinely help answer the question… use ONLY the candidates shown… refer to each dataset by its title"* |
| Temperature | 0 everywhere (answering and judging) |
| Seed | 20260731 (query selection) |
| Judge | `gpt-5.4-nano` at temperature 0, DeepEval's five metrics, one fixed judge for every cell |

The grid is 6 formats × 2 payloads × 3 models = **36 arms**, each evaluated on
the same 50 queries = **1,800 cells** (the original two-model grid was 24 arms /
1,200 cells; the mini arm's 600 cells were appended to the same run). A *cell*
is one (query, format, payload, model) combination and is the unit of every
artifact downstream of retrieval.

### What is measured

Three kinds of number come out, from three different stages:

- **Exact retrieval metrics** (Stage R, no LLM): Recall@1/5/10, MRR, nDCG@10
  against ground-truth concept-ids.
- **Cost** (Stage A): prompt and completion tokens, latency, dollars per answer.
- **Judged quality** (Stage J): Correctness, Faithfulness, Answer Relevancy,
  Contextual Relevancy, Contextual Recall — five LLM-graded scores in [0, 1].

Only Correctness knows the ground truth and measures end-to-end success; the
others grade the answer's discipline or the retrieval's precision. See
`ANALYSIS_50Q.md` §C for the per-metric definitions.

---

## 2. Design principles

The pipeline is shaped by five rules. Every stage and artifact below follows from them.

1. **Stages produce frozen artifacts; the next stage reads, never recomputes.**
   Retrieval is ranked once and written to disk; answering replays those ranked
   lists; judging re-assembles contexts from the ids the answer stage logged.
   Nothing is silently re-run, so a score can always be traced to the exact
   inputs that produced it.
2. **Cheap-and-exact is separated from expensive-and-stochastic.** Retrieval is
   deterministic and ~40 s per payload; it runs on all 511 queries. LLM answering
   and judging cost hours and dollars; they run on a 50-query slice. The split
   means retrieval conclusions rest on n = 511 while the LLM stages still see
   every expert query.
3. **Same bytes, end to end.** Every render cache carries a manifest fingerprint.
   Stage R pins it; Stages A and J verify it before doing anything. If the cache
   has been rebuilt since retrieval, the stage refuses to run rather than feed
   the models different bytes than the ones retrieval ranked.
4. **Fail loudly, never average in a zero.** A judge failure records `None` for
   that metric; a model outage records an `error` on the cell. Means and paired
   comparisons skip `None`; they never count it as 0. Fingerprint mismatches and
   unavailable models stop the run.
5. **Never overwrite; always resumable.** Every run gets its own `runs/<timestamp>/`.
   Cells append as they finish and are flushed immediately; an interrupted run
   resumed with `--run-id` skips checkpointed cells. Every raw LLM call — prompt,
   response, tokens, latency — is logged per call in `runs/<rid>/llm/*.jsonl`.

---

## 3. The workflow at a glance

```
                       ┌─────────────────────────────────────────────────────────────┐
 data/queries_full     │ Stage R  Retrieval only, no LLM                             │
 (511 queries)    ───▶ │   embed query → search chunked index → max-pool → top-10    │
 data/chroma/*         │   score vs ground truth: R@k, MRR, nDCG                     │
                       └─────────────────────────────┬───────────────────────────────┘
                                                     │ runs/<R-faceted>/chunked_retrieval.jsonl
                                                     │ runs/<R-unfaceted>/chunked_retrieval.jsonl
                                                     ▼
                       ┌─────────────────────────────────────────────────────────────┐
                       │ Stage S  Query selection (runs inside Stage A)              │
                       │   all 11 SME + n−11 synthetic, topic-stratified, seeded;    │
                       │   --select-only stops here; --superset-of nests a run       │
                       └─────────────────────────────┬───────────────────────────────┘
                                                     │ runs/<A>/selection.json
                                                     ▼
                       ┌─────────────────────────────────────────────────────────────┐
 data/format_cache*    │ Stage A  Answers, replayed                                  │
 (fingerprint-    ───▶ │   per cell: read top-10 ids → load renderings → build       │
  checked)             │   prompt → call model @ T=0 → log tokens/latency            │
                       │   (--workers N for cloud cells; --seed-answers-from copies  │
                       │    a nested run's frozen cells instead of re-answering)     │
                       └─────────────────────────────┬───────────────────────────────┘
                                                     │ runs/<A>/answers.jsonl  (+ llm/answer.jsonl)
                                                     ▼
                       ┌─────────────────────────────────────────────────────────────┐
 data/format_cache*    │ Stage J  Judgement                                          │
 (fingerprint-    ───▶ │   per cell: re-assemble contexts → DeepEval × 5 metrics     │
  checked)             │   (or a --metrics subset) under one fixed judge             │
                       │   → scores, reasons, errors                                 │
                       └─────────────────────────────┬───────────────────────────────┘
                                                     │ runs/<A>/judgements.jsonl  (+ llm/judge.jsonl)
                                                     ▼
                       ┌─────────────────────────────────────────────────────────────┐
                       │ Analysis  join R + A + J on (query, fmt, payload, model)    │
                       │   per-arm table, paired Δ with 95% CIs, cost, Pareto charts │
                       └─────────────────────────────┬───────────────────────────────┘
                                                     │ runs/<A>/analysis.json, pareto.png, faithfulness.png
                                                     ▼
                                              ANALYSIS_50Q.md
```

One Stage A/J/Analysis run (`<A>`) reads **two** Stage R runs — one per payload.
For the reference run: R-faceted = `20260814T205849Z`, R-unfaceted =
`20260814T205912Z`, A = `20260814T210927Z` (nano + glimmer answered 2026-08-14/15,
mini appended 2026-08-19). The 300-query follow-on run is A = `20260819T183313Z`
(selection + seeded cells only so far), reading the same two Stage R runs.

---

## 4. The stages

### Stage R — Retrieval (no LLM)

**Script.** `scripts/unfaceted_chunked_eval.py eval`

**Input.** All 511 queries; one chunked ChromaDB index per payload
(`data/chroma/{faceted,unfaceted}_chunked/`), each holding one collection per
format; the render cache the index was built from.

**What it does, per (query, format, payload).**
1. Embed the query text with bge-large.
2. Query the format's collection for the top k×12 = 120 chunks (oversampling,
   because several chunks of one record may rank).
3. Pool chunk hits to a record score by `max` (a record is as good as its best
   chunk).
4. Rank records, take the top 10.
5. Score the ranked list against the query's ground-truth concept-ids:
   Recall@1/5/10, MRR, nDCG@10.

**Output.** `chunked_retrieval.jsonl` (one row per format × query, with the
ranked ids *and their pooled distances*), `chunked_summary.json` (config,
per-slice metric tables, and the render-cache fingerprint), `chunked_rows.csv`.

**Why it is its own stage.** It is deterministic, exact and cheap, so it runs on
the full query set and its ranked lists become the fixed input to every LLM
cell. Freezing them guarantees that all six (payload × model) arms of a given
(query, format) answer from identical contexts.

**Why retrieval quality and answer quality are measured separately.** A format
can be easy to embed (prose wins retrieval) and no easier to read back, or hard
to embed (JSON-LD) and fine once the right records are in context. The pipeline
measures both so they can be compared — and they turn out to disagree.

### Stage S — Query selection

**Script.** First step of `scripts/answer_stage.py`; `--select-only` runs it
alone (writes `selection.json`, makes no model calls).

**What it does.** Draws the LLM-stage slice from the 511: **all 11 SME queries**
plus **n − 11 synthetic**, stratified so the science-topic mix matches the full
set (largest-remainder quotas per topic, alphabetical tie-break, seed 20260731 —
the selection rebuilds byte-identically). Before selecting, it checks that both
Stage R runs carry the identical query set.

**Nesting a smaller slice.** A plain `--n 300` draw shares only 34 of the 50
reference queries (per-topic `random.sample` is not prefix-stable across sizes),
which would throw away their 1,800 answered and judged cells. `--superset-of
<run>` fixes that: within each topic, that run's queries are taken first and the
seeded sampler fills the rest of the quota. The 300-query slice in
`20260819T183313Z` was built this way — it contains all 50 reference queries and
its synthetic topic mix matches the 50 and the full set to within ~0.5 pp per
topic (quotas scale 1→7, 9→72, 10→72). The default (no flag) draw is unchanged,
so the reference 50 still rebuild byte-identically.

**Output.** `selection.json` — the chosen query ids, per-topic quotas, models,
k, git commit, the render-cache fingerprints of both retrieval runs, the
composition caveat written into the file itself, and (when nested)
`superset_of` / `superset_of_run`.

**Why 50, and why all the SME queries.** Judging costs ~22 judge calls per cell;
1,200 cells was ~$40 and a day of local-model latency, and the same grid at 511
queries would extrapolate to ~$400. Fifty keeps the cost bounded while holding
the hard expert slice at full strength.

**The caveat that travels with it.** SME queries are 22% of the 50-query slice
(3.7% of the 300) vs 2.2% of the full set and score far lower than synthetic
ones. Pooled absolute levels are therefore not comparable across slices or with
full-set runs; paired comparisons (which hold the query fixed) are unaffected.

### Stage A — Answers

**Script.** `scripts/answer_stage.py`

**Input.** The two Stage R run ids (`--retrieval`), the two format caches, the
model list.

**What it does, per cell** (query × format × payload × model):
1. Read the top-10 concept-ids for that (query, format, payload) from the Stage
   R row. **No new retrieval happens.**
2. Verify the format cache's manifest fingerprint against the one Stage R
   pinned; refuse to run on mismatch.
3. Load each id's rendering from the cache and build the prompt:
   `ANSWER_SYSTEM` + `Question: …` + the 10 candidates as a numbered list
   `[1]…[10]`, each candidate being the full rendering in that format.
4. Call the model at temperature 0. For Ollama, pass an explicit `num_ctx` sized
   from the prompt length (Ollama otherwise serves a ~4k window and silently
   truncates — a confound that would hit the unfaceted arm hardest).
5. Append the cell to `answers.jsonl` and flush (lock-protected, so concurrent
   workers never interleave a line).

**Concurrency.** `--workers N` answers the OpenAI cells N at a time (the
checkpoint and the raw-call log are both lock-protected). Ollama cells always
run one at a time regardless — the local server serialises requests anyway,
and a queue of 100k-token prompts would only time out. The mini arm was answered
at 24 workers.

**Seeding from a nested run.** `--seed-answers-from <run>` copies that run's
`answers.jsonl` (and `judgements.jsonl`, if present) rows that fall inside this
run's grid — selected queries × requested models × payloads × formats — into
this run before anything is answered. It refuses unless both runs are bound to
the same Stage R runs and render-cache fingerprints (so the copied answers were
produced from exactly the contexts this run would replay), skips cells with an
`error` or already present, is idempotent, and stamps each copied row with
`seeded_from`; the row's `run_id` stays the producing run's. The 300-query run
was seeded with the reference run's cells this way.

**Adding a model to an existing run.** Resume with `--run-id <run> --models
<new>`: the selection rebuilds identically, the existing cells' keys differ in
the model coordinate so they are untouched, and only the new arm's cells are
answered. This is how `gpt-5.4-mini` (600 cells) was added to the reference run
on 2026-08-19. Note `selection.json`'s `models` field records the *latest*
invocation's models, not the union.

**Output.** `answers.jsonl` (one row per cell — schema in §5.3),
`answers_summary.json` (mean tokens/latency/cost per payload × format × model,
regenerated after every invocation), `selection.json`, `llm/answer.jsonl`
(every raw call, failures included).

**Why it exists.** Stage R says whether the right record was in the top 10. It
cannot say whether a model can *read* a given format and pick the right dataset
out of ten candidates — which is the study's actual question. Stage A produces
the answers Stage J grades, and on its own it yields the **cost axis**: prompt
tokens per query (dollars for the cloud models) and latency (for the
prefill-bound local model).

### Stage J — Judgement

**Script.** `scripts/judge_stage.py`

**Input.** `answers.jsonl` (`--answers`); the format caches, fingerprint-checked
again, to re-assemble each cell's 10 contexts from its logged `retrieved` ids;
the query set, to build the expected answer
(`"The relevant collections are: <ground-truth titles>."`).

**What it does, per answered cell.** Run DeepEval's five metrics (or the
`--metrics` subset) under one fixed judge (default `openai:gpt-5.4-nano`,
temperature 0), `--workers N` cells concurrently. Roughly 22 judge calls per
cell for all five; ~5 for correctness + faithfulness alone, which is how the mini
arm was judged:

| Metric | Sees | ~Calls |
|---|---|---:|
| Correctness (GEval) | question, answer, expected answer | 1–2 |
| Faithfulness | answer, contexts | 3–4 |
| Answer Relevancy | question, answer | 3 |
| Contextual Relevancy | question, contexts | ~10–11 |
| Contextual Recall | expected answer, contexts | 2 |

Each metric's score and the judge's rationale are recorded. A metric whose judge
call fails (typically the judge emitting unparseable JSON) is recorded as `None`
with the error string — never as 0. The checkpoint key includes the judge id, so
a second judge's pass over the same cells coexists in the file; `--report`
summarises and, when two judges exist, prints their per-cell correlation.

**Output.** `judgements.jsonl` (schema in §5.4), `judge_summary.json`,
`llm/judge.jsonl`.

**Judge constraint.** The local models under test must not judge — a model
grading its own answers biases the model-vs-model axis. The chosen judge, nano,
is nevertheless one of the three answering models (and a sibling of a second,
mini), so the script records a `self_judge` caveat. Comparisons *within* a model
are clean; the model-vs-model comparisons are the ones to read with care. In the
reference run the observed ordering is glimmer > nano > mini — the local
competitor is rated above the judge's own answers, which argues against simple
self-preference, but nano-vs-mini cannot be separated from "the judge prefers
nano-style answers" with this judge alone. A ~$2 `gpt-5-mini` calibration slice
(or a non-OpenAI judge) is the direct check.

**Metric subsets and the checkpoint.** The checkpoint key is (cell, judge), not
(cell, judge, metrics): a cell judged on two metrics is "done" and a later
invocation with the full five will skip it. To add metrics to already-judged
cells, filter to them and judge under a different `--judge` id, or extend the
key. The mini cells currently carry correctness and faithfulness only.

### Analysis

**Script.** `scripts/analyze_50q.py`

**What it does.** Joins the three frozen artifacts on (query, format, payload,
model): exact retrieval metrics from R, tokens/latency/cost from A, the five
scores from J. Produces the per-arm table, the paired differences along each axis
(payload, model, format-vs-JSON) with 95% t-intervals over per-cell differences,
the dollar and latency tables, the SME/synthetic split, and the two Pareto
charts. Writes `analysis.json`, `pareto.png`, `faithfulness.png`.

The script is model-agnostic: the model axis differences every model against
the nano baseline and every other pair (`mini_minus_nano`, `glimmer_minus_nano`,
`mini_minus_glimmer`, each also split by payload), the Pareto charts get one
panel per model present, and dollar figures come from a `PRICES` table
(nano $0.20/$1.25, mini $0.75/$4.50 per M tokens in/out, OpenAI standard tier as
of 2026-08-19; local models have no dollar cost). Cells missing a metric (the
mini arm has no AnsRel/CtxRel/CtxRec) are skipped for that metric, never zeroed.

**Why paired.** Each comparison holds the other coordinates fixed and differences
the two arms on the *same cell*. That removes query-difficulty variance (SME
queries are 3–5× harder than synthetic), which is what makes n = 50 informative.
`None` metrics drop out pairwise.

---

## 5. Artifacts and their schemas

Directory layout for one full run (reference ids shown):

```
runs/20260814T205849Z/            Stage R, faceted
  chunked_retrieval.jsonl
  chunked_summary.json
  chunked_rows.csv
runs/20260814T205912Z/            Stage R, unfaceted
  (same three files)
runs/20260814T210927Z/            Stages S, A, J, Analysis (50 queries, 3 models)
  selection.json
  answers.jsonl                   1,800 cells
  answers_summary.json
  judgements.jsonl                1,800 cells
  judge_summary.json
  analysis.json
  pareto.png
  faithfulness.png
  llm/answer.jsonl                every raw answer call
  llm/judge.jsonl                 every raw judge call
runs/20260819T183313Z/            Stage S for the 300-query slice, + seeded cells
  selection.json                  300 ids, superset_of = the 50 above
  answers.jsonl                   1,200 rows copied from 20260814T210927Z (seeded_from)
  judgements.jsonl                1,200 rows copied likewise
```

(The 300-query run's seeded rows predate the mini arm: they hold the nano and
glimmer cells for the 50 nested queries. Re-running the seed command with
`--models openai:gpt-5.4-nano openai:gpt-5.4-mini ollama:muse-glimmer:30b-mlx`
copies the 600 mini cells too — seeding only copies models in the run's grid.)

### 5.1 `chunked_retrieval.jsonl` (Stage R) — one row per format × query

| Field | Example | Meaning |
|---|---|---|
| `query_id` | `"sme-001"` | Query identifier. |
| `source` | `"sme"` / `"synthetic"` | Query slice. |
| `topic` | `"OCEANS"`, `""` | GCMD science topic (empty for SME queries). |
| `query_text` | `"How can we assess agricultural drought…"` | The text that was embedded. |
| `payload` | `"faceted"` | Which index was searched. |
| `format` | `"json"` | Which collection within it. |
| `pool` | `"max"` | Chunk→record pooling rule used. |
| `expected` | `["C1235316218-GES_DISC"]` | Ground-truth concept-ids. |
| `retrieved` | `[{"concept_id": "C1214612351-SCIOPS", "distance": 0.3117}, …]` | Top-10 records in rank order with pooled distance — the replayable part. |
| `recall@1`, `recall@5`, `recall@10`, `mrr`, `ndcg@10` | `0.0 … 1.0` | Exact metrics for this row. |

### 5.2 `selection.json` (Stage S)

| Field | Meaning |
|---|---|
| `run_id`, `created`, `git_commit`, `git_dirty` | Provenance of the Stage A run. |
| `retrieval_runs` | Map of Stage R run id → `{payload, render_cache: {dir, fingerprint}}` — the caches and fingerprints this run is bound to. |
| `models`, `k`, `formats` | The arms this run will answer. |
| `query_ids` | The selected ids (50 in the reference run, 300 in `20260819T183313Z`). |
| `n`, `seed`, `sme`, `synthetic` | Slice size, seed, and composition (50 / 20260731 / 11 / 39; 300 / 20260731 / 11 / 289). |
| `quota` | Per-topic synthetic quotas from largest-remainder allocation. |
| `full_set` | `{queries: 511, sme: 11, synthetic: 500}` — what the slice was drawn from. |
| `sme_share_note` | The composition caveat, in words. |
| `superset_of`, `superset_of_run` | Only when built with `--superset-of`: the nested run's query ids and its run id. `null`/absent otherwise. |
| `model_problems` | Models that failed the availability check (empty on a clean run; `--select-only` skips the check). |

### 5.3 `answers.jsonl` (Stage A) — one row per cell, 1,800 rows in the reference run

**Provenance**

| Field | Example | Meaning |
|---|---|---|
| `run_id` | `"20260814T210927Z"` | The Stage A run that *produced* the cell. For a row copied by `--seed-answers-from` this stays the producing run's id, not the directory it now sits in. |
| `retrieval_run` | `"20260814T205849Z"` | The Stage R run this cell replayed — join key back to `chunked_retrieval.jsonl`. Differs by payload. |
| `seeded_from` | `"20260814T210927Z"` | Only on copied rows: the run the row was copied from. Absent on rows answered in place. |

**Grid coordinates**

| Field | Example | Meaning |
|---|---|---|
| `payload` | `"faceted"` / `"unfaceted"` | Projection or raw record in context. |
| `fmt` | `"json"`, `"yaml"`, `"toon"`, `"csv"`, `"jsonld"`, `"mat"` | Serialisation the model read. |
| `query_id` | `"sme-001"` | The query. |
| `source` | `"sme"` / `"synthetic"` | Query slice (for the SME-vs-synthetic split). |
| `topic` | `"OCEANS"`, `""` | GCMD topic; empty for SME. |
| `provider` | `"openai"` / `"ollama"` | Backend that served the model. |
| `model` | `"gpt-5.4-nano"` / `"gpt-5.4-mini"` / `"muse-glimmer:30b-mlx"` | The answering model. |

`(payload, fmt, query_id, model)` uniquely identifies a cell and is the join key
used by the analysis.

**What the model was given**

| Field | Meaning |
|---|---|
| `expected` | Ground-truth concept-ids for the query (carried along; Stage J derives the expected-answer string from these). |
| `retrieved` | The 10 concept-ids actually placed in the prompt, in rank order, copied from the Stage R row. Stage J re-assembles contexts from exactly these. |
| `prompt_sha256` | SHA-256 of `ANSWER_SYSTEM + "\x00" + prompt` — the full text sent. Proves which bytes produced an answer; the raw prompt itself is in `llm/answer.jsonl`. |

**What came back**

| Field | Meaning |
|---|---|
| `answer` | The model's full response text — what Stage J grades. |
| `prompt_tokens` | Input tokens reported by the provider. Source of the "Prompt tok" column and the Pareto x-axis. |
| `completion_tokens` | Output tokens. |
| `latency_s` | Wall-clock seconds for the call. |
| `cost_usd` | Tokens × the price table in `airm.llm` (`data/prices.json`). **`null` in every row of the reference run** — neither nano nor mini is in the loaded price table and Ollama has no price. The dollar figures in the analysis are computed by `analyze_50q.py` from tokens × its own `PRICES` table (nano $0.20/$1.25, mini $0.75/$4.50 per M). Add the models to `data/prices.json` before a rerun if you want this populated. |
| `error` | Error string if the call failed, else `null` (0 errors in the reference run). |

Not in the row: the Ollama `num_ctx` used — it is logged per call in
`llm/answer.jsonl`.

### 5.4 `judgements.jsonl` (Stage J) — one row per judged cell, 1,800 rows in the reference run

**Join key** — same meaning as in `answers.jsonl`: `run_id`, `payload`, `fmt`,
`query_id`, `source`, `provider`, `model` (the **answering** model, not the
judge).

**Who judged, with what**

| Field | Example | Meaning |
|---|---|---|
| `judge` | `"openai:gpt-5.4-nano"` | The judge model. Part of the checkpoint key, so a second judge's pass appends rows rather than overwriting. |
| `metrics` | `["correctness", "faithfulness", "answer_relevancy", "contextual_relevancy", "contextual_recall"]` | Metrics requested for this cell (a `--metrics` subset run lists fewer — the 600 mini cells list `["correctness", "faithfulness"]`). |

**The verdict**

| Field | Meaning |
|---|---|
| `scores` | Dict metric → float in [0, 1], or `null` if that metric's judge call failed. E.g. `{"correctness": 0.1, "faithfulness": 0.857, "answer_relevancy": 1.0, "contextual_relevancy": 0.242, "contextual_recall": 0.0}`. The fractions are DeepEval's ratios (0.857 = 6 of 7 claims supported). |
| `reasons` | Dict metric → the judge's free-text rationale for the score. The thing to read when a score looks surprising. |
| `errors` | Dict metric → error string, only for metrics that failed; `{}` otherwise. In the reference run 35 of the original 1,200 cells (2.9%) carry one, almost always `contextual_relevancy: "ValueError: Evaluation LLM outputted an invalid JSON…"`; 3 of the 600 mini cells carry a faithfulness error. The matching `scores[metric]` is `null` and is skipped, never zeroed. |
| `latency_s` | Seconds to judge the whole cell (all ~22 calls). Mean ~94 s in the original pass; far less for the two-metric mini cells. |

Not in the row: the judge's individual calls (prompts, responses, tokens) —
those are in `llm/judge.jsonl`, one line per call, which is where the judging
cost is totalled from. The expected-answer string and the contexts are not
stored; they are re-derived from `answers.jsonl` plus the fingerprinted cache.

### 5.5 `llm/answer.jsonl` and `llm/judge.jsonl` — one row per raw LLM call

| Field | Meaning |
|---|---|
| `ts` | ISO timestamp of the call. |
| `run_id`, `purpose` | The run; `"answer"` or `"judge"`. |
| `provider`, `model` | Who served the call. For judge calls this is the **judge**; the graded model is in `answer_model`. |
| `messages` | The full chat messages sent (system + user). |
| `response` | The full response text. |
| `prompt_tokens`, `completion_tokens`, `total_tokens` | Provider-reported usage. |
| `cost_usd` | From the price table (see `cost_usd` note above). |
| `latency_s`, `attempts`, `finish_reason`, `error` | Per-call outcome; failures are logged too. |
| `fmt`, `query_id`, `payload` | Which cell the call belongs to. |
| `answer_model` | Judge calls only: the answering model whose cell was being graded. |

### 5.6 Summary and analysis files

| File | Contents |
|---|---|
| `chunked_summary.json` | Stage R config (encoder, chunking, pooling), per-slice metric tables, render-cache fingerprint. |
| `answers_summary.json` | Mean tokens / latency / cost per payload × format × model; regenerated after every Stage A invocation. |
| `judge_summary.json` | Per-metric means per arm under each judge, error counts, the `self_judge` caveat; regenerated after every Stage J invocation. |
| `analysis.json` | Everything `ANALYSIS_50Q.md` tabulates: per-arm means, paired deltas with CIs, cost tables, slice split. |
| `pareto.png`, `faithfulness.png` | Correctness / faithfulness vs prompt tokens, one panel per model, faceted and unfaceted points of each format joined. |

---

## 6. The audit chain

Any judged score can be walked back to raw bytes without re-running anything:

```
judgements.jsonl   row (payload, fmt, query_id, model) → scores, reasons
      │
      ▼  same key
answers.jsonl      cell → retrieved ids, prompt_sha256, answer, tokens
      │
      ├──▶ llm/answer.jsonl   the raw call: full prompt, full response
      │
      ▼  (retrieval_run, fmt, query_id)
chunked_retrieval.jsonl   row → ranked ids with distances, exact metrics
      │
      ▼  selection.json → retrieval_runs[<rid>].render_cache.fingerprint
data/format_cache{,_unfaceted}/   the exact renderings the prompt was built from
```

And for the judge side, `llm/judge.jsonl` filtered on `(fmt, query_id, payload,
answer_model)` gives every judge prompt and response behind that cell's scores.

Cross-cutting guarantees that make the chain hold: content parity across formats
(hard-gated at cache build); temperature 0 everywhere; fingerprint checks before
every LLM stage; `None`-not-zero on failure; one directory per run, append-only.

---

## 7. Running and resuming

Reference commands (already run for the reference ids; ~40 s per payload for R,
hours for A on the local model, ~$40 and a few hours for J at 24 workers):

```bash
# Stage R — once per payload, all 511 queries
uv run python scripts/unfaceted_chunked_eval.py eval --payload faceted   --queries data/queries_full.jsonl --pool max --no-chart
uv run python scripts/unfaceted_chunked_eval.py eval --payload unfaceted --queries data/queries_full.jsonl --pool max --no-chart

# Stage S + A — one run over both retrieval runs and the two original models
uv run python scripts/answer_stage.py \
    --retrieval 20260814T205849Z 20260814T205912Z \
    --models openai:gpt-5.4-nano ollama:muse-glimmer:30b-mlx \
    --n 50 --k 10
#   --run-id <RID>   resume an interrupted run (skips checkpointed cells)
#   --limit N        cap the selected queries, for a quick pass
#   --smoke          2 formats × first model × 5 queries, faceted only
#   --workers N      answer OpenAI cells N at a time (Ollama cells stay sequential)
#   --select-only    Stage S only: write selection.json and stop (no model calls)
#   --superset-of R  make the slice contain runs/R/selection.json's queries, so a
#                    larger slice nests a smaller one and its cells can be reused
#   --seed-answers-from R  copy runs/R's in-grid answered (+judged) cells into this run

# Add a third model to the reference run (done 2026-08-19): same selection,
# existing cells untouched, 600 new mini cells at 24 workers
uv run python scripts/answer_stage.py \
    --retrieval 20260814T205849Z 20260814T205912Z \
    --n 50 --run-id 20260814T210927Z \
    --models openai:gpt-5.4-mini --workers 24

# Stage S only — the 300-query slice (all 11 SME + 289 synthetic, same topic
# quotas as the 50, and a superset of the 50-query reference run):
uv run python scripts/answer_stage.py \
    --retrieval 20260814T205849Z 20260814T205912Z \
    --n 300 --superset-of 20260814T210927Z --select-only
#   -> runs/20260819T183313Z/selection.json

# Seed the nested 50 queries' frozen cells (1,200 answers + 1,200 judgements)
# from the reference run, so only the 250 new queries need answering/judging.
# Refuses unless both runs pin the same Stage R runs and cache fingerprints;
# copied rows keep their original run_id and gain "seeded_from". Idempotent.
uv run python scripts/answer_stage.py \
    --retrieval 20260814T205849Z 20260814T205912Z \
    --n 300 --superset-of 20260814T210927Z --run-id 20260819T183313Z \
    --seed-answers-from 20260814T210927Z --select-only

# Stage A on the remaining 13,200 cells (drop --select-only; --models picks arms;
# --workers fans out the OpenAI cells, Ollama cells always run one at a time):
uv run python scripts/answer_stage.py \
    --retrieval 20260814T205849Z 20260814T205912Z \
    --n 300 --superset-of 20260814T210927Z --run-id 20260819T183313Z \
    --workers 24

# Stage J — judge the frozen answers (--answers takes the run id, not a path)
uv run python scripts/judge_stage.py \
    --answers 20260814T210927Z \
    --judge openai:gpt-5.4-nano --workers 24
#   --models/--payloads/--formats/--metrics   judge a subset
#   --limit N                                 first N cells (e.g. calibration slice)
#   --report                                  summarise; compare judges if two exist

# Judge only the new mini cells on the two headline metrics (done 2026-08-19;
# already-judged cells are skipped by the (cell, judge) checkpoint key)
uv run python scripts/judge_stage.py \
    --answers 20260814T210927Z \
    --judge openai:gpt-5.4-nano \
    --metrics correctness faithfulness --workers 24

# Judge calibration (optional, ~$2): a second judge on 60 cells, then compare
uv run python scripts/judge_stage.py --answers 20260814T210927Z \
    --judge openai:gpt-5-mini --limit 60
uv run python scripts/judge_stage.py --answers 20260814T210927Z --report

# Analysis (model-agnostic: one Pareto panel and one set of paired deltas per model present)
uv run python scripts/analyze_50q.py
```

Resumption rules: Stage A and J both checkpoint per cell and append; re-invoking
with the same run / answers file continues from where it stopped. Stage R is
cheap enough to simply re-run; a re-run into a new run id reproduces the prior
one exactly (one 1e-4 ANN tie-ordering difference was the only nondeterminism
ever observed).

---

## 8. Extending the pipeline

Because stages are decoupled by files, most extensions touch one stage:

- **A new format or payload** → rebuild the render cache (parity gate runs),
  re-index, re-run Stage R for that payload, then Stage A/J for the new arms
  only (`--formats` / `--payloads` filters). Existing cells are untouched.
- **A new answering model** → Stage A with `--run-id <existing> --models <new>`
  on the same retrieval runs, then Stage J on the new cells (the checkpoint key
  skips the old ones). Retrieval is reused as-is. Add the model's prices to
  `PRICES` in `analyze_50q.py` and a label to `MODEL_LABEL`; everything else
  (model-axis pairs, chart panels) adapts. This is exactly how `gpt-5.4-mini`
  was added.
- **A different judge** → Stage J with `--judge`; rows coexist with the nano
  rows, `--report` compares them per cell.
- **A bigger query slice** → Stage A with `--n N --superset-of <small run>`
  (a plain `--n` does *not* keep the smaller slice's queries — per-topic
  sampling is not prefix-stable), then `--seed-answers-from <small run>` so the
  nested queries' cells are copied rather than re-bought. Only the new queries
  cost anything. The 300-query run `20260819T183313Z` is at this point: selected
  and seeded, answers pending.
- **Only a subset of metrics** (e.g. dropping Contextual Relevancy, which Stage
  R already measures exactly, halves judge cost) → `--metrics`.

Things that *do* invalidate downstream artifacts: rebuilding the render cache
(fingerprint changes → Stages A and J refuse to run against old retrieval),
changing the encoder or chunking (→ re-run Stage R), changing `ANSWER_SYSTEM`
(→ prompt hashes change; re-run Stage A).
