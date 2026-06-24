# Experiment 1 — Format Ablation: Workflow, Methods, and Results

This document is a detailed walk-through of **what Experiment 1 builds, how it
scores, where each number comes from, and how to read the output**. It is meant
to be read alongside the design `README.md` and implementation `plan.md` in this
folder, and the authoritative spec in `Report.md` §6. The implementation lives
in `experiments/exp1_format_ablation/run.py` and leans on the shared `airm/`
harness.

---

## 1. The question under test

The project's thesis is that the *representation* (the textual format) of NASA
CMR metadata changes how well an LLM-based system can **retrieve**, **select
fields from**, and **reason over** that metadata — and that the best
representation is **model-dependent and must be measured**, not assumed.

Experiment 1 isolates the first link in that chain: **format → retrieval and
end-to-end answer accuracy**. It is a controlled *ablation*: a study where one
factor is varied while everything else is held fixed, so any change in the
outcome can be attributed to that factor.

- **Independent variable (the thing we vary):** the *renderer* — how a single
  CMR record is turned into a string.
- **Held constant:** the corpus of records, the query workload + ground truth,
  the embedding model, the retrieval algorithm, and (per model) the answering
  prompt.
- **Dependent variables (the things we measure):** retrieval quality
  (Recall@k, MRR, nDCG), end-to-end answer accuracy, token cost per record, and
  robustness of the format differences.

### 1.1 What a "record" and a "representation" are

A CMR record is a **UMM-JSON** item: a dict shaped like
`{"meta": {...}, "umm": {...}}`. `meta["concept-id"]` (e.g.
`C2264132902-GES_DISC`) is the stable identifier and the **ground-truth key** —
correctness is always decided by comparing concept-ids.

The four representations are the renderers registered in
`airm/representations.py` under `RENDERERS`, applied in this canonical order:

| key | what it produces | rough size |
|---|---|---|
| `raw_umm_json` | the canonical UMM record, pretty-printed JSON | large (~3.9k tokens/rec) |
| `flattened_jsonld` | a Schema.org `Dataset` / STAC-style flattened JSON-LD projection, nulls dropped | compact (~570 tokens/rec) |
| `dot_breadcrumb` | every leaf of the UMM tree as `a.b.c[0].d = value` breadcrumb lines | largest (~4.3k tokens/rec) |
| `metadata_as_text` | a natural-language "Metadata-as-Text" prose summary | smallest (~460 tokens/rec) |

### 1.2 Holding *content* constant while *format* varies

A fair ablation requires that the four renderers express **the same facts** —
otherwise we would be measuring content differences, not format differences.
`representations.py` enforces this with a shared extraction layer:

- A family of `extract_*` helpers (`extract_title`, `extract_summary`,
  `extract_platforms`, `extract_variables`, `extract_bbox`, `extract_temporal`,
  `extract_quality`, `extract_data_center`) pull the common facets out of the
  UMM tree **once**.
- `facets(record)` bundles them into a single dict — the common content payload.
- Each renderer then *arranges that same payload differently*. For example,
  `metadata_as_text` writes "Spatial coverage spans −180° to 180° longitude…",
  while `flattened_jsonld` emits a GeoJSON-style `"box": "south west north
  east"`, and `raw_umm_json` keeps the original nested
  `BoundingRectangles`. Same fact, four formats.

> Note: `raw_umm_json` and `dot_breadcrumb` operate over the *full* UMM tree, so
> they carry incidental extra fields that the two curated projections drop. This
> is intentional — it is part of what "raw" vs "curated" formats means in
> practice — but it is why those two are both larger and (as the results show)
> weaker.

**Renderers must be deterministic and must not call an LLM.** This is a hard
rule: retrieval has to be reproducible, and `metadata_as_text` is the target of
the auto-tune loop, so it stays a self-contained template.

---

## 2. Inputs the experiment loads

At startup `main()` loads two fixtures:

- **Corpus** — `load_corpus()` reads `data/corpus.jsonl`, real CMR collections
  spanning many Earth-science domains. (The latest run used 481 records; the
  number grows as the corpus is rebuilt.)
- **Queries + ground truth** — `load_queries()` reads `data/queries.yaml`. Each
  `Query` (defined in `airm/queries.py`) carries:
  - `id`, `question` (natural language),
  - `relevant` — the list of ground-truth concept-ids that correctly answer it,
  - optional `difficulty` (`easy`/`medium`/`hard`), `fields` (CMR fields needed
    to answer), `paper` (provenance), and `notes`.

`corpus_by_id` indexes the corpus by concept-id for O(1) lookup during the
answer stage. A reproducible random **sample** of queries (`--max-queries`,
default 20, seeded by `--seed`) is drawn for the expensive stages (answering and
auto-tuning); the cheap retrieval stage runs over **all** queries.

---

## 3. The pipeline, stage by stage

The run is an "honest RAG" pipeline (retrieve first, then answer only from what
was retrieved) followed by a validate-and-improve loop.

```
data/corpus.jsonl (records)  +  data/queries.yaml (questions + ground-truth ids)
        │
        ▼  for each of the 4 representations
   render every record → embed (local, deterministic) → cosine search → top-k ids
        │
        ├─►  recall@k / mrr / ndcg per query        ◄── computed here, NO LLM involved
        │         └─► summarize: per-rep means + bootstrap 95% CIs
        │         └─► robustness: re-run across seeds with paraphrased queries
        │
        └─►  top-k candidates → LLM picks one concept_id → graded accuracy (format × model)
                                                              └─► model-dependence verdict + Pareto frontier

   (separately) auto-tune: LLM hill-climbs the metadata_as_text template on Recall@k
```

### 3.1 Stage 1 — Retrieval (where the headline metrics come from, *before any LLM call*)

This is the most important part to understand: **the core metrics are produced
purely by embeddings and set arithmetic. No language model is prompted.**

For each representation, `evaluate_retrieval()` (in `airm/run.py`) does:

1. **Build a vector index.** `Index.build()` (`airm/embeddings.py`) renders
   every corpus record with that representation's `render_fn`, then embeds all
   the resulting strings with a **local** sentence-transformers model,
   `BAAI/bge-small-en-v1.5`. Embeddings are **L2-normalized**, so a dot product
   equals cosine similarity. Running locally makes retrieval free, offline, and
   reproducible (the first run downloads the model, ~tens of MB, which is why a
   cold test takes ~40s).

2. **Score every query.** Each query's `question` text is embedded the same way;
   `Index.search()` computes `matrix @ q` (one cosine score per record),
   `argsort`s descending, and returns the top-k concept-ids (`--k`, default 10).

3. **Compute retrieval metrics** by comparing the ranked concept-id list against
   the ground-truth set `q.relevant`. All three live in `airm/metrics.py` and
   are pure rank/set math:

   - **Recall@k** — `recall_at_k`: of the relevant items, what fraction appear
     in the top-k. `hits / len(relevant)`. Answers *"did retrieval surface the
     right datasets at all?"*
   - **MRR (Mean Reciprocal Rank)** — `mrr`: `1 / rank` of the *first* relevant
     hit (0 if none). Rewards putting a correct answer near the top.
   - **nDCG@k (normalized Discounted Cumulative Gain)** — `ndcg_at_k`: a
     rank-discounted score (each hit contributes `1/log2(rank+1)`) normalized by
     the ideal ordering. A graded measure of *how well-ordered* the hits are.

Each query yields one `RetrievalResult` (representation, query_id, the three
metrics, and the `top_ids`). All results across all representations are written
to `retrieval.jsonl`, and the `top_ids` are stashed in a `retrieved[(rep,
query_id)]` map for reuse by the answer stage (so the LLM is graded on exactly
the candidates retrieval produced — no leakage).

#### Aggregation and confidence intervals

`summarize()` (`airm/run.py`) groups results by representation and reports, per
representation:

- **mean Recall@k with a bootstrap 95% confidence interval.** A *confidence
  interval* expresses how much the mean would wobble if we resampled the
  queries. `metrics.bootstrap_ci` does a **percentile bootstrap**: it resamples
  the per-query recall scores with replacement 1000 times, takes the mean of
  each resample, and reports the 2.5th and 97.5th percentiles as `[lo, hi]`.
  Overlapping CIs between two formats mean we cannot confidently say one beats
  the other.
- mean MRR and mean nDCG.

#### Token cost

`tokens_per_record()` renders a sample of records (cap 100) under each format
and averages `metrics.count_tokens` (tiktoken, `cl100k_base`, with a ~4-chars/token
fallback). This is the **cost axis** for the later Pareto analysis — bigger
renderings cost more to embed and to stuff into a prompt.

#### Why the index is rebuilt — and how the embedding cache makes that cheap

`Index.build` does not load a prebuilt vector store; it **renders every record
and embeds from source** each time it is called. Within one run that happens
several times over the *same* corpus: once per representation in the retrieval
stage, again per representation in the robustness stage (an identical index),
and once per template in the auto-tune loop. This is deliberate — it guarantees
the vectors always match the current corpus *and* the current renderer code,
which is essential for auto-tune (the renderer text changes every round, so any
index keyed only on "the corpus" would be silently wrong).

To avoid paying for that redundancy, `embed()` (`airm/embeddings.py`) is backed
by a **content-addressed disk cache** under `data/embed_cache/` (gitignored).
The cache key is a SHA-256 of the embedding-model name plus the exact texts, so
a hit is returned only for byte-identical input. The retrieval and robustness
indices for a given representation therefore embed once and reuse; change the
corpus or any renderer (including each tuned template) and the texts differ, the
key differs, and the vectors are recomputed — correct by construction, never
stale. Query vectors are cached the same way, so a question embedded once is
reused across all four representations. In practice a warm rebuild is ~70× faster
than a cold one. Bypass the cache with `AIRM_NO_EMBED_CACHE=1`, and delete
`data/embed_cache/` any time to reclaim space (it is rebuilt on demand).

### 3.2 Stage 2 — Robustness (still no end-to-end answering)

A single retrieval score could be a fluke of exact query wording. The robustness
stage (`run_robustness` → `airm/improve.py`) asks: *are the format differences
real, or noise?*

For each representation it re-runs retrieval over the sampled queries across
**three seeds** `(0, 1, 2)`, and at each seed perturbs the queries with
`paraphrase_queries` (default `shuffle_words`: lightly reorders the interior
tokens of each question while preserving the first/last word **and the
ground-truth relevance set**). `robustness_check` pools the per-query recall
scores across seeds and computes a bootstrap CI (`RobustnessReport`).

Then `cis_overlap` checks each pair of representations: if their Recall@k CIs
**overlap**, the difference between those two formats is flagged as **not
robustly distinguishable**. This is the experiment's built-in skepticism — it
prevents reading a meaningful gap into two formats that are statistically tied.

### 3.3 Stage 3 — Answer (the only stage that calls an LLM for grading)

This stage measures **end-to-end answer accuracy**: given retrieved candidates,
can a model pick the right dataset?

Model selection:
- `--models` lets you pin a comma-separated list; otherwise `available_models()`
  returns every configured tier (keyed cloud providers + locally pulled Ollama
  models).
- A provider that isn't configured raises `ProviderUnavailable`, which is caught
  to **skip that tier and log it** — a sweep never crashes over a missing key.
- The report design wants **≥3 models** (to expose model-dependence); the runner
  logs a `WARNING` if fewer are available but still proceeds.

For each `(representation, sampled query)`, `run_answer_stage`:

1. Takes the top `--answer-k` (default 5) retrieved concept-ids for that
   `(rep, query)` from the retrieval map.
2. Re-renders those candidate records **in the same representation**, each
   tagged with its `concept_id`, into a prompt (`build_answer_prompt`). The
   system prompt (`ANSWER_SYSTEM`) frames the model as a NASA metadata search
   assistant and demands a bare concept-id reply.
3. Calls each model via `airm.llm.complete()` (`max_tokens` default 512 — generous
   so a reasoning model can finish a `<think>` block *and* still emit the id).
4. Parses the answer with `extract_concept_id`, which first runs `clean_answer`
   to **strip any `<think>…</think>` preamble** (keeping only text after the last
   `</think>`) so a concept-id the model merely *considered* while reasoning is
   not mistaken for its final pick, then regex-matches the `C\d+-[A-Z0-9_]+`
   pattern.
5. Grades `correct = predicted in set(q.relevant)`.

Every row records the prediction, correctness, prompt/output token counts, the
raw output, and whether a relevant id was even among the candidates
(`retrieved_in_candidates`) — which separates *retrieval* failures from
*selection* failures. Rows are written to `answers.jsonl`, and with verbose
logging on (default), every prompt and raw model output is streamed to the run
log.

### 3.4 Stage 4 — Auto-tune (agentic improvement of one renderer)

The validate-and-improve loop's second half (`run_auto_tune` →
`improve.auto_tune`) tries to *improve* the `metadata_as_text` format rather than
just measure it.

- The optimization target is a **templated** version of the prose renderer:
  `DEFAULT_TEMPLATE`, a string with named placeholders (`{title}`,
  `{data_center}`, `{instruments}`, `{platforms}`, `{variables}`, `{spatial}`,
  `{temporal}`, `{quality}`, `{summary}`) filled by `flatten_facets()`.
- `score_fn(template)` builds an index from that template and returns mean
  Recall@k over the sampled queries; an invalid template (bad placeholder /
  format error) scores `-inf` and is rejected.
- `auto_tune` is a **hill-climb**: each round it shows the current best template
  and its score to the optimizer model (`models[0]`), asks for a rewrite
  (constrained to keep the placeholders, via the `context` string), re-scores
  it, and **keeps the revision only if it scores higher**. It stops after
  `--tune-rounds` rounds or `patience` (2) non-improving rounds, whichever comes
  first.
- Guardrail: the tuner only ever mutates the text passed to it; it never edits
  source files. The winning template is persisted to
  `tuned_metadata_as_text.txt` by the runner.

---

## 4. Outputs and how to read them

Each run creates a fresh timestamped directory `results/<UTC-timestamp>/` (and
repoints a `results/latest` symlink), containing:

| file | contents |
|---|---|
| `retrieval.jsonl` | one row per (representation, query) with the three retrieval metrics and `top_ids` |
| `answers.jsonl` | one row per (representation, model, query) with prediction, correctness, token counts, raw output |
| `tuned_metadata_as_text.txt` | the best template found by auto-tune |
| `run-<ts>.log` | the full verbose run log (status lines + every prompt/output) |
| `summary.md` | the human-readable report assembled by `write_summary` |

`write_summary` produces five sections:

1. **Retrieval table** — Recall@k + CI, MRR, nDCG, tokens/rec, sorted by recall.
2. **Robustness** — which format pairs have overlapping CIs (not distinguishable).
3. **End-to-end answer accuracy** — a format × model accuracy matrix.
4. **Model-dependence verdict** — the best format *per model*; if more than one
   distinct winner appears across models, it prints **"Model-dependent"**
   (supporting the thesis), else **"Consistent"**.
5. **Pareto frontier** — over `(accuracy, token cost)` points,
   `metrics.pareto_frontier` marks the formats that are not *dominated* (no other
   format is both at-least-as-accurate **and** at-least-as-cheap). These are the
   defensible choices on the accuracy-vs-cost trade-off.

### Running it

```
uv run python -m experiments.exp1_format_ablation.run [options]
```

Useful flags: `--k` (retrieval depth, 10), `--answer-k` (candidates shown, 5),
`--max-queries` (sample size, 20), `--seed` (0), `--tune-rounds` (3),
`--models "a,b,c"` (pin tiers), `--no-answer`, `--no-tune`, `--no-verbose`.

---

## 5. Latest results (run `20260624T203903Z`)

This run scored **481 corpus records** against **554 queries** for retrieval,
with a **20-query sample** for the answer and auto-tune stages. Only **one
model** (`gpt-5.4-nano`) was configured — below the design's ≥3 target, which
the runner flagged with a `WARNING`; the model-dependence verdict is therefore
not yet meaningful (see caveats).

### 5.1 Retrieval (all 554 queries)

| representation | Recall@10 | 95% CI | MRR | nDCG | tokens/rec |
|---|---|---|---|---|---|
| `metadata_as_text` | **0.883** | [0.856, 0.908] | **0.683** | **0.731** | **463** |
| `flattened_jsonld` | 0.812 | [0.780, 0.845] | 0.588 | 0.642 | 574 |
| `dot_breadcrumb` | 0.592 | [0.551, 0.635] | 0.399 | 0.446 | 4267 |
| `raw_umm_json` | 0.543 | [0.500, 0.585] | 0.370 | 0.412 | 3888 |

**Reading it:** the two *curated, compact* formats clearly beat the two *raw,
bulky* ones on every retrieval metric, and do so at ~7–9× lower token cost.
`metadata_as_text` (natural-language prose) leads outright — its CI [0.856,
0.908] sits entirely above `flattened_jsonld`'s [0.780, 0.845], so that gap is
statistically clean. The two bulky formats land at the bottom: dumping the full
UMM tree (raw JSON or dot-breadcrumbs) dilutes the embedding with boilerplate and
structural noise, hurting both recall and ranking. The headline takeaway: **for
embedding-based retrieval, prose and tight JSON-LD dominate raw dumps on both
quality and cost.**

### 5.2 Robustness — are the differences real?

Pairs whose Recall@k CIs **overlap** (difference *not* robustly distinguishable):

- `raw_umm_json` ≈ `dot_breadcrumb`
- `flattened_jsonld` ≈ `dot_breadcrumb`

So the *two weak formats* (`raw_umm_json`, `dot_breadcrumb`) are statistically
tied with each other, and `dot_breadcrumb` is close enough to `flattened_jsonld`
under paraphrase that we shouldn't over-claim a gap there. But
`metadata_as_text`'s lead is **not** in the overlap list — it stands apart from
the field, which is the robust, defensible finding.

> The robustness recall figures in the run log (e.g. `metadata_as_text`
> recall=0.967 CI[0.917, 1.000]) differ from the §5.1 table because robustness
> runs on the **20-query sample with shuffled-word paraphrases across 3 seeds**,
> whereas the table is plain retrieval over **all 554 queries**. They are
> measuring related but different things; don't expect them to match.

### 5.3 End-to-end answer accuracy (format × model)

| representation | gpt-5.4-nano |
|---|---|
| `metadata_as_text` | **0.90** |
| `flattened_jsonld` | 0.75 |
| `dot_breadcrumb` | 0.60 |
| `raw_umm_json` | 0.40 |

The answer-stage accuracy ordering **mirrors the retrieval ordering exactly**:
the format that retrieves best also lets the model pick best. `metadata_as_text`
reaches 0.90 — the model both *sees* the right candidate more often and *reads*
the prose more reliably than raw JSON. `raw_umm_json` is worst (0.40): even when
a correct candidate is present, the verbose nested JSON is harder for the model
to adjudicate.

### 5.4 Model-dependence verdict

- best format for **gpt-5.4-nano**: `metadata_as_text`
- **Consistent**: 1 distinct best-format across 1 model.

This is reported as "Consistent" only because a single model was configured. The
central thesis — that the best format *varies by model* — **cannot be confirmed
or refuted from this run.** It needs the ≥3-model sweep the design calls for
(e.g. a cloud tier plus a couple of local Ollama models).

### 5.5 Pareto frontier (accuracy vs token cost)

- `raw_umm_json`: acc=0.40, tokens/rec=3888
- `flattened_jsonld`: acc=0.75, tokens/rec=574
- `dot_breadcrumb`: acc=0.60, tokens/rec=4267
- `metadata_as_text`: acc=0.90, tokens/rec=463 **⬅ frontier**

`metadata_as_text` is the *only* point on the frontier: it is simultaneously the
**most accurate and the cheapest**, so it dominates every other format on both
axes at once. There is no accuracy-vs-cost trade-off to negotiate here — the
prose format is the unambiguous pick for this corpus and this model.

### 5.6 Auto-tune

- baseline templated Recall@k: **0.900**
- tuned Recall@k: **0.950** (achieved in round 1; then plateaued)
- winning template saved to `results/tuned_metadata_as_text.txt`:

  ```
  {title} — NASA Earth-science dataset at {data_center}. Summary: {summary}.
  Instruments: {instruments}. Platforms: {platforms}. Variables: {variables}.
  Spatial coverage: {spatial}. Temporal coverage: {temporal}. Data quality: {quality}.
  ```

The optimizer found a rewrite that lifted sampled Recall@k from 0.900 → 0.950 on
its first try — a labeled-prefix, semicolon-light phrasing that front-loads the
title and summary. It then plateaued (no further gain within patience), so the
loop stopped. **Caveat carried in the summary:** the tuned template's effect on
*cross-model answer accuracy* was not re-evaluated in this run — that is a
documented follow-up.

### 5.7 Caveats for this run

- **Single model.** Only `gpt-5.4-nano` was available, below the ≥3 the design
  wants. The model-dependence question — the project's headline claim — remains
  open until more tiers are configured.
- **Small answer/tune sample.** Answer accuracy and auto-tune used a 20-query
  sample; the percentages are indicative, not tight estimates. Retrieval and its
  CIs, by contrast, use all 554 queries and are solid.
- **Ground truth is partly auto-seeded.** `queries.yaml` started from heuristic
  seeds meant to be hand-checked and broadened; some `relevant` sets may be
  narrow, which can understate recall for queries that legitimately have several
  correct datasets.
- **Tuned template not re-scored end-to-end.** The +0.05 recall gain is a
  retrieval-only result.

### 5.8 Bottom line

For this corpus and model, **format matters a lot and the natural-language
`metadata_as_text` representation wins outright** — best retrieval, best answer
accuracy, lowest token cost, sole point on the Pareto frontier — and its lead is
robust to query paraphrasing. Raw, exhaustive dumps (`raw_umm_json`,
`dot_breadcrumb`) are both the most expensive and the weakest. The one thing this
run *cannot* yet establish is whether that winner holds across models; that
requires the multi-model sweep.

---

## 6. Three-model sweep (run `20260624T210800Z`)

This run repeats §5 with the **≥3-model answer stage the design calls for**,
configuring three OpenAI tiers — `gpt-5.4-nano`, `gpt-4.1-mini`, `gpt-4o-mini` —
via:

```
uv run python -m experiments.exp1_format_ablation.run --models "gpt-5.4-nano,gpt-4.1-mini,gpt-4o-mini"
```

Same **481 records / 554 queries**, same 20-query sample. Retrieval, robustness,
and auto-tune are **model-independent** (retrieval embeds locally; the optimizer
is `models[0]` = `gpt-5.4-nano`, unchanged from §5), so §5.1, §5.2, and §5.6
carry over **identically**. Only the answer stage changes — and with three
models present, the **model-dependence verdict is now meaningful** for the first
time.

### 6.1 End-to-end answer accuracy (format × model)

| representation | gpt-5.4-nano | gpt-4.1-mini | gpt-4o-mini |
|---|---|---|---|
| `metadata_as_text` | **0.85** | **0.90** | **0.85** |
| `flattened_jsonld` | 0.80 | 0.75 | 0.75 |
| `dot_breadcrumb` | 0.60 | 0.60 | 0.60 |
| `raw_umm_json` | 0.40 | 0.40 | 0.40 |

The accuracy ordering is **stable across all three models** and still mirrors the
retrieval ordering: `metadata_as_text` on top (0.85–0.90), then
`flattened_jsonld`, then `dot_breadcrumb`, with `raw_umm_json` last at a flat
0.40 for every model. Per-model wobble is small and confined to the middle
(`gpt-5.4-nano` edges `flattened_jsonld` to 0.80; `gpt-4.1-mini` peaks
`metadata_as_text` at 0.90) — exactly the kind of noise expected from a 20-query
sample.

### 6.2 Model-dependence verdict

- best format for **gpt-5.4-nano**: `metadata_as_text`
- best format for **gpt-4.1-mini**: `metadata_as_text`
- best format for **gpt-4o-mini**: `metadata_as_text`
- **Consistent**: 1 distinct best-format across 3 models.

This now clears the design's ≥3-model bar, so the verdict carries weight:
`metadata_as_text` is the best representation for **every** model tested. The
thesis allows for a model-dependent winner; here, across three models, the
winner does **not** vary. Important scope caveat: all three are **OpenAI-family**
models, so this tests robustness *within* a family, not *across* families — a
local Ollama tier or a Claude tier could still flip the ranking, and that
cross-family sweep remains the open question.

### 6.3 Pareto frontier (accuracy vs token cost)

Accuracy is now averaged across the three models:

- `raw_umm_json`: acc=0.40, tokens/rec=3888
- `flattened_jsonld`: acc=0.77, tokens/rec=574
- `dot_breadcrumb`: acc=0.60, tokens/rec=4267
- `metadata_as_text`: acc=0.87, tokens/rec=463 **⬅ frontier**

`metadata_as_text` remains the **sole frontier point** — most accurate *and*
cheapest — under the three-model average, reinforcing §5.5.

### 6.4 What's new vs §5

The single open question §5 flagged — *does the winner hold across models?* — is
now answered **for the OpenAI family: yes**. `metadata_as_text` wins on all three
tiers, on both accuracy and cost. The remaining gap is cross-*family* coverage
(Claude / local Ollama), which would stress whether the prose format's lead is a
property of the representation itself or of OpenAI tokenization/training.
