# Analysis: 50 queries, faceted vs unfaceted, retrieval → answers → judgement

**Run `20260814T210927Z`** — 1,800 answered cells (50 queries × 6 formats × 2
payloads × 3 models), every cell judged by `gpt-5.4-nano`, joined to exact
retrieval metrics from runs `20260814T205849Z` (faceted) and `20260814T205912Z`
(unfaceted). Status date 2026-08-19: the `gpt-5.4-mini` arm (600 cells) was
added on 2026-08-19 and judged on correctness and faithfulness only; the
original 1,200 nano/glimmer cells carry all five metrics.

**In one paragraph.** Putting a curated 32-field projection of a CMR record in
front of the model produces *more* correct answers than the full raw record
(+0.04 correctness, CI excludes zero, and the same +0.04 inside each of the
three models) at 1.9–3.7× fewer prompt tokens; the local 30B model beats the
cloud nano model by +0.07 even under nano's own judging, and earns that on the
hard expert questions; the larger cloud model, gpt-5.4-mini, scores *below*
nano (−0.05) at 3.6× the price; within a payload, format matters little for
answering (CSV, JSON-LD and TOON are measurably worse than JSON by 0.03–0.04)
even though it matters a lot for retrieval and for cost. The cheapest
configurations are also the most accurate.

- Script: [`scripts/analyze_50q.py`](../scripts/analyze_50q.py)
- Data: `runs/20260814T210927Z/{answers,judgements}.jsonl`,
  `runs/{20260814T205849Z,20260814T205912Z}/chunked_retrieval.jsonl`
- Tables: `runs/20260814T210927Z/analysis.json` · Charts:
  `runs/20260814T210927Z/pareto.png` (correctness vs context cost) ·
  `runs/20260814T210927Z/faithfulness.png` (faithfulness vs context cost)
- Pipeline detail and per-stage commands: [`PIPELINE_50Q.md`](PIPELINE_50Q.md)

Sections: [A. Context](#a-context--what-was-being-tested-and-why) ·
[B. Workflow](#b-the-workflow--four-stages-then-analysis) ·
[C. Reading the numbers](#c-how-to-read-the-numbers) · [0. Raw results](#0-raw-results--every-arm-all-metrics) ·
[1–6. Findings](#1-headline-the-projection-beats-the-raw-record-at-a-fraction-of-the-cost) · [Caveats](#caveats)

---

## A. Context — what was being tested, and why

**The question.** How should a metadata provider like NASA's Common Metadata
Repository (CMR) represent its collection records so that a large language model
can *find* the right dataset and *reason* about it well — and at what token
cost? [`Report.md`](Report.md) argues there is no a-priori best representation:
it is model-dependent and has to be measured. This run is the measurement.

**The corpus.** 2,584 real CMR collection records pulled from the public search
API, each rendered 6 ways × 2 payloads = 12 renderings per record, all cached
with a content fingerprint so any later stage can prove it is reading the same
bytes.

**Two payloads — *what* goes into the representation.**

| Payload | What it is | Size |
|---|---|---|
| **faceted** | A fixed 32-field projection of the UMM record (`airm.facets`): title, abstract, platform, instrument, science keywords, temporal/spatial extent, resolution, processing level, DOI, data centre, version, quality fields, … One facet per UMM field, nothing aggregated, nothing pre-rendered, CMR placeholder strings (`"Not provided"`) dropped as absences | ~18% of the raw record's scalar values; 786–1,465 bge tokens per record depending on format |
| **unfaceted** | The record CMR actually serves — all 45 top-level UMM-C fields, 398 distinct paths, contacts, addresses, URLs, metadata dates, everything (`airm.unfaceted`) | 1,462–7,012 bge tokens per record |

The faceted payload is the study's main line — the "derived AI projection" a
provider would publish. The unfaceted payload is the control: the status quo,
and the answer to "does any of this survive if you distrust the projection?"

**Six formats — *how* the payload is serialised.** Content parity is hard-gated
at cache build: all six renderings of a record carry identical facts (machine
formats decode to the same normalised fact set; prose is checked for 100% value
coverage).

| Format | Rendering |
|---|---|
| `json` | `json.dumps` of the payload tree — the baseline every format comparison is relative to |
| `yaml` | block-style `yaml.safe_dump`, key order preserved |
| `toon` | the official `toon_format` encoder, unpatched |
| `csv` | two-column `path,value` long form over the flattened tree — the only lossless CSV for a nested record |
| `jsonld` | schema.org `Dataset`; for the unfaceted payload, the ~40 UMM fields schema.org has no term for travel as `PropertyValue` entries |
| `mat` | Metadata-as-Text — real natural-language prose written from a template per UMM field |

**The queries.** 511 in total (`data/queries_full.jsonl`): **11 SME** queries
written verbatim by domain experts (research questions like *"How did sea-ice
concentration and thickness in the Kara Sea change each March from 2020 to
2025?"*, ground truth = the CMR collections that answer them) and **500
synthetic** queries generated *from* indexed records (so ground truth is
in-corpus by construction), each filtered to not leak identifiers and to be
retrievable by at least one format. This run uses a 50-query slice — see Stage S.

**Two answering models,** one cloud, one local — the "format winner is
model-dependent" hypothesis needs at least two independent model families:

- `gpt-5.4-nano` (OpenAI, cloud, $0.20/$1.25 per M tokens in/out)
- `gpt-5.4-mini` (OpenAI, cloud, $0.75/$4.50 per M tokens in/out) — added
  2026-08-19 as a third arm, to separate "cloud vs local" from "bigger vs smaller"
- `muse-glimmer:30b-mlx` (30B, served locally through Ollama; costs wall-clock, not dollars)

**One fixed judge:** `gpt-5.4-nano` at temperature 0, running DeepEval's five
metrics. The judge is a separate config knob from the models under test — a
judge that varied with the tested model would confound the very axis being
measured. Note the judge is also one of the answering models; §2 and Caveats
discuss what that does and does not compromise.

**Held constant everywhere:** encoder `BAAI/bge-large-en-v1.5` (1024-dim,
512-token window), chunking 510 tokens / 64 overlap, max-pooling of chunk hits to
records, k = 10 records in context, temperature 0, seed 20260731, the same
answer prompt (`airm.exp2.ANSWER_SYSTEM`: *"name the candidates that genuinely
help answer the question… use ONLY the candidates shown… refer to each dataset
by its title"*).

**The experimental grid** is therefore 6 formats × 2 payloads × 3 models = 36
arms, each evaluated on the same 50 queries — 1,800 (query, format, payload,
model) cells.

---

## B. The workflow — four stages, then analysis

The evaluation is split into stages that each produce a **frozen artifact the
next stage reads rather than recomputes**. Nothing is silently re-run between
stages, so every judged score traces back to the exact retrieval that produced
its contexts: `judgements.jsonl` → cell in `answers.jsonl` (prompt hash,
retrieved ids) → `chunked_retrieval.jsonl` row (distances, metrics) →
fingerprinted cache bytes → raw calls in `runs/<rid>/llm/*.jsonl`.

```
Stage R   retrieval only, no LLM        511 queries × 6 formats × 2 payloads
   ↓  runs/{20260814T205849Z,20260814T205912Z}/chunked_retrieval.jsonl
Stage S   50-query slice selection      seeded, stratified, recorded
   ↓  runs/20260814T210927Z/selection.json
Stage A   LLM answers, replayed         50q × 6 fmt × 2 payloads × 3 models = 1,800 cells
   ↓  runs/20260814T210927Z/answers.jsonl  (+ llm/answer.jsonl raw calls)
Stage J   DeepEval judging              1,200 cells × ~22 judge calls (+600 mini cells × ~5), one fixed judge
   ↓  runs/20260814T210927Z/judgements.jsonl  (+ llm/judge.jsonl raw calls)
Analysis  join R + A + J                paired deltas, CIs, cost, Pareto charts
   ↓  runs/20260814T210927Z/{analysis.json,pareto.png,faithfulness.png} + this document
```

### Stage R — Retrieval (no LLM) · `scripts/unfaceted_chunked_eval.py`

**Input.** All 511 queries; the two chunked ChromaDB indexes
(`data/chroma/{faceted,unfaceted}_chunked/`), one collection per format.

**What it does.** For each (query, format, payload): embed the query with bge-large,
fetch the top k×12 = 120 chunks, pool chunk hits to a record score by `max`,
rank records, take the top 10, and score the ranked list against the
ground-truth concept-ids. Exact metrics — Recall@1/5/10, MRR, nDCG@10 — no
model in the loop, so they are free and repeatable.

**Output.** `chunked_retrieval.jsonl`: one row per (format × query) with the
query text, the ranked top-10 concept-ids **with pooled distances**, expected
ids, and the metrics; plus `chunked_summary.json` carrying the encoder/chunking
config and the **render-cache fingerprint** (`e99575eaf32a4fa5` faceted /
`3ef54e75af6c6775` unfaceted) that pins the exact bytes the index was built from.

**Why it is separate.** Retrieval is deterministic and cheap; answering and
judging are neither. Freezing the ranked lists means the 1,800 LLM cells all
replay *the same* retrieval — and the retrieval ranking can be analysed at full
511-query strength (`CHUNKED_RETRIEVAL_511Q.md`) while the LLM stages run on a
slice.

**Result carried forward.** Metadata-as-Text leads R@10 on both payloads
(faceted .884 / unfaceted .849), then JSON, YAML, TOON; JSON-LD and CSV last.
The R@10 / nDCG@10 columns in §0 are these numbers restricted to the 50 queries.

### Stage S — Query selection · runs inside `scripts/answer_stage.py`

**What it does.** Draws 50 of the 511: **all 11 SME queries** (no sampling
decision to make about eleven) plus **39 synthetic**, stratified so the topic
mix matches the full set (largest-remainder quotas per science topic —
ATMOSPHERE 10, LAND SURFACE 9, OCEANS 9, one each for the small topics —
alphabetical tie-break, seed 20260731). Both retrieval runs are checked to carry
the identical query set before selection.

**Output.** `selection.json`: the 50 query ids, quotas, models, k, git commit,
render-cache fingerprints, and the composition caveat written into the file
itself: SME queries are 22% of the slice vs 2.2% of the full set.

**Why 50.** 1,200 LLM cells × ~22 judge calls is what a full nano-judged pass
costs (~$40); the same grid at 511 queries would extrapolate to ~$400 of judging and days
of local-model latency. Fifty with all eleven expert queries in it keeps the hard
slice at full strength.

### Stage A — Answers · `scripts/answer_stage.py`

**Input.** `selection.json`, the two `chunked_retrieval.jsonl` logs, the two
format caches.

**What it does, per cell** (query × format × payload × model):

1. Read the top-10 concept-ids for that (query, format, payload) from the Stage R
   log — **no new retrieval**.
2. Load each id's rendering from the format cache. First verify the cache
   manifest fingerprint against the one Stage R pinned; if the cache was rebuilt
   since retrieval, refuse to run rather than show the model different bytes
   than the ones retrieval ranked.
3. Build the prompt: `ANSWER_SYSTEM` + the question + the 10 candidates as a
   numbered list `[1]…[10]`, each candidate being the full rendering in that
   format. Call the model at temperature 0. Ollama calls carry an explicit
   `num_ctx` sized from the prompt — Ollama otherwise serves a ~4k window and
   silently truncates, a confound that would hit the unfaceted arm hardest.
4. Append the cell to `answers.jsonl` and flush: query id, payload, format,
   model, the retrieved ids actually used, SHA-256 of the full prompt, the
   answer text, prompt/completion tokens, latency, cost, any error. Resumable
   by run id; interrupted cells are skipped, never re-answered.

**Output.** `answers.jsonl` (1,800 cells, 0 errors; the 600 mini cells were
appended on 2026-08-19 by resuming the run with `--models openai:gpt-5.4-mini
--workers 24`), `answers_summary.json`
(mean tokens/latency/cost per arm), `llm/answer.jsonl` (every raw call, full
prompt and response). The *Prompt tok / Compl tok / Lat s* columns in §0 come
from here.

**What it measures on its own.** The operational cost of each representation in
use — prompt tokens per answered query (dollars for nano) and latency (for the
prefill-bound local model). Unfaceted JSON-LD costs 3.7× the prompt tokens of
faceted JSON-LD, and 2.7× the wall-clock of unfaceted prose on local serving.

### Stage J — Judgement · `scripts/judge_stage.py`

**Input.** `answers.jsonl`; the format caches (fingerprint-checked again) to
re-assemble each cell's 10 contexts from its logged concept-ids; the query set
for the expected answer (`"The relevant collections are: <ground-truth titles>."`).

**What it does.** For every answered cell, run DeepEval's five metrics under
one fixed judge, `gpt-5.4-nano` at temperature 0, 24 workers. ~22 judge calls
per cell, every one logged to `llm/judge.jsonl`. A judge failure (nano
occasionally emits unparseable JSON, almost always on contextual relevancy)
records `None` for that metric — never zero — and the checkpoint key includes
the judge id so a second judge's pass can coexist for calibration.

**Output.** `judgements.jsonl` (1,800 cells; 35 of the original 1,200 = 2.9%
carry one `None` metric, 3 of the 600 mini cells a `None` faithfulness),
`judge_summary.json`. Actual cost of the original pass: 25,066 judge calls,
91.8M in + 17.2M out tokens, **$39.80**, mean 94 s per cell. The mini cells
were judged with `--metrics correctness faithfulness` only (~5 calls per cell).

**Judge constraint.** The local models under test (glimmer, qwen) must not
judge — a model grading its own answers biases the model-vs-model axis. The
judge chosen, nano, is nevertheless one of the two answering models, so the
script records a `self_judge` caveat: format and payload comparisons *within*
nano are clean; the nano-vs-glimmer comparison is the one to read carefully. §2
shows the observed direction of any such bias favours the competitor, and a ~$2
`gpt-5-mini` calibration slice remains available.

### Analysis · `scripts/analyze_50q.py`

**What it does.** Joins the three frozen artifacts on (query, format, payload,
model): exact retrieval metrics from R, token/latency/cost from A, the five
judge scores from J. Builds the per-arm table (§0), the paired comparisons along
each axis (§1 payload, §2 model, §3 format), the dollar and latency tables (§4),
the SME/synthetic split (§5), and the two Pareto charts. Writes `analysis.json`.

---

## C. How to read the numbers

**Everything is paired.** Each comparison holds (query, format, model) or
(query, format, payload) fixed and takes the difference between the two arms on
that same cell, then reports the mean difference with a 95% t-interval over the
per-cell differences. Pairing removes the query-difficulty variance (SME queries
are 3–5× harder than synthetic ones), which is what makes n = 50 informative.
An effect is reported as a finding only when its CI excludes zero.

**`None` drops out pairwise.** A judge-errored metric (2.9% of cells, one metric
each) is `None`; a paired difference involving it is skipped, never counted as
zero.

**Retrieval columns are exact; judge columns are graded.** R@10 and nDCG@10 are
computed against ground-truth concept-ids with no model involved. The five
DeepEval columns are LLM-graded in [0, 1] by `gpt-5.4-nano`. Where a judge-graded
retrieval-side metric disagrees with the exact one, trust the exact one.

### What the five DeepEval metrics measure

Each metric sees a different subset of the cell — that subset defines what it
can and cannot penalise:

| Metric | Inputs it sees | How the score is produced | What a low score means here |
|---|---|---|---|
| **Correct** (Correctness, GEval) | question, answer, **expected answer** (the ground-truth collection titles) | The judge grades the answer against a rubric: naming a collection that genuinely measures the requested phenomenon is correct; omitting an expected one is a partial failure; asserting a wrong one is a full failure; wording/order don't count | The answer named the wrong collections, or missed expected ones. The only metric that knows the ground truth — the headline quality number |
| **Faithful** (Faithfulness) | answer, **retrieved contexts** | The judge extracts factual claims from the answer, extracts truths from the 10 contexts, and checks each claim for contradiction; score = fraction of claims supported | The model asserted things the retrieved metadata doesn't support — the **hallucination** measure. It does not care whether the answer is *correct*, only whether it is grounded |
| **AnsRel** (Answer Relevancy) | question, answer | The judge splits the answer into statements and marks each relevant/irrelevant to the question; score = relevant fraction | The answer wanders off-topic. Penalises verbosity per extra statement — which is why long-form glimmer (~660 output tokens) scores below terse nano (~260) largely on style |
| **CtxRel** (Contextual Relevancy) | question, **retrieved contexts** | The judge marks each of the 10 retrieved documents relevant/irrelevant to the question; score ≈ relevant fraction | Most retrieved documents don't bear on the question. Never sees the answer — it grades **retrieval precision**, not the model |
| **CtxRec** (Contextual Recall) | expected answer, **retrieved contexts** | The judge checks whether each part of the expected answer can be attributed to some retrieved context; score = attributable fraction | The information needed for a perfect answer wasn't in the contexts (or, for prose renderings, was there but hard to attribute). Also a retrieval-side metric — it never sees the model's answer |

Two structural notes that follow from the inputs column:

- **Only Correctness measures end-to-end success.** Faithfulness and AnsRel
  grade the *answer's* discipline (grounded, on-topic) without knowing the
  truth; CtxRel and CtxRec grade the *retrieval* without seeing the answer. A
  cell can be perfectly faithful and relevant while naming the wrong dataset —
  faithfully summarising bad contexts.
- **CtxRel and CtxRec are judge-graded shadows of what Stage R measures
  exactly** (R@10, nDCG against ground-truth concept-ids). Where they disagree
  with the exact metrics, trust the exact ones; they are kept because they see
  the *text* of the contexts rather than just record identity.

---

## 0. Raw results — every arm, all metrics

Mean per cell over the 50 queries (11 SME + 39 synthetic, pooled — see §5 for
the slice split). Judge metrics are nano-judged DeepEval scores (defined in §C); R@10/nDCG@10
are Stage R's exact retrieval metrics restricted to the same 50 queries; tokens
and latency are Stage A's measurements per answer call (10 records in context).

**Bold** marks the best value in each column within a payload block (higher is
better for the judge and retrieval columns; lower is better for tokens and
latency). Ties at display precision are both bolded.

### gpt-5.4-nano (cloud)

| Payload | Format | Correct | Faithful | AnsRel | CtxRel | CtxRec | R@10 | nDCG@10 | Prompt tok | Compl tok | Lat s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| faceted | MaT | 0.726 | 0.846 | 0.699 | 0.072 | 0.361 | 0.737 | **0.653** | **9,607** | 282 | 2.3 |
| faceted | JSON | 0.746 | 0.833 | 0.696 | **0.165** | 0.650 | **0.785** | 0.642 | 10,593 | 258 | **2.1** |
| faceted | YAML | **0.780** | **0.879** | 0.793 | 0.146 | **0.705** | 0.769 | 0.645 | 12,070 | 254 | 2.2 |
| faceted | TOON | 0.734 | 0.839 | 0.763 | 0.134 | 0.647 | 0.745 | 0.628 | 11,587 | **231** | 2.1 |
| faceted | CSV | 0.698 | 0.839 | **0.807** | 0.139 | 0.651 | 0.736 | 0.650 | 12,052 | 247 | 2.2 |
| faceted | JSON-LD | 0.702 | 0.872 | 0.719 | 0.129 | 0.615 | 0.665 | 0.532 | 13,503 | 242 | 2.2 |
| unfaceted | MaT | **0.708** | **0.905** | **0.753** | 0.114 | 0.591 | 0.701 | 0.611 | **17,791** | **253** | **2.4** |
| unfaceted | JSON | **0.708** | 0.896 | 0.708 | 0.148 | 0.682 | **0.715** | **0.663** | 21,498 | 279 | 2.5 |
| unfaceted | YAML | 0.706 | 0.872 | 0.713 | 0.144 | 0.688 | 0.701 | 0.579 | 23,814 | 264 | 2.5 |
| unfaceted | TOON | 0.676 | 0.879 | 0.715 | 0.135 | **0.697** | 0.641 | 0.568 | 23,198 | 268 | 2.5 |
| unfaceted | CSV | 0.632 | 0.842 | 0.666 | **0.173** | 0.647 | 0.615 | 0.505 | 32,338 | 313 | 2.8 |
| unfaceted | JSON-LD | 0.696 | 0.821 | 0.679 | 0.151 | 0.603 | 0.655 | 0.527 | 49,983 | 275 | 3.0 |

### gpt-5.4-mini (cloud) — added 2026-08-19, correctness + faithfulness only

| Payload | Format | Correct | Faithful | R@10 | nDCG@10 | Prompt tok | Compl tok | Lat s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| faceted | MaT | 0.690 | **0.887** | 0.737 | **0.653** | **9,607** | 189 | 2.2 |
| faceted | JSON | 0.692 | 0.836 | **0.785** | 0.642 | 10,593 | **169** | 2.2 |
| faceted | YAML | 0.678 | 0.869 | 0.769 | 0.645 | 12,070 | 185 | 2.0 |
| faceted | TOON | **0.694** | 0.844 | 0.745 | 0.628 | 11,587 | 173 | **1.8** |
| faceted | CSV | 0.676 | 0.823 | 0.736 | 0.650 | 12,052 | 171 | **1.8** |
| faceted | JSON-LD | 0.660 | 0.843 | 0.665 | 0.532 | 13,503 | 170 | 2.0 |
| unfaceted | MaT | 0.672 | 0.849 | 0.701 | 0.611 | **17,791** | **173** | **2.2** |
| unfaceted | JSON | **0.690** | 0.887 | **0.715** | **0.663** | 21,498 | 206 | 2.5 |
| unfaceted | YAML | 0.632 | 0.889 | 0.701 | 0.579 | 23,814 | 187 | 2.3 |
| unfaceted | TOON | 0.588 | 0.875 | 0.641 | 0.568 | 23,198 | 218 | 2.6 |
| unfaceted | CSV | 0.630 | **0.894** | 0.615 | 0.505 | 32,338 | 218 | 3.7 |
| unfaceted | JSON-LD | 0.644 | 0.837 | 0.655 | 0.527 | 49,983 | 201 | 3.2 |

Mini was judged on the two headline metrics only (AnsRel/CtxRel/CtxRec not
run, to keep the judge cost at ~¼). Its prompt-token counts equal nano's to the
token — same tokenizer, same bytes — which is the by-construction check that
the two cloud arms answered from identical prompts. Mini writes the shortest
answers of the three models (~170–220 completion tokens vs ~250–310 for nano
and ~590–720 for glimmer).

### muse-glimmer:30b-mlx (local)

| Payload | Format | Correct | Faithful | AnsRel | CtxRel | CtxRec | R@10 | nDCG@10 | Prompt tok | Compl tok | Lat s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| faceted | MaT | **0.826** | 0.820 | 0.653 | 0.073 | 0.519 | 0.737 | **0.653** | **9,419** | 664 | **80.7** |
| faceted | JSON | 0.812 | 0.882 | 0.572 | 0.148 | 0.615 | **0.785** | 0.642 | 10,300 | 660 | 82.8 |
| faceted | YAML | 0.760 | 0.837 | 0.621 | **0.153** | 0.690 | 0.769 | 0.645 | 11,839 | 672 | 96.7 |
| faceted | TOON | 0.810 | **0.889** | 0.581 | 0.138 | 0.657 | 0.745 | 0.628 | 11,368 | **588** | 91.0 |
| faceted | CSV | 0.802 | 0.864 | **0.709** | 0.135 | **0.691** | 0.736 | 0.650 | 11,828 | 666 | 92.1 |
| faceted | JSON-LD | 0.760 | 0.864 | 0.553 | 0.122 | 0.640 | 0.665 | 0.532 | 13,128 | 636 | 103.3 |
| unfaceted | MaT | 0.758 | 0.865 | 0.625 | 0.115 | 0.666 | 0.701 | 0.611 | **17,532** | 674 | **132.6** |
| unfaceted | JSON | **0.786** | 0.888 | 0.568 | 0.153 | **0.703** | **0.715** | **0.663** | 20,810 | 669 | 153.1 |
| unfaceted | YAML | 0.758 | **0.909** | 0.619 | 0.157 | 0.677 | 0.701 | 0.579 | 23,588 | 675 | 172.4 |
| unfaceted | TOON | 0.734 | 0.877 | **0.656** | 0.138 | 0.586 | 0.641 | 0.568 | 22,998 | 708 | 168.6 |
| unfaceted | CSV | 0.754 | 0.892 | 0.564 | **0.179** | 0.609 | 0.615 | 0.505 | 32,400 | 721 | 231.9 |
| unfaceted | JSON-LD | 0.754 | 0.841 | 0.577 | 0.148 | 0.606 | 0.655 | 0.527 | 49,982 | **629** | 351.8 |

Reading notes for the softer columns:

- **CtxRel (contextual relevancy)** is low everywhere by construction — queries
  expect 1–2 records among the 10 retrieved, so 8–9 contexts are "irrelevant"
  in every arm. It tracks retrieval precision, which R@10/nDCG already measure
  exactly; it is retained here for completeness, excluded from headline claims.
- **CtxRec (contextual recall)** asks whether the expected answer is derivable
  from the contexts. Faceted MaT's low value (0.36–0.52) is the one notable
  outlier — the judge finds the expected collection titles harder to attribute
  to prose contexts than to structured ones, even when the answer is correct.
- **AnsRel** differences between models are largely stylistic: glimmer writes
  longer answers (~660 vs ~260 completion tokens), which the relevancy metric
  penalises per off-topic statement.
- Retrieval columns are identical across the three model tables by
  construction — retrieval depends only on (payload, format).

## 1. Headline: the projection beats the raw record, at a fraction of the cost

Paired difference in correctness, faceted − unfaceted, same query/format/model:

| Comparison | Δ correctness | 95% CI | n |
|---|---:|---|---:|
| **Faceted − unfaceted, all cells** | **+0.040** | [+0.022, +0.058] | 900 |
| within gpt-5.4-nano | +0.043 | [+0.011, +0.076] | 300 |
| within gpt-5.4-mini | +0.039 | [+0.010, +0.068] | 300 |
| within muse-glimmer | +0.038 | [+0.004, +0.071] | 300 |

The CI excludes zero overall and within each of the three models, and the
per-model effect is the same size (+0.04) for two cloud models and a local one
— the payload effect does not depend on which model reads the context. The 32-field faceted
projection — ~18% of the raw record's scalar values — produces **more correct
answers than the full record**, while costing **1.9–3.7× fewer prompt tokens**
(e.g. 9.6k vs 17.8k for prose, 13.5k vs 50.0k for JSON-LD). More metadata in
context is not better metadata: the deferred UMM content (contacts, URLs,
projects, bookkeeping) appears to act as distraction, not signal.

Faithfulness moves the *other* way, barely: faceted − unfaceted = −0.020
[−0.035, −0.004] (n=896). Both payloads sit at 0.82–0.91; the raw record gives the model
slightly more text to ground quotes in. The effect is an order of magnitude
smaller than the correctness gain and does not change the conclusion.
`faithfulness.png` shows the shape: where the correctness chart slopes
down-right (more tokens, worse answers), the faithfulness chart drifts weakly
up-right within its narrow 0.82–0.91 band — with the two most expensive arms
(unfaceted JSON-LD, ~50k tokens) falling back to the bottom of the band, so
even on this metric the extra tokens stop paying at the far end.

## 2. Model: glimmer > nano > mini — under nano's own judging

| Comparison | Δ correctness | 95% CI | n |
|---|---:|---|---:|
| **Glimmer − nano, all cells** | **+0.067** | [+0.045, +0.088] | 600 |
| within faceted | +0.064 | [+0.036, +0.092] | 300 |
| within unfaceted | +0.070 | [+0.037, +0.102] | 300 |
| **Mini − nano, all cells** | **−0.047** | [−0.063, −0.031] | 600 |
| within faceted | −0.049 | [−0.069, −0.030] | 300 |
| within unfaceted | −0.045 | [−0.070, −0.020] | 300 |
| **Mini − glimmer, all cells** | **−0.114** | [−0.134, −0.094] | 600 |

The judge *is* gpt-5.4-nano, and it rates the local competitor's answers above
its own by a clear margin — the self-judge concern, had it materialised, would
have pushed the other way. It also rates its larger sibling, gpt-5.4-mini,
*below* its own answers. Two readings are possible and the data here cannot
separate them: mini really does answer this task worse (its answers are the
shortest of the three, ~170–220 tokens, and may name fewer of the relevant
candidates), or nano-as-judge prefers nano-style answers. The glimmer result
argues against pure self-preference; a second judge (the `gpt-5-mini`
calibration slice, or a non-OpenAI judge) is the direct check. Faithfulness
shows no model difference on any pair (all CIs span zero: glimmer − nano +0.009,
mini − nano +0.001, mini − glimmer −0.008).

The gap concentrates on the hard queries. Mean correctness on the SME slice
across arms: glimmer 0.54 (range 0.46–0.63), nano 0.38 (0.30–0.48), mini 0.35
(0.25–0.44); on the synthetic slice: 0.84 / 0.80 / 0.75. Glimmer leads nano by
roughly +0.10–0.25 per arm on SME queries (e.g. faceted CSV 0.627 vs 0.373,
mini 0.400), while on synthetic queries the three are closer. The bigger local
model earns its latency on exactly the questions that are research questions
rather than dataset descriptions.

## 3. Format: a small effect, and the retrieval winner is not the answering winner

Paired against the JSON baseline (pooled over payloads × 3 models, n=300 each):

| Format | Δ correctness vs JSON | 95% CI | Significant? |
|---|---:|---|---|
| Metadata-as-Text | −0.009 | [−0.037, +0.019] | no |
| YAML | −0.020 | [−0.048, +0.008] | no |
| **TOON** | **−0.033** | [−0.063, −0.003] | **yes, worse** |
| **JSON-LD** | **−0.036** | [−0.066, −0.006] | **yes, worse** |
| **CSV** | **−0.040** | [−0.068, −0.013] | **yes, worse** |

No format beats JSON for answering. With the third model the CIs tighten and
three formats are now measurably worse than JSON — CSV (the path,value long
form is hardest to read back), JSON-LD, and TOON — while prose and YAML remain
indistinguishable from it. The spread is still ~0.04: comparable to the payload
effect and smaller than the model effect. The TOON result is driven by the
unfaceted arm on mini (0.588, the lowest cell mean in the grid).

The instructive split: **Metadata-as-Text wins retrieval** (Stage R: R@10 .884
faceted, first in every configuration) **but not answering** — while JSON-LD's
poor retrieval (R@10 .665 faceted, last) doesn't stop its answers from being
mid-pack once the right records are in context. Retrieval quality and
answer-stage usability are different properties of a format, and this pipeline
measured them separately. Recall@10 and correctness correlate only weakly
across arms.

## 4. Cost and latency, measured

Dollars per 1,000 queries answered (input + output; nano at $0.20/$1.25 per M,
mini at $0.75/$4.50 per M — OpenAI standard-tier prices as of 2026-08-19):

| Format | nano faceted | nano unfaceted | mini faceted | mini unfaceted |
|---|---:|---:|---:|---:|
| Metadata-as-Text | **$2.27** | $3.87 | **$8.06** | $14.12 |
| JSON | $2.44 | $4.65 | $8.70 | $17.05 |
| TOON | $2.61 | $4.97 | $9.47 | $18.38 |
| CSV | $2.72 | $6.86 | $9.81 | $25.24 |
| YAML | $2.73 | $5.09 | $9.89 | $18.70 |
| JSON-LD | $3.00 | **$10.34** | $10.89 | **$38.39** |

Mini costs 3.6–3.7× nano per query on identical prompts and, per §2, answers
less correctly — on this task the extra spend buys nothing. The most expensive
cell in the grid (mini, unfaceted JSON-LD, $38 per 1k queries) is 17× the
cheapest (nano, faceted MaT, $2.27) and 0.05 *less* correct.

Muse-glimmer pays in wall-clock instead: 81–103s/query faceted vs 133–352s
unfaceted — unfaceted JSON-LD takes 2.7× the latency of unfaceted prose for
identical content on prefill-bound local serving.

The corner-to-corner spread is the operational story: **faceted MaT/JSON on
nano at ~$2.30–2.40 and ~0.73–0.75 correctness (0.81–0.83 on glimmer) versus
unfaceted JSON-LD at $10.30 (nano) to $38 (mini) and 0.64–0.75.** The cheapest configurations are also the most accurate — there is
no cost/quality trade to agonise over. See `pareto.png`.

## 5. SME vs synthetic — the composition caveat, quantified

Correctness by slice (ranges across arms): SME 0.25–0.63, synthetic 0.68–0.91.
The 11 expert queries are 3–5× harder than the synthetic ones and are 22% of
this slice against 2.2% of the full query set, so pooled numbers here are not
comparable with pooled numbers from full-set runs. All paired comparisons above
are immune (pairing holds the query fixed); only absolute levels carry the
caveat.

## 6. What this says for the study's question

1. **Payload dominates format.** Choosing *what* metadata goes into context (a
   curated projection vs the raw record) moves correctness twice as much as any
   format choice, in the opposite direction from token cost — the projection is
   both better and 2–3.7× cheaper. This is the strongest evidence yet for the
   canonical-UMM → derived-AI-projection architecture.
2. **Within the projection, format matters little for answering** — JSON,
   YAML and prose are statistically indistinguishable; CSV, JSON-LD and TOON
   are measurably worse, by ~0.03–0.04. Format choice can therefore be made on
   retrieval performance (where prose wins) and operational grounds (tooling,
   cost) rather than answer quality, avoiding the three that underperform.
3. **Model choice matters more than format** and interacts with question
   difficulty: the local 30B model justifies its latency on expert-style
   questions specifically — and "bigger cloud model" is not "better": mini
   costs 3.6× nano and scores 0.05 lower under this judge. Model-dependence
   of the best representation, the study's premise, shows up here as the
   ranking of *models* changing what the best cost/quality corner is, while
   the payload ranking (faceted first) is the same for all three.

## Caveats

- **Judge:** all quality scores are nano-judged; nano grades its own answers on
  a third of the matrix, and its sibling mini's on another third (observed
  direction: glimmer > nano > mini — the self-judge concern cannot be ruled out
  for the mini comparison, see §2). 35 of the original 1,200 cells (2.9%) have
  one `None` metric from judge JSON-parse failures, concentrated in contextual
  relevancy; 3 of the 600 mini cells have a `None` faithfulness.
- **The mini arm was judged on correctness and faithfulness only**, so it has
  no AnsRel/CtxRel/CtxRec values; its pooled comparisons in §1–§3 use the two
  metrics that exist for every cell. It was added on 2026-08-19 by resuming the
  same run (`answer_stage.py --run-id 20260814T210927Z --models
  openai:gpt-5.4-mini`) against the same frozen retrieval lists and caches. A ~$2 mini calibration slice remains available
  (`judge_stage.py --judge openai:gpt-5-mini --limit 60`, then `--report`).
- **Contextual relevancy is excluded** from headline tables: with 1–2 expected
  records among 10 retrieved it measures retrieval precision, which Stage R
  measures exactly and for free.
- **The faceted payload is a summary** (~18% of source scalars, identically
  across formats). "Faceted beats unfaceted" is a claim about *this* projection,
  not about summaries in general.
- **n = 50 queries** (11 SME + 39 synthetic, seeded, stratified). Effects with
  CIs excluding zero are reported as findings; everything else as
  indistinguishable.
- Retrieval metrics come from the chunked bge-large indexes over 2,584 records
  (max-pooling); the chunk-count verbosity caveat from
  `CHUNKED_RETRIEVAL_511Q.md` applies to them unchanged.
