# Staged pipeline: 50 queries, faceted + unfaceted, retrieval → answers → judgement

The RAG evaluation split into three stages, each producing a frozen artifact the next
stage **reads** rather than recomputes. Every LLM input is reconstructable after the
fact: retrieval log → concept-ids → fingerprinted cache bytes → prompt hash → logged raw
call. Nothing is silently recomputed between stages, so a judged score can always be
traced back to the exact retrieval that produced its contexts.

```
Stage R   retrieval only, no LLM       511 queries x 6 formats x 2 payloads    ✅ DONE
   ↓  runs/<rid>/chunked_retrieval.jsonl
Stage S   50-query slice selection     seeded, stratified, recorded            ✅ DONE (runs inside Stage A)
   ↓  runs/<rid>/selection.json
Stage A   LLM answers, replayed        50q x 6 fmt x 2 payloads x 2 models     ✅ DONE — 1,200/1,200 cells, 0 errors
   ↓  runs/<rid>/answers.jsonl + runs/<rid>/llm/answer.jsonl
Stage J   DeepEval judging             per answered cell x ~22 judge calls     ✅ DONE — 1,200/1,200 judged by gpt-5.4-nano (~$39.80)
   ↓  runs/<rid>/judgements.jsonl + runs/<rid>/llm/judge.jsonl
```

**The working run id for everything below: `20260814T210927Z`.**
Status date: 2026-08-16.

---

## Next steps

Judging is **done**: 1,200/1,200 cells judged by gpt-5.4-nano at 24 workers —
25,066 judge calls, 91.8M in + 17.2M out tokens, actual cost **$39.80** (est. was
~$38). Mean 94s/cell. 35 cells (2.9%) carry one failed metric — nano emitting
unparseable JSON, almost all on contextual relevancy — recorded as `None`, excluded
from means, never counted as zero.

**Early signals in the pooled summary** (per-slice analysis pending): glimmer
out-scores nano on correctness in 11 of 12 format×payload arms — the nano judge rates
the *other* model's answers higher, evidence against self-judge inflation. Faceted
beats unfaceted on correctness in most arms at 2–3× fewer tokens. Format spread
within a payload is modest (correctness ~0.70–0.83 faceted). Faithfulness is
uniformly high (0.82–0.91) for both models.

### The one remaining step: analysis and report

Joins three frozen artifacts — `chunked_retrieval.jsonl` (exact retrieval metrics),
`answers_summary.json` (token cost), `judgements.jsonl` (quality) — into the
accuracy-vs-cost picture per (payload × format × model). Ask Claude to build it, or
start from `runs/20260814T210927Z/judge_summary.json`, which `judge_stage` regenerates
after every invocation.

> **Footnote — judge calibration, if ever needed.** The judgements file supports a
> second judge without re-running anything: judge the same cells with
> `--judge openai:gpt-5-mini --limit 60`, then `--report` prints per-metric means for
> each judge and the per-cell correlation between them (~$2). Worth doing later if a
> nano-judged result looks suspicious — in particular the self-judge concern, since
> nano grades its own answers on half the matrix; nano-vs-mini agreement on those
> cells is the direct check. Until then, all reported scores are nano-judged and say
> so in `judge_summary.json`.

---

## Stage R — Retrieval only ✅ DONE

**What it does.** For every query, embeds the query text with `BAAI/bge-large-en-v1.5`
and searches each format's chunked ChromaDB collection (2,584 records, 510-token chunks,
64-token overlap), oversamples k×12 chunks, pools chunk hits to record scores
(max-pooling), and scores the ranked list against ground-truth concept-ids. **No LLM is
involved** — the metrics (Recall@1/5/10, MRR, nDCG@10) are exact, free, and repeatable.

**What was run.** The full 511-query set (`data/queries_full.jsonl`: 11 SME + 500
synthetic), both payloads:

| Run | Payload | Rows | Verified against prior study |
|---|---|---:|---|
| `runs/20260814T205849Z` | faceted (32-field projection) | 3,066 | **exact match** with `runs/20260810T234350Z` |
| `runs/20260814T205912Z` | unfaceted (raw UMM record) | 3,066 | matches to within 0.0001 on 1 of 180 cells (ANN tie-ordering noise) |

**Artifacts per run:**

- `chunked_retrieval.jsonl` — the replayable log, one row per (format × query): query
  text, ranked top-10 concept-ids **with pooled distances**, expected ids, and the exact
  retrieval metrics. This is the file Stage A replays.
- `chunked_summary.json` — config (encoder, chunking, pooling), per-slice metric tables,
  and the **render-cache fingerprint** (`e99575eaf32a4fa5` faceted /
  `3ef54e75af6c6775` unfaceted) pinning the exact rendering bytes the index was built
  from.
- `chunked_rows.csv` — flat per-query metric rows.

**Result (all 511 queries, max-pooling, Recall@10):** Metadata-as-Text leads on both
payloads (faceted .884 / unfaceted .849), then JSON, YAML, TOON, JSON-LD, CSV last.
Read with the standing caveat: verbose formats get more chunks and therefore more
chances to match under max-pooling, and they still lose.

**Command (for reference — already run, ~40s per payload):**

```bash
uv run python scripts/unfaceted_chunked_eval.py eval --payload faceted   --queries data/queries_full.jsonl --pool max --no-chart
uv run python scripts/unfaceted_chunked_eval.py eval --payload unfaceted --queries data/queries_full.jsonl --pool max --no-chart
```

---

## Stage S — Query selection ✅ DONE (runs as the first step of Stage A)

**What it does.** Draws the 50-query slice from the 511: **all 11 SME queries**
(the expert half — no sampling decision to make about eleven) plus **39 synthetic**,
stratified so the topic mix matches the full set (largest-remainder quotas, alphabetical
tie-break, seed `20260731` — the study's seed, so the selection rebuilds
byte-identically). Both retrieval runs are checked to carry the identical query set
before selection.

**Artifact.** `runs/20260814T210927Z/selection.json` — the chosen query ids, per-topic
quota, models, k, and the composition caveat, recorded with git commit and timestamp.

**Known composition shift, stated up front:** SME queries are 22% of this slice against
2.2% of the full set, and they score far lower than synthetic ones. Pooled numbers
therefore overweight the hard expert queries — compare the `sme` and `synthetic` slices
separately, never the pooled mean across studies.

---

## Stage A — Answers ✅ DONE — 1,200/1,200 cells, 0 errors

**Script:** `scripts/answer_stage.py`

**What it does, per cell** (one cell = query × format × payload × model):

1. Reads the top-10 concept-ids for that (query, format) from the Stage R log — **no
   new retrieval happens in this stage.**
2. Loads each id's rendering from the format cache (`data/format_cache/` faceted,
   `data/format_cache_unfaceted/` unfaceted). Before any call, the cache's manifest
   fingerprint is verified against the one Stage R pinned — if the cache was rebuilt
   since retrieval, the stage refuses to run rather than feed the models different
   bytes than the ones retrieval ranked.
3. Builds the answer prompt (`airm.exp2.ANSWER_SYSTEM` + numbered candidate list —
   the same prompt Experiment 2 defines) and calls the model at temperature 0.
   Ollama calls carry an explicit `num_ctx` sized from the prompt, because Ollama
   otherwise serves a ~4k window and **silently truncates** — a confound that would
   have hit the unfaceted arm hardest.
4. Appends the cell to `answers.jsonl` and flushes: query id, payload, format, model,
   the retrieved ids actually used, a SHA-256 of the full prompt, the answer text,
   prompt/completion tokens, latency, cost, and any error. Interrupt at any point;
   `--run-id <RID>` resumes, skipping checkpointed cells.

**Every raw call** — full prompt, full response, token counts, the `num_ctx` used —
is logged to `runs/<rid>/llm/answer.jsonl` for audit, per call, failures included.

### Done: gpt-5.4-nano arm — 600/600 cells, 0 errors (~$2.60 spent)

Mean prompt tokens per answer call (10 documents in context) — the measured
operational cost of each representation in use:

| Format | Faceted | Unfaceted | Unfaceted / Faceted |
|---|---:|---:|---:|
| Metadata-as-Text | **9,607** | **17,791** | 1.9× |
| JSON | 10,593 | 21,498 | 2.0× |
| TOON | 11,587 | 23,198 | 2.0× |
| CSV | 12,052 | 32,338 | 2.7× |
| YAML | 12,070 | 23,814 | 2.0× |
| JSON-LD | 13,503 | 49,983 | 3.7× |

JSON-LD unfaceted costs **2.8× more input tokens per query answered** than
Metadata-as-Text unfaceted. Latency: ~2.1–3.0s/call across all twelve arms.

### Done: muse-glimmer:30b-mlx arm — 600/600 cells, 0 errors

Prompt tokens track nano's within a few percent (same prompts, different tokenizer) —
the new information is **latency on local serving, which scales directly with format
verbosity** because a local model is prefill-bound:

| Format | Faceted lat/call | Unfaceted lat/call | Unfaceted prompt tokens |
|---|---:|---:|---:|
| Metadata-as-Text | **80.7s** | **132.6s** | 17,532 |
| JSON | 82.8s | 153.1s | 20,810 |
| TOON | 91.0s | 168.6s | 22,998 |
| YAML | 96.7s | 172.4s | 23,588 |
| CSV | 92.1s | 231.9s | 32,400 |
| JSON-LD | 103.3s | **351.8s** | 49,982 |

On local hardware, answering from unfaceted JSON-LD takes **2.7× the wall-clock** of
unfaceted Metadata-as-Text for identical content — the token cost of a verbose format
is paid in latency, not dollars, when serving locally.

**Artifacts:** `answers.jsonl` (cells), `answers_summary.json` (regenerated after every
invocation — mean tokens/latency/cost per payload × format × model), `selection.json`,
`llm/answer.jsonl`.

**Smoke-run artifacts kept:** `runs/20260814T210345Z` (nano, 10 cells),
`runs/20260814T210433Z` (glimmer, 4 cells).

---

## Stage J — Judgement ✅ BUILT + SMOKED · full pass PENDING

**Script:** `scripts/judge_stage.py`

**What it does.** Reads the frozen `answers.jsonl`, re-assembles each cell's contexts
from the logged concept-ids (fingerprint-checked, same as Stage A), and scores every
answered cell with DeepEval's five metrics under one **fixed judge** (default
`openai:gpt-5.4-nano`), with `--workers N` concurrency (per-thread judge + metrics):

| Metric | What it measures | ~Judge calls/cell |
|---|---|---:|
| Correctness (GEval) | answer vs ground-truth collection titles | 1–2 |
| Faithfulness | claims in the answer supported by the contexts (hallucination) | 3–4 |
| Answer Relevancy | statements relevant to the question | 3 |
| Contextual Relevancy | each retrieved doc's relevance to the question | ~10–11 |
| Contextual Recall | expected answer attributable to contexts | 2 |

Scores append to `runs/<rid>/judgements.jsonl` (checkpoint key includes the judge, so
different judges' passes coexist — that is what makes calibration a first-class flow);
every judge call is logged to `runs/<rid>/llm/judge.jsonl`. A judge failure records
`None`, never zero. `--report` summarises, and compares judges per-cell when two exist.

**Smoke result (1 cell, 22 judge calls, 155s, all metrics scored, 0 errors):**
`sme-001`/faceted/csv → correctness 0.10, faithfulness 0.86, answer relevancy 1.00,
contextual relevancy 0.24, contextual recall 0.00. Sensible: sme-001 is the known-hard
query whose retrieval finds nothing relevant — low correctness/recall is the *right*
verdict, while high relevancy/faithfulness shows the model answered honestly from bad
contexts.

**Three real bugs found by the smoke, all fixed:**

1. `evaluate.py`'s judge wrapper had the ABC first in its MRO
   (`_Judge(DeepEvalBaseLLM, LoggedJudge)`), so the abstract methods shadowed the
   concrete ones and the class could never be instantiated. Latent since Step 10 —
   only ever tested against stubs; exp2's judged path would have crashed identically.
2. DeepEval's rich progress spinner hangs in captured/non-TTY contexts —
   `measure()` never reached the judge. Metrics are now measured with
   `_show_indicator=False`.
3. The per-cell trace metadata used the key `model`, colliding with
   `CallLogger.log()`'s own `model` argument — every judge call raised **after** the
   paid API call returned. Renamed to `answer_model`.

**Cost, from measured tokens and verified pricing** (gpt-5.4-nano $0.20/M input,
$1.25/M output; gpt-5-mini ~$0.25/M + $2/M — verify before quoting in a write-up):

| | Tokens | nano | mini |
|---|---|---:|---:|
| Judge nano's 600 answers | ~45M in + ~8.4M out | **~$19** | ~$35 |
| Judge glimmer's 600 answers | ~same | ~$19 | ~$35 |
| **All 1,200 cells** | ~90M in + ~17M out | **~$38** | ~$70 |

Half the judge cost is **output** tokens — nano reasons at ~14k output tokens per
judged cell — which the old input-only estimates missed. Cost lever if needed:
`--metrics correctness faithfulness answer_relevancy contextual_recall` drops
Contextual Relevancy (~10 of the 22 calls/cell; retrieval quality is already measured
exactly in Stage R) and roughly halves both time and cost.

**Judge constraint:** qwen3.6 and muse-glimmer must not judge — glimmer is under test,
and any model grading its own answers biases the model-vs-model axis. The script
prints and records a `self_judge` caveat when the judge overlaps the answer models
(nano judging nano's cells is exactly this — acceptable for format comparisons within
one model, compromised for nano-vs-glimmer comparisons; the mini calibration slice is
the check on that too).

---

## Cross-cutting guarantees

- **Content parity upstream:** all six formats of a record carry identical facts
  (hard-gated at cache build); the faceted payload is a 32-field projection (~18% of
  source scalars) — that caveat travels with every faceted result.
- **Determinism:** temperature 0 everywhere, seed 20260731 for selection, retrieval
  exact. The only nondeterminism observed in the whole pipeline so far is one 1e-4 ANN
  tie in Stage R.
- **Fail loudly:** fingerprint mismatches, unavailable models, and judge outages stop
  the run or record explicit errors; nothing is averaged in as zero.
- **Never overwrite:** every run gets its own `runs/<timestamp>/`; resume appends.
- **Audit chain:** for any judged score — `judgements.jsonl` → cell in `answers.jsonl`
  (prompt hash, retrieved ids) → `chunked_retrieval.jsonl` row (distances, metrics) →
  fingerprinted cache bytes → raw calls in `runs/<rid>/llm/*.jsonl`.

## Current state, in one table

| Stage | State | Artifact |
|---|---|---|
| R — retrieval, 511q, both payloads | ✅ run + verified | `runs/20260814T205849Z`, `runs/20260814T205912Z` |
| S — 50-query slice | ✅ done | `runs/20260814T210927Z/selection.json` |
| A — answers, nano arm | ✅ **600/600, 0 errors**, token table above (~$2.60) | `runs/20260814T210927Z/answers.jsonl` |
| A — answers, glimmer arm | ✅ **600/600, 0 errors**, latency table above | same run id |
| J — judging | ✅ **1,200/1,200 judged** by gpt-5.4-nano, $39.80, 2.9% cells with one `None` metric | `judgements.jsonl`, `judge_summary.json` |
| Analysis / report | ✅ **done** — see [`ANALYSIS_50Q.md`](ANALYSIS_50Q.md) | `scripts/analyze_50q.py`, `runs/20260814T210927Z/{analysis.json,pareto.png}` |
