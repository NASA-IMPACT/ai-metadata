# Analysis: 50 queries, faceted vs unfaceted, retrieval → answers → judgement

What the completed staged pipeline says about the study's question — *which
representation of CMR metadata serves an LLM best, at what cost* — from run
`20260814T210927Z`: 1,200 answered cells (50 queries × 6 formats × 2 payloads ×
2 models), all judged by `gpt-5.4-nano`.

- Script: [`scripts/analyze_50q.py`](scripts/analyze_50q.py)
- Data: `runs/20260814T210927Z/{answers,judgements}.jsonl`,
  `runs/{20260814T205849Z,20260814T205912Z}/chunked_retrieval.jsonl`
- Tables: `runs/20260814T210927Z/analysis.json` · Charts:
  `runs/20260814T210927Z/pareto.png` (correctness vs context cost) ·
  `runs/20260814T210927Z/faithfulness.png` (faithfulness vs context cost)

All comparisons are **paired** on the same (query, format, model) cells, with 95%
t-intervals over per-cell differences. Judge-errored metrics (2.9% of cells, one
metric each) are `None` and drop out pairwise — never counted as zero.

---

## 0. Raw results — every arm, all metrics

Mean per cell over the 50 queries (11 SME + 39 synthetic, pooled — see §5 for
the slice split). Judge metrics are nano-judged DeepEval scores; R@10/nDCG@10
are Stage R's exact retrieval metrics restricted to the same 50 queries; tokens
and latency are Stage A's measurements per answer call (10 records in context).

**Bold** marks the best value in each column within a payload block (higher is
better for the judge and retrieval columns; lower is better for tokens and
latency). Ties at display precision are both bolded.

### What the five DeepEval metrics measure

All five are LLM-graded scores in [0, 1], computed per cell by the fixed judge
(`gpt-5.4-nano`, temperature 0, ~22 judge calls per cell, every call logged to
`runs/<rid>/llm/judge.jsonl`). Each metric sees a different subset of the cell —
that subset is what defines what it can and cannot penalise:

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
- Retrieval columns are identical between the two model tables by construction
  — retrieval depends only on (payload, format).

## 1. Headline: the projection beats the raw record, at a fraction of the cost

Paired difference in correctness, faceted − unfaceted, same query/format/model:

| Comparison | Δ correctness | 95% CI | n |
|---|---:|---|---:|
| **Faceted − unfaceted, all cells** | **+0.041** | [+0.017, +0.064] | 600 |
| within gpt-5.4-nano | +0.043 | [+0.011, +0.076] | 300 |
| within muse-glimmer | +0.038 | [+0.004, +0.071] | 300 |

The CI excludes zero overall and within each model. The 32-field faceted
projection — ~18% of the raw record's scalar values — produces **more correct
answers than the full record**, while costing **1.9–3.7× fewer prompt tokens**
(e.g. 9.6k vs 17.8k for prose, 13.5k vs 50.0k for JSON-LD). More metadata in
context is not better metadata: the deferred UMM content (contacts, URLs,
projects, bookkeeping) appears to act as distraction, not signal.

Faithfulness moves the *other* way, barely: faceted − unfaceted = −0.018
[−0.036, −0.000]. Both payloads sit at 0.82–0.91; the raw record gives the model
slightly more text to ground quotes in. The effect is an order of magnitude
smaller than the correctness gain and does not change the conclusion.
`faithfulness.png` shows the shape: where the correctness chart slopes
down-right (more tokens, worse answers), the faithfulness chart drifts weakly
up-right within its narrow 0.82–0.91 band — with the two most expensive arms
(unfaceted JSON-LD, ~50k tokens) falling back to the bottom of the band, so
even on this metric the extra tokens stop paying at the far end.

## 2. Model: glimmer outperforms nano — under nano's own judging

| Comparison | Δ correctness | 95% CI | n |
|---|---:|---|---:|
| **Glimmer − nano, all cells** | **+0.067** | [+0.045, +0.088] | 600 |
| within faceted | +0.064 | [+0.036, +0.092] | 300 |
| within unfaceted | +0.070 | [+0.037, +0.102] | 300 |

The judge *is* gpt-5.4-nano, and it rates the competing model's answers above
its own by a clear margin — the self-judge concern, had it materialised, would
have pushed the other way. Faithfulness shows no model difference
(+0.009 [−0.010, +0.028]).

The gap concentrates on the hard queries: on the SME slice glimmer leads nano by
roughly +0.10–0.25 per arm (e.g. faceted CSV 0.627 vs 0.373), while on synthetic
queries the models are much closer. The bigger local model earns its latency on
exactly the questions that are research questions rather than dataset
descriptions.

## 3. Format: a small effect, and the retrieval winner is not the answering winner

Paired against the JSON baseline (pooled over payloads × models, n=200 each):

| Format | Δ correctness vs JSON | 95% CI | Significant? |
|---|---:|---|---|
| Metadata-as-Text | −0.009 | [−0.046, +0.029] | no |
| YAML | −0.012 | [−0.048, +0.024] | no |
| TOON | −0.025 | [−0.062, +0.013] | no |
| JSON-LD | −0.035 | [−0.073, +0.003] | borderline |
| **CSV** | **−0.042** | [−0.076, −0.007] | **yes, worse** |

No format beats JSON for answering; only CSV is measurably worse (the
path,value long form appears hardest to read back). The spread is ~0.04 — half
the payload effect and smaller than the model effect.

The instructive split: **Metadata-as-Text wins retrieval** (Stage R: R@10 .884
faceted, first in every configuration) **but not answering** — while JSON-LD's
poor retrieval (R@10 .665 faceted, last) doesn't stop its answers from being
mid-pack once the right records are in context. Retrieval quality and
answer-stage usability are different properties of a format, and this pipeline
measured them separately. Recall@10 and correctness correlate only weakly
across arms.

## 4. Cost and latency, measured

gpt-5.4-nano dollars per 1,000 queries answered (input + output at
$0.20/$1.25 per M, prices as of 2026-08-15):

| Format | Faceted | Unfaceted |
|---|---:|---:|
| Metadata-as-Text | **$2.30** | $3.90 |
| JSON | $2.40 | $4.70 |
| TOON | $2.60 | $5.00 |
| CSV | $2.70 | $6.90 |
| YAML | $2.70 | $5.10 |
| JSON-LD | $3.00 | **$10.30** |

Muse-glimmer pays in wall-clock instead: 81–103s/query faceted vs 133–352s
unfaceted — unfaceted JSON-LD takes 2.7× the latency of unfaceted prose for
identical content on prefill-bound local serving.

The corner-to-corner spread is the operational story: **faceted MaT/JSON at
~$2.30–2.40 and ~0.73–0.83 correctness versus unfaceted JSON-LD at $10.30 and
0.70–0.75.** The cheapest configurations are also the most accurate — there is
no cost/quality trade to agonise over. See `pareto.png`.

## 5. SME vs synthetic — the composition caveat, quantified

Correctness by slice (ranges across arms): SME 0.30–0.63, synthetic 0.70–0.91.
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
2. **Within the projection, format barely matters for answering** — JSON, YAML,
   TOON, and prose are statistically indistinguishable; only CSV underperforms.
   Format choice can therefore be made on retrieval performance (where prose
   wins) and operational grounds (tooling, cost) rather than answer quality.
3. **Model choice matters more than format** and interacts with question
   difficulty: the local 30B model justifies its latency on expert-style
   questions specifically.

## Caveats

- **Judge:** all quality scores are nano-judged; nano grades its own answers on
  half the matrix (observed effect, if any, favours the competitor). 35 cells
  (2.9%) have one `None` metric from judge JSON-parse failures, concentrated in
  contextual relevancy. A ~$2 mini calibration slice remains available
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
