# Findings — How should metadata be formatted for AI?

**Scope:** what Experiment 1 (format ablation) and Experiment 6 (format vs.
content) actually establish, and what they do not. Audit date 2026-07-21.
`Report.md` remains the authoritative design spec; this document records
measured results and their limits.

---

## TL;DR

1. **The popular "flatten metadata to prose for the AI" result was a content
   artifact, not a format effect.** It came from comparing a full UMM dump
   against a curated summary — different facts, not different formats.
2. **Content curation is the dominant lever.** Trimming to ~9 relevant fields is
   worth ~0.30 Recall@10 and a ~6× token reduction. No format choice comes close.
3. **Once content is held constant, format barely moves retrieval.** All four
   formats land within 0.812–0.883 with every pairwise CI overlapping.
4. **Every *reasoning*-axis claim in both experiments is currently unsupported by
   its own data.** All pairwise CIs overlap on both experiments. The
   "model-dependent winner" and "the axes disagree" verdicts are point-estimate
   gaps the data cannot resolve.

Findings 1–3 are solid (554 queries). Finding 4 is the correction that limits
what can be published.

---

## 1. The original result was confounded

June 2026 run, `experiments/exp1_format_ablation/results/20260624T210800Z/summary.md`:

| representation | Recall@10 | tokens/rec |
|---|---|---|
| metadata_as_text | 0.883 | 463 |
| flattened_jsonld | 0.812 | 574 |
| dot_breadcrumb | 0.592 | 4267 |
| raw_umm_json | 0.543 | 3888 |

Verdict at the time: *"Consistent — 1 distinct best-format across 3 models"*,
with prose the sole Pareto point. Read naively, this says prose wins and JSON
loses by 0.34.

The two losing renderers were rendering the **whole UMM tree**, while the two
winners rendered a curated 9-field `facets()` payload. They did not contain the
same facts, so the comparison could not isolate format. The switch is
`RENDERERS` vs. `RENDERERS_FAIR` in `airm/representations.py:322,344`;
`umm_facet_subset()` (`:133`) is what equalizes them. This was caught by the
2026-07-16 audit in `experiments/exp1_format_ablation/WORKFLOW.md:538-645`, which
added the `--fair` flag; Exp 6 then promoted the confound to an explicit
independent variable.

---

## 2. Content scope is the dominant lever

Exp 6 crosses format × content scope
(`experiments/exp6_format_vs_content/results/20260716T213021Z/summary.md`, H1 —
timestamped rather than `results/latest`, which moves with each run):

| format | facet Recall@10 | full Recall@10 | Δ | facet tokens | full tokens |
|---|---|---|---|---|---|
| raw_json | 0.852 | 0.543 | **+0.309** | 689 | 3888 |
| breadcrumb | 0.877 | 0.592 | **+0.285** | 609 | 4267 |

Curating to ~9 relevant fields buys ~0.3 Recall@10 *and* cuts tokens ~6×. The
dropped fields — URLs, contacts, DOIs, metadata dates — dilute the embedding.
**H1 holds decisively.**

---

## 3. With content held constant, format barely moves retrieval

Exp 1 `--fair` (`results/20260716T190813Z/summary.md`) and Exp 6 facet-scope
agree exactly:

| representation | Recall@10 | 95% CI (cluster) | MRR | nDCG | tokens/rec |
|---|---|---|---|---|---|
| metadata_as_text | 0.883 | [0.824, 0.933] | 0.683 | 0.731 | 463 |
| dot_breadcrumb | 0.877 | [0.823, 0.922] | 0.658 | 0.711 | 609 |
| raw_umm_json | 0.852 | [0.795, 0.904] | 0.624 | 0.679 | 689 |
| flattened_jsonld | 0.812 | [0.750, 0.873] | 0.588 | 0.642 | 574 |

Spread is 0.070 and **every pairwise CI overlaps**. Raw JSON rose 0.543 → 0.852
purely from content equalization. **H2 holds: there is no retrieval format
winner.**

In proportion: the content effect (0.31) is **~4.4× the entire format spread
(0.07)**. Roughly 80% of prose's apparent advantage was content pruning, and the
remaining ~20% is not distinguishable from zero.

### Cost, and tuning beats choosing

`metadata_as_text` is the cheapest at 463 tokens/record (~8× cheaper than a full
dump) and the sole Pareto point. Separately, auto-tuning the prose *template*
moved Recall@10 from 0.843 to **0.950 on a held-out split** — a larger gain than
any format swap. Tuning the representation you have beats agonizing over which
representation to pick.

---

## 4. The correction: the reasoning axis proves nothing yet

Both experiments reported a second axis — decoupled reasoning, where every format
is shown the *same* gold-containing candidate set, isolating selection from
retrieval. This is the one axis where format could still matter, and it carries
every model-dependence claim in the project.

It had no confidence intervals. Applying the cluster bootstrap the harness
already provides (`airm.metrics.bootstrap_ci_clustered`, clustered by gold
dataset) gives:

**Exp 1 decoupled** (from `results/20260716T190813Z/answers.jsonl`):

| representation | accuracy | 95% CI (cluster) | n | distinct datasets |
|---|---|---|---|---|
| metadata_as_text | 0.780 | [0.564, 0.956] | 50 | 9 |
| flattened_jsonld | 0.760 | [0.600, 0.920] | 50 | 9 |
| dot_breadcrumb | 0.740 | [0.545, 0.911] | 50 | 9 |
| raw_umm_json | 0.660 | [0.460, 0.844] | 50 | 9 |

**All 6 pairwise CIs overlap**, over only 9 distinct gold datasets, with
intervals spanning roughly ±0.2.

**Exp 6 reasoning** (from `results/20260716T213021Z/answers.jsonl`):

| condition | accuracy | 95% CI (cluster) | n | distinct datasets |
|---|---|---|---|---|
| raw_json__full | 1.000 | [1.000, 1.000] | 36 | 10 |
| breadcrumb__full | 0.972 | [0.909, 1.000] | 36 | 10 |
| breadcrumb__facet | 0.972 | [0.909, 1.000] | 36 | 10 |
| jsonld__facet | 0.972 | [0.909, 1.000] | 36 | 10 |
| mat__facet | 0.972 | [0.909, 1.000] | 36 | 10 |
| raw_json__facet | 0.944 | [0.861, 1.000] | 36 | 10 |

**All 15 pairwise CIs overlap**, over 10 distinct datasets, at ceiling
(0.94–1.00). `raw_json__full` "winning" is a single query out of 12.

### What this invalidates

- **Exp 1's "model-dependent: 3 distinct best-formats across 5 models."** A
  per-model argmax over cells that move in 0.1 steps, with no interval, is not
  evidence of model-dependence in either direction.
- **Exp 6's H3 "axes DISAGREE."** The reasoning winner's CI fully overlaps the
  retrieval winner's, so the disagreement is not established.
- **Exp 6's H4 entirely.** The executed run used the three OpenAI tiers, which
  saturate at 0.92–1.00, rather than the five local families
  `experiments/exp6_format_vs_content/plan.md:65-68` specifies precisely so
  model-dependence could be tested across families.

### One hypothesis that was wrong

The `results/20260716T212622Z` llama3.2 run scored 0.00 on five of six
conditions, which looked like a parse or context-truncation bug. It is not.
Reconstructing the candidate sets confirms gold was present in every prompt and
**every prediction was a legitimately shown candidate id**. The model simply
picked wrong, on n=2. There is no bug; there is also nothing to conclude.

### Other limits worth noting

- The exp1 before/after changed **two** things at once — the `--fair` flag *and*
  the model set (3 OpenAI → 5 Ollama). The flip from "consistent" to
  "model-dependent" is therefore not cleanly attributable to content
  equalization.
- Grading is exact concept-id match. `clean_answer()` strips `<think>` blocks
  (`exp1/run.py:156-165`), but nothing distinguishes a refusal from a wrong pick.
- `flattened_jsonld` and `metadata_as_text` are facet-only by construction, so
  the content axis is currently testable on only two of four formats.

---

## 5. Answering the central question

Given what survives measurement:

1. **Curate fields before arguing about syntax.** This is the one large,
   well-supported effect. A provider deciding how to serve metadata should spend
   its effort choosing *which fields* to expose, not which bracket style.
2. **For the embedding/retrieval path**, serve curated facet-scope prose
   (`metadata_as_text`), template-tuned. It ties on recall, wins on cost, and is
   Pareto-optimal.
3. **For the reasoning/agent path**, the honest answer is *we don't know yet*.
   The measurement that would settle it is now in place but has not been run at
   adequate sample size.
4. **The harness matters more than any one result** (`Report.md` §7). That claim
   is strongly supported by this audit: the same corpus and queries produced
   opposite headline conclusions depending on one confound and one missing
   confidence interval.

Note that claims 1–2 are narrower than "flatten everything to prose," and arrive
at a superficially similar recommendation for a completely different reason:
prose is a good default because it is *cheap and curated*, not because prose is
intrinsically easier for models to read. The audit's conditional-accuracy
recomputation (`WORKFLOW.md:582-586`) in fact found prose *worst* at reasoning
(0.72 vs. 0.85–0.90) once retrieval recall was factored out.

---

## 6. Changes made to the harness (2026-07-21)

Applied to both `experiments/exp1_format_ablation/run.py` and
`experiments/exp6_format_vs_content/run.py`:

- **Cluster-stratified sampling.** `sample_queries` now draws round-robin across
  gold clusters instead of uniformly at random. At n=20 exp1 covers 20 distinct
  datasets instead of 9; at n=150 both cover all 46 available.
- **CIs on the reasoning axis**, cluster-bootstrapped by gold dataset exactly as
  retrieval already was, plus a pairwise-overlap table.
- **Interval-aware verdicts.** Summaries now state when no condition is
  distinguishable, warn when fewer than 20 clusters back the sample, mark H3 as
  "not established" when the two winners' CIs overlap, and warn on H4 when models
  are ceiling-saturated.
- **`--max-queries` default raised from 12 to 150.**

Verification:

- `uv run pytest` passes (34 tests).
- A retrieval-only rerun (`results/20260721T142325Z`) reproduces the Axis-1
  numbers exactly, so nothing regressed.
- The summary writer was exercised end-to-end against the existing
  `20260716T213021Z` answer data: the CI table, the overlap list, the
  "no condition is statistically distinguishable" banner, the <20-cluster
  warning, the H4 ceiling-saturation warning, and the H3 "not established"
  note all render correctly. The CI tables in §4 above are that output.
- **Not** verified: a fresh run that calls models and then writes the new
  summary. A 20-query × 2-local-model run was started for this and was killed
  before the reasoning stage finished, leaving a partial
  `results/20260721T142510Z` (retrieval only, no `summary.md`). Its `latest`
  symlink was repointed to the last complete run. The model-calling path itself
  is unchanged by this work, so the untested surface is the join between it and
  the new reporting code.

## 7. What to run next

1. **Re-run both reasoning axes at `--max-queries 150`** across one consistent
   model set (the 5 Ollama families + 3 OpenAI tiers), so exp1 and exp6 are
   comparable and the model-set confound is removed. This is the single blocking
   item for H3 and H4.
2. **Add `full`-scope variants for `jsonld` and `mat`** to complete the 4×2 grid
   and test whether "reasoning wants completeness" is about content or about JSON.
3. **Re-run the auto-tuned template through the reasoning axis** — it is
   currently tuned for retrieval only; `WORKFLOW.md` already flags cross-model
   re-evaluation as an open follow-up.

Reproduce with:

```
uv run pytest
uv run python -m experiments.exp1_format_ablation.run --fair --max-queries 150
uv run python -m experiments.exp6_format_vs_content.run --max-queries 150
```

Each writes a fresh UTC-timestamped `results/` directory. Confirm the new
summaries carry reasoning-axis CIs and that no condition reports a degenerate
0.00 across all queries.
