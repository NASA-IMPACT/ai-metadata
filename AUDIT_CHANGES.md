# Audit changes, results, and the `--fair` tag

_2026-07-16 — explanation of the audit-driven changes to the `airm/` harness and
Experiment 1, what the corrected results mean, and what the `--fair` mode is._

This document is a plain-English companion to the code changes. For the numbers
in context see `experiments/exp1_format_ablation/WORKFLOW.md` §7; for the raw
output see `experiments/exp1_format_ablation/results/latest/`.

---

## What the `--fair` tag is

This is the single most important change, so start here.

Experiment 1 asks: *does the **format** of a metadata record change how well
retrieval works?* To answer that cleanly, every format has to describe **the same
facts** — otherwise you are measuring content differences, not format differences.

The original code violated this. Two of the four renderers behaved differently
from the other two:

- `flattened_jsonld` and `metadata_as_text` were built from a curated set of
  **9 fields** (title, summary, platforms, variables, bounding box, temporal
  range, quality, data center) — extracted by the `facets()` function.
- `raw_umm_json` and `dot_breadcrumb` dumped the **entire** raw NASA record —
  dozens of extra fields (contact info, URLs, DOIs, metadata dates, use
  constraints, and so on).

So when `raw_umm_json` scored 0.54 and `metadata_as_text` scored 0.88, you could
not tell whether that was because prose is a better *format*, or simply because
the raw dump was carrying a big pile of **extra content** the curated ones never
saw. Format and content were tangled together.

**`--fair` untangles them.** A new function `umm_facet_subset()` rebuilds a record
containing *only* those same 9 facets, and a second renderer registry
`RENDERERS_FAIR` feeds all four formats that identical facet content.
`flattened_jsonld` and `metadata_as_text` are unchanged (they already used
facets); the two "raw" renderers now render only the facet subset instead of the
whole tree.

So:

- **default mode** = the old behavior (raw formats see more content) — kept for
  comparison.
- **`--fair` mode** = all four formats describe exactly the same facts, differing
  *only* in how they are laid out (nested JSON vs. `a.b.c = value` breadcrumbs
  vs. Schema.org vs. prose).

Only `--fair` mode actually isolates the variable the experiment claims to test.

---

## What the latest results mean

Running `--fair` produced the headline correction:

| format | default (raw sees more) | **fair (same content)** |
|---|---|---|
| metadata_as_text | 0.883 | 0.883 |
| dot_breadcrumb | 0.592 | **0.877** |
| raw_umm_json | 0.543 | **0.852** |
| flattened_jsonld | 0.812 | 0.812 |

The two "raw" formats jump from ~0.55 up to ~0.86 once they are restricted to the
same facts. Held constant, **all four formats land at 0.81–0.88 with overlapping
confidence intervals** — meaning the format differences are small and barely
distinguishable.

The plain-English takeaway: the original run's dramatic "prose format wins, raw
JSON loses" story was **mostly an artifact of content, not format**. The raw dump
was not bad because it was JSON — it was worse because it was stuffing the
embedding with dozens of irrelevant extra fields, which *diluted* the signal.
Give raw JSON the same clean 9 facts and it retrieves nearly as well as prose.

Two more corrections came from the other new pieces:

- **The "answer accuracy" story was circular.** The old run showed answer
  accuracy tracking retrieval and concluded "prose is easier for the model to
  reason over." But answer accuracy was *capped* by retrieval — a model cannot
  pick the right dataset it never retrieved. A new **decoupled** stage shows every
  format the *same* candidate set (with the correct answer guaranteed present), so
  only the rendering differs. Result: when the right answer is actually in front
  of the model, **format barely matters, and raw JSON is often best** — the
  opposite of the original claim. Conditional on the gold answer being present,
  raw JSON was adjudicated correctly 90% of the time vs. prose at 72%.

- **Model-dependence is real, and now genuinely tested.** The old runs used one
  model (then three, all OpenAI), so they could not test the project's actual
  thesis: *the best format depends on the model.* This run used five different
  model families. The best format split three ways across them (deepseek/gpt-oss
  preferred raw JSON, gemma/llama preferred flattened JSON-LD, qwen preferred
  dot-breadcrumb). So "prose wins outright" was only true inside the OpenAI
  family — across families **there is no single best format**, which is exactly
  what the report predicted.

---

## The code changes, by purpose

The changes fall into three tiers.

### P0 — the changes that fix the conclusions (the important ones)

- `umm_facet_subset()` + `RENDERERS_FAIR` + the `--fair` flag → content-held
  comparison (explained above).
- The **decoupled answer stage** (`build_shared_candidates()` + a shared-candidate
  mode in `run_answer_stage`) → measures reasoning-over-format independent of
  retrieval.
- `split_by_cluster()` in `queries.py` → a train/validation/test split so the
  auto-tuner is optimized on one slice and **re-scored on a held-out slice it
  never saw**. Previously the tuner was trained and reported on the same queries,
  which inflates the number. The split is *cluster-aware*: the 554 queries are
  really paraphrase clusters around 46 datasets, so whole clusters go to one side
  to prevent leakage.

### P1 — honest statistics + no silent failures

- `bootstrap_ci_clustered()` → confidence intervals now resample by *dataset*
  (46 clusters) instead of by *query* (554 correlated paraphrases). The old CIs
  were artificially narrow because near-duplicate queries were treated as
  independent. This is why the new CIs are visibly wider — they are more honest.
- `check_relevance_coverage()` → logs any query whose "correct answer" is not in
  the corpus (those silently scored 0 before) and excludes queries with no ground
  truth, instead of letting them quietly drag every score down.
- `llm.py` → cloud API errors (bad model id, rate limit) now cause a model to be
  *skipped* rather than crashing the whole sweep.

### P2 — correctness hardening

- `recall_at_k`/`ndcg_at_k` count *distinct* hits (cannot exceed 1.0 on duplicate
  ids).
- The temporal-format validator was rewritten to actually parse ISO-8601 dates
  (it was rejecting valid NASA timestamps and accepting impossible ones like
  month 13).
- Deterministic seeding for the paraphrase perturbation, atomic cache writes with
  corrupt-file recovery, and small doc fixes (the corpus is 481 records, not 437).

Plus 12 new unit tests (`tests/test_audit_fixes.py`); the full suite is 34/34.

---

## One-line summary

The experiment's *machinery* was mostly sound, but its *experimental design* let
content leak into a format comparison and let retrieval leak into a reasoning
comparison. The fixes isolate each variable, and once isolated, the strong
original conclusions soften into "format effects are small and model-dependent" —
which is both more defensible and more aligned with the report's actual thesis.
