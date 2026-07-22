# Experiment 6 — Format vs. content for CMR ingestion & retrieval

> *Follow-up to Experiment 1, motivated by a common but confounded claim (see
> "Motivation" below). Report.md §3.2, §4, §6.*

## Motivation

A widespread piece of RAG folklore — and the framing of the conversation that
prompted this experiment — says the "AI-ready" CMR format is **Metadata-as-Text
(MaT)**: "flatten everything to prose, raw JSON is noisy and loses." It is
usually argued with a side-by-side of a *full* nested UMM record against a *short*
prose summary, concluding that the prose **format** wins.

That argument is confounded, and the Experiment 1 audit (2026-07-16) quantified
why: the full record and the summary do not contain the **same facts**. The raw
dump carries dozens of extra fields (URLs, contacts, DOIs, metadata dates) the
summary drops, so the comparison measures *content*, not *format*. When content
is held constant, raw JSON's Recall@10 rose from 0.54 → 0.85 and all four formats
converged to 0.81–0.88 with overlapping CIs. The apparent "format effect" was
mostly information content: the extra fields *diluted* the embedding.

This experiment makes that confound the explicit independent variable, and
separates the axes the folklore conflates.

## Claim tested

Once **content** is held constant, **format** has only a small, model-dependent
effect on how well an LLM ingests and retrieves CMR records — and the three axes
(embedding retrieval, LLM reasoning over the record, token cost) can **disagree**,
so no single format is universally "best."

## Hypotheses

- **H1 (content dominates).** Holding format fixed and varying content scope
  (facet-only vs. full UMM tree) changes retrieval more than holding content
  fixed and varying format. Concretely: `raw_json` and `breadcrumb` retrieve much
  better at `facet` scope than at `full` scope.
- **H2 (format effect is small).** At `facet` scope, the four formats' Recall@10
  CIs overlap — no clear format winner on retrieval.
- **H3 (axes disagree).** The format that wins retrieval is not necessarily the
  one the model reasons best over; MaT's retrieval edge does not carry to the
  decoupled reasoning axis.
- **H4 (model-dependence).** The best format on the reasoning axis differs across
  model families, consistent with the report's thesis.

## Design

- **Independent variable 1 — format (4):** `raw_umm_json`, `flattened_jsonld`,
  `dot_breadcrumb`, `metadata_as_text`.
- **Independent variable 2 — content scope (2):** `facet` (the shared 9-field
  `facets()` payload) vs. `full` (the whole UMM tree). Separable only for the two
  full-tree renderers; `flattened_jsonld` and `metadata_as_text` are facet-only
  by construction. **Six conditions:**

  | condition | format | content scope | source renderer |
  |---|---|---|---|
  | `raw_json__full`    | raw JSON       | full  | `RENDERERS["raw_umm_json"]` |
  | `raw_json__facet`   | raw JSON       | facet | `RENDERERS_FAIR["raw_umm_json"]` |
  | `breadcrumb__full`  | dot-breadcrumb | full  | `RENDERERS["dot_breadcrumb"]` |
  | `breadcrumb__facet` | dot-breadcrumb | facet | `RENDERERS_FAIR["dot_breadcrumb"]` |
  | `jsonld__facet`     | flattened JSON-LD | facet | `RENDERERS_FAIR["flattened_jsonld"]` |
  | `mat__facet`        | metadata-as-text  | facet | `RENDERERS_FAIR["metadata_as_text"]` |

- **Independent variable 3 — model:** the reasoning axis runs across
  `airm.llm.available_models()`; target the five local Ollama **families** already
  pulled (`llama3.2`, `gemma3:4b`, `qwen3:8b`, `gpt-oss:20b`, `deepseek-r1`) so
  model-dependence is tested across families, not one vendor.
- **Held fixed:** corpus (`data/corpus.jsonl`), query set (`data/queries.yaml`),
  embedding model, retriever, `k`.

## Metrics — three axes, reported separately (never collapsed into one "winner")

1. **Retrieval** — Recall@10, MRR, nDCG@10 per condition, with
   cluster-bootstrapped CIs (`airm.metrics.bootstrap_ci_clustered`, clustered by
   ground-truth dataset — the query set is paraphrase-clustered).
2. **Reasoning (decoupled)** — every condition is shown the **same**
   gold-containing candidate set, rendered in that condition; grade the model's
   pick. This isolates reasoning-over-format from retrieval recall.
3. **Cost** — tokens/record per condition; a Pareto frontier over
   (reasoning accuracy, token cost).

## Comparisons the analysis must surface

- **Content effect** — within `raw_json` and within `breadcrumb`,
  `facet` vs. `full` (tests H1; expected large).
- **Format effect** — across the four formats at `facet` scope (tests H2;
  expected small, overlapping CIs).
- **Axis disagreement** — best retrieval condition vs. best decoupled-reasoning
  condition (tests H3).
- **Model-dependence** — best format per model on the reasoning axis (tests H4).

## Success criteria

- A condition × model table for each axis, with CIs where applicable.
- An explicit verdict on: (a) is **content scope** or **format** the dominant
  factor in retrieval? (b) is there a **universal** best format, or is it
  **model-dependent**? Framed as *"which of the pasted conversation's claims
  survive measurement."*

## Status of the results so far (audit, 2026-07-21)

**Axis 1 (retrieval) is solid.** H1 and H2 both hold with wide margins and are
reproducible: content scope moves Recall@10 by ~0.30 while the whole format
spread at facet scope is 0.07 with all six pairwise CIs overlapping. The content
effect is ~4.4× the format effect.

**Axis 2 (reasoning) is not yet interpretable, and neither H3 nor H4 is
established.** The runs to date used `--max-queries 12` or fewer:

- Pooling the 213021Z OpenAI run over models and bootstrapping by gold dataset
  gives **all 15 condition pairs overlapping**, over only **10 distinct gold
  datasets**. `raw_json__full` at 1.00 vs. the 0.97 field is a single query, so
  the "axes disagree" verdict (H3) rests on a gap the data cannot resolve.
- That run also used the three OpenAI tiers, which saturate at 0.92–1.00, rather
  than the five local families this design specifies for H4. **H4 is untested
  here**; the only evidence for model-dependence is exp 1's local-model run —
  and that axis has the same problem (all six format pairs overlap over 9
  clusters).
- The 212622Z llama3.2 zeros are **not** a parse or truncation bug. Gold was
  present in every candidate set and every prediction was a legitimately shown
  candidate id; the model simply picked wrong, on n=2. Nothing to fix, nothing
  to conclude.

The summary writer now reports cluster-bootstrapped CIs on the reasoning axis and
refuses to let a point-estimate gap stand as a verdict. `--max-queries` now
defaults to 150 and samples round-robin across gold clusters (46 available), so
the bootstrap has enough independent units to resolve a real difference.

## Relationship to other experiments

- **Extends Exp 1** by adding content scope as a controlled second variable
  (Exp 1's `--fair` flag holds content constant but does not vary it as an IV).
- **Distinct from Exp 2** (field-description quality) — Exp 6 does not alter field
  *descriptions*; it varies which *records-worth-of-content* each format sees.
- Reuses the shared harness only; **no changes to `airm/` or existing
  experiments.**
