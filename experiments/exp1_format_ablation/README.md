# Experiment 1 — Format ablation on real CMR records

> *Report.md §6, Experiment 1 (foundational). Adapts the RAGMATE protocol to CMR.*

## Claim tested

Representation format materially changes how well an LLM retrieves and answers
over metadata, and **the best format is model-dependent** — so "flatten
everything to prose for the AI" is not automatically correct (Report §3.2, §4).

## Hypothesis

Across the four representations, no single format wins for all models. Expect
`metadata_as_text` and `flattened_jsonld` to lead on retrieval recall, while
`raw_umm_json` may help precision/faithfulness for some models. The *ranking*
will differ by model.

## Design

- **Independent variables:** representation (4) × model (≥3).
  - Representations: `raw_umm_json`, `flattened_jsonld`, `dot_breadcrumb`,
    `metadata_as_text` (`airm.representations.RENDERERS`).
  - Models: `airm.llm.available_models()` (Claude + OpenAI + Ollama).
- **Held fixed:** corpus (`data/corpus.jsonl`), query set (`data/queries.yaml`),
  embedding model, retriever, k.
- **Two measured stages:**
  1. *Retrieval* — embed each record under one representation, score queries.
     (Embedding format effect; model-independent.)
  2. *End-to-end answer* — give the LLM the top-k rendered records + the
     question; grade the answer. (Where the model-dependence appears.)

## Metrics

- Recall@k, MRR, nDCG@k (`airm.metrics`) for retrieval.
- End-to-end answer accuracy (exact concept-id / graded match) per model.
- Tokens-per-record + a Pareto plot of accuracy vs token cost
  (`airm.metrics.pareto_frontier`).

## Success criteria

- A representation × model accuracy table with bootstrap CIs.
- Evidence for or against model-dependence: does the best representation differ
  across models with non-overlapping CIs?

## Validate + improve loop

- **Robustness:** re-run retrieval across seeds and paraphrased queries; flag
  representation differences whose CIs overlap.
- **Auto-tune target:** the `metadata_as_text` renderer template — hill-climb it
  to maximize Recall@k, then re-check whether it changes the cross-model ranking.
