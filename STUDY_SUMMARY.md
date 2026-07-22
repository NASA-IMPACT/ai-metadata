# Making Scientific Metadata AI-Ready: Study Summary

## Introduction

Scientific data repositories like NASA's Common Metadata Repository (CMR) — which describes over a billion data files across ~10,000 collections — were designed for human catalog users and traditional search APIs. Increasingly, though, the "user" is a large language model: someone asks an LLM to *"find high-resolution vegetation-height data from the ATLAS instrument over the Amazon in December 2024,"* and the model must retrieve, interpret, and reason over CMR's deeply nested UMM-JSON metadata records to answer.

Two research communities bear on how repositories should prepare for this. The **data-stewardship tradition** (FAIR principles, provenance, reproducibility) has spent a decade defining machine-actionable metadata in principle. The **RAG/NLP literature** measures how models actually behave when fed structured text — and finds that representation effects are real, large, and model-dependent. This study sits at their intersection.

## Problem statement

A popular piece of advice says repositories should "flatten everything into prose for the AI" — that nested JSON is noisy and natural-language summaries are the AI-ready format. The literature suggests this is at best half-right: structured formats can *improve* faithfulness even as prose helps embedding recall, format winners vary by model, and richer field descriptions may matter more than format at all. The claims bundled into "LLM-ready metadata" — flatter representations retrieve better, described fields are used better, linked-data structure aids discovery, agentic (MCP) access beats static retrieval — are largely untested on real scientific metadata.

**The core problem:** we don't actually know which representation of CMR metadata serves LLMs best, and the popular comparisons are confounded — a full nested record versus a short prose summary differ in *content* (which facts are present), not just *format*, so any observed difference can't be attributed to formatting.

## Central hypothesis

The best metadata representation is **not universal — it is consumption-mode- and model-dependent, and must be measured empirically**. A repository should therefore serve multiple coordinated representations (canonical record, embedding-tuned summary, described fields, machine-actionable provenance), validated per use. Corollary hypotheses: rich field descriptions are the highest-leverage lever; and once content is held constant, the format effect on retrieval is small, while retrieval, reasoning, and token cost can each favor different formats.

## Experiments

All experiments run over a shared harness: ~437 real CMR collection records, a query set with known ground-truth answers, a fixed local embedding model and retriever, and multiple LLM tiers (Anthropic, OpenAI, local Ollama families).

1. **Format ablation (foundational).** Render each record four ways — raw UMM JSON, flattened JSON-LD, dot-notation breadcrumb, natural-language Metadata-as-Text — holding everything else fixed. Measure Recall@k, MRR, and answer accuracy across ≥3 models, since the literature predicts a model-dependent winner.

2. **Description-quality ablation.** Same queries, three schema conditions: bare field names; names plus one-line descriptions; descriptions plus enums and examples. Measure whether the model selects the right fields and formats valid values (ISO-8601 ranges, bounding boxes). Tests the "describe fields like a textbook" claim.

3. **Embedded vs. connected metadata.** Compare flattening metadata into the embedded chunk against a dual-encoder that embeds content and metadata separately. Measure retrieval quality *and* operational cost (re-indexing time when metadata updates), operationalizing the stewardship literature's core design tension.

4. **Sparse-metadata stress test.** Deliberately delete fields (spatial bounds, quality flags, format) and measure whether each representation degrades gracefully — does the model correctly report the gap, or hallucinate the missing value?

5. **Static representation vs. agentic/MCP retrieval.** Pit pure vector RAG, metadata-filter-then-RAG, and an agent calling a CMR MCP tool against the same queries. Measure accuracy, latency, and API-call count to test whether agentic discovery beats good static representation or just adds overhead.

6. **Format vs. content (follow-up to Exp 1).** The confound in the "flatten to prose" folklore becomes the explicit independent variable: cross format (4 levels) with content scope (facet-only vs. full UMM tree) for six conditions, evaluated on three separately-reported axes — retrieval, decoupled reasoning over identical candidate sets, and token cost. Its hypotheses: content scope dominates format in retrieval (H1); at equal content, format effects are small with overlapping CIs (H2); the axes can disagree, so no single format wins everything (H3); and the best reasoning format differs across model families (H4).

**Shared metrics:** Recall@k, MRR/nDCG for retrieval; exact-match field and value accuracy for tool use; annotated hallucination rate for faithfulness; tokens-per-record and re-index latency for cost — summarized as an accuracy-vs-token-cost Pareto frontier across representations.
