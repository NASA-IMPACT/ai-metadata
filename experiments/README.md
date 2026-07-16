# Experiments

Five experiments from `Report.md` §6, each isolating one claim about how the
*representation* of NASA CMR metadata affects LLM use. All share the `airm/`
harness (CMR data, renderers, embeddings, cross-provider LLM client, metrics,
runner, improve loop) and the same corpus (`data/corpus.jsonl`, 481 real CMR
collections) and query set (`data/queries.yaml`).

Each folder has:
- `README.md` — the design: claim tested, hypothesis, conditions, metrics, success criteria.
- `plan.md` — implementation: harness modules used, run steps, and the validate+improve loop.

The `run.py`/`config.yaml`/`results/` for each are coded **later, one experiment
at a time** (this phase scaffolds docs + the shared harness only).

| # | Folder | Tests | Headline metric |
|---|--------|-------|-----------------|
| 1 | `exp1_format_ablation` | format × model interaction | Recall@k, MRR, answer accuracy |
| 2 | `exp2_description_quality` | field-description value | field-selection F1, value-format validity |
| 3 | `exp3_embedded_vs_connected` | embedded vs connected metadata | retrieval quality + re-index latency |
| 4 | `exp4_sparse_metadata` | missing-metadata behavior | hallucination vs correct-gap rate |
| 5 | `exp5_agentic_mcp` | static repr vs agentic/MCP | accuracy, latency, API-call count |

## Cross-cutting methodology

- **Models:** cross-provider via `airm.llm.available_models()` — Claude + OpenAI
  (when keyed) + locally pulled Ollama models. Experiment 1 requires ≥3.
- **Embeddings:** local `sentence-transformers` (BAAI/bge-small-en-v1.5).
- **Validate + improve loop (every experiment):**
  - *Robustness* (`airm.improve.robustness_check`): re-run across seeds /
    paraphrased queries / shuffled corpus, report bootstrap CIs, flag results
    whose CIs overlap as not robust.
  - *Auto-tune* (`airm.improve.auto_tune`): agentic hill-climb on that
    experiment's editable artifact (a renderer template or prompt) until the
    headline metric plateaus.
