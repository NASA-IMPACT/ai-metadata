# Experiment 1 — Implementation plan

## Harness modules used
- `airm.cmr.load_corpus` / `airm.queries.load_queries` — data.
- `airm.representations.RENDERERS` — the 4 format conditions.
- `airm.run.evaluate_retrieval`, `airm.run.summarize` — retrieval stage (already built).
- `airm.llm.complete`, `airm.llm.available_models` — answer stage.
- `airm.metrics` — recall/MRR/nDCG, token count, Pareto.
- `airm.improve` — robustness + auto-tune.

## Steps (coded later in `run.py`)
1. Load corpus + queries; resolve model list via `available_models()` (assert ≥3).
2. **Retrieval stage:** for each representation, `evaluate_retrieval(...)` → JSONL;
   `summarize(...)` for per-format Recall@k/MRR with CIs. Embedding is model-independent,
   so this runs once per representation.
3. **Answer stage:** for each (model, representation, query): render top-k retrieved
   records, build the QA prompt, `llm.complete(...)`, grade against ground-truth
   concept-id(s). Log tokens for the Pareto axis. Skip tiers `available_models()` omits.
4. Aggregate a representation × model accuracy table; build the accuracy-vs-cost Pareto.

## Run
```
uv run python -m experiments.exp1_format_ablation.run --k 10
```

## Validate + improve loop
- `robustness_check` over seeds {0,1,2} with `paraphrase_queries`; report CIs;
  mark format pairs with overlapping CIs as "not distinguishable".
- `auto_tune(initial=metadata_as_text_template, score_fn=recall_at_k, model=<claude>)`:
  hill-climb the NL template; persist the winner to a versioned file under
  `results/` and note whether the cross-model ranking shifts.

## Outputs
Each run writes to its own timestamped folder `results/<UTC-timestamp>/` (with a
`results/latest` symlink pointing at the most recent run), so subsequent runs are
logged separately rather than overwriting:
- `results/<ts>/retrieval.jsonl`, `results/<ts>/answers.jsonl`
- `results/<ts>/summary.md` (tables + Pareto), `results/<ts>/tuned_metadata_as_text.txt`
