# Experiment 3 — Implementation plan

## Harness modules used
- `airm.embeddings.Index` (embedded / condition A).
- `airm.embeddings.DualIndex` + `reembed_metadata` (connected / condition B, update timing).
- `airm.representations.metadata_as_text` (A chunk), `extract_summary` (B content),
  `facets`-derived metadata string (B metadata).
- `airm.metrics` retrieval metrics + `bootstrap_ci`; `time.perf_counter` for cost.
- `airm.improve` — robustness (alpha/seed sweep) + auto-tune (chunk composition).

## Steps (coded later in `run.py`)
1. Build `Index` (A) and `DualIndex` (B) over the same corpus.
2. Score all queries under both; aggregate Recall@k/MRR/nDCG with CIs.
3. **Update simulation:** apply a metadata mutation (e.g. bump `Version`, edit
   `CollectionProgress`) across records; time:
   - A: full `Index.build` rebuild.
   - B: `DualIndex.reembed_metadata` only.
   Record latency and the ratio.
4. Sweep `alpha ∈ {0.3,0.5,0.7}` for B.

## Run
```
uv run python -m experiments.exp3_embedded_vs_connected.run --k 10
```

## Validate + improve loop
- `robustness_check` over seeds × alpha; flag if A vs B retrieval CIs overlap
  (i.e., quality is statistically tied, so the cost axis decides).
- `auto_tune` on the embedded-chunk composition template (score = Recall@k with a
  token-length penalty); persist to `results/tuned_chunk.txt`.

## Outputs
- `results/retrieval.jsonl`, `results/update_cost.json`, `results/summary.md`
