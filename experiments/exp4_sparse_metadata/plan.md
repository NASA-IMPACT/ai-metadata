# Experiment 4 — Implementation plan

## Harness modules used
- `airm.representations` renderers + `facets` (to know which fields exist).
- A small `ablate(record, fields)` helper (to add in `run.py`) that drops UMM
  subtrees (`SpatialExtent`, `ProcessingLevel`/`CollectionProgress`, `TemporalExtents`).
- `airm.llm.complete` / `available_models`.
- `airm.metrics.FaithfulnessTally`, `bootstrap_ci`.
- `airm.improve` — robustness + auto-tune of the provenance prompt.

## Grading
- Each answer auto-labeled into `correct` / `correct_gap_report` / `fabrication`
  by a rubric + LLM judge (the ablated truth is known: if a field was dropped, the
  faithful answer is "not specified"). Judge prompt lives in `run.py`.

## Steps (coded later in `run.py`)
1. For each ablation level, produce ablated copies of the corpus.
2. For each (representation, ablation, question): render, ask the model, judge the
   answer, `FaithfulnessTally.add(label)`.
3. Aggregate hallucination + gap-reporting rates with CIs; plot the degradation curve.

## Run
```
uv run python -m experiments.exp4_sparse_metadata.run
```

## Validate + improve loop
- `robustness_check` over ablation seeds + paraphrases; CIs on hallucination rate.
- `auto_tune` on the provenance/`summary_statement` prompt; score =
  `gap_report_rate - fabrication_rate`; persist to `results/tuned_provenance_prompt.txt`.

## Outputs
- `results/judgements.jsonl`, `results/degradation.json`, `results/summary.md`
