# Experiment 2 — Implementation plan

## Harness modules used
- `airm.schema.render_fields` (conditions), `airm.schema.FIELDS` (auto-tune target).
- `airm.llm.complete` / `available_models`.
- `airm.metrics.field_selection_f1`, `value_format_valid`, `bootstrap_ci`.
- `airm.improve` — robustness + auto-tune.

## To author (later)
- `workload.yaml`: ~30 NL requests, each with gold `{fields: [...], values: {...}}`.
  Draw scenarios from corpus facets (instrument, variable, bbox, temporal).

## Steps (coded later in `run.py`)
1. For each condition in `airm.schema.CONDITIONS`, build a system prompt that
   presents the field schema and asks the model to emit a JSON CMR query object.
2. For each (model, condition, request): `llm.complete(...)`, parse JSON, compute
   field-selection F1 vs gold and value-format validity for `temporal`/`bounding_box`.
3. Aggregate per-condition means + CIs; report the bare→described→rich deltas.

## Run
```
uv run python -m experiments.exp2_description_quality.run
```

## Validate + improve loop
- `robustness_check` across models + paraphrased requests; verify the (a)<(b)<(c)
  ordering has non-overlapping CIs.
- `auto_tune` on the serialized `FIELDS` description text; score = mean(F1, value-valid);
  persist improved descriptions to `results/tuned_fields.json`.

## Outputs
- `results/predictions.jsonl`, `results/summary.md`, `results/tuned_fields.json`
