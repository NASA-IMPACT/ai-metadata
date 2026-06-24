# Experiment 2 — Description-quality ablation

> *Report.md §6, Experiment 2. Backed by Gorilla / ToolAlpaca / Databricks (§3.3).*

## Claim tested

Treating CMR's searchable parameters as tool/function parameters, **richer field
descriptions and enums sharply improve correct field selection and value
formatting** — the cheapest, highest-leverage change (Report §3.3, Rec. 2).

## Hypothesis

A large jump from condition (a) → (c): bare field names produce wrong fields and
malformed values; descriptions + enums + examples yield correct field choice and
valid ISO-8601 ranges / bounding boxes.

## Design

- **Independent variable:** schema presentation, 3 conditions
  (`airm.schema.render_fields`): `bare`, `described`, `rich`.
- **Workload:** natural-language requests (e.g. *"canopy height over the Amazon
  in December 2024"*) the model must translate into a CMR query object
  (which fields, what values). Gold field-sets and value formats authored per query.
- **Held fixed:** model set, query workload; only the schema text varies.

## Metrics

- Field-selection F1 (`airm.metrics.field_selection_f1`).
- Value-format validity (`airm.metrics.value_format_valid`) for `temporal`
  (ISO-8601 range) and `bounding_box` (west,south,east,north in range).
- Per-condition accuracy with bootstrap CIs.

## Success criteria

- Monotone improvement bare → described → rich on both metrics, with the
  (a)→(c) gap's CI excluding zero.

## Validate + improve loop

- **Robustness:** re-run across models and paraphrased requests; confirm the
  ordering holds with non-overlapping CIs.
- **Auto-tune target:** the per-field `description`/`enum_hint`/`example` text in
  `airm.schema.FIELDS` — hill-climb to maximize field-selection F1 + value validity.
