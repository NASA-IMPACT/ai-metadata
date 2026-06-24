# Experiment 4 — Sparse-metadata / dark-data stress test

> *Report.md §6, Experiment 4. Tests the "what happens when metadata is missing"
> thread; uses the Input Matters annotation method (§3.2).*

## Claim tested

When metadata is missing, a good representation should **report the gap rather
than fabricate** an answer. A `summary_statement` + machine-actionable provenance
(PROV-O / quality flags) should reduce hallucination (Report §5 Rec. 4, 6).

## Hypothesis

As fields are ablated, naive representations hallucinate plausible values;
representations that carry an explicit summary + provenance/quality status
degrade gracefully, reporting "not specified" more and fabricating less.

## Design

- **Independent variables:** ablation level (drop spatial bounds / quality flag /
  format / temporal, singly and combined) × representation.
- **Workload:** questions that target the ablated field (e.g. "what is the spatial
  coverage?") so the correct answer is sometimes *"the record doesn't say."*
- **Held fixed:** model set, base records (ablation applied programmatically).

## Metrics

- Faithfulness tally per (representation, ablation): `correct` /
  `correct_gap_report` / `fabrication` (`airm.metrics.FaithfulnessTally`).
- Hallucination rate = fabrication / total; gap-reporting rate =
  correct_gap_report / (correct_gap_report + fabrication).
- Graceful-degradation curve: accuracy vs. fraction of fields ablated.

## Success criteria

- Representations carrying an explicit summary/provenance show a lower
  hallucination rate at every ablation level, with non-overlapping CIs at the
  most-ablated level.

## Validate + improve loop

- **Robustness:** multiple ablation seeds and paraphrased questions; bootstrap CIs
  on hallucination rate.
- **Auto-tune target:** the provenance / `summary_statement` prompt prepended to
  records — hill-climb to *minimize* fabrication while keeping `correct` high.
