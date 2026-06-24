# Experiment 3 — Embedded vs. connected metadata

> *Report.md §6, Experiment 3. Operationalizes Leipzig et al.'s embedded/connected
> tension (§2.4) with RAGMATE's encoders (§3.1).*

## Claim tested

The core design tension: **metadata embedded *inside* the chunk** (one vector)
vs. **metadata connected via a separate encoder** (content + metadata vectors,
combined). Embedding aids cohesion/recall but is costly to maintain (any metadata
edit forces re-embedding the whole chunk); connecting preserves updateability.
Neither literature declares a universal winner (Report §4).

## Hypothesis

Retrieval quality is comparable (connected within CI of embedded), but the
connected dual-encoder is **far cheaper to update**: a metadata change re-embeds
only the metadata side, not the full chunk.

## Design

- **Condition A (embedded):** single-encoder `Index` over `metadata_as_text`
  (content + metadata flattened into one chunk).
- **Condition B (connected):** `DualIndex` — content (`Abstract`) and metadata
  (facets) embedded separately, scores combined with weight `alpha`.
- **Held fixed:** same records, same queries, same embedding model.
- **Update simulation:** mutate a metadata field across the corpus, then time the
  re-index path for each condition (`DualIndex.reembed_metadata` vs full rebuild).

## Metrics

- Retrieval: Recall@k, MRR, nDCG@k for both conditions.
- Operational cost: wall-clock re-index latency after a simulated metadata update;
  vectors recomputed per update.

## Success criteria

- Quantified quality difference (with CIs) **and** an update-cost ratio showing the
  connected design's maintenance advantage — making the trade-off explicit.

## Validate + improve loop

- **Robustness:** sweep `alpha` and seeds; report retrieval CIs; confirm the
  cost advantage is stable.
- **Auto-tune target:** the chunk composition for condition A (what facets go into
  the embedded chunk) — hill-climb to maximize Recall@k under a token budget.
