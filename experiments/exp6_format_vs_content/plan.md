# Experiment 6 — Implementation plan

> Design-only for now (README.md + this plan). `run.py` is coded in a follow-up.
> **Constraint: reuse `airm/` by import; change no existing code.**

## Harness modules reused (import only)

- `airm.cmr.load_corpus` / `airm.queries.load_queries` — data.
- `airm.queries.split_by_cluster` — cluster-aware train/val/test split (keeps
  paraphrases of one gold dataset on one side).
- `airm.representations.RENDERERS` — full-tree `raw_umm_json`, `dot_breadcrumb`.
- `airm.representations.RENDERERS_FAIR` — facet-scope variants of all four
  formats (built on `umm_facet_subset`). The 6 conditions are assembled from
  these two registries; **no new renderers are needed.**
- `airm.run.evaluate_retrieval`, `summarize(clusters=...)`,
  `check_relevance_coverage`, `new_run_dir`, `write_jsonl`, `set_seed`.
- `airm.metrics.count_tokens`, `pareto_frontier`, `bootstrap_ci_clustered`.
- `airm.llm.available_models`, `complete`, `ProviderUnavailable`.

## The 6 conditions (assembled in run.py, not in airm)

```python
from airm.representations import RENDERERS, RENDERERS_FAIR
CONDITIONS = {
    "raw_json__full":    RENDERERS["raw_umm_json"],
    "raw_json__facet":   RENDERERS_FAIR["raw_umm_json"],
    "breadcrumb__full":  RENDERERS["dot_breadcrumb"],
    "breadcrumb__facet": RENDERERS_FAIR["dot_breadcrumb"],
    "jsonld__facet":     RENDERERS_FAIR["flattened_jsonld"],
    "mat__facet":        RENDERERS_FAIR["metadata_as_text"],
}
```

## Steps (coded later in `run.py`)

1. **Load + guard.** `load_corpus()`, `load_queries()`; run
   `check_relevance_coverage(corpus, queries)` and log dangling/empty ground
   truth; keep only queries with reachable ground truth (mirror Exp 1).
2. **Cluster map + split.** `clusters = {q.id: q.relevant[0] for q in scored}`;
   `split_by_cluster(scored)` for the decoupled reasoning sample and any tuning.
3. **Retrieval axis (per condition).** `evaluate_retrieval(corpus, scored, fn,
   cond, k=10)` → JSONL; `summarize(..., clusters=clusters)` for Recall@10 / MRR /
   nDCG with cluster CIs. Embedding is model-independent → once per condition.
4. **Cost axis.** `tokens/record = mean(count_tokens(fn(r)))` over a record sample.
5. **Reasoning axis (decoupled).** Reimplement compactly in exp6 (do **not**
   import exp1 internals, so exp1 stays untouched):
   - Build one shared candidate set per sampled query = gold id + fixed
     distractors from a reference retrieval, capped at `answer_k`, deterministic
     shuffle (same construction proven in exp1's `build_shared_candidates`).
   - For each (model, condition, query): render the *same* candidates in that
     condition, prompt the model to pick a concept-id, grade against gold.
   - Skip any model `available_models()` omits (`ProviderUnavailable`), and log
     the skip — never silently drop a tier (CLAUDE.md convention).
6. **Verdicts.** Compute and write: content-effect deltas (facet vs full within
   raw_json / breadcrumb), format-effect spread + CI overlap at facet scope,
   axis-disagreement (best retrieval vs best reasoning condition), and per-model
   best format. Build the accuracy-vs-cost Pareto (`pareto_frontier`).

## Run (later)

```
# fast, deterministic — retrieval + cost only, no models needed
uv run python -m experiments.exp6_format_vs_content.run --no-answer

# full three-axis run across the five local families
uv run python -m experiments.exp6_format_vs_content.run \
  --models "llama3.2:latest,gemma3:4b,qwen3:8b,gpt-oss:20b,deepseek-r1:latest" \
  --max-queries 10 --answer-max-tokens 1024
```

## Outputs

Timestamped `results/<UTC>/` (with a `results/latest` symlink via
`airm.run.new_run_dir`):
- `retrieval.jsonl`, `answers.jsonl` (rows tagged with condition + model),
- `summary.md` — three-axis tables + the four verdicts (H1–H4).

## Verification

- `--no-answer` run: confirm `raw_json__facet` / `breadcrumb__facet` Recall@10 ≫
  their `__full` counterparts (content effect, H1) and that the four facet-scope
  formats' CIs overlap (small format effect, H2).
- Full run: `summary.md` reports a per-model best format on the reasoning axis
  (H4) and flags where the retrieval winner ≠ the reasoning winner (H3).
- No diffs anywhere under `airm/` or other `experiments/` folders.
