# Experiment 5 — Implementation plan

## Harness modules used
- `airm.run.evaluate_retrieval` + `airm.representations.metadata_as_text` (Arch A).
- `airm.cmr.search_collections` with `extra_params` for hard filters (Arch B, C tool).
- `airm.llm.complete` / `available_models` (answering + agent loop).
- `airm.metrics` (accuracy, Pareto); `time.perf_counter` + a call counter (cost).
- `airm.improve` — robustness + auto-tune of tool descriptions.

## To author (later)
- A difficulty-stratified query set (simple lookups vs. multi-constraint spatial+
  temporal+variable), each with gold concept-id(s).
- A minimal tool spec for the CMR search (parameters from `airm.schema.FIELDS`),
  plus an optional MCP client path (`nasa/earthdata-mcp` / `podaac/cmr-mcp`).

## Steps (coded later in `run.py`)
1. **Arch A:** vector RAG over flattened records; answer; score.
2. **Arch B:** LLM emits a CMR filter (reuse Exp 2 parsing) → `search_collections`
   pre-filter → RAG over survivors → answer.
3. **Arch C:** agent loop — model issues tool calls to CMR/MCP, observes results,
   refines, returns a concept-id; cap iterations and count calls.
4. Compare accuracy, latency, API-call count; build the Pareto; stratify by difficulty.

## Run
```
uv run python -m experiments.exp5_agentic_mcp.run
```

## Validate + improve loop
- `robustness_check` across difficulty strata + seeds; CIs on accuracy and calls.
- `auto_tune` on the agent tool-description text; score = accuracy − λ·api_calls;
  persist to `results/tuned_tool_spec.json`.

## Outputs
- `results/runs.jsonl` (per arch/query: answer, latency, calls), `results/summary.md`

## Dependency / fallback note
Arch C requires network + (optionally) a running MCP server. If unavailable, log
the skip and report A vs B only — never silently drop C.
