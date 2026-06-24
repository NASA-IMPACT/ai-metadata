# Experiment 5 — Static representation vs. agentic / MCP retrieval

> *Report.md §6, Experiment 5. Existence proofs: `nasa/earthdata-mcp`,
> `podaac/cmr-mcp` (§3.5).*

## Claim tested

Does **agentic discovery** (an LLM calling a CMR tool/MCP server) beat a good
*static* representation, or merely add latency and API calls? (Report §3.5, Rec. 5.)

## Hypothesis

Agentic retrieval wins on hard, multi-constraint queries (combining spatial +
temporal + variable filters) but adds latency and API calls; for simple lookups,
good static representation matches it at lower cost.

## Design — three architectures, same queries

- **A. Pure vector RAG:** embed flattened records (`metadata_as_text`), retrieve top-k,
  answer. (Reuses `airm.run.evaluate_retrieval` + an answer step.)
- **B. Filter → RAG:** parse the query into a CMR metadata filter (bbox/temporal/
  keyword), apply as a hard pre-filter, then RAG over survivors.
- **C. Agentic / MCP:** the model calls the CMR API (or `cmr-mcp`/`earthdata-mcp`)
  as a tool, iterating queries until it finds the dataset.

## Metrics

- Answer accuracy (correct concept-id) per architecture.
- Latency (wall-clock) and API-call count per query.
- Accuracy-vs-cost Pareto across the three.

## Success criteria

- A clear regime map: which architecture wins on simple vs. multi-constraint
  queries, and at what latency/API-call cost — i.e., when agentic access is worth
  it versus when good static representation suffices.

## Validate + improve loop

- **Robustness:** query difficulty strata (simple vs. multi-constraint) × seeds;
  CIs on accuracy and call count.
- **Auto-tune target:** the **tool descriptions** exposed to the agent (the §3.3
  field-description lever applied to tool specs) — hill-climb to raise accuracy
  while reducing API-call count.

## Note

Architecture C needs network/MCP access to CMR; it degrades to A+B if the MCP
server or network is unavailable, logging the skip (no silent cap).
