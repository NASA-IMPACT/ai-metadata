# ai-metadata

An empirical research harness for **"Making Scientific Metadata AI-Ready"** — testing
how the *representation* of NASA Common Metadata Repository (CMR) records affects how
well an LLM can retrieve, select fields from, and reason over them.

The motivating question (from [`Report.md`](Report.md)): how should a provider like
NASA's CMR restructure its metadata so large language models can use it well? The
report's thesis is that there is no single best format — it is **model-dependent and
must be measured** — and it proposes five experiments to find out. This repo implements
those experiments on **real CMR data** pulled from the public search API.

## Status

| # | Experiment | Tests | State |
