# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A research harness implementing the five experiments from `Report.md` §6
("Making Scientific Metadata AI-Ready"). The thesis under test: how the
*representation* of NASA CMR (Common Metadata Repository) metadata affects how
well an LLM can retrieve, select fields from, and reason over it — and that the
best representation is **model-dependent and must be measured**. `Report.md` is
the authoritative spec; read it before changing experiment design.

## Architecture

- **`airm/`** — the shared harness (one concern per module):
  - `cmr.py` — fetch + cache CMR collection/granule records from the public
    search API (`https://cmr.earthdata.nasa.gov/search`, no auth). A "record" is
    a UMM-JSON item: `{"meta": {...}, "umm": {...}}`. That whole dict flows
    through the harness; `meta["concept-id"]` is the stable id / ground-truth key.
  - `representations.py` — the independent variable of Exp 1: four renderers
    (`raw_umm_json`, `flattened_jsonld`, `dot_breadcrumb`, `metadata_as_text`) in
    the `RENDERERS` registry. `facets()` extracts shared content once so format
    varies while content is held roughly constant. `metadata_as_text` is the
    auto-tune target.
  - `schema.py` — Exp 2 field-description conditions (`bare`/`described`/`rich`).
  - `embeddings.py` — local sentence-transformers (`BAAI/bge-small-en-v1.5`);
    `Index` (single-encoder) and `DualIndex` (content+metadata, for Exp 3).
  - `llm.py` — cross-provider `complete()` dispatching by model-id prefix to
    Anthropic / OpenAI / local Ollama. **Missing provider → `ProviderUnavailable`,
    which callers catch to *skip* a tier, never crash a sweep.** Use
    `available_models()` to get the configured set.
  - `queries.py`, `metrics.py`, `run.py`, `improve.py` — query/ground-truth I/O;
    retrieval + tool-use + faithfulness + cost metrics; the runner (retrieval eval
    is already implemented in `run.evaluate_retrieval`); and the validate+improve
    loop (`robustness_check` + agentic `auto_tune`).
- **`experiments/expN_*/`** — one folder per experiment, each with a design
  `README.md` + implementation `plan.md`. The per-experiment `run.py` is written
  later, one at a time; the harness is the stable foundation they build on.
- **`data/`** — `corpus.jsonl` (437 real CMR collections across ~15 domains),
  `queries.yaml` (auto-seeded ground truth — meant to be hand-checked/broadened),
  `cmr_cache/` (per-record cache, gitignored).

## Environment & commands

- Python `>=3.13` (pinned in `.python-version`), managed by [`uv`](https://docs.astral.sh/uv/).
- Run tests: `uv run pytest` (config in `pyproject.toml`; `pythonpath=["."]` makes
  the top-level `airm` package importable — there is no `src/` layout).
- Run a single test: `uv run pytest tests/test_harness.py::test_smoke_retrieval_pipeline`
- Add a dependency: `uv add <package>` (`--dev` for dev deps).
- Demo the renderers on one record: `uv run python -m airm.representations --keyword "sea ice"`
- Rebuild the corpus: `uv run python -m airm.cmr <keywords...> -n 30`

## Provider config

Copy `.env.example` → `.env`. Cloud tiers (Anthropic, OpenAI) need keys; the
Ollama tier needs models pulled locally (`ollama pull llama3.1`). The suite runs
with whatever subset is configured — Exp 1 asserts ≥3 models are available.

## Conventions

- The first run of any embedding code downloads the sentence-transformers model
  (~tens of MB); tests take ~40s cold for this reason.
- Renderers must stay deterministic (no LLM calls) so retrieval is reproducible.
- Never silently drop coverage in an experiment (skipped model tier, unreachable
  MCP server, truncated corpus) — `log` it.
