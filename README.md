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
|---|------------|-------|-------|
| 1 | Format ablation (4 representations × N models) | retrieval, MRR, answer accuracy | **runner built** (`experiments/exp1_format_ablation/run.py`) |
| 2 | Description-quality ablation | field-selection F1, value validity | designed (docs) |
| 3 | Embedded vs. connected metadata | retrieval + re-index cost | designed (docs) |
| 4 | Sparse-metadata stress test | hallucination vs. gap-reporting | designed (docs) |
| 5 | Static vs. agentic/MCP retrieval | accuracy, latency, API calls | designed (docs) |

Experiments 2–5 have design `README.md` + `plan.md` in their folders; their runners are
coded one at a time on the shared harness.

## Layout

```
airm/                         # shared harness (one concern per module)
  cmr.py                      # fetch + cache CMR records (public API, no auth)
  representations.py          # 4 record renderers + shared facet extractors
  schema.py                   # field-description conditions (Exp 2)
  embeddings.py               # local sentence-transformers Index / DualIndex
  llm.py                      # cross-provider complete(): OpenAI / Ollama / Anthropic
  queries.py, metrics.py      # query+ground-truth I/O; retrieval/tool/faithfulness/cost metrics
  run.py, improve.py          # runner (+ timestamped run dirs); robustness + auto-tune loop
data/
  corpus.jsonl                # 437 real CMR collections across ~15 domains
  queries.yaml                # query set with ground-truth concept-ids
  cmr_cache/                  # per-record cache (gitignored)
experiments/expN_*/           # one folder per experiment: README.md + plan.md (+ run.py)
tests/                        # harness + experiment unit/smoke tests
```

See [`CLAUDE.md`](CLAUDE.md) for a deeper architecture tour.

## Setup

Requires Python ≥3.13 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                       # install dependencies
cp .env.example .env          # then fill in the providers you want
```

**Providers** (the harness uses whatever subset is configured and skips the rest):

- **Ollama** (local, no key) — `ollama pull llama3.2` (and any others you want to compare).
- **OpenAI** (optional) — set `OPENAI_API_KEY`.
- **Anthropic** (optional) — set `ANTHROPIC_API_KEY`.

Embeddings always run locally via `sentence-transformers` (no key needed); the model
downloads on first use.

## Usage

```bash
# Render all four representations of one record (sanity check)
uv run python -m airm.representations --keyword "sea ice concentration"

# Rebuild the CMR corpus
uv run python -m airm.cmr "vegetation height" "sea surface temperature" -n 30

# Run Experiment 1 (pin fast models to bound cost; omit --models to use all configured)
uv run python -m experiments.exp1_format_ablation.run \
    --max-queries 20 --models "llama3.2:latest,gemma3:4b"

# Retrieval + robustness only (no LLM calls, fast)
uv run python -m experiments.exp1_format_ablation.run --no-answer --no-tune
```

Each run writes to its own timestamped folder
`experiments/<exp>/results/<UTC-timestamp>/` (with a `results/latest` symlink), so
subsequent runs are logged separately. Outputs: `retrieval.jsonl`, `answers.jsonl`,
`summary.md` (tables + Pareto + model-dependence verdict), `tuned_metadata_as_text.txt`.

## Tests

```bash
uv run pytest                 # full suite
uv run pytest tests/test_exp1.py   # one file
```

## Caveats

- `data/queries.yaml` is **auto-seeded** (one relevant record per query, generic
  phrasing) — runnable now, but hand-broaden the relevance sets before drawing strong
  conclusions; absolute Recall numbers will rise.
- The report's Experiment 1 design wants **≥3 models**; reach it with the two OpenAI
  tiers and/or several pulled Ollama models. Reasoning models (e.g. `deepseek-r1`,
  `qwen3`) work but are slow and emit `<think>` blocks (handled by the runner).
- This is a research harness, not data-management advice — see `Report.md` §7.
