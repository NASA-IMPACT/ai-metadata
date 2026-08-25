# ai-metadata

An empirical research harness for **"AI-Optimized Science Metadata Formats"** —
measuring how the *representation* of NASA Common Metadata Repository (CMR)
records affects how well an LLM can retrieve and reason over them, and at what
token cost.

The motivating question (from [`reports/Report.md`](reports/Report.md)): how
should a provider like CMR restructure its metadata so large language models
can use it well? The thesis: **the best representation is not universal — it
is consumption-mode- and model-dependent, and must be measured.** This repo is
the measurement: 2,590 real CMR records × 6 serializations × 2 content
payloads × 3 models — 31,080 parity-gated renderings, a 511-query retrieval
study, and 10,800 judged answers, all runnable and audited end to end.

## Headline result

**Equal-or-better on every axis at 40% of the cost.** A curated 32-field
projection of the UMM record beat the raw record on correctness (+5.8%
relative) and retrieval (+8.4% Recall@10) at −60% prompt tokens, for every
model tested — while the worst possible *format* choice costs at most
2.7–8.5%. Content curation dominates serialization; prose (Metadata-as-Text)
is the strongest single default; JSON-LD is the one representation to avoid
for LLM consumption. See [`reports/Study_Summary.md`](reports/Study_Summary.md)
for the full findings, hypothesis verdicts, and the AI-readiness playbook.

## Reports

All write-ups live in [`reports/`](reports/):

| Report | What it covers |
|---|---|
| [`Study_Summary.md`](reports/Study_Summary.md) | **Start here** — the whole study: hypotheses, all findings, figures, caveats, playbook |
| [`Report.md`](reports/Report.md) | The literature synthesis and original five-experiment program |
| [`EXPERIMENTS.md`](reports/EXPERIMENTS.md) | The build log: design, steps, verification, Experiment 1 (token economy) results |
| [`CHUNKED_RETRIEVAL_511Q.md`](reports/CHUNKED_RETRIEVAL_511Q.md) | Retrieval study — 511 queries, chunked bge, faceted vs unfaceted |
| [`STAGED_PIPELINE.md`](reports/STAGED_PIPELINE.md) | How the retrieve → answer → judge pipeline works; every artifact and field |
| [`ANALYSIS_300Q.md`](reports/ANALYSIS_300Q.md) | The 300-query judged run: payload / format / model effects with CIs |
| [`ANALYSIS_50Q.md`](reports/ANALYSIS_50Q.md) | The 50-query reference run the 300-query slice nests |
| [`PIPELINE_50Q.md`](reports/PIPELINE_50Q.md) | Status tracker for the 50-query run |
| [`FORMATS.md`](reports/FORMATS.md) | Each format explained: derivation, examples, corpus coverage, present-vs-informative audit |
| [`NEXT_EXPERIMENTS.md`](reports/NEXT_EXPERIMENTS.md) | Gap analysis against the original program; what to build next |

Slides: [`slides/ai_metadata_formats_slides.pdf`](slides/ai_metadata_formats_slides.pdf)
(source `.tex` and paste-ready `.md` alongside).

## The experiments

| # | Experiment | What it measures | State |
|---|---|---|---|
| 1 | Token economy | Serialization cost at constant content, 2 payloads × 6 formats, paired vs JSON | ✅ done |
| 2 | Chunked retrieval | Recall/MRR/nDCG per format × payload, 511 queries, bge-large chunked | ✅ done |
| 3 | Retrieval → QA → judge | End-to-end correctness/faithfulness/cost, 36 arms × 300 queries | ✅ done |
| 4 | Enhanced retrieval | LLM-written summaries appended to every rendering; retrieval delta | ✅ done |
| — | Corpus coverage audit | Sizes, tokens, field-level present-vs-informative | ✅ done |
| — | Description ablation · sparse-metadata stress test · agentic/MCP arm | (original program's Exp 2/4/5) | planned |

## Repo layout

```
reports/          all write-ups (see table above)
src/airm/         the library: facets, formats, parity, retrieval, LLM stages, judging
scripts/          runnable stages and analyses (answer_stage, judge_stage, analyze_50q,
                  cache_coverage, field_coverage, summary_retrieval, summary_append_eval, …)
data/             cmr_cache (raw records) · format_cache* (renderings) · chroma/ (indexes)
runs/             timestamped experiment artifacts — every number traces to a file here
slides/           the results deck (pdf / tex / md)
tests/            offline + live test suites
```

## Quickstart

```bash
uv sync                                       # install
uv run pytest                                 # offline test suite
uv run python -m airm.exp1                    # Experiment 1: token economy (no API key)
uv run python scripts/cache_coverage.py       # corpus size/token audit
uv run python scripts/field_coverage.py       # present-vs-informative field audit

# the LLM stages need OPENAI_API_KEY (and Ollama for the local arm)
uv run python scripts/answer_stage.py --help
uv run python scripts/judge_stage.py --help
```

Provenance: every run directory carries git commit, config, and call logs;
caches are content-fingerprinted; format parity is hard-gated at build.
