# AI-Optimized Science Metadata Formats — Study Summary

**One paragraph.** We asked how a metadata provider like NASA's Common
Metadata Repository (CMR) should represent collection records so that LLMs
can find and reason about them, and at what cost. Across 2,590 real CMR
records × 6 serializations × 2 content payloads × 3 models — 31,080 cached
renderings, a 511-query retrieval study, and 10,800 judged answers — the
answer is: **what you put in context matters far more than how you serialize
it.** A curated 32-field projection beat the raw UMM record on correctness
(+0.045) at 40% of the token cost, for every model tested; the worst format
mistake moves correctness only 1/1.7 as much, and format effects are
model-dependent besides; prose won retrieval everywhere and is the cheapest
rendering; JSON-LD lost everywhere it could lose. A field-level
audit explains part of the win: CMR's required-field compliance is perfect,
but only 7 of 13 required fields are reliably *informative* — the rest are
heavily padded with placeholders that distort embeddings far out of
proportion to their ~2% token cost.

| | |
|---|---|
| Runs | `20260819T183313Z` (LLM stages, 300 queries) · `*_full_*` (retrieval, 511 queries) · `runs/cache_coverage/` (corpus audit) |
| Detail documents | [`ANALYSIS_300Q.md`](ANALYSIS_300Q.md) · [`CHUNKED_RETRIEVAL_511Q.md`](CHUNKED_RETRIEVAL_511Q.md) · [`FORMATS.md`](FORMATS.md) · [`STAGED_PIPELINE.md`](STAGED_PIPELINE.md) · [`Report.md`](Report.md) |
| Slides | [`slides/ai_metadata_formats_slides.pdf`](../slides/ai_metadata_formats_slides.pdf) |

---

## 1. The question and the four hypotheses

The pitch: **the best representation is not universal — it is
consumption-mode- and model-dependent, and must be measured empirically.**
Made falsifiable by crossing *format* against *content scope* — the confound
the field has been ignoring.

| Hypothesis | Verdict | Evidence in brief |
|---|---|---|
| **H1** Content dominates — field selection matters more than format | **Supported** | Choosing *what* goes in context moves correctness **1.7×** more than the worst format mistake (+0.045 vs −0.027) — while also cutting tokens 60% and lifting recall 8.4%, gains no format choice matches on any axis, let alone all three at once |
| **H2** Format effect is small at equal content | **Supported\*** | Going from the best format (MaT) to the worst (CSV) costs only **4.4%** relative correctness (0.812 → 0.776), and only CSV and JSON-LD are even statistically distinguishable from JSON. \*The one true outlier is JSON-LD, and only at retrieval: 5–9% worse Recall@10 while costing up to 2.2× the tokens |
| **H3** The axes disagree — retrieval, reasoning, cost rank formats differently | **Supported** | JSON wins raw-record reasoning for 3 of 3 models but is 5th of 6 at retrieval; prose is 1st at retrieval in all four conditions and cheapest, but mid-pack at raw-record reasoning — no format wins every axis |
| **H4** Best format varies by model family | **Suggestive** | Three models, three different curated-payload winners (YAML / MaT / TOON) — the leaderboard reshuffled every time the model changed. But each winner leads its runner-up by only 0.002–0.006, well inside the ~±0.03 per-model uncertainty at n=300 — so *that winners vary* is supported; *which* format wins for which model is not settled |

The honest framing of H1: on correctness alone, content beats format by
1.25–1.7× (payload +0.045 vs the worst single format penalty −0.027, or the
full best-to-worst format spread 0.036) — real but modest. The dominance
claim rests on the content effect being **the same sign on every axis at
once** (+0.045 correctness, +8.4% recall, −60% tokens), while format choices
only ever trade one axis against another.

**The numbers behind the verdicts** (run `20260819T183313Z`; paired
comparisons with 95% CIs from `analysis.json`; retrieval from the 511-query
Stage R runs):

| # | Comparison | Value |
|---|---|---|
| H1 | Payload effect, correctness (paired, n=5,400) | **+0.045** [+0.038, +0.052] |
| H1 | Largest format effect, correctness (CSV vs JSON, n=1,800) | −0.027 [−0.039, −0.015] → content/format ratio **1.7×** |
| H1 | Payload effect, prompt tokens | **−16,533**/query [−16,870, −16,197] = −60% (27,719 → 11,186) |
| H1 | Payload effect, retrieval (arm-mean R@10) | +0.067 = **+8.4%** (511 q; +9.4% on the 300-q slice) |
| H2 | Best→worst format, correctness, per model × payload (equal content, n=300 each) | faceted: **2.7–5.2%** relative (glimmer toon→jsonld 0.864→0.840; nano yaml→jsonld 0.850→0.822; mini mat→jsonld 0.782→0.742) · unfaceted: **6.3–8.5%** (json→csv for all three models) |
| H2 | Pooled format contrasts whose CI excludes zero (vs JSON, n=1,800) | CSV −0.027, JSON-LD −0.015 (YAML grazes it at −0.0003); MaT and TOON indistinguishable |
| H2 | The JSON-LD retrieval outlier | R@10 −0.041 faceted / −0.060 unfaceted vs JSON (**5–9%** worse) at **+22% / +124%** tokens |
| H3 | MaT rank: retrieval vs raw-record reasoning | **1st of 6** (all four conditions) vs mid-pack |
| H3 | JSON rank: retrieval vs raw-record reasoning | 5th of 6 (faceted) vs **1st for 3 of 3 models** |
| H3 | JSON-LD rank: retrieval vs reasoning vs cost | 6th of 6 vs mid-pack vs most expensive (**2.2×** JSON, unfaceted) |
| H4 | Faceted winners per model (winner − runner-up gap; per-model uncertainty ≈ ±0.03 at n=300) | nano→YAML 0.850 (+0.003 over MaT) · mini→MaT 0.782 (+0.006 over TOON) · glimmer→TOON 0.864 (+0.002 over MaT) |
| H4 | Raw-record winner per model | JSON, **3 of 3 models** (0.811 / 0.740 / 0.837) |

---

## 2. What was run

**Three independent axes**

- **Payload** — *faceted*: a curated 32-field projection of the UMM record
  (placeholders dropped, nothing aggregated or pre-rendered) vs *unfaceted*:
  the raw record CMR serves (45 top-level fields, 311 distinct leaf paths).
- **Format** — JSON, YAML, TOON, CSV, JSON-LD, Metadata-as-Text (real
  prose). All six carry identical facts within a payload — parity is
  hard-gated at cache build, so format effects are representation-only.
- **Model** — `gpt-5.4-nano`, `gpt-5.4-mini` (cloud); `muse-glimmer:30b`
  (local 32B).

**Held fixed** — encoder `BAAI/bge-large-en-v1.5`, 510-token chunks with max
pooling, k=10 records per prompt, temperature 0, seeded query selection, one
judge (`gpt-5.4-nano`, DeepEval).

**Scale** — 2,590 records × 12 renderings; 511 queries for retrieval (11
SME-written + 500 synthetic); a 300-query slice for the LLM stages; 6 × 2 × 3
= 36 arms × 300 queries = **10,800 judged cells**. Spend: $71 OpenAI + ~145 h
local generation.

**Staged pipeline** — Stage R (exact retrieval, no LLM) → Stage A (answers,
frozen) → Stage J (judging, checkpointed). Every number traces to an
artifact; stages re-run without touching upstream results.

---

## 3. Retrieval findings (Stage R — exact, 511 queries)

![Chunked retrieval over faceted records](../runs/faceted_full_pooling.png)

![Chunked retrieval over unfaceted records](../runs/unfaceted_full_pooling.png)

- **Metadata-as-Text is first in all four conditions** (both payloads × both
  pooling rules) on all five metrics — and it is simultaneously the
  **cheapest format to embed** (fewest tokens, fewest chunks). Under this
  encoder there is no cost/quality trade-off to make.
- **JSON-LD is last or near-last everywhere**, at 1.7–4.4× the chunks of
  prose. It costs the most and retrieves the worst.
- **Faceted ≥ unfaceted for every format** (ΔRecall@10 +0.01 prose to +0.14
  CSV) — with the caveat that the synthetic query generator consumed a field
  summary that structurally resembles the faceted projection, so the
  payload gap at retrieval is an observation, not a finding.
- CSV is payload-dependent: 4th faceted, last unfaceted. An earlier "CSV is
  the stable negative" reading was retracted at 511 queries.
- No format fits the 512-token embedding window at even the median record —
  the justification for chunked retrieval.

---

## 4. Reasoning findings (Stages A/J — 300 queries, all arms complete)

![Correctness vs context cost](../runs/20260819T183313Z/pareto.png)

**The payload axis is the one large, unambiguous effect.** Faceted −
unfaceted correctness: **+0.045 [+0.038, +0.052]** (n=5,400 paired), holding
per model — nano +0.049, mini +0.051, glimmer +0.036 — at **16,533 fewer
prompt tokens per query (−60%)**. The curated projection is better *and*
cheaper for every model tested. Nothing else in the run is close.

![Faithfulness vs context cost](../runs/20260819T183313Z/faithfulness.png)

**Faithfulness buys no counter-argument.** Raw records recover only −0.006
[−0.012, −0.001] of faithfulness, at 2.5× the tokens and −0.045 correctness.
Not a trade worth making.

**All 36 arms** (run `20260819T183313Z`, n=300 each; **bold** = best
correctness / faithfulness / cheapest tokens within each model × payload
block, and best Recall@10 per payload; local-model $ not metered):

| Model | Payload | Format | Correct. | Faithful. | Prompt tok | $/1k q | R@10 |
|---|---|---|---:|---:|---:|---:|---:|
| nano | faceted | yaml | **0.850** | **0.863** | 11,758 | 2.63 | 0.865 |
| nano | faceted | mat | 0.847 | 0.839 | **9,334** | **2.16** | **0.873** |
| nano | faceted | toon | 0.843 | 0.856 | 11,347 | 2.54 | 0.858 |
| nano | faceted | json | 0.828 | 0.844 | 10,393 | 2.38 | 0.831 |
| nano | faceted | csv | 0.827 | 0.844 | 11,959 | 2.69 | 0.863 |
| nano | faceted | jsonld | 0.822 | 0.857 | 12,833 | 2.84 | 0.788 |
| nano | unfaceted | json | **0.811** | **0.892** | 21,533 | 4.64 | **0.806** |
| nano | unfaceted | mat | 0.810 | 0.865 | **17,178** | **3.75** | 0.827 |
| nano | unfaceted | jsonld | 0.796 | 0.843 | 48,587 | 10.04 | 0.766 |
| nano | unfaceted | yaml | 0.793 | 0.874 | 24,490 | 5.22 | 0.770 |
| nano | unfaceted | toon | 0.773 | 0.859 | 23,159 | 4.95 | 0.764 |
| nano | unfaceted | csv | 0.742 | 0.871 | 32,128 | 6.74 | 0.706 |
| mini | faceted | mat | **0.782** | 0.847 | **9,334** | **7.83** | **0.873** |
| mini | faceted | toon | 0.776 | **0.865** | 11,347 | 9.28 | 0.858 |
| mini | faceted | csv | 0.762 | 0.861 | 11,959 | 9.77 | 0.863 |
| mini | faceted | json | 0.755 | 0.852 | 10,393 | 8.57 | 0.831 |
| mini | faceted | yaml | 0.751 | 0.857 | 11,758 | 9.62 | 0.865 |
| mini | faceted | jsonld | 0.742 | 0.846 | 12,833 | 10.43 | 0.788 |
| mini | unfaceted | json | **0.740** | 0.861 | 21,533 | 17.10 | **0.806** |
| mini | unfaceted | mat | 0.734 | 0.836 | **17,178** | **13.70** | 0.827 |
| mini | unfaceted | jsonld | 0.704 | 0.829 | 48,587 | 37.36 | 0.766 |
| mini | unfaceted | yaml | 0.702 | 0.862 | 24,490 | 19.30 | 0.770 |
| mini | unfaceted | toon | 0.699 | 0.864 | 23,159 | 18.34 | 0.764 |
| mini | unfaceted | csv | 0.684 | **0.886** | 32,128 | 25.07 | 0.706 |
| glimmer | faceted | toon | **0.864** | 0.886 | 11,128 | — | 0.858 |
| glimmer | faceted | mat | 0.862 | 0.878 | **9,146** | — | **0.873** |
| glimmer | faceted | csv | 0.856 | 0.888 | 11,730 | — | 0.863 |
| glimmer | faceted | yaml | 0.846 | 0.877 | 11,535 | — | 0.865 |
| glimmer | faceted | json | 0.845 | **0.891** | 10,095 | — | 0.831 |
| glimmer | faceted | jsonld | 0.840 | 0.881 | 12,470 | — | 0.788 |
| glimmer | unfaceted | json | **0.837** | 0.886 | 20,834 | — | **0.806** |
| glimmer | unfaceted | mat | 0.833 | 0.887 | **16,911** | — | 0.827 |
| glimmer | unfaceted | jsonld | 0.823 | 0.877 | 47,733 | — | 0.766 |
| glimmer | unfaceted | toon | 0.812 | 0.882 | 22,936 | — | 0.764 |
| glimmer | unfaceted | yaml | 0.810 | 0.884 | 24,246 | — | 0.770 |
| glimmer | unfaceted | csv | 0.784 | **0.892** | 32,143 | — | 0.706 |

**Format effects are real, small, and model-dependent.** Paired vs JSON
(n=1,800 each): MaT +0.009, TOON −0.008, YAML −0.011, JSON-LD −0.015, CSV
−0.027 — only CSV and JSON-LD clear their intervals, and by less than the
payload effect. Faceted winners split three ways (nano→YAML 0.850, mini→MaT
0.782, glimmer→TOON 0.864); **under the raw record JSON wins for all three
models** — explicit structure appears to help when there are hundreds of
paths to navigate. Within-model spreads overlap the CIs: ordering is
suggestive, not established.

**Best→worst format deltas at equal content** (correctness, n=300 per cell;
descriptive post-hoc extremes, so read as ranges rather than tested
contrasts):

| Model | Payload | Best | Worst | Δ | relative |
|---|---|---|---|---:|---:|
| nano | faceted | YAML 0.850 | JSON-LD 0.822 | 0.028 | 3.3% |
| mini | faceted | MaT 0.782 | JSON-LD 0.742 | 0.041 | 5.2% |
| glimmer | faceted | TOON 0.864 | JSON-LD 0.840 | 0.023 | 2.7% |
| nano | unfaceted | JSON 0.811 | CSV 0.742 | 0.069 | 8.5% |
| mini | unfaceted | JSON 0.740 | CSV 0.684 | 0.056 | 7.6% |
| glimmer | unfaceted | JSON 0.837 | CSV 0.784 | 0.053 | 6.3% |

Two regularities: **format matters roughly twice as much on the raw record
as on the curated one** (6.3–8.5% vs 2.7–5.2% relative) — curation shrinks
format sensitivity, itself an H1-flavored result — and the extremes are
payload-driven, not model-driven: JSON-LD is always the faceted loser, CSV
always the unfaceted loser, JSON the unfaceted winner for all three models;
model identity only reshuffles the faceted podium.

**The model axis is interesting and least trustworthy.** Paired correctness
(n=3,600): mini − nano −0.076; glimmer − nano +0.022; mini − glimmer −0.098.
Read literally, a local 32B beats both cloud models — but the judge *is*
nano, grading its own answers in two of three comparisons. The one
judge-neutral comparison (mini vs glimmer) is the largest effect, and
glimmer's win over nano is biased against glimmer — the two comparisons that
don't flatter the judge both favour the local model. Also instructive:
glimmer's edge read +0.067 on the 50-query slice and +0.022 at 300 — the
small slice was SME-heavy, glimmer's best terrain. Glimmer is additionally
the most *faithful* arm (+0.025 over nano).

**Recall gates correctness.** Joining Stage R's exact Recall@10 with judged
correctness per cell (10,800 pairs): the per-query correlation is r = 0.74–
0.81 per model. When the expected record is fully retrieved, correctness runs
0.83–0.93; when it is fully missed, 0.33–0.43 (the judge grants partial
credit for related-but-wrong answers, so a miss floors near 0.35, not 0).
Model quality differences live almost entirely in the *hit* condition — mini
loses ~0.10 to nano/glimmer even when the right record is in context.
Decomposing the payload effect the same way: with retrieval equalized (both
payloads at R@10 = 1) faceted still wins, but by +0.012 rather than +0.045 —
**about three quarters of the payload effect is better retrieval; the
remaining quarter is better reasoning over the leaner context.**

**Expert questions are the hard part.** SME-written queries score ~0.3–0.4
correctness below synthetic ones for every model (synthetic queries inherit
vocabulary from the records that generated them). Glimmer has the smallest
expert gap and the best expert scores — but n=11 SME queries is far too few
to conclude anything.

---

## 5. Corpus coverage findings (sizes, tokens, fields)

**Size and tokens** (all 31,080 files measured; tiktoken `o200k_base`, with
a Qwen3-8B cross-check within 5–8% at identical ranking — the differences
are length, not tokenizer vocabulary):

| | Faceted | Unfaceted | ratio |
|---|---:|---:|---:|
| Corpus size | 53.7 MB | 141.9 MB | 2.6× |
| Mean tokens/record (cheapest fmt: MaT) | 805 | 1,476 | 1.8× |
| Mean tokens/record (dearest fmt: JSON-LD) | 1,110 | 4,295 | 3.9× |
| Leaf values/record (mean) | 39.6 | 149.1 | 3.8× |

Every payload × format cell has a heavy tail — p95 ≈ 2× the mean, max 20–30×
the median; the corpus worst case renders at 74,528 bge tokens.

**Field-level: present is not the same as informative.** Two mechanical
measures per field — *present* (the source value exists, placeholders
included) and *informative* (it survives the projection's placeholder rule):

- Faceted: **22.8 of 32 facets present vs 20.7 informative** per record —
  two facets per record are placeholder noise.
- Unfaceted: **23.5 of 45 fields present (52.2%) vs 21.0 informative
  (46.8%)**; range 14–35. Half the schema is absent from a typical record.
- All 13 UMM-required fields are 100% present, but **only 7 are 100%
  informative**: `DOI` collapses to 43.7%, `ProcessingLevel` to 25.6%,
  `Version` to 54.8%, `Platforms` to 68.3%, `CollectionProgress` to 76.3%,
  `Abstract` to 99.7%. **Required means present; it does not mean
  informative.**
- Biggest placeholder pools: `ProcessingLevel` (1,927 records of
  `"Not provided"`), `DOI` (1,458 MissingReason stubs), `Version` (1,170),
  platform names (932), `CollectionProgress` (613), `data_format` (334).
- Of 311 distinct leaf paths, **six are never informative in any record** —
  led by `DOI.MissingReason` (1,458) and `DOI.Explanation` (1,432): ~2,900
  field instances whose entire content is "we don't know", rendered and
  embedded in every unfaceted representation.
- The sparse tail caps everything: `spatial_resolution` 4.7%,
  `temporal_resolution` 2.5%, `techniques` 2.4%. No serialization can answer
  a question about a field that isn't there.

This audit reframes the headline result: faceting doesn't just shrink the
context, it **removes anti-signal** — identical content-free strings that
make unrelated records look alike in embedding space.

**How placeholders actually do their damage** (measured by stripping every
placeholder leaf from the unfaceted JSON renderings and re-counting):

- **Token cost is the small part: ~36 tokens/record, 1.9%** of the
  unfaceted rendering (median 39, p95 71; ~93k tokens corpus-wide; ~360
  tokens of "we don't know" in every k=10 prompt). The −60% payload saving
  comes overwhelmingly from dropping verbose-but-real content — contacts,
  addresses, metadata dates, citations — not placeholders.
- **Embedding pollution is the disproportionate part**: the same ~40-token
  strings recur verbatim in 600–1,900+ records each, pulling unrelated
  records toward each other in vector space and diluting every chunk they
  occupy — a similarity distortion token counts don't capture.
- **Faithfulness, counterintuitively, they *inflate*.** A placeholder is
  perfectly groundable — the unfaceted prose literally says "version Not
  provided", and a model echoing it is faithful while informing no one.
  Consistent with the observed sign: the raw payload scores slightly
  *higher* faithfulness (+0.006) despite −0.045 correctness. Faithfulness
  rewards grounding, not information; placeholders are free grounding — one
  more reason correctness, not faithfulness, is the end-to-end metric.
- **Open question**: whether *dropping* placeholders (absence) invites
  hallucinated specifics where "Unknown" would have invited an honest gap
  report — the designed-but-unrun sparse-metadata stress test (the original
  `Report.md` program's Experiment 4) is built to answer exactly this.

---

## 6. Follow-on experiment: LLM summaries appended to every rendering

**Question.** The study's `mat` rendering is deterministic template prose. If
an LLM instead *writes* a dense summary of each record and that summary is
appended to every format rendering, does retrieval improve?

**What was done** (`scripts/summary_retrieval.py` generate,
`scripts/summary_append_eval.py` cache/tokens/build/eval):

1. `gpt-5.4-nano` (temperature 0) summarised each record's cached MaT
   rendering, once per payload — 2,590 records × 2 = 5,180 calls, ~$4.30 at
   nano pricing, every call audit-logged. Summaries live in
   `data/format_cache_summary/` (from faceted MaT) and
   `data/format_cache_unfaceted_summary/` (from raw-record MaT).
2. Each summary was appended (after a `"Summary: "` separator) to each of the
   record's six format renderings, producing two NEW caches —
   `data/format_cache_plus_summary/`, `data/format_cache_unfaceted_plus_summary/`
   — the originals untouched. The combined document is for *embedding*, not
   parsing (prose after JSON deliberately breaks its syntax).
3. Chunked retrieval ran with the identical stack as the format study
   (bge-large-en-v1.5, 510-token chunks / 64 overlap, max pooling, the same
   511 queries) against fresh chroma databases
   (`data/chroma/{faceted,unfaceted}_plus_summary_chunked/`).

**The generation prompt** (system: *"You write dense, factual prose summaries
of Earth-science dataset metadata records, optimized for semantic-search
retrieval."*):

> Summarize the following dataset metadata record as ONE dense prose
> paragraph of roughly 120-180 words.
>
> Rules:
> - Preserve exactly: the dataset title, short name, organisation and
>   data-center names, platform and instrument names, variable and
>   science-keyword terms, and spatial/temporal coverage values.
> - Plain prose only: no markdown, no lists, no headings.
> - Use only facts stated in the record; never add outside knowledge.
> - Skip placeholders ("Not provided", "Unknown", absent DOIs) and say
>   nothing about missing information.

**Result — the summary improves retrieval in all 12 payload × format cells**
(Recall@10, max-pool, 511 queries, vs the frozen no-summary baselines):

*Faceted payload:*

| Format | R@10 before | with summary | Δ | relative |
|---|---:|---:|---:|---:|
| JSON-LD | 0.813 | 0.878 | +0.066 | **+8.1%** |
| CSV | 0.872 | 0.911 | +0.039 | **+4.5%** |
| TOON | 0.873 | 0.911 | +0.038 | **+4.4%** |
| JSON | 0.854 | 0.887 | +0.034 | **+4.0%** |
| YAML | 0.874 | 0.905 | +0.031 | **+3.5%** |
| MaT | 0.884 | 0.903 | +0.019 | **+2.1%** |

*Unfaceted payload:*

| Format | R@10 before | with summary | Δ | relative |
|---|---:|---:|---:|---:|
| CSV | 0.728 | 0.803 | +0.076 | **+10.4%** |
| JSON-LD | 0.771 | 0.847 | +0.076 | **+9.9%** |
| JSON | 0.831 | 0.874 | +0.043 | **+5.2%** |
| TOON | 0.787 | 0.828 | +0.041 | **+5.2%** |
| YAML | 0.802 | 0.838 | +0.035 | **+4.4%** |
| MaT | 0.849 | 0.883 | +0.033 | **+3.9%** |

**Reading it:**

- **No exceptions**: every cell improves, +2.1% to +10.4% relative. New
  corpus bests: faceted CSV/TOON + summary at 0.911 (the previous best was
  faceted MaT at 0.884).
- **Gains are inversely proportional to how prose-like the baseline already
  was** — the worst retrievers (JSON-LD, unfaceted CSV) gain 8–10%, prose
  gains 2–4%. A prose head partially insures any format against its own
  retrieval weaknesses: the format spread shrinks from 0.071 to 0.033
  (faceted) and 0.121 to 0.080 (unfaceted).
- **The payload hierarchy survives**: the best unfaceted+summary cell (MaT,
  0.883) only just reaches the *baseline* faceted MaT (0.884). A summary
  bolted onto the raw record nearly buys back the curation advantage at
  retrieval — but curation + summary still beats it, at far fewer tokens.
- **Token cost of the append** (below): a uniform +402 tokens/record faceted
  and +430 unfaceted — the summary is identical across formats by
  construction, so the *relative* cost is highest exactly where the retrieval
  gain is lowest (MaT) and lowest where the gain is highest (JSON-LD).

**Token consumption per format** (mean tiktoken `o200k_base` per record,
from `runs/summary_append/tokens.json`):

| Format | Faceted | + summary | Δ | | Unfaceted | + summary | Δ |
|---|---:|---:|---:|---|---:|---:|---:|
| MaT | 805 | 1,207 | **+50.0%** | | 1,476 | 1,906 | **+29.1%** |
| JSON | 913 | 1,315 | +44.0% | | 1,918 | 2,348 | +22.4% |
| TOON | 931 | 1,333 | +43.2% | | 2,026 | 2,457 | +21.3% |
| YAML | 971 | 1,373 | +41.4% | | 2,100 | 2,531 | +20.5% |
| CSV | 984 | 1,386 | +40.9% | | 2,780 | 3,210 | +15.5% |
| JSON-LD | 1,110 | 1,512 | **+36.2%** | | 4,295 | 4,726 | **+10.0%** |

**Caveats**: appended text adds chunks (e.g. faceted JSON 2.72 → 3.61 per
record) and max-pooling rewards chunk count, so part of the gain is bought by
length — mean-pool gains are smaller but consistently positive. And the
summaries were written by nano from the record's own MaT text, so they share
vocabulary provenance with the synthetic queries. Reasoning-stage effects
(does the summary help *answers*?) are unmeasured — retrieval only.

Artifacts: `runs/summary_append/{retrieval_eval.json,
retrieval_eval_faceted.json, retrieval_rows*.csv, tokens.json}`; summary
caches carry `manifest.json` with model, prompt hash, and call-log path.

---

## 7. Caveats

1. **The judge is a model under test.** All quality scores are nano-graded;
   the model ranking is the least trustworthy result. Format and payload
   effects are within-judge comparisons and far more robust.
2. **Query provenance leans faceted, and most of the payload effect flows
   through retrieval.** The synthetic generator consumed a curated field
   summary, so the retrieval payload gap carries that asterisk — and each
   payload retrieves its own contexts, so the end-to-end +0.045 inherits it
   in part: conditional on *both* payloads fully retrieving the expected
   record, the faceted advantage is +0.012 [+0.008, +0.016] (n=3,918) — a
   real but much smaller pure reasoning-side gain. The remaining ~three
   quarters of the effect is retrieval-mediated.
3. **One encoder, one chunking scheme.** Prose ranked last under a different
   encoder on whole documents and first here — retrieval rankings are
   encoder-conditional.
4. **Format rankings within a model are not statistically separated** — only
   the pooled CSV and JSON-LD penalties clear their intervals.
5. **Expert coverage is thin** — 11 SME queries carry all realistic
   difficulty.
6. **Informative ≠ correct.** The coverage audit checks whether a value
   asserts a fact, not whether the fact is true; and the placeholder rule is
   a finite list, so informative shares are upper bounds.
7. **One corpus snapshot.** 2,590 records, not all of CMR; placeholder
   practices vary by provider; numbers drift (an older "398 paths" figure no
   longer reproduces — today it is 311 within `umm`).

---

## 8. Answers to the research questions

The four hypotheses were the falsifiable form of two broader research
questions. Mapping the findings back:

**RQ1 — How do LLMs interact with and utilize domain-specific metadata in
complex archives like CMR?**

In two distinct modes with different preferences — and the bottleneck is
what reaches the model, not how it is spelled.

- **Consumption is two-stage, and the stages disagree (H3).** The embedding
  stage prefers prose (MaT first in all four retrieval conditions); the
  reasoning stage prefers explicit structure when the record is large and
  deeply nested (JSON wins raw-record reasoning for 3 of 3 models). No
  single representation serves both stages best.
- **Retrieval gates everything.** Per-query correctness correlates with
  exact Recall@10 at r = 0.74–0.81; a retrieved record yields 0.83–0.93
  correctness, a missed one 0.33–0.43. An LLM's "use" of metadata is
  mostly determined before it ever sees a token of it.
- **Models are robust to serialization but sensitive to content bloat
  (H1, H2).** At equal content, the best-to-worst format spread is
  2.7–5.2% relative; but the raw 45-field record costs −0.045 correctness
  at 2.5× the tokens versus a curated projection. Format sensitivity
  roughly doubles on the raw record — structure earns its keep only when
  there are hundreds of paths to navigate.
- **Placeholders are consumed as anti-signal.** ~40-token "Not provided"
  stubs recur verbatim across 600–1,900+ records, pulling unrelated
  records together in embedding space — and they *inflate* faithfulness
  while informing no one, since echoing a placeholder is perfectly
  grounded.
- **Preferences are model-conditional (H4), and domain vocabulary is the
  hard part.** Three models produced three different curated-payload
  winners, and SME-written queries score ~0.3–0.4 below synthetic ones —
  the vocabulary gap between how experts ask and how records are written
  dwarfs every representation effect measured.

**RQ2 — How can metadata architectures be optimized to improve AI search,
retrieval, and accuracy?**

Curate the content, serve mode-specific representations, and fix
informativeness before format — the full prescription is §9; the evidence
in one breath:

- **Serve a curated projection as the AI-facing representation.** The
  32-field faceted payload is the one intervention that wins every axis
  at once: +0.045 correctness, +8.4% Recall@10, −60% tokens, for every
  model tested (H1). Keep the canonical record as the record of authority.
- **Match representation to consumption mode.** Prose (MaT) for anything
  that gets embedded — best retrieval everywhere and cheapest; structured
  JSON where an agent must reason over a full raw record; JSON-LD for
  neither (worst retriever, most expensive rendering).
- **Optimize informative presence, not schema compliance.** All 13
  required fields are 100% present but only 7 are reliably informative;
  placeholder remediation (1,927 processing levels, 1,458 DOIs, 1,170
  versions) and populating the sparse tail (resolution fields at 2–5%)
  raise the ceiling no serialization choice can touch.
- **An appended LLM summary is a cheap retrieval upgrade** — +2.1% to
  +10.4% Recall@10 in all 12 cells, largest exactly where the base format
  retrieves worst (reasoning-stage effect still unmeasured).
- **Institutionalize the measurement, not the winner.** Format rankings
  are model- and encoder-conditional and reshuffle as models change; the
  durable asset is the audited evaluation harness re-run at each corpus
  refresh (H4's real lesson).

---

## 9. What to do about it — the AI-readiness playbook

Ordered by evidence-to-effort. The through-line: **most of AI-readiness is
not AI work — it is metadata hygiene plus serving a curated, prose-friendly
projection.**

**Tier 1 — fix what the data says (highest leverage, no ML required)**

1. **Track "informative presence per required field."** Compliance
   dashboards show 100% green while `ProcessingLevel` is real in 25.6% of
   records. The `MissingReason` machinery makes the honest number a query,
   not a project. What gets measured gets fixed.
2. **Run placeholder-remediation campaigns** against the gap league table
   (1,927 processing levels, 1,458 DOIs, 1,170 versions — enumerable per
   provider, often resolvable from the data itself).
3. **Populate the sparse tail where users actually ask questions** —
   resolution fields at 2–5% make whole question classes unanswerable
   regardless of representation.

**Tier 2 — fix how the data is served (this study's direct evidence)**

4. **Publish a curated projection as a first-class AI representation**
   (+0.045 correctness at 40% of the tokens, every model). Keep the
   canonical UMM record untouched as the record of authority.
5. **Ship a Metadata-as-Text summary per collection** — best retrieval in
   every condition, cheapest to embed, top-two reasoning; one deterministic
   template per schema field, no generation risk.
6. **Do not invest in JSON-LD for LLM consumption** — worst retriever, most
   expensive rendering. Serve it for web interoperability if wanted.

**Tier 3 — fix the interface (strong literature evidence, untested here)**

7. **Inline field descriptions and enums** everywhere an agent looks — the
   function-calling literature predicts the largest single gain (30%+); this
   is the original program's designed-but-unrun description-quality ablation
   (`Report.md` §6, its "Experiment 2" — not the chunked-retrieval
   experiment numbered 2 in this study's write-ups).
8. **Distinguish "absent" from "unknown" explicitly for the model** — UMM's
   `MissingReason` is good design serialized badly; "DOI: none assigned"
   lets a model report gaps instead of guessing.
9. **Maintain the semantic-search MCP path** (`nasa/earthdata-mcp` exists)
   so agents are told the tools and parameters rather than guessing URLs.

**Tier 4 — fix the process**

10. **Fund the harness, not a format.** Winners are model- and
    encoder-dependent and models change; the repeatable, audited evaluation
    pipeline is the durable artifact. Re-run the coverage scripts each
    corpus refresh; track trends, not snapshots.

---

## 10. Future work

- **Neutral-judge re-run** (mini or glimmer as judge) to bound
  self-preference — the model axis is the most interesting and least
  trustworthy result.
- **Second encoder** over the same chunks to break the encoder/chunking
  confound behind prose's retrieval win.
- **More SME queries** — the single highest-leverage data investment.
- **Summary experiment, reasoning stage** — §6 measured retrieval only; run
  Stages A/J over the `+summary` caches to see whether the appended summary
  also helps answers, and re-score the SME slice specifically.
- From the original `Report.md` program (its numbering, not this study's):
  **field-description ablation** (its Exp 2), **direct comprehension**
  without retrieval (its Exp 5), **sparse-metadata stress test** measuring
  hallucination vs honest gap-reporting (its Exp 4), and an **agentic/MCP
  arm** against static representation.

---

## 11. Where everything lives

| Artifact | Path |
|---|---|
| Judged cells + summary | `runs/20260819T183313Z/{judgements.jsonl, judge_summary.json}` |
| Analysis + figures | `runs/20260819T183313Z/{analysis.json, pareto.png, faithfulness.png}` |
| Retrieval studies | `runs/{faceted,unfaceted}_full_{max,mean}/`, `runs/*_full_pooling.png` |
| Corpus audit | `runs/cache_coverage/{coverage.json, field_coverage.json, per_record.csv, field_presence.csv}` |
| Rendering caches | `data/format_cache/`, `data/format_cache_unfaceted/` (2,590 × 12, parity-gated) |
| Summary experiment | `runs/summary_append/`, `data/format_cache{,_unfaceted}_summary/`, `data/format_cache{,_unfaceted}_plus_summary/`, `data/chroma/*_plus_summary_chunked/` |
| Scripts | `scripts/{judge_stage.py, analyze_50q.py, cache_coverage.py, field_coverage.py, summary_retrieval.py, summary_append_eval.py}` |
| Slide deck | `slides/ai_metadata_formats_slides.{pdf,md,tex}` |

*Prepared 2026-08-24, summary experiment added 2026-08-25; runs
`20260819T183313Z`, `*_full_*`, the `cache_coverage` audit, and
`runs/summary_append/`; judge `gpt-5.4-nano` at temperature 0 throughout.*
