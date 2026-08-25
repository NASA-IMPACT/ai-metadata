# AI-Optimized Science Metadata Formats

Slide-by-slide text, mirroring `ai_metadata_formats_slides.pdf`. Figures referenced by repo path.

---

## Slide 1 — Title

**AI-Optimized Science Metadata Formats**

*Does representation matter when an LLM searches and reads CMR metadata?*

Faceted vs. raw UMM × six serializations × three models, measured end to end

Bernard Benson · August 2026 · runs `20260819T183313Z` (LLM) and `*_full_*` (retrieval)

---

## Slide 2 — The pitch we made

> The best representation is **not universal** — it is consumption-mode- and model-dependent, and must be measured empirically.

Made falsifiable by crossing **format** against **content scope** — the confound the field has been ignoring.

- **H1 — Content dominates**: field selection matters more than format for retrieval.
- **H2 — Format effect is small**: at equal content, representations barely differ.
- **H3 — The axes disagree**: retrieval, reasoning, and cost need not rank formats alike.
- **H4 — Best format varies**: the top reasoning representation differs by model family.

---

## Slide 3 — Verdict up front

| Hypothesis | Verdict | Why |
|---|---|---|
| **H1** Content dominates | **Supported** | Payload effect is the largest in the study, on every axis, for every model. |
| **H2** Format effect is small | **Supported\*** | At equal content, the largest format penalty is about half the payload effect. \*JSON-LD is a real retrieval outlier. |
| **H3** The axes disagree | **Supported** | JSON wins raw-record reasoning but sits 5th of 6 in retrieval; JSON-LD is last in retrieval yet mid-pack in reasoning. |
| **H4** Best format varies | **Suggestive** | Three models, three different curated-payload winners — but within-model spreads overlap their confidence intervals. |

Every verdict comes from one audited pipeline: 10,800 judged answers over real CMR records.

---

## Slide 4 — Experimental setup: the grid

**Three independent axes**

- **Payload** — *faceted*: curated 32-field projection vs. *unfaceted*: raw UMM record (45 top-level fields, 398 paths)
- **Format** — JSON, YAML, TOON, CSV, JSON-LD, Metadata-as-Text (prose); all six carry identical facts (parity hard-gated at render)
- **Model** — `gpt-5.4-nano`, `gpt-5.4-mini` (cloud); `muse-glimmer:30b` (local 32B)

**Held fixed** — encoder (bge-large-en-v1.5), 510-token chunks, max pooling, k=10 records in context, temperature 0, seed, one judge

**Scale**

- 2,584 real CMR collection records × 12 renderings
- 511 queries for retrieval (11 SME + 500 synthetic)
- 300-query slice for LLM stages
- 6 × 2 × 3 = 36 arms × 300 queries = **10,800 judged cells**
- Spend: $71 OpenAI + ~145 h local generation

---

## Slide 5 — Experimental setup: staged pipeline, every number auditable

| Stage | What it does | Measures | LLM involved |
|---|---|---|---|
| **R** Retrieve | chunked vector search per format/payload | Recall@k, MRR, nDCG@10 | none — exact |
| **A** Answer | top-10 records into each model, temp 0 | prompt/completion tokens, latency, $ | the 3 under test |
| **J** Judge | DeepEval metrics on frozen answers | correctness, faithfulness, … | `gpt-5.4-nano` |

- Stages are frozen artifacts — judging re-runs never touch answers; checkpointed and resumable
- Correctness is the end-to-end metric (knows ground truth); the rest are diagnostics
- Content parity across formats is enforced, so format effects are representation-only

The same 60% token gap shows up before the LLM ever runs: faceted renders at 786–1,465 bge tokens/record; unfaceted at 1,462–7,012.

---

## Slide 6 — Stage R: retrieval — prose wins, JSON-LD loses, faceted leads

*Figure: `runs/faceted_full_pooling.png`*

- **Metadata-as-Text is first in all four conditions** (both payloads × both poolings), on all five metrics — and it is also the **cheapest** format to embed
- **JSON-LD is last or near-last everywhere**, at 1.7–4.4× the chunks of prose
- Faceted ≥ unfaceted for **every** format: ΔR@10 +0.01 (prose) to +0.14 (CSV) — with a query-provenance caveat, later

---

## Slide 7 — Stage A/J: correctness vs. context cost (300 queries, all arms)

*Figure: `runs/20260819T183313Z/pareto.png`*

- Faceted sits up and to the left of unfaceted in every panel: **more correct at ~40% of the tokens**
- Payload effect +0.045 correctness [+0.038, +0.052], n=5,400 pairs; holds per model
- Cheapest strong point: faceted MaT ≈ 9.1–9.3k tokens/query, top-two correctness for all models

---

## Slide 8 — Stage A/J: faithfulness — no trade-off worth making

*Figure: `runs/20260819T183313Z/faithfulness.png`*

- Raw records buy −0.006 [−0.012, −0.001] faithfulness back — barely detectable, at 2.5× the tokens and −0.045 correctness
- The local model is the most grounded arm: glimmer +0.025 faithfulness over nano
- Faithfulness never separates the formats the way correctness separates the payloads

---

## Slide 9 — H1: Content dominates — **Supported**

**The payload axis is the largest effect in the study, on all three measurement axes.**

| Axis | Faceted vs. unfaceted | Largest format effect | Ratio |
|---|---|---|---|
| Reasoning (correctness) | +0.045 [+0.038, +0.052] | CSV −0.027 vs. JSON | ~1.7× |
| Retrieval (R@10) | +0.01 to +0.14, every format | JSON-LD −0.07 faceted | similar† |
| Cost (prompt tokens) | −16,533/query (−60%) | ~−3,500 within payload | ~5× |

- Per model: nano +0.049, mini +0.051, glimmer +0.036 — all clear of zero at n=1,800
- The curated projection is **better and cheaper for every model tested** — the run's headline finding, and it is not close
- †Retrieval caveat: the synthetic query generator saw a field summary that structurally resembles the faceted projection — payload gap there is an observation, not a finding

---

## Slide 10 — H2: Format effect is small — **Supported**, with one outlier

**At equal content, serialization moves correctness by at most half the payload effect.**

Reasoning, paired vs. JSON (n=1,800 each):

| Format | Correctness vs. JSON |
|---|---|
| MaT (prose) | +0.009 [−0.003, +0.020] |
| TOON | −0.008 [−0.019, +0.003] |
| YAML | −0.011 [−0.022, −0.000] |
| JSON-LD | −0.015 [−0.027, −0.003] |
| CSV | −0.027 [−0.039, −0.015] |

- Only CSV and JSON-LD clearly separate from JSON — and by less than the payload's 0.045
- **Format matters far less than what you put in it**
- The outlier: at *retrieval*, JSON-LD trails by 0.07–0.10 R@10 while costing the most tokens — format is not *free* at the embedding stage

---

## Slide 11 — H3: The axes disagree — **Supported**

**No single format wins retrieval, reasoning, and cost at once — rankings genuinely diverge.**

| Format | Retrieval (chunked) | Reasoning, faceted | Reasoning, raw | Cost |
|---|---|---|---|---|
| MaT (prose) | **1st, all 4 conditions** | top-two, all models | mid | **cheapest** |
| JSON | 5th of 6 (faceted) | mid | **1st, all 3 models** | mid |
| CSV | 4th faceted / **last** raw | worst pooled penalty | last (2 of 3) | high (raw) |
| JSON-LD | **last** everywhere | mid-pack | mid | **highest** |

- Even the judge's diagnostics disagree: contextual recall runs *counter* to correctness (faceted MaT 0.36 vs. raw JSON-LD 0.60 — while beating it 0.847 to 0.796 on correctness)
- Consequence: a repository cannot pick one representation off a single benchmark axis — exactly the claim in the pitch

---

## Slide 12 — H4: Best format varies by model — **Suggestive**

**Three models, three different winners under the curated payload — but the gaps are inside the error bars.**

| Model | Faceted winner | Raw-record winner |
|---|---|---|
| `gpt-5.4-nano` | YAML (0.850) | JSON (0.811) |
| `gpt-5.4-mini` | MaT (0.782) | JSON (0.740) |
| `muse-glimmer:30b` | TOON (0.864) | JSON (0.837) |

- Under the **raw record, JSON wins for all three** — explicit structure appears to help when there are 398 paths to navigate; content scope *modulates* format sensitivity
- Within-model spreads (0.028–0.040) overlap the paired CIs (±0.011–0.012): the *ordering* is suggestive, not established — H4 needs more SME-hard queries to close
- What *is* established: no format is safe to hard-code across models

---

## Slide 13 — The model axis: interesting, and least trustworthy

Paired correctness (n=3,600 each):

| Comparison | Δ correctness |
|---|---|
| mini − nano | −0.076 |
| glimmer − nano | +0.022 |
| mini − glimmer | −0.098 |

Read literally: a local 32B beats both cloud models, and the small cloud model beats the larger one.

**Before believing that:**

- The judge **is** `gpt-5.4-nano` — it grades its own answers in two of three comparisons, and wins both. Self-preference cannot be separated from quality here
- The one **judge-neutral** comparison (mini vs. glimmer) is the *largest* effect — and glimmer's win over nano is biased *against* glimmer, so it is conservative
- Slice size matters: glimmer's edge read +0.067 at 50 queries, +0.022 at 300 — the small slice was SME-heavy, glimmer's best terrain

---

## Slide 14 — The hard part: expert questions

**SME-written queries score ~0.3–0.4 correctness below synthetic ones, for every model.**

| Model (faceted, range over formats) | SME (n=11) | Synthetic (n=289) | Gap |
|---|---|---|---|
| `gpt-5.4-nano` | 0.36–0.48 | 0.84–0.87 | ~0.43 |
| `gpt-5.4-mini` | 0.28–0.42 | 0.76–0.80 | ~0.40 |
| `muse-glimmer:30b` | **0.47–0.63** | 0.85–0.87 | ~0.31 |

- Synthetic queries inherit vocabulary from the records they were generated from; expert questions do not — **the SME slice is the realistic difficulty**
- Pooled means are dominated by slice composition; compare per-slice, never pooled
- The local model has the smallest expert gap and the best expert scores — worth chasing, but n=11 is far too few expert queries to conclude anything

---

## Slide 15 — What we would push back on ourselves

- **The judge is a model under test.** All quality scores are nano-graded; the model ranking is the result we trust least. (The format and payload effects are within-judge comparisons and far more robust.)
- **Query provenance leans faceted, and ~¾ of the payload effect is retrieval-mediated.** The synthetic generator consumed a curated field summary, and each payload retrieves its own contexts. With retrieval equalized (both payloads at R@10 = 1), faceted still wins — but by +0.012, not +0.045.
- **One encoder, one chunking scheme.** Prose ranked *last* under a different encoder on whole documents and first here — retrieval rankings are encoder-conditional. Chunk boundaries also cut JSON mid-object but prose mid-paragraph.
- **Format rankings within a model are not statistically separated** — only the pooled CSV and JSON-LD penalties clear their intervals.
- **Expert coverage is thin**: 11 SME queries carry all realistic difficulty.

---

## Slide 16 — Future work

**Close the loops this run opened**

- **Neutral judge re-run** — re-grade a slice with mini or glimmer as judge to bound self-preference; the pipeline already supports coexisting judges per cell
- **Second encoder** — one more index (same chunks, different encoder) separates the encoder/chunking confound behind prose's retrieval win
- **More SME queries** — the realistic slice is 11 questions; this is the single highest-leverage data investment

**Next experiments from the original program**

- **Field-description ablation** — bare names vs. inline descriptions vs. enums; the literature predicts the largest single gain (Gorilla, ToolAlpaca: +30%)
- **Direct comprehension** — deterministic Q&A over single records, no retrieval, no judge
- **Sparse-metadata stress test** — ablate fields, measure hallucination vs. honest gap-reporting per representation
- **Agentic/MCP arm** — static representation vs. tool-calling agent (`nasa/earthdata-mcp` already exists)

---

## Slide 17 — Takeaways for CMR

1. **Publish a curated projection as a first-class AI representation.** A 32-field facet of UMM was more correct *and* 60% cheaper than the raw record for every model tested. This is the actionable finding, and it is not close.
2. **Ship a Metadata-as-Text summary alongside it.** Prose won retrieval in every condition, is the cheapest rendering to embed, and stayed top-two on reasoning — the strongest single default for embedding pipelines.
3. **Keep the canonical UMM record untouched.** The projection is an *added* representation; precision, provenance, and audit stay with the record of authority.
4. **Do not assume JSON-LD helps LLM consumption.** It was the most expensive rendering and the worst retriever here. Its value is interoperability and linking — serve it for that, not as LLM context.
5. **Content curation first, format second.** Deciding *which fields* go into context beats any serialization debate by a wide margin.
6. **Fund the harness, not a format.** Winners are model- and encoder-dependent and models change; a repeatable, audited evaluation pipeline is the durable artifact. (This one is open and resumable.)

---

## Slide 18 — Close

**Measure, don't assume.**

The best representation is not universal — and now that's a measurement, not a slogan.

- Pipeline & analysis: `ai-metadata` repo — `reports/STAGED_PIPELINE.md`, `reports/ANALYSIS_300Q.md`
- Retrieval study: `reports/CHUNKED_RETRIEVAL_511Q.md`
- Literature synthesis: `reports/Report.md`
- Contact: Bernard Benson
