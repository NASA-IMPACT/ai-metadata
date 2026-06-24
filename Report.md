# Making Scientific Metadata AI-Ready

### How repositories like NASA's Common Metadata Repository should restructure metadata for large language models — a synthesis of the data-stewardship and RAG literatures

---

## Executive summary

The question "how should a metadata provider like NASA's Common Metadata Repository (CMR) restructure its holdings so that large language models can use them well?" sits at the meeting point of two research communities that have, until recently, developed in parallel.

The first is the **data-stewardship tradition** — FAIR, "metadata-as-data-intelligence," provenance documentation, and reproducible-research scholarship — which has spent a decade making metadata *machine-actionable* for autonomous software agents. The second is the **recent NLP / retrieval-augmented-generation (RAG) literature**, which measures how language models actually consume structured text and where they fail.

The central finding of this report is that these two literatures point to a **consistent but more disciplined conclusion than the popular "just flatten everything into prose for the AI" advice**. The stewardship literature shows that good metadata is identified, richly described, interoperably encoded, and accompanied by provenance — a standard that the FAIR principles articulated in 2016 and that the 2026 "AI-ready data" literature now reframes explicitly for AI. The RAG literature shows that *representation matters empirically and per-model*: structure can both help (faithfulness, disambiguation) and hurt (token cost, reasoning), and the best format is an experimental question rather than a fixed answer.

The defensible recommendation for a provider like CMR is therefore **a multi-representation, provenance-rich, FAIR-aligned metadata layer**: keep the precise canonical record, add a flattened natural-language summary for embedding, expose rich inline field descriptions for agentic tool use, and carry machine-actionable provenance — then validate each representation empirically. The closing section proposes five experiments to do exactly that.

---

## 1. Background: what is actually being asked

The motivating scenario is a user asking an LLM something like *"find high-resolution vegetation-height data from the ATLAS instrument over the Amazon in December 2024,"* and the model translating that into a structured query against CMR's Unified Metadata Model (UMM). CMR is large enough that representation choices have real consequences: as of 2024 it described over a billion data files across roughly 10,000 collections, plus tens of thousands more from international partners.

The proposal under examination — that providers should serve flatter, more descriptive, more semantically linked, "LLM-ready" metadata — breaks into four testable claims:

1. Flatter or natural-language representations improve retrieval over deeply nested structures.
2. Inline field descriptions and richer schemas improve machine use.
3. Knowledge-graph / linked-data structure improves discovery.
4. Agentic access (e.g., Model Context Protocol servers) improves the end-to-end workflow.

The papers supplied for this report supply the **principled, governance-level foundation** for those claims; the RAG literature reviewed previously supplies the **empirical, measurement-level evidence**. The two are combined below.

---

## 2. The data-stewardship foundation

### 2.1 FAIR established "machine-actionability" as the goal — before LLMs existed

The [FAIR Guiding Principles (Wilkinson et al., 2016)](https://www.nature.com/articles/sdata201618) are the keystone. Their distinctive contribution, relative to peer initiatives of the time, was the explicit insistence that data and metadata be usable not only by human scholars but by *computational agents* acting on their behalf. The paper introduces machine-actionability as a *continuum*: a well-described object lets an agent identify what it is, judge whether it is useful for the current task, determine whether it is usable under licensing constraints, and then act — much as a human would.

Several FAIR sub-principles map directly onto the CMR question:

- **F2 / R1** — data described with *rich* metadata, "a plurality of accurate and relevant attributes." This is the principled version of "richer schemas help."
- **I1** — metadata use a formal, shared, broadly applicable knowledge-representation language. This anticipates the JSON-LD / RDF / knowledge-graph recommendation.
- **I3** — metadata include *qualified references to other metadata*. This is the cross-linking ("Instrument → Platform → Variable") argument, stated as a principle.
- **R1.2** — metadata carry detailed provenance.
- **A2** — metadata remain accessible even when the underlying data are not.

Crucially, FAIR is **implementation-agnostic**: it specifies desiderata, not formats. That is exactly why it does not, on its own, answer "JSON vs. flattened text vs. graph" — it tells you the metadata must be richly described and interoperably encoded, but leaves the encoding to be chosen and, as the RAG literature shows, *measured*. FAIR also warns against over-reliance on bespoke parsers for every data type, favouring general, open interoperability technologies — a caution worth remembering when designing any single "LLM-optimized" format.

### 2.2 Metadata is "intelligent data" — and AI both consumes and produces it

The editorial framing the [*Metadata as Data Intelligence* special issue (Greenberg et al., 2023)](https://direct.mit.edu/dint/article/5/1/1/115168/Metadata-as-Data-Intelligence) positions metadata as value-added, "intelligent" data, and notes that interest in it has accelerated precisely because of two forces: the global adoption of FAIR (which is heavily metadata-driven) and the rise of AI/ML. The issue is useful here because it catalogs the specific sub-problems the CMR question touches:

- **Crosswalks to Schema.org** — [Wu, Richard, Verhey, Castro, Cecconi & Juty (2023), *An Analysis of Crosswalks from Research Data Schemas to Schema.org*](https://direct.mit.edu/dint/article/5/1/100/113281/An-Analysis-of-Crosswalks-from-Research-Data) (*Data Intelligence* 5(1):100–121; DOI 10.1162/dint_a_00186). Emerging from the Research Data Alliance Metadata Interest Group, it surveys which schemas participating repositories use and maps fourteen research-data schemas onto Schema.org. The verified findings: most *descriptive* metadata is interoperable across the schemas, *rights* metadata is the most inconsistently mapped, and a large gap exists in *structural* metadata and in the controlled vocabularies needed to specify property values. This is direct evidence on the JSON-LD/Schema.org recommendation — feasible for description, materially harder for licensing and structure.
- **Automated metadata annotation with ML** — [Wu, Brandhorst, Marinescu, Moré López, Hlava & Busch (2023), *Automated metadata annotation: What is and is not possible with machine learning*](https://direct.mit.edu/dint/article/5/1/122/113178/Automated-metadata-annotation-What-is-and-is-not) (*Data Intelligence* 5(1):122–138; DOI 10.1162/dint_a_00162). Note the *same lead author* (Mingfang Wu) but a different team and topic. Drawing on three use cases in cultural-heritage and research-data catalogs, it argues that automated annotation is only as good as the training data or domain rules available, and that one must know what a pre-trained model was trained on to gauge its limitations and biases. Critically for a repository like CMR: scholarly and historical content is often *not* available in consumable, homogenized, interoperable formats at the volume ML needs — with science and medicine as partial exceptions. The implication is that AI can help *generate* the AI-ready metadata layer, but not unsupervised, and not uniformly across domains.
- **Continuous metadata** — [Underwood (2023), *Continuous Metadata in Continuous Integration, Stream Processing and Enterprise DataOps*](https://direct.mit.edu/dint/article/5/1/275/114946/Continuous-Metadata-in-Continuous-Integration) (*Data Intelligence* 5(1):275–288). Metadata implementations tend to favor centralized, static records, which is at odds with streaming and cloud-native architectures; metadata is often produced *continuously*, so one-off capture is inadequate — directly relevant to CMR's near-real-time and latency-laden granule records.
- **Transparency** — [Gillman (2023), *Achieving Transparency: A Metadata Perspective*](https://direct.mit.edu/dint/article/5/1/261/114955/Achieving-Transparency-A-Metadata-Perspective) (*Data Intelligence* 5(1):261–274). Aimed at government statistical agencies, it frames transparency as providing *sufficient* documentation and gives three conditions — conforming to a specification, providing quality metadata, and offering a usable interface — a useful checklist for what a trustworthy CMR record must expose.
- **Cognitive metadata modeling** — [Liu, Fu & Liu (2023), *Metadata as a Methodological Commons: From Aboutness Description to Cognitive Modeling*](https://direct.mit.edu/dint/article/5/1/289/114768/Metadata-as-a-Methodological-Commons-From) (*Data Intelligence* 5(1):289–302). Explicitly motivated by the rise of large labeled datasets, ChatGPT, and knowledge graphs, it catalogs the operations (entity definition, ontology modeling, alignment, enrichment) that underpin linked-data and knowledge-graph metadata — the semantic-encoding backbone the report's KG recommendation relies on.

### 2.3 Provenance is the part of "AI-ready metadata" most easily overlooked

[Kale et al. (2023)](https://direct.mit.edu/dint/article/5/1/139/109494/Provenance-documentation-to-enable-explainable-and), in the same issue and notably **NASA-funded**, review provenance as a route to explainable and trustworthy AI. Their framing matters for CMR because the "what happens when metadata is missing / how do I trust this record" thread in the original conversation is fundamentally a provenance problem.

Key points usable in an argument:

- Provenance is best understood as a *subset of metadata* that additionally records the interrelationships and derivation history among objects — the who, what, when, where, and why.
- The W3C **PROV** family ([PROV-DM, PROV-O](https://www.w3.org/TR/prov-o/)) provides a standard, machine-actionable model built on entities, activities, and agents — i.e., an existing, FAIR-compatible way to encode the "DataCenter / Version / processing-history" facts an LLM needs to judge reliability.
- Earth-science-relevant tooling already exists: **Geoweaver** (cloud workflows for earth-science AI) and **MetaClip**, which records climate-product provenance in **JSON-LD** and embeds it with the output. MetaClip is a concrete existence proof that the "flattened, linked, machine-readable summary" idea is already practiced in a NASA-adjacent domain.
- Provenance granularity should be matched to stakeholder need; high-stakes domains warrant heavier documentation. For CMR this argues for *layered* provenance rather than a single fixed verbosity.

### 2.4 Reproducibility reframes metadata as an "analytic stack," and names the core design tension

[Leipzig et al. (2021)](https://www.cell.com/patterns/fulltext/S2666-3899(21)00170-7) review metadata standards for reproducible computational research across an *analytic stack* — input data, tools, notebooks/reports, pipelines, and publications. Their most useful contribution for this report is a clean articulation of the central design choice:

> **Embedded vs. connected metadata.** Metadata can travel *inside* the object it describes, or live *outside* it and be linked.

This is precisely the choice a provider like CMR faces when deciding between a flattened, self-contained "Metadata-as-Text" summary (embedded) and a cross-linked knowledge graph (connected). The reproducibility literature treats this as a genuine trade-off rather than a solved question — which aligns with the RAG findings in §3 that embedding metadata in the chunk improves cohesion but raises re-indexing cost.

### 2.5 The 2026 reframing: from FAIR to "AI-ready"

[Greenberg & An (2026), *The metadata ecosystem and AI: enabling FAIR and AI-ready data* (AI Magazine)](https://onlinelibrary.wiley.com/doi/full/10.1002/aaai.70060), is the most recent and most directly on-point of the supplied papers. It argues that as AI is embedded across science, high-quality metadata describing **datasets, models, and workflows** supports FAIR, strengthens reproducibility, and enables explicit evaluation of *AI-readiness* by making data and models interpretable, traceable, and structurally consistent.

It also supplies a governance distinction useful for CMR specifically: general-purpose standards (e.g., Dublin Core) act as flexible guidelines, whereas domain-specific standards tend to be *prescriptive* — the paper cites the Crystallographic Information Framework (required unit-cell parameters, symmetry groups, atomic positions) and the **NetCDF Climate and Forecast (CF) conventions** (mandated coordinate variables for time, latitude, longitude, and controlled vocabularies). CMR's UMM is on the prescriptive end of that spectrum, which is an *asset* for AI-readiness (consistency, mandatory fields) even as its verbosity is a liability for naive embedding.

---

## 3. How LLMs actually consume metadata (the empirical layer)

The stewardship literature says *what* good metadata is. The RAG literature measures *how a model behaves* when fed it. The two must be combined, because a record can be perfectly FAIR and still retrieve poorly if it is encoded in a way the model parses badly.

### 3.1 Metadata-as-text genuinely helps retrieval — with a maintenance cost

The closest direct study is [*Utilizing Metadata for Better Retrieval-Augmented Generation* (Yousuf et al., 2026; arXiv:2601.11863)](https://arxiv.org/abs/2601.11863), with the public RAGMATE-10K dataset. Comparing a plain-text baseline against metadata-as-text (prefix/suffix), a dual-encoder unified embedding, late fusion, and query reformulation, it finds prefixing and unified embeddings consistently beat the plain-text baseline. The mechanism is the useful part for an argument: integrating metadata increases intra-document cohesion, reduces inter-document confusion, and widens the separation between relevant and irrelevant chunks. The honest caveat — prefixing forces re-embedding the whole index on any metadata update — is itself an argument for the *connected* (dual-encoder) design over naive *embedded* flattening, echoing Leipzig et al.'s tension.

### 3.2 Format matters, but the winner is model-dependent — beware "flatten to prose"

A cluster of studies shows representation has large, measurable effects:

- *Does Prompt Formatting Have Any Impact on LLM Performance?* ([arXiv:2411.10541](https://arxiv.org/html/2411.10541v1)) — across plain text, Markdown, YAML, and JSON, format materially changes accuracy and there is no universal winner.
- *Input Matters* ([arXiv:2510.21034](https://arxiv.org/pdf/2510.21034)) — on NBA play-by-play with hand-annotated errors, **JSON input cut factual-error rates by roughly two-thirds versus unstructured text**. This cuts *against* naive prose-flattening: structure improved faithfulness.
- *CallNavi* ([arXiv:2501.05255](https://arxiv.org/pdf/2501.05255)) — JSON outperformed YAML for tool input/output.

The recurring explanation is the **training-data-prevalence hypothesis**: models do best with formats they saw most in training, not with theoretically token-efficient ones. The implication for CMR is that "flatten to natural language" is not automatically correct; the right representation must be tested per target model, and structured encodings may *help* faithfulness even as summaries help recall.

### 3.3 Field descriptions are the highest-leverage, best-supported lever

If CMR's search parameters are treated as tool parameters, the function-calling literature is decisive on the value of *describing* fields well:

- [Gorilla (Patil et al., 2023; arXiv:2305.15334)](https://arxiv.org/abs/2305.15334) and [ToolAlpaca (Tang et al., 2023; arXiv:2306.05301)](https://arxiv.org/abs/2306.05301) report that precise parameter descriptions and enum constraints improve parameter-generation accuracy by over 30%.
- [Databricks' function-calling evaluation](https://www.databricks.com/blog/unpacking-function-calling-eval) found that improving how tool relevance is described raised a Llama-3-8B model's relevance detection from roughly 20% to 78%.

This is the empirical backing for FAIR's R1 ("rich attributes") and for the original proposal's "describe fields like a textbook." It is also the cheapest change to ship.

### 3.4 Knowledge-graph / linked-data structure: promising, domain-relevant

- [*Structured Linked Data as a Memory Layer for Agent-Orchestrated Retrieval* (arXiv:2603.10700)](https://arxiv.org/pdf/2603.10700) tests whether Schema.org markup and dereferenceable entity pages improve standard and agentic RAG across four domains — a direct test of FAIR's I1/I3 in an LLM setting.
- [*Towards Intelligent Geospatial Data Discovery* (arXiv:2603.20670)](https://arxiv.org/pdf/2603.20670) builds a knowledge-graph-driven multi-agent framework over geospatial catalogs, STAC, and OGC standards — the closest analogue to a CMR-specific system.
- [*Autonomous GIS* (arXiv:2305.06453)](https://arxiv.org/pdf/2305.06453) names the exact open problem: practical strategies for LLMs to discover, filter, and select high-quality geospatial datasets by resolution, coverage, and accuracy.

### 3.5 Agentic / MCP access already exists for CMR

The "agentic discovery" recommendation is no longer hypothetical. There is an official [`nasa/earthdata-mcp`](https://github.com/nasa/earthdata-mcp) server implementing embedding-based **semantic search** over Earthdata (an enrichment pipeline that validates, embeds, and stores records — i.e., the "AI-optimized" path, not a raw proxy), and a [`podaac/cmr-mcp`](https://github.com/podaac/cmr-mcp) server from a NASA DAAC. These are partial existence proofs that the representation question is live and being engineered now.

> *Note on a prior citation:* a "ReSearch Framework (2025/2026)" arXiv paper referenced in the originating Gemini conversation could not be verified and should not be cited without locating the primary source. The MCP claim, by contrast, checks out.

---

## 4. Synthesis: where the two literatures agree and disagree

**Points of strong convergence**

- *Rich description beats sparse description.* FAIR R1, the metadata-as-data-intelligence issue, and the tool-description experiments all agree, from principle and from measurement respectively. This is the safest, highest-value recommendation.
- *Interoperable, linked encoding is the long-run target.* FAIR I1/I3, the Schema.org crosswalk study, and the linked-data RAG experiment all push toward graph-structured, vocabulary-backed metadata.
- *Provenance is a first-class requirement, not an extra.* FAIR R1.2, Kale et al., and the AI-readiness paper converge: to let a model judge reliability and to make the system trustworthy, provenance must be machine-actionable (PROV-O, JSON-LD as in MetaClip).

**Points of productive tension**

- *Embedded vs. connected metadata* (Leipzig et al.) maps onto *flatten-into-chunk vs. dual-encoder* (RAGMATE). Embedding aids cohesion and recall but is costly to maintain and can bloat tokens; connecting preserves precision and updateability but demands graph infrastructure. Neither literature declares a universal winner.
- *Structure vs. flattening.* The popular "flatten for the AI" advice is only half-right: summaries help recall and embedding cohesion, but structured encodings can *reduce hallucination* and the optimal format is model-specific.

**The combined thesis.** The strongest, least-attackable position is not "flatten everything for LLMs." It is: *a provider like CMR should serve metadata in multiple coordinated representations — a precise canonical UMM record, a flattened natural-language summary tuned for embedding, rich inline field descriptions for agentic tool use, and machine-actionable provenance — because the optimal representation depends on the consumption mode and the model, and should be chosen empirically.* Every cluster of evidence in this report supports that claim, and FAIR's modularity (apply principles incrementally, in any combination) makes it implementable as a staged roadmap rather than a rebuild.

---

## 5. Recommendations for a provider like CMR

1. **Add an `llm_summary` / Metadata-as-Text field per collection and granule.** A dense natural-language sentence capturing dataset, instrument, variables, spatial/temporal coverage, and quality status. Justified by RAGMATE (cohesion, recall) and Greenberg & An (AI-readiness). Keep the canonical UMM record untouched — this is an *added* representation, not a replacement.

2. **Put descriptions and enums inline in the schema.** Every searchable UMM field should carry a one-line semantic description and, where applicable, controlled-vocabulary enums (GCMD keywords). Highest evidence-to-effort ratio (Gorilla/ToolAlpaca; Databricks; FAIR R1).

3. **Serve a lightweight JSON-LD / Schema.org `Dataset` projection.** Backed by FAIR I1/I3 and the Schema.org crosswalk study — while budgeting for the study's finding that rights and structural metadata map poorly and need manual attention.

4. **Expose machine-actionable provenance in PROV-O / JSON-LD.** Version, data center, processing latency, and quality maturity as queryable provenance, following MetaClip's pattern. This directly serves the "is this record trustworthy / why is it missing" need.

5. **Maintain an official semantic-search MCP tool with described tools and enums** (building on `nasa/earthdata-mcp`), so agents are *told* the available functions and parameters rather than guessing URLs.

6. **Standardize a quality/maturity flag** ("beta / provisional / validated") to enable model-side guardrails for "scientific-grade" requests.

7. **Treat the choice of representation as continuously evaluated, not fixed** — because format effects are model-dependent and models change.

---

## 6. Proposed experiments

Ordered cheapest-first; each isolates one claim.

**Experiment 1 — Format ablation on real CMR records (foundational).**
Take 300–500 CMR collection/granule records. Render each as (a) raw UMM JSON, (b) flattened JSON-LD/STAC, (c) dot-notation breadcrumb, (d) natural-language Metadata-as-Text. Build a question set with known answers ("which dataset covers canopy height over the Amazon, Dec 2024?"). Hold embedding model and retriever fixed; vary only format. Measure Recall@k, MRR, and end-to-end answer accuracy. **Run across ≥3 models**, because the format literature predicts a model-dependent winner. (Adapts the RAGMATE protocol to CMR.)

**Experiment 2 — Description-quality ablation.**
Same query workload, three schema conditions: bare field names; names + one-line inline descriptions; descriptions + enums + examples. Measure correct field selection and correct value formatting (valid ISO-8601 temporal ranges, valid bounding boxes). The tool-use literature predicts a large (a)→(c) jump; confirming it on CMR fields is a clean result.

**Experiment 3 — Embedded vs. connected metadata (tests the core tension).**
Compare flattened-into-chunk embedding against a dual-encoder that embeds content and metadata separately, on identical records and queries. Measure retrieval quality *and* operational cost (re-indexing time on a simulated metadata update). Operationalizes Leipzig et al.'s embedded/connected distinction with RAGMATE's encoders.

**Experiment 4 — Sparse-metadata / dark-data stress test.**
Deliberately ablate fields (drop spatial bounds, quality flags, format) and measure graceful degradation and *hallucination-vs.-correct-gap-reporting* rates per representation. Tests the "what happens when metadata is missing" thread; quantifies whether a `summary_statement` and PROV-O provenance reduce fabrication.

**Experiment 5 — Static representation vs. agentic/MCP retrieval.**
Three architectures on the same queries: pure vector RAG over flattened records; hard metadata filter → RAG; agent calling the CMR MCP server as a tool. Measure accuracy, latency, and API-call count. Tests whether agentic discovery beats good static representation or merely adds overhead.

**Suggested shared metrics:** Recall@k, MRR/nDCG for retrieval; exact-match field/value accuracy for tool use; annotated factual-error and hallucination rate (following the *Input Matters* annotation method) for faithfulness; tokens-per-record and re-index latency for cost; and a Pareto plot of accuracy vs. token cost across representations.

---

## 7. Caveats and limitations

- **Recency and verification.** Several core references are very recent (2026 preprints); findings may shift on peer review. One citation circulating in the originating conversation (the "ReSearch Framework") could not be verified and is excluded.
- **Domain transfer.** Much of the format and tool-use evidence comes from non-geospatial domains (regulatory filings, sports play-by-play, generic APIs). The geospatial-specific evidence (Autonomous GIS, the geospatial KG paper) is thinner and more architectural than experimental — which is exactly why Experiments 1–5 are framed on CMR data directly.
- **Model dependence.** Because the best representation varies by model and models change, no single recommended format should be treated as permanent; the evaluation harness matters more than any one result.
- **This is a research synthesis, not data-management or procurement advice.** Implementation choices at a specific repository should involve that repository's stewards and standards bodies.

---

## References

**Supplied papers (data-stewardship strand)**

- Wilkinson, M. D., et al. (2016). The FAIR Guiding Principles for scientific data management and stewardship. *Scientific Data* 3:160018. https://www.nature.com/articles/sdata201618
- Greenberg, J., et al. (2023). Metadata as Data Intelligence (editorial). *Data Intelligence* 5(1):1–5. https://direct.mit.edu/dint/article/5/1/1/115168/Metadata-as-Data-Intelligence — also at https://www.sciengine.com/DI/doi/10.1162/dint_r_00024
- Wu, M., Richard, S. M., Verhey, C., Castro, L. J., Cecconi, B., & Juty, N. (2023). An Analysis of Crosswalks from Research Data Schemas to Schema.org. *Data Intelligence* 5(1):100–121. DOI 10.1162/dint_a_00186. https://direct.mit.edu/dint/article/5/1/100/113281/An-Analysis-of-Crosswalks-from-Research-Data
- Wu, M., Brandhorst, H., Marinescu, M.-C., Moré López, J., Hlava, M., & Busch, J. (2023). Automated metadata annotation: What is and is not possible with machine learning. *Data Intelligence* 5(1):122–138. DOI 10.1162/dint_a_00162. https://direct.mit.edu/dint/article/5/1/122/113178/Automated-metadata-annotation-What-is-and-is-not
- Leipzig, J., Nüst, D., Hoyt, C. T., Ram, K., & Greenberg, J. (2021). The role of metadata in reproducible computational research. *Patterns* 2(9):100322. https://www.cell.com/patterns/fulltext/S2666-3899(21)00170-7 (preprint: https://arxiv.org/abs/2006.08589)
- Kale, A., Nguyen, T., Harris, F. C. Jr., et al. (2023). Provenance documentation to enable explainable and trustworthy AI: A literature review. *Data Intelligence* 5(1):139–162. https://direct.mit.edu/dint/article/5/1/139/109494/Provenance-documentation-to-enable-explainable-and
- Wang, Y. X., Luo, L. Q., & Li, G. J. (2023). Research on Intelligent Organization and Application of Multi-source Heterogeneous Knowledge Resources for Energy Internet. *Data Intelligence* 5(1):75–99. https://direct.mit.edu/dint/article/5/1/75/113284/Research-on-Intelligent-Organization-and
- Underwood, M. (2023). Continuous Metadata in Continuous Integration, Stream Processing and Enterprise DataOps. *Data Intelligence* 5(1):275–288. DOI 10.1162/dint_a_00193. https://direct.mit.edu/dint/article/5/1/275/114946/Continuous-Metadata-in-Continuous-Integration
- Gillman, D. (2023). Achieving Transparency: A Metadata Perspective. *Data Intelligence* 5(1):261–274. DOI 10.1162/dint_a_00188. https://direct.mit.edu/dint/article/5/1/261/114955/Achieving-Transparency-A-Metadata-Perspective
- Liu, W., Fu, Y., & Liu, Q. (2023). Metadata as a Methodological Commons: From Aboutness Description to Cognitive Modeling. *Data Intelligence* 5(1):289–302. DOI 10.1162/dint_a_00189. https://direct.mit.edu/dint/article/5/1/289/114768/Metadata-as-a-Methodological-Commons-From
- Greenberg, J., & An, Y. (2026). The metadata ecosystem and AI: Enabling FAIR and AI-ready data. *AI Magazine* 47:e70060. https://onlinelibrary.wiley.com/doi/full/10.1002/aaai.70060

**RAG / LLM-consumption strand (from prior research)**

- Yousuf, R. B., et al. (2026). Utilizing Metadata for Better Retrieval-Augmented Generation. arXiv:2601.11863. https://arxiv.org/abs/2601.11863 (code/dataset: https://github.com/raquibvt/RAGMate)
- Does Prompt Formatting Have Any Impact on LLM Performance? arXiv:2411.10541. https://arxiv.org/html/2411.10541v1
- Sundararajan, B., Sripada, S., & Reiter, E. Input Matters: Evaluating Input Structure's Impact on LLM Summaries of Sports Play-by-Play. arXiv:2510.21034. https://arxiv.org/pdf/2510.21034
- CallNavi: A Challenge and Empirical Study on LLM Function Calling and Routing. arXiv:2501.05255. https://arxiv.org/pdf/2501.05255
- Volpini, A., Raad, E., Gamba, B., & Riccitelli, D. Structured Linked Data as a Memory Layer for Agent-Orchestrated Retrieval. arXiv:2603.10700. https://arxiv.org/pdf/2603.10700
- Towards Intelligent Geospatial Data Discovery: a knowledge graph-driven multi-agent framework powered by large language models. arXiv:2603.20670. https://arxiv.org/pdf/2603.20670
- Autonomous GIS: the next-generation AI-powered GIS. arXiv:2305.06453. https://arxiv.org/pdf/2305.06453
- Patil, S. G., Zhang, T., Wang, X., & Gonzalez, J. E. (2023). Gorilla: Large Language Model Connected with Massive APIs. arXiv:2305.15334. https://arxiv.org/abs/2305.15334
- Tang, Q., Deng, Z., Lin, H., Han, X., Liang, Q., Cao, B., & Sun, L. (2023). ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases. arXiv:2306.05301. https://arxiv.org/abs/2306.05301
- Databricks. Beyond the Leaderboard: Unpacking Function Calling Evaluation. https://www.databricks.com/blog/unpacking-function-calling-eval
- W3C. PROV-O: The PROV Ontology (W3C Recommendation). https://www.w3.org/TR/prov-o/

**Infrastructure / existence proofs**

- NASA Earthdata MCP server (semantic search via embeddings): https://github.com/nasa/earthdata-mcp
- PO.DAAC CMR MCP server: https://github.com/podaac/cmr-mcp
- NASA Common Metadata Repository overview: https://www.earthdata.nasa.gov/about/esdis/eosdis/cmr

*Prepared as a research synthesis. Not legal, financial, or data-management advice.*