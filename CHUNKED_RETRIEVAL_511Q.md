# Chunked retrieval by format, at 511 queries

One encoder, one chunking scheme, one record set, one query set. The only variable
is whether the record is the 32-field **faceted** projection or the raw **unfaceted**
UMM document.

No LLM is involved in the measurement. It is exact, free and repeatable.

- Script: [`scripts/unfaceted_chunked_eval.py`](scripts/unfaceted_chunked_eval.py)
- Queries: `data/queries_full.jsonl` — 511 (11 SME, 500 synthetic)
- Indexes: `data/chroma/faceted_chunked/` · `data/chroma/unfaceted_chunked/`
- Runs: `runs/{faceted,unfaceted}_full_{max,mean}/`

---

## 1. Held constant

| | |
|---|---|
| Encoder | `BAAI/bge-large-en-v1.5`, 1024-dim, **512-token window** |
| Chunking | 510-token windows, 64-token overlap, one vector per chunk |
| Pooling | chunk → record by `max` and by `mean`, both reported |
| Records | 2,584 × 6 formats (the same set the faceted full index uses) |
| Queries | 511 — 11 SME, 500 synthetic |
| Oversample | top-`k`×12 chunks fetched before pooling |

Ground truth: **521 unique expected concept-ids, all 521 present** in both indexes.
No query is unanswerable, so every Recall@k below is measured rather than partly
undefined.

---

## 2. Token cost per format

Mean tokens per document, measured with bge's own wordpiece tokenizer over all 2,590
cached renderings.

| Format | Faceted mean | median | p95 | over 512 | Unfaceted mean | median | p95 | over 512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| JSON-LD | 1,465 | 1,333 | 2,705 | 98.3% | **7,012** | 6,195 | 12,993 | 100% |
| JSON | 1,087 | 991 | 2,095 | 85.6% | 2,738 | 2,484 | 4,880 | 100% |
| CSV | 1,024 | 905 | 2,057 | 80.7% | 3,203 | 2,897 | 5,840 | 100% |
| TOON | 966 | 872 | 1,917 | 79.4% | 1,947 | 1,774 | 3,527 | 100% |
| YAML | 934 | 840 | 1,868 | 77.8% | 1,878 | 1,702 | 3,475 | 100% |
| Metadata-as-Text | **786** | 717 | 1,526 | 73.9% | **1,462** | 1,328 | 2,768 | 99.5% |

**Every format's mean exceeds the 512-token window under both payloads** — even the
cheapest. That is the justification for chunking here rather than embedding whole
documents: with this encoder there is no payload and no format that fits.

### What that costs in chunks

| Format | Faceted chunks | /record | Unfaceted chunks | /record | Ratio |
|---|---:|---:|---:|---:|---:|
| JSON-LD | 9,261 | 3.58 | 40,965 | 15.85 | 4.4× |
| JSON | 7,035 | 2.72 | 16,560 | 6.41 | 2.4× |
| CSV | 6,673 | 2.58 | 19,273 | 7.46 | 2.9× |
| TOON | 6,389 | 2.47 | 12,088 | 4.68 | 1.9× |
| YAML | 6,222 | 2.41 | 11,666 | 4.51 | 1.9× |
| Metadata-as-Text | 5,358 | 2.07 | 9,335 | 3.61 | 1.7× |
| **Total** | **40,938** | 81 min | **109,887** | 98 min | 2.7× |

Chunk count is not neutral: under max-pooling more chunks means more independent
chances to match a query. The spread is 1.7× under the faceted payload and 4.4×
under the unfaceted one, so the faceted index is the cleaner of the two for reading
a format ranking. In both, **the format with the most chunks finishes last**, so the
bias never pays out.

---

## 3. Results

### 3.1 Faceted payload

**max-pooling**

| Format | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Metadata-as-Text | **0.641** | 0.830 | **0.884** | **0.730** | **0.763** |
| YAML | 0.617 | 0.818 | 0.874 | 0.708 | 0.744 |
| TOON | 0.610 | 0.829 | 0.873 | 0.709 | 0.745 |
| CSV | 0.624 | 0.830 | 0.872 | 0.713 | 0.748 |
| JSON | 0.580 | 0.793 | 0.854 | 0.680 | 0.717 |
| JSON-LD | 0.504 | 0.741 | 0.813 | 0.615 | 0.656 |

**mean-pooling**

| Format | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Metadata-as-Text | **0.618** | **0.797** | **0.862** | **0.705** | **0.738** |
| CSV | 0.570 | 0.781 | 0.851 | 0.668 | 0.706 |
| YAML | 0.553 | 0.791 | 0.844 | 0.656 | 0.697 |
| TOON | 0.537 | 0.757 | 0.842 | 0.644 | 0.685 |
| JSON | 0.517 | 0.739 | 0.818 | 0.624 | 0.664 |
| JSON-LD | 0.425 | 0.682 | 0.771 | 0.544 | 0.590 |

### 3.2 Unfaceted payload

**max-pooling**

| Format | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Metadata-as-Text | **0.591** | **0.797** | **0.849** | **0.683** | **0.719** |
| JSON | 0.554 | 0.752 | 0.831 | 0.650 | 0.686 |
| YAML | 0.501 | 0.728 | 0.802 | 0.614 | 0.653 |
| TOON | 0.496 | 0.714 | 0.787 | 0.601 | 0.639 |
| JSON-LD | 0.491 | 0.715 | 0.771 | 0.591 | 0.630 |
| CSV | 0.410 | 0.638 | 0.728 | 0.516 | 0.561 |

**mean-pooling**

| Format | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Metadata-as-Text | **0.553** | **0.781** | **0.843** | **0.656** | **0.697** |
| JSON | 0.505 | 0.751 | 0.808 | 0.616 | 0.656 |
| YAML | 0.476 | 0.705 | 0.776 | 0.581 | 0.620 |
| TOON | 0.471 | 0.677 | 0.765 | 0.572 | 0.612 |
| JSON-LD | 0.424 | 0.675 | 0.750 | 0.536 | 0.581 |
| CSV | 0.363 | 0.598 | 0.680 | 0.475 | 0.516 |

### 3.3 Rankings

```
faceted   max   Prose > YAML > TOON > CSV > JSON > JSON-LD
faceted   mean  Prose > CSV > YAML > TOON > JSON > JSON-LD
unfaceted max   Prose > JSON > YAML > TOON > JSON-LD > CSV
unfaceted mean  Prose > JSON > YAML > TOON > JSON-LD > CSV
```

---

## 4. What replicates

**Metadata-as-Text is first in all four conditions**, on every one of the five
metrics, under both payloads and both pooling rules. On the 131-query set it led
three of four (JSON took faceted max-pooling); quadrupling the synthetic queries
removed that exception. This is the clearest result the chunked indexes have
produced.

**JSON-LD is last or near-last in all four.** It is also the most expensive format
by a wide margin — 1,465 tokens faceted, 7,012 unfaceted, 4.4× the chunks of prose.
It costs the most and retrieves the worst under this encoder.

**Prose is both the cheapest and the most accurate.** Lowest mean token count under
both payloads, fewest chunks per record, first on every metric. Under this encoder
there is no cost/quality trade-off to make.

**The faceted payload beats the unfaceted one for every format**, by 0.01 (prose) to
0.14 (CSV) on Recall@10. See the caveat in §6 — the query generator's input was a
field summary, which structurally resembles the faceted projection.

---

## 5. What changed against the 131-query set

Recall@10 rose across the board, faceted max-pooling:

| Format | 131 queries | 511 queries | Δ |
|---|---:|---:|---:|
| CSV | 0.594 | 0.872 | **+0.278** |
| JSON | 0.590 | 0.854 | +0.264 |
| YAML | 0.614 | 0.874 | +0.260 |
| TOON | 0.658 | 0.873 | +0.215 |
| JSON-LD | 0.605 | 0.813 | +0.208 |
| Metadata-as-Text | 0.686 | 0.884 | +0.198 |

Same records, same encoder, same chunks — so the level shift belongs to the query
set, not the indexes. **Absolute scores from the two query sets are not comparable;
only rankings are.**

**CSV is the format that moved, and the earlier reading of it was wrong.** A previous
report called CSV "the stable negative" on the strength of the 131-query runs. At
511 queries it rises to 4th under faceted max-pooling (0.872, within 0.012 of
second) while staying last under the unfaceted payload (0.728). CSV is
payload-dependent, not uniformly weak. That earlier finding is retracted.

---

## 6. Caveats

**The query set is easier, and its provenance is not the old one.** The 500 synthetic
queries were generated by `gpt-5.4-nano` (`logs/llm/20260810T225555Z/query_gen.jsonl`)
from a **field summary** — title, science topic, keywords, measures, spatial and
temporal bounds, abstract — with an explicit instruction never to name the dataset,
short name, DOI, concept id, version or data centre. This is fairer than the previous
set, which was generated from the faceted rendering: no single format is privileged,
because all six carry those facts.

**But the generator's input resembles the faceted projection.** It is a curated field
subset, structurally closer to the 32-field faceted payload than to the raw UMM
record. That plausibly contributes to faceted scoring above unfaceted in every row of
§3, and is the reason the payload comparison in §4 is stated as an observation rather
than a finding. Within-payload format rankings are unaffected.

**Recall@k on the synthetic slice is a hit-rate.** 502 of 511 queries expect exactly
one record; 9 expect 2–9. The SME slice expects 1–9 and is genuine set recall. They
are reported separately in every summary for that reason.

**The SME slice is underpowered and should not be ranked.** n = 11, and it disagrees
with itself across pooling rules — JSON leads faceted max at 0.388 while YAML leads
faceted mean at 0.406, and prose places 4th or 5th in three of the four conditions.
Nothing in §4 rests on it.

**Chunk boundaries fall differently by format.** JSON cut mid-object is syntactically
meaningless; prose cut mid-paragraph still reads. This is not fixable by tuning and
may be part of why prose leads. It is measured only indirectly, through the results.

**Stored chunk text is detokenized, not the original bytes.** `chunk()` cuts on token
ids and decodes back, which lowercases and spaces out punctuation. The round trip is
identity at the token level (verified: 1,203 ids in, 1,203 out, identical), so the
vectors are exactly what the raw bytes would produce — but the documents in Chroma
will not byte-match `data/format_cache*/`.

---

## 7. Artefacts

```
runs/faceted_full_max/     runs/faceted_full_mean/
runs/unfaceted_full_max/   runs/unfaceted_full_mean/
  chunked_rows.csv        one row per (query × format), with the ranked ids
  chunked_summary.json    payload, db, queries_path, chunk counts, all cells
  chunked_retrieval.png   Recall@10 and nDCG@10 by format

runs/faceted_full_pooling.png     max vs mean, faceted
runs/unfaceted_full_pooling.png   max vs mean, unfaceted

data/chroma/faceted_chunked/      40,938 chunks · 81 min
data/chroma/unfaceted_chunked/   109,887 chunks · 98 min
  chunked_index_report.json       payload, db, per-format chunk stats, GT coverage
```

Every row in the CSV carries the top-10 concept-ids that produced its metrics, so any
cell can be traced back to the ranking behind it.

## 8. Reproducing

```bash
# build either index (~80-100 min each)
uv run python scripts/unfaceted_chunked_eval.py build --payload faceted
uv run python scripts/unfaceted_chunked_eval.py build --payload unfaceted

# score, both pooling rules
for p in faceted unfaceted; do for pool in max mean; do
  uv run python scripts/unfaceted_chunked_eval.py eval \
    --payload $p --pool $pool --queries data/queries_full.jsonl \
    --run-id ${p}_full_${pool}
done; done

# redraw from summaries on disk, no retrieval
uv run python scripts/unfaceted_chunked_eval.py plot \
  faceted_full_max faceted_full_mean --out runs/faceted_full_pooling.png
```

## 9. Open items

- **Encoder and chunking are still confounded.** Prose finishes last in the
  whole-document faceted index built with `gte-modernbert` and first here. Payload is
  eliminated — this run holds it fixed and prose still leads. Separating the
  remaining two requires a faceted + `gte-modernbert` + 510-token-chunk index, which
  changes only the retrieval unit against `runs/FULL2584/`.
- **No unchunked unfaceted index exists.** It needs an encoder with a ≥32k window:
  the worst unfaceted document is 74,528 tokens, and 32k drops 13 records of 2,590
  under the all-formats rule while 128k drops none. `Qwen3-Embedding-0.6B` (32k,
  1024-dim, open weights) is the candidate that keeps the run local.
- **Experiment 2 remains blocked** on a working `OPENAI_API_KEY`.
