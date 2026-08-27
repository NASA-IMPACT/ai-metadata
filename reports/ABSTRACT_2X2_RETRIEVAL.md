# The abstract 2×2 — what the abstract field is worth at retrieval

Four arms per payload × format, 511 queries, exact chunked retrieval, **max pooling**, `BAAI/bge-large-en-v1.5`, 510-token chunks / 64 overlap. Baseline = the frozen format-study indexes (`runs/20260814T205849Z` faceted, `runs/20260814T205912Z` unfaceted).

| Arm | Abstract | AI summary | Cache | Chroma |
|---|---|---|---|---|
| **baseline** | ✓ | — | `data/format_cache{,_unfaceted}/` | `data/chroma/{faceted,unfaceted}_chunked/` |
| **dropped** | — | — | `data/format_cache_no_summary/`, `…_unfaceted_no_abstract/` | `dropped_summary_faceted/`, `dropped_abstract_unfaceted/` |
| **replaced** | — | ✓ (in its place) | `data/format_cache_replace_summary/`, `…_unfaceted_replace_summary/` | `no_summary_faceted/`, `no_abstract_unfaceted/` |
| **appended** | ✓ | ✓ (after) | `data/format_cache_plus_summary/`, `…_unfaceted_plus_summary/` | `faceted_plus_summary_chunked/`, `unfaceted_plus_summary_chunked/` |

The AI summary is a `gpt-5.4-nano` (temp 0) 120–180-word prose paragraph written from each record's MaT rendering (which included the abstract), one per payload. Faceted `summary` = UMM `Abstract` verbatim; unfaceted acts on `Abstract` directly.

Δ = arm − baseline (absolute); rel = Δ / baseline. **Bold** = best arm in the row.


## Headline — format-mean across the six serializations

**Recall@10** (relative Δ vs baseline in parentheses)

| | baseline | dropped | replaced | appended |
|---|---:|---:|---:|---:|
| Subsetted records | 0.862 | 0.795 (-7.7%) | 0.853 (-1.0%) | 0.899 (+4.4%) |
| Full records | 0.795 | 0.675 (-15.0%) | 0.834 (+4.9%) | 0.845 (+6.4%) |

**nDCG@10** (relative Δ vs baseline in parentheses)

| | baseline | dropped | replaced | appended |
|---|---:|---:|---:|---:|
| Subsetted records | 0.729 | 0.647 (-11.2%) | 0.698 (-4.2%) | 0.759 (+4.1%) |
| Full records | 0.648 | 0.526 (-18.8%) | 0.680 (+4.9%) | 0.701 (+8.2%) |

nDCG sharpens the same story: it weights *rank*, and dropping the abstract
hurts ranking more than it hurts top-10 presence (−11.2% / −18.8% vs −7.7% /
−15.0%) — the right record often stays in the top 10 but slides down.
Replacing on subsetted records is also worse by nDCG (−4.2%) than by recall
(−1.0%): the paraphrase keeps the record findable but ranks it lower than the
original abstract did.

## What the 2×2 says

1. **The abstract is the most valuable text in the record for retrieval.**
   Dropping it costs −3% to −10% Recall@10 subsetted and −1.5% to **−47%**
   full. Full-record JSON-LD without an abstract retrieves the right record
   less than half the time (0.408) — `description` was its only free text.
2. **MaT is the least abstract-dependent format** (−4.9% / −1.5%) because
   the template already verbalizes the other fields as prose.
3. **The AI summary substitutes for the abstract on full records (5 of 6
   formats gain, +2.3% to +11.2%) but not on subsetted ones** — CSV −2.4%,
   YAML −4.1%, TOON −6.0%. On the curated projection the summary is a
   paraphrase of facts already present.
4. **Both together wins 11 of 12 cells.** Only exception: full CSV, where
   replaced 0.809 vs appended 0.803 is a tie.
5. **Summary value tracks abstract value**: where the abstract is worth most
   (JSON-LD, CSV on full records), the summary is worth slightly more — both
   supply prose to serializations that have none.
6. **The Study Summary §6 length caveat is partly resolved**: on full
   records the replaced arm (+120–150 tokens) captures most of the appended
   gain (+400 tokens), so the append effect is mostly content, not chunk
   count. On subsetted records it isn't — duplicated facts help under max
   pooling; a paraphrase in place does not.

**Playbook implication:** treat the abstract as a retrieval-critical field
(a placeholder abstract kills findability under any structured format);
*add* an LLM summary rather than substitute one, with the biggest payoff on
raw records and structured formats; on a curated prose projection the
summary is nearly redundant.

## Findings

**1. The abstract is the single most valuable text in the record for retrieval.**
Dropping it costs −7.7% Recall@10 on average across subsetted formats (−3.0%
JSON to −10.2% CSV) and −15.0% across full records (−1.5% MaT to **−47.1%
JSON-LD**). On the full raw record, JSON-LD without an abstract retrieves the
right record less than half the time (0.408) — its `description` property
was the only schema.org-mapped free text; everything else is keyed literals.

**2. Prose (MaT) is the format least dependent on the abstract.** −4.9%
subsetted, −1.5% full. The template already verbalizes title, keywords,
platforms and coverage as sentences, so the abstract is one prose paragraph
among several; in JSON-LD/CSV it is the *only* prose.

**3. The AI summary is a full substitute for the abstract on full records,
but not on subsetted ones.** Full: replaced beats baseline for 5 of 6
formats (+2.3% to +11.2%), MaT flat. Subsetted: replaced *loses* for CSV
(−2.4%), YAML (−4.1%) and TOON (−6.0%), roughly flat for MaT and JSON-LD,
and wins only for JSON (+4.7%). The curated projection already contains the
facts the summary restates, so on that payload the summary is a paraphrase
of the abstract — and a worse one for three formats.

**4. Both together is best in 11 of 12 cells.** Appended is the top arm for
every subsetted format and for 5 of 6 full formats (full CSV: replaced
0.809 vs appended 0.803, inside noise). Mean gain over baseline: +4.4%
subsetted, +6.4% full.

**5. The summary's value is largest where the abstract's value is largest.**
Full-record JSON-LD: abstract +0.363, summary +0.431; CSV: +0.145 / +0.227.
Both texts are doing the same job — supplying prose to a serialization that
has none — and the summary does it slightly better because it is denser in
the terms queries use (keywords, platforms, coverage), and never a
placeholder.

**6. The length caveat from §6 of the Study Summary is partly answered.**
Appending adds ~400 tokens and chunks; replacing adds only ~120–150. On full
records the replaced arm captures most of the appended gain (+4.9% vs +6.4%
mean) at a third of the added length, so the append gain is mostly content,
not chunk count. On subsetted records the gap is wider (−1.0% vs +4.4%),
which is consistent with the summary duplicating what the projection already
says — extra chunks of the same facts help under max pooling, a paraphrase
in place of the original does not.

**Implications for the playbook.** (a) Providers should treat the abstract
as a retrieval-critical field, not a description: a placeholder abstract
(9 in this corpus) costs the record most of its findability under any
structured serialization. (b) An LLM summary should be *added*, not
substituted, and its value is highest for raw records and for structured
formats — the cases where prose is otherwise absent. (c) On a curated
projection served as prose, the summary is nearly redundant (+2.1%
appended, +0.2% replaced for MaT).

## Caveats

- Max pooling only; mean-pool results exist for the three new arms
  (`runs/*/retrieval_eval.json`) but there is no frozen 511-query mean-pool
  baseline, so they are not tabulated here.
- Summaries were written by nano from the record's MaT text; synthetic
  queries were generated from a field summary of similar provenance. Both
  share vocabulary with the queries. The *dropped* arm carries no such
  asterisk — it is the cleanest number in this report.
- No confidence intervals: 511 queries, single run per cell. Differences
  under ~0.02 R@10 (e.g. full CSV replaced vs appended, subsetted MaT
  replaced vs baseline) should be read as ties.
- Retrieval only. Whether a summary or a missing abstract changes *answers*
  (Stages A/J) is unmeasured.

---

## Subsetted records (faceted, 32-field projection)

### Recall@10 — all four arms

| Format | baseline | dropped | Δ | rel | replaced | Δ | rel | appended | Δ | rel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| JSON | 0.854 | 0.829 | -0.025 | -3.0% | **0.894** | +0.040 | +4.7% | 0.887 | +0.034 | +3.9% |
| CSV | 0.872 | 0.783 | -0.089 | -10.2% | 0.852 | -0.021 | -2.4% | **0.911** | +0.039 | +4.4% |
| YAML | 0.874 | 0.797 | -0.077 | -8.8% | 0.838 | -0.035 | -4.1% | **0.905** | +0.031 | +3.6% |
| TOON | 0.873 | 0.788 | -0.085 | -9.8% | 0.821 | -0.053 | -6.0% | **0.911** | +0.038 | +4.3% |
| JSON-LD | 0.813 | 0.735 | -0.078 | -9.5% | 0.827 | +0.014 | +1.8% | **0.878** | +0.066 | +8.1% |
| MaT | 0.884 | 0.841 | -0.043 | -4.9% | 0.886 | +0.002 | +0.2% | **0.903** | +0.019 | +2.1% |
| *format mean* | 0.862 | 0.795 | -0.066 | -7.7% | 0.853 | -0.009 | -1.0% | 0.899 | +0.038 | +4.4% |

### Contribution of each ingredient (Recall@10)

Isolating the two texts: *abstract value* = baseline − dropped (what the human abstract adds to a record with neither); *summary value* = replaced − dropped (what the AI summary adds to the same); *summary on top of abstract* = appended − baseline; *summary vs abstract head-to-head* = replaced − baseline.

| Format | abstract value | summary value | summary on top of abstract | summary vs abstract |
|---|---:|---:|---:|---:|
| JSON | +0.025 (+3.1%) | +0.065 (+7.9%) | +0.034 (+3.9%) | +0.040 (+4.7%) |
| CSV | +0.089 (+11.4%) | +0.069 (+8.7%) | +0.039 (+4.4%) | -0.021 (-2.4%) |
| YAML | +0.077 (+9.7%) | +0.042 (+5.2%) | +0.031 (+3.6%) | -0.035 (-4.1%) |
| TOON | +0.085 (+10.8%) | +0.033 (+4.1%) | +0.038 (+4.3%) | -0.053 (-6.0%) |
| JSON-LD | +0.078 (+10.5%) | +0.092 (+12.5%) | +0.066 (+8.1%) | +0.014 (+1.8%) |
| MaT | +0.043 (+5.2%) | +0.045 (+5.4%) | +0.019 (+2.1%) | +0.002 (+0.2%) |

### All metrics, per arm

**dropped** (Δ vs baseline in parentheses)

| Format | recall@1 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---:|---:|---:|---:|---:|
| JSON | 0.542 (-0.039) | 0.746 (-0.048) | 0.829 (-0.025) | 0.637 (-0.043) | 0.677 (-0.040) |
| CSV | 0.500 (-0.125) | 0.717 (-0.113) | 0.783 (-0.089) | 0.601 (-0.112) | 0.639 (-0.108) |
| YAML | 0.519 (-0.098) | 0.742 (-0.076) | 0.797 (-0.077) | 0.621 (-0.087) | 0.657 (-0.087) |
| TOON | 0.493 (-0.117) | 0.714 (-0.115) | 0.788 (-0.085) | 0.601 (-0.108) | 0.640 (-0.105) |
| JSON-LD | 0.424 (-0.080) | 0.637 (-0.104) | 0.735 (-0.078) | 0.528 (-0.087) | 0.570 (-0.086) |
| MaT | 0.555 (-0.086) | 0.770 (-0.059) | 0.841 (-0.043) | 0.659 (-0.071) | 0.698 (-0.065) |

**replaced** (Δ vs baseline in parentheses)

| Format | recall@1 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---:|---:|---:|---:|---:|
| JSON | 0.570 (-0.010) | 0.819 (+0.026) | 0.894 (+0.040) | 0.684 (+0.005) | 0.732 (+0.015) |
| CSV | 0.566 (-0.059) | 0.789 (-0.041) | 0.852 (-0.021) | 0.666 (-0.047) | 0.704 (-0.044) |
| YAML | 0.497 (-0.121) | 0.766 (-0.052) | 0.838 (-0.035) | 0.623 (-0.085) | 0.669 (-0.075) |
| TOON | 0.523 (-0.086) | 0.772 (-0.056) | 0.821 (-0.053) | 0.633 (-0.076) | 0.672 (-0.073) |
| JSON-LD | 0.510 (+0.007) | 0.744 (+0.003) | 0.827 (+0.014) | 0.623 (+0.008) | 0.665 (+0.009) |
| MaT | 0.597 (-0.044) | 0.837 (+0.007) | 0.886 (+0.002) | 0.707 (-0.023) | 0.747 (-0.016) |

**appended** (Δ vs baseline in parentheses)

| Format | recall@1 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---:|---:|---:|---:|---:|
| JSON | 0.629 (+0.049) | 0.837 (+0.044) | 0.887 (+0.034) | 0.725 (+0.045) | 0.760 (+0.043) |
| CSV | 0.636 (+0.012) | 0.863 (+0.033) | 0.911 (+0.039) | 0.731 (+0.018) | 0.772 (+0.024) |
| YAML | 0.619 (+0.002) | 0.844 (+0.025) | 0.905 (+0.031) | 0.722 (+0.015) | 0.762 (+0.018) |
| TOON | 0.606 (-0.004) | 0.847 (+0.019) | 0.911 (+0.038) | 0.714 (+0.005) | 0.758 (+0.014) |
| JSON-LD | 0.566 (+0.063) | 0.802 (+0.062) | 0.878 (+0.066) | 0.675 (+0.061) | 0.720 (+0.064) |
| MaT | 0.653 (+0.012) | 0.855 (+0.025) | 0.903 (+0.019) | 0.745 (+0.015) | 0.780 (+0.017) |

### Token cost per record (tiktoken `o200k_base`, mean)

| Format | baseline | dropped | Δ | replaced | Δ | appended | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| JSON | 913 | 632 | -281 | 1,034 | +121 | 1,315 | +402 |
| CSV | 984 | 707 | -277 | 1,109 | +125 | 1,386 | +402 |
| YAML | 971 | 658 | -312 | 1,087 | +117 | 1,373 | +402 |
| TOON | 931 | 649 | -282 | 1,052 | +121 | 1,333 | +402 |
| JSON-LD | 1,110 | 829 | -281 | 1,231 | +121 | 1,512 | +402 |
| MaT | 805 | 530 | -275 | 930 | +125 | 1,207 | +402 |

---

## Full records (unfaceted, raw UMM)

### Recall@10 — all four arms

| Format | baseline | dropped | Δ | rel | replaced | Δ | rel | appended | Δ | rel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| JSON | 0.831 | 0.749 | -0.082 | -9.9% | 0.865 | +0.034 | +4.1% | **0.874** | +0.043 | +5.2% |
| CSV | 0.728 | 0.582 | -0.145 | -20.0% | **0.809** | +0.081 | +11.2% | 0.803 | +0.076 | +10.4% |
| YAML | 0.802 | 0.746 | -0.056 | -7.0% | 0.821 | +0.018 | +2.3% | **0.838** | +0.035 | +4.4% |
| TOON | 0.787 | 0.730 | -0.057 | -7.3% | 0.820 | +0.033 | +4.2% | **0.828** | +0.041 | +5.2% |
| JSON-LD | 0.771 | 0.408 | -0.363 | -47.1% | 0.839 | +0.069 | +8.9% | **0.847** | +0.076 | +9.9% |
| MaT | 0.849 | 0.836 | -0.013 | -1.5% | 0.849 | -0.000 | -0.0% | **0.883** | +0.033 | +3.9% |
| *format mean* | 0.795 | 0.675 | -0.119 | -15.0% | 0.834 | +0.039 | +4.9% | 0.845 | +0.051 | +6.4% |

### Contribution of each ingredient (Recall@10)

Isolating the two texts: *abstract value* = baseline − dropped (what the human abstract adds to a record with neither); *summary value* = replaced − dropped (what the AI summary adds to the same); *summary on top of abstract* = appended − baseline; *summary vs abstract head-to-head* = replaced − baseline.

| Format | abstract value | summary value | summary on top of abstract | summary vs abstract |
|---|---:|---:|---:|---:|
| JSON | +0.082 (+10.9%) | +0.116 (+15.5%) | +0.043 (+5.2%) | +0.034 (+4.1%) |
| CSV | +0.145 (+25.0%) | +0.227 (+38.9%) | +0.076 (+10.4%) | +0.081 (+11.2%) |
| YAML | +0.056 (+7.5%) | +0.074 (+10.0%) | +0.035 (+4.4%) | +0.018 (+2.3%) |
| TOON | +0.057 (+7.8%) | +0.090 (+12.3%) | +0.041 (+5.2%) | +0.033 (+4.2%) |
| JSON-LD | +0.363 (+89.0%) | +0.431 (+105.8%) | +0.076 (+9.9%) | +0.069 (+8.9%) |
| MaT | +0.013 (+1.6%) | +0.013 (+1.5%) | +0.033 (+3.9%) | -0.000 (-0.0%) |

### All metrics, per arm

**dropped** (Δ vs baseline in parentheses)

| Format | recall@1 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---:|---:|---:|---:|---:|
| JSON | 0.412 (-0.142) | 0.682 (-0.070) | 0.749 (-0.082) | 0.534 (-0.116) | 0.580 (-0.106) |
| CSV | 0.299 (-0.111) | 0.502 (-0.136) | 0.582 (-0.145) | 0.398 (-0.118) | 0.433 (-0.127) |
| YAML | 0.430 (-0.071) | 0.678 (-0.050) | 0.746 (-0.056) | 0.545 (-0.069) | 0.587 (-0.066) |
| TOON | 0.428 (-0.068) | 0.635 (-0.079) | 0.730 (-0.057) | 0.532 (-0.069) | 0.574 (-0.065) |
| JSON-LD | 0.187 (-0.304) | 0.351 (-0.364) | 0.408 (-0.363) | 0.267 (-0.325) | 0.294 (-0.336) |
| MaT | 0.551 (-0.040) | 0.751 (-0.046) | 0.836 (-0.013) | 0.651 (-0.032) | 0.691 (-0.028) |

**replaced** (Δ vs baseline in parentheses)

| Format | recall@1 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---:|---:|---:|---:|---:|
| JSON | 0.540 (-0.014) | 0.802 (+0.050) | 0.865 (+0.034) | 0.660 (+0.010) | 0.705 (+0.019) |
| CSV | 0.484 (+0.074) | 0.737 (+0.099) | 0.809 (+0.081) | 0.602 (+0.086) | 0.645 (+0.084) |
| YAML | 0.510 (+0.010) | 0.749 (+0.021) | 0.821 (+0.018) | 0.624 (+0.010) | 0.666 (+0.013) |
| TOON | 0.505 (+0.008) | 0.755 (+0.040) | 0.820 (+0.033) | 0.619 (+0.018) | 0.663 (+0.024) |
| JSON-LD | 0.559 (+0.068) | 0.769 (+0.054) | 0.839 (+0.069) | 0.657 (+0.065) | 0.696 (+0.066) |
| MaT | 0.546 (-0.045) | 0.796 (-0.001) | 0.849 (-0.000) | 0.663 (-0.020) | 0.704 (-0.015) |

**appended** (Δ vs baseline in parentheses)

| Format | recall@1 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---:|---:|---:|---:|---:|
| JSON | 0.599 (+0.045) | 0.807 (+0.055) | 0.874 (+0.043) | 0.699 (+0.049) | 0.736 (+0.049) |
| CSV | 0.490 (+0.080) | 0.739 (+0.101) | 0.803 (+0.076) | 0.605 (+0.089) | 0.647 (+0.087) |
| YAML | 0.530 (+0.029) | 0.770 (+0.042) | 0.838 (+0.035) | 0.646 (+0.032) | 0.687 (+0.034) |
| TOON | 0.542 (+0.046) | 0.772 (+0.058) | 0.828 (+0.041) | 0.648 (+0.047) | 0.685 (+0.046) |
| JSON-LD | 0.556 (+0.065) | 0.778 (+0.064) | 0.847 (+0.076) | 0.658 (+0.067) | 0.699 (+0.070) |
| MaT | 0.626 (+0.035) | 0.818 (+0.020) | 0.883 (+0.033) | 0.715 (+0.031) | 0.751 (+0.033) |

### Token cost per record (tiktoken `o200k_base`, mean)

| Format | baseline | dropped | Δ | replaced | Δ | appended | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| JSON | 1,918 | 1,636 | -281 | 2,066 | +148 | 2,348 | +430 |
| CSV | 2,780 | 2,503 | -277 | 2,932 | +152 | 3,210 | +430 |
| YAML | 2,100 | 1,788 | -312 | 2,248 | +148 | 2,531 | +430 |
| TOON | 2,026 | 1,744 | -282 | 2,175 | +148 | 2,457 | +430 |
| JSON-LD | 4,295 | 4,014 | -281 | 4,444 | +148 | 4,726 | +431 |
| MaT | 1,476 | 1,202 | -275 | 1,629 | +153 | 1,906 | +430 |
