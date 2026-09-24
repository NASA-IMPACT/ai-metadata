#!/usr/bin/env bash
# Build the faceted ChromaDB indexes with ModernBERT-large.
#
# Two databases, identical in schema to the gte-modernbert-base ones they sit
# beside -- same six `cmr_<fmt>` collections, same `{concept_id, topic, format}`
# metadata, same cosine space, same one-document-per-record chunking. The only
# thing that differs is the encoder, and therefore the vectors (1024-d rather
# than 768-d).
#
#   data/chroma/faceted_ModernBert       500 records, the study's stratified corpus
#   data/chroma/faceted_full_ModernBert  2,584 records, the full parity-gated cache
#
# Both are written to *new* paths. A collection built by one encoder cannot be
# queried by another, so overwriting the existing databases would strand every
# result that cites them.
#
# What to know before running this
# --------------------------------
# `answerdotai/ModernBERT-large` is a base masked-LM, not a retrieval model. It
# ships no trained pooling head, so sentence-transformers attaches mean pooling
# over raw MLM hidden states. Expect the retrieval scores to be *worse* than
# gte-modernbert-base (which is ModernBERT-base fine-tuned for search) -- that
# gap is the measurement, not a bug. The smoke test bears this out: every
# cosine distance lands in a narrow 0.35-0.41 band, the anisotropy that
# untuned MLM embeddings are known for.
#
# The window is 8,192, the same as gte-modernbert-base, and the tokenizer is the
# same ModernBERT tokenizer -- so the parity and truncation gates behave
# identically and nothing in the faceted corpus (max 6,325 tokens) is clipped.
#
# Usage
# -----
#   bash scripts/build_modernbert_index.sh              # both indexes
#   bash scripts/build_modernbert_index.sh --smoke      # 8 records, ~1 min
#   EMBED_BATCH=16 bash scripts/build_modernbert_index.sh   # CUDA with headroom
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${EMBED_MODEL:-answerdotai/ModernBERT-large}"

# 4 is tuned for MPS at an 8k window, where attention is quadratic in a 6,000+
# token document and 32 asks for a >10 GiB allocation. A CUDA box with real
# VRAM should raise it -- the build is entirely forward passes, so throughput
# scales with batch size until memory runs out.
BATCH="${EMBED_BATCH:-4}"

SUBSET_DB="${SUBSET_DB:-data/chroma/faceted_ModernBert}"
FULL_DB="${FULL_DB:-data/chroma/faceted_full_ModernBert}"

LIMIT=()
if [[ "${1:-}" == "--smoke" ]]; then
  LIMIT=(--limit 8)
  SUBSET_DB="${SUBSET_DB}_smoke"
  FULL_DB="${FULL_DB}_smoke"
  echo "SMOKE: 8 records per index, into *_smoke paths."
fi

echo "model : $MODEL"
echo "batch : $BATCH"
echo

# --- 500-record stratified corpus ----------------------------------------- #
echo "=== subset (corpus) -> $SUBSET_DB ==="
uv run python -m airm.index --build \
  --payload faceted \
  --records corpus \
  --embed-model "$MODEL" \
  --embed-batch "$BATCH" \
  --path "$SUBSET_DB" \
  "${LIMIT[@]}"

# --- full parity-gated cache ---------------------------------------------- #
# `--records all` re-runs the parity and window gates over all 2,590 cached
# records and writes selection_report.json alongside the index, so which six
# records were dropped is on disk rather than implied.
echo
echo "=== full (all cached) -> $FULL_DB ==="
uv run python -m airm.index --build \
  --payload faceted \
  --records all \
  --embed-model "$MODEL" \
  --embed-batch "$BATCH" \
  --path "$FULL_DB" \
  "${LIMIT[@]}"

echo
echo "=== sanity probe ==="
uv run python -m airm.index --probe "sea surface temperature from MODIS" \
  --payload faceted --embed-model "$MODEL" --path "$FULL_DB"

cat <<EOF

Done. Both index_report.json files record the encoder, so a later reader can
tell these apart from the gte-modernbert-base databases without guessing.

To score the full index against the multi-target query set -- note that
--embed-model must repeat here, or queries get encoded by the wrong model:

  uv run python scripts/multi_retrieval_eval.py \\
    --db $FULL_DB \\
    --embed-model $MODEL \\
    --run-id MULTI500_FULL2584_MODERNBERT
EOF
