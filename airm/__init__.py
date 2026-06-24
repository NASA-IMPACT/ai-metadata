"""airm — AI-Ready Metadata experiment harness.

Shared infrastructure for the five experiments proposed in ``Report.md``
("Making Scientific Metadata AI-Ready"). Each experiment tests how the
*representation* of NASA CMR metadata affects how well an LLM can retrieve,
select, and reason over it.

Modules
-------
cmr             Fetch + cache CMR collection/granule records (public API).
representations Four record renderers (raw UMM, JSON-LD/STAC, dot-breadcrumb, NL text).
schema          Field-description conditions for the description-quality experiment.
embeddings      Local sentence-transformers embedder + cosine vector store.
llm             Unified cross-provider client (Anthropic / OpenAI / Ollama).
queries         Query workload + ground-truth relevance.
metrics         Recall@k, MRR, nDCG, field/value accuracy, cost, Pareto helpers.
run             Experiment runner: config, seeding, JSONL logging.
improve         Validate + improve loop (robustness CIs + agentic auto-tune).
"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
CMR_CACHE_DIR = DATA_DIR / "cmr_cache"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
QUERIES_PATH = DATA_DIR / "queries.yaml"

__all__ = [
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "DATA_DIR",
    "CMR_CACHE_DIR",
    "CORPUS_PATH",
    "QUERIES_PATH",
]
