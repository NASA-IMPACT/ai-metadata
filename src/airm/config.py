"""Central configuration: paths, the format list, the model matrix, quotas.

Everything that a reader of the results would need to know in order to reproduce
them lives here rather than being scattered as literals through the runners.

Environment (loaded from ``.env`` if present):

``OPENAI_API_KEY``
    Required for Experiment 2 only. Experiment 1 needs no credentials.
``OLLAMA_HOST``
    Defaults to ``http://localhost:11434``.
``AIRM_OPENAI_MODELS`` / ``AIRM_OLLAMA_MODELS``
    Comma-separated overrides for the system-under-test model list.
``AIRM_JUDGE_MODEL``
    Overrides the fixed DeepEval judge model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# Paths. Everything is anchored to the repo root so runners work from any cwd.
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
CMR_CACHE_DIR = DATA_DIR / "cmr_cache"
FORMAT_CACHE_DIR = DATA_DIR / "format_cache"
CHROMA_DIR = DATA_DIR / "chroma"
LOGS_DIR = REPO_ROOT / "logs"
RUNS_DIR = REPO_ROOT / "runs"
SAMPLE_DATA_DIR = REPO_ROOT / "sample_data"

CORPUS_PATH = DATA_DIR / "corpus.jsonl"
GT_PATH = DATA_DIR / "gt.jsonl"
GT_DROPPED_PATH = DATA_DIR / "gt_dropped.json"
QUERIES_PATH = DATA_DIR / "queries.jsonl"
SME_XLSX_PATH = REPO_ROOT / "sample_sme_queries.xlsx"


def ensure_dirs() -> None:
    """Create every directory the harness writes into."""
    for d in (DATA_DIR, CMR_CACHE_DIR, FORMAT_CACHE_DIR, CHROMA_DIR, LOGS_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def new_run_id() -> str:
    """A sortable, filesystem-safe id for one experiment run."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# --------------------------------------------------------------------------- #
# Formats under test.
# --------------------------------------------------------------------------- #

#: The six representations. Order is fixed so tables and charts stay comparable
#: across runs; ``json`` leads because it is the baseline.
FORMATS: tuple[str, ...] = ("json", "csv", "yaml", "toon", "jsonld", "mat")

#: Every token count and score is also reported relative to this format.
BASELINE_FORMAT = "json"

#: Human-readable labels for report tables.
FORMAT_LABELS: dict[str, str] = {
    "json": "JSON",
    "csv": "CSV",
    "yaml": "YAML",
    "toon": "TOON",
    "jsonld": "JSON-LD",
    "mat": "Metadata-as-Text",
}


# --------------------------------------------------------------------------- #
# Corpus and query set.
# --------------------------------------------------------------------------- #

CORPUS_SIZE = 500

#: The GCMD science-keyword topics used to stratify the corpus, upper-cased.
#: Taken from CMR's own ``include_facets=v2`` listing rather than guessed: these
#: are the 14 topics with a substantial collection count (407 to 20,443). The
#: listing also contains a long tail of malformed values (``-Na-``,
#: ``Atmospheric``, a NOAA department name) which are data-entry artefacts, not
#: domains, and are deliberately excluded -- see ``corpus.topic_of``.
CMR_TOPICS: tuple[str, ...] = (
    "LAND SURFACE",
    "OCEANS",
    "ATMOSPHERE",
    "CRYOSPHERE",
    "BIOSPHERE",
    "BIOLOGICAL CLASSIFICATION",
    "TERRESTRIAL HYDROSPHERE",
    "HUMAN DIMENSIONS",
    "SOLID EARTH",
    "AGRICULTURE",
    "SUN-EARTH INTERACTIONS",
    "CLIMATE INDICATORS",
    "SPECTRAL/ENGINEERING",
    "PALEOCLIMATE",
)

#: Records whose science keywords name no canonical topic.
UNCLASSIFIED_TOPIC = "UNCLASSIFIED"

#: The three domains the study requires to be *equally* represented.
PRIMARY_TOPICS: tuple[str, ...] = ("LAND SURFACE", "OCEANS", "ATMOSPHERE")

#: The remaining topics, which must each still be represented.
SECONDARY_TOPICS: tuple[str, ...] = tuple(t for t in CMR_TOPICS if t not in PRIMARY_TOPICS)

#: Synthetic queries generated on top of the 11 expert-authored SME queries.
SYNTHETIC_QUERY_TOTAL = 120
SYNTHETIC_PER_PRIMARY_TOPIC = 30  # 3 x 30 = 90, the equal-representation requirement
MAX_QUERY_REGEN_ATTEMPTS = 3


def synthetic_quota(total: int = SYNTHETIC_QUERY_TOTAL) -> dict[str, int]:
    """How many synthetic queries to generate per topic.

    Land, Ocean and Atmosphere get an equal 30 each, as the study requires. The
    remaining 30 are spread as evenly as possible over the other topics so every
    domain is represented; with 11 secondary topics that is 2 each plus a
    remainder of 8 handed out one at a time.

    ``total`` scales that shape without changing it: the primary share stays at
    ``SYNTHETIC_PER_PRIMARY_TOPIC / SYNTHETIC_QUERY_TOTAL`` of the set each (a
    quarter, so three quarters go to the equal-representation requirement) and
    the rest is spread evenly as before. A larger query set therefore has the
    *same* topical distribution as the 120, which is what makes the two
    comparable; rebalancing on the way up would confound size with stratification.
    """
    per_primary = round(total * SYNTHETIC_PER_PRIMARY_TOPIC / SYNTHETIC_QUERY_TOTAL)
    quota = {t: per_primary for t in PRIMARY_TOPICS}
    remaining = total - per_primary * len(PRIMARY_TOPICS)
    per, extra = divmod(remaining, len(SECONDARY_TOPICS))
    for i, topic in enumerate(SECONDARY_TOPICS):
        quota[topic] = per + (1 if i < extra else 0)
    return quota

#: Minimum records per topic in the corpus -- a stratification floor, asserted
#: rather than hoped for.
MIN_RECORDS_PER_TOPIC = 3


# --------------------------------------------------------------------------- #
# Retrieval.
# --------------------------------------------------------------------------- #

#: Held fixed across every cell. The embedding model is not an independent
#: variable in either experiment; varying it would confound the format axis.
#:
#: The context window is the binding constraint, not retrieval quality. The
#: obvious default (``BAAI/bge-small-en-v1.5``) has a 512-token window, and the
#: rendered records overflow it *unevenly* -- 444 of 500 JSON-LD documents were
#: clipped against 205 of 500 prose ones. That silently converts a format
#: comparison into a truncation comparison, penalising exactly the verbose
#: formats under test. ``gte-modernbert-base`` takes 8192 tokens, comfortably
#: above the longest rendering in the corpus (6,325), so nothing is clipped.
#: It also loads without ``trust_remote_code``, unlike the other long-context
#: options considered (nomic-embed, jina-v2).
EMBED_MODEL = "Alibaba-NLP/gte-modernbert-base"

#: Asserted at index time: no document may exceed the embedding window, or the
#: format axis is confounded with truncation.
MAX_EMBED_TRUNCATIONS = 0

TOP_K = 10
RECALL_AT = (1, 5, 10)

CHROMA_COLLECTION_PREFIX = "cmr_"


def collection_name(fmt: str) -> str:
    return f"{CHROMA_COLLECTION_PREFIX}{fmt}"


# --------------------------------------------------------------------------- #
# Models.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelSpec:
    """One system-under-test model.

    ``price_in`` / ``price_out`` are USD per million tokens. They are ``None``
    for models with no published price (local Ollama models are free to run but
    not free to compare against cloud dollars, so their cost is reported as
    ``None`` rather than as ``0.0`` -- a zero would silently win every
    cost-per-accuracy comparison).
    """

    provider: str  # "openai" | "ollama"
    model: str
    price_in: float | None = None
    price_out: float | None = None

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"


#: USD per million tokens. Dated, and deliberately not baked into ModelSpec
#: defaults so a stale price is visible rather than assumed. Verify against
#: current provider pricing before quoting dollar figures in a write-up.
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    # model: (input, output)
}
PRICES_AS_OF = "unverified - populate before quoting dollar costs"


def _env_models(var: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(var, "").strip()
    if not raw:
        return default
    return tuple(m.strip() for m in raw.split(",") if m.strip())


#: OpenAI models are *validated against the live /v1/models listing* at startup
#: rather than trusted -- see ``airm.llm.resolve_models``. If a default here is
#: retired, the run fails with the actual available list instead of a 404 in the
#: middle of the matrix.
OPENAI_MODEL_NAMES = _env_models("AIRM_OPENAI_MODELS", ("gpt-5", "gpt-5-mini"))

#: The open-weights arm. ``qwen3.6`` is the current local model; ``gpt-oss:20b``
#: is also present on this machine and can be added via ``AIRM_OLLAMA_MODELS``
#: for a second open-weights family.
OLLAMA_MODEL_NAMES = _env_models("AIRM_OLLAMA_MODELS", ("qwen3.6:latest",))


def model_matrix() -> list[ModelSpec]:
    """The system-under-test models, in report order."""
    specs: list[ModelSpec] = []
    for name in OPENAI_MODEL_NAMES:
        price = PRICES_USD_PER_MTOK.get(name)
        specs.append(
            ModelSpec("openai", name, price[0] if price else None, price[1] if price else None)
        )
    for name in OLLAMA_MODEL_NAMES:
        specs.append(ModelSpec("ollama", name))
    return specs


#: The DeepEval judge. Held fixed across every cell and kept deliberately
#: separate from ``model_matrix()``: a judge that varies with the model under
#: test would confound the very axis being measured.
#:
#: Provider is overridable so the pipeline can be exercised locally
#: (``AIRM_JUDGE_PROVIDER=ollama AIRM_JUDGE_MODEL=qwen3.6:latest``), but the
#: reported study must use one fixed judge for every cell -- and which judge was
#: used is recorded in the run summary, because it is a real caveat on the
#: LLM-graded metrics rather than an implementation detail.
JUDGE_MODEL = os.getenv("AIRM_JUDGE_MODEL", "gpt-5-mini")
JUDGE_PROVIDER = os.getenv("AIRM_JUDGE_PROVIDER", "openai")


def judge_spec() -> "ModelSpec":
    price = PRICES_USD_PER_MTOK.get(JUDGE_MODEL)
    return ModelSpec(
        JUDGE_PROVIDER, JUDGE_MODEL, price[0] if price else None, price[1] if price else None
    )


# --------------------------------------------------------------------------- #
# Tokenisers (Experiment 1).
# --------------------------------------------------------------------------- #

#: Primary count -- the OpenAI-side number.
TIKTOKEN_ENCODING = "o200k_base"

#: Open-weights cross-check. Downloaded from the Hugging Face hub on first use;
#: if unavailable the run reports the tiktoken and character counts and says so
#: rather than silently dropping a column.
HF_TOKENIZER = "Qwen/Qwen3-8B"


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
