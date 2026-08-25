"""AI-ready metadata evaluation harness.

Two experiments over real NASA CMR collection records:

* **Experiment 1** — which metadata format costs the fewest tokens for the same
  canonical record, with content held constant across JSON, CSV, YAML, TOON,
  JSON-LD and natural-language Metadata-as-Text.
* **Experiment 2** — how model choice interacts with format on LLM retrieval,
  scored with DeepEval (correctness, faithfulness, answer relevancy, contextual
  relevancy, contextual recall) over ChromaDB indexes.

See ``EXPERIMENTS.md`` at the repo root for the running tracker.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
