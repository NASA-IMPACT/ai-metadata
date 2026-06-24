"""Fetch and cache NASA CMR metadata records.

Uses the public CMR search REST API (no authentication required), verified
during planning:

    GET https://cmr.earthdata.nasa.gov/search/collections.umm_json?keyword=...

A "record" here is one item from the ``items`` array of a ``*.umm_json``
response: a dict with ``meta`` (CMR-managed fields such as ``concept-id``) and
``umm`` (the Unified Metadata Model payload). The whole item is cached and
carried through the harness so renderers can use both halves.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from . import CMR_CACHE_DIR, CORPUS_PATH

CMR_BASE = "https://cmr.earthdata.nasa.gov/search"
_TIMEOUT = httpx.Timeout(30.0)


def concept_id(record: dict) -> str:
    """Return the stable CMR concept-id for a record (used as cache + GT key)."""
    return record["meta"]["concept-id"]


def _cache_path(cid: str) -> Path:
    return CMR_CACHE_DIR / f"{cid}.json"


def search_collections(
    keyword: str | None = None,
    count: int = 20,
    *,
    extra_params: dict | None = None,
    use_cache: bool = True,
) -> list[dict]:
    """Search CMR collections, returning up to ``count`` UMM-JSON items.

    Each returned item is also written to the per-concept-id disk cache so the
    corpus can be rebuilt offline.
    """
    params: dict = {"page_size": min(count, 2000)}
    if keyword:
        params["keyword"] = keyword
    if extra_params:
        params.update(extra_params)

    url = f"{CMR_BASE}/collections.umm_json"
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        items = resp.json().get("items", [])

    if use_cache:
        CMR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for item in items:
            _cache_path(concept_id(item)).write_text(json.dumps(item))
    return items


def get_granules(collection_concept_id: str, count: int = 5) -> list[dict]:
    """Return up to ``count`` granule UMM-JSON items for a collection."""
    params = {"collection_concept_id": collection_concept_id, "page_size": count}
    url = f"{CMR_BASE}/granules.umm_json"
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json().get("items", [])


def load_cached(cid: str) -> dict | None:
    """Load a single cached record by concept-id, or None if not cached."""
    path = _cache_path(cid)
    if path.exists():
        return json.loads(path.read_text())
    return None


def build_corpus(
    keywords: list[str],
    n: int = 20,
    *,
    out_path: Path = CORPUS_PATH,
) -> list[dict]:
    """Build a deduplicated corpus by unioning collection searches over keywords.

    Writes one JSON record per line to ``out_path`` (JSONL) and returns the
    records. ``n`` is per-keyword; the union is deduplicated by concept-id.
    """
    by_id: dict[str, dict] = {}
    for kw in keywords:
        for item in search_collections(kw, count=n):
            by_id[concept_id(item)] = item

    records = list(by_id.values())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return records


def load_corpus(path: Path = CORPUS_PATH) -> list[dict]:
    """Load a JSONL corpus written by :func:`build_corpus`."""
    if not path.exists():
        raise FileNotFoundError(
            f"No corpus at {path}. Run airm.cmr.build_corpus(...) first."
        )
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


if __name__ == "__main__":  # pragma: no cover - manual corpus build entrypoint
    import argparse

    parser = argparse.ArgumentParser(description="Build a CMR corpus.")
    parser.add_argument("keywords", nargs="+", help="search keywords")
    parser.add_argument("-n", type=int, default=20, help="records per keyword")
    args = parser.parse_args()
    recs = build_corpus(args.keywords, n=args.n)
    print(f"Wrote {len(recs)} unique records to {CORPUS_PATH}")
