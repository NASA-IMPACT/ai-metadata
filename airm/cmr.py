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


def _write_cache(cid: str, item: dict) -> None:
    """Persist one record to the disk cache atomically (tmp file + rename).

    A plain ``write_text`` leaves a truncated file if the process dies mid-write,
    which then crashes the next :func:`load_cached`. Writing to a temp file and
    renaming makes the publish atomic so readers never see a partial record.
    """
    path = _cache_path(cid)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(item))
    tmp.replace(path)


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
            _write_cache(concept_id(item), item)
    return items


def fetch_by_concept_ids(cids: list[str], *, use_cache: bool = True) -> list[dict]:
    """Fetch collection UMM-JSON items by exact concept-id (cache-first).

    Used to pull ground-truth target collections (e.g. from an external query
    benchmark) into the corpus so retrieval has something to find. Cached
    records are served from disk; only the misses hit the network, batched into
    a single ``concept_id``-repeated CMR query.
    """
    found: dict[str, dict] = {}
    misses: list[str] = []
    for cid in dict.fromkeys(cids):  # dedup, preserve order
        cached = load_cached(cid) if use_cache else None
        if cached is not None:
            found[cid] = cached
        else:
            misses.append(cid)

    if misses:
        url = f"{CMR_BASE}/collections.umm_json"
        params = [("page_size", str(min(len(misses), 2000)))]
        params += [("concept_id", cid) for cid in misses]
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            items = resp.json().get("items", [])
        if use_cache:
            CMR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for item in items:
            cid = concept_id(item)
            found[cid] = item
            if use_cache:
                _write_cache(cid, item)

    return [found[cid] for cid in dict.fromkeys(cids) if cid in found]


def extend_corpus(cids: list[str], *, path: Path = CORPUS_PATH) -> tuple[int, list[str]]:
    """Append any of ``cids`` missing from the corpus JSONL, fetching from CMR.

    Returns ``(n_added, still_missing)`` — concept-ids CMR could not return are
    reported, never silently dropped (see CLAUDE.md coverage convention).
    """
    existing = {concept_id(r) for r in load_corpus(path)} if path.exists() else set()
    wanted = [c for c in dict.fromkeys(cids) if c not in existing]
    if not wanted:
        return 0, []

    fetched = fetch_by_concept_ids(wanted)
    got = {concept_id(r) for r in fetched}
    still_missing = [c for c in wanted if c not in got]

    with path.open("a") as fh:
        for rec in fetched:
            fh.write(json.dumps(rec) + "\n")
    return len(fetched), still_missing


def get_granules(collection_concept_id: str, count: int = 5) -> list[dict]:
    """Return up to ``count`` granule UMM-JSON items for a collection."""
    params = {"collection_concept_id": collection_concept_id, "page_size": count}
    url = f"{CMR_BASE}/granules.umm_json"
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json().get("items", [])


def load_cached(cid: str) -> dict | None:
    """Load a single cached record by concept-id, or None if absent/corrupt."""
    path = _cache_path(cid)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        # truncated/corrupt cache file — treat as a miss so the caller re-fetches
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
