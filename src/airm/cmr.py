"""Fetch and cache NASA CMR collection metadata.

Uses the public CMR search REST API (no authentication required):

    GET https://cmr.earthdata.nasa.gov/search/collections.umm_json?keyword=...

A "record" here is one item from the ``items`` array of a ``*.umm_json``
response: a dict with ``meta`` (CMR-managed fields such as ``concept-id``) and
``umm`` (the Unified Metadata Model payload). The whole item is cached and
carried through the harness so renderers can use both halves.

Ported from the implementation deleted in ``fe00bb6``
(``git show fe00bb6^:airm/cmr.py``), with three additions this study needs:
topic-stratified search, chunked concept-id batches, and retries.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Iterator

import httpx

from .config import CMR_CACHE_DIR, CORPUS_PATH

CMR_BASE = "https://cmr.earthdata.nasa.gov/search"

_TIMEOUT = httpx.Timeout(60.0)

#: CMR's own hard cap on ``page_size``.
_MAX_PAGE_SIZE = 2000

#: Concept-ids are sent as repeated query parameters, so a large batch can blow
#: past the server's URL length limit. 50 keeps requests comfortably short while
#: still cutting round-trips by an order of magnitude.
_ID_CHUNK = 50

_RETRIES = 3
_BACKOFF_SECONDS = 2.0


class CMRError(RuntimeError):
    """Raised when CMR cannot be reached after retries."""


# --------------------------------------------------------------------------- #
# Record helpers.
# --------------------------------------------------------------------------- #


def concept_id(record: dict) -> str:
    """Return the stable CMR concept-id for a record (cache key and GT key)."""
    return record["meta"]["concept-id"]


def entry_title(record: dict) -> str:
    return record.get("umm", {}).get("EntryTitle", "")


# --------------------------------------------------------------------------- #
# Disk cache.
# --------------------------------------------------------------------------- #


def _cache_path(cid: str) -> Path:
    return CMR_CACHE_DIR / f"{cid}.json"


def _write_cache(cid: str, item: dict) -> None:
    """Persist one record atomically (temp file + rename).

    A plain ``write_text`` leaves a truncated file if the process dies mid-write,
    which then crashes the next :func:`load_cached`. Writing to a temp file and
    renaming makes the publish atomic so readers never see a partial record.
    """
    CMR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cid)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(item))
    tmp.replace(path)


def load_cached(cid: str) -> dict | None:
    """Load a cached record by concept-id, or ``None`` if absent or corrupt."""
    path = _cache_path(cid)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        # Truncated/corrupt cache file: treat as a miss so the caller re-fetches.
        return None


def is_cached(cid: str) -> bool:
    return _cache_path(cid).exists()


# --------------------------------------------------------------------------- #
# HTTP.
# --------------------------------------------------------------------------- #


def _get(url: str, params) -> dict:
    """GET with retries on transient network/5xx failures.

    A 4xx is a request we got wrong and is raised immediately; retrying it would
    only waste time and hide the bug.
    """
    last: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                resp = client.get(url, params=params)
            if resp.status_code >= 500:
                last = CMRError(f"CMR {resp.status_code}: {resp.text[:200]}")
            else:
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError:
            raise
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            last = exc
        if attempt < _RETRIES - 1:
            time.sleep(_BACKOFF_SECONDS * (attempt + 1))
    raise CMRError(f"CMR request failed after {_RETRIES} attempts: {last}")


# --------------------------------------------------------------------------- #
# Search.
# --------------------------------------------------------------------------- #


def search_collections(
    keyword: str | None = None,
    count: int = 20,
    *,
    extra_params: dict | None = None,
    use_cache: bool = True,
) -> list[dict]:
    """Search CMR collections, returning up to ``count`` UMM-JSON items.

    Each returned item is also written to the disk cache so the corpus can be
    rebuilt offline.
    """
    params: dict = {"page_size": min(count, _MAX_PAGE_SIZE)}
    if keyword:
        params["keyword"] = keyword
    if extra_params:
        params.update(extra_params)

    items = _get(f"{CMR_BASE}/collections.umm_json", params).get("items", [])
    if use_cache:
        for item in items:
            _write_cache(concept_id(item), item)
    return items


def search_by_topic(topic: str, count: int = 200, *, use_cache: bool = True) -> list[dict]:
    """Return up to ``count`` collections whose primary science keyword is ``topic``.

    This is the stratification primitive for the corpus: CMR indexes the GCMD
    science-keyword hierarchy, so filtering on ``science_keywords[0][topic]``
    gives a clean per-domain pool to sample from. Verified live against all 13
    topics in :data:`airm.config.CMR_TOPICS`.
    """
    return search_collections(
        count=count,
        extra_params={"science_keywords[0][topic]": topic},
        use_cache=use_cache,
    )


def _chunks(items: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fetch_by_concept_ids(cids: Iterable[str], *, use_cache: bool = True) -> list[dict]:
    """Fetch collections by exact concept-id, cache-first.

    Cached records are served from disk; only the misses hit the network, in
    chunked batches. Concept-ids CMR does not return are simply absent from the
    result -- callers are responsible for reporting them (see
    :func:`missing_concept_ids`), never for silently ignoring them.
    """
    ordered = list(dict.fromkeys(cids))  # dedup, preserve order
    found: dict[str, dict] = {}
    misses: list[str] = []
    for cid in ordered:
        cached = load_cached(cid) if use_cache else None
        if cached is not None:
            found[cid] = cached
        else:
            misses.append(cid)

    for chunk in _chunks(misses, _ID_CHUNK):
        params = [("page_size", str(len(chunk)))]
        params += [("concept_id", cid) for cid in chunk]
        for item in _get(f"{CMR_BASE}/collections.umm_json", params).get("items", []):
            cid = concept_id(item)
            found[cid] = item
            if use_cache:
                _write_cache(cid, item)

    return [found[cid] for cid in ordered if cid in found]


def missing_concept_ids(cids: Iterable[str], *, use_cache: bool = True) -> list[str]:
    """Return the concept-ids CMR cannot resolve -- for explicit reporting."""
    ordered = list(dict.fromkeys(cids))
    got = {concept_id(r) for r in fetch_by_concept_ids(ordered, use_cache=use_cache)}
    return [c for c in ordered if c not in got]


# --------------------------------------------------------------------------- #
# Corpus JSONL.
# --------------------------------------------------------------------------- #


def write_corpus(records: list[dict], path: Path = CORPUS_PATH) -> Path:
    """Write records as JSONL, one per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    tmp.replace(path)
    return path


def load_corpus(path: Path = CORPUS_PATH) -> list[dict]:
    """Load a JSONL corpus written by :func:`write_corpus`."""
    if not path.exists():
        raise FileNotFoundError(
            f"No corpus at {path}. Run `uv run python -m airm.corpus --build` first."
        )
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


if __name__ == "__main__":  # pragma: no cover - manual probe
    import argparse

    parser = argparse.ArgumentParser(description="Probe the CMR search API.")
    parser.add_argument("--concept-id", action="append", default=[])
    parser.add_argument("--topic")
    parser.add_argument("-n", type=int, default=5)
    args = parser.parse_args()

    if args.concept_id:
        for rec in fetch_by_concept_ids(args.concept_id):
            print(concept_id(rec), "|", entry_title(rec))
        for cid in missing_concept_ids(args.concept_id):
            print(f"MISSING: {cid}")
    elif args.topic:
        for rec in search_by_topic(args.topic, count=args.n):
            print(concept_id(rec), "|", entry_title(rec))
    else:
        parser.error("pass --concept-id or --topic")
