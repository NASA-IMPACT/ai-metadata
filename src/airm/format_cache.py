"""On-disk renderings — one file per (record, format), mirroring ``cmr_cache``.

``data/cmr_cache/<concept-id>.json`` holds the raw CMR item. This module gives
every format the same treatment::

    data/format_cache/json/C1234-PROV.json
    data/format_cache/csv/C1234-PROV.csv
    data/format_cache/yaml/C1234-PROV.yaml
    data/format_cache/toon/C1234-PROV.toon
    data/format_cache/jsonld/C1234-PROV.jsonld
    data/format_cache/mat/C1234-PROV.txt

so the mapping is one-to-one and total: every concept-id in ``cmr_cache`` has
exactly one file in every format directory, and every file in a format
directory corresponds to a cached record. :func:`verify` asserts both
directions rather than trusting them, because a *partial* cache is the failure
mode that matters -- it would silently drop records from an experiment.

Why cache at all
----------------
The renderings are deterministic, so this buys three things rather than one:

* **Inspectability.** ``diff``-able artefacts. The reason a format costs what it
  costs is legible without running Python.
* **Speed.** Rendering the full cache takes real time, and Experiment 1,
  Experiment 2 and the index build each need the same strings.
* **A stable referent.** Token counts, retrieval results and LLM answers can all
  be traced back to the exact bytes that produced them.

Staleness is the hazard a rendering cache introduces, so it is closed off rather
than documented away. The manifest records a :func:`fingerprint` over the source
of :mod:`airm.facets` and :mod:`airm.formats` -- every line of code that can
change a rendering. Any edit to either module changes the fingerprint, and
:func:`load` then refuses to serve stale bytes instead of quietly poisoning a
run with renderings from a previous version of the code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from . import cmr, formats
from .config import CMR_CACHE_DIR, FORMAT_CACHE_DIR, FORMATS
from .facets import facets

#: File extension per format. ``mat`` is prose, so it gets ``.txt`` rather than
#: a suffix implying a parseable structure it does not have.
EXTENSIONS: dict[str, str] = {
    "json": ".json",
    "csv": ".csv",
    "yaml": ".yaml",
    "toon": ".toon",
    "jsonld": ".jsonld",
    "mat": ".txt",
}

MANIFEST_NAME = "manifest.json"


class StaleCacheError(RuntimeError):
    """Raised when the cache was written by different rendering code."""


# --------------------------------------------------------------------------- #
# Fingerprint.
# --------------------------------------------------------------------------- #


def fingerprint() -> str:
    """A hash over every module that can change a rendering.

    ``facets`` decides *what* is rendered and ``formats`` decides *how*; nothing
    else touches the output. Hashing their source means a cache written before an
    edit to either is detectable as stale, which is the only reliable way to stop
    a rendering cache from silently outliving the code that produced it.
    """
    here = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ("facets.py", "formats.py"):
        digest.update((here / name).read_bytes())
    return digest.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Paths.
# --------------------------------------------------------------------------- #


def format_dir(fmt: str, *, root: Path | None = None) -> Path:
    return (root or FORMAT_CACHE_DIR) / fmt


def path_for(fmt: str, cid: str, *, root: Path | None = None) -> Path:
    try:
        ext = EXTENSIONS[fmt]
    except KeyError:
        raise KeyError(f"unknown format {fmt!r}; known: {', '.join(EXTENSIONS)}") from None
    return format_dir(fmt, root=root) / f"{cid}{ext}"


def cached_concept_ids(*, cache_dir: Path | None = None) -> list[str]:
    """Every concept-id with a raw record on disk, sorted."""
    return sorted(p.stem for p in (cache_dir or CMR_CACHE_DIR).glob("C*.json"))


def _write(path: Path, text: str) -> None:
    """Write atomically, for the same reason :func:`cmr._write_cache` does.

    A rendering interrupted mid-write would leave a truncated file that parses
    as valid-but-wrong for CSV and prose, which is worse than a crash.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Build.
# --------------------------------------------------------------------------- #


@dataclass
class CacheReport:
    """What was rendered, what was skipped, and by which version of the code."""

    fingerprint: str = ""
    records: int = 0
    written: dict[str, int] = field(default_factory=dict)
    skipped_unchanged: int = 0
    render_errors: list[dict] = field(default_factory=list)
    parity_failures: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint,
            "formats": list(FORMATS),
            "extensions": EXTENSIONS,
            "records": self.records,
            "written": self.written,
            "skipped_unchanged": self.skipped_unchanged,
            "render_errors": self.render_errors,
            "parity_failures": self.parity_failures,
        }


def build(
    concept_ids: list[str] | None = None,
    *,
    root: Path | None = None,
    cache_dir: Path | None = None,
    force: bool = False,
    check_parity: bool = True,
) -> CacheReport:
    """Render every cached record in every format.

    Incremental by default: a record whose files all exist is skipped unless
    ``force``. The fingerprint guard makes that safe -- a code change invalidates
    the whole cache at once, so "exists" cannot mean "exists but was rendered by
    older code".

    Parity is checked here rather than only at corpus-build time so a record that
    cannot be represented identically in all six formats is visible in the
    manifest from the moment it is cached, not just when it is sampled.
    """
    root = root or FORMAT_CACHE_DIR
    fp = fingerprint()
    report = CacheReport(fingerprint=fp)

    if force or _manifest_fingerprint(root) != fp:
        # Renderings on disk came from different code; every one of them is
        # suspect, so no file is treated as up to date this pass.
        force = True

    if concept_ids is None:
        concept_ids = cached_concept_ids(cache_dir=cache_dir)
    report.records = len(concept_ids)
    report.written = {fmt: 0 for fmt in FORMATS}

    for cid in concept_ids:
        paths = {fmt: path_for(fmt, cid, root=root) for fmt in FORMATS}
        if not force and all(p.exists() for p in paths.values()):
            report.skipped_unchanged += 1
            continue

        record = cmr.load_cached(cid) if cache_dir is None else _load_from(cache_dir, cid)
        if record is None:
            report.render_errors.append({"concept_id": cid, "error": "raw record missing"})
            continue

        try:
            payload = facets(record)
            rendered = {fmt: formats.render(fmt, payload) for fmt in FORMATS}
        except Exception as exc:  # noqa: BLE001 - one bad record must not stop the build
            report.render_errors.append(
                {"concept_id": cid, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue

        for fmt, text in rendered.items():
            _write(paths[fmt], text)
            report.written[fmt] += 1

        if check_parity:
            detail = formats.parity_report(payload)
            failed = {k: v for k, v in detail.items() if not v["ok"]}
            if failed:
                report.parity_failures.append({"concept_id": cid, "formats": failed})

    _write_manifest(root, report)
    return report


def _load_from(cache_dir: Path, cid: str) -> dict | None:
    path = cache_dir / f"{cid}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


# --------------------------------------------------------------------------- #
# Manifest.
# --------------------------------------------------------------------------- #


def manifest_path(root: Path | None = None) -> Path:
    return (root or FORMAT_CACHE_DIR) / MANIFEST_NAME


def load_manifest(root: Path | None = None) -> dict | None:
    path = manifest_path(root)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _manifest_fingerprint(root: Path | None = None) -> str | None:
    manifest = load_manifest(root)
    return manifest.get("fingerprint") if manifest else None


def _write_manifest(root: Path, report: CacheReport) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path(root).write_text(json.dumps(report.to_dict(), indent=2) + "\n")


# --------------------------------------------------------------------------- #
# Read.
# --------------------------------------------------------------------------- #


def load(fmt: str, cid: str, *, root: Path | None = None, strict: bool = True) -> str:
    """The cached rendering of ``cid`` in ``fmt``.

    ``strict`` compares the manifest fingerprint against the current code and
    raises :class:`StaleCacheError` on a mismatch. Pass ``strict=False`` only to
    inspect an old cache deliberately; an experiment must never do it, since the
    whole guarantee of the cache is that its bytes match its code.
    """
    if strict:
        stored = _manifest_fingerprint(root)
        current = fingerprint()
        if stored != current:
            raise StaleCacheError(
                f"format cache was written by rendering code {stored!r}, current is "
                f"{current!r}. Rebuild with `uv run python -m airm.format_cache --build`."
            )
    return path_for(fmt, cid, root=root).read_text(encoding="utf-8")


def load_all(fmt: str, *, root: Path | None = None, strict: bool = True) -> dict[str, str]:
    """Every cached rendering in ``fmt``, keyed by concept-id."""
    if strict:
        load_guard(root)
    ext = EXTENSIONS[fmt]
    return {
        p.name[: -len(ext)]: p.read_text(encoding="utf-8")
        for p in sorted(format_dir(fmt, root=root).glob(f"*{ext}"))
    }


def load_guard(root: Path | None = None) -> None:
    """Raise unless the cache on disk was written by the current code."""
    stored = _manifest_fingerprint(root)
    current = fingerprint()
    if stored != current:
        raise StaleCacheError(
            f"format cache was written by rendering code {stored!r}, current is "
            f"{current!r}. Rebuild with `uv run python -m airm.format_cache --build`."
        )


def is_current(root: Path | None = None) -> bool:
    """Whether the cache on disk was written by the current rendering code."""
    return _manifest_fingerprint(root) == fingerprint()


def render_many(
    fmt: str, items: list[tuple[str | None, dict]], *, root: Path | None = None
) -> list[str]:
    """Renderings for ``(concept_id, payload)`` pairs -- cache first, live otherwise.

    Callers get the speed of the cache without depending on it: a missing, stale
    or partial cache costs time, never correctness, because the fallback renders
    the same ``payload`` through the same code that filled the cache. The
    fingerprint is checked once for the whole batch rather than per record.
    """
    usable = is_current(root)
    out: list[str] = []
    for cid, payload in items:
        if usable and cid is not None:
            path = path_for(fmt, cid, root=root)
            if path.exists():
                out.append(path.read_text(encoding="utf-8"))
                continue
        out.append(formats.render(fmt, payload))
    return out


def render_all_cached(cid: str | None, payload: dict, *, root: Path | None = None) -> dict[str, str]:
    """Every format's rendering of one record, cache first."""
    usable = is_current(root)
    out: dict[str, str] = {}
    for fmt in FORMATS:
        if usable and cid is not None:
            path = path_for(fmt, cid, root=root)
            if path.exists():
                out[fmt] = path.read_text(encoding="utf-8")
                continue
        out[fmt] = formats.render(fmt, payload)
    return out


# --------------------------------------------------------------------------- #
# Verification.
# --------------------------------------------------------------------------- #


def verify(*, root: Path | None = None, cache_dir: Path | None = None) -> list[str]:
    """Hard-gate violations; empty when the mapping is one-to-one and current.

    Checks both directions. A format directory missing a record would silently
    shrink an experiment; one holding an *extra* record means a concept-id was
    dropped from ``cmr_cache`` and left a rendering behind, which would let a
    stale record back into a run.
    """
    problems: list[str] = []
    expected = set(cached_concept_ids(cache_dir=cache_dir))

    manifest = load_manifest(root)
    if manifest is None:
        return [f"no manifest at {manifest_path(root)}; the cache has never been built"]
    if manifest.get("fingerprint") != fingerprint():
        problems.append(
            f"stale: written by rendering code {manifest.get('fingerprint')!r}, "
            f"current is {fingerprint()!r}"
        )

    errored = {e["concept_id"] for e in manifest.get("render_errors") or []}
    renderable = expected - errored
    if errored:
        problems.append(f"{len(errored)} record(s) failed to render: {sorted(errored)[:5]}")

    for fmt in FORMATS:
        present = set(load_all(fmt, root=root, strict=False))
        missing = renderable - present
        extra = present - expected
        if missing:
            problems.append(
                f"{fmt}: {len(missing)} record(s) with no rendering: {sorted(missing)[:5]}"
            )
        if extra:
            problems.append(
                f"{fmt}: {len(extra)} rendering(s) with no raw record: {sorted(extra)[:5]}"
            )
    return problems


if __name__ == "__main__":  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(description="Render every cached CMR record in every format.")
    parser.add_argument("--build", action="store_true", help="render and write the cache")
    parser.add_argument("--force", action="store_true", help="re-render even if files exist")
    parser.add_argument("--verify", action="store_true", help="check the one-to-one mapping")
    parser.add_argument("--show", metavar="CONCEPT_ID", help="print every rendering of one record")
    args = parser.parse_args()

    if args.show:
        for fmt in FORMATS:
            text = load(fmt, args.show, strict=False)
            print(f"\n{'=' * 72}\n{fmt}  ({len(text)} chars)\n{'=' * 72}")
            print(text)
        raise SystemExit(0)

    if args.build:
        rep = build(force=args.force)
        print(f"fingerprint      : {rep.fingerprint}")
        print(f"records          : {rep.records}")
        print(f"skipped (current): {rep.skipped_unchanged}")
        print("written per format:")
        for fmt in FORMATS:
            print(f"  {fmt:<8} {rep.written.get(fmt, 0):>6}")
        if rep.render_errors:
            print(f"render errors    : {len(rep.render_errors)}")
            for err in rep.render_errors[:5]:
                print(f"  {err['concept_id']}: {err['error']}")
        if rep.parity_failures:
            print(f"parity failures  : {len(rep.parity_failures)}")
            for failure in rep.parity_failures[:5]:
                print(f"  {failure['concept_id']}: {list(failure['formats'])}")
        print(f"wrote {manifest_path()}")

    if args.verify or args.build:
        issues = verify()
        if issues:
            print("\nHARD GATE FAILURES:")
            for issue in issues:
                print(f"  - {issue}")
            raise SystemExit(1)
        print("\ncache is one-to-one with cmr_cache and current with the rendering code")

    if not (args.build or args.verify or args.show):
        parser.error("pass --build, --verify or --show")
