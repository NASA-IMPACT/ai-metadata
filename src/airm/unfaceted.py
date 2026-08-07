"""The unfaceted control — the same six formats over the *whole* raw UMM record.

:mod:`airm.facets` projects a CMR record onto 32 fields and every renderer reads
that projection. This module renders the record CMR actually serves: all 45
top-level UMM fields, 398 distinct paths, contacts and addresses and metadata
dates included. Same six formats, same parity discipline, no projection.

Why it exists
-------------
Experiment 1 reports a ``json_umm`` reference — what the status quo costs — but
only in JSON, so it answers "how much does faceting save?" and not "does the
format ranking survive without faceting?". Those are different questions, and
the second one is the one a reader who distrusts the projection will ask. This
cache answers it by holding content constant at the *other* extreme.

It is a control, not a competitor. Nothing in the study's main line reads it:
the corpus, the index and both experiments go through
:mod:`airm.format_cache`. Faceted and unfaceted renderings are **not**
comparable to each other record-for-record, because they carry different content
— which is the whole reason the faceted payload exists.

What breaks without a schema
----------------------------
Four of the six formats are schema-agnostic: JSON, YAML, TOON and CSV serialise
an arbitrary tree and read it back. The other two cannot, and their degradation
is the most informative thing this module produces:

* **JSON-LD** has schema.org properties for five UMM fields and nothing for the
  other forty. The rest can only be carried as ``PropertyValue`` entries keyed by
  path, which is legal JSON-LD and semantically inert — the ``@context`` buys
  nothing when every term is a literal path string.
* **Metadata-as-Text** cannot be prose. Prose requires knowing what a field
  *means* in order to write a sentence about it; over an arbitrary tree the best
  available rendering is one ``path is value`` clause per leaf. It satisfies the
  value-coverage parity check and reads like a database dump, because that is
  what it is.

Both are rendered anyway, and honestly labelled, because "this format needs a
schema and that schema is exactly what faceting provides" is a finding rather
than an obstacle.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import cmr, formats
from .config import CMR_CACHE_DIR, DATA_DIR, FORMATS
from .facets import fact_set, fact_values, flatten, scalar_str

#: Sibling of ``data/format_cache``, same layout, different payload.
UNFACETED_CACHE_DIR = DATA_DIR / "format_cache_unfaceted"

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
# Payload.
# --------------------------------------------------------------------------- #


def payload(record: dict) -> dict:
    """The unfaceted payload: the record's whole ``umm`` object, untouched.

    The analogue of :func:`airm.facets.facets`, and deliberately trivial. No
    projection, no placeholder filtering, no canonicalisation — the point of the
    control is that nothing has been decided on the reader's behalf.

    ``meta`` is excluded: it is CMR's record-keeping (revision id, ingest dates)
    rather than metadata about the dataset, and including it would put the
    concept-id in some formats and not others.
    """
    return record.get("umm", record)


# --------------------------------------------------------------------------- #
# JSON-LD over an arbitrary tree.
# --------------------------------------------------------------------------- #

#: The five UMM fields schema.org actually models. Everything else in the record
#: has no schema.org property and can only travel as a keyed literal.
JSONLD_MAPPED: tuple[tuple[str, str], ...] = (
    ("EntryTitle", "name"),
    ("Abstract", "description"),
    ("ShortName", "alternateName"),
    ("Version", "version"),
    ("DOI.DOI", "identifier"),
)


def render_jsonld(p: dict) -> str:
    """schema.org ``Dataset`` over a full UMM record.

    Five fields get real properties; every other leaf becomes a
    ``PropertyValue`` keyed by its flattened path. That is what JSON-LD without
    a schema degrades to, and the ratio — 5 modelled against ~400 paths — is the
    measurement worth taking.
    """
    doc: dict[str, Any] = {"@context": formats.JSONLD_CONTEXT, "@type": "Dataset"}
    leaves = flatten(p)
    mapped = dict(JSONLD_MAPPED)

    for path, value in leaves:
        if path in mapped:
            doc[mapped[path]] = value

    doc["additionalProperty"] = [
        {"@type": "PropertyValue", "name": path, "value": value}
        for path, value in leaves
        if path not in mapped
    ]
    return json.dumps(doc, ensure_ascii=False, separators=(",", ":"))


def parse_jsonld(text: str) -> dict:
    """Invert :func:`render_jsonld` by reassembling paths into a tree."""
    doc = json.loads(text)
    root: dict = {}
    for umm_path, schema_key in JSONLD_MAPPED:
        if schema_key in doc:
            formats._assign(root, formats._path_tokens(umm_path), doc[schema_key])
    for prop in doc.get("additionalProperty") or []:
        formats._assign(root, formats._path_tokens(prop["name"]), prop["value"])
    return root


# --------------------------------------------------------------------------- #
# Prose over an arbitrary tree.
# --------------------------------------------------------------------------- #

#: How a flattened path is spoken. ``SpatialExtent.HorizontalSpatialDomain`` ->
#: ``SpatialExtent > HorizontalSpatialDomain``, matching GCMD's own notation for
#: hierarchy so the prose at least reads consistently with the keyword paths.
PATH_SEPARATOR = " > "


def _speak_path(path: str) -> str:
    return path.replace(".", PATH_SEPARATOR)


def render_mat(p: dict) -> str:
    """One ``path is value`` clause per leaf.

    Not prose, and labelled as such. Writing a sentence about a field requires
    knowing what the field means; over 398 arbitrary paths there is no such
    knowledge, so the only faithful rendering is mechanical. Every leaf value
    appears verbatim, so the value-coverage parity check still applies.
    """
    lines = []
    for path, value in flatten(p):
        lines.append(f"{_speak_path(path)} is {scalar_str(value)}.")
    return " ".join(lines)


# --------------------------------------------------------------------------- #
# Registries and parity.
# --------------------------------------------------------------------------- #

RENDERERS = {
    "json": formats.render_json,
    "csv": formats.render_csv,
    "yaml": formats.render_yaml,
    "toon": formats.render_toon,
    "jsonld": render_jsonld,
    "mat": render_mat,
}

PARSERS = {
    "json": formats.parse_json,
    "csv": formats.parse_csv,
    "yaml": formats.parse_yaml,
    "toon": formats.parse_toon,
    "jsonld": parse_jsonld,
}

PROSE_FORMATS = ("mat",)


def render(fmt: str, p: dict) -> str:
    try:
        return RENDERERS[fmt](p)
    except KeyError:
        raise KeyError(f"unknown format {fmt!r}; known: {', '.join(RENDERERS)}") from None


def render_all(record: dict) -> dict[str, str]:
    p = payload(record)
    return {fmt: render(fmt, p) for fmt in FORMATS}


def parity_report(p: dict) -> dict[str, dict]:
    """Per-format parity for one unfaceted payload.

    Identical in shape to :func:`airm.formats.parity_report`, so the two caches
    are judged by the same standard rather than the control being graded gently.
    """
    expected = fact_set(p)
    report: dict[str, dict] = {}

    for fmt, parser in PARSERS.items():
        try:
            got = fact_set(parser(render(fmt, p)))
        except Exception as exc:  # noqa: BLE001 - any failure is a parity failure
            report[fmt] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            continue
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        report[fmt] = {"ok": not missing and not extra, "missing": missing, "extra": extra}

    # Dispatch through :func:`render` rather than calling ``render_mat``
    # directly, so ``RENDERERS`` is the single point of dispatch and a swapped
    # renderer is actually seen by the validator.
    for fmt in PROSE_FORMATS:
        text = render(fmt, p)
        missing_values = [v for v in fact_values(p) if v not in text]
        report[fmt] = {
            "ok": not missing_values,
            "missing": [("<value>", v) for v in missing_values],
            "extra": [],
        }
    return report


def check_record(record: dict) -> tuple[bool, dict[str, dict]]:
    report = parity_report(payload(record))
    return all(r["ok"] for r in report.values()), report


# --------------------------------------------------------------------------- #
# Paths and fingerprint.
# --------------------------------------------------------------------------- #


def fingerprint() -> str:
    """Hash over every module that can change an unfaceted rendering.

    ``formats.py`` supplies four of the six renderers and this module supplies
    the other two, so both are hashed. ``facets.py`` is deliberately *not*: the
    unfaceted payload does not go through it, which is the point, and hashing it
    would invalidate this cache every time the projection changed.
    """
    digest = hashlib.sha256()
    here = Path(__file__).parent
    for name in ("formats.py", "unfaceted.py"):
        digest.update((here / name).read_bytes())
    return digest.hexdigest()[:16]


def format_dir(fmt: str, *, root: Path | None = None) -> Path:
    return (root or UNFACETED_CACHE_DIR) / fmt


def path_for(fmt: str, cid: str, *, root: Path | None = None) -> Path:
    try:
        ext = EXTENSIONS[fmt]
    except KeyError:
        raise KeyError(f"unknown format {fmt!r}; known: {', '.join(EXTENSIONS)}") from None
    return format_dir(fmt, root=root) / f"{cid}{ext}"


def cached_concept_ids(*, cache_dir: Path | None = None) -> list[str]:
    return sorted(p.stem for p in (cache_dir or CMR_CACHE_DIR).glob("C*.json"))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Build.
# --------------------------------------------------------------------------- #


@dataclass
class CacheReport:
    fingerprint: str = ""
    records: int = 0
    written: dict[str, int] = field(default_factory=dict)
    skipped_unchanged: int = 0
    render_errors: list[dict] = field(default_factory=list)
    parity_failures: list[dict] = field(default_factory=list)
    #: Per-format count of records that failed to round-trip, so the TOON
    #: shortfall is a headline number rather than something to derive.
    parity_by_format: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint,
            "payload": "raw UMM (unfaceted)",
            "formats": list(FORMATS),
            "extensions": EXTENSIONS,
            "records": self.records,
            "written": self.written,
            "skipped_unchanged": self.skipped_unchanged,
            "render_errors": self.render_errors,
            "parity_by_format": self.parity_by_format,
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
    """Render every cached record, unfaceted, in every format."""
    root = root or UNFACETED_CACHE_DIR
    fp = fingerprint()
    report = CacheReport(fingerprint=fp)

    if force or _manifest_fingerprint(root) != fp:
        force = True

    if concept_ids is None:
        concept_ids = cached_concept_ids(cache_dir=cache_dir)
    report.records = len(concept_ids)
    report.written = {fmt: 0 for fmt in FORMATS}
    report.parity_by_format = {fmt: 0 for fmt in FORMATS}

    for cid in concept_ids:
        paths = {fmt: path_for(fmt, cid, root=root) for fmt in FORMATS}
        if not force and all(p.exists() for p in paths.values()):
            report.skipped_unchanged += 1
            continue

        record = cmr.load_cached(cid) if cache_dir is None else _load_from(cache_dir, cid)
        if record is None:
            report.render_errors.append({"concept_id": cid, "error": "raw record missing"})
            continue

        p = payload(record)
        try:
            rendered = {fmt: render(fmt, p) for fmt in FORMATS}
        except Exception as exc:  # noqa: BLE001 - one bad record must not stop the build
            report.render_errors.append(
                {"concept_id": cid, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue

        for fmt, text in rendered.items():
            _write(paths[fmt], text)
            report.written[fmt] += 1

        if check_parity:
            detail = parity_report(p)
            failed = {k: v for k, v in detail.items() if not v["ok"]}
            for fmt in failed:
                report.parity_by_format[fmt] += 1
            if failed:
                report.parity_failures.append(
                    {
                        "concept_id": cid,
                        # Full missing/extra lists run to thousands of entries on
                        # a raw record; the format and a sample is what a reader
                        # can act on.
                        "formats": {
                            k: v.get("error") or {
                                "missing": v["missing"][:3],
                                "extra": v["extra"][:3],
                                "n_missing": len(v["missing"]),
                                "n_extra": len(v["extra"]),
                            }
                            for k, v in failed.items()
                        },
                    }
                )

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
# Manifest and read.
# --------------------------------------------------------------------------- #


def manifest_path(root: Path | None = None) -> Path:
    return (root or UNFACETED_CACHE_DIR) / MANIFEST_NAME


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


def is_current(root: Path | None = None) -> bool:
    return _manifest_fingerprint(root) == fingerprint()


def load(fmt: str, cid: str, *, root: Path | None = None, strict: bool = True) -> str:
    if strict and not is_current(root):
        raise StaleCacheError(
            f"unfaceted cache was written by rendering code {_manifest_fingerprint(root)!r}, "
            f"current is {fingerprint()!r}. Rebuild with "
            f"`uv run python -m airm.unfaceted --build`."
        )
    return path_for(fmt, cid, root=root).read_text(encoding="utf-8")


def load_all(fmt: str, *, root: Path | None = None, strict: bool = True) -> dict[str, str]:
    if strict and not is_current(root):
        raise StaleCacheError("unfaceted cache is stale; rebuild it")
    ext = EXTENSIONS[fmt]
    return {
        p.name[: -len(ext)]: p.read_text(encoding="utf-8")
        for p in sorted(format_dir(fmt, root=root).glob(f"*{ext}"))
    }


# --------------------------------------------------------------------------- #
# Verification.
# --------------------------------------------------------------------------- #


def verify(*, root: Path | None = None, cache_dir: Path | None = None) -> list[str]:
    """Hard-gate violations; empty when the mapping is one-to-one and current.

    Parity failures are *not* a violation here. TOON cannot round-trip every raw
    UMM record, and that is a measured property of the control rather than a
    broken build -- unlike the faceted cache, where a parity failure means a
    record must leave the corpus.
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
            problems.append(f"{fmt}: {len(missing)} record(s) with no rendering: {sorted(missing)[:5]}")
        if extra:
            problems.append(f"{fmt}: {len(extra)} rendering(s) with no raw record: {sorted(extra)[:5]}")
    return problems


if __name__ == "__main__":  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(
        description="Render every cached CMR record, unfaceted, in every format."
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--show", metavar="CONCEPT_ID")
    args = parser.parse_args()

    if args.show:
        for fmt in FORMATS:
            text = load(fmt, args.show, strict=False)
            print(f"\n{'=' * 72}\n{fmt}  ({len(text):,} chars)\n{'=' * 72}")
            print(text[:2000] + ("\n... [truncated]" if len(text) > 2000 else ""))
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
                print(f"  {err['concept_id']}: {err['error'][:100]}")
        print("round-trip failures per format:")
        for fmt in FORMATS:
            n = rep.parity_by_format.get(fmt, 0)
            pct = 100 * n / rep.records if rep.records else 0
            print(f"  {fmt:<8} {n:>6}  ({pct:.1f}%)")
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
