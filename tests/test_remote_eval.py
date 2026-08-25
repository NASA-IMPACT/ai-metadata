"""Tests for the retrieval-only harness against the deployed vector database.

The endpoint is stubbed at the ChromaDB and embedding boundaries. What is under
test is everything this module actually owns: the four preflight gates, the
extraction of ranked concept-ids out of a Chroma response, nan handling in the
aggregate, the paired comparison against the baseline format, the truncation
audit, and the shape of what gets written to disk.

Metric values are asserted against numbers computed by hand rather than against
the code that produced them -- the same discipline as ``tests/test_evaluate.py``.
"""

from __future__ import annotations

import csv
import json
import math

import pytest

from airm import remote_eval
from airm.queries import Query
from airm.remote_eval import Backend, Endpoint, RemoteEvalError

FORMATS = ("json", "mat")

QUERIES = [
    # Two expected records: one is retrieved at rank 2 by json, neither by mat.
    Query("q-1", "sea ice thickness", ["A", "B"], "sme", None, ["Sea Ice"]),
    # One expected record, retrieved first by json and third by mat.
    Query("q-2", "soil moisture", ["C"], "synthetic", "LAND SURFACE", ["Soil Moisture"]),
]

RANKINGS = {
    ("q-1", "json"): ["X", "A", "Y"],
    ("q-1", "mat"): ["X", "Y", "Z"],
    ("q-2", "json"): ["C", "X", "Y"],
    ("q-2", "mat"): ["X", "Y", "C"],
}


class FakeCollection:
    """Enough of a Chroma collection for the gates and the search."""

    def __init__(self, fmt, ids, documents=None, vectors=None, metadata=None):
        self.fmt = fmt
        self._ids = list(ids)
        self._documents = documents or {c: f"{fmt} document for {c}" for c in ids}
        self._vectors = vectors or {c: [1.0, 0.0] for c in ids}
        self.metadata = metadata or {"hnsw:space": "cosine"}
        self.queried = 0

    def count(self):
        return len(self._ids)

    def get(self, ids=None, include=None, limit=None):
        include = include or []
        self.gets = getattr(self, "gets", 0) + 1
        chosen = [c for c in (ids if ids is not None else self._ids) if c in set(self._ids)]
        if limit is not None:
            chosen = chosen[:limit]
        out = {"ids": chosen}
        if "documents" in include:
            out["documents"] = [self._documents[c] for c in chosen]
        if "embeddings" in include:
            out["embeddings"] = [self._vectors[c] for c in chosen]
        return out

    def query(self, query_embeddings, n_results, include=None):
        self.queried += 1
        # The stub embeds each query as [index, 0.0]; recover the query from it.
        rows = []
        for vector in query_embeddings:
            query = QUERIES[int(vector[0])]
            rows.append(RANKINGS[(query.query_id, self.fmt)][:n_results])
        return {"ids": rows}


class FakeClient:
    def __init__(self, collections):
        self._collections = collections

    def get_collection(self, name):
        if name not in self._collections:
            raise ValueError(f"no collection {name}")
        return self._collections[name]


def make_backend(*, ids=("A", "B", "C", "X", "Y", "Z"), formats=FORMATS, prefix="nasa_cmr_",
                 lengths=None, window=512, drop=None):
    """A backend whose embeddings encode the query index, so rankings are exact."""
    drop = drop or {}
    collections = {}
    for fmt in formats:
        keep = [c for c in ids if c not in drop.get(fmt, ())]
        collections[f"{prefix}{fmt}"] = FakeCollection(fmt, keep)

    def embed(texts):
        vectors = []
        for text in texts:
            match = [i for i, q in enumerate(QUERIES) if q.text == text]
            vectors.append([float(match[0]) if match else 0.0, 0.0])
        return vectors

    def count_tokens(texts):
        if lengths is None:
            return [10 for _ in texts]
        # Keyed on the format prefix the fake documents carry.
        return [lengths.get(t.split()[0], 10) for t in texts]

    return Backend(client=FakeClient(collections), embed=embed, count_tokens=count_tokens,
                   window=window), collections


#: ``ssl=False`` explicitly: port 8000 is the plain-HTTP shape, and pinning it
#: here keeps the asserted endpoint URL independent of the production default.
ENDPOINT = Endpoint("stub-host", 8000, "nasa_cmr_", False)


# --------------------------------------------------------------------------- #
# Ranked ids and metrics.
# --------------------------------------------------------------------------- #


def test_ranked_ids_come_out_of_the_chroma_response_in_rank_order():
    backend, collections = make_backend()
    opened = remote_eval.open_collections(backend, ENDPOINT, FORMATS)
    ranked = remote_eval.search(backend, opened, QUERIES, k=3)

    assert ranked[("q-1", "json")] == ["X", "A", "Y"]
    assert ranked[("q-2", "mat")] == ["X", "Y", "C"]


def test_top_k_is_honoured():
    backend, _ = make_backend()
    opened = remote_eval.open_collections(backend, ENDPOINT, FORMATS)
    ranked = remote_eval.search(backend, opened, QUERIES, k=1)

    assert ranked[("q-1", "json")] == ["X"]


def test_queries_are_embedded_once_and_reused_across_formats():
    calls = {"n": 0}
    backend, _ = make_backend()
    inner = backend.embed

    def counting(texts):
        calls["n"] += 1
        return inner(texts)

    backend.embed = counting
    opened = remote_eval.open_collections(backend, ENDPOINT, FORMATS)
    remote_eval.search(backend, opened, QUERIES, k=3)

    # The vector does not depend on the format, so one batch covers all six.
    assert calls["n"] == 1


def test_batched_search_keeps_each_ranking_with_its_own_query(monkeypatch):
    """131 queries do not fit one request, so the offset arithmetic has to hold."""
    monkeypatch.setattr(remote_eval, "QUERY_BATCH", 1)
    backend, _ = make_backend()
    opened = remote_eval.open_collections(backend, ENDPOINT, FORMATS)
    ranked = remote_eval.search(backend, opened, QUERIES, k=3)

    assert ranked[("q-1", "json")] == ["X", "A", "Y"]
    assert ranked[("q-2", "json")] == ["C", "X", "Y"]


def test_a_short_ranking_batch_is_an_error():
    backend, collections = make_backend()
    collections["nasa_cmr_json"].query = lambda **kw: {"ids": []}
    opened = remote_eval.open_collections(backend, ENDPOINT, FORMATS)
    with pytest.raises(RemoteEvalError, match="asked for 2 rankings"):
        remote_eval.search(backend, opened, QUERIES, k=3)


def test_the_query_instruction_reaches_queries_and_never_documents():
    """bge prefixes queries only.

    The parity gate re-embeds a stored *document* through the same callable, so
    if the instruction were baked into the encoder it would compare a query
    vector against a document vector and fail for the wrong reason.
    """
    seen = []
    backend, collections = make_backend()
    inner = backend.embed

    def recording(texts):
        seen.extend(texts)
        return [[0.0, 0.0] for _ in texts] if texts[0].startswith("PREFIX:") else inner(texts)

    backend.embed = recording
    opened = remote_eval.open_collections(backend, ENDPOINT, FORMATS)

    remote_eval.check_embedding_parity(backend, collections["nasa_cmr_json"], "A", strict=False)
    assert seen == ["json document for A"]  # bare, no prefix

    seen.clear()
    remote_eval.search(backend, opened, QUERIES, k=3, instruction="PREFIX:")
    assert seen == ["PREFIX:sea ice thickness", "PREFIX:soil moisture"]


def test_live_backend_embeds_bare_so_the_parity_gate_stays_valid():
    """The instruction is not a constructor argument -- search applies it."""
    import inspect

    assert "instruction" not in inspect.signature(remote_eval.live_backend).parameters


def test_metrics_match_hand_computed_values():
    rows = remote_eval.score(QUERIES, RANKINGS, formats=FORMATS)
    by_key = {(r["query_id"], r["format"]): r for r in rows}

    # q-1/json: expects {A, B}, retrieved [X, A, Y]. One of two found by rank 3.
    row = by_key[("q-1", "json")]
    assert row["recall@1"] == 0.0
    assert row["recall@5"] == pytest.approx(0.5)
    assert row["mrr"] == pytest.approx(0.5)  # A is at rank 2
    # DCG = 1/log2(3) for the single hit at rank 2; ideal has two hits at ranks 1-2.
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert row["ndcg@10"] == pytest.approx((1 / math.log2(3)) / ideal)

    # q-2/mat: expects {C}, retrieved [X, Y, C].
    row = by_key[("q-2", "mat")]
    assert row["recall@1"] == 0.0
    assert row["recall@10"] == pytest.approx(1.0)
    assert row["mrr"] == pytest.approx(1 / 3)


def test_a_missing_ranking_is_an_error_not_a_silent_gap():
    partial = {k: v for k, v in RANKINGS.items() if k != ("q-2", "mat")}
    with pytest.raises(RemoteEvalError, match="No ranking recorded"):
        remote_eval.score(QUERIES, partial, formats=FORMATS)


# --------------------------------------------------------------------------- #
# Aggregation.
# --------------------------------------------------------------------------- #


def test_unmeasurable_queries_are_excluded_from_the_mean_not_averaged_in_as_zero():
    """A query with no expected records is unmeasurable, which is not a miss."""
    queries = [*QUERIES, Query("q-3", "no ground truth", [], "synthetic", None, [])]
    rankings = {**RANKINGS, ("q-3", "json"): ["X"], ("q-3", "mat"): ["X"]}
    rows = remote_eval.score(queries, rankings, formats=FORMATS)

    unmeasurable = [r for r in rows if r["query_id"] == "q-3"]
    assert all(r["recall@10"] != r["recall@10"] for r in unmeasurable)  # nan

    summary = remote_eval.summarise(rows, embed_model="m", formats=FORMATS)
    json_cell = summary["by_source"]["all"]["json|remote:m"]
    # Only q-1 (0.5) and q-2 (1.0) count; averaging the nan in as 0 would give 0.5.
    assert json_cell["recall@10"] == pytest.approx(0.75)


def test_sme_and_synthetic_slices_are_reported_separately():
    rows = remote_eval.score(QUERIES, RANKINGS, formats=FORMATS)
    summary = remote_eval.summarise(rows, embed_model="m", formats=FORMATS)

    assert set(summary["by_source"]) == {"all", "sme", "synthetic"}
    assert summary["by_source"]["sme"]["json|remote:m"]["recall@10"] == pytest.approx(0.5)
    assert summary["by_source"]["synthetic"]["json|remote:m"]["recall@10"] == pytest.approx(1.0)


def test_summary_keys_match_the_shape_report_py_consumes():
    rows = remote_eval.score(QUERIES, RANKINGS, formats=FORMATS)
    summary = remote_eval.summarise(rows, embed_model="bge", formats=FORMATS)

    assert "json|remote:bge" in summary["summary"]
    cell = summary["summary"]["json|remote:bge"]
    for key in ("recall@10", "mrr", "ndcg@10"):
        assert key in cell


def test_paired_delta_pairs_per_query_against_the_baseline():
    rows = remote_eval.score(QUERIES, RANKINGS, formats=FORMATS)
    summary = remote_eval.summarise(rows, embed_model="m", formats=FORMATS)

    # json Recall@10: 0.5, 1.0. mat: 0.0, 1.0. Mean per-query delta = -0.25.
    assert summary["paired_vs_baseline"]["formats"]["mat"]["recall@10"]["mean_delta"] == pytest.approx(-0.25)
    assert summary["ranking"]["order"] == ["json", "mat"]


# --------------------------------------------------------------------------- #
# The four preflight gates.
# --------------------------------------------------------------------------- #


def test_unreachable_endpoint_names_the_endpoint_and_the_curl():
    with pytest.raises(RemoteEvalError) as exc:
        remote_eval.connect(Endpoint("127.0.0.1", 1, "nasa_cmr_"), timeout=0.5)
    assert "127.0.0.1:1" in str(exc.value)
    assert "curl" in str(exc.value)


def test_a_missing_collection_fails_before_any_search():
    backend, _ = make_backend(formats=("json",))
    with pytest.raises(RemoteEvalError, match="Collections not found"):
        remote_eval.open_collections(backend, ENDPOINT, FORMATS)


def test_unequal_collection_sizes_are_fatal():
    backend, collections = make_backend()
    collections["nasa_cmr_mat"]._ids = ["A", "B"]
    opened = remote_eval.open_collections(backend, ENDPOINT, FORMATS)
    with pytest.raises(RemoteEvalError, match="same haystack"):
        remote_eval.check_counts(opened)


def test_an_empty_collection_is_fatal():
    backend, collections = make_backend()
    collections["nasa_cmr_mat"]._ids = []
    opened = remote_eval.open_collections(backend, ENDPOINT, FORMATS)
    with pytest.raises(RemoteEvalError, match="Empty collections"):
        remote_eval.check_counts(opened)


def test_embedding_parity_fails_when_the_encoder_is_a_different_model():
    backend, collections = make_backend()
    collection = collections["nasa_cmr_json"]
    collection._vectors["A"] = [1.0, 0.0]
    backend.embed = lambda texts: [[0.0, 1.0] for _ in texts]  # orthogonal

    with pytest.raises(RemoteEvalError, match="does not match the indexed vectors"):
        remote_eval.check_embedding_parity(backend, collection, "A")


def test_embedding_parity_passes_and_records_provenance():
    backend, collections = make_backend()
    collection = collections["nasa_cmr_json"]
    backend.embed = lambda texts: [[1.0, 0.0] for _ in texts]

    report = remote_eval.check_embedding_parity(backend, collection, "A")
    assert report["passed"] is True
    assert report["cosine_similarity"] == pytest.approx(1.0)
    assert report["hnsw_space"] == "cosine"
    assert report["stored_vector_norm"] == pytest.approx(1.0)


def test_embedding_mismatch_can_be_overridden_and_the_override_is_recorded():
    backend, collections = make_backend()
    backend.embed = lambda texts: [[0.0, 1.0] for _ in texts]

    report = remote_eval.check_embedding_parity(
        backend, collections["nasa_cmr_json"], "A", strict=False
    )
    assert report["passed"] is False
    assert "override" in report


def test_a_missing_ground_truth_id_is_fatal_by_default():
    backend, _ = make_backend(drop={"mat": ("B",)})
    opened = remote_eval.open_collections(backend, ENDPOINT, FORMATS)
    with pytest.raises(RemoteEvalError, match="unmeasurable"):
        remote_eval.check_ground_truth(opened, ["A", "B"])


def test_ground_truth_coverage_is_one_request_per_collection_not_one_per_id():
    """The lookup must stay out of the comprehension: 147 ids x 6 formats."""
    backend, collections = make_backend()
    opened = remote_eval.open_collections(backend, ENDPOINT, FORMATS)
    remote_eval.check_ground_truth(opened, ["A", "B", "C", "X"])

    assert collections["nasa_cmr_json"].gets == 1


def test_a_missing_ground_truth_id_can_be_recorded_instead():
    backend, _ = make_backend(drop={"mat": ("B",)})
    opened = remote_eval.open_collections(backend, ENDPOINT, FORMATS)
    missing = remote_eval.check_ground_truth(opened, ["A", "B"], strict=False)
    assert missing["mat"] == ["B"]
    assert missing["json"] == []


# --------------------------------------------------------------------------- #
# The truncation audit.
# --------------------------------------------------------------------------- #


def test_truncation_audit_counts_over_window_documents_per_format():
    backend, _ = make_backend(lengths={"json": 900, "mat": 100}, window=512)
    opened = remote_eval.open_collections(backend, ENDPOINT, FORMATS)
    audit = remote_eval.truncation_audit(backend, opened, sample=None)

    assert audit["by_format"]["json"]["over_window_share"] == pytest.approx(1.0)
    assert audit["by_format"]["mat"]["over_window_share"] == pytest.approx(0.0)
    assert "must not be read as a clean format comparison" in audit["caveat"]


def test_truncation_audit_says_so_when_nothing_is_clipped():
    backend, _ = make_backend(lengths={"json": 100, "mat": 100}, window=512)
    opened = remote_eval.open_collections(backend, ENDPOINT, FORMATS)
    audit = remote_eval.truncation_audit(backend, opened, sample=None)

    assert "No sampled document exceeds" in audit["caveat"]


def test_truncation_audit_samples_the_same_records_in_every_format():
    backend, _ = make_backend()
    opened = remote_eval.open_collections(backend, ENDPOINT, FORMATS)
    audit = remote_eval.truncation_audit(backend, opened, sample=3)

    assert audit["sampled_records"] == 3
    assert {v["sampled"] for v in audit["by_format"].values()} == {3}


def test_truncation_audit_can_be_skipped():
    backend, _ = make_backend()
    opened = remote_eval.open_collections(backend, ENDPOINT, FORMATS)
    assert remote_eval.truncation_audit(backend, opened, sample=0)["skipped"] is True


# --------------------------------------------------------------------------- #
# End to end, on disk.
# --------------------------------------------------------------------------- #


def test_run_writes_rows_and_a_summary_carrying_its_provenance(tmp_path):
    backend, _ = make_backend(lengths={"json": 900, "mat": 100}, window=512)
    summary = remote_eval.run(
        endpoint=ENDPOINT,
        backend=backend,
        queries=QUERIES,
        formats=FORMATS,
        embed_model="stub-model",
        out_dir=tmp_path,
        run_id="TESTRUN",
        truncation_sample=None,
        chart=False,
        allow_embedding_mismatch=True,
    )

    with (tmp_path / "remote_retrieval_rows.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(QUERIES) * len(FORMATS)
    assert list(rows[0]) == list(remote_eval.CSV_FIELDS)
    assert rows[0]["retrieved"] == "X|A|Y"

    written = json.loads((tmp_path / "remote_retrieval_summary.json").read_text())
    assert written["endpoint"] == "http://stub-host:8000"
    assert written["collections"] == {"json": "nasa_cmr_json", "mat": "nasa_cmr_mat"}
    assert written["embed_model"] == "stub-model"
    assert written["embed_window"] == 512
    assert written["top_k"] == 10
    assert written["queries"]["sme"] == 1 and written["queries"]["synthetic"] == 1
    assert "gates" in written and "truncation" in written
    assert summary["ranking"]["order"] == ["json", "mat"]


def test_run_carries_the_caveats_a_reader_of_these_numbers_needs(tmp_path):
    backend, _ = make_backend(lengths={"json": 900, "mat": 100}, window=512)
    summary = remote_eval.run(
        endpoint=ENDPOINT, backend=backend, queries=QUERIES, formats=FORMATS,
        embed_model="stub-model", out_dir=tmp_path, truncation_sample=None,
        chart=False, allow_embedding_mismatch=True,
    )
    joined = " ".join(summary["caveats"])
    assert "not comparable with" in joined  # the different-index warning
    assert "hit-rate" in joined  # the synthetic-slice semantics
    assert "does not carry over" in joined  # the retrievability gate
    assert summary["caveats"][0].startswith("100%")  # truncation leads when it bites


def test_run_refuses_an_empty_query_set(tmp_path):
    backend, _ = make_backend()
    with pytest.raises(RemoteEvalError, match="No evaluation queries"):
        remote_eval.run(endpoint=ENDPOINT, backend=backend, queries=[], out_dir=tmp_path)


def test_run_draws_a_chart(tmp_path):
    backend, _ = make_backend()
    summary = remote_eval.run(
        endpoint=ENDPOINT, backend=backend, queries=QUERIES, formats=FORMATS,
        embed_model="stub-model", out_dir=tmp_path, truncation_sample=0,
        chart=True, allow_embedding_mismatch=True,
    )
    assert (tmp_path / "remote_retrieval.png").exists()
    assert summary["chart"].endswith("remote_retrieval.png")


# --------------------------------------------------------------------------- #
# Which renderings the deployment holds.
# --------------------------------------------------------------------------- #


def test_payload_is_identified_by_an_exact_byte_match_against_the_local_caches(monkeypatch, tmp_path):
    faceted = tmp_path / "format_cache" / "json"
    unfaceted = tmp_path / "format_cache_unfaceted" / "json"
    faceted.mkdir(parents=True)
    unfaceted.mkdir(parents=True)
    (faceted / "C1-X.json").write_text('{"title": "short"}')
    (unfaceted / "C1-X.json").write_text('{"EntryTitle": "the whole record"}')
    monkeypatch.setattr(remote_eval, "DATA_DIR", tmp_path)

    assert remote_eval.identify_payload('{"title": "short"}', "C1-X", "json") == "faceted"
    assert remote_eval.identify_payload('{"EntryTitle": "the whole record"}', "C1-X", "json") == "unfaceted"
    assert remote_eval.identify_payload("something else", "C1-X", "json") == "unknown"


def test_prose_documents_are_looked_up_with_the_txt_suffix(monkeypatch, tmp_path):
    """mat renders to .txt deliberately -- a .mat suffix would be a lie."""
    mat = tmp_path / "format_cache" / "mat"
    mat.mkdir(parents=True)
    (mat / "C1-X.txt").write_text("This dataset is titled Something.")
    monkeypatch.setattr(remote_eval, "DATA_DIR", tmp_path)

    assert remote_eval.identify_payload("This dataset is titled Something.", "C1-X", "mat") == "faceted"


def test_format_table_renders_missing_metrics_as_a_dash_never_zero():
    rows = remote_eval.score(QUERIES, RANKINGS, formats=FORMATS)
    summary = remote_eval.summarise(rows, embed_model="m", formats=FORMATS)
    summary["by_source"]["all"]["mat|remote:m"]["mrr"] = None

    table = remote_eval.format_table(summary, "m")
    assert "—" in table
