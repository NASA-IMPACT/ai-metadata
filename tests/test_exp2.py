"""Tests for the Experiment 2 matrix runner and the report assembler.

The runner is stubbed at the retrieval and LLM boundaries: what is under test is
the matrix bookkeeping — checkpointing, resume, retrieval sharing across models,
and failure handling — not the providers.
"""

from __future__ import annotations

import json

import pytest

from airm import exp2, llm, report
from airm.config import ModelSpec
from airm.queries import Query

QUERIES = [
    Query("q-1", "How did sea ice change?", ["A"], "synthetic", "OCEANS", ["Sea Ice Daily"]),
    Query("q-2", "What about soil moisture?", ["B", "C"], "sme", None, ["Soil Moisture L3"]),
]
MODELS = [ModelSpec("ollama", "m1"), ModelSpec("ollama", "m2")]
FORMATS = ("json", "mat")


@pytest.fixture
def stubbed(monkeypatch):
    """Deterministic retrieval and answers; no judge."""
    calls = {"retrieve": 0, "ask": 0}

    def fake_query(fmt, text, k=10, **kw):
        calls["retrieve"] += 1
        ranked = ["A", "B", "X"] if fmt == "json" else ["X", "A", "B"]
        return [
            {"rank": i + 1, "concept_id": cid, "document": f"{fmt} doc for {cid}", "topic": "OCEANS", "distance": 0.1 * i}
            for i, cid in enumerate(ranked[:k])
        ]

    def fake_ask(spec, prompt, **kwargs):
        calls["ask"] += 1
        return llm.LLMResponse(
            text=f"{spec.model} says the relevant dataset is A.",
            provider=spec.provider,
            model=spec.model,
            prompt_tokens=100,
            completion_tokens=20,
            latency_s=0.5,
        )

    monkeypatch.setattr(exp2.index, "query", fake_query)
    monkeypatch.setattr(exp2, "ask", fake_ask)
    monkeypatch.setattr(exp2, "resolve_models", lambda specs: (list(specs), []))
    return calls


# --------------------------------------------------------------------------- #
# The matrix.
# --------------------------------------------------------------------------- #


def test_every_cell_in_the_matrix_is_produced(stubbed, tmp_path):
    result = exp2.run(
        formats=FORMATS, models=MODELS, queries=QUERIES, judge=False, out_dir=tmp_path, run_id="r"
    )
    assert result["cells"] == len(FORMATS) * len(MODELS) * len(QUERIES)


def test_retrieval_runs_once_per_format_and_query_not_once_per_model(stubbed, tmp_path):
    """Retrieval depends only on the format; re-running it per model is waste."""
    exp2.run(
        formats=FORMATS, models=MODELS, queries=QUERIES, judge=False, out_dir=tmp_path, run_id="r"
    )
    assert stubbed["retrieve"] == len(FORMATS) * len(QUERIES)
    assert stubbed["ask"] == len(FORMATS) * len(QUERIES) * len(MODELS)


def test_retrieval_metrics_are_computed_from_ground_truth(stubbed, tmp_path):
    exp2.run(
        formats=("json",), models=MODELS[:1], queries=QUERIES[:1], judge=False,
        out_dir=tmp_path, run_id="r",
    )
    row = json.loads((tmp_path / "exp2_cells.jsonl").read_text().splitlines()[0])
    assert row["retrieved"] == ["A", "B", "X"]
    assert row["expected"] == ["A"]
    assert row["metrics"]["recall@10"] == 1.0
    assert row["metrics"]["mrr"] == 1.0


def test_format_changes_the_ranking_and_therefore_the_metrics(stubbed, tmp_path):
    result = exp2.run(
        formats=FORMATS, models=MODELS[:1], queries=QUERIES[:1], judge=False,
        out_dir=tmp_path, run_id="r",
    )
    summary = result["summary"]
    assert summary["json|ollama:m1"]["mrr"] == 1.0
    assert summary["mat|ollama:m1"]["mrr"] == 0.5


# --------------------------------------------------------------------------- #
# Checkpointing and resume.
# --------------------------------------------------------------------------- #


def test_cells_are_checkpointed_as_they_complete(stubbed, tmp_path):
    exp2.run(
        formats=FORMATS, models=MODELS, queries=QUERIES, judge=False, out_dir=tmp_path, run_id="r"
    )
    lines = (tmp_path / "exp2_cells.jsonl").read_text().strip().splitlines()
    assert len(lines) == len(FORMATS) * len(MODELS) * len(QUERIES)
    for line in lines:
        json.loads(line)


def test_a_resumed_run_does_no_duplicate_work(stubbed, tmp_path):
    """A matrix this size will be interrupted; restarting from zero is fatal."""
    exp2.run(
        formats=FORMATS, models=MODELS, queries=QUERIES, judge=False, out_dir=tmp_path, run_id="r"
    )
    before = stubbed["ask"]

    result = exp2.run(
        formats=FORMATS, models=MODELS, queries=QUERIES, judge=False, out_dir=tmp_path, run_id="r"
    )
    assert stubbed["ask"] == before, "resume re-asked completed cells"
    assert result["cells"] == len(FORMATS) * len(MODELS) * len(QUERIES)


def test_resume_fills_only_the_missing_cells(stubbed, tmp_path):
    exp2.run(
        formats=("json",), models=MODELS[:1], queries=QUERIES, judge=False,
        out_dir=tmp_path, run_id="r",
    )
    before = stubbed["ask"]
    exp2.run(
        formats=FORMATS, models=MODELS[:1], queries=QUERIES, judge=False,
        out_dir=tmp_path, run_id="r",
    )
    assert stubbed["ask"] - before == len(QUERIES)  # only the "mat" cells


def test_resume_skips_the_vector_searches_for_completed_cells(stubbed, tmp_path):
    """Resuming a finished matrix must not redo every retrieval first."""
    exp2.run(
        formats=FORMATS, models=MODELS, queries=QUERIES, judge=False, out_dir=tmp_path, run_id="r"
    )
    before = stubbed["retrieve"]
    exp2.run(
        formats=FORMATS, models=MODELS, queries=QUERIES, judge=False, out_dir=tmp_path, run_id="r"
    )
    assert stubbed["retrieve"] == before, "fully-checkpointed resume re-ran retrieval"


def test_an_unavailable_judge_fails_fast_not_per_cell(stubbed, tmp_path, monkeypatch):
    """A dead judge must stop the run up front, not 401 through every metric."""
    monkeypatch.setattr(
        exp2, "resolve_models",
        lambda specs: (
            (list(specs), []) if any(s.provider == "ollama" for s in specs)
            else ([], ["openai: cannot list models (401)"])
        ),
    )
    with pytest.raises(llm.LLMError, match="judge is unavailable.*--no-judge"):
        exp2.run(
            formats=("json",), models=MODELS[:1], queries=QUERIES[:1], judge=True,
            out_dir=tmp_path, run_id="r",
        )
    # No cells were attempted, so nothing was checkpointed.
    assert not (tmp_path / "exp2_cells.jsonl").exists()


def test_summary_reports_answered_and_total_denominators_separately(
    stubbed, tmp_path, monkeypatch
):
    """Retrieval means cover all cells; judge/answer means only the answered."""
    calls = {"n": 0}

    def sometimes(spec, prompt, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise llm.LLMError("model unreachable")
        return llm.LLMResponse(
            text="answered", provider=spec.provider, model=spec.model,
            prompt_tokens=100, completion_tokens=20,
        )

    monkeypatch.setattr(exp2, "ask", sometimes)
    result = exp2.run(
        formats=("json",), models=MODELS[:1], queries=QUERIES, judge=False,
        out_dir=tmp_path, run_id="r",
    )
    row = result["summary"]["json|ollama:m1"]
    assert row["n"] == 2
    assert row["n_answered"] == 1


def test_resume_can_be_disabled(stubbed, tmp_path):
    exp2.run(
        formats=("json",), models=MODELS[:1], queries=QUERIES[:1], judge=False,
        out_dir=tmp_path, run_id="r",
    )
    before = stubbed["ask"]
    exp2.run(
        formats=("json",), models=MODELS[:1], queries=QUERIES[:1], judge=False,
        out_dir=tmp_path, run_id="r", resume=False,
    )
    assert stubbed["ask"] > before


# --------------------------------------------------------------------------- #
# Failures.
# --------------------------------------------------------------------------- #


def test_a_failed_answer_is_recorded_as_an_error_cell(monkeypatch, tmp_path, stubbed):
    def boom(*args, **kwargs):
        raise llm.LLMError("model unreachable")

    monkeypatch.setattr(exp2, "ask", boom)
    result = exp2.run(
        formats=("json",), models=MODELS[:1], queries=QUERIES[:1], judge=False,
        out_dir=tmp_path, run_id="r",
    )
    row = json.loads((tmp_path / "exp2_cells.jsonl").read_text().splitlines()[0])
    assert "model unreachable" in row["error"]
    # Retrieval metrics still stand: they do not depend on the model.
    assert row["metrics"]["recall@10"] == 1.0
    assert result["cells"] == 1


def test_unavailable_models_are_reported_in_the_summary(monkeypatch, tmp_path, stubbed):
    monkeypatch.setattr(
        exp2, "resolve_models", lambda specs: ([], ["openai: model 'gpt-5' not available"])
    )
    result = exp2.run(
        formats=("json",), models=MODELS, queries=QUERIES, judge=False,
        out_dir=tmp_path, run_id="r",
    )
    assert result["cells"] == 0
    assert "gpt-5" in result["model_problems"][0]


# --------------------------------------------------------------------------- #
# Prompting.
# --------------------------------------------------------------------------- #


def test_the_answer_prompt_numbers_every_candidate():
    prompt = exp2.answer_prompt("Q?", ["doc one", "doc two"])
    assert "[1]" in prompt and "[2]" in prompt
    assert "doc one" in prompt and "doc two" in prompt


def test_the_expected_answer_prefers_titles_over_concept_ids():
    assert "Sea Ice Daily" in exp2.expected_answer(QUERIES[0])
    bare = Query("q-3", "t", ["C9-X"], "sme", None, [])
    assert "C9-X" in exp2.expected_answer(bare)


def test_the_answer_system_prompt_forbids_inventing_datasets():
    assert "ONLY the candidates" in exp2.ANSWER_SYSTEM


# --------------------------------------------------------------------------- #
# Reporting.
# --------------------------------------------------------------------------- #


def test_report_renders_experiment_one(tmp_path):
    summary = {
        "records": 500,
        "tokenizers": {"hf": "Qwen/Qwen3-8B"},
        "summary_tiktoken": [
            {"format": "json", "n": 500, "mean_tokens": 587, "median_tokens": 516,
             "ci_low": 558, "ci_high": 617, "ratio_vs_json": 1.0, "ratio_ci_low": 1.0,
             "ratio_ci_high": 1.0, "pct_vs_json": 0.0},
            {"format": "mat", "n": 500, "mean_tokens": 574, "median_tokens": 503,
             "ci_low": 545, "ci_high": 604, "ratio_vs_json": 0.975, "ratio_ci_low": 0.973,
             "ratio_ci_high": 0.977, "pct_vs_json": -2.5},
        ],
        "summary_hf": [],
        "reference_raw_umm": {"n": 500, "mean_tokens": 1991, "median_tokens": 1770},
        "per_topic_tiktoken": {},
    }
    text = report.exp1_section(summary)
    assert "Metadata-as-Text" in text and "-2.5%" in text
    assert "1,991" in text and "3.5×" in text


def test_report_says_when_experiment_two_has_not_run():
    assert "No cells" in report.exp2_section({"summary": {}})


def test_report_flags_when_models_disagree_on_the_best_format():
    """This is the study's core question, so it must be stated explicitly."""
    summary = {
        "json|openai:gpt-5": {"n": 10, "recall@10": 0.9, "correctness": 0.8},
        "mat|openai:gpt-5": {"n": 10, "recall@10": 0.7, "correctness": 0.6},
        "json|ollama:qwen3.6": {"n": 10, "recall@10": 0.6, "correctness": 0.5},
        "mat|ollama:qwen3.6": {"n": 10, "recall@10": 0.8, "correctness": 0.7},
    }
    lines = "\n".join(report._winner_lines(summary))
    assert "models disagree" in lines
    assert "gpt-5 → json" in lines and "qwen3.6 → mat" in lines


def test_report_notes_agreement_when_one_format_wins_everywhere():
    summary = {
        "json|openai:gpt-5": {"n": 10, "recall@10": 0.9},
        "mat|openai:gpt-5": {"n": 10, "recall@10": 0.7},
        "json|ollama:qwen3.6": {"n": 10, "recall@10": 0.8},
        "mat|ollama:qwen3.6": {"n": 10, "recall@10": 0.6},
    }
    assert "all models agree" in "\n".join(report._winner_lines(summary))


def test_report_will_not_claim_model_dependence_from_one_model():
    summary = {"json|openai:gpt-5": {"n": 10, "recall@10": 0.9}}
    assert "not yet answerable" in "\n".join(report._winner_lines(summary))


def test_a_missing_metric_renders_as_a_dash_not_a_zero():
    assert report._fmt(None) == report.MISSING
    assert report._fmt(float("nan")) == report.MISSING
    assert report._fmt(0.5) == "0.500"


def test_build_writes_findings_even_with_no_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "latest_run", lambda *a, **k: None)
    path = report.build(tmp_path / "FINDINGS.md")
    text = path.read_text()
    assert "Not run yet" in text
    assert text.startswith("# Findings")
