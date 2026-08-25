"""Tests for the retrieval metrics and the judge plumbing.

The retrieval metrics are the study's only exact numbers, so they are pinned
against hand-computed values rather than against themselves.
"""

from __future__ import annotations

import math

import pytest

from airm import evaluate
from airm.config import ModelSpec

# --------------------------------------------------------------------------- #
# Recall.
# --------------------------------------------------------------------------- #


def test_recall_is_over_the_expected_set_not_a_hit_rate():
    """Several SME queries expect nine collections; 1-of-9 is not a hit."""
    retrieved = ["A", "B", "C"]
    assert evaluate.recall_at_k(retrieved, ["A", "X", "Y"], 10) == pytest.approx(1 / 3)


def test_recall_respects_the_cutoff():
    retrieved = ["X", "X", "X", "X", "X", "A"]
    assert evaluate.recall_at_k(retrieved, ["A"], 5) == 0.0
    assert evaluate.recall_at_k(retrieved, ["A"], 10) == 1.0


def test_recall_of_a_perfect_retrieval_is_one():
    assert evaluate.recall_at_k(["A", "B"], ["A", "B"], 10) == 1.0


def test_recall_with_no_expected_records_is_nan_not_zero():
    """No ground truth means unmeasurable, which is not the same as failure."""
    assert math.isnan(evaluate.recall_at_k(["A"], [], 10))


def test_duplicate_retrievals_do_not_inflate_recall():
    assert evaluate.recall_at_k(["A", "A", "A"], ["A", "B"], 10) == 0.5


# --------------------------------------------------------------------------- #
# MRR and nDCG.
# --------------------------------------------------------------------------- #


def test_reciprocal_rank_uses_the_first_correct_hit():
    assert evaluate.reciprocal_rank(["X", "A"], ["A"]) == 0.5
    assert evaluate.reciprocal_rank(["A", "X"], ["A"]) == 1.0


def test_reciprocal_rank_is_zero_when_nothing_was_found():
    assert evaluate.reciprocal_rank(["X", "Y"], ["A"]) == 0.0


def test_ndcg_rewards_ranking_the_answer_higher():
    """Recall@10 cannot tell rank 1 from rank 10; nDCG is why it is reported."""
    top = evaluate.ndcg_at_k(["A"] + ["X"] * 9, ["A"], 10)
    bottom = evaluate.ndcg_at_k(["X"] * 9 + ["A"], ["A"], 10)
    assert top == 1.0
    assert bottom < top
    assert evaluate.recall_at_k(["A"] + ["X"] * 9, ["A"], 10) == evaluate.recall_at_k(
        ["X"] * 9 + ["A"], ["A"], 10
    )


def test_ndcg_of_a_perfect_ranking_is_one():
    assert evaluate.ndcg_at_k(["A", "B", "C"], ["A", "B", "C"], 10) == pytest.approx(1.0)


def test_ndcg_matches_the_hand_computed_value():
    # One expected record at rank 2: DCG = 1/log2(3), ideal = 1/log2(2) = 1.
    assert evaluate.ndcg_at_k(["X", "A"], ["A"], 10) == pytest.approx(1 / math.log2(3))


# --------------------------------------------------------------------------- #
# The bundle.
# --------------------------------------------------------------------------- #


def test_retrieval_metrics_reports_every_declared_column():
    m = evaluate.retrieval_metrics(["A", "B"], ["B"])
    assert set(m) == {"recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"}


def test_retrieval_metrics_agree_with_the_individual_functions():
    retrieved, expected = ["X", "A", "B"], ["A", "B"]
    m = evaluate.retrieval_metrics(retrieved, expected)
    assert m["recall@10"] == evaluate.recall_at_k(retrieved, expected, 10)
    assert m["mrr"] == evaluate.reciprocal_rank(retrieved, expected)


# --------------------------------------------------------------------------- #
# The judge.
# --------------------------------------------------------------------------- #


def test_judge_routes_through_the_logger(tmp_path, monkeypatch):
    """Judge calls produce the reported scores; they must not bypass the log."""
    from airm import llm

    logger = llm.CallLogger(run_id="testrun", root=tmp_path)
    monkeypatch.setattr(
        llm,
        "_call_ollama",
        lambda spec, msgs, **kw: llm.LLMResponse(
            text="graded", provider="ollama", model="m", prompt_tokens=5, completion_tokens=1
        ),
    )

    judge = evaluate.LoggedJudge(ModelSpec("ollama", "m"), logger)
    judge.set_context(fmt="yaml", query_id="syn-001")
    assert judge.generate("grade this") == "graded"

    records = logger.read(llm.PURPOSE_JUDGE)
    assert len(records) == 1
    assert records[0]["fmt"] == "yaml"
    assert records[0]["messages"][0]["content"] == "grade this"


def test_judge_name_identifies_provider_and_model():
    judge = evaluate.LoggedJudge(ModelSpec("openai", "gpt-5-mini"))
    assert judge.get_model_name() == "openai:gpt-5-mini"


@pytest.mark.parametrize(
    "text",
    [
        '{"score": 0.8, "reason": "good"}',
        '```json\n{"score": 0.8, "reason": "good"}\n```',
        'Sure! Here you go:\n{"score": 0.8, "reason": "good"}\nHope that helps.',
    ],
)
def test_judge_output_is_parsed_through_common_model_wrappers(text):
    """Local models wrap JSON in fences and chatter; that is not a grading failure."""

    class Schema:
        def __init__(self, score, reason):
            self.score, self.reason = score, reason

    parsed = evaluate._coerce_schema(text, Schema)
    assert parsed.score == 0.8


def test_unparseable_judge_output_falls_back_to_the_raw_text(monkeypatch):
    """DeepEval's own trimAndLoadJson gets a shot before we call it an error."""
    from airm import llm

    monkeypatch.setattr(
        llm,
        "_call_ollama",
        lambda spec, msgs, **kw: llm.LLMResponse(
            text="score is high, no json here", provider="ollama", model="m"
        ),
    )

    class Schema:
        def __init__(self, **kw):
            pass

    judge = evaluate.LoggedJudge(ModelSpec("ollama", "m"))
    assert judge.generate("grade", schema=Schema) == "score is high, no json here"


def test_judge_exposes_the_attributes_deepeval_reads_directly():
    """metrics/utils touches model.name outside any try block."""
    judge = evaluate.LoggedJudge(ModelSpec("openai", "gpt-5-mini"))
    assert judge.name == "openai:gpt-5-mini"
    assert judge.model is judge.spec


def test_a_judge_failure_is_recorded_as_an_error_not_a_zero(monkeypatch):
    """A broken judge and a bad answer are different events."""

    class Exploding:
        score = None

        def measure(self, case):
            raise RuntimeError("judge unreachable")

    class Fine:
        score = 0.9
        reason = "ok"

        def measure(self, case):
            return None

    monkeypatch.setattr(
        evaluate, "make_test_case", lambda **kw: object()
    )
    result = evaluate.score_with_deepeval(
        question="q",
        answer="a",
        expected="e",
        contexts=["c"],
        judge=object(),
        metrics={"faithfulness": Exploding(), "answer_relevancy": Fine()},
    )
    assert result.scores["faithfulness"] is None
    assert "judge unreachable" in result.errors["faithfulness"]
    assert result.scores["answer_relevancy"] == 0.9


def test_the_five_declared_metrics_are_the_studys_five():
    assert set(evaluate.METRIC_NAMES) == {
        "correctness",
        "faithfulness",
        "answer_relevancy",
        "contextual_relevancy",
        "contextual_recall",
    }


def test_judge_description_records_who_graded(monkeypatch):
    desc = evaluate.judge_description()
    assert desc["provider"] and desc["model"]
    assert set(desc["metrics"]) == set(evaluate.METRIC_NAMES)
