"""Metrics for Experiment 2.

Two families, kept separate because they answer different questions and have
very different trust levels:

**Retrieval metrics** (Recall@k, MRR, nDCG) are computed from concept-ids against
ground truth. No model is involved, so they are exact, free, and immune to judge
drift. They measure whether the *representation* surfaces the right records.

**DeepEval metrics** (Correctness, Faithfulness, Answer Relevancy, Contextual
Relevancy, Contextual Recall) are LLM-graded and measure whether the model then
*used* what it retrieved. They are the study's stated metrics, and they are also
the softest numbers in it -- so the judge is held fixed across every cell, and
every judge call is logged like any other.

The judge is deliberately not drawn from the model matrix. If the model under
test also graded its own output, the model axis would be confounded with the
grading standard, and a "better" model would be partly measuring its own
leniency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .config import RECALL_AT, JUDGE_MODEL, JUDGE_PROVIDER, ModelSpec, judge_spec
from .llm import PURPOSE_JUDGE, CallLogger, complete

# --------------------------------------------------------------------------- #
# Retrieval metrics -- exact, no model involved.
# --------------------------------------------------------------------------- #


def recall_at_k(retrieved: Sequence[str], expected: Iterable[str], k: int) -> float:
    """Fraction of expected records found in the top ``k``.

    Set recall, not hit-rate: several SME queries expect up to nine collections,
    and a hit-rate would score "found one of nine" as a perfect result.
    """
    expected = set(expected)
    if not expected:
        return float("nan")
    return len(expected & set(retrieved[:k])) / len(expected)


def reciprocal_rank(retrieved: Sequence[str], expected: Iterable[str]) -> float:
    """1 / rank of the first correct record; 0 if none was retrieved."""
    expected = set(expected)
    for i, cid in enumerate(retrieved, start=1):
        if cid in expected:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], expected: Iterable[str], k: int) -> float:
    """Binary-relevance nDCG@k.

    Rewards putting correct records *near the top*, which Recall@k ignores
    entirely -- a format that ranks the answer 1st and one that ranks it 10th
    score identically on Recall@10 but very differently here.
    """
    expected = set(expected)
    if not expected:
        return float("nan")
    dcg = sum(1.0 / math.log2(i + 1) for i, cid in enumerate(retrieved[:k], start=1) if cid in expected)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(expected), k) + 1))
    return dcg / ideal if ideal else float("nan")


def retrieval_metrics(
    retrieved: Sequence[str],
    expected: Iterable[str],
    *,
    ks: tuple[int, ...] = RECALL_AT,
    ndcg_k: int = 10,
) -> dict[str, float]:
    expected = list(expected)
    out = {f"recall@{k}": recall_at_k(retrieved, expected, k) for k in ks}
    out["mrr"] = reciprocal_rank(retrieved, expected)
    out[f"ndcg@{ndcg_k}"] = ndcg_at_k(retrieved, expected, ndcg_k)
    return out


# --------------------------------------------------------------------------- #
# The logged judge.
# --------------------------------------------------------------------------- #


class LoggedJudge:
    """A ``DeepEvalBaseLLM`` that routes every judge call through :mod:`airm.llm`.

    DeepEval would otherwise talk to OpenAI directly, and the judge calls -- the
    ones that actually produce the reported scores -- would be the only LLM
    traffic in the study with no record of what was asked or answered.
    """

    def __init__(self, spec: ModelSpec | None = None, logger: CallLogger | None = None):
        self.spec = spec or judge_spec()
        self.logger = logger
        self._meta: dict = {}
        # DeepEvalBaseLLM.__init__ is bypassed (its signature wants a model
        # string), but parts of DeepEval read ``model.name``/``model.model``
        # directly -- e.g. the multimodal support checks in metrics/utils --
        # so set what it would have set.
        self.name = self.spec.key
        self.model = self.spec

    # DeepEval calls these.
    def load_model(self):
        return self.spec

    def get_model_name(self) -> str:
        return self.spec.key

    def set_context(self, **meta) -> None:
        """Tag subsequent judge calls with the cell they belong to."""
        self._meta = meta

    def generate(self, prompt: str, schema=None, **kwargs) -> str:
        response = complete(
            self.spec,
            [{"role": "user", "content": prompt}],
            purpose=PURPOSE_JUDGE,
            logger=self.logger,
            temperature=0.0,
            **self._meta,
        )
        if schema is not None:
            try:
                return _coerce_schema(response.text, schema)
            except Exception:  # noqa: BLE001 - defer to DeepEval's own parser
                # DeepEval checks isinstance(result, schema) and runs anything
                # else through its trimAndLoadJson fallback. Raising here would
                # pre-empt that: a response our parser cannot handle becomes a
                # metric error where DeepEval might have parsed it fine.
                return response.text
        return response.text

    async def a_generate(self, prompt: str, schema=None, **kwargs) -> str:
        return self.generate(prompt, schema=schema, **kwargs)


def _coerce_schema(text: str, schema):
    """Parse a judge response into DeepEval's expected pydantic schema."""
    import json

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        cleaned = cleaned[4:] if cleaned.startswith("json") else cleaned
        cleaned = cleaned.strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise
        data = json.loads(cleaned[start : end + 1])
    return schema(**data)


def _deepeval_judge(spec: ModelSpec | None, logger: CallLogger | None):
    """Wrap :class:`LoggedJudge` in DeepEval's base class at call time.

    Imported lazily so the retrieval metrics -- and the whole offline test suite
    -- do not depend on DeepEval being importable.
    """
    from deepeval.models import DeepEvalBaseLLM

    class _Judge(DeepEvalBaseLLM, LoggedJudge):
        def __init__(self, spec, logger):
            LoggedJudge.__init__(self, spec, logger)

    return _Judge(spec, logger)


# --------------------------------------------------------------------------- #
# DeepEval metrics.
# --------------------------------------------------------------------------- #

CORRECTNESS_CRITERIA = (
    "Determine whether the actual output correctly identifies the NASA Earth-science "
    "collections that answer the question, judged against the expected output. "
    "Naming a collection that genuinely measures the requested phenomenon over the "
    "requested region and period is correct. Omitting an expected collection is a "
    "partial failure; asserting a collection that does not measure the requested "
    "phenomenon is a full failure. Do not penalise differences in wording or ordering."
)

#: The five metrics the study reports, in the order they are stated.
METRIC_NAMES = (
    "correctness",
    "faithfulness",
    "answer_relevancy",
    "contextual_relevancy",
    "contextual_recall",
)


def build_metrics(judge, *, threshold: float = 0.5):
    """The five DeepEval metrics, all sharing one fixed judge."""
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
        GEval,
    )
    from deepeval.test_case import LLMTestCaseParams

    common = {"model": judge, "threshold": threshold, "async_mode": False}
    return {
        "correctness": GEval(
            name="Correctness",
            criteria=CORRECTNESS_CRITERIA,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            **common,
        ),
        "faithfulness": FaithfulnessMetric(**common),
        "answer_relevancy": AnswerRelevancyMetric(**common),
        "contextual_relevancy": ContextualRelevancyMetric(**common),
        "contextual_recall": ContextualRecallMetric(**common),
    }


def make_test_case(
    *, question: str, answer: str, expected: str, contexts: list[str]
):
    from deepeval.test_case import LLMTestCase

    return LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=expected,
        retrieval_context=contexts,
    )


@dataclass
class JudgeResult:
    scores: dict[str, float | None] = field(default_factory=dict)
    reasons: dict[str, str | None] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


def score_with_deepeval(
    *,
    question: str,
    answer: str,
    expected: str,
    contexts: list[str],
    judge=None,
    logger: CallLogger | None = None,
    metrics=None,
    **meta,
) -> JudgeResult:
    """Run all five DeepEval metrics on one cell.

    A metric that raises is recorded as an error with a ``None`` score rather
    than as a zero: a judge failure and a genuinely bad answer are different
    events, and averaging a failure in as zero would quietly drag a cell down.
    """
    judge = judge or _deepeval_judge(None, logger)
    if hasattr(judge, "set_context"):
        judge.set_context(**meta)
    metrics = metrics or build_metrics(judge)
    case = make_test_case(question=question, answer=answer, expected=expected, contexts=contexts)

    result = JudgeResult()
    for name, metric in metrics.items():
        try:
            metric.measure(case)
            result.scores[name] = metric.score
            result.reasons[name] = getattr(metric, "reason", None)
        except Exception as exc:  # noqa: BLE001 - judge failures are data
            result.scores[name] = None
            result.reasons[name] = None
            result.errors[name] = f"{type(exc).__name__}: {exc}"
    return result


def judge_description() -> dict:
    """Recorded in every run summary -- which judge graded the soft metrics."""
    return {"provider": JUDGE_PROVIDER, "model": JUDGE_MODEL, "metrics": list(METRIC_NAMES)}
