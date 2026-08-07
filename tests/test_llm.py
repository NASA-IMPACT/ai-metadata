"""Tests for the LLM wrapper and the call log.

The log is the study's audit trail. A missing record means a number in the
results table cannot be traced to the call that produced it, so the tests care
most about calls being logged on *failure* paths too.
"""

from __future__ import annotations

import json
import threading

import pytest

from airm import llm
from airm.config import ModelSpec

SPEC = ModelSpec("openai", "test-model")


@pytest.fixture
def logger(tmp_path):
    return llm.CallLogger(run_id="testrun", root=tmp_path)


def _response(text="ok", **kw):
    return llm.LLMResponse(
        text=text, provider="openai", model="test-model", prompt_tokens=10, completion_tokens=3, **kw
    )


# --------------------------------------------------------------------------- #
# Logging.
# --------------------------------------------------------------------------- #


def test_a_successful_call_is_logged_with_prompt_and_response(logger, monkeypatch):
    monkeypatch.setattr(llm, "_call_openai", lambda spec, msgs, **kw: _response("hello"))
    llm.ask(SPEC, "hi there", purpose=llm.PURPOSE_ANSWER, logger=logger)

    records = logger.read(llm.PURPOSE_ANSWER)
    assert len(records) == 1
    record = records[0]
    assert record["messages"][-1]["content"] == "hi there"
    assert record["response"] == "hello"
    assert record["prompt_tokens"] == 10
    assert record["total_tokens"] == 13
    assert record["error"] is None


def test_a_failing_call_is_still_logged(logger, monkeypatch):
    """A silent failure is the one thing the audit trail must never allow."""

    def boom(*args, **kwargs):
        raise RuntimeError("upstream exploded")

    monkeypatch.setattr(llm, "_call_openai", boom)
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)

    with pytest.raises(llm.LLMError):
        llm.ask(SPEC, "hi", purpose=llm.PURPOSE_ANSWER, logger=logger, max_retries=2)

    records = logger.read(llm.PURPOSE_ANSWER)
    assert len(records) == 1
    assert records[0]["response"] is None
    assert "upstream exploded" in records[0]["error"]


def test_metadata_is_written_through_so_a_row_can_be_traced(logger, monkeypatch):
    monkeypatch.setattr(llm, "_call_openai", lambda *a, **k: _response())
    llm.ask(SPEC, "q", purpose=llm.PURPOSE_ANSWER, logger=logger, fmt="toon", query_id="syn-007")
    record = logger.read(llm.PURPOSE_ANSWER)[0]
    assert record["fmt"] == "toon"
    assert record["query_id"] == "syn-007"


def test_purposes_are_kept_in_separate_files(logger, monkeypatch):
    monkeypatch.setattr(llm, "_call_openai", lambda *a, **k: _response())
    for purpose in llm.PURPOSES:
        llm.ask(SPEC, "q", purpose=purpose, logger=logger)
    assert logger.counts() == {p: 1 for p in llm.PURPOSES}
    assert logger.path_for(llm.PURPOSE_JUDGE).name == "judge.jsonl"


def test_judge_calls_have_their_own_stream(logger, monkeypatch):
    """Judge traffic must be auditable separately from answers."""
    monkeypatch.setattr(llm, "_call_openai", lambda *a, **k: _response())
    llm.ask(SPEC, "grade this", purpose=llm.PURPOSE_JUDGE, logger=logger)
    assert logger.counts()[llm.PURPOSE_JUDGE] == 1
    assert logger.counts()[llm.PURPOSE_ANSWER] == 0


def test_log_is_append_only_across_calls(logger, monkeypatch):
    monkeypatch.setattr(llm, "_call_openai", lambda *a, **k: _response())
    for i in range(5):
        llm.ask(SPEC, f"q{i}", purpose=llm.PURPOSE_ANSWER, logger=logger)
    assert len(logger.read(llm.PURPOSE_ANSWER)) == 5


def test_concurrent_writes_do_not_interleave(logger, monkeypatch):
    """Experiment 2 fans out; a torn line destroys the record it was written for."""
    monkeypatch.setattr(llm, "_call_openai", lambda *a, **k: _response("x" * 500))

    def call(i):
        llm.ask(SPEC, f"q{i}", purpose=llm.PURPOSE_ANSWER, logger=logger)

    threads = [threading.Thread(target=call, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = logger.path_for(llm.PURPOSE_ANSWER).read_text().strip().splitlines()
    assert len(lines) == 16
    for line in lines:
        json.loads(line)  # every line is a complete JSON object


# --------------------------------------------------------------------------- #
# Retries.
# --------------------------------------------------------------------------- #


def test_a_transient_failure_is_retried_and_the_attempt_count_recorded(logger, monkeypatch):
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("rate limited")
        return _response("recovered")

    monkeypatch.setattr(llm, "_call_openai", flaky)
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)

    response = llm.ask(SPEC, "q", purpose=llm.PURPOSE_ANSWER, logger=logger)
    assert response.text == "recovered"
    assert response.attempts == 3
    assert logger.read(llm.PURPOSE_ANSWER)[0]["attempts"] == 3


class _StatusError(RuntimeError):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def test_auth_and_bad_request_errors_are_not_retried(logger, monkeypatch):
    """Retrying a 401 four times multiplies the same failure and wastes ~20s."""
    calls = {"n": 0}

    def unauthorized(*args, **kwargs):
        calls["n"] += 1
        raise _StatusError(401)

    monkeypatch.setattr(llm, "_call_openai", unauthorized)
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)

    with pytest.raises(llm.LLMError, match="1 attempt"):
        llm.ask(SPEC, "q", purpose=llm.PURPOSE_ANSWER, logger=logger)
    assert calls["n"] == 1
    assert logger.read(llm.PURPOSE_ANSWER)[0]["attempts"] == 1


@pytest.mark.parametrize("status", [408, 429, 500, 503])
def test_transient_statuses_are_retried(status, monkeypatch):
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise _StatusError(status)
        return _response()

    monkeypatch.setattr(llm, "_call_openai", flaky)
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)
    assert llm.ask(SPEC, "q", purpose=llm.PURPOSE_ANSWER).attempts == 2


def test_errors_without_a_status_code_are_retried():
    assert llm._is_retryable(RuntimeError("connection reset"))
    assert not llm._is_retryable(_StatusError(404))
    assert llm._is_retryable(_StatusError(429))


# --------------------------------------------------------------------------- #
# Cost.
# --------------------------------------------------------------------------- #


def test_cost_is_none_when_no_price_is_configured(monkeypatch):
    """A fabricated zero would let unpriced models win every cost comparison."""
    monkeypatch.setattr(llm, "load_prices", lambda *a, **k: {})
    assert llm.cost_of("test-model", 1000, 500) is None


def test_cost_is_computed_when_a_price_is_configured(monkeypatch):
    monkeypatch.setattr(
        llm, "load_prices", lambda *a, **k: {"test-model": {"input": 1.0, "output": 4.0}}
    )
    # 1000 in @ $1/Mtok + 500 out @ $4/Mtok
    assert llm.cost_of("test-model", 1000, 500) == pytest.approx((1000 + 2000) / 1_000_000)


def test_missing_token_counts_yield_no_cost(monkeypatch):
    monkeypatch.setattr(
        llm, "load_prices", lambda *a, **k: {"test-model": {"input": 1.0, "output": 4.0}}
    )
    assert llm.cost_of("test-model", None, 500) is None


def test_a_corrupt_price_file_is_ignored_rather_than_crashing(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text("{not json")
    assert llm.load_prices(path) == {}


# --------------------------------------------------------------------------- #
# Model availability.
# --------------------------------------------------------------------------- #


def test_available_models_are_kept_and_missing_ones_reported(monkeypatch):
    monkeypatch.setattr(llm, "available_openai_models", lambda: ["gpt-5", "gpt-5-mini"])
    monkeypatch.setattr(llm, "available_ollama_models", lambda: ["qwen3.6:latest"])

    specs = [
        ModelSpec("openai", "gpt-5"),
        ModelSpec("openai", "gpt-4-retired"),
        ModelSpec("ollama", "qwen3.6:latest"),
    ]
    ok, problems = llm.resolve_models(specs)
    assert [s.model for s in ok] == ["gpt-5", "qwen3.6:latest"]
    assert len(problems) == 1 and "gpt-4-retired" in problems[0]


def test_an_unreachable_provider_is_reported_not_silently_skipped(monkeypatch):
    def boom():
        raise RuntimeError("401 invalid key")

    monkeypatch.setattr(llm, "available_openai_models", boom)
    ok, problems = llm.resolve_models([ModelSpec("openai", "gpt-5")])
    assert ok == []
    assert any("cannot list models" in p and "401" in p for p in problems)


def test_missing_api_key_says_which_experiment_still_works(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(llm.LLMError, match="Experiment 1 runs without it"):
        llm.openai_client()


# --------------------------------------------------------------------------- #
# Request shaping.
# --------------------------------------------------------------------------- #


def test_json_mode_maps_to_each_provider_convention(monkeypatch):
    seen = {}

    def capture_openai(spec, msgs, **kw):
        seen["openai"] = kw
        return _response()

    def capture_ollama(spec, msgs, **kw):
        seen["ollama"] = kw
        return llm.LLMResponse(text="{}", provider="ollama", model="m")

    monkeypatch.setattr(llm, "_call_openai", capture_openai)
    monkeypatch.setattr(llm, "_call_ollama", capture_ollama)

    llm.ask(SPEC, "q", purpose=llm.PURPOSE_ANSWER, json_mode=True)
    llm.ask(ModelSpec("ollama", "m"), "q", purpose=llm.PURPOSE_ANSWER, json_mode=True)

    assert seen["openai"]["response_format"] == {"type": "json_object"}
    assert seen["ollama"]["json_mode"] is True


def test_system_prompt_precedes_the_user_turn(monkeypatch):
    captured = {}

    def capture(spec, msgs, **kw):
        captured["messages"] = msgs
        return _response()

    monkeypatch.setattr(llm, "_call_openai", capture)
    llm.ask(SPEC, "user text", system="system text", purpose=llm.PURPOSE_ANSWER)

    assert [m["role"] for m in captured["messages"]] == ["system", "user"]
    assert captured["messages"][0]["content"] == "system text"


def test_a_prompt_without_a_system_message_has_only_the_user_turn(monkeypatch):
    captured = {}

    def capture(spec, msgs, **kw):
        captured["messages"] = msgs
        return _response()

    monkeypatch.setattr(llm, "_call_openai", capture)
    llm.ask(SPEC, "user text", purpose=llm.PURPOSE_ANSWER)

    assert [m["role"] for m in captured["messages"]] == ["user"]
