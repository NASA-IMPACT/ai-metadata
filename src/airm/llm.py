"""One interface over OpenAI and Ollama, with every call logged.

The logging is not incidental. Experiment 2 makes thousands of calls across
`format x model x query`, and three different *kinds* of call -- query
generation, answering, and DeepEval judging. When a number looks wrong later,
the only way to find out why is to read the exact prompt and the exact response
that produced it. So:

* every call goes through :func:`complete`;
* every call appends a JSONL record to ``logs/llm/<run_id>/<purpose>.jsonl``
  with the full prompt, the full response, token counts, latency and retries;
* **DeepEval judge calls route through here too** (see
  :class:`airm.evaluate.LoggedJudge`), so nothing evaluative happens off-log.

Cost
----
Token counts are exact and always recorded. Dollar costs are only computed when
a price is configured, because a wrong price is worse than an absent one: local
Ollama models have no list price, and a zero would let them win every
cost-per-accuracy comparison by default. Configure prices in ``data/prices.json``
(see :func:`load_prices`).
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import DATA_DIR, LOGS_DIR, ModelSpec, OLLAMA_HOST, new_run_id

RETRIES = 4
BACKOFF_SECONDS = 2.0

#: Purposes, one JSONL file each. Kept as constants so a typo cannot scatter
#: judge calls across two files and make the log look complete when it is not.
PURPOSE_QUERY_GEN = "query_gen"
PURPOSE_ANSWER = "answer"
PURPOSE_JUDGE = "judge"
PURPOSES = (PURPOSE_QUERY_GEN, PURPOSE_ANSWER, PURPOSE_JUDGE)


class LLMError(RuntimeError):
    """A call that could not be completed after retries."""


def _is_retryable(exc: Exception) -> bool:
    """Whether retrying this failure can possibly help.

    Both provider SDKs attach ``status_code`` to their HTTP errors (OpenAI's
    ``APIStatusError``, Ollama's ``ResponseError``). Timeouts (408), rate
    limits (429) and server errors (5xx) are transient; any other 4xx means the
    *request* is wrong -- a bad key, a retired model, malformed input -- and
    retrying it four times with backoff just multiplies the same failure.
    Errors with no status code (transport failures, connection resets) are
    retried.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and 400 <= status < 500:
        return status in (408, 429)
    return True


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float = 0.0
    attempts: int = 1
    cost_usd: float | None = None
    finish_reason: str | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens


# --------------------------------------------------------------------------- #
# Pricing.
# --------------------------------------------------------------------------- #

PRICES_PATH = DATA_DIR / "prices.json"


def load_prices(path: Path = PRICES_PATH) -> dict[str, dict[str, float]]:
    """Optional ``{model: {"input": usd_per_mtok, "output": usd_per_mtok}}``.

    Absent by design: prices change, and this harness must not bake in a number
    it cannot verify. With no file, dollar costs are reported as ``None`` and
    token counts carry the cost comparison.
    """
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def cost_of(model: str, prompt_tokens: int | None, completion_tokens: int | None) -> float | None:
    prices = load_prices().get(model)
    if not prices or prompt_tokens is None or completion_tokens is None:
        return None
    return (
        prompt_tokens * prices.get("input", 0.0) + completion_tokens * prices.get("output", 0.0)
    ) / 1_000_000


# --------------------------------------------------------------------------- #
# Logging.
# --------------------------------------------------------------------------- #


@dataclass
class CallLogger:
    """Append-only JSONL logger, one file per purpose.

    Thread-safe: Experiment 2 fans queries out concurrently, and interleaved
    partial writes would corrupt the very record needed to debug a bad number.
    """

    run_id: str = field(default_factory=new_run_id)
    root: Path = LOGS_DIR
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def dir(self) -> Path:
        return self.root / "llm" / self.run_id

    def path_for(self, purpose: str) -> Path:
        return self.dir / f"{purpose}.jsonl"

    def log(
        self,
        *,
        purpose: str,
        provider: str,
        model: str,
        messages: list[dict],
        response: LLMResponse | None,
        error: str | None = None,
        **meta: Any,
    ) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "purpose": purpose,
            "provider": provider,
            "model": model,
            "messages": messages,
            "response": response.text if response else None,
            "prompt_tokens": response.prompt_tokens if response else None,
            "completion_tokens": response.completion_tokens if response else None,
            "total_tokens": response.total_tokens if response else None,
            "cost_usd": response.cost_usd if response else None,
            "latency_s": round(response.latency_s, 4) if response else None,
            "attempts": response.attempts if response else None,
            "finish_reason": response.finish_reason if response else None,
            "error": error,
            **meta,
        }
        path = self.path_for(purpose)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read(self, purpose: str) -> list[dict]:
        path = self.path_for(purpose)
        if not path.exists():
            return []
        with path.open() as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def counts(self) -> dict[str, int]:
        return {p: len(self.read(p)) for p in PURPOSES}


# --------------------------------------------------------------------------- #
# Providers.
# --------------------------------------------------------------------------- #


#: Per-request ceiling. The SDK default is 600s, which is not a timeout so much
#: as an afternoon: a single hung socket blocks the whole run for ten minutes
#: before :func:`complete` is even told there is a problem. Generation and
#: judging are both short prompts with short answers -- a call still running
#: after a minute is not slow, it is stuck, and the retry is the cheaper move.
OPENAI_TIMEOUT_SECONDS = 60.0


def openai_client():
    from openai import OpenAI

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise LLMError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in; "
            "Experiment 1 runs without it, Experiment 2 does not."
        )
    # ``max_retries=0`` hands retry control to :func:`complete`, which logs every
    # attempt. The SDK's own retries are invisible to the log, so leaving them on
    # makes ``query_gen.jsonl`` undercount what was actually sent -- and the two
    # layers multiply: 3 SDK attempts inside 4 of ours is 12 requests per call.
    return OpenAI(api_key=key, timeout=OPENAI_TIMEOUT_SECONDS, max_retries=0)


def ollama_client():
    from ollama import Client

    return Client(host=OLLAMA_HOST)


def _call_openai(spec: ModelSpec, messages: list[dict], **kwargs) -> LLMResponse:
    client = openai_client()
    started = time.time()
    resp = client.chat.completions.create(model=spec.model, messages=messages, **kwargs)
    usage = resp.usage
    choice = resp.choices[0]
    return LLMResponse(
        text=choice.message.content or "",
        provider="openai",
        model=spec.model,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        latency_s=time.time() - started,
        finish_reason=choice.finish_reason,
    )


#: Models whose reasoning cannot be switched off, only *separated*. Ollama's
#: ``think=False`` does not stop a harmony-format model (gpt-oss) from
#: reasoning; it only stops the reasoning being split into its own field, so it
#: lands in ``content`` and corrupts the answer -- under ``format="json"`` the
#: result is unparseable prose followed by a mangled brace. Left as substrings
#: so ``gpt-oss:20b``, ``gpt-oss:120b`` and future tags all match.
ALWAYS_THINK = ("gpt-oss",)


def _thinks_regardless(model: str) -> bool:
    return any(name in model.lower() for name in ALWAYS_THINK)


def _call_ollama(spec: ModelSpec, messages: list[dict], **kwargs) -> LLMResponse:
    client = ollama_client()
    options = kwargs.pop("options", {})
    if "temperature" in kwargs:
        options["temperature"] = kwargs.pop("temperature")
    fmt = "json" if kwargs.pop("json_mode", False) else None
    think = True if _thinks_regardless(spec.model) else False

    started = time.time()
    try:
        resp = client.chat(
            model=spec.model,
            messages=messages,
            options=options or None,
            format=fmt,
            think=think,
        )
    except Exception as exc:  # noqa: BLE001 - narrow on the message below
        # think=False keeps reasoning models (qwen3) from spending tokens on
        # hidden thought, but Ollama rejects the parameter outright for models
        # without the capability. Retry once without it rather than ruling those
        # models out of the matrix.
        if "think" not in str(exc).lower():
            raise
        resp = client.chat(
            model=spec.model,
            messages=messages,
            options=options or None,
            format=fmt,
        )
    # Reasoning is returned separately and deliberately dropped from ``text``:
    # it is not the answer. Its tokens still land in ``eval_count`` below, so
    # the cost of thinking stays visible in the accounting.
    return LLMResponse(
        text=resp.get("message", {}).get("content", "") or "",
        provider="ollama",
        model=spec.model,
        prompt_tokens=resp.get("prompt_eval_count"),
        completion_tokens=resp.get("eval_count"),
        latency_s=time.time() - started,
        finish_reason=resp.get("done_reason"),
    )


# --------------------------------------------------------------------------- #
# The one entry point.
# --------------------------------------------------------------------------- #


def complete(
    spec: ModelSpec,
    messages: list[dict],
    *,
    purpose: str,
    logger: CallLogger | None = None,
    json_mode: bool = False,
    temperature: float | None = 0.0,
    max_retries: int = RETRIES,
    **meta: Any,
) -> LLMResponse:
    """Call ``spec`` with ``messages``; log the call whether it succeeds or not.

    ``meta`` (format, query id, ...) is written straight into the log record so a
    row in the results table can be traced back to the call that produced it.
    """
    kwargs: dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if json_mode:
        if spec.provider == "openai":
            kwargs["response_format"] = {"type": "json_object"}
        else:
            kwargs["json_mode"] = True

    caller = _call_openai if spec.provider == "openai" else _call_ollama
    last: Exception | None = None

    attempts = 0
    for attempt in range(1, max_retries + 1):
        attempts = attempt
        try:
            response = caller(spec, messages, **kwargs)
            response.attempts = attempt
            response.cost_usd = cost_of(
                spec.model, response.prompt_tokens, response.completion_tokens
            )
            if logger:
                logger.log(
                    purpose=purpose,
                    provider=spec.provider,
                    model=spec.model,
                    messages=messages,
                    response=response,
                    **meta,
                )
            return response
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise widely
            last = exc
            if not _is_retryable(exc):
                break
            if attempt < max_retries:
                time.sleep(BACKOFF_SECONDS * attempt)

    if logger:
        logger.log(
            purpose=purpose,
            provider=spec.provider,
            model=spec.model,
            messages=messages,
            response=None,
            error=f"{type(last).__name__}: {last}",
            attempts=attempts,
            **meta,
        )
    raise LLMError(f"{spec.key} failed after {attempts} attempt(s): {last}") from last


def ask(
    spec: ModelSpec,
    prompt: str,
    *,
    system: str | None = None,
    **kwargs,
) -> LLMResponse:
    """Convenience wrapper for a single-turn prompt."""
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    return complete(spec, messages, **kwargs)


# --------------------------------------------------------------------------- #
# Availability.
# --------------------------------------------------------------------------- #


def available_openai_models() -> list[str]:
    return sorted(m.id for m in openai_client().models.list().data)


def available_ollama_models() -> list[str]:
    return sorted(m.get("model", "") for m in ollama_client().list().get("models", []))


def resolve_models(specs: Iterable[ModelSpec]) -> tuple[list[ModelSpec], list[str]]:
    """Split ``specs`` into those the providers actually offer and problems.

    Checked once at startup rather than discovered as a 404 twenty minutes into
    a matrix run. A retired model name is reported alongside what *is* available.
    """
    specs = list(specs)
    problems: list[str] = []
    ok: list[ModelSpec] = []

    wanted = {s.provider for s in specs}
    catalogue: dict[str, set[str]] = {}
    for provider, lister in (("openai", available_openai_models), ("ollama", available_ollama_models)):
        if provider not in wanted:
            continue
        try:
            catalogue[provider] = set(lister())
        except Exception as exc:  # noqa: BLE001 - unreachable provider
            problems.append(f"{provider}: cannot list models ({type(exc).__name__}: {exc})")

    for spec in specs:
        names = catalogue.get(spec.provider)
        if names is None:
            continue
        if spec.model in names:
            ok.append(spec)
        else:
            near = sorted(n for n in names if spec.model.split(":")[0].split("-")[0] in n)[:8]
            problems.append(
                f"{spec.provider}: model {spec.model!r} not available"
                + (f"; did you mean one of {near}?" if near else "")
            )
    return ok, problems


if __name__ == "__main__":  # pragma: no cover - CLI smoke test
    import argparse

    from .config import model_matrix

    parser = argparse.ArgumentParser(description="Smoke-test the LLM providers.")
    parser.add_argument("--prompt", default="Reply with exactly: ok")
    args = parser.parse_args()

    specs = model_matrix()
    usable, issues = resolve_models(specs)
    print("configured models:")
    for s in specs:
        print(f"  {s.key:<28} {'available' if s in usable else 'UNAVAILABLE'}")
    for issue in issues:
        print(f"  ! {issue}")

    log = CallLogger()
    print(f"\nlogging to {log.dir}")
    for spec in usable:
        try:
            r = ask(spec, args.prompt, purpose=PURPOSE_ANSWER, logger=log, probe=True)
            print(
                f"  {spec.key:<28} {r.latency_s:5.2f}s  "
                f"in={r.prompt_tokens} out={r.completion_tokens}  {r.text.strip()[:50]!r}"
            )
        except LLMError as exc:
            print(f"  {spec.key:<28} FAILED: {exc}")

    print(f"\nlog records: {log.counts()}")
