"""Unified cross-provider LLM client.

A single ``complete()`` dispatches by model-id prefix to:

* Anthropic   -- ids starting with "claude"      (needs ANTHROPIC_API_KEY)
* OpenAI      -- ids starting with "gpt"/"o"      (needs OPENAI_API_KEY)
* Ollama      -- anything else, served locally    (OLLAMA_HOST, default :11434)

Missing cloud keys or an unreachable Ollama raise :class:`ProviderUnavailable`,
which the runner catches to *skip* a model tier rather than crash a sweep.
This lets the suite run with whatever providers are configured (e.g. Ollama +
Claude) and fill in others later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Canonical model tiers referenced by the experiments. The open tier is
# populated dynamically from whatever is pulled in Ollama.
CLAUDE_MODELS = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"]
OPENAI_MODELS = ["gpt-5.4-nano", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini"]


class ProviderUnavailable(RuntimeError):
    """Raised when a model's provider is not configured/reachable."""


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


def provider_for(model: str) -> str:
    m = model.lower()
    # Ollama tags are "name:tag" (e.g. gpt-oss:20b, llama3.2:latest). A colon
    # means a locally served model even when the name starts with "gpt" — cloud
    # model ids never contain one, so check this before the prefix rules.
    if ":" in m:
        return "ollama"
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    return "ollama"


def _complete_anthropic(model, system, prompt, max_tokens, temperature) -> Completion:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ProviderUnavailable("ANTHROPIC_API_KEY not set")
    import anthropic

    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model=model,
        system=system or "",
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    return Completion(text, model, msg.usage.input_tokens, msg.usage.output_tokens)


def _complete_openai(model, system, prompt, max_tokens, temperature) -> Completion:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ProviderUnavailable("OPENAI_API_KEY not set")
    from openai import OpenAI

    client = OpenAI(api_key=key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # GPT-5 / o-series reasoning models renamed ``max_tokens`` to
    # ``max_completion_tokens`` and accept only the default temperature; the
    # older gpt-4.x chat models keep the classic params. Route accordingly so a
    # single complete() works across both.
    m = model.lower()
    newer = m.startswith("gpt-5") or m.startswith(("o1", "o3", "o4"))
    kwargs: dict = {"model": model, "messages": messages}
    if newer:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = temperature
    resp = client.chat.completions.create(**kwargs)
    usage = resp.usage
    return Completion(
        resp.choices[0].message.content or "",
        model,
        getattr(usage, "prompt_tokens", 0),
        getattr(usage, "completion_tokens", 0),
    )


def _ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def _complete_ollama(model, system, prompt, max_tokens, temperature) -> Completion:
    import httpx

    url = f"{_ollama_host().rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "messages": (
            ([{"role": "system", "content": system}] if system else [])
            + [{"role": "user", "content": prompt}]
        ),
    }
    try:
        resp = httpx.post(url, json=payload, timeout=httpx.Timeout(300.0))
        resp.raise_for_status()
    except httpx.HTTPError as exc:  # connection refused, 404 model, etc.
        raise ProviderUnavailable(f"Ollama unavailable for {model}: {exc}") from exc
    data = resp.json()
    return Completion(
        data.get("message", {}).get("content", ""),
        model,
        data.get("prompt_eval_count", 0),
        data.get("eval_count", 0),
    )


def complete(
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> Completion:
    """Generate a completion from any configured provider.

    Raises :class:`ProviderUnavailable` if the model's provider is not ready.
    """
    provider = provider_for(model)
    dispatch = {
        "anthropic": _complete_anthropic,
        "openai": _complete_openai,
        "ollama": _complete_ollama,
    }[provider]
    return dispatch(model, system, prompt, max_tokens, temperature)


def list_ollama_models() -> list[str]:
    """Return locally pulled Ollama model names, or [] if Ollama is down."""
    import httpx

    try:
        resp = httpx.get(f"{_ollama_host().rstrip('/')}/api/tags", timeout=5.0)
        resp.raise_for_status()
    except httpx.HTTPError:
        return []
    return [m["name"] for m in resp.json().get("models", [])]


def available_models(
    candidates: list[str] | None = None,
) -> list[str]:
    """Filter a candidate model list to those whose providers are configured.

    With no candidates, returns Claude + OpenAI tiers (if keyed) plus all
    locally pulled Ollama models.
    """
    if candidates is None:
        candidates = list(CLAUDE_MODELS) + list(OPENAI_MODELS) + list_ollama_models()

    ready: list[str] = []
    ollama_present = set(list_ollama_models())
    for model in candidates:
        provider = provider_for(model)
        if provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            ready.append(model)
        elif provider == "openai" and os.environ.get("OPENAI_API_KEY"):
            ready.append(model)
        elif provider == "ollama" and model in ollama_present:
            ready.append(model)
    return ready
