"""Token counting for Experiment 1.

Three measures per rendered string, because no single one is trustworthy alone:

``tiktoken`` (``o200k_base``)
    The OpenAI-side number, and the primary result.
Hugging Face ``AutoTokenizer``
    An open-weights cross-check. Tokeniser vocabularies differ enough that a
    format could win on one and lose on another -- which is itself a finding, and
    one that a single-tokenizer study would report as a fact about formats.
Characters and bytes
    A tokenizer-independent control. If the two tokenizers disagree with each
    other but both track characters, the effect is length, not vocabulary.

A tokenizer that cannot be loaded yields ``None``, which the report shows as an
absent column. It is never silently replaced with a character estimate.
"""

from __future__ import annotations

import functools

from .config import HF_TOKENIZER, TIKTOKEN_ENCODING


class TokenizerUnavailable(RuntimeError):
    """Raised by the loaders; callers degrade to ``None`` and say so."""


@functools.lru_cache(maxsize=4)
def tiktoken_encoder(name: str = TIKTOKEN_ENCODING):
    import tiktoken

    try:
        return tiktoken.get_encoding(name)
    except Exception as exc:  # noqa: BLE001 - surfaced as unavailability
        raise TokenizerUnavailable(f"tiktoken {name!r}: {exc}") from exc


@functools.lru_cache(maxsize=4)
def hf_tokenizer(name: str = HF_TOKENIZER):
    """Load a Hugging Face tokenizer, downloading it on first use."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name)
    except Exception as exc:  # noqa: BLE001 - network or gated repo
        raise TokenizerUnavailable(f"hf {name!r}: {exc}") from exc


def count_tiktoken(text: str, name: str = TIKTOKEN_ENCODING) -> int:
    return len(tiktoken_encoder(name).encode(text))


def count_hf(text: str, name: str = HF_TOKENIZER) -> int:
    return len(hf_tokenizer(name).encode(text, add_special_tokens=False))


def measure(text: str, *, hf: bool = True) -> dict[str, int | None]:
    """All measures for one rendered string.

    ``hf_tokens`` is ``None`` when the tokenizer could not be loaded -- an
    honestly missing column beats a fabricated one.
    """
    out: dict[str, int | None] = {
        "chars": len(text),
        "bytes": len(text.encode("utf-8")),
        "tiktoken": count_tiktoken(text),
        "hf_tokens": None,
    }
    if hf:
        try:
            out["hf_tokens"] = count_hf(text)
        except TokenizerUnavailable:
            pass
    return out


def hf_available() -> tuple[bool, str]:
    """``(available, detail)`` -- used to record why a column is absent."""
    try:
        hf_tokenizer()
    except TokenizerUnavailable as exc:
        return False, str(exc)
    return True, HF_TOKENIZER
