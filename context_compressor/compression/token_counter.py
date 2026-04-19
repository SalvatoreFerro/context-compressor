"""
Token counting utilities.

Uses tiktoken for accurate counts. Falls back to word-based estimation
if tiktoken is not available or the model is unknown.
"""

from __future__ import annotations

from collections.abc import Sequence


def _get_encoder(model: str):
    try:
        import tiktoken
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding(model)
    except ImportError:
        return None


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Count tokens in a string."""
    enc = _get_encoder(model)
    if enc is not None:
        return len(enc.encode(text))
    # Fallback: ~1.3 tokens per word (empirical average)
    return int(len(text.split()) * 1.3)


def count_messages_tokens(messages: Sequence[dict], model: str = "cl100k_base") -> int:
    """
    Count total tokens in a messages list, including role overhead.

    OpenAI format adds ~4 tokens per message for role/metadata overhead.
    """
    total = 0
    for msg in messages:
        total += 4  # per-message overhead
        total += count_tokens(msg.get("role", ""), model)
        total += count_tokens(msg.get("content", "") or "", model)
    total += 2  # reply priming
    return total


def estimate_chars_for_tokens(n_tokens: int) -> int:
    """Rough estimate: ~4 chars per token for English text."""
    return n_tokens * 4
