"""
Anthropic adapter — wraps the Anthropic client with automatic context compression.

    # Before
    client = anthropic.Anthropic(api_key="...")
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022", messages=messages, max_tokens=1024
    )

    # After
    client = AnthropicAdapter(api_key="...", model="claude-3-5-sonnet-20241022")
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022", messages=messages, max_tokens=1024
    )
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from context_compressor.compression.compressor import CompressionResult, ContextCompressor
from context_compressor.compression.config import CompressorConfig


class AnthropicAdapter:
    """
    Anthropic client wrapper with automatic context compression.

    Note: Anthropic's API separates system prompts from messages.
    This adapter handles that correctly — system role messages are
    extracted and passed as the `system` parameter automatically.

    Args:
        api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
        model: Default model for token counting preset.
        config: Custom CompressorConfig. Auto-detected from model if None.
        **kwargs: Passed through to anthropic.Anthropic().

    Attributes:
        last_compression: CompressionResult from the most recent call.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-sonnet-20241022",
        config: CompressorConfig | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required for AnthropicAdapter. "
                "Install it with: pip install context-compressor[anthropic]"
            )

        self._client = anthropic.Anthropic(api_key=api_key, **kwargs)
        self._default_model = model
        cfg = config or CompressorConfig.for_model(model)
        self._compressor = ContextCompressor(cfg)
        self.last_compression: CompressionResult | None = None
        self.messages = _MessagesNamespace(self)

    def _compress(self, messages: Sequence[dict]) -> list[dict]:
        result = self._compressor.compress_with_stats(messages)
        self.last_compression = result
        return result.messages

    @property
    def compressor(self) -> ContextCompressor:
        return self._compressor

    @property
    def raw_client(self):
        return self._client


class _MessagesNamespace:
    def __init__(self, adapter: AnthropicAdapter) -> None:
        self._adapter = adapter

    def create(self, messages: Sequence[dict], **kwargs: Any) -> Any:
        """
        Compress messages then call anthropic.messages.create().

        Automatically handles system message extraction for Anthropic's API format.
        """
        messages = list(messages)

        # Extract system messages (Anthropic requires them as a separate param)
        system_parts: list[str] = []
        non_system = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if content:
                    system_parts.append(content)
            else:
                non_system.append(msg)

        compressed = self._adapter._compress(non_system)

        # Build final kwargs
        call_kwargs = dict(kwargs)
        if system_parts and "system" not in call_kwargs:
            call_kwargs["system"] = "\n\n".join(system_parts)

        return self._adapter._client.messages.create(
            messages=compressed, **call_kwargs
        )
