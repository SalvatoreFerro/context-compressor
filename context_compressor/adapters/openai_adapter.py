"""
OpenAI adapter — wraps the OpenAI client with automatic context compression.

Drop-in replacement: swap `openai.OpenAI()` for `OpenAIAdapter()`.

    # Before
    client = openai.OpenAI(api_key="...")
    response = client.chat.completions.create(model="gpt-4o", messages=messages)

    # After — identical API surface
    client = OpenAIAdapter(api_key="...", model="gpt-4o")
    response = client.chat.completions.create(model="gpt-4o", messages=messages)
    # Compression happens automatically. Access stats via client.last_compression
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from context_compressor.compression.compressor import CompressionResult, ContextCompressor
from context_compressor.compression.config import CompressorConfig


class OpenAIAdapter:
    """
    OpenAI client wrapper with automatic context compression.

    Args:
        api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        model: Default model (used for token counting preset).
        config: Custom CompressorConfig. Auto-detected from model if None.
        **kwargs: Passed through to openai.OpenAI().

    Attributes:
        last_compression: CompressionResult from the most recent call.
                          None if no compression was needed.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        config: Optional[CompressorConfig] = None,
        **kwargs: Any,
    ) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAIAdapter. "
                "Install it with: pip install context-compressor[openai]"
            )

        self._client = openai.OpenAI(api_key=api_key, **kwargs)
        self._default_model = model
        cfg = config or CompressorConfig.for_model(model)
        self._compressor = ContextCompressor(cfg)
        self.last_compression: Optional[CompressionResult] = None
        # Expose chat.completions interface
        self.chat = _ChatNamespace(self)

    def _compress(self, messages: Sequence[dict], model: Optional[str] = None) -> List[dict]:
        result = self._compressor.compress_with_stats(messages)
        self.last_compression = result
        return result.messages

    @property
    def compressor(self) -> ContextCompressor:
        return self._compressor

    @property
    def raw_client(self):
        """Access the underlying openai.OpenAI() client directly."""
        return self._client


class _ChatNamespace:
    def __init__(self, adapter: OpenAIAdapter) -> None:
        self._adapter = adapter
        self.completions = _CompletionsNamespace(adapter)


class _CompletionsNamespace:
    def __init__(self, adapter: OpenAIAdapter) -> None:
        self._adapter = adapter

    def create(self, messages: Sequence[dict], **kwargs: Any) -> Any:
        """
        Compress messages then call openai.chat.completions.create().

        All kwargs are passed through unchanged.
        """
        compressed = self._adapter._compress(messages, kwargs.get("model"))
        return self._adapter._client.chat.completions.create(
            messages=compressed, **kwargs
        )
