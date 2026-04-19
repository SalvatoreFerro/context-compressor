"""
CompressorConfig — all tuneable parameters in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from context_compressor.scoring.signals import SignalWeights


@dataclass
class CompressorConfig:
    """
    Configuration for ContextCompressor.

    Args:
        max_tokens: Hard limit on output token count. The compressor will
                    always produce a message list within this budget.
                    Default: 4096 (fits most models' practical limits).

        model: Tokenizer model name (for tiktoken). Use the model you're
               sending to, e.g. "gpt-4o", "gpt-3.5-turbo", "cl100k_base".
               Default: "cl100k_base" (works for all OpenAI + Anthropic models).

        preserve_threshold: Messages with score >= this are always kept verbatim.
                            Default: 0.7

        compress_threshold: Messages with score in [compress_threshold, preserve_threshold)
                            are included as one-line summaries.
                            Messages below compress_threshold are dropped.
                            Default: 0.3

        summary_prefix: Prefix added to compressed message summaries.
                        Default: "[summarized] "

        always_preserve_last_n: Always keep the last N messages verbatim,
                                regardless of score. Ensures the model always
                                sees the most recent context.
                                Default: 4

        signal_weights: Custom signal weights for the importance scorer.
                        Default: balanced SignalWeights().

        recency_half_life: Half-life in number of messages for recency decay.
                           Default: 10
    """

    max_tokens: int = 4096
    model: str = "cl100k_base"
    preserve_threshold: float = 0.7
    compress_threshold: float = 0.3
    summary_prefix: str = "[summarized] "
    always_preserve_last_n: int = 4
    signal_weights: SignalWeights = field(default_factory=SignalWeights)
    recency_half_life: int = 10

    def validate(self) -> None:
        if self.max_tokens < 256:
            raise ValueError("max_tokens must be >= 256")
        if not (0.0 <= self.compress_threshold < self.preserve_threshold <= 1.0):
            raise ValueError(
                "Must satisfy: 0 <= compress_threshold < preserve_threshold <= 1"
            )
        if self.always_preserve_last_n < 0:
            raise ValueError("always_preserve_last_n must be >= 0")
        self.signal_weights.validate()

    @classmethod
    def for_model(cls, model: str, max_tokens: int | None = None) -> CompressorConfig:
        """
        Preset configs for common models.

        Usage:
            config = CompressorConfig.for_model("gpt-4o")
            config = CompressorConfig.for_model("claude-3-5-sonnet-20241022")
        """
        presets = {
            # OpenAI
            "gpt-3.5-turbo": 4096,
            "gpt-4": 8192,
            "gpt-4o": 16384,
            "gpt-4o-mini": 16384,
            # Anthropic
            "claude-3-haiku-20240307": 16384,
            "claude-3-5-sonnet-20241022": 32768,
            "claude-3-opus-20240229": 32768,
            # Meta
            "llama3": 8192,
        }

        # Match by prefix or substring
        matched_tokens = None
        for key, tokens in presets.items():
            if model == key or model.startswith(key) or key in model:
                matched_tokens = tokens
                break

        token_limit = max_tokens or matched_tokens or 8192
        return cls(max_tokens=token_limit, model="cl100k_base")
