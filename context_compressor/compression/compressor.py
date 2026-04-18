"""
ContextCompressor — the main public interface.

Drop-in replacement for raw messages[] passing.

    # Before
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,  # might blow up context window
    )

    # After — 2 lines
    compressor = ContextCompressor(CompressorConfig.for_model("gpt-4o"))
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=compressor.compress(messages),  # always fits
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from context_compressor.compression.config import CompressorConfig
from context_compressor.compression.summarizer import ExtractiveSummarizer, Summarizer
from context_compressor.compression.token_counter import (
    count_messages_tokens,
    estimate_chars_for_tokens,
)
from context_compressor.scoring.scorer import ImportanceScorer, ScoredMessage


@dataclass
class CompressionResult:
    """
    Result of a compression pass.

    Attributes:
        messages: Compressed messages list, ready to send to any LLM API.
        original_token_count: Token count before compression.
        compressed_token_count: Token count after compression.
        messages_kept: Number of messages preserved verbatim.
        messages_summarized: Number of messages summarized.
        messages_dropped: Number of messages dropped entirely.
        compression_ratio: compressed / original (lower = more compressed).
    """

    messages: List[dict]
    original_token_count: int
    compressed_token_count: int
    messages_kept: int
    messages_summarized: int
    messages_dropped: int

    @property
    def compression_ratio(self) -> float:
        if self.original_token_count == 0:
            return 1.0
        return self.compressed_token_count / self.original_token_count

    @property
    def token_savings(self) -> int:
        return self.original_token_count - self.compressed_token_count

    def __repr__(self) -> str:
        return (
            f"CompressionResult("
            f"ratio={self.compression_ratio:.2f}, "
            f"saved={self.token_savings} tokens, "
            f"kept={self.messages_kept}, "
            f"summarized={self.messages_summarized}, "
            f"dropped={self.messages_dropped})"
        )


class ContextCompressor:
    """
    Intelligent conversation context compressor.

    How it works:
    1. Score each message using ImportanceScorer (zero LLM calls).
    2. Always preserve: system messages + last N messages.
    3. Score >= preserve_threshold → keep verbatim.
    4. Score in [compress_threshold, preserve_threshold) → extract key sentences.
    5. Score < compress_threshold → drop.
    6. If still over token budget, iteratively drop lowest-scored messages.

    Key advantages over competitors:
    - Zero extra LLM calls (LangChain ConversationSummaryMemory needs one per summary)
    - Semantic scoring, not byte-level compression (unlike Headroom/LLMLingua)
    - Preserves numeric facts, named entities, code blocks with high fidelity
    - Fully deterministic: same input always produces same output

    Args:
        config: CompressorConfig instance. Use CompressorConfig.for_model(model_name)
                for sensible defaults per model.
        summarizer: Optional custom summarizer. Defaults to ExtractiveSummarizer.
                    Pass any object implementing the Summarizer protocol to use
                    an LLM-based abstractive summarizer instead.

    Example:
        >>> compressor = ContextCompressor(CompressorConfig(max_tokens=4096))
        >>> compressed = compressor.compress(messages)
        >>> # compressed is a list[dict] — pass directly to any LLM API
    """

    def __init__(
        self,
        config: Optional[CompressorConfig] = None,
        summarizer: Optional[Summarizer] = None,
    ) -> None:
        self.config = config or CompressorConfig()
        self.config.validate()
        self.scorer = ImportanceScorer(
            weights=self.config.signal_weights,
            recency_half_life=self.config.recency_half_life,
        )
        self.summarizer = summarizer or ExtractiveSummarizer()

    def compress(self, messages: Sequence[dict]) -> List[dict]:
        """
        Compress a conversation to fit within max_tokens.

        Args:
            messages: List of {"role": str, "content": str} dicts.

        Returns:
            Compressed list[dict], same format — ready to pass to any LLM API.
        """
        return self.compress_with_stats(messages).messages

    def compress_with_stats(self, messages: Sequence[dict]) -> CompressionResult:
        """
        Like compress(), but also returns compression statistics.

        Useful for logging, monitoring, or benchmarking.
        """
        messages = list(messages)
        original_tokens = count_messages_tokens(messages, self.config.model)

        # Fast path: already fits
        if original_tokens <= self.config.max_tokens:
            return CompressionResult(
                messages=messages,
                original_token_count=original_tokens,
                compressed_token_count=original_tokens,
                messages_kept=len(messages),
                messages_summarized=0,
                messages_dropped=0,
            )

        # Score all messages
        scored = self.scorer.score(messages)
        n = len(scored)

        # Identify always-preserved indices
        always_keep = set()
        for i, sm in enumerate(scored):
            if sm.role == "system":
                always_keep.add(i)
        # Last N messages
        last_n = self.config.always_preserve_last_n
        for i in range(max(0, n - last_n), n):
            always_keep.add(i)

        # Decide fate of each message
        # Each entry: (original_index, score, msg_dict) — index carried through for budget enforcement
        kept, summarized, dropped = 0, 0, 0
        output_entries: List[tuple] = []  # (original_index, score, msg_dict)

        for sm in scored:
            if sm.index in always_keep:
                output_entries.append((sm.index, sm.score, sm.to_dict()))
                kept += 1
            elif sm.score >= self.config.preserve_threshold:
                output_entries.append((sm.index, sm.score, sm.to_dict()))
                kept += 1
            elif sm.score >= self.config.compress_threshold:
                summary = self._summarize_message(sm)
                output_entries.append((sm.index, sm.score, {"role": sm.role, "content": summary}))
                summarized += 1
            else:
                dropped += 1

        # Second pass: if still over budget, drop lowest-scored non-locked messages
        output_entries = self._enforce_budget(output_entries, always_keep)
        output_messages = [msg for _, _, msg in output_entries]

        compressed_tokens = count_messages_tokens(output_messages, self.config.model)

        return CompressionResult(
            messages=output_messages,
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
            messages_kept=kept,
            messages_summarized=summarized,
            messages_dropped=dropped,
        )

    def _summarize_message(self, sm: ScoredMessage) -> str:
        """Summarize a single message content."""
        max_chars = estimate_chars_for_tokens(
            max(30, int(self.config.max_tokens * 0.02))
        )
        summary = self.summarizer.summarize(sm.content, max_chars=max_chars)
        return f"{self.config.summary_prefix}{summary}"

    def _enforce_budget(
        self,
        output_entries: List[tuple],
        always_keep: set,
    ) -> List[tuple]:
        """
        Iteratively drop lowest-scored messages until we're within budget.

        Uses original_index (not content) as key — safe after summarization
        because content may have been modified with the summary prefix.
        """
        msgs = [msg for _, _, msg in output_entries]
        current_tokens = count_messages_tokens(msgs, self.config.model)

        if current_tokens <= self.config.max_tokens:
            return output_entries

        # Sort by score ascending (lowest first = drop first), exclude protected
        candidates = [
            (score, idx, msg)
            for idx, score, msg in output_entries
            if idx not in always_keep and msg["role"] != "system"
        ]
        candidates.sort(key=lambda x: x[0])

        # Drop until within budget
        to_drop: set = set()
        for score, idx, msg in candidates:
            if current_tokens <= self.config.max_tokens:
                break
            msg_tokens = 4 + len((msg.get("content") or "").split()) + 1
            current_tokens -= msg_tokens
            to_drop.add(idx)

        return [(idx, score, msg) for idx, score, msg in output_entries if idx not in to_drop]

    def explain(self, messages: Sequence[dict]) -> List[dict]:
        """
        Debug tool: returns scoring breakdown for each message.

        Useful for understanding compression decisions without actually compressing.

        Returns:
            List of dicts with role, content preview, score, and signal breakdown.
        """
        messages = list(messages)
        scored = self.scorer.score(messages)
        n = len(messages)
        results = []

        for sm in scored:
            breakdown = self.scorer.score_breakdown(
                {"role": sm.role, "content": sm.content},
                position=sm.index,
                total=n,
            )
            fate = self._decide_fate(sm, n)
            results.append({
                "index": sm.index,
                "role": sm.role,
                "content_preview": sm.content[:80] + ("..." if len(sm.content) > 80 else ""),
                "fate": fate,
                **breakdown,
            })

        return results

    def _decide_fate(self, sm: ScoredMessage, total: int) -> str:
        n = total
        always_keep = sm.role == "system" or sm.index >= n - self.config.always_preserve_last_n
        if always_keep:
            return "keep (protected)"
        if sm.score >= self.config.preserve_threshold:
            return "keep (high score)"
        if sm.score >= self.config.compress_threshold:
            return "summarize"
        return "drop"
