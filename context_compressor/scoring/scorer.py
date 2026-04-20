"""
ImportanceScorer — assigns a score in [0, 1] to each conversation message.

Design principles vs. competitors:
- Zero LLM calls (unlike LangChain ConversationSummaryMemory)
- Semantic signals, not byte-level compression (unlike Headroom/LLMLingua)
- Recency-aware with exponential decay
- Deterministic and reproducible (testable)
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from context_compressor.scoring.signals import (
    SignalWeights,
    code_density,
    explicit_density,
    named_entity_density,
    numeric_density,
    question_density,
)


class ScoredMessage:
    """A conversation message with its computed importance score."""

    __slots__ = ("role", "content", "score", "index", "_original_msg")

    def __init__(
        self,
        role: str,
        content: str,
        score: float,
        index: int,
        original_msg: dict | None = None,
    ) -> None:
        self.role = role
        self.content = content
        self.score = score
        self.index = index
        self._original_msg = original_msg

    def to_dict(self) -> dict:
        if self._original_msg is not None:
            return dict(self._original_msg)
        return {"role": self.role, "content": self.content}

    def __repr__(self) -> str:
        return f"ScoredMessage(role={self.role!r}, score={self.score:.3f}, index={self.index})"


class ImportanceScorer:
    """
    Scores each message in a conversation by importance.

    Importance = weighted combination of:
      - Recency (exponential decay from end of conversation)
      - Numeric content (numbers, dates, measurements)
      - Explicit anchors ("remember", "always", "must", etc.)
      - Named entities (proper nouns)
      - Code content (blocks, keywords)
      - Questions

    System messages are always assigned score 1.0 (never compressed).

    Args:
        weights: Signal weights configuration. Defaults to balanced weights.
        recency_half_life: How many messages back before recency score halves.
                           Default 10 = message 10 turns ago has 0.5x recency.
    """

    def __init__(
        self,
        weights: SignalWeights | None = None,
        recency_half_life: int = 10,
    ) -> None:
        self.weights = weights or SignalWeights()
        self.weights.validate()
        self.recency_half_life = recency_half_life

    def score(self, messages: Sequence[dict]) -> list[ScoredMessage]:
        """
        Score all messages in a conversation.

        Args:
            messages: List of {"role": str, "content": str} dicts.

        Returns:
            List of ScoredMessage, same order as input.
        """
        scored: list[ScoredMessage] = []
        n = len(messages)

        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "") or ""

            # System messages are always preserved
            if role == "system":
                scored.append(ScoredMessage(role, content, 1.0, i, original_msg=msg))
                continue

            # Recency: exponential decay, most recent = 1.0
            distance_from_end = n - 1 - i
            signals = self._compute_signals(content, distance_from_end)

            w = self.weights
            score = sum(
                getattr(w, f"{name}_weight") * value
                for name, value in signals.items()
            )

            # Clamp to [0, 1]
            score = max(0.0, min(1.0, score))
            scored.append(ScoredMessage(role, content, score, i, original_msg=msg))

        return scored

    def _compute_signals(self, content: str, distance_from_end: int) -> dict[str, float]:
        """Compute all signal values for a message's content."""
        recency = math.exp(
            -distance_from_end * math.log(2) / self.recency_half_life
        )
        return {
            "recency": recency,
            "numeric": numeric_density(content),
            "explicit": explicit_density(content),
            "named_entity": named_entity_density(content),
            "code": code_density(content),
            "question": question_density(content),
        }

    def score_breakdown(self, message: dict, position: int, total: int) -> dict:
        """
        Returns a detailed breakdown of signal contributions for a single message.
        Useful for debugging and understanding why a message was kept/compressed.
        """
        role = message.get("role", "user")
        content = message.get("content", "") or ""

        if role == "system":
            return {"role": role, "final_score": 1.0, "reason": "system message always preserved"}

        distance_from_end = total - 1 - position
        signals = self._compute_signals(content, distance_from_end)

        w = self.weights
        weighted = {
            name: (value, getattr(w, f"{name}_weight"))
            for name, value in signals.items()
        }
        final_score = sum(v * wt for v, wt in weighted.values())

        return {
            "role": role,
            "signals": {
                k: {"raw": v, "weight": wt, "contribution": v * wt}
                for k, (v, wt) in weighted.items()
            },
            "final_score": round(min(1.0, final_score), 4),
        }
