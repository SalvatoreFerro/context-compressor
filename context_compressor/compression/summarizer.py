"""
Lightweight extractive summarizer — no LLM calls required.

Strategy:
1. Split content into sentences.
2. Score each sentence by signal density (numeric, explicit, entities).
3. Keep top-K sentences that fit within a token budget.
4. Preserve sentence order for readability.

This is intentionally simple and fast. If you want abstractive summaries,
you can plug in your own LLM-based summarizer via the `Summarizer` protocol.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class Summarizer(Protocol):
    """Protocol for custom summarizers. Plug in your own LLM-based one."""

    def summarize(self, text: str, max_chars: int = 200) -> str:
        ...


class ExtractiveSummarizer:
    """
    Extractive summarizer: selects the most informative sentences.

    Unlike LangChain's ConversationSummaryMemory, this:
    - Requires zero LLM calls
    - Is deterministic and testable
    - Preserves numeric and factual content with higher fidelity
    """

    # Sentence splitter: ends at . ! ? followed by space or end
    _SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

    def summarize(self, text: str, max_chars: int = 200) -> str:
        if len(text) <= max_chars:
            return text

        sentences = self._split_sentences(text)
        if not sentences:
            return text[:max_chars]

        # Score each sentence
        scored = [(self._sentence_score(s), i, s) for i, s in enumerate(sentences)]
        scored.sort(key=lambda x: -x[0])

        # Greedily pick top sentences within budget
        selected: list[tuple] = []
        budget = max_chars

        for score, idx, sentence in scored:
            if len(sentence) + 2 <= budget:  # +2 for ". "
                selected.append((idx, sentence))
                budget -= len(sentence) + 2
            if budget <= 0:
                break

        if not selected:
            # Fallback: truncate
            return text[:max_chars - 3] + "..."

        # Restore original order
        selected.sort(key=lambda x: x[0])
        return " ".join(s for _, s in selected)

    def _split_sentences(self, text: str) -> list[str]:
        sentences = self._SENTENCE_RE.split(text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _sentence_score(self, sentence: str) -> float:
        """Higher = more informative."""
        score = 0.0

        # Numbers and measurements are high signal
        if re.search(r"\d", sentence):
            score += 2.0

        # Explicit importance anchors
        if re.search(
            r"\b(?:important|must|always|never|remember|critical|key|deadline)\b",
            sentence,
            re.IGNORECASE,
        ):
            score += 2.5

        # Named entities (capitalized bigrams)
        entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", sentence)
        score += len(entities) * 0.5

        # Code content
        if re.search(r"`|def |class |import ", sentence):
            score += 1.5

        # Prefer medium-length sentences (not too short, not rambling)
        length = len(sentence)
        if 20 <= length <= 150:
            score += 0.5

        return score
