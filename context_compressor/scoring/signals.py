"""
Signal weights for importance scoring.

Each signal contributes a score in [0, 1] range.
Final score is a weighted sum, normalized to [0, 1].
"""

import re
from dataclasses import dataclass


@dataclass
class SignalWeights:
    """
    Controls how much each signal contributes to the importance score.

    Tune these to match your use case:
    - Customer support: raise recency_weight, lower numeric_weight
    - Technical assistant: raise numeric_weight and code_weight
    - General chat: defaults work well
    """

    recency_weight: float = 0.30       # Recent messages matter more
    numeric_weight: float = 0.20       # Numbers, dates, measurements
    explicit_weight: float = 0.20      # "remember", "important", "always", "never"
    named_entity_weight: float = 0.15  # Proper nouns, capitalized entities
    code_weight: float = 0.10          # Code blocks, technical content
    question_weight: float = 0.05      # Direct questions (often need context)

    def validate(self) -> None:
        total = (
            self.recency_weight
            + self.numeric_weight
            + self.explicit_weight
            + self.named_entity_weight
            + self.code_weight
            + self.question_weight
        )
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Signal weights must sum to 1.0, got {total:.3f}")


# ── Signal extraction helpers ──────────────────────────────────────────────────

# Matches numbers, percentages, dates, currencies, measurements
_NUMERIC_RE = re.compile(
    r"""
    \b\d{4}-\d{2}-\d{2}\b          # ISO date
    | \b\d{1,2}/\d{1,2}/\d{2,4}\b  # US date
    | \$[\d,]+(?:\.\d+)?            # currency
    | \b\d+(?:\.\d+)?%\b            # percentage
    | \b\d{3,}\b                    # large numbers (IDs, ports, etc.)
    | \b\d+(?:\.\d+)?\s*(?:px|ms|kb|mb|gb|tb|rpm|fps|hz)\b  # units
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Explicit memory anchors — user is flagging something as important
_EXPLICIT_RE = re.compile(
    r"\b(?:"
    r"remember|always|never|must|critical|important|key|note that|"
    r"don['\u2019]t forget|keep in mind|make sure|be aware|"
    r"my name is|i am|i'm|our|we are|the goal is|the task is|"
    r"deadline|due|by [a-z]+day|constraint"
    r")\b",
    re.IGNORECASE,
)

# Named entities: sequences of 2+ capitalized words (naive but zero-dependency)
_NAMED_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")

# Code signals
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```|`[^`]+`")
_CODE_KEYWORDS_RE = re.compile(
    r"\b(?:def |class |import |from |function |const |let |var |async |await |"
    r"SELECT |INSERT |UPDATE |DELETE |CREATE |DROP )\b"
)

# Questions
_QUESTION_RE = re.compile(r"\?")


def numeric_density(text: str) -> float:
    """Ratio of numeric-signal characters to total characters."""
    matches = _NUMERIC_RE.findall(text)
    if not text:
        return 0.0
    matched_chars = sum(len(m) for m in matches)
    return min(matched_chars / max(len(text), 1) * 8, 1.0)


def explicit_density(text: str) -> float:
    """Presence of explicit importance anchors (capped at 1.0)."""
    count = len(_EXPLICIT_RE.findall(text))
    return min(count / 3.0, 1.0)


def named_entity_density(text: str) -> float:
    """Ratio of named entity tokens to total words."""
    words = text.split()
    if not words:
        return 0.0
    entities = _NAMED_ENTITY_RE.findall(text)
    entity_words = sum(len(e.split()) for e in entities)
    return min(entity_words / max(len(words), 1) * 5, 1.0)


def code_density(text: str) -> float:
    """Presence of code blocks or keywords."""
    if _CODE_BLOCK_RE.search(text):
        return 1.0
    kw_count = len(_CODE_KEYWORDS_RE.findall(text))
    return min(kw_count / 2.0, 1.0)


def question_density(text: str) -> float:
    """Whether the message contains questions."""
    count = len(_QUESTION_RE.findall(text))
    return min(count / 2.0, 1.0)


# Registry for easy iteration
SIGNAL_EXTRACTORS: list[str] = [
    "numeric",
    "explicit",
    "named_entity",
    "code",
    "question",
]
