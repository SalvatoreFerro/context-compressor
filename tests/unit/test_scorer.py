"""Unit tests for ImportanceScorer."""

import pytest
from context_compressor.scoring.scorer import ImportanceScorer
from context_compressor.scoring.signals import SignalWeights


@pytest.fixture
def scorer():
    return ImportanceScorer()


class TestSystemMessages:
    def test_system_message_always_score_one(self, scorer):
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        scored = scorer.score(messages)
        assert scored[0].score == 1.0

    def test_system_message_mixed_conversation(self, scorer):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        scored = scorer.score(messages)
        assert scored[0].score == 1.0
        assert scored[1].score < 1.0
        assert scored[2].score < 1.0


class TestRecency:
    def test_recent_messages_score_higher(self, scorer):
        messages = [
            {"role": "user", "content": "hello"},  # old
            {"role": "user", "content": "hello"},  # recent
        ]
        scored = scorer.score(messages)
        assert scored[1].score > scored[0].score

    def test_last_message_has_highest_recency(self, scorer):
        messages = [{"role": "user", "content": "msg"} for _ in range(20)]
        scored = scorer.score(messages)
        # Last message should have highest score
        scores = [s.score for s in scored if s.role != "system"]
        assert scores[-1] == max(scores)


class TestNumericSignal:
    def test_numbers_increase_score(self, scorer):
        low = [{"role": "user", "content": "hi how are you"}]
        high = [{"role": "user", "content": "The budget is $50,000 and deadline is 2025-06-15"}]

        scored_low = scorer.score(low)
        scored_high = scorer.score(high)

        assert scored_high[0].score > scored_low[0].score

    def test_iso_date_detected(self):
        from context_compressor.scoring.signals import numeric_density
        assert numeric_density("Meeting on 2025-06-15") > 0

    def test_currency_detected(self):
        from context_compressor.scoring.signals import numeric_density
        assert numeric_density("Cost: $12,500") > 0


class TestExplicitSignal:
    def test_remember_anchor_increases_score(self, scorer):
        low = [{"role": "user", "content": "today was okay"}]
        high = [{"role": "user", "content": "always remember my name is John"}]

        sl = scorer.score(low)[0].score
        sh = scorer.score(high)[0].score
        assert sh > sl

    def test_must_and_critical_detected(self):
        from context_compressor.scoring.signals import explicit_density
        assert explicit_density("You must never do this. Critical constraint.") > 0


class TestCodeSignal:
    def test_code_block_detected(self):
        from context_compressor.scoring.signals import code_density
        assert code_density("```python\ndef foo(): pass\n```") == 1.0

    def test_inline_code_detected(self):
        from context_compressor.scoring.signals import code_density
        assert code_density("Use `async def` for this") > 0

    def test_code_keywords_detected(self):
        from context_compressor.scoring.signals import code_density
        assert code_density("import numpy as np") > 0


class TestScoreBreakdown:
    def test_breakdown_returns_all_signals(self, scorer):
        msg = {"role": "user", "content": "Remember: the budget is $50,000"}
        breakdown = scorer.score_breakdown(msg, position=0, total=1)
        assert "signals" in breakdown
        assert "final_score" in breakdown
        for signal in ["recency", "numeric", "explicit", "named_entity", "code", "question"]:
            assert signal in breakdown["signals"]

    def test_system_breakdown_returns_reason(self, scorer):
        msg = {"role": "system", "content": "You are helpful."}
        breakdown = scorer.score_breakdown(msg, position=0, total=1)
        assert breakdown["final_score"] == 1.0
        assert "reason" in breakdown


class TestCustomWeights:
    def test_zero_recency_weight(self):
        weights = SignalWeights(
            recency_weight=0.0,
            numeric_weight=0.25,
            explicit_weight=0.25,
            named_entity_weight=0.25,
            code_weight=0.15,
            question_weight=0.10,
        )
        scorer = ImportanceScorer(weights=weights)
        messages = [{"role": "user", "content": "hello"} for _ in range(5)]
        scored = scorer.score(messages)
        # Without recency, all plain messages should score similarly
        scores = [s.score for s in scored]
        assert max(scores) - min(scores) < 0.1

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            w = SignalWeights(recency_weight=0.9)
            w.validate()
