"""Unit tests for ExtractiveSummarizer."""

import pytest
from context_compressor.compression.summarizer import ExtractiveSummarizer


@pytest.fixture
def summarizer():
    return ExtractiveSummarizer()


class TestBasicSummarization:
    def test_short_text_unchanged(self, summarizer):
        text = "Hello world."
        result = summarizer.summarize(text, max_chars=200)
        assert result == text

    def test_long_text_truncated_to_budget(self, summarizer):
        text = "This is sentence one. " * 20
        result = summarizer.summarize(text, max_chars=100)
        assert len(result) <= 100 + 20  # some slack for sentence boundaries

    def test_output_is_string(self, summarizer):
        result = summarizer.summarize("Some text here.", max_chars=50)
        assert isinstance(result, str)

    def test_empty_string(self, summarizer):
        result = summarizer.summarize("", max_chars=100)
        assert isinstance(result, str)

    def test_single_sentence(self, summarizer):
        text = "The budget is $50,000 for this project."
        result = summarizer.summarize(text, max_chars=200)
        assert result == text


class TestImportantContentPreserved:
    def test_numeric_sentence_preserved(self, summarizer):
        text = (
            "Today was a nice day. "
            "The project deadline is 2025-12-31 and budget is $85,000. "
            "Weather was good. "
            "Nothing else happened."
        )
        result = summarizer.summarize(text, max_chars=120)
        assert "$85,000" in result or "2025-12-31" in result or "85,000" in result

    def test_explicit_anchor_preserved(self, summarizer):
        text = (
            "How are you doing today? "
            "Remember: you must never use AWS services. "
            "It was a good meeting. "
            "See you tomorrow."
        )
        result = summarizer.summarize(text, max_chars=100)
        assert "AWS" in result or "never" in result or "must" in result

    def test_named_entity_preserved(self, summarizer):
        text = (
            "So that was fine. "
            "Maria Chen is the lead architect and must approve all changes. "
            "Okay sounds good. "
            "Let me know."
        )
        result = summarizer.summarize(text, max_chars=100)
        assert "Maria Chen" in result or "Maria" in result

    def test_code_content_preserved(self, summarizer):
        text = (
            "Some general chat here. "
            "Use `async def` for this function to avoid blocking. "
            "Okay thanks. "
            "See you later."
        )
        result = summarizer.summarize(text, max_chars=100)
        assert "async" in result or "def" in result


class TestFallback:
    def test_very_tight_budget_uses_truncation(self, summarizer):
        text = "A" * 500
        result = summarizer.summarize(text, max_chars=20)
        assert len(result) <= 23  # 20 + "..." margin
        assert isinstance(result, str)

    def test_result_never_exceeds_double_budget(self, summarizer):
        """Reasonable upper bound — summarizer is extractive, not perfect."""
        text = "Short sentence. " * 30
        result = summarizer.summarize(text, max_chars=80)
        assert len(result) <= 160  # 2x is generous upper bound


class TestSentenceScoring:
    def test_numeric_scores_higher(self, summarizer):
        numeric = "The total cost is $12,500 and completion is 2025-06-01."
        plain = "The meeting was okay today."
        assert summarizer._sentence_score(numeric) > summarizer._sentence_score(plain)

    def test_explicit_anchor_scores_higher(self, summarizer):
        anchored = "You must always remember this critical constraint."
        plain = "That sounds fine to me."
        assert summarizer._sentence_score(anchored) > summarizer._sentence_score(plain)
