"""Unit tests for ContextCompressor."""

import pytest

from context_compressor.compression.compressor import ContextCompressor
from context_compressor.compression.config import CompressorConfig


def make_messages(n: int, role: str = "user", content: str = "hello world") -> list:
    return [{"role": role, "content": content} for _ in range(n)]


def make_conversation(pairs: int) -> list:
    """Create a realistic back-and-forth conversation."""
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(pairs):
        messages.append({"role": "user", "content": f"Question {i}: what is the answer?"})
        messages.append({"role": "assistant", "content": f"Answer {i}: here is the explanation."})
    return messages


@pytest.fixture
def compressor():
    return ContextCompressor(CompressorConfig(max_tokens=512, always_preserve_last_n=2))


class TestFastPath:
    def test_short_conversation_unchanged(self):
        compressor = ContextCompressor(CompressorConfig(max_tokens=10000))
        messages = make_conversation(3)
        result = compressor.compress_with_stats(messages)
        assert result.compression_ratio == 1.0
        assert result.messages == messages

    def test_empty_messages_unchanged(self):
        compressor = ContextCompressor()
        result = compressor.compress_with_stats([])
        assert result.messages == []


class TestAlwaysPreserve:
    def test_system_message_always_kept(self, compressor):
        messages = [
            {"role": "system", "content": "Critical system prompt."},
        ] + make_messages(20)
        result = compressor.compress_with_stats(messages)
        system_msgs = [m for m in result.messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "Critical system prompt."

    def test_last_n_always_kept(self, compressor):
        messages = make_conversation(10)
        last_two = messages[-2:]
        result = compressor.compress_with_stats(messages)
        # Last 2 messages should appear verbatim (content preserved)
        result_contents = [m["content"] for m in result.messages]
        for msg in last_two:
            assert msg["content"] in result_contents


class TestCompressionStats:
    def test_stats_populated(self):
        compressor = ContextCompressor(CompressorConfig(max_tokens=512))
        messages = make_conversation(10)
        result = compressor.compress_with_stats(messages)
        assert result.original_token_count > 0
        assert result.compressed_token_count > 0
        assert result.compressed_token_count <= result.original_token_count

    def test_compression_ratio_range(self):
        compressor = ContextCompressor(CompressorConfig(max_tokens=512))
        messages = make_conversation(10)
        result = compressor.compress_with_stats(messages)
        assert 0.0 < result.compression_ratio <= 1.0

    def test_token_savings_positive(self):
        compressor = ContextCompressor(CompressorConfig(max_tokens=512))
        messages = make_conversation(15)
        result = compressor.compress_with_stats(messages)
        assert result.token_savings >= 0


class TestHighImportancePreserved:
    def test_numeric_content_preserved(self):
        compressor = ContextCompressor(CompressorConfig(max_tokens=300, always_preserve_last_n=1))
        important_msg = {
            "role": "user",
            "content": (
                "Remember: the project budget is $150,000"
                " and deadline is 2025-12-31. Critical constraint."
            ),
        }
        filler = make_messages(20, content="okay sounds good")
        messages = [important_msg] + filler
        result = compressor.compress_with_stats(messages)
        # Important message content should appear (verbatim or summarized)
        all_content = " ".join(m["content"] for m in result.messages)
        assert "$150,000" in all_content or "150" in all_content

    def test_code_blocks_preserved(self):
        compressor = ContextCompressor(CompressorConfig(max_tokens=500, always_preserve_last_n=1))
        code_msg = {
            "role": "assistant",
            "content": (
                "```python\ndef authenticate(user_id: str) -> bool:\n"
                "    return db.check(user_id)\n```"
            ),
        }
        filler = make_messages(15, content="ok thanks")
        messages = filler + [code_msg]
        result = compressor.compress_with_stats(messages)
        all_content = " ".join(m["content"] for m in result.messages)
        assert "authenticate" in all_content or "python" in all_content.lower()


class TestExplainMethod:
    def test_explain_returns_all_messages(self):
        compressor = ContextCompressor()
        messages = make_conversation(3)
        explanation = compressor.explain(messages)
        assert len(explanation) == len(messages)

    def test_explain_has_required_fields(self):
        compressor = ContextCompressor()
        messages = [{"role": "user", "content": "hello"}]
        explanation = compressor.explain(messages)
        assert "fate" in explanation[0]
        assert "final_score" in explanation[0]
        assert "role" in explanation[0]

    def test_system_fate_is_protected(self):
        compressor = ContextCompressor()
        messages = [{"role": "system", "content": "You are helpful."}]
        explanation = compressor.explain(messages)
        assert "keep" in explanation[0]["fate"]


class TestConfig:
    def test_for_model_gpt4o(self):
        config = CompressorConfig.for_model("gpt-4o")
        assert config.max_tokens > 0

    def test_for_model_claude(self):
        config = CompressorConfig.for_model("claude-3-5-sonnet-20241022")
        assert config.max_tokens > 0

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError):
            CompressorConfig(max_tokens=50).validate()

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            CompressorConfig(
                preserve_threshold=0.2, compress_threshold=0.8
            ).validate()


class TestExports:
    def test_compression_result_importable(self):
        from context_compressor import CompressionResult
        assert CompressionResult is not None

    def test_signal_weights_importable(self):
        from context_compressor import SignalWeights
        assert SignalWeights is not None
