"""
Unit tests for OpenAI and Anthropic adapters.

Uses unittest.mock to avoid real API calls.
Tests verify that:
1. Adapters compress messages before forwarding
2. last_compression is populated after each call
3. System messages are handled correctly (Anthropic)
4. ImportError is raised with helpful message when SDK not installed
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from context_compressor.compression.config import CompressorConfig

# ── Helpers ────────────────────────────────────────────────────────────────────

def make_long_conversation(n: int = 30) -> list:
    msgs = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(n):
        msgs.append({"role": "user", "content": f"Message {i}: " + "hello " * 10})
        msgs.append({"role": "assistant", "content": f"Reply {i}: " + "sure " * 10})
    return msgs


# ── OpenAI Adapter ─────────────────────────────────────────────────────────────

class TestOpenAIAdapter:
    def test_import_error_without_openai(self):
        """Raises ImportError with helpful message when openai not installed."""
        with patch.dict(sys.modules, {"openai": None}):
            from importlib import reload

            import context_compressor.adapters.openai_adapter as mod
            with pytest.raises(ImportError, match="openai package"):
                reload(mod)
                mod.OpenAIAdapter()

    def test_compress_called_before_api(self):
        """Messages are compressed before being sent to the API."""
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock()

        with patch.dict(sys.modules, {"openai": mock_openai}):
            from context_compressor.adapters.openai_adapter import OpenAIAdapter
            adapter = OpenAIAdapter(
                api_key="test-key",
                model="gpt-4o",
                config=CompressorConfig(max_tokens=512),
            )
            long_messages = make_long_conversation(20)
            adapter.chat.completions.create(model="gpt-4o", messages=long_messages)

            # The API was called
            assert mock_client.chat.completions.create.called
            # Messages passed to API are <= original (compression happened or fast-path)
            call_args = mock_client.chat.completions.create.call_args
            sent_messages = (
                call_args.kwargs.get("messages")
                or (call_args.args[0] if call_args.args else [])
            )
            assert len(sent_messages) <= len(long_messages)

    def test_last_compression_populated(self):
        """last_compression is set after each call."""
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock()

        with patch.dict(sys.modules, {"openai": mock_openai}):
            from context_compressor.adapters.openai_adapter import OpenAIAdapter
            adapter = OpenAIAdapter(api_key="test", config=CompressorConfig(max_tokens=512))
            assert adapter.last_compression is None

            adapter.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hello"}],
            )
            assert adapter.last_compression is not None

    def test_raw_client_accessible(self):
        """raw_client exposes underlying openai client."""
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict(sys.modules, {"openai": mock_openai}):
            from context_compressor.adapters.openai_adapter import OpenAIAdapter
            adapter = OpenAIAdapter(api_key="test", config=CompressorConfig(max_tokens=512))
            assert adapter.raw_client is mock_client

    def test_kwargs_passed_through(self):
        """Extra kwargs (temperature, max_tokens, etc.) reach the underlying API."""
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock()

        with patch.dict(sys.modules, {"openai": mock_openai}):
            from context_compressor.adapters.openai_adapter import OpenAIAdapter
            adapter = OpenAIAdapter(api_key="test", config=CompressorConfig(max_tokens=512))
            adapter.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.7,
                max_tokens=256,
            )
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs.get("temperature") == 0.7
            assert call_kwargs.get("max_tokens") == 256


# ── Anthropic Adapter ──────────────────────────────────────────────────────────

class TestAnthropicAdapter:
    def test_import_error_without_anthropic(self):
        """Raises ImportError with helpful message when anthropic not installed."""
        with patch.dict(sys.modules, {"anthropic": None}):
            from importlib import reload

            import context_compressor.adapters.anthropic_adapter as mod
            with pytest.raises(ImportError, match="anthropic package"):
                reload(mod)
                mod.AnthropicAdapter()

    def test_system_message_extracted(self):
        """System messages are extracted and passed as `system` kwarg."""
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock()

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            from context_compressor.adapters.anthropic_adapter import AnthropicAdapter
            adapter = AnthropicAdapter(
                api_key="test",
                config=CompressorConfig(max_tokens=512),
            )
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ]
            adapter.messages.create(
                model="claude-3-5-sonnet-20241022",
                messages=messages,
                max_tokens=1024,
            )
            call_kwargs = mock_client.messages.create.call_args.kwargs
            # System prompt extracted correctly
            assert call_kwargs.get("system") == "You are a helpful assistant."
            # System message not in messages list
            sent_messages = call_kwargs.get("messages", [])
            assert all(m.get("role") != "system" for m in sent_messages)

    def test_last_compression_populated(self):
        """last_compression is set after each call."""
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock()

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            from context_compressor.adapters.anthropic_adapter import AnthropicAdapter
            adapter = AnthropicAdapter(api_key="test", config=CompressorConfig(max_tokens=512))
            assert adapter.last_compression is None

            adapter.messages.create(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=100,
            )
            assert adapter.last_compression is not None

    def test_existing_system_kwarg_not_overridden(self):
        """If caller already passes `system=`, it is not overridden."""
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock()

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            from context_compressor.adapters.anthropic_adapter import AnthropicAdapter
            adapter = AnthropicAdapter(api_key="test", config=CompressorConfig(max_tokens=512))
            adapter.messages.create(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
                system="Custom system prompt.",
            )
            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert call_kwargs.get("system") == "Custom system prompt."

    def test_raw_client_accessible(self):
        """raw_client exposes underlying anthropic client."""
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            from context_compressor.adapters.anthropic_adapter import AnthropicAdapter
            adapter = AnthropicAdapter(api_key="test", config=CompressorConfig(max_tokens=512))
            assert adapter.raw_client is mock_client

    def test_multiple_system_messages_concatenated(self):
        """Multiple system messages should be concatenated, not overwritten."""
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock()

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            from context_compressor.adapters.anthropic_adapter import AnthropicAdapter
            adapter = AnthropicAdapter(
                api_key="test",
                config=CompressorConfig(max_tokens=4096),
            )
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "system", "content": "Always respond in English."},
                {"role": "user", "content": "Hello"},
            ]
            adapter.messages.create(
                model="claude-3-5-sonnet-20241022",
                messages=messages,
                max_tokens=1024,
            )
            call_kwargs = mock_client.messages.create.call_args.kwargs
            system_val = call_kwargs.get("system")
            assert "You are a helpful assistant." in system_val
            assert "Always respond in English." in system_val
