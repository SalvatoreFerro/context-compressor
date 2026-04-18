"""
Basic usage examples for context-compressor.

Run this file directly — no API keys required.
"""

from context_compressor import ContextCompressor, CompressorConfig

# ── Example 1: Basic compression ──────────────────────────────────────────────

print("=" * 60)
print("Example 1: Basic compression")
print("=" * 60)

messages = [
    {"role": "system", "content": "You are a project management assistant."},
    {"role": "user", "content": "Hi there, how are you?"},
    {"role": "assistant", "content": "I'm doing well! How can I help?"},
    {"role": "user", "content": "okay"},
    {"role": "assistant", "content": "Let me know when you're ready."},
    {"role": "user", "content": "yep"},
    {"role": "assistant", "content": "Sure."},
    {"role": "user", "content": "The project budget is $85,000. Deadline: 2025-12-01. Critical: no AWS services allowed."},
    {"role": "assistant", "content": "Got it. Budget $85,000, deadline December 1st, no AWS."},
    {"role": "user", "content": "ok cool"},
    {"role": "assistant", "content": "Understood."},
    {"role": "user", "content": "What should we start with?"},
]

compressor = ContextCompressor(CompressorConfig(max_tokens=512))
result = compressor.compress_with_stats(messages)

print(f"\nOriginal:   {result.original_token_count} tokens, {len(messages)} messages")
print(f"Compressed: {result.compressed_token_count} tokens, {len(result.messages)} messages")
print(f"Reduction:  {1 - result.compression_ratio:.0%} ({result.token_savings} tokens saved)")
print(f"Kept: {result.messages_kept} | Summarized: {result.messages_summarized} | Dropped: {result.messages_dropped}")

print("\nCompressed messages:")
for msg in result.messages:
    preview = msg["content"][:70] + ("..." if len(msg["content"]) > 70 else "")
    print(f"  [{msg['role']:9s}] {preview}")


# ── Example 2: Explain compression decisions ───────────────────────────────────

print("\n" + "=" * 60)
print("Example 2: Explain compression decisions")
print("=" * 60)

explanation = compressor.explain(messages)
print()
for item in explanation:
    print(f"  [{item['fate']:22s}] score={item['final_score']:.3f}  {item['content_preview']}")


# ── Example 3: Custom config for a technical assistant ────────────────────────

print("\n" + "=" * 60)
print("Example 3: Custom config (technical assistant)")
print("=" * 60)

from context_compressor.scoring.signals import SignalWeights

tech_config = CompressorConfig(
    max_tokens=1024,
    signal_weights=SignalWeights(
        recency_weight=0.20,
        numeric_weight=0.15,
        explicit_weight=0.20,
        named_entity_weight=0.10,
        code_weight=0.30,   # Code matters most for a dev assistant
        question_weight=0.05,
    ),
)

tech_compressor = ContextCompressor(tech_config)

code_conversation = [
    {"role": "system", "content": "You are a Python expert."},
    {"role": "user", "content": "How do I read a file?"},
    {"role": "assistant", "content": "Use `open()`: `with open('file.txt') as f: data = f.read()`"},
    {"role": "user", "content": "ok"},
    {"role": "assistant", "content": "Sure."},
    {"role": "user", "content": "ok"},
    {"role": "assistant", "content": "Got it."},
    {"role": "user", "content": "Now write me an async file reader:\n```python\nimport asyncio\nimport aiofiles\n\nasync def read_file(path: str) -> str:\n    async with aiofiles.open(path) as f:\n        return await f.read()\n```"},
    {"role": "assistant", "content": "That's correct. `aiofiles` is the standard library for async file I/O."},
    {"role": "user", "content": "How do I handle errors in that function?"},
]

result = tech_compressor.compress_with_stats(code_conversation)
print(f"\nCode-heavy conversation: {result.original_token_count} → {result.compressed_token_count} tokens")
print(f"Code messages preserved: check that async code block survived compression")

all_content = " ".join(m["content"] for m in result.messages)
print(f"  'aiofiles' in output: {'aiofiles' in all_content}")
print(f"  'async' in output:    {'async' in all_content}")
