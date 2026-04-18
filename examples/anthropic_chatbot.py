"""
Real-world example: Anthropic Claude chatbot with automatic context compression.

This script runs a terminal chatbot powered by Claude that never hits the
context window limit, no matter how long the conversation gets.

Requirements:
    pip install "context-compressor[anthropic]"

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/anthropic_chatbot.py
"""

import os
from context_compressor.adapters import AnthropicAdapter
from context_compressor.compression.config import CompressorConfig

# ── Setup ──────────────────────────────────────────────────────────────────────

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise SystemExit("Set the ANTHROPIC_API_KEY environment variable first.")

MODEL = "claude-3-5-sonnet-20241022"

# Drop-in replacement for anthropic.Anthropic()
# Cap at 4096 tokens to demonstrate compression.
# In production, use CompressorConfig.for_model(MODEL) for the real limit.
client = AnthropicAdapter(
    api_key=api_key,
    model=MODEL,
    config=CompressorConfig(
        max_tokens=4096,
        preserve_threshold=0.7,
        compress_threshold=0.3,
        always_preserve_last_n=4,
    ),
)

# Conversation history — the system prompt is extracted automatically
# by AnthropicAdapter and passed as the `system` parameter.
messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant. "
            "Answer concisely and clearly."
        ),
    }
]

print(f"Anthropic chatbot ({MODEL}) with context-compressor")
print("Type 'quit' to exit, 'stats' to see compression info.")
print("-" * 50)

# ── Chat loop ──────────────────────────────────────────────────────────────────

while True:
    try:
        user_input = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        break

    if not user_input:
        continue

    if user_input.lower() == "quit":
        print("Bye!")
        break

    if user_input.lower() == "stats":
        if client.last_compression:
            c = client.last_compression
            print(f"\n[Compression stats]")
            print(f"  Tokens:      {c.original_token_count} → {c.compressed_token_count}")
            print(f"  Saved:       {c.token_savings} tokens ({1 - c.compression_ratio:.0%} reduction)")
            print(f"  Messages:    kept={c.messages_kept}, summarized={c.messages_summarized}, dropped={c.messages_dropped}")
        else:
            print("[No compression has occurred yet — conversation fits in the window]")
        continue

    # Add user message to history
    messages.append({"role": "user", "content": user_input})

    # Call API — compression + system prompt extraction happen automatically
    response = client.messages.create(
        model=MODEL,
        messages=messages,
        max_tokens=512,
    )

    assistant_reply = response.content[0].text
    messages.append({"role": "assistant", "content": assistant_reply})

    print(f"\nClaude: {assistant_reply}")

    # Show a brief compression hint if compression kicked in
    if client.last_compression and client.last_compression.token_savings > 0:
        c = client.last_compression
        print(f"  [context-compressor: {c.token_savings} tokens saved | type 'stats' for details]")
