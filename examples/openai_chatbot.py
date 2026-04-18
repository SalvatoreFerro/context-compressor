"""
Real-world example: OpenAI chatbot with automatic context compression.

This script runs a terminal chatbot that never hits the context window limit,
no matter how long the conversation gets.

Requirements:
    pip install "context-compressor[openai]"

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/openai_chatbot.py
"""

import os
from context_compressor.adapters import OpenAIAdapter
from context_compressor.compression.config import CompressorConfig

# ── Setup ──────────────────────────────────────────────────────────────────────

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("Set the OPENAI_API_KEY environment variable first.")

# Drop-in replacement for openai.OpenAI()
# We cap at 2048 tokens to demonstrate compression kicking in quickly.
# In production, use CompressorConfig.for_model("gpt-4o") for the real limit.
client = OpenAIAdapter(
    api_key=api_key,
    model="gpt-4o",
    config=CompressorConfig(
        max_tokens=2048,
        preserve_threshold=0.7,
        compress_threshold=0.3,
        always_preserve_last_n=4,
    ),
)

# Conversation history — grows with every turn
messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant. "
            "Answer concisely and clearly."
        ),
    }
]

print("OpenAI chatbot with context-compressor")
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

    # Call API — compression happens automatically inside the adapter
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=512,
        temperature=0.7,
    )

    assistant_reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_reply})

    print(f"\nAssistant: {assistant_reply}")

    # Show a brief compression hint if compression kicked in
    if client.last_compression and client.last_compression.token_savings > 0:
        c = client.last_compression
        print(f"  [context-compressor: {c.token_savings} tokens saved | type 'stats' for details]")
