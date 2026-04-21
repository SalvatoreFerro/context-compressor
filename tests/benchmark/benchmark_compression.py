"""
Reproducible benchmark: context-compressor vs. naive truncation.

Measures:
- Token reduction (%)
- Information retention (key facts preserved after compression)
- Processing time (ms per compression)

Run with:
    python tests/benchmark/benchmark_compression.py

No API keys required — fully offline.
"""

from __future__ import annotations

import time

from context_compressor.compression.compressor import ContextCompressor
from context_compressor.compression.config import CompressorConfig
from context_compressor.compression.token_counter import count_messages_tokens

# ── Test conversations ──────────────────────────────────────────────────────

FACT_RICH_CONVERSATION: list[dict] = [
    {"role": "system", "content": "You are a project management assistant."},
    {"role": "user", "content": "Hi, I need help with our Q4 project."},
    {"role": "assistant", "content": "Sure, I'm happy to help. What's the project about?"},
    {
        "role": "user",
        "content": "It's a data pipeline. Our budget is $85,000 and deadline is 2025-12-01.",
    },
    {"role": "assistant", "content": "Got it. $85,000 budget, December 1st deadline."},
    {
        "role": "user",
        "content": "We're using Python 3.12 and Apache Kafka for the streaming layer.",
    },
    {"role": "assistant", "content": "Good choices. Kafka handles high-throughput streaming well."},
    {"role": "user", "content": "sounds good"},
    {"role": "assistant", "content": "Let me know if you have questions."},
    {"role": "user", "content": "ok"},
    {"role": "assistant", "content": "Sure."},
    {"role": "user", "content": "yep"},
    {"role": "assistant", "content": "Okay."},
    {
        "role": "user",
        "content": "Critical: the team lead is Maria Chen, she must approve all PRs.",
    },
    {"role": "assistant", "content": "Noted. Maria Chen must approve all pull requests."},
    {"role": "user", "content": "yeah"},
    {"role": "assistant", "content": "Understood."},
    {"role": "user", "content": "Also, remember we have a hard constraint: no AWS services."},
    {"role": "assistant", "content": "Constraint noted: no AWS services allowed."},
    {"role": "user", "content": "What should we prioritize first?"},
]

KEY_FACTS = [
    "$85,000",
    "2025-12-01",
    "Python",
    "Kafka",
    "Maria Chen",
    "no AWS",
]

LONG_CHATTER_CONVERSATION: list[dict] = [
    {"role": "system", "content": "You are a helpful assistant."},
] + [
    {
        "role": "user" if i % 2 == 0 else "assistant",
        "content": f"Message {i}: " + ("okay " * 15),
    }
    for i in range(50)
]


def naive_truncation(messages: list[dict], max_tokens: int) -> list[dict]:
    """Baseline: keep only the most recent messages that fit."""
    result = []
    tokens = 0
    for msg in reversed(messages):
        t = count_messages_tokens([msg])
        if tokens + t > max_tokens:
            break
        result.insert(0, msg)
        tokens += t
    return result


def check_fact_retention(messages: list[dict], facts: list[str]) -> dict:
    all_content = " ".join(m.get("content", "") for m in messages)
    retained = [f for f in facts if f in all_content]
    return {
        "retained": retained,
        "dropped": [f for f in facts if f not in retained],
        "retention_rate": len(retained) / len(facts),
    }


def run_benchmark():
    print("=" * 60)
    print("context-compressor benchmark")
    print("=" * 60)

    configs = [
        ("Tight (512 tokens)", CompressorConfig(max_tokens=512)),
        ("Medium (1024 tokens)", CompressorConfig(max_tokens=1024)),
        ("Relaxed (2048 tokens)", CompressorConfig(max_tokens=2048)),
    ]

    conversations = [
        ("Fact-rich (20 msgs)", FACT_RICH_CONVERSATION, KEY_FACTS),
        ("Long chatter (51 msgs)", LONG_CHATTER_CONVERSATION, []),
    ]

    for conv_name, messages, facts in conversations:
        original_tokens = count_messages_tokens(messages)
        print(f"\n── {conv_name} ({original_tokens} tokens original) ──")

        for config_name, config in configs:
            compressor = ContextCompressor(config)

            # Benchmark context-compressor
            start = time.perf_counter()
            result = compressor.compress_with_stats(messages)
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Benchmark naive truncation
            naive = naive_truncation(messages, config.max_tokens)
            naive_tokens = count_messages_tokens(naive)

            print(f"\n  {config_name}")
            print("  context-compressor:")
            orig = result.original_token_count
            comp = result.compressed_token_count
            ratio = result.compression_ratio
            print(f"    tokens:    {orig} → {comp} ({ratio:.0%})")
            kept = result.messages_kept
            summ = result.messages_summarized
            drop = result.messages_dropped
            print(f"    kept/sum/drop: {kept}/{summ}/{drop}")
            print(f"    time:      {elapsed_ms:.1f}ms")

            if facts:
                retention = check_fact_retention(result.messages, facts)
                rate = retention["retention_rate"]
                n_ret = len(retention["retained"])
                print(f"    fact retention: {rate:.0%} ({n_ret}/{len(facts)} facts)")
                if retention["dropped"]:
                    print(f"    dropped facts: {retention['dropped']}")

            print("  naive truncation:")
            naive_ratio = naive_tokens / original_tokens
            print(f"    tokens:    {original_tokens} → {naive_tokens} ({naive_ratio:.0%})")
            print(f"    messages:  {len(messages)} → {len(naive)}")
            if facts:
                naive_retention = check_fact_retention(naive, facts)
                n_rate = naive_retention["retention_rate"]
                naive_ret_count = len(naive_retention["retained"])
                print(
                    f"    fact retention: {n_rate:.0%} "
                    f"({naive_ret_count}/{len(facts)} facts)"
                )

    print("\n" + "=" * 60)
    print("Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
