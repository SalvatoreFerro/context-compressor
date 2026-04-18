# context-compressor

**Keep infinite conversations within any LLM context window — zero extra API calls.**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://github.com/SalvatoreFerro/context-compressor/actions/workflows/tests.yml/badge.svg)](https://github.com/SalvatoreFerro/context-compressor/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/SalvatoreFerro/context-compressor/branch/main/graph/badge.svg)](https://codecov.io/gh/SalvatoreFerro/context-compressor)

---

Every developer building LLM applications eventually hits the same wall: **the context window**. As conversations grow, you're forced to choose between silently truncating messages (losing critical context) or paying for ever-larger context windows.

`context-compressor` solves this with a smarter approach: it scores each message by importance and compresses the least valuable ones — keeping numbers, names, dates, code, and explicit instructions intact while dropping small talk and filler.

## 5-minute integration guide

Already have a working chatbot? Add context compression in 2 steps.

**Before** — your existing OpenAI code:

```python
import openai

client = openai.OpenAI(api_key="...")
messages = [{"role": "system", "content": "You are a helpful assistant."}]

while True:
    user_input = input("You: ")
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(model="gpt-4o", messages=messages)
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    # ❌ Problem: messages grows forever — eventually hits the context limit
```

**After** — swap 1 import, change 1 line:

```python
from context_compressor.adapters import OpenAIAdapter  # ← step 1

client = OpenAIAdapter(api_key="...", model="gpt-4o")  # ← step 2: same API surface
messages = [{"role": "system", "content": "You are a helpful assistant."}]

while True:
    user_input = input("You: ")
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(model="gpt-4o", messages=messages)
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    # ✅ messages is automatically compressed before each API call
    # ✅ Critical facts (numbers, names, dates, code) are always preserved
    # ✅ Zero extra API calls — compression is purely algorithmic
```

Same for Anthropic — replace `anthropic.Anthropic()` with `AnthropicAdapter()`.  
System prompt handling is automatic.

Full runnable examples: [`examples/openai_chatbot.py`](examples/openai_chatbot.py) · [`examples/anthropic_chatbot.py`](examples/anthropic_chatbot.py)

---

## Why not just use LangChain's ConversationSummaryMemory?

| | context-compressor | LangChain ConversationSummaryMemory | Naive truncation |
|---|---|---|---|
| **Extra LLM calls** | ❌ Zero | ✅ One per summary | ❌ Zero |
| **Semantic scoring** | ✅ Yes | ⚠️ No (summarizes everything equally) | ❌ No |
| **Fact preservation** | ✅ Numbers/dates/names protected | ⚠️ Can lose fine details | ❌ Oldest messages lost |
| **Deterministic** | ✅ Yes | ❌ No (LLM-generated summaries vary) | ✅ Yes |
| **Drop-in API** | ✅ 2 lines | ⚠️ Requires LangChain | ✅ |
| **Customizable** | ✅ Per signal | ❌ Limited | ❌ None |

## Installation

```bash
pip install context-compressor

# With OpenAI support
pip install "context-compressor[openai]"

# With Anthropic support
pip install "context-compressor[anthropic]"
```

## Quickstart

### Standalone (any LLM provider)

```python
from context_compressor import ContextCompressor, CompressorConfig

# Auto-configure for your model
compressor = ContextCompressor(CompressorConfig.for_model("gpt-4o"))

# Drop-in: compress before sending
compressed_messages = compressor.compress(messages)

# With stats
result = compressor.compress_with_stats(messages)
print(result)
# CompressionResult(ratio=0.31, saved=1847 tokens, kept=8, summarized=12, dropped=23)
```

### OpenAI drop-in

```python
from context_compressor.adapters import OpenAIAdapter

# Replace openai.OpenAI() with OpenAIAdapter()
client = OpenAIAdapter(api_key="...", model="gpt-4o")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,  # compressed automatically
)

# Inspect what happened
print(client.last_compression)
```

### Anthropic drop-in

```python
from context_compressor.adapters import AnthropicAdapter

client = AnthropicAdapter(api_key="...", model="claude-3-5-sonnet-20241022")

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=messages,
    max_tokens=1024,
)
```

## How it works

Every message receives an **importance score** from 0 to 1, computed in milliseconds with zero API calls:

| Signal | What it detects | Default weight |
|---|---|---|
| **Recency** | Distance from end of conversation (exponential decay) | 30% |
| **Numeric** | Numbers, dates, currencies, measurements | 20% |
| **Explicit** | "remember", "always", "must", "critical", "deadline" | 20% |
| **Named entity** | Proper nouns, capitalized sequences | 15% |
| **Code** | Code blocks, `def`, `class`, `import` | 10% |
| **Question** | Direct questions | 5% |

Based on the score, each message is:
- **Score ≥ 0.7** → kept verbatim
- **Score 0.3–0.7** → extractively summarized (most informative sentences preserved)
- **Score < 0.3** → dropped
- **Last N messages** → always kept (configurable)
- **System messages** → always kept (score = 1.0)

If the result is still over budget, the lowest-scored messages are iteratively dropped until it fits.

## Benchmark

On a 51-message conversation heavy with filler ("okay", "sounds good", "yep"):

```
context-compressor:  1263 → 113 tokens  (91% reduction)  |  0.5ms
naive truncation:    1263 → 452 tokens  (64% reduction)  |  0ms
```

On a 20-message fact-rich conversation (budget, deadlines, names, constraints):

```
context-compressor:  100% fact retention (6/6 key facts preserved)
naive truncation:    Drops oldest messages — critical facts silently lost
```

## Configuration

```python
from context_compressor import CompressorConfig
from context_compressor.scoring.signals import SignalWeights

config = CompressorConfig(
    max_tokens=4096,              # Hard token budget
    preserve_threshold=0.7,       # Score >= this: keep verbatim
    compress_threshold=0.3,       # Score < this: drop; between: summarize
    always_preserve_last_n=4,     # Always keep last N messages
    signal_weights=SignalWeights(
        recency_weight=0.30,
        numeric_weight=0.20,
        explicit_weight=0.20,
        named_entity_weight=0.15,
        code_weight=0.10,
        question_weight=0.05,
    ),
    recency_half_life=10,         # Messages 10 turns back score 0.5x recency
)
```

Presets for common models:

```python
config = CompressorConfig.for_model("gpt-4o")
config = CompressorConfig.for_model("claude-3-5-sonnet-20241022")
config = CompressorConfig.for_model("gpt-3.5-turbo")
```

## Debug: explain compression decisions

```python
explanation = compressor.explain(messages)
for msg in explanation:
    print(f"[{msg['fate']:20s}] score={msg['final_score']:.3f}  {msg['content_preview']}")

# [keep (protected)     ] score=1.000  You are a helpful assistant.
# [drop                 ] score=0.112  okay sounds good...
# [keep (high score)    ] score=0.847  Remember: budget is $85,000 and deadline is...
# [keep (protected)     ] score=0.621  What should we prioritize first?
```

## Custom summarizer

Plug in your own LLM-based abstractive summarizer:

```python
from context_compressor.compression.summarizer import Summarizer

class MyLLMSummarizer:
    def summarize(self, text: str, max_chars: int = 200) -> str:
        # Call your preferred LLM here
        return your_llm.summarize(text, max_length=max_chars)

compressor = ContextCompressor(config, summarizer=MyLLMSummarizer())
```

## Running tests

```bash
pip install "context-compressor[dev]"
pytest tests/unit/
python tests/benchmark/benchmark_compression.py
```

## License

MIT
