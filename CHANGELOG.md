# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-04-19

### Added
- `ImportanceScorer`: deterministic message scoring with 6 configurable signals
  (recency, numeric, explicit, named entity, code, question)
- `ContextCompressor`: main compression pipeline with preserve/summarize/drop tiers
- `ExtractiveSummarizer`: zero-LLM-call extractive summarization
- `CompressorConfig`: full configuration with per-model presets
- `SignalWeights`: customizable signal weights for domain-specific tuning
- `OpenAIAdapter`: drop-in wrapper for `openai.OpenAI()` with auto-compression
- `AnthropicAdapter`: drop-in wrapper for `anthropic.Anthropic()` with system prompt handling
- `compressor.explain()`: debug tool for understanding compression decisions
- `compress_with_stats()`: compression with full telemetry (ratio, savings, counts)
- 55 unit tests, 91% coverage
- Reproducible offline benchmark vs naive truncation
