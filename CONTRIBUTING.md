# Contributing to context-compressor

Thank you for considering contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/SalvatoreFerro/context-compressor
cd context-compressor
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/unit/
```

## Code style

This project uses [ruff](https://github.com/astral-sh/ruff) for linting:

```bash
ruff check context_compressor/
```

## How to contribute

1. Fork the repo and create a branch: `git checkout -b feat/your-feature`
2. Make your changes
3. Add or update tests — coverage should not decrease
4. Run the full test suite: `pytest tests/unit/`
5. Open a pull request with a clear description

## Areas where contributions are especially welcome

- New signal types for the importance scorer
- Abstractive summarizer integrations (GPT-4o-mini, Claude Haiku)
- Additional provider adapters (Gemini, Ollama, Mistral)
- Performance improvements on very long conversations
- Better sentence splitting (abbreviations, URLs)

## Reporting bugs

Open an issue on GitHub with:
- Python version
- A minimal reproducible example
- Expected vs actual behavior
