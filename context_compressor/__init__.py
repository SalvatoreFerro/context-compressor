"""
context-compressor: Intelligent conversation context compression for LLM applications.

Keep infinite conversations within any context window — zero extra API calls,
zero information loss on what matters.
"""

from context_compressor.compression.compressor import CompressionResult, ContextCompressor
from context_compressor.compression.config import CompressorConfig
from context_compressor.scoring.scorer import ImportanceScorer
from context_compressor.scoring.signals import SignalWeights

__all__ = [
    "ContextCompressor",
    "CompressorConfig",
    "CompressionResult",
    "ImportanceScorer",
    "SignalWeights",
]
__version__ = "0.1.0"
