"""Core package for Undogmatic experiments."""

from .llm_scorer import LLMScorer, ScoreParsingError, ScoreResult, score_text

__all__ = ["LLMScorer", "ScoreParsingError", "ScoreResult", "score_text"]
