"""Core package for Undogmatic experiments."""

from .control_samples import (  # noqa: F401 - re-exported for convenience
    ControlSample,
    VALID_LABELS,
    iter_control_samples,
    load_control_samples,
)
from .llm_scorer import LLMScorer, ScoreParsingError, ScoreResult, score_text

__all__ = [
    "LLMScorer",
    "ScoreParsingError",
    "ScoreResult",
    "score_text",
    "ControlSample",
    "VALID_LABELS",
    "iter_control_samples",
    "load_control_samples",
]
