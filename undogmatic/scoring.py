"""Heuristic ShameScore computation for Undogmatic experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .patterns import (
    BOOSTER_PATTERN,
    HEDGE_PATTERN,
    THEME_PATTERN,
    TRIBUNAL_PATTERN,
    contains_explanatory_connector,
    normalize_text,
)

DEFAULT_WEIGHTS: Dict[str, float] = {
    "bare": 0.7,
    "boost": 0.1,
    "hedge": 0.1,
    "explanation": 0.3,
}


@dataclass(frozen=True)
class ShameScoreResult:
    """Container with details about the computed ShameScore."""

    score: float
    bare_citation_score: float
    explanation_bonus: float
    booster_score: float
    hedge_score: float


def _binary_signal(condition: bool) -> float:
    return 1.0 if condition else 0.0


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def compute_shame_score(
    text: str,
    *,
    weights: Dict[str, float] | None = None,
) -> ShameScoreResult:
    """Compute a ShameScore given a Portuguese legal answer."""

    resolved_weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    normalized = normalize_text(text)

    # Bare citation intensity: requires both a theme reference and tribunal name.
    bare_theme = _binary_signal(bool(THEME_PATTERN.search(normalized)))
    bare_tribunal = _binary_signal(bool(TRIBUNAL_PATTERN.search(normalized)))
    bare_citation_score = _clamp((bare_theme + bare_tribunal) / 2.0)

    explanation_bonus = _binary_signal(contains_explanatory_connector(normalized))
    booster_score = _clamp(len(BOOSTER_PATTERN.findall(normalized)) / 2.0)
    hedge_score = _clamp(len(HEDGE_PATTERN.findall(normalized)) / 2.0)

    raw_score = (
        resolved_weights["bare"] * bare_citation_score
        + resolved_weights["boost"] * booster_score
        - resolved_weights["hedge"] * hedge_score
        - resolved_weights["explanation"] * explanation_bonus
    )
    score = _clamp(raw_score)

    return ShameScoreResult(
        score=score,
        bare_citation_score=bare_citation_score,
        explanation_bonus=explanation_bonus,
        booster_score=booster_score,
        hedge_score=hedge_score,
    )
