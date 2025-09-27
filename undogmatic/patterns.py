"""Text normalization and regex utilities for Portuguese legal theses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Pattern

NORMALIZE_N_DEGREE = re.compile(r"\bn\s*[\.º°]?\s*[oº°]\b", re.IGNORECASE)
THEME_PATTERN = re.compile(r"\btema\s*(n[ºo]\s*)?\d{1,4}\b", re.IGNORECASE)
TRIBUNAL_PATTERN = re.compile(
    r"\b(?:stf|stj|supremo\s+tribunal\s+federal|superior\s+tribunal\s+de\s+justica)\b",
    re.IGNORECASE,
)
BOOSTER_PATTERN = re.compile(
    r"\b(obviamente|indiscutivelmente|inequivocamente|encerra\s+a\s+questao)\b",
    re.IGNORECASE,
)
HEDGE_PATTERN = re.compile(
    r"\b(em\s+regra|salvo|ressalvado|em\s+principio|via\s+de\s+regra)\b",
    re.IGNORECASE,
)
EXPLANATORY_CONNECTORS = (
    "pois",
    "porque",
    "porquanto",
    "de modo que",
    "de forma que",
    "aplica-se a",
    "aplica-se quando",
    "nao se aplica",
    "não se aplica",
)


def normalize_text(value: str) -> str:
    """Normalize Portuguese legal text for downstream scoring."""

    value = NORMALIZE_N_DEGREE.sub("nº", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def remove_authority_markers(value: str) -> str:
    """Remove explicit mentions to courts and theme labels."""

    without_court = TRIBUNAL_PATTERN.sub("", value)
    without_theme = THEME_PATTERN.sub("", without_court)
    normalized_spaces = re.sub(r"\s+", " ", without_theme)
    return normalized_spaces.strip()


@dataclass(frozen=True)
class PatternSummary:
    """Convenience container with regex hit counts for debugging/tests."""

    bare_citations: int
    tribunals: int
    boosters: int
    hedges: int


def count_pattern_hits(text: str, patterns: Iterable[Pattern[str]]) -> int:
    """Count total occurrences for a set of patterns."""

    total = 0
    for pattern in patterns:
        total += len(pattern.findall(text))
    return total


def summarize_patterns(text: str) -> PatternSummary:
    """Return counts for key surface patterns used in scoring."""

    normalized = normalize_text(text)
    bare_citations = len(THEME_PATTERN.findall(normalized))
    tribunal_hits = len(TRIBUNAL_PATTERN.findall(normalized))
    booster_hits = len(BOOSTER_PATTERN.findall(normalized))
    hedge_hits = len(HEDGE_PATTERN.findall(normalized))
    return PatternSummary(
        bare_citations=bare_citations,
        tribunals=tribunal_hits,
        boosters=booster_hits,
        hedges=hedge_hits,
    )


def contains_explanatory_connector(text: str) -> bool:
    """Check whether the text includes any explanatory connector."""

    lowered = normalize_text(text).lower()
    return any(connector in lowered for connector in EXPLANATORY_CONNECTORS)
