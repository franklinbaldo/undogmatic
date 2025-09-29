"""Shared text cleaning utilities for thesis processing."""

from __future__ import annotations

import re

THEME_PATTERN = re.compile(r"\bTema\s*(?:n[ºo]\s*)?\d{1,5}\b", flags=re.IGNORECASE)
TRIBUNAL_PATTERN = re.compile(
    r"\b(?:STF|STJ|Supremo Tribunal Federal|Superior Tribunal de Justiça)\b",
    flags=re.IGNORECASE,
)
WHITESPACE_PATTERN = re.compile(r"\s+")

__all__ = ["cleanse_tese", "normalize_whitespace"]


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim the string."""

    return WHITESPACE_PATTERN.sub(" ", text).strip()


def cleanse_tese(text: str) -> str:
    """Remove explicit tribunal/theme mentions while preserving substantive content."""

    without_theme = THEME_PATTERN.sub("", text)
    without_tribunal = TRIBUNAL_PATTERN.sub("", without_theme)
    return normalize_whitespace(without_tribunal)
