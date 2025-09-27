"""Helpers to load curated control prompts for ShameScore calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

__all__ = ["ControlSample", "VALID_LABELS", "load_control_samples", "iter_control_samples"]

DEFAULT_PATH = Path("data/curated/control_samples.jsonl")
VALID_LABELS: Sequence[str] = ("hubristic", "neutral", "grounded")


@dataclass(frozen=True)
class ControlSample:
    """Single text snippet with an expected hubris profile."""

    id: str
    label: str
    text: str
    notes: str | None = None


def _coerce_path(path: Path | str | None) -> Path:
    if path is None:
        return DEFAULT_PATH
    if isinstance(path, Path):
        return path
    return Path(path)


def iter_control_samples(path: Path | str | None = None) -> Iterator[ControlSample]:
    """Yield control samples from a JSONL file lazily."""

    target = _coerce_path(path)
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            label = payload["label"].strip()
            if label not in VALID_LABELS:
                raise ValueError(f"Unknown control label: {label}")
            notes = payload.get("notes")
            if isinstance(notes, str):
                notes = notes.strip() or None
            yield ControlSample(
                id=payload["id"],
                label=label,
                text=payload["text"].strip(),
                notes=notes,
            )


def load_control_samples(path: Path | str | None = None) -> List[ControlSample]:
    """Return a list of control samples from the curated dataset."""

    return list(iter_control_samples(path))
