from __future__ import annotations

import json
from pathlib import Path

import pytest

from undogmatic.control_samples import (
    VALID_LABELS,
    ControlSample,
    iter_control_samples,
    load_control_samples,
)


def test_iter_control_samples_custom_path(tmp_path: Path) -> None:
    data = [
        {"id": "a", "label": "hubristic", "text": "Tema 1 resolve tudo."},
        {"id": "b", "label": "neutral", "text": "Audiência designada."},
    ]
    path = tmp_path / "controls.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in data), encoding="utf-8")

    samples = list(iter_control_samples(path))
    assert [sample.id for sample in samples] == ["a", "b"]
    assert all(isinstance(sample, ControlSample) for sample in samples)


def test_load_control_samples_default_dataset() -> None:
    samples = load_control_samples()
    assert samples, "expected curated control samples to be available"
    labels = {sample.label for sample in samples}
    assert labels.issubset(set(VALID_LABELS))
    assert all(sample.text for sample in samples)


def test_iter_control_samples_rejects_unknown_labels(tmp_path: Path) -> None:
    path = tmp_path / "controls.jsonl"
    path.write_text(json.dumps({"id": "x", "label": "other", "text": "..."}), encoding="utf-8")

    with pytest.raises(ValueError):
        list(iter_control_samples(path))
