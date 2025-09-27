from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pytest

from undogmatic.llm_scorer import LLMScorer, ScoreParsingError
from undogmatic import prompts


class QueueClient:
    def __init__(self, responses: Iterable[str]) -> None:
        self._responses = list(responses)

    def complete(self, *, system: str, user: str) -> str:  # pragma: no cover - simple passthrough
        if not self._responses:
            raise RuntimeError("No more responses queued")
        return self._responses.pop(0)


def test_render_user_prompt_includes_text() -> None:
    prompt = prompts.render_user_prompt("Texto de teste")
    assert "Texto de teste" in prompt
    assert "Avalie o texto" in prompt


def test_score_text_logs_and_parses(tmp_path: Path) -> None:
    response = json.dumps({"shame_score": 70, "confidence": 80, "rationale": "Pouca análise."})
    scorer = LLMScorer(client=QueueClient([response]), log_dir=tmp_path)

    result = scorer.score_text(
        "STF Tema 123", metadata={"id": "STF-123", "variant": "authority_only"}
    )

    assert result.shame_score == 70
    assert result.confidence == 80
    date_dirs = list(tmp_path.iterdir())
    assert date_dirs, "expected log directory to be created"
    prompt_path = next(date_dirs[0].rglob("prompt.json"))
    response_path = next(date_dirs[0].rglob("response.json"))
    prompt_payload = json.loads(prompt_path.read_text(encoding="utf-8"))
    assert prompt_payload["metadata"]["id"] == "STF-123"
    response_payload = json.loads(response_path.read_text(encoding="utf-8"))
    assert response_payload["parsed"]["shame_score"] == 70


def test_score_text_retries_invalid_json(tmp_path: Path) -> None:
    bad = "{invalid"
    good = json.dumps({"shame_score": 10, "confidence": 90, "rationale": "Explica."})
    scorer = LLMScorer(client=QueueClient([bad, good]), log_dir=tmp_path)

    result = scorer.score_text("explica", metadata={"id": "case", "variant": "explained_only"})
    assert result.shame_score == 10


def test_score_text_raises_after_exhausting_retries(tmp_path: Path) -> None:
    response = "not-json"
    scorer = LLMScorer(
        client=QueueClient([response, response, response]), log_dir=tmp_path, max_retries=1
    )

    with pytest.raises(ScoreParsingError):
        scorer.score_text("falha", metadata={"id": "fail"})
