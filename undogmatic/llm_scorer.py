"""LLM-backed ShameScore computation utilities."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Optional, Protocol
from uuid import uuid4

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from . import prompts

TEMPERATURE = 0.0
TOP_P = 1.0


class CompletionClient(Protocol):
    """Protocol for chat completion clients."""

    def complete(self, *, system: str, user: str) -> str: ...


class OpenAIChatClient:
    """Minimal OpenAI Chat Completions client using the REST API."""

    def __init__(
        self, *, api_key: str, model: str, base_url: str | None = None, timeout: float = 60.0
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1/chat/completions"
        self.timeout = timeout
        self._session = requests.Session()

    def complete(self, *, system: str, user: str) -> str:  # pragma: no cover - thin wrapper
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = self._session.post(
            self.base_url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


class ScoreParsingError(RuntimeError):
    """Raised when the LLM response cannot be parsed into the expected schema."""


class ScoreResult(BaseModel):
    shame_score: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=100)
    rationale: str

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - passthrough
        return super().model_dump(*args, **kwargs)


@dataclass
class LoggedInteraction:
    """Metadata about a logged scorer interaction."""

    directory: Path
    prompt_path: Path
    response_path: Path


class LLMScorer:
    """High-level interface for computing ShameScores via an LLM."""

    def __init__(
        self,
        *,
        client: CompletionClient | None = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        log_dir: Path | str | None = None,
        max_retries: int = 2,
    ) -> None:
        load_dotenv()
        self.provider = provider or os.getenv("LLM_PROVIDER", "openai")
        self.model = model or os.getenv("LLM_MODEL")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.log_dir = Path(log_dir or "runs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.log_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries

        if client is not None:
            self.client = client
        else:
            self.client = self._build_client()

    def _build_client(self) -> CompletionClient:
        if self.provider.lower() != "openai":
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        if not self.api_key or not self.model:
            raise ValueError("LLM_API_KEY and LLM_MODEL must be set for OpenAI provider")
        return OpenAIChatClient(api_key=self.api_key, model=self.model, base_url=self.base_url)

    def score_text(self, text: str, *, metadata: Optional[Dict[str, Any]] = None) -> ScoreResult:
        system_prompt = prompts.SYSTEM_PROMPT
        user_prompt = prompts.render_user_prompt(text)
        attempts = 0
        raw_response: str | None = None
        parsed: ScoreResult | None = None

        cached = self._read_cache(text)
        if cached is not None:
            self._log_interaction(
                system_prompt,
                user_prompt,
                None,
                cached,
                metadata,
                success=True,
                cache_hit=True,
            )
            return cached

        while attempts <= self.max_retries:
            raw_response = self.client.complete(system=system_prompt, user=user_prompt)
            try:
                payload = json.loads(raw_response)
                parsed = ScoreResult.model_validate(payload)
                break
            except (json.JSONDecodeError, ValidationError) as exc:
                attempts += 1
                if attempts > self.max_retries:
                    self._log_interaction(
                        system_prompt,
                        user_prompt,
                        raw_response,
                        parsed,
                        metadata,
                        success=False,
                        error=str(exc),
                    )
                    raise ScoreParsingError("Failed to parse LLM response") from exc
                continue

        assert parsed is not None  # for type checkers
        self._write_cache(text, parsed)
        self._log_interaction(
            system_prompt,
            user_prompt,
            raw_response,
            parsed,
            metadata,
            success=True,
        )
        return parsed

    def _log_interaction(
        self,
        system_prompt: str,
        user_prompt: str,
        raw_response: str | None,
        parsed: ScoreResult | None,
        metadata: Optional[Dict[str, Any]],
        *,
        success: bool,
        error: Optional[str] = None,
        cache_hit: bool = False,
    ) -> LoggedInteraction:
        timestamp = datetime.now(timezone.utc)
        date_dir = self.log_dir / timestamp.strftime("%Y%m%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        slug_parts = []
        if metadata:
            for key in ("id", "variant"):
                value = metadata.get(key)
                if value:
                    slug_parts.append(str(value))
        slug = "-".join(slug_parts) if slug_parts else uuid4().hex[:8]
        entry_dir = date_dir / f"{timestamp.strftime('%H%M%S')}-{slug}"
        entry_dir.mkdir(parents=True, exist_ok=True)

        prompt_path = entry_dir / "prompt.json"
        response_path = entry_dir / "response.json"

        prompt_payload = {
            "system": system_prompt,
            "user": user_prompt,
            "metadata": metadata or {},
        }
        response_payload: Dict[str, Any] = {
            "raw": raw_response,
            "parsed": parsed.model_dump() if parsed else None,
            "success": success,
            "error": error,
            "cache_hit": cache_hit,
        }
        prompt_path.write_text(
            json.dumps(prompt_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        response_path.write_text(
            json.dumps(response_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return LoggedInteraction(
            directory=entry_dir, prompt_path=prompt_path, response_path=response_path
        )

    def _cache_path(self, text: str) -> Path:
        digest = sha1(text.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, text: str) -> ScoreResult | None:
        cache_path = self._cache_path(text)
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return ScoreResult.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValidationError):
            # Corrupted cache entries are ignored and removed to avoid repeated failures.
            try:
                cache_path.unlink()
            except OSError:
                pass
            return None

    def _write_cache(self, text: str, result: ScoreResult) -> None:
        cache_path = self._cache_path(text)
        cache_payload = result.model_dump()
        cache_path.write_text(
            json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def score_text(text: str, *, metadata: Optional[Dict[str, Any]] = None) -> ScoreResult:
    """Convenience function that proxies to a default scorer instance."""

    scorer = LLMScorer()
    return scorer.score_text(text, metadata=metadata)
