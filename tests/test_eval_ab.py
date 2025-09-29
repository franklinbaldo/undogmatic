from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from undogmatic.eval_ab import ABPair, compute_summary, run, score_pairs
from undogmatic.llm_scorer import LLMScorer


class StaticClient:
    def __init__(self, score: float) -> None:
        payload = json.dumps({"shame_score": score, "confidence": 90, "rationale": "ok"})
        self.responses = [payload, payload, payload, payload]

    def complete(self, *, system: str, user: str) -> str:  # pragma: no cover - deterministic
        return self.responses.pop(0)


def test_score_pairs_creates_csv(tmp_path: Path) -> None:
    def simple_scorer(text: str) -> dict:
        return {"shame_score": 10.0, "confidence": 90.0, "rationale": text}

    pairs = [ABPair(id="X", authority_only="A", explained_only="B")]
    df, csv_path = score_pairs(
        pairs,
        simple_scorer,
        run_label="unit-test",
        log_dir=tmp_path,
    )

    assert csv_path.exists()
    assert "unit-test" in csv_path.name
    assert isinstance(df, pd.DataFrame)
    assert df.loc[0, "delta"] == 0


def test_score_pairs_uses_scorer_log_dir(tmp_path: Path) -> None:
    scorer_log_dir = tmp_path / "scorer-logs"
    scorer = LLMScorer(client=StaticClient(55.0), log_dir=scorer_log_dir)
    pairs = [ABPair(id="Y", authority_only="A", explained_only="B")]

    df, csv_path = score_pairs(pairs, scorer, run_label="default-dir")

    assert csv_path.exists()
    assert csv_path.parent.parent == scorer_log_dir
    assert df.loc[0, "authority_score"] == df.loc[0, "explained_score"] == 55.0


def test_compute_summary_handles_empty_dataframe() -> None:
    empty_df = pd.DataFrame()
    summary = compute_summary(empty_df)
    assert pd.isna(summary.authority_mean)
    assert pd.isna(summary.rank_biserial)


def _write_pairs(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "1", "authority_only": "A", "explained_only": "B"}),
                json.dumps({"id": "2", "authority_only": "C", "explained_only": "D"}),
            ]
        ),
        encoding="utf-8",
    )


def test_run_with_injected_scorer(tmp_path: Path) -> None:
    ab_path = tmp_path / "pairs.jsonl"
    _write_pairs(ab_path)
    report_path = tmp_path / "report.md"
    csv_path = tmp_path / "results.csv"
    scorer = LLMScorer(client=StaticClient(40.0), log_dir=tmp_path / "runs")

    with patch("undogmatic.eval_ab._get_scorer", side_effect=AssertionError):
        df, summary, generated_csv = run(
            ab_path,
            report_path=report_path,
            csv_path=csv_path,
            scorer=scorer,
            backend="llm",
            run_label="integration",
        )

    assert report_path.exists()
    assert csv_path.exists()
    assert generated_csv.exists()
    assert len(df) == 2
    assert summary.authority_mean == summary.explained_mean == 40.0
    assert (tmp_path / "runs").is_dir()
    assert scorer.log_dir in generated_csv.parents


def test_run_uses_default_backend_when_no_scorer(tmp_path: Path) -> None:
    ab_path = tmp_path / "pairs.jsonl"
    _write_pairs(ab_path)

    def fake_scorer(text: str, metadata: dict | None = None) -> dict:
        variant = metadata.get("variant") if metadata else None
        score = 60 if variant == "authority_only" else 40
        return {"shame_score": score, "confidence": 99, "rationale": "stub"}

    with patch("undogmatic.eval_ab._get_scorer", return_value=fake_scorer) as mock_get:
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            df, summary, generated_csv = run(ab_path)
        finally:
            os.chdir(cwd)

    assert mock_get.called
    assert not df.empty
    assert summary.authority_mean == 60
    assert summary.explained_mean == 40
    assert (tmp_path / "runs").is_dir()
    resolved_csv = generated_csv if generated_csv.is_absolute() else (tmp_path / generated_csv)
    assert resolved_csv.is_relative_to(tmp_path / "runs")
