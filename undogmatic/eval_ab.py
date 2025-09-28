"""A/B evaluation pipeline using the LLM-based ShameScore."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import stats


def _get_scorer(backend: str) -> Callable:
    """Dynamically import and return the appropriate scoring function."""
    if backend == "embed":
        from undogmatic.embed_scorer import score_text

        return score_text
    if backend == "llm":
        from .llm_scorer import LLMScorer, ScoreResult

        scorer_instance = LLMScorer()

        def score_func(text: str, metadata: Optional[dict] = None) -> dict:
            result: ScoreResult = scorer_instance.score_text(text, metadata=metadata)
            return result.model_dump()

        return score_func
    raise SystemExit(f"Unknown backend: {backend}")


@dataclass
class ABPair:
    """Container for a single STF/STJ thesis pair."""

    id: str
    authority_only: str
    explained_only: str


@dataclass
class ExperimentSummary:
    authority_mean: float
    explained_mean: float
    authority_median: float
    explained_median: float
    wilcoxon_stat: float
    wilcoxon_pvalue: float
    rank_biserial: float


def load_pairs(path: Path) -> List[ABPair]:
    pairs: List[ABPair] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            pairs.append(
                ABPair(
                    id=data["id"],
                    authority_only=data["authority_only"],
                    explained_only=data["explained_only"],
                )
            )
    return pairs


def score_pairs(
    pairs: Iterable[ABPair],
    scorer_func: Callable,
    *,
    run_label: Optional[str] = None,
    log_dir: Path,
) -> tuple[pd.DataFrame, Path]:
    records = []
    for pair in pairs:
        try:
            authority_res = scorer_func(
                pair.authority_only,
                metadata={"id": pair.id, "variant": "authority_only"},
            )
            explained_res = scorer_func(
                pair.explained_only,
                metadata={"id": pair.id, "variant": "explained_only"},
            )
        except TypeError:
            authority_res = scorer_func(pair.authority_only)
            explained_res = scorer_func(pair.explained_only)

        records.append(
            {
                "id": pair.id,
                "authority_score": authority_res["shame_score"],
                "authority_confidence": authority_res["confidence"],
                "authority_rationale": authority_res["rationale"],
                "explained_score": explained_res["shame_score"],
                "explained_confidence": explained_res["confidence"],
                "explained_rationale": explained_res["rationale"],
                "delta": authority_res["shame_score"] - explained_res["shame_score"],
            }
        )

    df = pd.DataFrame(records)
    timestamp = datetime.now(timezone.utc)
    date_dir = log_dir / timestamp.strftime("%Y%m%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    label = run_label or f"ab-{timestamp.strftime('%H%M%S')}-{uuid4().hex[:6]}"
    csv_path = date_dir / f"{label}-scores.csv"
    df.to_csv(csv_path, index=False)
    return df, csv_path


def compute_summary(df: pd.DataFrame) -> ExperimentSummary:
    if df.empty:
        return ExperimentSummary(*(float("nan"),) * 7)

    authority = df["authority_score"].to_numpy()
    explained = df["explained_score"].to_numpy()
    deltas = df["delta"].to_numpy()

    if np.allclose(deltas, 0):
        wilcoxon_stat = 0.0
        wilcoxon_pvalue = 1.0
    else:
        wilcoxon_result = stats.wilcoxon(
            authority, explained, zero_method="wilcox", correction=True
        )
        wilcoxon_stat = float(wilcoxon_result.statistic)
        wilcoxon_pvalue = float(wilcoxon_result.pvalue)
    positives = np.sum(deltas > 0)
    negatives = np.sum(deltas < 0)
    rank_biserial = (positives - negatives) / deltas.size if deltas.size else float("nan")

    return ExperimentSummary(
        authority_mean=float(np.mean(authority)),
        explained_mean=float(np.mean(explained)),
        authority_median=float(np.median(authority)),
        explained_median=float(np.median(explained)),
        wilcoxon_stat=wilcoxon_stat,
        wilcoxon_pvalue=wilcoxon_pvalue,
        rank_biserial=float(rank_biserial),
    )


def write_report(path: Path, summary: ExperimentSummary, pair_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = (
        "# ShameScore A/B Test\n\n"
        f"Quantidade de pares: {pair_count}\n\n"
        "| Métrica | Autoridade | Explicada |\n"
        "| --- | --- | --- |\n"
        f"| Média | {summary.authority_mean:.3f} | {summary.explained_mean:.3f} |\n"
        f"| Mediana | {summary.authority_median:.3f} | {summary.explained_median:.3f} |\n\n"
        f"Teste de Wilcoxon: estatística={summary.wilcoxon_stat:.3f}, p-valor={summary.wilcoxon_pvalue:.4f}.\n\n"
        f"Tamanho de efeito (rank-biserial): {summary.rank_biserial:.3f}.\n"
    )
    path.write_text(report, encoding="utf-8")


def run(
    input_path: Path,
    backend: str,
    *,
    report_path: Optional[Path] = None,
    csv_path: Optional[Path] = None,
    run_label: Optional[str] = None,
) -> tuple[pd.DataFrame, ExperimentSummary, Path]:
    scorer_func = _get_scorer(backend)
    log_dir = Path("runs")
    log_dir.mkdir(exist_ok=True)

    pairs = load_pairs(input_path)
    df, generated_csv = score_pairs(pairs, scorer_func, run_label=run_label, log_dir=log_dir)
    summary = compute_summary(df)

    if report_path:
        write_report(report_path, summary, len(pairs))
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

    return df, summary, generated_csv


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in", dest="input_path", type=Path, required=True, help="Arquivo JSONL com pares A/B"
    )
    parser.add_argument("--report", type=Path, help="Arquivo Markdown para resumo")
    parser.add_argument("--csv", type=Path, help="Caminho opcional para CSV detalhado")
    parser.add_argument("--run-label", type=str, help="Nome opcional para agrupar os logs")
    parser.add_argument(
        "--backend", default="embed", choices=["embed", "llm"], help="Scoring backend to use"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    run(
        args.input_path,
        backend=args.backend,
        report_path=args.report,
        csv_path=args.csv,
        run_label=args.run_label,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
