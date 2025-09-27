"""Run the ShameScore A/B experiment comparing authority vs explained theses."""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

from undogmatic.scoring import compute_shame_score


@dataclass
class ABRecord:
    id: str
    authority_only: str
    explained_only: str


def load_pairs(path: pathlib.Path) -> List[ABRecord]:
    pairs: List[ABRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            pairs.append(
                ABRecord(
                    id=data["id"],
                    authority_only=data["authority_only"],
                    explained_only=data["explained_only"],
                )
            )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", type=pathlib.Path, required=True, help="Arquivo JSONL com pares A/B")
    parser.add_argument("--report", type=pathlib.Path, required=True, help="Arquivo Markdown para resumo")
    parser.add_argument("--csv", type=pathlib.Path, help="Caminho opcional para CSV detalhado")
    args = parser.parse_args()

    pairs = load_pairs(args.input_path)
    authority_scores = []
    explained_scores = []
    deltas = []

    for pair in pairs:
        authority = compute_shame_score(pair.authority_only)
        explained = compute_shame_score(pair.explained_only)
        authority_scores.append(authority.score)
        explained_scores.append(explained.score)
        deltas.append(authority.score - explained.score)

    authority_array = np.array(authority_scores)
    explained_array = np.array(explained_scores)
    deltas_array = np.array(deltas)

    wilcoxon = stats.wilcoxon(authority_array, explained_array, zero_method="wilcox", correction=True)
    rank_biserial = 1 - (2 * np.sum(deltas_array < 0) / len(deltas_array))

    summary = {
        "authority_mean": float(np.mean(authority_array)) if len(authority_array) else float("nan"),
        "explained_mean": float(np.mean(explained_array)) if len(explained_array) else float("nan"),
        "authority_median": float(np.median(authority_array)) if len(authority_array) else float("nan"),
        "explained_median": float(np.median(explained_array)) if len(explained_array) else float("nan"),
        "wilcoxon_stat": float(wilcoxon.statistic) if deltas_array.size else float("nan"),
        "wilcoxon_pvalue": float(wilcoxon.pvalue) if deltas_array.size else float("nan"),
        "rank_biserial": float(rank_biserial) if deltas_array.size else float("nan"),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# ShameScore A/B Test\n\n"
        f"Quantidade de pares: {len(pairs)}\n\n"
        f"| Métrica | Autoridade | Explicada |\n"
        "| --- | --- | --- |\n"
        f"| Média | {summary['authority_mean']:.3f} | {summary['explained_mean']:.3f} |\n"
        f"| Mediana | {summary['authority_median']:.3f} | {summary['explained_median']:.3f} |\n"
        "\n"
        f"Teste de Wilcoxon: estatística={summary['wilcoxon_stat']:.3f}, p-valor={summary['wilcoxon_pvalue']:.4f}.\n\n"
        f"Tamanho de efeito (rank-biserial): {summary['rank_biserial']:.3f}.\n",
        encoding="utf-8",
    )

    if args.csv:
        df = pd.DataFrame(
            {
                "id": [pair.id for pair in pairs],
                "authority_score": authority_scores,
                "explained_score": explained_scores,
                "delta": deltas,
            }
        )
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
