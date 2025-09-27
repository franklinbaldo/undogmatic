"""Generate authority-only and explained-only variants from curated theses."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Iterable, Iterator

from undogmatic.utils import cleanse_tese


def load_seeds(path: pathlib.Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def generate_pairs(seeds: Iterable[dict]) -> Iterator[dict]:
    for seed in seeds:
        court = seed["court"]
        tema = seed["tema"]
        tese = seed["tese"]
        authority_only = f"{court} Tema {tema}"
        explained_only = cleanse_tese(tese)
        yield {
            "id": f"{court}-{tema}",
            "court": court,
            "tema": tema,
            "authority_only": authority_only,
            "explained_only": explained_only,
        }


def main() -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in", dest="input_path", type=pathlib.Path, required=True, help="Arquivo JSONL com teses"
    )
    parser.add_argument(
        "--out", dest="output_path", type=pathlib.Path, required=True, help="Arquivo JSONL de saída"
    )
    args = parser.parse_args()

    seeds = list(load_seeds(args.input_path))
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        for pair in generate_pairs(seeds):
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")


if __name__ == "__main__":  # pragma: no cover
    main()
