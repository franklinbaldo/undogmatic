"""Fetch STF repercussão geral theses and store raw HTML/JSON."""

from __future__ import annotations

import argparse
import pathlib
from datetime import datetime

import requests

DEFAULT_STF_URL = "https://portal.stf.jus.br/processos/listarTeses.asp"


def fetch_stf(url: str) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_STF_URL, help="Página oficial de teses do STF")
    parser.add_argument("--out", type=pathlib.Path, required=True, help="Arquivo de saída (HTML)")
    args = parser.parse_args()

    content = fetch_stf(args.url)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().isoformat()
    args.out.write_text(f"<!-- fetched_at={timestamp} -->\n" + content, encoding="utf-8")


if __name__ == "__main__":
    main()
