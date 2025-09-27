"""Parse raw STF/STJ theses and build a curated JSONL dataset."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Iterable, List, TypedDict

from bs4 import BeautifulSoup

from undogmatic.patterns import normalize_text, remove_authority_markers


class TemaRecord(TypedDict):
    court: str
    tema: int
    tese: str
    url: str


def _extract_text(node) -> str:
    return " ".join(node.stripped_strings)


def parse_stf_html(path: pathlib.Path) -> Iterable[TemaRecord]:
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    for link in soup.select("a"):  # heuristic placeholder
        text = _extract_text(link)
        if text.lower().startswith("tema"):
            try:
                tema = int("".join(ch for ch in text if ch.isdigit()))
            except ValueError:
                continue
            tese_node = link.find_next("p")
            if tese_node is None:
                continue
            tese = normalize_text(_extract_text(tese_node))
            yield {
                "court": "STF",
                "tema": tema,
                "tese": tese,
                "url": link.get("href", ""),
            }


def parse_stj_html(path: pathlib.Path) -> Iterable[TemaRecord]:
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    for header in soup.find_all(["h2", "h3", "h4"]):
        text = _extract_text(header)
        if text.lower().startswith("tema"):
            try:
                tema = int("".join(ch for ch in text if ch.isdigit()))
            except ValueError:
                continue
            tese_node = header.find_next("p")
            if tese_node is None:
                continue
            tese = normalize_text(_extract_text(tese_node))
            link = header.find("a")
            url = link.get("href") if link else ""
            yield {
                "court": "STJ",
                "tema": tema,
                "tese": tese,
                "url": url or "",
            }


def build_ab_pairs(records: Iterable[TemaRecord]) -> List[dict]:
    ab_pairs = []
    for record in records:
        authority_only = f"{record['court']} Tema {record['tema']}"
        explained_only = remove_authority_markers(record["tese"])
        ab_pairs.append(
            {
                "id": f"{record['court']}-{record['tema']}",
                "court": record["court"],
                "tema": record["tema"],
                "authority_only": authority_only.strip(),
                "explained_only": explained_only,
            }
        )
    return ab_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=pathlib.Path, required=True, help="Diretório com HTML bruto")
    parser.add_argument("--out", type=pathlib.Path, required=True, help="Diretório de saída para JSONL")
    args = parser.parse_args()

    stf_path = args.raw / "stf.html"
    stj_path = args.raw / "stj.html"

    records: List[TemaRecord] = []
    if stf_path.exists():
        records.extend(parse_stf_html(stf_path))
    if stj_path.exists():
        records.extend(parse_stj_html(stj_path))

    args.out.mkdir(parents=True, exist_ok=True)

    curated_path = args.out / "temas.jsonl"
    with curated_path.open("w", encoding="utf-8") as handle:
        for record in records:
            if not record["tese"]:
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    ab_pairs = build_ab_pairs(records)
    ab_path = args.out / "ab_pairs.jsonl"
    with ab_path.open("w", encoding="utf-8") as handle:
        for pair in ab_pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
