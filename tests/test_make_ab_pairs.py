from __future__ import annotations

from pathlib import Path

from scripts.make_ab_pairs import cleanse_tese, generate_pairs


def test_cleanse_tese_removes_theme_and_tribunal() -> None:
    text = "O STF no Tema 123 fixou tese sobre responsabilidade."
    cleansed = cleanse_tese(text)
    assert "STF" not in cleansed
    assert "Tema" not in cleansed
    assert "responsabilidade" in cleansed


def test_generate_pairs(tmp_path: Path) -> None:
    seeds = [
        {"court": "STF", "tema": 1, "tese": "STF Tema 1 define algo"},
        {"court": "STJ", "tema": 2, "tese": "Superior Tribunal de Justiça entende X"},
    ]
    pairs = list(generate_pairs(seeds))
    assert pairs[0]["authority_only"] == "STF Tema 1"
    assert "Superior Tribunal de Justiça" not in pairs[1]["explained_only"]
    assert len(pairs) == 2
