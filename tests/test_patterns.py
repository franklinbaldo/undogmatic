from undogmatic import patterns


def test_normalize_text_replaces_degree_symbol():
    assert patterns.normalize_text("Tema n.o 123") == "Tema nº 123"


def test_remove_authority_markers():
    text = "STF Tema 123 trata de pensão"
    assert patterns.remove_authority_markers(text) == "trata de pensão"


def test_summarize_patterns_counts_variations():
    summary = patterns.summarize_patterns("Tema nº 321 do STF")
    assert summary.bare_citations == 1
    assert summary.tribunals == 1


def test_contains_explanatory_connector():
    assert patterns.contains_explanatory_connector("O benefício se aplica porque há previsão legal")
    assert not patterns.contains_explanatory_connector("STJ Tema 999")
