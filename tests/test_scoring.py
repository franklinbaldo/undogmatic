from undogmatic.scoring import compute_shame_score


def test_bare_citation_increases_score():
    baseline = compute_shame_score("Trata da aplicação da lei porque há previsão expressa")
    with_citation = compute_shame_score("STF Tema 123. Trata da aplicação da lei porque há previsão expressa")
    assert with_citation.bare_citation_score >= baseline.bare_citation_score
    assert with_citation.score >= baseline.score


def test_explanation_reduces_score():
    citation_only = compute_shame_score("STJ Tema 456")
    explained = compute_shame_score("STJ Tema 456 porque trata de matéria tributária")
    assert explained.explanation_bonus >= citation_only.explanation_bonus
    assert explained.score <= citation_only.score
