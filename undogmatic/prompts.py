"""Prompt templates for the LLM-based ShameScore evaluator."""

from __future__ import annotations

from textwrap import dedent

SYSTEM_PROMPT = dedent(
    """
    Você é um avaliador jurídico rigoroso. Sua tarefa é medir o quanto um trecho
    incorre em "hubris": quando ele apela à autoridade (cita tribunal/tema/súmula)
    como substituto de análise ou justificativa.

    - "shame_score" (0 a 100) significa: quanto mais alto, mais o texto depende
      de autoridade sem explicar a razão, o escopo, os limites ou as condições.
    - Ignore "hedges" superficiais ("talvez", "em regra") se não houver
      substância analítica.
    - Considere "explicação" somente quando houver razões, escopo, exceções,
      condições ou comparação com casos concretos.
    - Responda ESTRITAMENTE em JSON com as chaves: shame_score, confidence,
      rationale (1–2 frases).
    """
).strip()


def render_user_prompt(text: str) -> str:
    """Render the user-facing prompt inserting the text to be evaluated."""

    return dedent(
        f"""
        Avalie o texto abaixo. Produza apenas JSON.

        Escala ancorada:
        - 0 = análise sólida, sem apelo de autoridade.
        - 25 = leve menção à autoridade, mas com explicação prevalente.
        - 50 = equilíbrio: alguma justificativa, alguma autoridade
          substituindo análise.
        - 75 = dependência forte de autoridade com explicação fraca.
        - 100 = quase só autoridade (ex.: "STF Tema 1234 resolve o caso"), sem
          razões.

        Texto:
        "{text}"
        """
    ).strip()
