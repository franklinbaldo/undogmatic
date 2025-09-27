# Whitepaper: ShameScore via LLM Evaluation for STF/STJ Theses

## Abstract

This document describes the UNDOGMATIC proof of concept for measuring “hubris” in legal answers that
reference Brazilian higher court precedents. Instead of relying on handcrafted regexes and
lexicons, the project now treats a Large Language Model (LLM) as the principal evaluator. For every
text segment we obtain a **ShameScore** (0–100) that reflects how much the answer depends on naked
appeals to authority (e.g., “STF Tema 1234”) without providing analysis. The POC compares two
variants per thesis: an authority-only snippet versus an explanation-only snippet. We expect the
LLM to rate the first variant with higher ShameScore than the second. The artefacts—dataset seed,
A/B pairs, logs, and statistics—are designed to be reproducible and auditable.

**Keywords:** ShameScore, legal AI, hubris detection, LLM evaluation, STF, STJ.

## 1. Motivation

Legal assistants built on LLMs often sound confident even when they only cite a precedent label or a
theme number. For Brazilian practitioners this is especially risky: citing “Tema ####” or a court
acronym without justification can be perceived as rhetorical overreach. Our goal is to quantify that
risk, enabling downstream systems (and humans) to penalize hubris and reward explanations.

## 2. Scope of the POC

The current cycle focuses on a small, manually curated dataset (10–30 theses from STF/STJ). Each
item produces two textual variants:

- **authority_only** – e.g., “STF Tema 1234.”
- **explained_only** – the thesis paraphrased without the court label.

By scoring both variants, we obtain per-item deltas that support statistical tests (Wilcoxon signed
rank) and qualitative inspection. Future iterations may add a third hybrid condition (“explain +
cite”) and larger datasets via scraping.

## 3. Methodology

### 3.1 Prompt design

We deploy the evaluator with a system prompt that defines hubris in Portuguese and enumerates an
anchored scale (0, 25, 50, 75, 100). The user prompt contains the text being judged and explicit
instructions to return JSON with three keys:

```json
{
  "shame_score": 0-100,
  "confidence": 0-100,
  "rationale": "1-2 frases"
}
```

The prompt asks the model to ignore superficial hedges and only reward genuine explanations (scope,
conditions, exceptions). Responses must be valid JSON; the client retries up to two times if
parsing fails.

### 3.2 LLM scorer

`undogmatic/llm_scorer.py` exposes `score_text(text: str) -> ScoreResult`. Responsibilities:

1. Load environment variables (`LLM_API_KEY`, `LLM_PROVIDER`, `LLM_MODEL`).
2. Render system/user prompts from `undogmatic/prompts.py`.
3. Call the provider with deterministic parameters (temperature 0.0, top_p 1.0).
4. Parse JSON into a Pydantic model.
5. Persist logs under `runs/YYYYMMDD/ID/` (prompt, raw response, parsed result).

### 3.3 Evaluation pipeline

`undogmatic/eval_ab.py` orchestrates the experiment:

1. Read `data/curated/ab_pairs.jsonl` (fields: `id`, `authority_only`, `explained_only`).
2. Score both variants and record the outputs in `runs/<date>/scores.csv`.
3. Compute summary statistics (mean, median, IQR) per group.
4. Perform the Wilcoxon signed-rank test on paired differences and report rank-biserial effect size.
5. Render a textual report (`reports/ab_test.md`) and optional plots (boxplot, histogram).

### 3.4 Reliability checks

To ensure the evaluator is stable:

- **Back-to-back runs:** execute the scorer twice on the same inputs and measure Pearson/Spearman
correlations. Low variance is expected when temperature/top_p are fixed.
- **Multi-model agreement:** if two providers are available, compare their scores to detect drift or
systematic bias.
- **Control prompts:** feed clearly humble sentences (“O tema exige análise cuidadosa…”) and clearly
hubristic ones (“Este tema encerra definitivamente a questão.”) to verify the anchors.

## 4. Implementation status

The repository currently contains legacy rule-based modules (`patterns.py`, `scoring.py`) from an
earlier iteration. They will be deprecated once the LLM scorer and evaluation scripts are merged.
The `TODO.md` file outlines the migration plan (P0–P7), including CLI integration and optional
embedding distillation for cheaper batch scoring.

## 5. Limitations

- **Data scarcity:** Manual curation keeps the seed set small. The statistical test may have low
power until more theses are added.
- **LLM subjectivity:** The evaluator inherits biases from the chosen provider. Back-to-back and
multi-provider checks are necessary safeguards.
- **Prompt gaming:** Although the prompt warns against hedge-spam, models might still be fooled by
well-crafted rhetoric. Future work includes adversarial examples and human evaluation.
- **Cost and latency:** API calls for every variant can be expensive. Distilling the signal into a
cheaper embedding-based regressor is an optional follow-up (P5).

## 6. Roadmap

Short-term priorities:

1. Finalizar `llm_scorer.py` e `eval_ab.py` com logging completo.
2. Adicionar testes (`tests/test_llm_scorer.py`) que verificam parsing, ancoragem e retentativa.
3. Incluir condição “explicação + citação” e amostras negativas/positivas controladas.
4. Documentar o protocolo de execução no README e publicar resultados iniciais em `reports/`.

Medium-term extensions:

- Distilar o avaliador em um regressor de embeddings (professor–aluno) para scoring em lote.
- Integrar o ShameScore como função de recompensa em pipelines de RL (reward = −score_norm).
- Expandir para outras cortes (TST, súmulas) mantendo o mesmo protocolo experimental.

## 7. Ethical considerations

The ShameScore is a heuristic for rhetorical caution; it does not certify factual accuracy. Always
cross-check precedents and context manually. Logs may include sensitive text—sanitize them before
sharing. The tool aims to discourage argumentum ad verecundiam, not to automate legal reasoning.

## 8. Contact

Contribuições são bem-vindas via issues ou pull requests no repositório. Para dúvidas, abra uma
issue descrevendo o cenário de uso e os modelos LLM empregados.

