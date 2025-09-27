# TODO — POC UNDOGMATIC (LLM-first STF/STJ)

**Objetivo.** Demonstrar, com dados reais de teses (STF/STJ), que respostas que **apenas citam o número do tema + tribunal** têm **ShameScore** maior do que respostas que **explicam a tese** com suas próprias palavras, usando um avaliador LLM-first reprodutível.

---

## P0) Setup & fundações
- [x] Estrutura base do pacote (`undogmatic/llm_scorer.py`, `undogmatic/eval_ab.py`, `undogmatic/prompts.py`).
- [x] `pyproject.toml` com dependências essenciais (`requests`, `pydantic`, `numpy`, `pandas`, `scipy`, `python-dotenv`).
- [x] Scripts utilitários (`scripts/make_ab_pairs.py`, `scripts/run_ab_test.py`).
- [x] Testes automatizados (`tests/test_llm_scorer.py`, `tests/test_eval_ab.py`, `tests/test_make_ab_pairs.py`).
- [x] Hooks de estilo/CI mínimos (`.pre-commit-config.yaml`, workflow de lint + pytest).

## P1) Dataset mínimo (em andamento)
- [ ] Montar `data/curated/temas_seed.jsonl` (10–30 teses) a partir das páginas oficiais do STF/STJ.
- [x] Gerar pares A/B com `scripts/make_ab_pairs.py` → `data/curated/ab_pairs.jsonl`.
- [ ] Registrar amostras de controle (frases neutras/hubristas) para checar ancoragem do avaliador.

## P2) LLM-Scorer
- [x] Implementar `LLMScorer.score_text` com prompts ancorados, retries de parsing e logging em `runs/YYYYMMDD/...`.
- [ ] Integrar provedores alternativos (ex.: Azure OpenAI, Anthropic) via adaptadores opcionais.
- [ ] Acrescentar suporte a "batelada" (score de múltiplos textos com limitação de QPS e backoff).

## P3) Experimento A/B pareado
- [x] `undogmatic/eval_ab.py` roda o pipeline completo e gera CSV + Markdown.
- [ ] Acrescentar visualizações (boxplot, histograma) no relatório ou salvar como artefatos externos.
- [ ] Parametrizar `eval_ab.run` para repetir múltiplos passes automaticamente (p/ estabilidade).

## P4) Robustez e confiabilidade
- [ ] Back-to-back runs: medir Pearson/Spearman de duas execuções consecutivas (mesmo modelo).
- [ ] Concordância entre avaliadores (dois provedores ou duas versões do modelo).
- [ ] Controles positivos/negativos explícitos no relatório para sanity check.

## P5) Distilação (opcional)
- [ ] Rotular lote ampliado (200–500 textos) com o LLM "professor".
- [ ] Extrair embeddings e treinar regressor (Ridge/ElasticNet) para prever ShameScore.
- [ ] Avaliar correlação/erro e salvar `models/shame_regressor.pkl` + função `predict_embed`.

## P6) CLI e documentação
- [ ] Empacotar CLI (`undogmatic score`, `undogmatic abtest`) com `typer` ou `argparse`.
- [x] README/WHITEPAPER atualizados com fluxo LLM-first.
- [ ] Documentar política de uso + checklist de privacidade/logs.

## P7) Próximos passos
- [ ] Terceira condição experimental: “explicação + citação” para medir mitigação de hubris.
- [ ] Anti-gaming: prompt tweaks + testes com hedge/booster spam.
- [ ] Generalização para outros órgãos (TST, súmulas) mantendo protocolo.
- [ ] Investigar uso do ShameScore como função de recompensa (RLHF/TRL).
