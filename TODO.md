# TODO — POC UNDOGMATIC (STF/STJ)

**Objetivo.** Demonstrar, com dados reais de teses (STF/STJ), que respostas que **apenas citam o número do tema + tribunal** apresentam **ShameScore** maior do que respostas que **explicam a tese** sem invocar o rótulo de autoridade (sem “STF/STJ” nem “Tema ####”).

---

## 0) Setup do projeto
- [ ] Criar estrutura:

```
undogmatic/
    __init__.py
    scoring.py           # sinais + agregação
    patterns.py          # regex / normalização PT-BR
data/
    raw/{stf,stj}/       # capturas HTML/JSON brutas
    curated/{stf,stj}.jsonl  # schema unificado
scripts/
    fetch_stf.py
    fetch_stj.py
    build_dataset.py
    run_ab_test.py
reports/
    ab_test_summary.md
tests/
    test_patterns.py
    test_scoring.py
pyproject.toml
```

- [ ] `pyproject.toml` com deps mínimas: `requests`, `beautifulsoup4`, `regex`, `pydantic`, `numpy`, `scikit-learn`, `sentence-transformers` (ou mockar encoder nesta POC).
- [ ] `pre-commit` (black/ruff/isort) e CI simples (lint+pytest).

## 1) Ingestão de dados (STF/STJ)
- [ ] **STF**: baixar tabela/listas oficiais de **Teses de Repercussão Geral** (nº do tema, enunciado/tese, URL). Salvar HTML bruto e extrair para `data/curated/stf.jsonl`.  
    Campos: `{"court":"STF","tema":int,"tese":str,"url":str}`
- [ ] **STJ**: baixar **Temas Repetitivos** (nº do tema, enunciado/tese, URL). Salvar bruto e extrair para `data/curated/stj.jsonl`.  
    Campos: `{"court":"STJ","tema":int,"tese":str,"url":str}`
- [ ] `build_dataset.py`: concatenar em `data/curated/temas.jsonl` e filtrar itens com `tese` não vazia.

> **Notas de scraping**
> - Tentar primeiro endpoints/textos oficiais; se vier só HTML, usar BeautifulSoup.
> - Guardar os brutos em `data/raw/...` para reprodutibilidade.
> - Normalizar espaços, remover “Tema …” dentro do **texto** da tese quando fizer a variante “explicada”.

## 2) Geração dos dois estilos de resposta por item
Para cada `{court, tema, tese}` gerar:
- [ ] **Variant A — autoridade-nua**:  
    `f"{court} Tema {tema}"`  
    (Opcional: adicionar “decidiu que…”, **sem** colar o texto da tese; foco é a referência de autoridade.)
- [ ] **Variant B — tese-explicada**:  
    Texto da `tese`, **removendo** menções explícitas de tribunal/tema (regex abaixo), mantendo o conteúdo normativo/descritivo.

Salvar amostras em `data/curated/ab_pairs.jsonl`:
```json
{"id":"STF-1234","court":"STF","tema":1234,"authority_only":"STF Tema 1234","explained_only":"[tese sem tribunal/tema]"}
```

## 3) Definição do ShameScore (índice de hubris)

Implementar em undogmatic/scoring.py:

- [ ] Sinal 1 — BareCitationScore (0–1): match de padrões de citação “nua”.
  - Regex PT para tema: r"\b(Tema\s*(n[ºo]\s*)?|\bT\.)\s*\d{1,4}\b"
  - Tribunal explícito: r"\b(STF|STJ)\b"
  - Heurística: pontuar alto se há tema ± tribunal e ausência de verbos explicativos (“decide/define/estabelece/entende/segundo…” seguido de razão).

- [ ] Sinal 2 — ExplanationBonus (0–1): presença de conectivos que introduzem fundamentação/escopo (“pois”, “porque”, “de modo que”, “aplica-se a”, “não se aplica quando…”). Pode ser proxy simples: contagem normalizada de conectivos/regra.

- [ ] Sinal 3 — Booster/Hedge (opcional nesta POC, pesos baixos):
  - Boosters (↑Shame): “obviamente”, “indiscutivelmente”, “encerra a questão”, “inequivocamente…”
  - Hedges (↓Shame): “em regra”, “salvo”, “ressalvado”, “em princípio”, “via de regra…”

- [ ] Agregação (linear): ShameScore = w_bare * BareCitationScore + w_boost * BoosterScore - w_hedge * HedgeScore - w_expl  * ExplanationBonus
  - Começar com w_bare=0.7, w_expl=0.3, w_boost=0.1, w_hedge=0.1. Ajustar depois.


## 4) Experimento A/B (pareado por tema)

- [ ] run_ab_test.py:

  - Carregar ab_pairs.jsonl.
  - Calcular ShameScore para authority_only e explained_only por tema.
  - Estatísticas:
    - Média, mediana, IQR por grupo.
    - Diferença pareada (Δ = A − B) por tema.
    - Teste pareado (Wilcoxon). Reportar tamanho de efeito (r de rank-biserial).


  - Saídas:
    - reports/ab_test_summary.md com tabela e interpretação.
    - CSV com resultados por tema para auditoria.



**Critério de sucesso da POC**

- [ ] Distribuição de ShameScore(authority_only) dominando ShameScore(explained_only) (média/mediana maiores, p-valor baixo, efeito moderado+).


## 5) Regex/normalização (patterns.py)

- [ ] Normalizar caixa, trocar “nº/n°/No/No.” para nº.

- [ ] Remover padrões de tribunal no variant B:

  - Tribunal: \bSTF\b|\bSTJ\b|\bSupremo Tribunal Federal\b|\bSuperior Tribunal de Justiça\b

  - Tema: \bTema\s*(n[ºo]\s*)?\d{1,4}\b


- [ ] Ponto de atenção: não remover conteúdo jurídico substantivo (ex.: “Tema de prova” etc.); limitar a remoção a sequências que batam com tema de precedente.


## 6) Testes (mínimos)

- [ ] test_patterns.py: cobre matches de “Tema 123”, “Tema nº 123”, “Tema 123/STF”, variações com espaços.

- [ ] test_scoring.py: garante monotonicidade:

  - Se adiciono “STF Tema 123”, ↑BareCitationScore.

  - Se adiciono “porque”, “aplica-se quando…”, ↑ExplanationBonus ⇒ ↓ShameScore.



## 7) Execução (como rodar)

- [ ] `python scripts/fetch_stf.py  --out data/raw/stf.html`

- [ ] `python scripts/fetch_stj.py  --out data/raw/stj.html`

- [ ] `python scripts/build_dataset.py --in data/raw --out data/curated`

- [ ] `python scripts/run_ab_test.py --in data/curated/ab_pairs.jsonl --report reports/ab_test_summary.md`


## 8) Relato e limites

- [ ] Documentar no reports/ab_test_summary.md:

  - Premissa: diferença de forma, não de conteúdo jurídico.

  - Riscos: enunciados de tese já são normativos (poucos “hedges”), então o ganho vem sobretudo de remover o gatilho de autoridade e premiar conectivos explicativos.

  - Próximos passos: incluir explicação + citação (terceira condição), anti-gaming (hedge-spam), e calibração com rótulos humanos.


### Observações rápidas sobre fontes e coleta

- **STF — Teses de Repercussão Geral**: página oficial com listagem e teses; dá para extrair nº do tema, enunciado e links. 1  
- **STJ — Temas Repetitivos**: página “Precedentes qualificados”; quando o índice principal estiver instável, use também páginas auxiliares (NUGEP/tribunais) que replicam a numeração/teses para obter o enunciado. 2  
- **Tabelas dos NUGEPs (TJSP)** costumam trazer **tema, status e ementa resumida**, úteis como fallback. 3
