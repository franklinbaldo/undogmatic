# MANAGER-INTEL.md

## Visão geral
- **O que este repo faz:** O projeto Undogmatic é uma prova de conceito (POC) focada em medir o quanto respostas jurídicas no Brasil dependem de "autoridade sem explicação". O objetivo principal é avaliar teses (STF/STJ) através de um avaliador LLM-first reprodutível que gera um **ShameScore** justificando a ausência de uma explicação própria.
- **Stack tecnológica principal:** Python 3.10+, dependências de ciência de dados (Numpy, Pandas, SciPy, Scikit-learn), Pydantic, pytest e clientes HTTP (`requests`, `python-dotenv`) para integração com provedores de modelos de linguagem (LLMs).
- **Estado atual:** Ativo e Experimental. O projeto está em transição, tendo abandonado um pipeline legado baseado em regex/lexicons para adotar uma abordagem totalmente "LLM-first".

## Issues e PRs abertas
- **Issues Abertas por Prioridade:**
  1. **[Alta] Issue #10:** Handle scorer objects in A/B evaluation (Associada ao PR #10). Expande a flexibilidade do `eval_ab` para aceitar instâncias de `LLMScorer`.
  2. **[Média] Issue #14:** Revert 11 feat/embed gemma backend (Associada ao PR #14).
- **PRs Abertas e Estado Delas:**
  1. **PR #10 (Aberto):** "Handle scorer objects in A/B evaluation" - Modifica o pipeline de avaliação para suportar callables e objetos `LLMScorer`, normalizando logs e diretórios.
  2. **PR #14 (Aberto):** "Revert 11 feat/embed gemma backend" - Reversão de código em estado aberto.

## Backlog priorizado (para o manager)
1. **Adicionar checagens de estabilidade (Back-to-back runs) e controles**
   - **Descrição:** Executar passes consecutivos para medir correlações (Pearson/Spearman) e adicionar amostras de controle explícitas (positivas/negativas) no sanity check.
   - **Por que é prioridade:** Para validar o **ShameScore**. A credibilidade de métricas baseadas em LLM exige a demonstração de estabilidade e baixa variância quando os parâmetros do modelo (temperature = 0) estão travados.
2. **Incluir condição "explicação + citação" no experimento**
   - **Descrição:** Criar uma terceira variante nos testes A/B que misture explicação real e citação de autoridade.
   - **Por que é prioridade:** Demonstra a mitigação real do *hubris*. Vai enriquecer muito o relatório final (`ab_test.md`) e corroborar o Whitepaper de que é possível ter a autoridade sem a penalidade de "ShameScore".
3. **Integrar provedores alternativos e duplo avaliador**
   - **Descrição:** Adicionar adaptadores para provedores como Azure OpenAI ou Anthropic e medir concordância (agreement) entre dois modelos distintos.
   - **Por que é prioridade:** Garante que o score não é um mero viés da API específica (ex: OpenAI) sendo usada.
4. **Empacotamento em CLI (Command Line Interface)**
   - **Descrição:** Utilizar bibliotecas como `typer` ou `argparse` para unificar comandos como `undogmatic score` e `undogmatic abtest`.
   - **Por que é prioridade:** Diminui a fricção de uso, melhora a ergonomia da ferramenta para terceiros e encerra a transição para pacotes maduros (P6 do TODO).
5. **Distilação de LLM para Regressor (Opcional, porém de alto impacto)**
   - **Descrição:** Utilizar os outputs gerados pelo LLM ("professor") em 200-500 textos para extrair embeddings e treinar um modelo leve (Ridge/ElasticNet) para prever o score localmente.
   - **Por que é prioridade:** Otimização severa de custo e latência. APIs de LLM são caras e lentas. Ter um regressor `shame_regressor.pkl` permitirá o uso de avaliação massiva em produção.

## Contexto para Jules
- **O que precisa saber para trabalhar bem neste repo:**
  Este é um repositório guiado por experimentação e pesquisa de dados (*data-science research*). A lógica central migrou e agora está focada na interação com LLMs através de prompts ancorados localizados em `undogmatic/prompts.py` e inferência em `undogmatic/llm_scorer.py`. Ao realizar testes com LLMs ou invocar APIs, utilize o sistema de **cache** (`runs/cache/`) para não desperdiçar créditos realizando chamadas em textos repetidos. Sempre verifique o `TODO.md` e o `WHITEPAPER.md` pois atuam como norteadores das decisões arquiteturais da POC.
- **Convenções de código e arquivos-chave:**
  - Código fonte e domínio foca na pasta `undogmatic/`.
  - Scripts de transformação/pipeline em `scripts/`. Dados e curadorias em `data/curated/`. Relatórios devem focar em `reports/`.
  - **Formatação:** Segue diretrizes estritas do `black` (100 caracteres), `isort`, e linter `ruff`.
  - **Testes:** Use `pytest` executado a partir da raiz do repositório. O foco dos testes unitários é a garantia do parser JSON da resposta do LLM e integridade das funções (`test_llm_scorer.py`, `test_eval_ab.py`).
  - Lembre-se que há `pre-commit` hooks configurados no projeto.
