# UNDOGMATIC

> Undoing naïve deference to overstated mandates by making hubris measurable.

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

## Visão geral

UNDOGMATIC é uma prova de conceito voltada para o ecossistema jurídico brasileiro. O objetivo é
medir, de forma reprodutível, quanto uma resposta se apoia em **autoridade sem explicação** – o
nosso **ShameScore**. A nova direção do projeto abandona o pipeline de regex + lexicons e adota um
avaliador LLM-first capaz de atribuir uma nota (0–100) para cada texto, sempre com justificativa e
confiança. A hipótese que queremos demonstrar é simples: respostas que só citam “Tema ####/STF|STJ”
exibem ShameScore maior do que respostas que explicam a tese com palavras próprias.

Esta é uma POC de pesquisa, não aconselhamento jurídico. Use sempre com revisão humana.

## Abordagem LLM-first

A versão atual do roadmap prioriza um ciclo de avaliação enxuto:

1. **Dataset semente** (`data/curated/temas_seed.jsonl`): 10–30 teses copiadas das páginas oficiais
do STF/STJ.
2. **Geração A/B** (`scripts/make_ab_pairs.py`): cria pares `authority_only` vs. `explained_only`
para cada tese.
3. **ShameScore por LLM** (`undogmatic/llm_scorer.py`): envia o texto para um modelo de linguagem
   com um prompt ancorado em PT-BR e retorna `shame_score`, `confidence` e `rationale`. As respostas
   válidas são armazenadas em cache (`runs/cache/<sha1>.json`) para evitar chamadas repetidas.
4. **Experimento pareado** (`undogmatic/eval_ab.py`): compara a distribuição do score entre os dois
estilos e aplica Wilcoxon + tamanho de efeito.
5. **Relatório** (`reports/ab_test.md`): consolida tabelas, gráficos e interpretação.

> Nota: A versão atual já removeu o pipeline legado de regex/lexicons. O módulo
> `undogmatic.llm_scorer` é a fonte de verdade para o ShameScore.

## Estrutura esperada do repositório

```
undogmatic/
├── undogmatic/
│   ├── __init__.py
│   ├── llm_scorer.py        # contrato score_text + logging de prompts/respostas
│   ├── prompts.py           # templates de system/user prompt (PT-BR)
│   └── eval_ab.py           # estatística pareada, geração de relatórios
├── scripts/
│   └── make_ab_pairs.py     # transforma teses em variantes A/B
├── data/
│   └── curated/
│       ├── temas_seed.jsonl      # amostra manual
│       ├── ab_pairs.jsonl        # saídas do script make_ab_pairs
│       └── control_samples.jsonl # frases neutras/hubristas p/ ancoragem
├── reports/
│   └── ab_test.md           # resumo do experimento
├── tests/
│   └── test_llm_scorer.py   # testes rápidos de parsing JSON e anchors
├── README.md
├── TODO.md
└── WHITEPAPER.md
```

Alguns destes arquivos ainda estão em desenvolvimento – consulte o `TODO.md` para o plano detalhado
de entrega (P0–P7).

## Requisitos e instalação

1. Garanta Python 3.10+.
2. Crie e ative um ambiente virtual.
3. Instale as dependências:

```bash
pip install -e .[dev]
```

Principais bibliotecas: `pydantic`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `tqdm`, `python-dotenv`
e um cliente de API para o provedor de LLM escolhido.

### Configuração de credenciais

Crie um arquivo `.env` na raiz com as variáveis necessárias, por exemplo:

```
LLM_API_KEY="sua-chave"
LLM_PROVIDER="openai"
LLM_MODEL="gpt-4.1-mini"
# Opcional para proxies/self-host: LLM_BASE_URL="https://api.openai.com/v1/chat/completions"
```

Os scripts carregarão essas variáveis automaticamente (`python-dotenv`). Configure temperatura `0.0`
e `top_p=1.0` para reprodutibilidade. Caso utilize um proxy ou Azure OpenAI, ajuste `LLM_BASE_URL`.

## Fluxo de trabalho da POC

1. Revise `data/curated/temas_seed.jsonl` (o repositório já inclui 12 teses STF/STJ como ponto de partida) e ajuste conforme necessário.
   Para sanity check rápido, utilize também `data/curated/control_samples.jsonl`, que traz frases neutras, grounded e altamente hubristas para calibrar o avaliador.
2. Gere pares A/B:
   ```bash
   python scripts/make_ab_pairs.py --in data/curated/temas_seed.jsonl --out data/curated/ab_pairs.jsonl
   ```
3. Rode o avaliador e experimento (saída principal em Markdown, CSV opcional para análises):
   ```bash
   python -m undogmatic.eval_ab --in data/curated/ab_pairs.jsonl --report reports/ab_test.md --csv reports/ab_results.csv
   ```
4. Consulte `reports/ab_test.md` e os artefatos em `runs/<data>/` para análise completa. Reutilize o
   cache automático (`runs/cache/`) sempre que repetir o mesmo texto para economizar créditos.

> `scripts/run_ab_test.py` permanece no repositório apenas por retrocompatibilidade; utilize o comando
> `python -m undogmatic.eval_ab` descrito acima como caminho canônico.

## Testes

Testes unitários focam no parsing do retorno do LLM e na estabilidade dos anchors:

```bash
pytest
```

Para garantir consistência, execute dois passes consecutivos do avaliador e compare as correlações
(Pearson/Spearman) entre as execuções.

## Roadmap imediato

- [ ] Migrar completamente para `llm_scorer.py` + `eval_ab.py`.
- [ ] Adicionar checagem de estabilidade (passes consecutivos) e dupla avaliação (dois modelos).
- [ ] Incluir condição “explicação + citação” como terceira variante.
- [ ] Documentar limites éticos e anti-gaming (hedge-spam) no whitepaper.

## Licença

MIT. Veja `LICENSE`.

## Contato

Abra uma issue ou envie PRs – contribuições são bem-vindas.

