# Experimento A/B — ShameScore (POC)

- Data/Horário: ver diretório `runs/` e timestamps dos artefatos.
- Repositório/versão: workspace local (tests: 14/14 OK).
- Contato: time Undogmatic.

## Desenho do experimento
- Objetivo: avaliar se a variante “autoridade nua” (“STF/STJ Tema N”) recebe ShameScore maior do que a variante “tese explicada” (o enunciado sem apelos explícitos à autoridade).
- Métrica: `ShameScore` (0..100). Maior = mais apelo à autoridade sem análise.
- Comparação: pareamento 1:1 (mesmo tema), teste de Wilcoxon para medidas pareadas e tamanho de efeito (rank-biserial).

## Dados
- Fonte: `data/curated/temas_seed.jsonl` (12 entradas STF/STJ).
- Construção dos pares: `scripts/make_ab_pairs.py` gerou `data/curated/ab_pairs.jsonl` com:
  - `authority_only`: "<TRIBUNAL> Tema <N>".
  - `explained_only`: texto da tese limpo (remoção de tribunal/tema).

## Backend e configuração
- Backend usado: `embed` (local TF‑IDF), sem chamadas externas.
  - Implementação: `undogmatic/embed_scorer.py` (prototipagem por similaridade contra dois protótipos PT‑BR: “hubris” e “humility”).
  - Determinístico (temperatura não se aplica; sem ruído). Reprodutível no mesmo código/seed.
- Alternativa não usada aqui: LLM (`undogmatic/llm_scorer.py`) com saída JSON estrita e cache; pode ser ativada para rodadas futuras.
- Execução:
  - `python scripts/make_ab_pairs.py --in data/curated/temas_seed.jsonl --out data/curated/ab_pairs.jsonl`
  - `python -m undogmatic.eval_ab --in data/curated/ab_pairs.jsonl --backend embed --report reports/ab_test.md --csv reports/ab_results.csv --run-label gemma-embed-local`

## Resultados
- Tabela agregada (de `reports/ab_test.md`):
  - Média: autoridade 56.0 | explicada 48.0
  - Mediana: autoridade 56.0 | explicada 47.0
  - Wilcoxon pareado: estatística=0.000, p‑valor=0.0010
  - Tamanho de efeito (rank‑biserial): 0.917
- Arquivos gerados:
  - Resumo: `reports/ab_test.md`
  - Detalhado (por par): `reports/ab_results.csv`
  - CSV de execução (com label/timestamp): dentro de `runs/YYYYMMDD/`.

## Interpretação
- Direção esperada: “autoridade nua” deve pontuar mais alto. Foi observado (Δ>0 na maioria dos pares; média e mediana maiores para a variante autoridade).
- Significância: p‑valor de 0.001 sugere diferença estatisticamente significativa no conjunto testado (n=12), sob a métrica do protótipo.
- Magnitude: rank‑biserial ≈ 0.92 indica efeito forte.

## “Foi só simulação?”
- Não. Os scores foram computados por um modelo de embeddings local (TF‑IDF) implementado em `undogmatic/embed_scorer.py`, comparando similaridade com protótipos de “hubris” vs “humility”.
- Não houve chamadas à API de LLM/embeddings durante esta rodada (modo local). Os resultados são determinísticos e reprodutíveis.
- Obs.: este backend é um baseline heurístico, não um julgamento semântico completo. Para “LLM‑first” real, ativar `--backend llm` com chave e modelo configurados.

## Limitações e riscos
- Heurístico: o espaço TF‑IDF captura vocabulário superficial; pode supervalorizar menções de autoridade e subcapturar explicações sutis.
- Conjunto pequeno (12 pares); apesar do p‑valor baixo, a generalização requer mais temas.
- Protótipos curtos: a escolha de frases‑âncora influencia o escore; convém ampliar e calibrar.

## Próximos passos
- Rodar com backend LLM (`--backend llm`) usando um modelo configurado para PT‑BR e saída JSON estrita; comparar resultados com o baseline de embeddings.
- Habilitar embeddings via API Gemma (ou similar) em `EMBED_BACKEND=api` e reexecutar para ver ganhos sobre TF‑IDF local.
- Ampliar `temas_seed.jsonl` (30–50 pares), adicionar amostras de controle e checks de consistência.
- Registrar custo/latência por item e ativar cache de respostas (já suportado no scorer LLM).

## Reprodutibilidade
- Testes: `pytest -q` (14/14 aprovados na execução).
- Lint/format (pendente de pequenos ajustes em imports): `ruff check .` e `black --check .`.
- Ambiente: Python ≥3.11, deps em `pyproject.toml`. Não são necessários modelos pesados para o backend local.

