# AGENTS.md — UNDOGMATIC

> **Papel do agente (Codex CLI):** atuar como dev auxiliar neste repo para implementar e manter um avaliador de “hubris” jurídico (ShameScore) com abordagem **LLM-first**, rodar o A/B (autoridade nua vs tese explicada), manter qualidade (lint, testes) e abrir PRs bem descritos.

---

## 1) Escopo e metas

**Faça:**
- Implementar/ajustar o pipeline **LLM-first**:
  - `undogmatic/prompts.py` (prompt PT-BR; saída **JSON estrito**).
  - `undogmatic/llm_scorer.py` (função `score_text` com temperatura 0.0).
  - `scripts/make_ab_pairs.py` (gera pares A/B a partir de `temas_seed.jsonl`).
  - `scripts/eval_ab.py` (A/B pareado, Wilcoxon, relatório em `reports/ab_test.md`).
- Melhorar DX: cache opcional de scores, CLI simples, README enxuto.
- Abrir PRs pequenos, testados e com diffs legíveis.

**Evite (sem pedido explícito):**
- Scraping pesado/automático de tribunais (por ora use seed manual).
- Alterar `WHITEPAPER.md` de forma substantiva.
- Introduzir novas dependências pesadas sem justificar.

**Não faça:**
- Comitar segredos (`OPENAI_API_KEY`) ou dados sensíveis.
- Subir artefatos grandes/temporários (ver ignorados abaixo).

---

## 2) Setup do ambiente

**Requisitos:**
- Python 3.11+
- `pip` (ou `uv`) e Git
- Chave de API OpenAI em variável de ambiente

**Comandos de setup (execute nesta ordem):**
```bash
# criar venv (opcional)
python -m venv .venv && source .venv/bin/activate

# instalar deps
pip install -U pip wheel
pip install -r requirements.txt || pip install pandas scipy scikit-learn pydantic tqdm python-dotenv openai

# preparar diretórios esperados
mkdir -p data/curated reports runs

Variáveis de ambiente (crie .env se necessário):

OPENAI_API_KEY=...
UNDOGMATIC_MODEL=gpt-4o-mini
# opcional:
# UNDOGMATIC_PROVIDER=openai
# LLM_BASE_URL=https://api.openai.com/v1
```

---

## 3) Convenções de código

Estilo: Black + Ruff + Isort (quando configurados); nomes descritivos, docstrings curtas.

Idioma: Código e comentários em inglês; prompts/README em PT-BR ok.

Arquitetura mínima:

undogmatic/ → biblioteca (sem I/O pesado na importação).

scripts/ → binários simples (argumentos por CLI).

data/curated/ → pequenos datasets de exemplo (até ~100 linhas).

reports/ → saídas textuais, CSVs pequenos e imagens leves.

runs/ → logs temporários de execuções (não versionar).

---

## 4) Execução do fluxo principal (A/B test)

1. Criar/atualizar semente (data/curated/temas_seed.jsonl):

```
{"court":"STF","tema":123,"tese":"[enunciado da tese]"}
{"court":"STJ","tema":456,"tese":"[enunciado da tese]"}
```

2. Gerar pares A/B:

```
python scripts/make_ab_pairs.py --in data/curated/temas_seed.jsonl --out data/curated/ab_pairs.jsonl
```

3. Rodar avaliação A/B:

```
python scripts/eval_ab.py --in data/curated/ab_pairs.jsonl --out-md reports/ab_test.md --out-csv reports/ab_results.csv
```

4. Verificar resultado esperado:

```
Δ = score_authority − score_explained positivo na maioria dos itens.

Relatório reports/ab_test.md com Wilcoxon e interpretação curta.
```

---

## 5) Testes, lint e qualidade

Smoke tests (se existirem):
pytest -q
Obs.: testes não devem chamar a API real (mockar I/O).

Lint/format:
ruff check . • black --check .

Antes de abrir PR: garantir que comandos acima passam.

---

## 6) Política de arquivos ignorados

Adicionar/garantir no .gitignore:

```
# artefatos e dados locais
.runs/
runs/
data/raw/
reports/*.csv
.env
.venv/
```

---

## 7) Commits, branches e PRs

Branching: feat/<curto>, fix/<curto>, docs/<curto>.

Mensagens de commit: imperativo curto (“add scorer cache”, “fix json parse”).

Template do PR (copiar/colar):

```
Motivação: (1–2 frases)

Mudanças: bullets

Como testar: comandos

Riscos/limites: curto

Checklist: lint, tests, report atualizado (se aplicável)
```

---

## 8) Regras de operação do agente (Codex CLI)

Explique o plano antes de modificar muitos arquivos; proponha diffs pequenos.

Peça confirmação para:

Instalar novas dependências.

Remover/refatorar arquivos públicos (README.md, WHITEPAPER.md).

Preferências ao editar:

Escrever código idempotente e determinístico (temperatura 0.0 no LLM).

Tratar erros de parse JSON com 1–2 retentativas.

Não executar scraping; use a semente manual.

Segurança:

Nunca exibir ou gravar chaves de API.

Não enviar dados do repositório para serviços externos (além da API LLM).

Performance de custo:

Se implementar cache de scores, hashear o texto (ex.: SHA1) e armazenar em runs/cache/.

Evitar chamadas duplicadas de LLM no mesmo run.

---

## 9) Tarefas priorizadas (backlog curto)

1. Garantir estrutura mínima (undogmatic/ + scripts/ + data/curated/).

2. Implementar llm_scorer.py com saída JSON estrita e retentativa.

3. Gerar ab_pairs.jsonl a partir de 10–30 teses (seed manual).

4. Rodar eval_ab.py e publicar reports/ab_test.md.

5. Adicionar CLI leve (python -m undogmatic.score --text "...") [opcional].

6. Documentar no README.md “Como rodar a POC”.

---

## 10) Glossário rápido para o agente

Authority-only / autoridade nua: “STF Tema 1234” (sem explicação).

Explained-only / tese explicada: conteúdo da tese sem citar tribunal/tema.

ShameScore (0..100): maior = mais “apelo à autoridade” sem análise.
