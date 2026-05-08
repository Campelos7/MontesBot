# MontesBot (Interface Web com Uvicorn)

MontesBot é um chatbot local com interface web, alimentado por `Ollama` e regras/conteúdo em `Skill.md`.

O fluxo principal do projeto é:
- iniciar API local com Uvicorn;
- abrir a interface no browser;
- conversar via endpoint `/chat`.

## Requisitos

- Python 3.11+
- Ollama instalado
- Modelo local no Ollama (ex.: `llama3.1:8b`)

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuração (`.env`)

Na raiz do projeto:

```env
OLLAMA_MODEL=llama3.1:8b
CHAT_MAX_MESSAGE_CHARS=4000
```

## Execução 

1) Arranca o Ollama:

```bash
ollama serve
```

2) Noutro terminal, arranca a app:

```bash
python -m uvicorn api.main:app --reload --port 8000
```

3) Abre no browser:

- [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Endpoints ativos

- `GET /` -> frontend (`frontend/index.html`)
- `POST /chat` -> resposta do bot
- `GET /health` -> estado simples da app

## Estrutura principal

```text
MontesBot/
├── api/
│   └── main.py
├── bot/
│   ├── rag.py
│   └── message_sanitize.py
├── frontend/
│   └── index.html
├── Skill.md
├── knowledge_base.json
├── project_env.py
├── requirements.txt
└── requirements-dev.txt
```

## Testes

```bash
pip install -r requirements-dev.txt
pytest -q
```
# MontesBot (Ollama + Skill.md)

MontesBot é um chatbot local em Python que responde com base em `Skill.md` e usa um modelo local via `Ollama`.

O foco atual do projeto é simples:
- chat local no terminal;
- respostas em PT-PT;
- contexto e regras definidos em `Skill.md`.

## Requisitos

- Python 3.11+
- Ollama instalado e ativo
- Um modelo local (ex.: `llama3.1:8b`)

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuração

Cria um ficheiro `.env` na raiz (opcional, mas recomendado):

```env
OLLAMA_MODEL=llama3.1:8b
CHAT_MAX_MESSAGE_CHARS=4000
```

## Execução (CLI)

1) Garante que o Ollama está ativo:

```bash
ollama serve
```

2) Noutro terminal, inicia o chat:

```bash
python chat_cli.py
```

Comandos úteis no chat:
- `/exit`
- `/quit`

## Estrutura principal

```text
MontesBot/
├── bot/
│   ├── rag.py               # lógica de resposta e heurísticas
│   └── message_sanitize.py  # sanitização e limites de entrada
├── Skill.md                 # base de conhecimento e regras de resposta
├── knowledge_base.json      # dados estruturados de apoio
├── chat_cli.py              # interface terminal (sem API)
└── project_env.py           # carregamento do .env da raiz
```

## Testes

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Notas

- O projeto mantém histórico em memória por sessão.
- Se a informação não estiver disponível, o bot devolve fallback seguro.
- A qualidade das respostas depende diretamente da qualidade e atualização do `Skill.md`.
