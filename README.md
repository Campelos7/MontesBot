
COMANDOS PARA INSERIR NO TERMINAL:

py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

PARA DAR RUN:
python -m uvicorn api.main:app --reload --port 8000

PARA FECHAR RUNS EM SEGUNDO PLANO:
taskkill /F /IM python.exe



## MontesBot — UTAD RAG Chatbot

MontesBot is a Retrieval-Augmented Generation (RAG) chatbot that collects and indexes public information from the University of Trás-os-Montes e Alto Douro website (`utad.pt`) and exposes it through a simple web-based chat interface.

The assistant is designed for accessibility and clarity, especially for seniors and users with low digital literacy, always answering in European Portuguese.

### Architecture Overview

```text
                +----------------------+
                |   UTAD Website       |
                |      (utad.pt)       |
                +----------+-----------+
                           |
                   HTTP (polite, robots.txt)
                           |
                 +---------v----------+
                 |   Scraper          |
                 | (BeautifulSoup4,   |
                 |  Requests,         |
                 |  APScheduler)      |
                 +---------+----------+
                           |
             Raw docs (JSONL) + Python objects
                           |
                 +---------v----------+
                 |   Indexer          |
                 | (LangChain,        |
                 |  ChromaDB,         |
                 |  Groq embeddings)  |
                 +---------+----------+
                           |
                 Vector embeddings
                           |
                 +---------v----------+
                 |   RAG Bot          |
                 | (LangChain         |
                 |  RetrievalQA +     |
                 |  Claude / GPT-4o)  |
                 +---------+----------+
                           |
                 Answers + sources
                           |
        +------------------v------------------+
        |           FastAPI API               |
        |  /chat  /health  /stats  /scrape   |
        +------------------+------------------+
                           |
                    JSON over HTTP
                           |
                 +---------v----------+
                 |  Frontend (SPA)   |
                 |  HTML/CSS/JS      |
                 |  "MontesBot 🤖"   |
                 +-------------------+
```

### Project Structure

```text
MontesBot/
├── scraper/
│   └── scraper.py          # UTAD web scraper and APScheduler integration
├── database/
│   └── indexer.py          # ChromaDB indexer and search helpers
├── bot/
│   └── rag.py              # RAG chatbot logic and LLM orchestration
├── api/
│   └── main.py             # FastAPI application and HTTP endpoints
├── frontend/
│   └── index.html          # Single-page chat frontend
├── .env                    # Environment variables (example template)
├── requirements.txt        # Python dependencies with pinned versions
└── README.md               # This file
```

---

## Features

- **RAG pipeline** using LangChain + ChromaDB as local vector store.
- **Polite UTAD scraper** with robots.txt support and request delays.
- **APScheduler** job that runs scraping and indexing automatically (daily at 03:00 by default).
- **FastAPI backend** exposing chat, health, stats, and a protected manual scraping endpoint.
- **LLM provider switching** between Claude (Anthropic) and GPT-4o (OpenAI) via environment variable.
- **Per-session conversation history** managed in memory.
- **Fallback real-time fetch** from `utad.pt` when the vector store has no relevant context.
- **Accessible, single-screen frontend** in Portuguese, designed for clarity and ease of use.

---

## Setup Instructions

### 1. Requirements

- Python **3.11+**
- Access to:
  - **Anthropic API key** (Claude) and/or
  - **OpenAI API key** (GPT-4o + embeddings)
- Internet access to `utad.pt` from the machine running the scraper.

### 2. Clone the Repository

```bash
git clone <your-repo-url> MontesBot
cd MontesBot
```

### 3. Create a Virtual Environment

It is recommended to use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or
.venv\Scripts\activate      # Windows
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Copy `.env` and fill in the required values:

```bash
cp .env .env.local  # optional, if you want a separate file
```

Environment variables used:

- **ANTHROPIC_API_KEY**  
  Anthropic API key for Claude models.

- **OPENAI_API_KEY**  
  OpenAI API key for GPT-4o and embeddings.

- **LLM_PROVIDER**  
  - `claude` — use Anthropic Claude (default)  
  - `openai` — use OpenAI GPT-4o

- **SCRAPE_INTERVAL_HOURS**  
  - `24` (default) → schedule once per day at 03:00.
  - Any other integer value → schedule scraper every N hours.

- **CHROMA_DB_PATH**  
  Local folder path used by ChromaDB to persist the vector store.  
  Default: `./chroma_db`

- **API_PORT**  
  Port used when running `uvicorn`. Default: `8000`.

- **SCRAPE_ADMIN_TOKEN**  
  Secret token required to call the protected `POST /scrape` endpoint.  
  Do **not** commit this token to version control.

> **Important:** No API keys are hardcoded in the codebase. All credentials must be provided via environment variables.

---

## Running MontesBot Locally

### 1. Start the API Server

From the project root:

```bash
uvicorn api.main:app --reload --port 8000
```

The `--reload` flag is convenient during development; in production you should run without it and behind a proper process manager or container.

On startup:

- The FastAPI app is initialized.
- The APScheduler background scheduler is started.
- A scraping job is scheduled (daily at 03:00 by default).

### 2. Open the Frontend

Open the `frontend/index.html` file directly in your browser, or serve the `frontend` folder with a simple static server (for example using VS Code Live Server or `python -m http.server` from the `frontend` directory).

The frontend expects the API to be accessible at:

- `http://localhost:8000`

If you use a different port or host, adjust `API_BASE` in `frontend/index.html`.

---

## API Documentation

All responses intended for end users are in **European Portuguese**. Error messages are simple and clear.

### `POST /chat`

Send a message to MontesBot and receive an answer plus the list of sources used.

- **Request body**

```json
{
  "session_id": "string",
  "message": "string"
}
```

- **Response (200)**

```json
{
  "response": "Texto de resposta em Português europeu...",
  "sources": [
    {
      "source_url": "https://www.utad.pt/...",
      "title": "Título da página",
      "category": "Academic | Services & Contacts | Institutional | Events & News | RealtimeFallback"
    }
  ]
}
```

- **Rate limiting**

  - Maximum **20 requests por minuto por sessão**.
  - If exceeded, you receive HTTP `429` with a friendly error message in Portuguese.

### `GET /health`

Simple health check for monitoring and automation.

- **Response (200)**

```json
{
  "status": "ok",
  "documents_indexed": 1234
}
```

### `POST /scrape`

Manually trigger a scraping + indexing run in the background.

- **Protection**
  - Requires the `SCRAPE_ADMIN_TOKEN` environment variable to be set.
  - Client must send header: `X-Admin-Token: <your-token>`.

- **Response (200)**

```json
{
  "message": "Raspagem iniciada em segundo plano.",
  "scraped": 0,
  "indexed": 0
}
```

The real numbers for scraped and indexed documents are logged server-side.

### `GET /stats`

Returns a simple count of currently indexed vector entries.

- **Response (200)**

```json
{
  "documents_indexed": 1234
}
```

---

## Internals

### Scraper (`scraper/scraper.py`)

- Uses **Requests** + **BeautifulSoup4** to crawl `utad.pt`.
- Respects `robots.txt` using Python's standard `RobotFileParser`.
- Restricts crawling to the `utad.pt` domain and subdomains.
- Follows links recursively with a configurable maximum number of pages.
- Sleeps **1–2 seconds between requests** to avoid overloading the site.
- Extracts and cleans text, removing empty lines and normalizing whitespace.
- Categorizes pages heuristically into:
  - Academic
  - Services & Contacts
  - Institutional
  - Events & News
  - Uncategorized (fallback)
- Saves a copy of all scraped documents to `scraper/data/raw_documents.jsonl` with:
  - `url`
  - `title`
  - `content`
  - `date_scraped` (ISO 8601, UTC)
  - `category`
- Exposes `run_scraper_and_index()` (scrape + persist + index) and `start_scheduler()` to wire into FastAPI.

### Indexer (`database/indexer.py`)

- Uses **ChromaDB** via `langchain-chroma` as a local vector store.
- Uses **OpenAI embeddings** via `langchain-openai` (requires `OPENAI_API_KEY`).
- Uses `TokenTextSplitter` (`langchain-text-splitters`) to chunk text:
  - **chunk_size**: 500 tokens
  - **chunk_overlap**: 50 tokens
- Stores the following metadata for each chunk:
  - `source_url`
  - `title`
  - `category`
  - `date_scraped`
- Implements an **upsert strategy** per `source_url`:
  - Delete existing chunks for a URL.
  - Insert freshly created chunks for the same URL.
- Provides:
  - `index_documents(scraped_documents)` → number of chunks stored.
  - `search(query, n_results=5)` → list of `Document` results (used internally).
  - `get_document_count()` → number of stored vector entries.

### RAG Bot (`bot/rag.py`)

- Maintains an in-memory **per-session conversation history**.
- Chooses the LLM based on `LLM_PROVIDER`:
  - `claude` → `ChatAnthropic` (Claude Sonnet model, configurable).
  - `openai` → `ChatOpenAI` (GPT-4o, configurable).
- Builds a **RetrievalQA** chain with:
  - ChromaDB retriever (`k=5`).
  - A custom **Portuguese system prompt** enforcing:
    - European Portuguese.
    - Short, clear sentences.
    - Explanation of academic jargon in simple language.
    - Numbered steps for longer answers.
    - No bare “I do not know”; always provide alternatives or references to `utad.pt`.
    - Friendly, calm, patient tone.
    - A final **“Esta resposta foi útil?”** check in every answer.
- Before calling RetrievalQA:
  - It pre-checks the vector store for results.
  - If no results exist, it performs a **real-time search** on `utad.pt`:
    - Queries `https://www.utad.pt/?s=<query>`.
    - Fetches a small number of result pages.
    - Builds temporary context for a one-off answer.
  - If even the fallback fails, the LLM is called with a clear explanation that no context is available, and the bot guides the user as best as possible.

### API (`api/main.py`)

- **FastAPI** application with CORS enabled for local development.
- Endpoints:
  - `POST /chat` → forwards messages to the RAG bot.
  - `GET /health` → status and indexed document count.
  - `POST /scrape` → protected manual scraping trigger.
  - `GET /stats` → number of indexed vector entries.
- **Rate limiting**:
  - Simple in-memory sliding window: **20 requests/minute per `session_id`**.
  - Requests over the limit receive HTTP 429 with a clear Portuguese message.
- On startup:
  - Calls `start_scheduler()` to boot APScheduler and schedule daily scraping.

### Frontend (`frontend/index.html`)

- Single static HTML file (no build tools required).
- Key characteristics:
  - Large body text (minimum 18px).
  - High contrast UTAD color palette:
    - Blue `#003a70`.
    - Accent `#0066cc`.
  - Clean, single-screen layout with:
    - Bot header (“MontesBot 🤖”).
    - Scrollable chat area with clear bubbles for user and bot.
    - Typing indicator: “MontesBot está a pensar...”.
    - Suggested starter questions on first load.
    - Clear error messages in plain Portuguese.
  - Fully responsive and mobile-friendly.
- Communicates with `POST /chat` at `http://localhost:8000/chat`.

---

## Tech Stack

- **Language:** Python 3.11+
- **Web framework:** FastAPI
- **Scraping:** Requests, BeautifulSoup4
- **Scheduling:** APScheduler
- **Vector DB:** ChromaDB + langchain-chroma
- **RAG / Orchestration:** LangChain
- **LLMs:** Anthropic Claude, OpenAI GPT-4o (selectable)
- **Embeddings:** OpenAI embeddings via langchain-openai
- **Frontend:** HTML/CSS/JavaScript (single file)
- **Config:** python-dotenv

---

## Team

- **Team members**
  - Filipe Santos
  - Francisco Veiga
  - Duarte Carvalho
  - Sérgio Vieira
  - Tomás Campelos

- **University**
  - UTAD — Interação Pessoa-Computador 2025/2026

