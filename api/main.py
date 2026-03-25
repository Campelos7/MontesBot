import logging
import os
import time
import asyncio
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List
from uuid import uuid4

from fastapi import (
    BackgroundTasks,
    FastAPI,
    Header,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from bot.message_sanitize import (
    get_chat_max_message_chars,
    sanitize_chat_message,
)
from bot.rag import get_answer
from project_env import load_project_env

load_project_env()

CHAT_MAX_MESSAGE_CHARS = get_chat_max_message_chars()
from database.indexer import get_document_count, get_vector_store
from scraper.scraper import get_last_scrape_date, run_scraper_and_index, start_scheduler


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""

    session_id: str | None = Field(
        default=None,
        description="Unique session identifier for the user. If omitted, the server generates one.",
    )
    message: str = Field(
        ...,
        description="User message in natural language.",
        max_length=CHAT_MAX_MESSAGE_CHARS,
    )

    @field_validator("message", mode="before")
    @classmethod
    def _sanitize_message(cls, value: object) -> object:
        if isinstance(value, str):
            return sanitize_chat_message(value)
        return value


class ChatResponse(BaseModel):
    """Response body for the /chat endpoint."""

    response: str
    sources: List[Dict[str, str]]
    session_id: str


class HealthResponse(BaseModel):
    """Simple health check response schema."""

    status: str
    documents_indexed: int


class StatsResponse(BaseModel):
    """Statistics about the indexed knowledge base."""

    documents_indexed: int


class IndexStatusResponse(BaseModel):
    """Status information about the vector index and scraping."""

    documents_indexed: int
    last_scrape_date: str | None


class ScrapeResponse(BaseModel):
    """Response schema for manual scrape trigger."""

    message: str
    scraped: int
    indexed: int


RATE_LIMIT_REQUESTS_PER_MINUTE = 20
_RATE_LIMIT_WINDOW_SECONDS = 60
_REQUEST_LOG: Dict[str, Deque[float]] = {}


def _check_rate_limit(session_id: str) -> None:
    """
    Enforce a simple in-memory rate limit per session.

    If the limit is exceeded, raise an HTTPException with 429 status.
    """
    now = time.time()
    history = _REQUEST_LOG.setdefault(session_id, deque())

    # Remove entries older than the configured window.
    while history and now - history[0] > _RATE_LIMIT_WINDOW_SECONDS:
        history.popleft()

    if len(history) >= RATE_LIMIT_REQUESTS_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail=(
                "Estão a ser feitos demasiados pedidos para esta sessão. "
                "Por favor espera um pouco antes de continuar."
            ),
        )

    history.append(now)


def get_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    load_project_env()

    app = FastAPI(title="MontesBot API", version="1.0.0")

    # Resolve path to the single-page frontend so that users can open
    # http://localhost:8000 and immediately see the chatbot.
    frontend_path = (
        Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    )

    # Configure CORS for local development / frontend.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def _on_startup() -> None:  # noqa: D401
        """FastAPI startup event handler to initialize background scheduler."""
        try:
            start_scheduler()
            LOGGER.info("APScheduler for scraping started successfully")
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to start APScheduler on startup: %s", exc)

        # Warm up vector store + embeddings in the background so the first chat
        # request is much faster (avoids first-load embedding initialization).
        async def _warm_vector_store() -> None:
            try:
                await asyncio.to_thread(get_vector_store)
                LOGGER.info("Vector store warm-up completed")
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Vector store warm-up failed: %s", exc)

        async def _ensure_index_populated() -> None:
            """
            On cold start, make sure the Chroma index has documents.

            If it is empty, trigger a full scrape + index run once.
            """
            try:
                count = await asyncio.to_thread(get_document_count)
                if count > 0:
                    LOGGER.info("Chroma index already has %s documents; skipping initial scrape", count)
                    return

                LOGGER.info("Chroma index appears empty; starting initial scrape and index run")
                summary = await asyncio.to_thread(run_scraper_and_index)
                LOGGER.info(
                    "Initial scrape and index completed on startup: scraped=%s indexed=%s",
                    summary.get("scraped", 0),
                    summary.get("indexed", 0),
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Failed to ensure initial index population on startup: %s", exc)

        asyncio.create_task(_warm_vector_store())
        asyncio.create_task(_ensure_index_populated())

    @app.get("/", response_class=FileResponse)
    async def root() -> FileResponse:
        """
        Serve the MontesBot single-page frontend at the API root.

        This allows users to open http://localhost:8000 and immediately
        see the chatbot interface without any extra steps.
        """
        if not frontend_path.is_file():
            # If, for alguma razão, o ficheiro não existir, devolvemos
            # uma mensagem simples em vez de deixarmos o servidor falhar.
            raise HTTPException(
                status_code=500,
                detail=(
                    "O ficheiro do frontend (index.html) não foi encontrado. "
                    "Verifica se a pasta 'frontend' existe no mesmo projeto."
                ),
            )
        return FileResponse(path=str(frontend_path))

    @app.post("/chat", response_model=ChatResponse)
    async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
        """
        Chat endpoint that forwards user messages to the RAG bot.

        Applies per-session rate limiting and returns the bot's answer
        along with a list of source documents used.
        """
        session_id = payload.session_id or str(uuid4())
        _check_rate_limit(session_id)

        if not payload.message.strip():
            raise HTTPException(
                status_code=400,
                detail="A mensagem não pode estar vazia.",
            )

        try:
            answer, sources = get_answer(session_id, payload.message)
            return ChatResponse(response=answer, sources=sources, session_id=session_id)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Unexpected error in /chat: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=(
                    "Ocorreu um problema ao processar a tua mensagem. "
                    "Por favor tenta novamente mais tarde."
                ),
            ) from exc

    @app.get("/health", response_model=HealthResponse)
    async def health_endpoint() -> HealthResponse:
        """Lightweight health-check endpoint for monitoring."""
        try:
            doc_count = get_document_count()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Error retrieving document count in /health: %s", exc)
            doc_count = 0
        return HealthResponse(status="ok", documents_indexed=doc_count)

    @app.get("/stats", response_model=StatsResponse)
    async def stats_endpoint() -> StatsResponse:
        """Return basic statistics about the indexed knowledge base."""
        try:
            doc_count = get_document_count()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Error retrieving document count in /stats: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=(
                    "Não foi possível obter estatísticas neste momento. "
                    "Por favor tenta mais tarde."
                ),
            ) from exc

        return StatsResponse(documents_indexed=doc_count)

    @app.get("/index-status", response_model=IndexStatusResponse)
    async def index_status_endpoint() -> IndexStatusResponse:
        """
        Return index statistics and last scrape date for monitoring.
        """
        try:
            doc_count = get_document_count()
            last_scrape = get_last_scrape_date()
            return IndexStatusResponse(
                documents_indexed=doc_count,
                last_scrape_date=last_scrape,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Error retrieving index status: %s", exc)
            raise HTTPException(
                status_code=500,
                detail="Não foi possível obter o estado do índice neste momento.",
            ) from exc

    @app.post("/scrape", response_model=ScrapeResponse)
    async def scrape_endpoint(
        background_tasks: BackgroundTasks,
        admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> ScrapeResponse:
        """
        Manually trigger a scraping run.

        This endpoint is protected via a shared admin token in the
        SCRAPE_ADMIN_TOKEN environment variable. The client must send it
        in the X-Admin-Token header.
        """
        expected_token = os.getenv("SCRAPE_ADMIN_TOKEN")
        if not expected_token:
            raise HTTPException(
                status_code=503,
                detail=(
                    "A funcionalidade de raspagem manual não está configurada "
                    "neste servidor."
                ),
            )

        if not admin_token or admin_token != expected_token:
            raise HTTPException(
                status_code=403,
                detail="Operação não autorizada.",
            )

        def _run() -> None:
            try:
                run_scraper_and_index()
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Background scraping job failed: %s", exc)

        background_tasks.add_task(_run)
        # For immediate feedback we return zeros; real numbers will be in logs.
        return ScrapeResponse(
            message="Raspagem iniciada em segundo plano.",
            scraped=0,
            indexed=0,
        )

    return app


app = get_app()

