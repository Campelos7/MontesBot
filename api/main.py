import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List

from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    FastAPI,
    Header,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from bot.rag import get_answer
from database.indexer import get_document_count
from scraper.scraper import run_scraper_and_index, start_scheduler


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""

    session_id: str = Field(..., description="Unique session identifier for the user.")
    message: str = Field(..., description="User message in natural language.")


class ChatResponse(BaseModel):
    """Response body for the /chat endpoint."""

    response: str
    sources: List[Dict[str, str]]


class HealthResponse(BaseModel):
    """Simple health check response schema."""

    status: str
    documents_indexed: int


class StatsResponse(BaseModel):
    """Statistics about the indexed knowledge base."""

    documents_indexed: int


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
    # Load environment variables from .env at application startup so that
    # API keys and configuration are available to all modules.
    load_dotenv()

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
        _check_rate_limit(payload.session_id)

        if not payload.message.strip():
            raise HTTPException(
                status_code=400,
                detail="A mensagem não pode estar vazia.",
            )

        try:
            answer, sources = get_answer(payload.session_id, payload.message)
            return ChatResponse(response=answer, sources=sources)
        except HTTPException:
            raise
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

