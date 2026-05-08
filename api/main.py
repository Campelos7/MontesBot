import logging
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List
from uuid import uuid4

from fastapi import FastAPI, HTTPException
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
    """Create and configure the FastAPI application instance for local UI."""
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
        return HealthResponse(status="ok")

    return app


app = get_app()

