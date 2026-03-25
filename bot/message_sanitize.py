"""Sanitização e limites do texto das mensagens do utilizador."""

from __future__ import annotations

import os
import re
import unicodedata

# Caracteres de controlo (exceto tab, LF, CR) e DEL — evita NUL e binário acidental no LLM.
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def get_chat_max_message_chars() -> int:
    """Limite máximo de caracteres (configurável por CHAT_MAX_MESSAGE_CHARS)."""
    try:
        n = int(os.environ.get("CHAT_MAX_MESSAGE_CHARS", "4000"))
    except ValueError:
        n = 4000
    return max(256, min(n, 32000))


def get_llm_request_timeout_sec() -> float:
    """Timeout HTTP para pedidos ao provedor LLM (Groq, Gemini, etc.)."""
    try:
        t = float(os.environ.get("LLM_REQUEST_TIMEOUT_SEC", "60"))
    except ValueError:
        t = 60.0
    return max(5.0, min(t, 300.0))


def sanitize_chat_message(text: str) -> str:
    """
    Normaliza Unicode (NFC), remove controlos perigosos e faz strip.

    Não trunca — o limite de tamanho é aplicado pelo validador da API e por get_answer.
    """
    if not isinstance(text, str):
        raise TypeError("message must be a string")
    text = unicodedata.normalize("NFC", text)
    text = _CTRL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()
