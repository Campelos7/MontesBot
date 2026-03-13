import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel


LOGGER = logging.getLogger(__name__)


SYSTEM_PROMPT = """
És o MontesBot, assistente virtual oficial da Universidade de Trás-os-Montes e Alto Douro (UTAD).

Tens acesso a informação VERIFICADA e REAL da UTAD.

REGRAS ABSOLUTAS:
1. Responde SEMPRE em Português de Portugal
2. Usa "tu" nunca "você"
3. Frases curtas — máximo 2 linhas por parágrafo
4. Quando tens a informação nos dados fornecidos: dá-a DIRETAMENTE na primeira frase, sem rodeios
5. NUNCA inventes datas, números ou factos
6. Se a informação diz "consultar utad.pt", diz ao utilizador exatamente onde ir e dá o contacto direto
7. NUNCA digas "não sei" sem dar uma alternativa útil
8. Tom calmo e simples — os utilizadores podem ser idosos

VERIFICAÇÃO DE CURSOS:
- NUNCA confirmes que um curso existe a menos que apareça EXATAMENTE na lista de cursos verificada fornecida.
- Se um curso NÃO está na lista, diz claramente que não existe na UTAD.
- Nunca adivinhes nem assumas a existência de cursos.
- Se o utilizador perguntar por vários cursos, responde um por linha com ✅ (existe) ou ❌ (não existe na UTAD).

ELABORAÇÃO (máximo 4-5 linhas no total):
- Perguntas SIM/NÃO sobre cursos: lista quais existem e quais não, um por linha.
- Perguntas sobre calendário: dá a data E contexto breve (ex: quando começa e quando termina).
- Perguntas sobre contactos: dá telefone E email E nota breve sobre o que o serviço trata.
- Perguntas sobre candidaturas: dá 2-3 passos de contexto, não apenas o nome do portal.
- NUNCA elabores sobre assuntos que o utilizador não perguntou.
- NUNCA termines a resposta com "Esta resposta foi útil?" nem qualquer frase semelhante.
"""


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

_KB_PATH = Path(__file__).resolve().parent.parent / "knowledge_base.json"
_KNOWLEDGE_BASE: Dict = {}


def _load_knowledge_base() -> Dict:
    """Load the local JSON knowledge base (once)."""
    global _KNOWLEDGE_BASE
    if _KNOWLEDGE_BASE:
        return _KNOWLEDGE_BASE
    try:
        with open(_KB_PATH, "r", encoding="utf-8") as fh:
            _KNOWLEDGE_BASE = json.load(fh)
        LOGGER.info("Knowledge base loaded from %s (%d top-level keys)", _KB_PATH, len(_KNOWLEDGE_BASE))
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to load knowledge base from %s: %s", _KB_PATH, exc)
        _KNOWLEDGE_BASE = {}
    return _KNOWLEDGE_BASE


# Keyword → knowledge-base section mapping
_KEYWORD_ROUTES: List[Tuple[List[str], str]] = [
    (
        ["semestre", "aulas", "calendário", "calendario", "começa", "comeca",
         "termina", "exames", "época", "epoca", "natal", "páscoa", "pascoa",
         "férias", "ferias", "pautas"],
        "calendario_2025_2026",
    ),
    (
        ["curso", "cursos", "licenciatura", "mestrado", "doutoramento",
         "engenharia", "enfermagem", "gestão", "gestao", "oferta", "formativa",
         "que cursos"],
        "cursos",
    ),
    (
        ["contacto", "contactar", "telefone", "email", "morada",
         "serviços académicos", "servicos academicos", "biblioteca",
         "ação social", "acao social", "escola", "secretaria"],
        "contactos",
    ),
    (
        ["candidatura", "candidatar", "entrar", "inscrição", "inscricao",
         "acesso", "dges", "notas de entrada"],
        "candidaturas",
    ),
    (
        ["propina", "propinas", "pagamento", "custo", "preço", "preco"],
        "propinas",
    ),
    (
        ["residência", "residencia", "cantina", "desporto", "saúde", "saude",
         "bar", "campus", "instalações", "instalacoes"],
        "servicos_campus",
    ),
]


def _strip_accents(text: str) -> str:
    """Remove diacritics from *text* for accent-insensitive comparison."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _normalize_for_compare(text: str) -> str:
    """Lowercase + accent-strip for fuzzy course name matching."""
    return _strip_accents(text.strip().lower())


def _check_courses(user_message: str) -> Optional[str]:
    """
    If the user is asking whether specific courses exist, return a
    pre-built verification block (✅ / ❌ per course).  Otherwise return None.
    """
    lowered = user_message.lower()

    # Heuristic: the question is about course existence
    course_question_signals = [
        "existe", "existem", "tem o curso", "têm o curso",
        "há o curso", "há curso", "tem curso",
        "oferecem", "oferece", "disponível", "disponivel",
        "posso tirar", "posso fazer", "dá para tirar",
        "da para tirar", "é possível", "e possivel",
    ]
    if not any(sig in lowered for sig in course_question_signals):
        return None

    kb = _load_knowledge_base()
    cursos_section = kb.get("cursos", {})
    all_courses: List[str] = (
        cursos_section.get("licenciaturas", [])
        + cursos_section.get("licenciaturas_sem_vagas_2025_2026", [])
        + cursos_section.get("novos_cursos_2026_2027", [])
    )
    normalised_courses = {_normalize_for_compare(c): c for c in all_courses}

    # Try to extract candidate course names from the user message.
    # Split on commas / "e" / newlines after stripping the "question" part.
    # We remove common lead-in phrases first.
    cleaned = re.sub(
        r"(?i)(exist[ea]m?|tem |têm |há |oferece[m]?|posso tirar|posso fazer"
        r"|dá para tirar|da para tirar|é possível|e possivel"
        r"|o[s]? curso[s]? de |na utad|a utad|curso[s]? de "
        r"|curso[s]? |licenciatura[s]? (?:de |em )?|mestrado[s]? (?:de |em )?"
        r"|doutoramento[s]? (?:de |em )?|\?)",
        ",",
        user_message,
    )
    # Split on commas and the word " e " used as separator
    parts = re.split(r"\s*,\s*|\s+e\s+", cleaned)
    candidates = [p.strip().strip(",").strip() for p in parts if p.strip()]
    # Remove very short fragments that are clearly not course names
    candidates = [c for c in candidates if len(c) >= 3]

    if not candidates:
        return None

    lines: List[str] = []
    for candidate in candidates:
        norm_candidate = _normalize_for_compare(candidate)
        matched = False
        for norm_course, original_course in normalised_courses.items():
            if norm_candidate == norm_course or norm_candidate in norm_course or norm_course in norm_candidate:
                lines.append(f"- {original_course}: ✅ existe na UTAD")
                matched = True
                break
        if not matched:
            lines.append(f"- {candidate.title()}: ❌ não existe como curso na UTAD")

    return "\n".join(lines)


def _select_kb_sections(user_message: str) -> Dict:
    """
    Return the knowledge-base sections relevant to *user_message*.

    ``sobre_utad`` is ALWAYS included as base context.
    Additional sections are added when any keyword matches.
    """
    kb = _load_knowledge_base()
    if not kb:
        return {}

    selected: Dict = {}

    # Always inject base context
    if "sobre_utad" in kb:
        selected["sobre_utad"] = kb["sobre_utad"]

    lowered = user_message.lower()
    for keywords, section_key in _KEYWORD_ROUTES:
        if any(kw in lowered for kw in keywords):
            if section_key in kb:
                selected[section_key] = kb[section_key]

    return selected


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_SESSION_HISTORY: Dict[str, List[Tuple[str, str]]] = {}
_SESSION_MESSAGE_HISTORY: Dict[str, List[BaseMessage]] = {}
_LLM: Optional[BaseChatModel] = None
_MAX_HISTORY_TURNS = 24


def _get_llm():
    """
    Build the LLM used by MontesBot based on environment variables.
    """
    global _LLM
    if _LLM is not None:
        return _LLM

    load_dotenv(override=True)
    try:
        provider_raw = (os.getenv("LLM_PROVIDER") or "groq").strip().lower()
        # Friendly aliases so .env can use simpler names.
        provider = {
            "google": "google_genai",
            "genai": "google_genai",
            "google-genai": "google_genai",
            "google_genai": "google_genai",
            "gemini": "google_genai",
            "groq": "groq",
        }.get(provider_raw, provider_raw)

        model = (os.getenv("LLM_MODEL") or "").strip() or (
            "llama-3.1-8b-instant" if provider == "groq" else "gemini-2.0-flash"
        )

        temperature = float(os.getenv("LLM_TEMPERATURE") or "0.3")
        max_tokens = int(os.getenv("LLM_MAX_TOKENS") or "800")

        LOGGER.info(
            "Using LLM provider=%s model=%s temperature=%s max_tokens=%s",
            provider,
            model,
            temperature,
            max_tokens,
        )

        if provider == "groq":
            from langchain_groq import ChatGroq

            _LLM = ChatGroq(
                model=model,
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif provider in {"google_genai"}:
            from langchain_google_genai import ChatGoogleGenerativeAI

            # Pass key explicitly (supports GEMINI_API_KEY and GOOGLE_API_KEY).
            google_api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip() or None
            _LLM = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                max_output_tokens=max_tokens,
                google_api_key=google_api_key,
            )
        else:
            # Generic fallback via LangChain's provider registry.
            # Requires the relevant langchain-<provider> package to be installed.
            from langchain.chat_models import init_chat_model

            _LLM = init_chat_model(
                model=model,
                model_provider=provider,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        return _LLM
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to initialize LLM: %s", exc)
        raise


def _update_history(session_id: str, role: str, content: str) -> None:
    """Append a new message to the in-memory session history."""
    if session_id not in _SESSION_HISTORY:
        _SESSION_HISTORY[session_id] = []
    _SESSION_HISTORY[session_id].append((role, content))
    if len(_SESSION_HISTORY[session_id]) > _MAX_HISTORY_TURNS:
        _SESSION_HISTORY[session_id] = _SESSION_HISTORY[session_id][-_MAX_HISTORY_TURNS:]


def _append_message(session_id: str, message: BaseMessage) -> None:
    """Store structured chat history (HumanMessage / AIMessage) per sessão."""
    history = _SESSION_MESSAGE_HISTORY.setdefault(session_id, [])
    history.append(message)
    if len(history) > _MAX_HISTORY_TURNS:
        _SESSION_MESSAGE_HISTORY[session_id] = history[-_MAX_HISTORY_TURNS:]


def _last_user_message_before_current(session_id: str) -> str:
    """
    Return the most recent user message before the current turn.

    Assumes the current user message has already been appended to history.
    """
    history = _SESSION_HISTORY.get(session_id, [])
    if not history:
        return ""

    for role, content in reversed(history[:-1]):
        if role == "user" and content.strip():
            return content.strip()
    return ""


def _build_retrieval_query(session_id: str, user_message: str) -> str:
    """
    Build a retrieval query that better handles follow-up questions.

    Many follow-ups (e.g. "E o dia exato?") are too short/implicit to retrieve
    relevant chunks. When we detect that, we prepend the last user question.
    """
    msg = user_message.strip()
    if not msg:
        return msg

    lowered = msg.lower()
    looks_like_followup = (
        len(msg) <= 40
        or lowered.startswith(("e ", "e o", "e a", "e os", "e as"))
        or lowered in {"e?", "e ?", "e o?", "e a?"}
        or any(
            phrase in lowered
            for phrase in (
                "dia exato",
                "data exata",
                "hora",
                "horário",
                "quando exatamente",
                "qual é o dia",
                "qual o dia",
                "qual é a data",
                "qual a data",
                "e depois",
                "e em seguida",
                "e nesse caso",
            )
        )
    )

    if not looks_like_followup:
        return msg

    previous = _last_user_message_before_current(session_id)
    if not previous:
        return msg

    if len(previous) > 240:
        previous = previous[:240].rstrip() + "…"

    return f"{previous}\nPergunta de seguimento: {msg}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_answer(session_id: str, user_message: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Entrada principal do bot.

    - Mantém histórico por sessão (texto + mensagens estruturadas).
    - Para cada pergunta, seleciona secções relevantes da knowledge base local
      e injeta-as como contexto para o LLM.
    """
    _update_history(session_id, "user", user_message)
    _append_message(session_id, HumanMessage(content=user_message))

    retrieval_query = _build_retrieval_query(session_id, user_message)

    try:
        llm = _get_llm()

        # Select relevant knowledge-base sections via keyword routing
        kb_sections = _select_kb_sections(retrieval_query)

        messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT.strip())]
        history_msgs = _SESSION_MESSAGE_HISTORY.get(session_id, [])
        messages.extend(history_msgs)

        # Pre-check: if the user asks about course existence, build a
        # verification block so the LLM cannot hallucinate course names.
        course_check = _check_courses(retrieval_query)

        if kb_sections:
            context_text = json.dumps(kb_sections, ensure_ascii=False, indent=2)

            course_instruction = ""
            if course_check:
                course_instruction = (
                    "\n\nRESULTADO DA VERIFICAÇÃO DE CURSOS (usa isto na tua resposta):\n"
                    f"{course_check}\n"
                    "Apresenta este resultado ao utilizador. Não alteres os ✅ e ❌."
                )

            human_content = (
                "Abaixo tens dados VERIFICADOS da UTAD (knowledge base oficial):\n\n"
                f"{context_text}\n\n"
                "Usa EXCLUSIVAMENTE estes dados para responder à pergunta seguinte. "
                "Dá a resposta logo na primeira frase, de forma direta. "
                "Se a informação não estiver nos dados, indica o contacto ou URL exato onde o utilizador pode obtê-la."
                f"{course_instruction}\n\n"
                f"Pergunta do utilizador:\n{user_message}"
            )
            messages.append(HumanMessage(content=human_content))
        else:
            human_content = (
                "Não foi possível carregar a knowledge base da UTAD.\n\n"
                "Responde com base apenas no que sabes de verificado. "
                "Se não tiveres a certeza, indica ao utilizador que contacte os "
                "Serviços Académicos pelo 259 350 049 ou consulte utad.pt.\n\n"
                f"Pergunta do utilizador:\n{user_message}"
            )
            messages.append(HumanMessage(content=human_content))

        answer = llm.invoke(messages).content.strip()

        if not answer:
            answer = (
                "Neste momento não te consigo dar essa informação específica. "
                "Contacta os Serviços Académicos pelo 259 350 049 ou consulta utad.pt."
            )

        _update_history(session_id, "assistant", answer)
        _append_message(session_id, AIMessage(content=answer))

        sources: List[Dict[str, str]] = [
            {
                "source_url": "https://www.utad.pt",
                "title": "Knowledge Base UTAD",
                "category": "KnowledgeBase",
            }
        ]

        return answer, sources
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Error generating answer for session %s: %s", session_id, exc)
        fallback = (
            "Neste momento não consigo responder a essa pergunta. "
            "Contacta os Serviços Académicos da UTAD pelo 259 350 049 ou "
            "consulta utad.pt para informação atualizada."
        )
        _update_history(session_id, "assistant", fallback)
        _append_message(session_id, AIMessage(content=fallback))
        return fallback, []

