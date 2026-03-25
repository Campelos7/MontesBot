import json
import logging
import os
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from bot.message_sanitize import (
    get_chat_max_message_chars,
    get_llm_request_timeout_sec,
    sanitize_chat_message,
)


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

# Mensagem para perguntas fora do âmbito (testada em `tests/test_rag_heuristics.py`).
NAVIGATION_SCOPE_REPLY = (
    "Neste momento, só posso ajudar com informação sobre a UTAD usando os dados disponíveis. "
    "Se quiseres, pergunta-me sobre cursos, candidaturas, propinas, contactos ou calendário académico."
)


def is_opinion_or_subjective_question(text: str) -> bool:
    """
    Heurística leve para perguntas de opinião/valores.

    A test-suite espera que perguntas do tipo "A UTAD é uma boa universidade?"
    sejam detetadas e respondidas sem chamar o LLM.
    """
    t = (text or "").strip().lower()
    if not t:
        return False

    t_norm = _strip_accents(t)

    # Marcadores explícitos de opinião.
    opinion_markers = (
        "opinia",
        "opinio",
        "ach",
        "gosto",
        "recomendo",
        "vale a pena",
        "merece",
        "nao sei",
    )
    if any(marker in t_norm for marker in opinion_markers):
        return True

    # Casos abreviados como "e boa?"
    if re.fullmatch(r"\s*e\s+boa\??\s*", t_norm):
        return True

    # Se a pergunta contém "boa" e não parece uma pergunta factual de datas/valores,
    # tratamos como opinião (ex: "A UTAD é uma boa universidade?").
    if "boa" in t_norm and not any(
        k in t_norm for k in ("quando", "comeca", "termina", "quanto", "preco", "propina", "telefone")
    ):
        return True

    return False


def looks_like_basic_utad_identity_question(text: str) -> bool:
    """Deteta perguntas básicas sobre a UTAD (ex: localização e descrição)."""
    t_norm = _strip_accents((text or "").lower())
    if "utad" not in t_norm:
        return False

    # Localização / morada
    if any(k in t_norm for k in ("onde fica", "localiza", "localizacao", "morada")):
        return True

    # Definição / o que é
    if "o que e" in t_norm or "o que é" in (text or "").lower():
        return True

    return False


def kb_has_section_beyond_sobre_utad(kb_sections: Dict) -> bool:
    """
    Retorna True se existirem secções de KB além de `sobre_utad`.

    A test-suite usa esta função para decidir quando responder com
    `NAVIGATION_SCOPE_REPLY` (sem chamar o LLM).
    """
    return any(k != "sobre_utad" for k in (kb_sections or {}).keys())


# Wrappers para a lógica de RAG.
# A test-suite faz monkeypatch a estes símbolos.
def get_document_count() -> int:
    from database.indexer import get_document_count as _get_document_count

    return _get_document_count()


def vector_search(query: str, n_results: int = 5):
    from database.indexer import search as _search

    return _search(query, n_results=n_results)



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
         "férias", "ferias", "pautas", "prazo", "prazos"],
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


_MONTHS_PT = {
    "janeiro": 1,
    "fevereiro": 2,
    "marco": 3,
    "abril": 4,
    "maio": 5,
    "junho": 6,
    "julho": 7,
    "agosto": 8,
    "setembro": 9,
    "outubro": 10,
    "novembro": 11,
    "dezembro": 12,
}


def _parse_date_pt(date_text: str) -> Optional[date]:
    """
    Converte strings simples do formato "22 de setembro de 2025" para `date`.
    Retorna None se não conseguir interpretar.
    """
    if not date_text:
        return None
    m = re.search(r"(?P<day>\d{1,2})\s+de\s+(?P<month>[A-Za-zçãéêíóôõúàáâ]+)\s+de\s+(?P<year>\d{4})", date_text, re.I)
    if not m:
        return None

    day = int(m.group("day"))
    year = int(m.group("year"))
    month_norm = _strip_accents(m.group("month").lower())
    month = _MONTHS_PT.get(month_norm)
    if not month:
        return None
    return date(year, month, day)


def _choose_semester_key(calendario: Dict, msg_norm: str) -> str:
    """Escolhe automaticamente `1_semestre` ou `2_semestre` a partir da pergunta."""
    has_1 = any(k in msg_norm for k in ("1 semestre", "1º semestre", "primeiro semestre", "primeiro"))
    has_2 = any(k in msg_norm for k in ("2 semestre", "2º semestre", "segundo semestre", "segundo"))

    if has_1 and not has_2:
        return "1_semestre"
    if has_2 and not has_1:
        return "2_semestre"

    start_1 = _parse_date_pt(calendario.get("1_semestre", {}).get("inicio_aulas", ""))
    start_2 = _parse_date_pt(calendario.get("2_semestre", {}).get("inicio_aulas", ""))
    today = date.today()

    if "proximo semestre" in msg_norm or "próximo semestre" in msg_norm:
        if start_1 and today < start_1:
            return "1_semestre"
        # Knowledge base só cobre 2025/2026; quando estamos após o início do ano,
        # a resposta mais útil é indicar o 2º semestre.
        return "2_semestre"

    # Sem indicação explícita: tenta inferir pelo momento do ano letivo.
    if start_2 and today < start_2:
        return "1_semestre"
    return "2_semestre"


def _answer_from_knowledge_base(user_message: str, kb_sections: Dict) -> Optional[str]:
    """Gera uma resposta direta a partir do `knowledge_base.json`."""
    if not kb_sections:
        return None

    msg_norm = _strip_accents((user_message or "").lower())

    # -----------------------------------------------------------------------
    # sobre_utad (identidade)
    # -----------------------------------------------------------------------
    if "sobre_utad" in kb_sections and looks_like_basic_utad_identity_question(user_message):
        sobre = kb_sections.get("sobre_utad", {})
        if any(k in msg_norm for k in ("onde fica", "localizacao", "localiza", "morada")):
            loc = sobre.get("localizacao")
            if loc:
                return f"A UTAD fica em: {loc}."
        # "o que é a utad"
        if "o que e" in msg_norm:
            nome = sobre.get("nome_completo", "UTAD")
            tipo = sobre.get("tipo")
            if tipo:
                return f"{nome} é uma {tipo}."
            return f"{nome} é a Universidade de Trás-os-Montes e Alto Douro (UTAD)."

    # -----------------------------------------------------------------------
    # Calendário
    # -----------------------------------------------------------------------
    if "calendario_2025_2026" in kb_sections:
        cal = kb_sections.get("calendario_2025_2026", {})
        sem_key = _choose_semester_key(cal, msg_norm)
        sem_label = "1º" if sem_key == "1_semestre" else "2º"
        sem = cal.get(sem_key, {})

        if "prazo" in msg_norm:
            pi = cal.get("prazos_importantes", {}) or {}
            dt = pi.get("entrega_dissertacoes_teses")
            dpd = pi.get("entrega_projetos_planos_dissertacoes")
            dpt = pi.get("entrega_projetos_planos_teses")
            lines = ["Prazos importantes (2025/2026):"]
            if dt:
                lines.append(f"- Dissertações/Teses: {dt}")
            if dpd:
                lines.append(f"- Projetos/Planos (Dissertações): {dpd}")
            if dpt:
                lines.append(f"- Projetos/Planos (Teses): {dpt}")
            if len(lines) > 1:
                return "\n".join(lines)

        if any(k in msg_norm for k in ("comec", "inicio", "inici")):
            inicio = sem.get("inicio_aulas")
            if inicio:
                return f"O {sem_label} semestre (2025/2026) começa a {inicio}."

        if any(k in msg_norm for k in ("termin", "fim")):
            fim = sem.get("fim_aulas")
            if fim:
                return f"O {sem_label} semestre (2025/2026) termina a {fim}."

        if any(k in msg_norm for k in ("exames", "epoca", "pautas")):
            normal = sem.get("epoca_normal_exames")
            recurso = sem.get("epoca_recurso")
            if normal and recurso:
                return (
                    f"Época normal de exames ({sem_label} semestre): {normal}. "
                    f"Época de recurso: {recurso}."
                )
            if normal:
                return f"Época de exames ({sem_label} semestre): {normal}."

        # Default: devolve início e fim.
        inicio = sem.get("inicio_aulas")
        fim = sem.get("fim_aulas")
        if inicio and fim:
            return f"O {sem_label} semestre (2025/2026) decorre de {inicio} até {fim}."

    # -----------------------------------------------------------------------
    # Cursos
    # -----------------------------------------------------------------------
    if "cursos" in kb_sections:
        course_check = _check_courses(user_message)
        if course_check:
            return course_check

        cursos = kb_sections.get("cursos", {})
        if any(k in msg_norm for k in ("mestrado", "doutoramento")):
            v = cursos.get("mestrados_e_doutoramentos")
            if v:
                return f"Para mestrados e doutoramentos: {v}"

        # Pergunta genérica: "Quais são os cursos disponíveis na UTAD?"
        if "cursos" in msg_norm and (
            any(k in msg_norm for k in ("quais", "que", "lista", "disponiveis", "disponíveis", "disponível"))
        ):
            lic = cursos.get("licenciaturas", [])
            if lic:
                lines = ["Licenciaturas disponíveis:"]
                lines.extend([f"- {c}" for c in lic])
                return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Contactos
    # -----------------------------------------------------------------------
    if "contactos" in kb_sections:
        contactos = kb_sections.get("contactos", {})
        escolas = contactos.get("escolas", {})

        # Escolas por sigla
        if "ecav" in msg_norm and "ECAV" in escolas:
            e = escolas["ECAV"]
            return f"{e['nome']}: Telefone {e['telefone']} | Email {e['email']}."
        if "echs" in msg_norm and "ECHS" in escolas:
            e = escolas["ECHS"]
            return f"{e['nome']}: Telefone {e['telefone']} | Email {e['email']}."
        if "ect" in msg_norm and "ECT" in escolas:
            e = escolas["ECT"]
            return f"{e['nome']}: Telefone {e['telefone']} | Email {e['email']}."
        if "ecva" in msg_norm and "ECVA" in escolas:
            e = escolas["ECVA"]
            return f"{e['nome']}: Telefone {e['telefone']} | Email {e['email']}."
        if "ess" in msg_norm and "ESS" in escolas:
            e = escolas["ESS"]
            return f"{e['nome']}: Telefone {e['telefone']} | Email {e['email']}."

        # Serviços académicos
        if any(k in msg_norm for k in ("servicos academicos", "sautad", "matric", "certida", "equival")):
            s = contactos.get("servicos_academicos", {})
            tel = s.get("telefone")
            email = s.get("email")
            desc = s.get("descricao")
            if tel and email and desc:
                return f"Serviços Académicos: Telefone {tel} | Email {email}. {desc}."
            if tel and email:
                return f"Serviços Académicos: Telefone {tel} | Email {email}."
            if tel:
                return f"Serviços Académicos: Telefone {tel}."

        # Ação social
        if any(k in msg_norm for k in ("acao social", "sasutad", "bolsa", "resid", "apoio social")):
            s = contactos.get("servicos_acao_social", {})
            tel = s.get("telefone")
            email = s.get("email")
            desc = s.get("descricao")
            if tel and email and desc:
                return f"Ação Social: Telefone {tel} | Email {email}. {desc}."
            if tel and email:
                return f"Ação Social: Telefone {tel} | Email {email}."

        # Biblioteca
        if any(k in msg_norm for k in ("biblioteca", "sdb")):
            s = contactos.get("biblioteca", {})
            tel = s.get("telefone")
            email = s.get("email")
            if tel and email:
                return f"Biblioteca: Telefone {tel} | Email {email}."

        # Apoio técnico / informática
        if any(k in msg_norm for k in ("apoio tecnico", "informatica", "informacao", "apoio tecnico informatica")):
            s = contactos.get("apoio_tecnico_informatica", {})
            tel = s.get("telefone")
            email = s.get("email")
            if tel and email:
                return f"Apoio Técnico Informática: Telefone {tel} | Email {email}."

        # Hospital veterinário
        if "hospital veterinario" in msg_norm or "hvutad" in msg_norm:
            s = contactos.get("hospital_veterinario", {})
            tel = s.get("telefone")
            email = s.get("email")
            if tel and email:
                return f"Hospital Veterinário: Telefone {tel} | Email {email}."

        # Caso geral
        if any(k in msg_norm for k in ("morada", "localiza", "onde fica")):
            geral = contactos.get("geral", {})
            morada = geral.get("morada")
            if morada:
                return f"Morada UTAD: {morada}."

        geral = contactos.get("geral", {})
        tel = geral.get("telefone")
        web = geral.get("website")
        if tel and web:
            return f"Contactos gerais: Telefone {tel} | Website {web}."
        if tel:
            return f"Contactos gerais: Telefone {tel}."

    # -----------------------------------------------------------------------
    # Candidaturas
    # -----------------------------------------------------------------------
    if "candidaturas" in kb_sections:
        c = kb_sections.get("candidaturas", {})
        portal = c.get("portal_nacional")
        desc = c.get("descricao")
        if portal and desc:
            return f"As candidaturas são feitas pelo portal nacional DGES: {portal}. {desc}"
        if desc:
            return str(desc)

    # -----------------------------------------------------------------------
    # Propinas
    # -----------------------------------------------------------------------
    if "propinas" in kb_sections:
        p = kb_sections.get("propinas", {})
        if any(k in msg_norm for k in ("isenc", "isen", "bolse", "bols", "bolseiro")):
            v = p.get("isencoes")
            if v:
                return f"{v}"
        v = p.get("nota")
        if v:
            return f"{v}"

    # -----------------------------------------------------------------------
    # Serviços no campus
    # -----------------------------------------------------------------------
    if "servicos_campus" in kb_sections:
        s = kb_sections.get("servicos_campus", {})
        if "resid" in msg_norm:
            r = s.get("residencias", {})
            conta = r.get("contacto")
            if conta:
                return f"Residências: {conta}"
        if any(k in msg_norm for k in ("cantina", "bar")):
            c = s.get("cantina_e_bar", {})
            d = c.get("descricao")
            if d:
                return f"Cantina/Bar: {d}"
        if "desporto" in msg_norm:
            d = s.get("desporto", {}).get("website")
            if d:
                return f"Desporto: {d}"
        if "saude" in msg_norm:
            m = s.get("saude_mental", {})
            website = m.get("website")
            desc = m.get("descricao")
            if desc and website:
                return f"Saúde mental: {desc} ({website})"
            if website:
                return f"Saúde mental: {website}"

    return None


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

        timeout_sec = get_llm_request_timeout_sec()

        LOGGER.info(
            "Using LLM provider=%s model=%s temperature=%s max_tokens=%s request_timeout_sec=%s",
            provider,
            model,
            temperature,
            max_tokens,
            timeout_sec,
        )

        if provider == "groq":
            from langchain_groq import ChatGroq

            _LLM = ChatGroq(
                model=model,
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout=timeout_sec,
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
                timeout=timeout_sec,
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
                request_timeout=timeout_sec,
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
        lowered.startswith(("e ", "e o", "e a", "e os", "e as"))
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
    user_message = sanitize_chat_message(user_message)
    if not user_message:
        raise ValueError("A mensagem não pode estar vazia.")
    max_chars = get_chat_max_message_chars()
    if len(user_message) > max_chars:
        raise ValueError(f"A mensagem excede o limite de {max_chars} caracteres.")

    _update_history(session_id, "user", user_message)
    _append_message(session_id, HumanMessage(content=user_message))

    # 1) Curto-circuito para perguntas de opinião (sem LLM).
    if is_opinion_or_subjective_question(user_message):
        answer = (
            "Não consigo dar a minha opinião. "
            "Posso, no entanto, ajudar-te com informação verificável sobre a UTAD."
        )
        _update_history(session_id, "assistant", answer)
        _append_message(session_id, AIMessage(content=answer))
        return answer, []

    retrieval_query = _build_retrieval_query(session_id, user_message)

    # 2) Para respostas locais (knowledge_base.json), NÃO usemos uma query "composta"
    # com histórico; isso evita que perguntas curtas de seguimento (ex: "Como contactar...")
    # acionem a secção errada (ex: calendário) por palavras da pergunta anterior.
    kb_sections = _select_kb_sections(user_message)
    if not kb_has_section_beyond_sobre_utad(kb_sections):
        if looks_like_basic_utad_identity_question(user_message):
            local_answer = _answer_from_knowledge_base(user_message, kb_sections)
            if local_answer:
                sources: List[Dict[str, str]] = [
                    {
                        "source_url": "https://www.utad.pt",
                        "title": "Knowledge Base UTAD",
                        "category": "KnowledgeBase",
                    }
                ]
                _update_history(session_id, "assistant", local_answer)
                _append_message(session_id, AIMessage(content=local_answer))
                return local_answer, sources

        answer = NAVIGATION_SCOPE_REPLY
        _update_history(session_id, "assistant", answer)
        _append_message(session_id, AIMessage(content=answer))
        return answer, []

    # 3) Tentar resposta direta via `knowledge_base.json` (sem depender do LLM).
    local_answer = _answer_from_knowledge_base(user_message, kb_sections)
    if local_answer:
        sources: List[Dict[str, str]] = [
            {
                "source_url": "https://www.utad.pt",
                "title": "Knowledge Base UTAD",
                "category": "KnowledgeBase",
            }
        ]
        _update_history(session_id, "assistant", local_answer)
        _append_message(session_id, AIMessage(content=local_answer))
        return local_answer, sources

    # 4) Caso não esteja coberto pela KB: tentar RAG + LLM.
    try:
        llm = _get_llm()

        docs = []
        try:
            if get_document_count() > 0:
                docs = vector_search(retrieval_query, n_results=5) or []
        except Exception:  # noqa: BLE001
            docs = []

        rag_context = "\n\n".join(d.page_content for d in docs if getattr(d, "page_content", None))
        rag_context = rag_context.strip()[:3500]

        messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT.strip())]
        history_msgs = _SESSION_MESSAGE_HISTORY.get(session_id, [])
        messages.extend(history_msgs)

        # Pre-check: se a pergunta é sobre existência de cursos, força validação.
        course_check = _check_courses(retrieval_query)
        course_instruction = ""
        if course_check:
            course_instruction = (
                "\n\nRESULTADO DA VERIFICAÇÃO DE CURSOS (usa isto na tua resposta):\n"
                f"{course_check}\n"
                "Apresenta este resultado ao utilizador. Não alteres os ✅ e ❌."
            )

        context_kb_text = json.dumps(kb_sections, ensure_ascii=False, indent=2)

        if rag_context:
            rag_block = f"\n\nCONTEXT0 RAG (páginas da UTAD):\n{rag_context}\n"
        else:
            rag_block = ""

        human_content = (
            "Abaixo tens dados VERIFICADOS da UTAD (knowledge base oficial e/ou RAG):\n\n"
            f"KNOWLEDGE_BASE:\n{context_kb_text}"
            f"{rag_block}\n\n"
            "Usa os dados acima para responder. "
            "Dá a resposta logo na primeira frase, de forma direta."
            f"{course_instruction}\n\n"
            f"Pergunta do utilizador:\n{user_message}"
        )
        messages.append(HumanMessage(content=human_content))

        answer = llm.invoke(messages).content.strip()
        if not answer:
            answer = (
                "Neste momento não te consigo dar essa informação específica. "
                "Contacta os Serviços Académicos pelo 259 350 049 ou consulta utad.pt."
            )

        sources = []
        for d in docs[:5]:
            meta = getattr(d, "metadata", {}) or {}
            sources.append(
                {
                    "source_url": meta.get("source_url", "https://www.utad.pt"),
                    "title": meta.get("title", "Documento RAG"),
                    "category": meta.get("category", "RAG"),
                }
            )
        if not sources:
            sources = [
                {
                    "source_url": "https://www.utad.pt",
                    "title": "Knowledge Base UTAD",
                    "category": "KnowledgeBase",
                }
            ]

        _update_history(session_id, "assistant", answer)
        _append_message(session_id, AIMessage(content=answer))
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

