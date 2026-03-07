import logging
import os
from typing import Dict, List, Tuple

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic

from database.indexer import _get_chroma_db_path, DEFAULT_COLLECTION_NAME


LOGGER = logging.getLogger(__name__)


SYSTEM_PROMPT_PT = """
És o MontesBot, um assistente virtual da Universidade de Trás-os-Montes e Alto Douro (UTAD).

Regras obrigatórias:
- Responde sempre em Português europeu.
- Usa frases curtas e claras; evita parágrafos longos e densos.
- Evita jargão académico; quando tiveres de o usar, explica-o em linguagem simples.
- Se a resposta for longa, organiza-a em passos numerados.
- Nunca digas apenas «não sei»; sugere sempre alternativas ou remete para secções relevantes de https://www.utad.pt.
- Tolera erros ortográficos e perguntas pouco claras, tentando perceber a intenção do utilizador.
- Mantém um tom amigável, calmo e paciente.
- No fim de cada resposta, pergunta sempre se a explicação foi útil.

Tens acesso a excertos de páginas da UTAD na secção CONTEXTO. Usa-os como fonte principal.
Se a informação não estiver claramente no contexto, explica isso, mas tenta ainda assim orientar o utilizador
(por exemplo, indicando serviços da UTAD, páginas prováveis ou contactos úteis).
"""


_SESSION_HISTORY: Dict[str, List[Tuple[str, str]]] = {}


def _get_llm():
    """
    Build the configured LLM based on environment variables.

    LLM_PROVIDER can be "claude" or "openai". Default is "claude".
    """
    # Garantimos que o conteúdo atual do .env sobrepõe qualquer variável
    # de ambiente antiga deixada na sessão de desenvolvimento.
    load_dotenv(override=True)

    provider = os.getenv("LLM_PROVIDER", "claude").lower()

    try:
        if provider == "openai":
            model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
            LOGGER.info("Using OpenAI model %s for MontesBot", model_name)
            return ChatOpenAI(model=model_name, temperature=0.2)

        # Default to Anthropic Claude.
        model_name = os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-5-sonnet-20241022")
        LOGGER.info("Using Anthropic model %s for MontesBot", model_name)
        return ChatAnthropic(model=model_name, temperature=0.2)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to initialize LLM: %s", exc)
        raise


def _get_vector_store() -> Chroma:
    """Return a Chroma vector store configured consistently with the indexer."""
    persist_directory = _get_chroma_db_path()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # If no OpenAI key is configured, we skip vector search entirely and
        # rely on the real-time UTAD fallback. This keeps the bot funcional
        # com apenas a chave da Anthropic, ainda que sem RAG completo.
        LOGGER.warning(
            "OPENAI_API_KEY is not set; ChromaDB search will be desativada "
            "e o bot usará apenas o fallback em tempo real de utad.pt."
        )
        # Return a Chroma instance with a dummy embedding function that will
        # never actually be called, as the search function handles this case.
        return Chroma(
            collection_name=DEFAULT_COLLECTION_NAME,
            embedding_function=OpenAIEmbeddings(),  # not used sem chave válida
            persist_directory=persist_directory,
        )

    embeddings = OpenAIEmbeddings()
    return Chroma(
        collection_name=DEFAULT_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )


def _format_history(session_id: str, max_turns: int = 5) -> str:
    """Serialize the recent conversation history for inclusion in the prompt."""
    history = _SESSION_HISTORY.get(session_id, [])
    if not history:
        return "Sem histórico relevante."

    # Only keep the last N turns for brevity.
    recent = history[-max_turns:]
    lines: List[str] = []
    for role, content in recent:
        etiqueta = "Utilizador" if role == "user" else "MontesBot"
        lines.append(f"{etiqueta}: {content}")
    return "\n".join(lines)


def _update_history(session_id: str, role: str, content: str) -> None:
    """Append a new message to the in-memory session history."""
    if session_id not in _SESSION_HISTORY:
        _SESSION_HISTORY[session_id] = []
    _SESSION_HISTORY[session_id].append((role, content))


def _search_vector_store(query: str, n_results: int = 5) -> List[Document]:
    """Direct similarity search wrapper for pre-checking context availability."""
    try:
        # Se não houver chave OpenAI, saltamos esta fase e deixamos o RAG
        # depender apenas do fallback em tempo real.
        if not os.getenv("OPENAI_API_KEY"):
            LOGGER.info(
                "OPENAI_API_KEY não está configurada; a pesquisa em ChromaDB "
                "não será usada para esta pergunta."
            )
            return []

        vector_store = _get_vector_store()
        return vector_store.similarity_search(query, k=n_results)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Vector search failed for query '%s': %s", query, exc)
        return []


def _fetch_realtime_utad(query: str, max_pages: int = 2) -> List[Document]:
    """
    Fallback: perform a small, real-time fetch of UTAD pages related to the query.

    This does not index content in Chroma; it only builds temporary context
    for the current answer when the vector store has no relevant entries.
    """
    try:
        base_search_url = "https://www.utad.pt/?s="
        session = requests.Session()
        headers = {
            "User-Agent": "MontesBotFallback/1.0 (+https://www.utad.pt)"
        }

        search_url = base_search_url + requests.utils.quote(query)
        LOGGER.info("Performing fallback search on UTAD: %s", search_url)
        search_resp = session.get(search_url, headers=headers, timeout=10)
        if search_resp.status_code != 200:
            LOGGER.warning(
                "Fallback UTAD search returned status %s", search_resp.status_code
            )
            return []

        soup = BeautifulSoup(search_resp.text, "html.parser")
        result_links: List[str] = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "utad.pt" in href and href not in result_links:
                result_links.append(href)
            if len(result_links) >= max_pages:
                break

        documents: List[Document] = []
        for url in result_links:
            try:
                resp = session.get(url, headers=headers, timeout=10)
                if resp.status_code != 200:
                    continue
                page_soup = BeautifulSoup(resp.text, "html.parser")
                text = page_soup.get_text(separator="\n")
                cleaned = "\n".join(
                    line.strip() for line in text.splitlines() if line.strip()
                )
                if not cleaned:
                    continue
                title = page_soup.title.string.strip() if page_soup.title else url
                documents.append(
                    Document(
                        page_content=cleaned,
                        metadata={
                            "source_url": url,
                            "title": title,
                            "category": "RealtimeFallback",
                        },
                    )
                )
            except Exception as inner_exc:  # noqa: BLE001
                LOGGER.error("Error fetching fallback page %s: %s", url, inner_exc)

        LOGGER.info("Fallback UTAD fetch produced %s documents", len(documents))
        return documents
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Realtime UTAD fetch failed: %s", exc)
        return []


def get_answer(session_id: str, user_message: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Main entry point for the RAG bot.

    - Maintains per-session conversation history.
    - Uses ChromaDB as primary context source via RetrievalQA.
    - If no context is found, performs a small real-time fetch from utad.pt.

    Returns a tuple of (answer, sources) where sources is a list of metadata
    dictionaries with at least 'source_url' and 'title'.
    """
    _update_history(session_id, "user", user_message)

    history_str = _format_history(session_id)

    try:
        llm = _get_llm()

        # Primeiro tentamos usar o contexto existente no ChromaDB.
        source_docs: List[Document] = _search_vector_store(user_message, n_results=5)

        if not source_docs:
            # Se não houver contexto na base vetorial, usamos o fallback em tempo real.
            temp_docs = _fetch_realtime_utad(user_message)
            source_docs = temp_docs

        if source_docs:
            joined_context = "\n\n---\n\n".join(
                doc.page_content[:4000] for doc in source_docs
            )
            prompt = (
                SYSTEM_PROMPT_PT.strip()
                + "\n\n"
                + "CONTEXTO:\n"
                + joined_context
                + "\n\nHistórico recente da conversa:\n"
                + history_str
                + "\n\nPergunta do utilizador:\n"
                + user_message
            )
            answer = llm.invoke(prompt).content.strip()
        else:
            # Último recurso: sem qualquer contexto documental.
            prompt = (
                SYSTEM_PROMPT_PT.strip()
                + "\n\n"
                + "Não tens contexto adicional de documentos. "
                + "Com base no teu conhecimento geral e bom senso, tenta orientar o utilizador:\n"
                + f"Pergunta: {user_message}"
            )
            answer = llm.invoke(prompt).content.strip()

        if not answer:
            answer = (
                "Não encontrei informação clara nos dados disponíveis. "
                "Recomendo consultar diretamente o site da UTAD ou contactar os serviços relevantes. "
                "Esta resposta foi útil?"
            )
        else:
            # Ensure the answer ends with the required follow-up question.
            if not answer.strip().endswith("?"):
                answer = answer.rstrip() + "\n\nEsta resposta foi útil?"

        _update_history(session_id, "assistant", answer)

        sources: List[Dict[str, str]] = []
        for doc in source_docs:
            metadata = doc.metadata or {}
            sources.append(
                {
                    "source_url": metadata.get("source_url", ""),
                    "title": metadata.get("title", ""),
                    "category": metadata.get("category", ""),
                }
            )

        return answer, sources
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Error generating answer for session %s: %s", session_id, exc)
        message = str(exc)
        fallback = (
            "Ocorreu um problema ao falar com o serviço de IA que gera as respostas. "
            "Mensagem técnica recebida do fornecedor externo: "
            f"\"{message}\". "
            "Por favor pede ao responsável técnico para verificar as chaves e o estado da conta configurada no ficheiro .env. "
            "Entretanto, consulta diretamente o site da UTAD para informação atualizada. "
            "Esta resposta foi útil?"
        )
        _update_history(session_id, "assistant", fallback)
        return fallback, []

