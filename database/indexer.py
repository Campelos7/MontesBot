import logging
import os
from typing import Dict, Iterable, List, Optional

from chromadb.errors import ChromaError
from langchain_chroma import Chroma
from langchain_text_splitters import TokenTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


LOGGER = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "montesbot_documents"

_EMBEDDINGS: Optional[HuggingFaceEmbeddings] = None
_VECTOR_STORE: Optional[Chroma] = None


def _get_chroma_db_path() -> str:
    """Resolve the ChromaDB persistence path from environment variables."""
    return os.getenv("CHROMA_DB_PATH", "./chroma_db")


def _build_embeddings():
    """
    Build the embedding model used for all document indexing and search.
    Uses HuggingFace embeddings so we don't depend on OpenAI or Gemini.
    """
    global _EMBEDDINGS
    if _EMBEDDINGS is not None:
        return _EMBEDDINGS

    model_name = os.getenv(
        "EMBEDDINGS_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    _EMBEDDINGS = HuggingFaceEmbeddings(model_name=model_name)
    return _EMBEDDINGS


def _build_text_splitter() -> TokenTextSplitter:
    """Create a token-based text splitter with the configured chunk size and overlap."""
    return TokenTextSplitter(chunk_size=500, chunk_overlap=50)


def _get_vector_store() -> Chroma:
    """Instantiate or connect to the Chroma vector store."""
    global _VECTOR_STORE
    if _VECTOR_STORE is not None:
        return _VECTOR_STORE

    persist_directory = _get_chroma_db_path()
    os.makedirs(persist_directory, exist_ok=True)

    try:
        embeddings = _build_embeddings()
        _VECTOR_STORE = Chroma(
            collection_name=DEFAULT_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )
        return _VECTOR_STORE
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to initialize Chroma vector store: %s", exc)
        raise


def _to_langchain_documents(scraped_documents: Iterable[object]) -> List[Document]:
    """
    Convert scraped document-like objects to LangChain Document objects.

    Each element is expected to expose at least the attributes:
    - url
    - title
    - content
    - category
    - date_scraped
    """
    lc_docs: List[Document] = []
    for doc in scraped_documents:
        metadata: Dict[str, str] = {
            "source_url": doc.url,
            "title": doc.title,
            "category": doc.category,
            "date_scraped": doc.date_scraped,
        }
        lc_docs.append(Document(page_content=doc.content, metadata=metadata))
    return lc_docs


def index_documents(scraped_documents: Iterable[object]) -> int:
    """
    Index a collection of scraped documents in ChromaDB with upsert behavior.

    Documents are first converted to LangChain Document objects, then split into
    500-token chunks with 50-token overlap. Existing chunks for the same source_url
    are removed before inserting new ones, implementing an upsert strategy.

    Returns the number of chunks stored.
    """
    try:
        vector_store = _get_vector_store()
        splitter = _build_text_splitter()
        base_documents = _to_langchain_documents(scraped_documents)

        if not base_documents:
            LOGGER.info("No documents received for indexing")
            return 0

        # Split into token-based chunks.
        chunked_documents = splitter.split_documents(base_documents)

        # Group documents by source_url for clean up / upsert.
        by_source: Dict[str, List[Document]] = {}
        for doc in chunked_documents:
            source_url = doc.metadata.get("source_url", "unknown")
            by_source.setdefault(source_url, []).append(doc)

        total_chunks = 0
        for source_url, docs_for_source in by_source.items():
            try:
                # Delete old chunks for the same page to implement upsert.
                vector_store.delete(where={"source_url": source_url})
            except ChromaError as exc:
                LOGGER.warning(
                    "Failed to delete existing chunks for %s: %s",
                    source_url,
                    exc,
                )

            try:
                vector_store.add_documents(docs_for_source)
                total_chunks += len(docs_for_source)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Failed to add documents for %s: %s", source_url, exc)

        try:
            vector_store.persist()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to persist ChromaDB: %s", exc)

        LOGGER.info("Indexed %s chunks into ChromaDB", total_chunks)
        return total_chunks
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Indexing operation failed: %s", exc)
        return 0


def get_vector_store() -> Chroma:
    """
    Public helper to obtain the shared Chroma vector store instance.

    This is used by both the indexer and the RAG bot to ensure that the same
    embedding configuration is always applied.
    """
    return _get_vector_store()


def search(query: str, n_results: int = 5) -> List[Document]:
    """
    Perform a semantic similarity search in the vector database.

    Returns a list of LangChain Document objects ordered by relevance.
    """
    try:
        vector_store = _get_vector_store()
        results = vector_store.similarity_search(query, k=n_results)
        LOGGER.info("Search for query '%s' returned %s results", query, len(results))
        return results
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Search operation failed for query '%s': %s", query, exc)
        return []


def get_document_count() -> int:
    """
    Return the number of stored vector entries in the Chroma collection.

    Implementation uses the underlying Chroma collection count; if counting
    fails, zero is returned and the error is logged.
    """
    try:
        vector_store = _get_vector_store()
        # Access underlying Chroma collection when available.
        collection = getattr(vector_store, "_collection", None)
        if collection is None:
            LOGGER.warning("Chroma vector store has no underlying _collection attribute")
            return 0
        return int(collection.count())
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to count documents in ChromaDB: %s", exc)
        return 0

