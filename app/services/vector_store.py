"""
app/services/vector_store.py
──────────────────────────────────────────────────────────────
Manages the FAISS vector store lifecycle:
  1. Build the index once from documents.
  2. Persist it to disk (avoids re-embedding on every restart).
  3. Load it from disk on subsequent runs.
  4. Expose a retriever for the RAG chain.

Key design decisions:
  - HuggingFace all-MiniLM-L6-v2 embeddings (fast, offline-capable).
  - FAISS-CPU for local, production-ready similarity search.
  - Single function `get_or_create_vector_store()` encapsulates
    the build-vs-load decision.
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings
from app.core.logger import logger


# ── Embedding model (singleton) ───────────────────────────────

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a cached HuggingFaceEmbeddings instance.
    The model is downloaded once and reused across all calls.
    """
    global _embeddings
    if _embeddings is None:
        logger.info(f"Loading embedding model: {settings.embedding_model_name}")
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,   # Cosine similarity
                "batch_size": 32,
            },
        )
        logger.info("Embedding model loaded successfully")
    return _embeddings


# ── FAISS Index ────────────────────────────────────────────────

def create_vector_store(documents: list[Document]) -> FAISS:
    """
    Build a FAISS vector store from a list of Documents.
    Embeds all documents and persists the index to disk.

    Args:
        documents: List of LangChain Documents to embed.

    Returns:
        A FAISS vector store instance.
    """
    if not documents:
        raise ValueError("Cannot create a vector store from an empty document list.")

    logger.info(f"Building FAISS index from {len(documents)} documents…")
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(documents, embeddings)

    # Persist to disk for reuse
    index_path = settings.faiss_index_path_obj
    index_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_path))
    logger.info(f"FAISS index saved to '{index_path}'")

    return vector_store


def load_vector_store() -> FAISS:
    """
    Load a previously persisted FAISS index from disk.

    Returns:
        A FAISS vector store instance.

    Raises:
        FileNotFoundError: If the index directory doesn't exist.
    """
    index_path = settings.faiss_index_path_obj
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{index_path}'. "
            "Run `scripts/build_index.py` first to create the index."
        )
    logger.info(f"Loading FAISS index from '{index_path}'")
    embeddings = get_embeddings()
    vector_store = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,  # Safe: we created this file ourselves
    )
    logger.info("FAISS index loaded successfully")
    return vector_store


def get_or_create_vector_store(
    documents: list[Document] | None = None,
) -> FAISS:
    """
    Smart loader: returns an existing FAISS index if found on disk,
    otherwise builds and saves a new one from `documents`.

    This is the primary entry point used by the application.
    Call it at startup; the result is cached in the RAG service.

    Args:
        documents: Required only if the index doesn't exist yet.

    Returns:
        FAISS vector store ready for similarity search.

    Raises:
        ValueError: If no index exists and no documents are provided.
    """
    index_path = settings.faiss_index_path_obj
    if index_path.exists():
        logger.info("Existing FAISS index found — loading from disk (no re-embedding)")
        return load_vector_store()

    if documents is None:
        raise ValueError(
            "No FAISS index found on disk and no documents provided to build one. "
            "Provide `documents` argument or run `scripts/build_index.py`."
        )
    logger.info("No existing index — building new FAISS index from documents")
    return create_vector_store(documents)


def create_retriever(
    vector_store: FAISS,
    top_k: int | None = None,
) -> VectorStoreRetriever:
    """
    Create a LangChain retriever from the vector store.

    Args:
        vector_store: A loaded or created FAISS instance.
        top_k: Number of documents to retrieve per query.
                Defaults to settings.top_k_results (3).

    Returns:
        VectorStoreRetriever configured for similarity search.
    """
    k = top_k or settings.top_k_results
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    logger.debug(f"Retriever created with top_k={k}")
    return retriever


def delete_vector_store() -> None:
    """
    Delete the FAISS index from disk.
    Useful when the dataset is updated and the index must be rebuilt.
    """
    import shutil
    index_path = settings.faiss_index_path_obj
    if index_path.exists():
        shutil.rmtree(index_path)
        logger.info(f"FAISS index deleted from '{index_path}'")
    else:
        logger.warning(f"No FAISS index found at '{index_path}' — nothing to delete")
