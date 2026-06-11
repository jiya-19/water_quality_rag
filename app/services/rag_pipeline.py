"""
app/services/rag_pipeline.py
──────────────────────────────────────────────────────────────
The core RAG (Retrieval-Augmented Generation) pipeline.

Architecture:
  User query
      │
      ▼
  [Topic Guard]  ──── off-topic ────► Standard rejection message
      │
      ▼
  [FAISS Retriever]  (top-k similarity search)
      │
      ▼
  [Prompt Template]  (retrieved context + question)
      │
      ▼
  [Groq LLM]  (llama-3.1-8b-instant or configured model)
      │
      ▼
  [Structured Response]

The pipeline is built as a LangChain LCEL (LangChain Expression
Language) chain for clean composition and future streaming support.
"""

from typing import Any

from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from app.core.config import settings
from app.core.logger import logger
from app.services.data_loader import load_all_documents
from app.services.vector_store import create_retriever, get_or_create_vector_store
from app.utils.topic_guard import get_off_topic_response, is_water_quality_related


# ── System Prompt ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Water Quality Assistant for a Water Quality Monitoring Dashboard.
Your sole purpose is to answer questions about:
- Water quality parameters (pH, DO, BOD, TDS, Turbidity, Nitrate, Coliform)
- Water Quality Index (WQI) and its categories
- Water bodies, rivers, lakes, and reservoirs
- Water pollution and contamination
- WHO drinking water guidelines and standards

STRICT RULES:
1. ONLY answer questions related to water quality topics listed above.
2. If asked about anything unrelated (politics, sports, entertainment, cooking, etc.), respond EXACTLY with:
   "I am a Water Quality Assistant and can only answer questions related to water quality data, WQI, water bodies, pollution indicators, and WHO water quality standards."
3. Base your answers PRIMARILY on the retrieved context provided below.
4. When context is insufficient, you may use general water quality knowledge.
5. Always cite specific values from the context when discussing a water body.
6. Be precise with units: mg/L for DO/BOD/TDS/Nitrate, NTU for Turbidity, CFU/100mL for Coliform.
7. When discussing WQI categories, reference the scale: Excellent (90-100), Good (70-89), Medium (50-69), Bad (25-49), Very Bad (0-24).

Retrieved Context:
{context}

Answer the following question based on the context above:
"""

HUMAN_PROMPT = "{question}"


# ── Prompt Template ────────────────────────────────────────────

def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])


# ── Context Formatter ──────────────────────────────────────────

def _format_docs(docs: list[Document]) -> str:
    """
    Format retrieved documents into a single context string.
    Numbers each source for traceability.
    """
    if not docs:
        return "No relevant data found in the knowledge base."
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        label = (
            doc.metadata.get("water_body", "Dataset Record")
            if source == "dataset"
            else doc.metadata.get("topic", "WHO Guideline")
        )
        parts.append(f"[Source {i} — {source.upper()} | {label}]\n{doc.page_content}")
    return "\n\n".join(parts)


# ── RAG Service Class ──────────────────────────────────────────

class WaterQualityRAGService:
    """
    Singleton service managing the full RAG pipeline lifecycle.

    Usage:
        service = WaterQualityRAGService()
        service.initialize()   # Call once at startup
        response = service.generate_response("What is the WQI of Sabarmati?")
    """

    def __init__(self) -> None:
        self._llm: ChatGroq | None = None
        self._retriever = None
        self._chain = None
        self._is_initialized = False

    def initialize(self, csv_path: str | None = None) -> None:
        """
        Initialize the full RAG pipeline:
          1. Load / create the FAISS vector store.
          2. Create the Groq LLM client.
          3. Build the LCEL chain.

        Args:
            csv_path: Optional override for the CSV dataset path.
        """
        if self._is_initialized:
            logger.debug("RAG service already initialized — skipping")
            return

        logger.info("Initializing Water Quality RAG service…")

        # Step 1: Load or build vector store
        documents = load_all_documents(csv_path)
        vector_store = get_or_create_vector_store(documents)
        self._retriever = create_retriever(vector_store)

        # Step 2: Initialize Groq LLM
        self._llm = ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=settings.groq_model_name,
            temperature=0.1,          # Low temp for factual, consistent answers
            max_tokens=1024,
        )

        # Step 3: Build LCEL chain
        prompt = _build_prompt()
        self._chain = (
            {
                "context": self._retriever | _format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self._llm
            | StrOutputParser()
        )

        self._is_initialized = True
        logger.info("RAG service initialized successfully")

    def generate_response(self, query: str) -> dict[str, Any]:
        """
        Generate an answer for the given user query using RAG.

        Applies topic guard before invoking the LLM.

        Args:
            query: The user's question string.

        Returns:
            Dict with keys:
              - answer (str): The generated answer or rejection message.
              - sources (list[dict]): Metadata of retrieved documents.
              - is_on_topic (bool): Whether the query passed the topic guard.
              - model (str): The LLM model used.
        """
        if not self._is_initialized:
            raise RuntimeError(
                "RAG service is not initialized. Call `service.initialize()` first."
            )

        # ── Topic Guard ──────────────────────────────────────
        if not is_water_quality_related(query):
            logger.info(f"Off-topic query blocked: '{query[:80]}'")
            return {
                "answer": get_off_topic_response(),
                "sources": [],
                "is_on_topic": False,
                "model": settings.groq_model_name,
            }

        # ── Retrieval + Generation ───────────────────────────
        logger.info(f"Processing query: '{query[:80]}'")
        try:
            # Retrieve relevant documents for source attribution
            retrieved_docs: list[Document] = self._retriever.invoke(query)

            # Generate answer via the LCEL chain
            answer: str = self._chain.invoke(query)

            # Build source metadata for API / UI display
            sources = [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "water_body": doc.metadata.get("water_body", ""),
                    "location": doc.metadata.get("location", ""),
                    "topic": doc.metadata.get("topic", ""),
                    "wqi": doc.metadata.get("wqi", None),
                    "wqi_category": doc.metadata.get("wqi_category", ""),
                }
                for doc in retrieved_docs
            ]

            logger.info(f"Response generated | sources retrieved: {len(sources)}")
            return {
                "answer": answer,
                "sources": sources,
                "is_on_topic": True,
                "model": settings.groq_model_name,
            }

        except Exception as exc:
            logger.exception("Error generating response")
            return {
                "answer": (
                    "I encountered an error while processing your question. "
                    "Please try again or rephrase your query."
                ),
                "sources": [],
                "is_on_topic": True,
                "model": settings.groq_model_name,
                "error": str(exc),
            }

    @property
    def is_ready(self) -> bool:
        """True if the service has been successfully initialized."""
        return self._is_initialized


# ── Module-level singleton ────────────────────────────────────
# Import `rag_service` in API routes and Streamlit app.
# Call rag_service.initialize() once at startup.
rag_service = WaterQualityRAGService()


# ── Convenience functions (for FastAPI dependency injection) ──

def get_rag_service() -> WaterQualityRAGService:
    """FastAPI dependency: returns the initialized RAG service."""
    return rag_service
