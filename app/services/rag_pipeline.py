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

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.core.config import settings
from app.core.logger import logger
from app.services.simple_retriever import SimpleRetriever
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

def _format_matched_rows(matched_items: list[dict]) -> str:
    """
    Format matched rows and WHO guidelines into a single context string.
    Numbers each source for traceability.
    """
    if not matched_items:
        return "No relevant data found in the knowledge base."
    parts = []
    for i, item in enumerate(matched_items, 1):
        source = item.get("source", "unknown")
        if source == "dataset":
            water_body = item.get("water_body", "Unknown")
            location = item.get("location", "Unknown")
            wqi = item.get("wqi", "N/A")
            category = item.get("wqi_category", "Unknown")
            ph = item.get("ph", "N/A")
            do = item.get("do", "N/A")
            bod = item.get("bod", "N/A")
            tds = item.get("tds", "N/A")
            turbidity = item.get("turbidity", "N/A")
            nitrate = item.get("nitrate", "N/A")
            coliform = item.get("coliform", "N/A")

            content = (
                f"Water Body: {water_body}\n"
                f"Location: {location}\n"
                f"WQI: {wqi}\n"
                f"Category: {category}\n"
                f"pH: {ph}\n"
                f"DO: {do}\n"
                f"BOD: {bod}\n"
                f"TDS: {tds}\n"
                f"Turbidity: {turbidity}\n"
                f"Nitrate: {nitrate}\n"
                f"Coliform: {coliform}"
            )
            label = water_body
        else:
            topic = item.get("topic", "WHO Guideline")
            content = item.get("content", "")
            label = topic

        parts.append(f"[Source {i} — {source.upper()} | {label}]\n{content}")
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
        self._retriever: SimpleRetriever | None = None
        self._is_initialized = False

    def initialize(self, csv_path: str | None = None) -> None:
        """
        Initialize the RAG service:
          1. Instantiate SimpleRetriever (loads CSV).
          2. Create the Groq LLM client.

        Args:
            csv_path: Optional override for the CSV dataset path.
        """
        if self._is_initialized:
            logger.debug("RAG service already initialized — skipping")
            return

        logger.info("Initializing Water Quality RAG service using SimpleRetriever...")

        # Step 1: Initialize Retriever
        self._retriever = SimpleRetriever(csv_path)

        # Step 2: Initialize Groq LLM
        logger.info("Initializing Groq...")
        self._llm = ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=settings.groq_model_name,
            temperature=0.1,
            max_tokens=1024,
        )
        logger.info("Groq initialized")

        self._is_initialized = True
        logger.info("RAG service initialized successfully")

    def generate_response(self, query: str) -> dict[str, Any]:
        """
        Generate an answer for the given user query using RAG.

        Applies topic guard using retrieval-score-based detection.

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
            logger.info("Lazy loading RAG service...")
            self.initialize()

        # ── Retrieval + Score-based Topic Guard ───────────────────────────
        logger.info(f"Processing query: '{query[:80]}'")
        try:
            # Retrieve relevant records
            matched_rows = self._retriever.search(query)

            # Check top result relevance score
            top_score = matched_rows[0].get("score", 0.0) if matched_rows else 0.0

            # Pass 1 from topic guard: check for strong off-topic signals
            from app.utils.topic_guard import OFF_TOPIC_STRONG_SIGNALS
            query_lower = query.lower()
            has_strong_off_topic_signal = any(signal in query_lower for signal in OFF_TOPIC_STRONG_SIGNALS)

            # Accept only if score is above the configured relevance threshold and there are no off-topic signals
            is_on_topic = (top_score >= settings.relevance_threshold) and not has_strong_off_topic_signal

            if not is_on_topic:
                logger.info(f"Query blocked by score-based topic guard (score: {top_score:.1f}, off-topic: {has_strong_off_topic_signal}): '{query[:80]}'")
                return {
                    "answer": get_off_topic_response(),
                    "sources": [],
                    "is_on_topic": False,
                    "model": settings.groq_model_name,
                }

            # Build context from matched rows
            context_str = _format_matched_rows(matched_rows)

            # Generate answer via prompt template + Groq invocation
            prompt = _build_prompt()
            formatted_prompt = prompt.format_messages(context=context_str, question=query)
            
            response = self._llm.invoke(formatted_prompt)
            answer = response.content

            # Build source metadata for API / UI display exactly as before
            sources = [
                {
                    "source": item.get("source", "unknown"),
                    "water_body": item.get("water_body", ""),
                    "location": item.get("location", ""),
                    "topic": item.get("topic", ""),
                    "wqi": item.get("wqi", None),
                    "wqi_category": item.get("wqi_category", ""),
                }
                for item in matched_rows
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
