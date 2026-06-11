from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logger import logger, setup_logger
from app.services.data_loader import load_data
from app.services.rag_pipeline import WaterQualityRAGService, get_rag_service
from app.services.vector_store import delete_vector_store


# ── Lifespan: startup / shutdown ──────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logger()
    logger.info("Starting Water Quality RAG API...")
    yield
    logger.info("API shutdown")


# ── FastAPI app ────────────────────────────────────────────────

app = FastAPI(
    title="Water Quality RAG API",
    description=(
        "RAG-powered chatbot API for water quality monitoring. "
        "Provides natural language Q&A over water body data and WHO guidelines."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS middleware (for React dashboard) ─────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for /chat endpoint."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The user's water quality question",
        example="What is the WQI of the Sabarmati River?",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session ID for conversation tracking",
    )


class SourceDocument(BaseModel):
    """Metadata about a retrieved source document."""
    source: str
    water_body: str = ""
    location: str = ""
    topic: str = ""
    wqi: float | None = None
    wqi_category: str = ""


class ChatResponse(BaseModel):
    """Response body for /chat endpoint."""
    answer: str
    sources: list[SourceDocument]
    is_on_topic: bool
    model: str
    session_id: str | None = None


class CompareRequest(BaseModel):
    """Request body for /compare endpoint."""
    body_a: str = Field(..., description="First water body name", example="Sabarmati River")
    body_b: str = Field(..., description="Second water body name", example="Tapi River")


class WaterBodyData(BaseModel):
    """Structured water body data for GET responses."""
    water_body: str
    location: str
    latitude: float
    longitude: float
    ph: float
    do: float
    bod: float
    tds: float
    turbidity: float
    nitrate: float
    coliform: float
    wqi: float
    wqi_category: str


class HealthResponse(BaseModel):
    status: str
    rag_ready: bool
    model: str
    version: str = "1.0.0"


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(service: WaterQualityRAGService = Depends(get_rag_service)):
    """Health check endpoint — verifies RAG service is operational."""
    return HealthResponse(
        status="ok" if service.is_ready else "degraded",
        rag_ready=service.is_ready,
        model=settings.groq_model_name,
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chatbot"])
async def chat(
    request: ChatRequest,
    service: WaterQualityRAGService = Depends(get_rag_service),
):
    """
    Ask a water quality question.
    Returns an AI-generated answer grounded in retrieved water body data
    and WHO guidelines. Off-topic questions are rejected.
    """
    result = service.generate_response(request.question)
    return ChatResponse(
        answer=result["answer"],
        sources=[SourceDocument(**s) for s in result.get("sources", [])],
        is_on_topic=result["is_on_topic"],
        model=result["model"],
        session_id=request.session_id,
    )


@app.post("/compare", response_model=ChatResponse, tags=["Analysis"])
async def compare_water_bodies(
    request: CompareRequest,
    service: WaterQualityRAGService = Depends(get_rag_service),
):
    """
    Compare water quality between two water bodies.
    Internally constructs a comparison question and routes
    it through the full RAG pipeline.
    """
    question = (
        f"Compare the water quality of {request.body_a} and {request.body_b}. "
        f"Include their WQI scores, WQI categories, and key parameter differences "
        f"such as pH, DO, BOD, TDS, Turbidity, Nitrate, and Coliform. "
        f"Which water body has better quality and why?"
    )
    result = service.generate_response(question)
    return ChatResponse(
        answer=result["answer"],
        sources=[SourceDocument(**s) for s in result.get("sources", [])],
        is_on_topic=result["is_on_topic"],
        model=result["model"],
    )


@app.get("/waterbody/{name}", response_model=WaterBodyData, tags=["Data"])
async def get_water_body(name: str):
    """
    Retrieve raw data for a specific water body by name.
    Performs a case-insensitive partial match on the name field.
    """
    try:
        df = load_data()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    mask = df["Water Body Name"].str.lower().str.contains(name.lower(), na=False)
    matches = df[mask]

    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No water body found matching '{name}'",
        )

    row = matches.iloc[0]
    return WaterBodyData(
        water_body=str(row["Water Body Name"]),
        location=str(row["Location"]),
        latitude=float(row["Latitude"]),
        longitude=float(row["Longitude"]),
        ph=float(row["pH"]),
        do=float(row["Dissolved Oxygen (DO)"]),
        bod=float(row["Biological Oxygen Demand (BOD)"]),
        tds=float(row["Total Dissolved Solids (TDS)"]),
        turbidity=float(row["Turbidity"]),
        nitrate=float(row["Nitrate"]),
        coliform=float(row["Coliform"]),
        wqi=float(row["Water Quality Index (WQI)"]),
        wqi_category=str(row["Water Quality Category"]),
    )


@app.get("/wqi/{location}", tags=["Data"])
async def get_wqi_by_location(
    location: str,
    category: str | None = Query(default=None, description="Filter by WQI category"),
):
    """
    Get WQI data for all water bodies in a given location.
    Optionally filter by WQI category (Excellent, Good, Medium, Bad, Very Bad).

    Example: GET /wqi/Surat?category=Good
    """
    try:
        df = load_data()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    mask = df["Location"].str.lower().str.contains(location.lower(), na=False)
    matches = df[mask]

    if category:
        matches = matches[
            matches["Water Quality Category"].str.lower() == category.lower()
        ]

    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No water bodies found in location '{location}'"
                   + (f" with category '{category}'" if category else ""),
        )

    results = []
    for _, row in matches.iterrows():
        results.append({
            "water_body": str(row["Water Body Name"]),
            "location": str(row["Location"]),
            "wqi": float(row["Water Quality Index (WQI)"]),
            "wqi_category": str(row["Water Quality Category"]),
            "latitude": float(row["Latitude"]),
            "longitude": float(row["Longitude"]),
        })

    return {
        "location": location,
        "total_results": len(results),
        "water_bodies": results,
    }


@app.get("/waterbodies", tags=["Data"])
async def list_water_bodies(
    category: str | None = Query(default=None, description="Filter by WQI category"),
    min_wqi: float | None = Query(default=None, description="Minimum WQI score"),
    max_wqi: float | None = Query(default=None, description="Maximum WQI score"),
):
    """
    List all water bodies, optionally filtered by WQI range or category.
    Useful for the React dashboard map view.
    """
    try:
        df = load_data()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    if category:
        df = df[df["Water Quality Category"].str.lower() == category.lower()]
    if min_wqi is not None:
        df = df[df["Water Quality Index (WQI)"] >= min_wqi]
    if max_wqi is not None:
        df = df[df["Water Quality Index (WQI)"] <= max_wqi]

    return {
        "total": len(df),
        "water_bodies": df.to_dict(orient="records"),
    }


@app.post("/admin/rebuild-index", tags=["Admin"])
async def rebuild_index(service: WaterQualityRAGService = Depends(get_rag_service)):
    """
    Delete the current FAISS index and rebuild it from scratch.
    Use this endpoint after updating the dataset CSV.

    ⚠️  Protect this endpoint with authentication in production.
    """
    logger.warning("Index rebuild requested via API")
    delete_vector_store()

    # Reinitialize forces a fresh build
    service._is_initialized = False
    service.initialize()

    return {"status": "success", "message": "FAISS index rebuilt successfully"}

@app.get("/")
async def root():
    return {
        "message": "Water Quality RAG API Running",
        "status": "ok"
    }