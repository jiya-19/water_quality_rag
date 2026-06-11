import pandas as pd
from pathlib import Path
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logger import logger
from app.data.who_guidelines import WHO_KNOWLEDGE_BASE


# ── Helper: Row → natural language text ──────────────────────

def _row_to_text(row: pd.Series) -> str:
    """
    Convert one CSV row into a descriptive natural language
    paragraph. This gives the embedding model rich context
    rather than raw key=value pairs.
    """
    return (
        f"Water body: {row.get('Water Body Name', 'Unknown')} "
        f"located in {row.get('Location', 'Unknown')} "
        f"(Latitude: {row.get('Latitude', 'N/A')}, "
        f"Longitude: {row.get('Longitude', 'N/A')}). "
        f"Year = {row.get('Year', 'Unknown')}. "
        f"Water quality parameters: "
        f"pH = {row.get('pH', 'N/A')}, "
        f"Dissolved Oxygen (DO) = {row.get('Dissolved Oxygen (DO)', 'N/A')} mg/L, "
        f"Biological Oxygen Demand (BOD) = {row.get('Biological Oxygen Demand (BOD)', 'N/A')} mg/L, "
        f"Total Dissolved Solids (TDS) = {row.get('Total Dissolved Solids (TDS)', 'N/A')} mg/L, "
        f"Turbidity = {row.get('Turbidity', 'N/A')} NTU, "
        f"Nitrate = {row.get('Nitrate', 'N/A')} mg/L, "
        f"Coliform = {row.get('Coliform', 'N/A')} CFU/100 mL. "
        f"Water Quality Index (WQI) = {row.get('Water Quality Index (WQI)', 'N/A')}. "
        f"Water Quality Category: {row.get('Water Quality Category', 'Unknown')}."
    )


def _row_to_metadata(row: pd.Series, idx: int) -> dict:
    """
    Extract structured metadata from a row.
    Metadata is stored alongside embeddings and returned
    with each retrieved Document — useful for API responses.
    """
    return {
        "source": "dataset",
        "row_index": idx,
        "water_body": str(row.get("Water Body Name", "")),
        "location": str(row.get("Location", "")),
        "latitude": float(row.get("Latitude", 0.0) or 0.0),
        "longitude": float(row.get("Longitude", 0.0) or 0.0),
        "year": int(row.get("Year", 0) or 0),
        "ph": float(row.get("pH", 0.0) or 0.0),
        "do": float(row.get("Dissolved Oxygen (DO)", 0.0) or 0.0),
        "bod": float(row.get("Biological Oxygen Demand (BOD)", 0.0) or 0.0),
        "tds": float(row.get("Total Dissolved Solids (TDS)", 0.0) or 0.0),
        "turbidity": float(row.get("Turbidity", 0.0) or 0.0),
        "nitrate": float(row.get("Nitrate", 0.0) or 0.0),
        "coliform": float(row.get("Coliform", 0.0) or 0.0),
        "wqi": float(row.get("Water Quality Index (WQI)", 0.0) or 0.0),
        "wqi_category": str(row.get("Water Quality Category", "")),
    }


# ── Public API ────────────────────────────────────────────────

def load_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """
    Load the water quality CSV dataset into a DataFrame.

    Args:
        csv_path: Override path; defaults to settings.dataset_path.

    Returns:
        pd.DataFrame with cleaned column names.

    Raises:
        FileNotFoundError: If the CSV does not exist at the given path.
    """
    if csv_path:
        path = Path(csv_path)
        if not path.is_absolute() and not path.exists():
            from app.core.config import PROJECT_ROOT
            alt_path = PROJECT_ROOT / path
            if alt_path.exists():
                path = alt_path
    else:
        path = settings.dataset_path_obj

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Please place your CSV file at the configured DATASET_PATH "
            "or pass the path explicitly."
        )
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()   # Remove accidental whitespace
    logger.info(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns from '{path}'")
    return df


def dataset_to_documents(df: pd.DataFrame) -> list[Document]:
    """
    Convert a water quality DataFrame into a list of
    LangChain Documents suitable for embedding.

    Each row becomes one Document where:
      - page_content = natural language summary of the row
      - metadata     = structured dict with all numeric fields

    Args:
        df: DataFrame returned by load_data().

    Returns:
        List of LangChain Document objects.
    """
    documents = []
    for idx, row in df.iterrows():
        doc = Document(
            page_content=_row_to_text(row),
            metadata=_row_to_metadata(row, int(idx)),  # type: ignore[arg-type]
        )
        documents.append(doc)
    logger.info(f"Converted {len(documents)} rows to LangChain Documents")
    return documents


def who_guidelines_to_documents() -> list[Document]:
    """
    Convert the static WHO guideline knowledge base into
    LangChain Documents.

    Returns:
        List of LangChain Document objects for WHO guidelines.
    """
    documents = []
    for item in WHO_KNOWLEDGE_BASE:
        doc = Document(
            page_content=item["content"],
            metadata={
                "source": "who_guidelines",
                "topic": item["topic"],
            },
        )
        documents.append(doc)
    logger.info(f"Loaded {len(documents)} WHO guideline Documents")
    return documents


def load_all_documents(csv_path: str | Path | None = None) -> list[Document]:
    """
    Convenience function: load dataset + WHO guidelines,
    merge into a single document list for the vector store.

    Args:
        csv_path: Optional override for the CSV path.

    Returns:
        Combined list of Documents from both sources.
    """
    df = load_data(csv_path)
    dataset_docs = dataset_to_documents(df)
    who_docs = who_guidelines_to_documents()
    all_docs = dataset_docs + who_docs
    logger.info(
        f"Total documents for vector store: {len(all_docs)} "
        f"({len(dataset_docs)} dataset + {len(who_docs)} WHO)"
    )
    return all_docs
