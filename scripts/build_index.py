"""
scripts/build_index.py
──────────────────────────────────────────────────────────────
Standalone script to build (or rebuild) the FAISS vector index.

Run this once before starting the Streamlit app or FastAPI server,
or any time the dataset is updated.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --csv path/to/custom_data.csv
    python scripts/build_index.py --force   # Rebuild even if index exists

The script:
  1. Loads the water quality CSV dataset.
  2. Loads the WHO guidelines knowledge base.
  3. Converts all records to LangChain Documents.
  4. Generates HuggingFace embeddings (all-MiniLM-L6-v2).
  5. Saves the FAISS index to the path defined in .env.
"""

import argparse
import sys
import time
from pathlib import Path

# ── Ensure project root is on path ────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.core.logger import logger, setup_logger
from app.services.data_loader import load_all_documents
from app.services.vector_store import (
    create_vector_store,
    delete_vector_store,
)


def main(csv_path: str | None = None, force: bool = False) -> None:
    setup_logger()

    logger.info("=" * 60)
    logger.info("Water Quality FAISS Index Builder")
    logger.info("=" * 60)

    index_path = settings.faiss_index_path_obj

    # ── Check if rebuild is needed ────────────────────────────
    if index_path.exists() and not force:
        logger.warning(
            f"FAISS index already exists at '{index_path}'. "
            "Use --force to rebuild it."
        )
        print(f"\n✅ Index already exists at: {index_path}")
        print("   Use --force flag to rebuild it.\n")
        return

    if index_path.exists() and force:
        logger.info("Force rebuild: deleting existing index…")
        delete_vector_store()

    # ── Load documents ────────────────────────────────────────
    logger.info("Loading documents from dataset and WHO guidelines…")
    t0 = time.time()
    try:
        documents = load_all_documents(csv_path)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        print(f"\n❌ Error: {exc}")
        print(f"   Expected CSV at: {settings.dataset_path}")
        sys.exit(1)

    logger.info(f"Loaded {len(documents)} documents in {time.time() - t0:.1f}s")

    # ── Build index ───────────────────────────────────────────
    logger.info("Building FAISS index (this may take a minute)…")
    t1 = time.time()
    vector_store = create_vector_store(documents)
    elapsed = time.time() - t1

    # ── Sanity check: test a retrieval ────────────────────────
    logger.info("Running sanity check retrieval…")
    test_query = "What is the water quality of Sabarmati River?"
    results = vector_store.similarity_search(test_query, k=2)
    logger.info(f"Sanity check: retrieved {len(results)} documents for test query")
    for i, doc in enumerate(results):
        logger.debug(f"  Result {i+1}: {doc.page_content[:80]}…")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ FAISS Index Built Successfully")
    print("=" * 60)
    print(f"  Documents indexed : {len(documents)}")
    print(f"  Index saved at    : {settings.faiss_index_path}")
    print(f"  Time taken        : {elapsed:.1f}s")
    print(f"  Embedding model   : {settings.embedding_model_name}")
    print(f"  Sanity check      : {len(results)} docs retrieved ✓")
    print("=" * 60)
    print("\nYou can now start the app:")
    print("  Streamlit : streamlit run app/ui/streamlit_app.py")
    print("  FastAPI   : uvicorn app.api.main:app --reload\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build or rebuild the FAISS vector index for Water Quality RAG"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to the water quality CSV dataset (overrides .env DATASET_PATH)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the index even if it already exists",
    )
    args = parser.parse_args()
    main(csv_path=args.csv, force=args.force)
