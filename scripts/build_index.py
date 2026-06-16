"""
scripts/build_index.py
──────────────────────────────────────────────────────────────
Helper script to verify the SimpleRetriever search and ranking logic.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --csv path/to/custom_data.csv
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
from app.services.simple_retriever import SimpleRetriever


def main(csv_path: str | None = None, force: bool = False) -> None:
    setup_logger()

    logger.info("=" * 60)
    logger.info("Water Quality SimpleRetriever Verification")
    logger.info("=" * 60)

    # ── Check if rebuild is needed ────────────────────────────
    # ── Initialize SimpleRetriever ────────────────────────────────
    logger.info("Initializing SimpleRetriever...")
    t0 = time.time()
    try:
        retriever = SimpleRetriever(csv_path)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        print(f"\n[ERROR] Error: {exc}")
        print(f"   Expected CSV at: {settings.dataset_path}")
        sys.exit(1)

    elapsed = time.time() - t0
    logger.info(f"Loaded retriever and dataset in {elapsed:.2f}s")

    # ── Sanity check: test a retrieval ────────────────────────
    logger.info("Running sanity check retrieval 1 (Water Body)…")
    test_query_1 = "What is the water quality of MAT RIVER?"
    t1 = time.time()
    results_1 = retriever.search(test_query_1, top_k=5)
    query_elapsed_1 = time.time() - t1

    logger.info("Running sanity check retrieval 2 (WHO Guideline)…")
    test_query_2 = "What is the WHO limit for pH?"
    t2 = time.time()
    results_2 = retriever.search(test_query_2, top_k=5)
    query_elapsed_2 = time.time() - t2

    print("\n" + "=" * 60)
    print("[OK] SimpleRetriever Operational")
    print("=" * 60)
    print(f"  Dataset size      : {len(retriever.dataset)} rows")
    print(f"  Query 1           : '{test_query_1}'")
    print(f"  Retrieval time 1  : {query_elapsed_1*1000:.1f}ms")
    print("  Top 3 Matches for Query 1:")
    for i, res in enumerate(results_1[:3], 1):
        source_type = res.get("source", "unknown").upper()
        if source_type == "DATASET":
            print(f"    {i}. [{source_type}] Water Body: {res.get('water_body')} | Location: {res.get('location')} | WQI: {res.get('wqi')}")
        else:
            print(f"    {i}. [{source_type}] Topic: {res.get('topic')}")
            
    print("-" * 60)
    print(f"  Query 2           : '{test_query_2}'")
    print(f"  Retrieval time 2  : {query_elapsed_2*1000:.1f}ms")
    print("  Top 3 Matches for Query 2:")
    for i, res in enumerate(results_2[:3], 1):
        source_type = res.get("source", "unknown").upper()
        if source_type == "DATASET":
            print(f"    {i}. [{source_type}] Water Body: {res.get('water_body')} | Location: {res.get('location')} | WQI: {res.get('wqi')}")
        else:
            print(f"    {i}. [{source_type}] Topic: {res.get('topic')}")
    print("=" * 60)
    print("\nYou can now start the app:")
    print("  Streamlit : streamlit run app/ui/streamlit_app.py")
    print("  FastAPI   : uvicorn app.api.main:app --reload\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify SimpleRetriever search capabilities for Water Quality RAG"
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
        help="Ignored (kept for backwards compatibility)",
    )
    args = parser.parse_args()
    main(csv_path=args.csv, force=args.force)
