"""
tests/test_services.py
──────────────────────────────────────────────────────────────
Unit tests for the Water Quality RAG services.

Tests are designed to run without a Groq API key or FAISS index
(data loading and topic guard tests), with lightweight mocking
for LLM-dependent tests.

Run:
    pytest tests/ -v
    pytest tests/ -v --tb=short    # shorter tracebacks
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
import pandas as pd
from langchain_core.documents import Document


# ── Data Loader Tests ─────────────────────────────────────────

class TestDataLoader:
    """Tests for app/services/data_loader.py"""

    def test_load_sample_csv(self, tmp_path):
        """load_data() should return a DataFrame from the sample CSV."""
        from app.services.data_loader import load_data

        csv_content = (
            "Water Body Name,Location,Latitude,Longitude,pH,"
            "Dissolved Oxygen (DO),Biological Oxygen Demand (BOD),"
            "Total Dissolved Solids (TDS),Turbidity,Nitrate,Coliform,"
            "Water Quality Index (WQI),Water Quality Category\n"
            "Test River,Test City,23.0,72.0,7.2,6.5,2.1,310,3.4,18.0,10,68.5,Good\n"
        )
        csv_file = tmp_path / "test_data.csv"
        csv_file.write_text(csv_content)

        df = load_data(csv_path=csv_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["Water Body Name"] == "Test River"

    def test_load_data_file_not_found(self):
        """load_data() should raise FileNotFoundError for missing file."""
        from app.services.data_loader import load_data
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            load_data(csv_path="/nonexistent/path/data.csv")

    def test_dataset_to_documents(self):
        """dataset_to_documents() should produce one Document per row."""
        from app.services.data_loader import dataset_to_documents

        df = pd.DataFrame([{
            "Water Body Name": "Sabarmati River",
            "Location": "Ahmedabad",
            "Latitude": 23.02,
            "Longitude": 72.57,
            "pH": 7.2,
            "Dissolved Oxygen (DO)": 5.8,
            "Biological Oxygen Demand (BOD)": 3.2,
            "Total Dissolved Solids (TDS)": 410,
            "Turbidity": 6.1,
            "Nitrate": 28.5,
            "Coliform": 45,
            "Water Quality Index (WQI)": 52.3,
            "Water Quality Category": "Medium",
        }])
        docs = dataset_to_documents(df)
        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert "Sabarmati River" in docs[0].page_content
        assert docs[0].metadata["source"] == "dataset"
        assert docs[0].metadata["water_body"] == "Sabarmati River"
        assert docs[0].metadata["wqi"] == 52.3

    def test_who_guidelines_to_documents(self):
        """who_guidelines_to_documents() should return non-empty document list."""
        from app.services.data_loader import who_guidelines_to_documents
        docs = who_guidelines_to_documents()
        assert len(docs) > 0
        for doc in docs:
            assert doc.metadata["source"] == "who_guidelines"
            assert "topic" in doc.metadata
            assert len(doc.page_content) > 50

    def test_load_all_documents_merges_sources(self, tmp_path):
        """load_all_documents() should merge dataset + WHO guidelines."""
        from app.services.data_loader import load_all_documents, who_guidelines_to_documents

        csv_content = (
            "Water Body Name,Location,Latitude,Longitude,pH,"
            "Dissolved Oxygen (DO),Biological Oxygen Demand (BOD),"
            "Total Dissolved Solids (TDS),Turbidity,Nitrate,Coliform,"
            "Water Quality Index (WQI),Water Quality Category\n"
            "River A,City A,23.0,72.0,7.0,6.0,2.0,300,3.0,15.0,5,70.0,Good\n"
            "River B,City B,22.0,73.0,7.5,7.0,1.5,250,2.0,10.0,2,85.0,Good\n"
        )
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(csv_content)

        all_docs = load_all_documents(csv_path=csv_file)
        who_count = len(who_guidelines_to_documents())

        # Should have 2 dataset docs + N WHO docs
        assert len(all_docs) == 2 + who_count

        sources = {doc.metadata["source"] for doc in all_docs}
        assert "dataset" in sources
        assert "who_guidelines" in sources


# ── Topic Guard Tests ─────────────────────────────────────────

class TestTopicGuard:
    """Tests for app/utils/topic_guard.py"""

    def test_water_quality_query_allowed(self):
        """On-topic water quality queries should pass the guard."""
        from app.utils.topic_guard import is_water_quality_related
        water_queries = [
            "What is the WQI of Sabarmati River?",
            "Is the pH of Tapi River within WHO guidelines?",
            "Which water bodies have high BOD?",
            "What is a safe level of nitrate in drinking water?",
            "Tell me about dissolved oxygen levels in rivers",
            "What does WQI of 45 mean?",
            "WHO drinking water standards for turbidity",
        ]
        for query in water_queries:
            assert is_water_quality_related(query), f"Should be allowed: '{query}'"

    def test_off_topic_query_blocked(self):
        """Off-topic queries should be blocked by the guard."""
        from app.utils.topic_guard import is_water_quality_related
        off_topic_queries = [
            "Who won the cricket match yesterday?",
            "What is the latest IPL score?",
            "Tell me about the upcoming election",
            "Recommend a good movie to watch",
            "What is the stock price of Apple?",
        ]
        for query in off_topic_queries:
            assert not is_water_quality_related(query), f"Should be blocked: '{query}'"

    def test_off_topic_response_is_string(self):
        """The rejection message should be a non-empty string."""
        from app.utils.topic_guard import get_off_topic_response
        msg = get_off_topic_response()
        assert isinstance(msg, str)
        assert len(msg) > 20
        assert "Water Quality" in msg


# ── WHO Guidelines Tests ──────────────────────────────────────

class TestWHOGuidelines:
    """Tests for app/data/who_guidelines.py"""

    def test_who_knowledge_base_structure(self):
        """WHO_KNOWLEDGE_BASE should be a list of dicts with required keys."""
        from app.data.who_guidelines import WHO_KNOWLEDGE_BASE
        assert isinstance(WHO_KNOWLEDGE_BASE, list)
        assert len(WHO_KNOWLEDGE_BASE) >= 8

        for item in WHO_KNOWLEDGE_BASE:
            assert "topic" in item, f"Missing 'topic' key in item: {item}"
            assert "content" in item, f"Missing 'content' key in item: {item}"
            assert len(item["content"]) > 50

    def test_who_topics_include_required_parameters(self):
        """WHO guidelines should cover all critical water quality parameters."""
        from app.data.who_guidelines import WHO_KNOWLEDGE_BASE
        topics = {item["topic"].lower() for item in WHO_KNOWLEDGE_BASE}
        required_topics = {"ph", "tds", "turbidity", "nitrate"}
        for topic in required_topics:
            assert any(topic in t for t in topics), f"Missing WHO guideline for: {topic}"


# ── Config Tests ──────────────────────────────────────────────

class TestConfig:
    """Tests for app/core/config.py"""

    def test_settings_defaults_exist(self):
        """Settings should have default values for non-secret fields."""
        from app.core.config import settings
        assert settings.groq_model_name != ""
        assert settings.embedding_model_name != ""
        assert settings.top_k_results > 0
        assert settings.app_port > 0

    def test_cors_origins_list_parsing(self):
        """cors_origins_list should parse comma-separated strings correctly."""
        from app.core.config import Settings
        s = Settings(cors_origins="http://localhost:3000,http://localhost:5173")
        origins = s.cors_origins_list
        assert len(origins) == 2
        assert "http://localhost:3000" in origins
        assert "http://localhost:5173" in origins
