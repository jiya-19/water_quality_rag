"""
app/core/config.py
──────────────────────────────────────────────────────────────
Centralised settings loaded from environment variables (.env).
Uses Pydantic BaseSettings for type-safe config management.
All modules should import `settings` from here — never read
os.environ directly.
"""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application-wide settings derived from .env / environment."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Groq LLM ────────────────────────────────────────────
    groq_api_key: str = ""
    groq_model_name: str = "llama-3.1-8b-instant"

    # ── Embeddings ──────────────────────────────────────────
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── FAISS / Vector Store ─────────────────────────────────
    faiss_index_path: str = "./app/data/faiss_index"

    # ── Dataset ─────────────────────────────────────────────
    dataset_path: str = "./app/data/water_quality_data.csv"

    # ── RAG ─────────────────────────────────────────────────
    top_k_results: int = 3

    # ── Application ─────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # ── CORS ────────────────────────────────────────────────
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def faiss_index_path_obj(self) -> Path:
        p = Path(self.faiss_index_path)
        if p.is_absolute():
            return p
        return PROJECT_ROOT / p

    @property
    def dataset_path_obj(self) -> Path:
        p = Path(self.dataset_path)
        if p.is_absolute():
            return p
        return PROJECT_ROOT / p

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached singleton Settings instance.
    Use @lru_cache so .env is read only once per process.
    """
    return Settings()


# Module-level shortcut — import this everywhere
settings: Settings = get_settings()
