"""
app/core/logger.py
──────────────────────────────────────────────────────────────
Centralised logging using Loguru.
All modules should `from app.core.logger import logger`.
"""

import sys
from loguru import logger as _logger
from app.core.config import settings


def setup_logger() -> None:
    """
    Configure Loguru with the log level set in settings.
    Call once at application startup (main.py / streamlit_app.py).
    """
    _logger.remove()          # Remove the default stderr sink
    _logger.add(
        sys.stderr,
        level=settings.log_level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    # Persist logs to file in production
    if settings.is_production:
        _logger.add(
            "logs/water_quality_rag_{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
            encoding="utf-8",
        )


# Re-export the configured logger
logger = _logger
