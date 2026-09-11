from pydantic_settings import BaseSettings
from typing import Optional, List
import os

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "CycloNex"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Database — auto-detects PostgreSQL (Railway) or SQLite (local)
    DATABASE_URL: str = "sqlite+aiosqlite:///./cyclonex.db"

    @property
    def async_database_url(self) -> str:
        """Convert sync postgres:// → async postgresql+asyncpg://"""
        url = self.DATABASE_URL
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql://") and "asyncpg" not in url:
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url

    # Security
    SECRET_KEY: str = "cyclonex-dev-secret-change-in-production"

    # Redis (optional — falls back to in-memory)
    REDIS_URL: Optional[str] = None

    # ML Model
    MODEL_WEIGHTS_PATH: str = "./ml_weights/cyclonex_model.pt"
    USE_GPU: bool = False

    # Data sources
    DATA_SOURCE: str = "synthetic"
    IBTRACS_CSV_PATH: str = "./data/ibtracs_ni/ibtracs_NI.csv"

    # CORS — Vercel frontend + all origins for Railway
    ALLOWED_ORIGINS: List[str] = [
        "https://cyclonex-iota.vercel.app",
        "https://*.vercel.app",
        "http://localhost:5173",
        "http://localhost:4173",
        "*",
    ]

    # Frontend URL
    FRONTEND_URL: str = "https://cyclonex-iota.vercel.app"

    # XAI output directory
    XAI_OUTPUT_DIR: str = "./static/xai"

    # Port (Railway sets $PORT automatically)
    PORT: int = 8000

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
