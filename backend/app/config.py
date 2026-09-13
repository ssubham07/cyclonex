from pydantic_settings import BaseSettings
from typing import Optional, List
import os


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "CycloNex"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Database
    # Render sets DATABASE_URL automatically when PostgreSQL is added
    # Local dev falls back to SQLite
    DATABASE_URL: str = "sqlite+aiosqlite:///./cyclonex.db"

    @property
    def async_database_url(self) -> str:
        """Convert postgres:// → postgresql+asyncpg:// for async SQLAlchemy."""
        url = self.DATABASE_URL
        if url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql+asyncpg://", 1)
        if url.startswith("postgresql://") and "+asyncpg" not in url:
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url

    # Security
    SECRET_KEY: str = "cyclonex-dev-key-change-in-production"

    # ML Model
    MODEL_WEIGHTS_PATH: str = "./ml_weights/cyclonex_model.pt"
    USE_GPU: bool = False

    # Data
    DATA_SOURCE: str = "synthetic"
    IBTRACS_CSV_PATH: str = "./data/ibtracs_ni/ibtracs_NI.csv"

    # CORS — allow Vercel frontend + localhost dev
    # On Render: set FRONTEND_URL env var to your Vercel URL
    FRONTEND_URL: str = "https://cyclonex-iota.vercel.app"

    @property
    def cors_origins(self) -> List[str]:
        return [
            self.FRONTEND_URL,
            "https://cyclonex-iota.vercel.app",
            "https://*.vercel.app",
            "http://localhost:5173",
            "http://localhost:4173",
            "http://127.0.0.1:5173",
        ]

    # XAI output
    XAI_OUTPUT_DIR: str = "./static/xai"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
