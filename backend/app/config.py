from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "CycloNex"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./cyclonex.db"

    # Redis (optional — falls back to in-memory)
    REDIS_URL: Optional[str] = None

    # ML Model
    MODEL_WEIGHTS_PATH: str = "./ml_weights/cyclonex_model.pt"
    USE_GPU: bool = False

    # Data sources
    DATA_SOURCE: str = "synthetic"  # "synthetic" | "ibtracs" | "mosdac"
    IBTRACS_CSV_PATH: str = "./data/ibtracs_ni/ibtracs_NI.csv"

    # CORS
    ALLOWED_ORIGINS: list = [
        "http://localhost:5173",
        "http://localhost:4173",
        "http://localhost:3000",
        "*"
    ]

    # XAI output directory
    XAI_OUTPUT_DIR: str = "./static/xai"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
