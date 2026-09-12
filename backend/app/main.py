"""
CycloNex — AI Cyclone Intelligence System
FastAPI Main Application — Production Ready (Render.com / Railway)
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import init_db
from app.api import cyclones, predict, alerts, review, reports, metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB, seed data, warm up model."""
    logger.info("🌪️  CycloNex starting up...")

    # Init database tables
    await init_db()
    logger.info("✅ Database initialized")

    # Seed demo data
    try:
        from app.services.seed import seed_database
        await seed_database()
    except Exception as e:
        logger.warning(f"Seed skipped: {e}")

    # Pre-load model (non-blocking — synthetic fallback if weights missing)
    try:
        from app.ml.inference import get_model
        get_model()
    except Exception as e:
        logger.warning(f"Model pre-load: {e}")

    # Ensure static/XAI output dir exists
    os.makedirs(settings.XAI_OUTPUT_DIR, exist_ok=True)
    os.makedirs("./static", exist_ok=True)

    port = os.environ.get("PORT", "8000")
    logger.info(f"🚀 CycloNex ready — port {port} — /docs")
    yield

    logger.info("🛑 CycloNex shutting down...")


app = FastAPI(
    title="CycloNex API",
    description=(
        "AI/ML Decision-Support System for Tropical Cyclone Intelligence — North Indian Ocean. "
        "Built for SIH 2026 (SIH26070). "
        "**Disclaimer:** Not an official IMD system. Official warnings issued exclusively by IMD."
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS: allow Vercel frontend + all origins ────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ── Static files (XAI heatmap images) ───────────────────────────────────────
os.makedirs("./static", exist_ok=True)
app.mount("/static", StaticFiles(directory="./static"), name="static")

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(cyclones.router)
app.include_router(predict.router)
app.include_router(alerts.router)
app.include_router(review.router)
app.include_router(reports.router)
app.include_router(metrics.router, prefix="/api/v1")


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/api/v1/health", tags=["health"])
async def health():
    from app.ml.inference import _model_loaded, _model
    try:
        from app.database import engine
        import sqlalchemy
        async with engine.connect() as conn:
            await conn.execute(sqlalchemy.text("SELECT 1"))
        db_ok = True
    except Exception as ex:
        logger.warning(f"DB health check failed: {ex}")
        db_ok = False

    return JSONResponse({
        "status": "ok",
        "model_loaded": _model is not None,
        "model_weights_found": os.path.exists(settings.MODEL_WEIGHTS_PATH),
        "data_source": settings.DATA_SOURCE,
        "db_connected": db_ok,
        "version": settings.APP_VERSION,
        "frontend": settings.FRONTEND_URL,
        "disclaimer": "CycloNex is an AI/ML prototype. Not an official IMD warning system.",
    })


@app.get("/", tags=["root"])
async def root():
    return JSONResponse({
        "name": "CycloNex API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
        "frontend": settings.FRONTEND_URL,
        "status": "online",
    })
