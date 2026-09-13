"""
CycloNex API — Production Backend
Serves REST API only. Frontend is deployed on Vercel.
Deploys to Render.com as a Python web service.
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
    logger.info("🌪️  CycloNex API starting...")

    await init_db()
    logger.info("✅ Database ready")

    try:
        from app.services.seed import seed_database
        await seed_database()
    except Exception as e:
        logger.warning(f"Seed skipped: {e}")

    try:
        from app.ml.inference import get_model
        get_model()
    except Exception as e:
        logger.warning(f"Model pre-load (synthetic fallback active): {e}")

    # Ensure XAI static dir exists
    os.makedirs("./static/xai", exist_ok=True)

    port = os.environ.get("PORT", "8000")
    logger.info(f"🚀 CycloNex ready on :{port}  docs=/docs  health=/api/v1/health")
    yield
    logger.info("🛑 CycloNex shutting down")


app = FastAPI(
    title="CycloNex API",
    description=(
        "AI/ML Decision-Support System for Tropical Cyclone Intelligence.\n\n"
        "Built for **SIH 2026** (Problem SIH26070).\n\n"
        "⚠️ **Disclaimer**: Not an official IMD system."
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
# allow_credentials must be False when allow_origins includes "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # allows Vercel + any device
    allow_credentials=False,      # required when origins=["*"]
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Authorization"],
)

# ── Static files (XAI heatmaps) ──────────────────────────────────────────────
os.makedirs("./static/xai", exist_ok=True)
app.mount("/static", StaticFiles(directory="./static"), name="static")

# ── API Routers ───────────────────────────────────────────────────────────────
app.include_router(cyclones.router)   # prefix: /api/v1/cyclones
app.include_router(predict.router)   # prefix: /api/v1 (predict.py has it baked in)
app.include_router(alerts.router)    # prefix: /api/v1/alerts
app.include_router(review.router)
app.include_router(reports.router)
app.include_router(metrics.router, prefix="/api/v1")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/api/v1/health", tags=["health"])
async def health():
    from app.ml.inference import _model
    db_ok = False
    try:
        from app.database import engine
        import sqlalchemy
        async with engine.connect() as conn:
            await conn.execute(sqlalchemy.text("SELECT 1"))
        db_ok = True
    except Exception as ex:
        logger.warning(f"DB health: {ex}")

    return JSONResponse({
        "status": "ok",
        "version": settings.APP_VERSION,
        "db_connected": db_ok,
        "model_loaded": _model is not None,
        "model_weights_found": os.path.exists(settings.MODEL_WEIGHTS_PATH),
        "data_source": settings.DATA_SOURCE,
        "cors_origins": settings.cors_origins,
        "disclaimer": "Research prototype. Not an official IMD system.",
    })


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", tags=["root"])
async def root():
    return JSONResponse({
        "name": "CycloNex API",
        "version": settings.APP_VERSION,
        "status": "online",
        "endpoints": {
            "health": "/api/v1/health",
            "docs": "/docs",
            "predict": "POST /api/v1/predict",
            "cyclones": "/api/v1/cyclones",
            "alerts": "/api/v1/alerts",
            "metrics": "/api/v1/metrics",
        },
        "frontend": "https://cyclonex-iota.vercel.app",
    })
