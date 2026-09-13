"""
CycloNex — AI Cyclone Intelligence System
Full-Stack FastAPI: serves React frontend + REST API
Deployable to Render.com free tier as a single service.
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

from app.config import settings
from app.database import init_db
from app.api import cyclones, predict, alerts, review, reports, metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Path to built React app (copied here during Render build step)
FRONTEND_DIR = Path(__file__).parent.parent / "static" / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🌪️  CycloNex starting up...")
    await init_db()
    logger.info("✅ Database initialized")

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

    os.makedirs("./static/xai", exist_ok=True)
    os.makedirs(str(FRONTEND_DIR), exist_ok=True)

    port = os.environ.get("PORT", "8000")
    logger.info(f"🚀 CycloNex ready on port {port}")
    logger.info(f"   API docs : /docs")
    logger.info(f"   Health   : /api/v1/health")
    logger.info(f"   Frontend : {'BUILT' if (FRONTEND_DIR / 'index.html').exists() else 'NOT BUILT'}")
    yield
    logger.info("🛑 CycloNex shutting down...")


app = FastAPI(
    title="CycloNex API",
    description="AI/ML Tropical Cyclone Intelligence — SIH 2026 (SIH26070)",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — open for all (needed for Vercel ↔ Render cross-origin during dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Static XAI images
os.makedirs("./static/xai", exist_ok=True)
app.mount("/static/xai", StaticFiles(directory="./static/xai"), name="xai")

# ── API Routers ──────────────────────────────────────────────────────────────
app.include_router(cyclones.router)
app.include_router(predict.router)
app.include_router(alerts.router)
app.include_router(review.router)
app.include_router(reports.router)
app.include_router(metrics.router, prefix="/api/v1")


# ── Health ───────────────────────────────────────────────────────────────────
@app.get("/api/v1/health", tags=["health"])
async def health():
    from app.ml.inference import _model
    try:
        from app.database import engine
        import sqlalchemy
        async with engine.connect() as conn:
            await conn.execute(sqlalchemy.text("SELECT 1"))
        db_ok = True
    except Exception as ex:
        logger.warning(f"DB check failed: {ex}")
        db_ok = False

    return JSONResponse({
        "status": "ok",
        "model_loaded": _model is not None,
        "model_weights_found": os.path.exists(settings.MODEL_WEIGHTS_PATH),
        "data_source": settings.DATA_SOURCE,
        "db_connected": db_ok,
        "version": settings.APP_VERSION,
        "frontend_built": (FRONTEND_DIR / "index.html").exists(),
        "disclaimer": "CycloNex is a research prototype. Not an official IMD system.",
    })


# ── Serve React SPA (must be LAST — catch-all) ──────────────────────────────
@app.get("/", tags=["frontend"])
async def serve_root():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({
        "name": "CycloNex API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
        "note": "Frontend not built. Run build script first.",
    })


if FRONTEND_DIR.exists() and (FRONTEND_DIR / "index.html").exists():
    # Mount React assets (JS/CSS/images)
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

    # SPA catch-all — serve index.html for all unmatched routes
    @app.get("/{full_path:path}", tags=["frontend"])
    async def serve_spa(full_path: str):
        # Don't intercept API or docs routes
        if full_path.startswith(("api/", "docs", "redoc", "static/", "assets/")):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        return FileResponse(str(FRONTEND_DIR / "index.html"))
