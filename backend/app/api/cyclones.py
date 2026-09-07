"""
GET /api/v1/cyclones      — list all cyclones
GET /api/v1/cyclones/{id} — full detail
GET /api/v1/cyclones/{id}/track     — track history + forecast
GET /api/v1/cyclones/{id}/intensity — intensity history
GET /api/v1/cyclones/{id}/explanation — XAI heatmap + evidence
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from typing import List, Optional
import base64
import os

from app.database import get_db
from app.models.cyclone import Cyclone, TrackPoint, Prediction
from app.schemas.cyclone import CycloneSummary, CycloneDetail, TrackPointOut, PredictionOut
from app.config import settings

router = APIRouter(prefix="/api/v1/cyclones", tags=["cyclones"])


@router.get("", response_model=List[CycloneSummary])
async def list_cyclones(
    status: Optional[str] = None,
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    q = select(Cyclone).order_by(Cyclone.updated_at.desc()).limit(limit)
    if status:
        q = q.where(Cyclone.status == status)
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/{cyclone_id}", response_model=CycloneDetail)
async def get_cyclone(cyclone_id: int, db: AsyncSession = Depends(get_db)):
    q = select(Cyclone).where(Cyclone.id == cyclone_id).options(
        selectinload(Cyclone.track_points)
    )
    result = await db.execute(q)
    cyclone = result.scalar_one_or_none()
    if not cyclone:
        raise HTTPException(status_code=404, detail=f"Cyclone {cyclone_id} not found")
    return cyclone


@router.get("/{cyclone_id}/track", response_model=List[TrackPointOut])
async def get_track(cyclone_id: int, db: AsyncSession = Depends(get_db)):
    q = select(TrackPoint).where(TrackPoint.cyclone_id == cyclone_id).order_by(TrackPoint.timestamp)
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/{cyclone_id}/intensity")
async def get_intensity(cyclone_id: int, db: AsyncSession = Depends(get_db)):
    """Return intensity time series (historical + predicted)."""
    q = select(TrackPoint).where(TrackPoint.cyclone_id == cyclone_id).order_by(TrackPoint.timestamp)
    result = await db.execute(q)
    pts = result.scalars().all()
    return [
        {
            "timestamp": pt.timestamp.isoformat(),
            "wind_kt": pt.max_wind_kt,
            "pressure_hpa": pt.pressure_hpa,
            "category": pt.category,
            "is_forecast": pt.is_forecast,
            "forecast_hour": pt.forecast_hour,
        }
        for pt in pts
    ]


@router.get("/{cyclone_id}/explanation")
async def get_explanation(cyclone_id: int, db: AsyncSession = Depends(get_db)):
    """Return XAI heatmap (base64 PNG) and textual evidence."""
    # Get latest prediction for this cyclone
    q = select(Prediction).where(Prediction.cyclone_id == cyclone_id).order_by(Prediction.timestamp.desc()).limit(1)
    result = await db.execute(q)
    pred = result.scalar_one_or_none()

    if not pred:
        raise HTTPException(status_code=404, detail="No prediction found for this cyclone")

    # Check for saved XAI image
    xai_b64 = None
    if pred.xai_image_path and os.path.exists(pred.xai_image_path):
        with open(pred.xai_image_path, "rb") as f:
            xai_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "cyclone_id": cyclone_id,
        "prediction_id": pred.id,
        "xai_image_base64": xai_b64,
        "xai_image_url": f"/static/xai/{os.path.basename(pred.xai_image_path)}" if pred.xai_image_path else None,
        "evidence": pred.xai_evidence,
        "confidence": pred.confidence,
        "category": pred.category,
        "intensity_trend": pred.intensity_trend,
        "timestamp": pred.timestamp.isoformat(),
    }
