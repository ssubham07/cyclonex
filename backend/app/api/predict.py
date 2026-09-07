"""
POST /api/v1/predict
Runs the full CycloNex ML pipeline:
  Preprocessing (OpenCV+NumPy+Pandas) → CNN+TL → LSTM/GRU → PipelinePrediction
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from ..ml.pipeline import run_pipeline
from ..database    import AsyncSessionLocal
from sqlalchemy    import select, func
from ..models.cyclone import Cyclone, TrackPoint, Prediction

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["predict"])


# ── Request schema ────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    data_source:  str  = "synthetic"
    cyclone_id:   Optional[int] = None
    use_xai:      bool = False
    # Override sliders (optional, from frontend)
    sea_surface_temperature: Optional[float] = None
    atmospheric_pressure:    Optional[float] = None
    humidity:                Optional[float] = None
    wind_shear:              Optional[float] = None
    latitude:                Optional[float] = None
    ocean_depth:             Optional[float] = None
    proximity_to_coastline:  Optional[float] = None


# ── Response schema ───────────────────────────────────────────────────────────
class TrackPoint_(BaseModel):
    hour:             int
    lat:              float
    lon:              float
    wind_kt:          Optional[float] = None
    confidence_radius_km: Optional[float] = None

class WindForecast_(BaseModel):
    hour:    int
    wind_kt: float
    lower:   float
    upper:   float

class LayerActivation_(BaseModel):
    layer:    str
    mean_act: float
    max_act:  Optional[float] = None
    n_maps:   Optional[int]   = None
    feat_dim: Optional[int]   = None

class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    # Core
    detected:           bool
    detection_prob:     float
    category:           Optional[str]    = None
    category_code:      Optional[int]    = None
    max_wind_kt:        Optional[float]  = None
    confidence:         float
    center_lat:         Optional[float]  = None
    center_lon:         Optional[float]  = None

    # LSTM outputs
    intensity_trend:    Optional[str]    = None
    intensity_probs:    Optional[List[float]] = None
    wind_24h_kt:        Optional[float]  = None
    track_forecast:     Optional[List[TrackPoint_]]   = None
    wind_forecast:      Optional[List[WindForecast_]] = None

    # CNN outputs
    cnn_class_probs:    Optional[List[float]]  = None
    cnn_layer_acts:     Optional[List[LayerActivation_]] = None
    cnn_feature_snippet: Optional[List[float]] = None

    # Preprocessing
    preprocess_stats:   Optional[Dict[str, Any]] = None
    channel_png:        Optional[str]  = None     # base64 OLR image

    # Meta
    xai_evidence:       Optional[str]  = None
    flagged_for_review: bool           = False
    pipeline_steps:     Optional[List[str]] = None
    model_stack:        Optional[Dict[str, Any]] = None
    timestamp:          Optional[str]  = None


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Full ML prediction pipeline:
      1. Preprocess satellite channels (OpenCV + NumPy + Pandas)
      2. CNN + Transfer-Learning detection/classification
      3. LSTM + GRU track & intensity forecast
    """
    # Determine cyclone parameters from DB or request
    center_lat     = req.latitude or 15.0
    center_lon     = 82.0
    intensity_proxy = 0.70
    track_points: list = []

    if req.cyclone_id:
        try:
            async with AsyncSessionLocal() as session:
                cy = await session.get(Cyclone, req.cyclone_id)
                if cy:
                    center_lat      = float(cy.current_lat  or center_lat)
                    center_lon      = float(cy.current_lon  or center_lon)
                    intensity_proxy = min(1.0, float(cy.max_wind_kt or 80) / 130.0)

                # Fetch last 8 historical track points
                stmt = (
                    select(TrackPoint)
                    .where(TrackPoint.cyclone_id == req.cyclone_id)
                    .where(TrackPoint.is_forecast == False)
                    .order_by(TrackPoint.timestamp.desc())
                    .limit(8)
                )
                result = await session.execute(stmt)
                tps = result.scalars().all()
                track_points = [
                    {"wind_kt": t.wind_kt, "pressure_hpa": t.pressure_hpa, "lat": t.lat, "lon": t.lon}
                    for t in reversed(tps)
                ]
        except Exception as e:
            logger.warning("DB fetch failed: %s", e)

    # Override with slider values if provided
    if req.sea_surface_temperature is not None:
        # Scale SST to intensity proxy
        sst = req.sea_surface_temperature
        intensity_proxy = max(0.1, min(1.0, (sst - 20.0) / 15.0))
    if req.latitude is not None:
        center_lat = req.latitude

    # Run the full pipeline
    try:
        pred = run_pipeline(
            center_lat=center_lat,
            center_lon=center_lon,
            intensity_proxy=intensity_proxy,
            track_points=track_points or None,
            cyclone_id=req.cyclone_id,
        )
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"ML pipeline error: {e}")

    # Save to DB (non-blocking)
    if req.cyclone_id:
        try:
            async with AsyncSessionLocal() as session:
                db_pred = Prediction(
                    cyclone_id       = req.cyclone_id,
                    detected         = pred.detected,
                    detection_prob   = pred.detection_prob,
                    category         = pred.category,
                    max_wind_kt      = pred.max_wind_kt,
                    confidence       = pred.confidence,
                    intensity_trend  = pred.intensity_trend,
                    xai_evidence     = pred.xai_evidence,
                    flagged_for_review = pred.flagged_for_review,
                )
                session.add(db_pred)
                await session.commit()
        except Exception as e:
            logger.warning("Prediction DB save failed: %s", e)

    # Build response
    track_fc = [
        TrackPoint_(hour=t["hour"], lat=t["lat"], lon=t["lon"],
                    wind_kt=t.get("wind_kt"), confidence_radius_km=t.get("radius_km"))
        for t in (pred.track_forecast or [])
    ]
    wind_fc = [
        WindForecast_(hour=w["hour"], wind_kt=w["wind_kt"], lower=w["lower"], upper=w["upper"])
        for w in (pred.wind_forecast or [])
    ]
    layer_acts = [
        LayerActivation_(layer=a["layer"], mean_act=a["mean_act"],
                         max_act=a.get("max_act"), n_maps=a.get("n_maps"), feat_dim=a.get("feat_dim"))
        for a in (pred.cnn_layer_acts or [])
    ]

    return PredictResponse(
        detected           = pred.detected,
        detection_prob     = pred.detection_prob,
        category           = pred.category,
        category_code      = pred.category_code,
        max_wind_kt        = round(pred.max_wind_kt, 1),
        confidence         = round(pred.confidence, 4),
        center_lat         = pred.center_lat,
        center_lon         = pred.center_lon,
        intensity_trend    = pred.intensity_trend,
        intensity_probs    = [round(p, 4) for p in (pred.intensity_probs or [])],
        wind_24h_kt        = round(pred.wind_24h_kt, 1) if pred.wind_24h_kt else None,
        track_forecast     = track_fc,
        wind_forecast      = wind_fc,
        cnn_class_probs    = [round(p, 4) for p in (pred.cnn_class_probs or [])],
        cnn_layer_acts     = layer_acts,
        cnn_feature_snippet= [round(v, 4) for v in (pred.cnn_feature_snippet or [])],
        preprocess_stats   = pred.preprocess_stats or {},
        channel_png        = pred.channel_png if req.use_xai else None,
        xai_evidence       = pred.xai_evidence,
        flagged_for_review = pred.flagged_for_review,
        pipeline_steps     = pred.pipeline_steps,
        model_stack        = pred.model_stack,
        timestamp          = pred.timestamp,
    )
