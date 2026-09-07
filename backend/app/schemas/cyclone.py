from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime


# ── IMD Category constants ──────────────────────────────────────────────────
IMD_CATEGORIES = {
    0: "Depression (D)",
    1: "Deep Depression (DD)",
    2: "Cyclonic Storm (CS)",
    3: "Severe Cyclonic Storm (SCS)",
    4: "Very Severe Cyclonic Storm (VSCS)",
    5: "Extremely Severe Cyclonic Storm (ESCS)",
    6: "Super Cyclone (SuCS)",
}

WIND_THRESHOLDS = [(17, 27), (28, 33), (34, 47), (48, 63), (64, 89), (90, 119), (120, 9999)]


def wind_to_category(wind_kt: float) -> tuple[int, str]:
    for idx, (lo, hi) in enumerate(WIND_THRESHOLDS):
        if lo <= wind_kt <= hi:
            return idx, IMD_CATEGORIES[idx]
    if wind_kt < 17:
        return -1, "Sub-Depression"
    return 6, IMD_CATEGORIES[6]


# ── Track point schemas ──────────────────────────────────────────────────────
class TrackPointOut(BaseModel):
    id: int
    timestamp: datetime
    lat: float
    lon: float
    max_wind_kt: Optional[float]
    pressure_hpa: Optional[float]
    category: Optional[str]
    is_forecast: bool
    forecast_hour: Optional[int]
    confidence_radius_km: Optional[float]

    model_config = {"from_attributes": True}


# ── Cyclone list / summary ───────────────────────────────────────────────────
class CycloneSummary(BaseModel):
    id: int
    name: str
    year: int
    status: str
    category: Optional[str]
    category_code: Optional[int]
    max_wind_kt: Optional[float]
    current_lat: Optional[float]
    current_lon: Optional[float]
    confidence: float
    flagged_for_review: bool
    source: str
    updated_at: datetime

    model_config = {"from_attributes": True}


# ── Cyclone full detail ──────────────────────────────────────────────────────
class CycloneDetail(CycloneSummary):
    min_pressure_hpa: Optional[float]
    extra: Optional[Any]
    track_points: List[TrackPointOut] = []

    model_config = {"from_attributes": True}


# ── Prediction ───────────────────────────────────────────────────────────────
class TrackForecastPoint(BaseModel):
    hour: int
    lat: float
    lon: float
    wind_kt: Optional[float]
    confidence_radius_km: Optional[float]


class PredictionOut(BaseModel):
    id: int
    cyclone_id: Optional[int]
    timestamp: datetime
    detection_prob: float
    detected: bool
    center_lat: Optional[float]
    center_lon: Optional[float]
    category: Optional[str]
    category_code: Optional[int]
    intensity_trend: Optional[str]
    max_wind_kt: Optional[float]
    confidence: float
    flagged_for_review: bool
    xai_image_url: Optional[str]
    xai_evidence: Optional[str]
    track_forecast: Optional[List[TrackForecastPoint]]

    model_config = {"from_attributes": True}


class PredictRequest(BaseModel):
    data_source: str = Field(default="synthetic", description="synthetic | ibtracs | mosdac")
    cyclone_id: Optional[int] = Field(default=None, description="If provided, run inference for this cyclone")
    use_xai: bool = Field(default=True)


# ── Alerts ───────────────────────────────────────────────────────────────────
class AlertOut(BaseModel):
    tier: str        # Advisory | Track Forecast | High Alert | Emergency
    color: str       # green | yellow | orange | red
    cyclone_id: int
    cyclone_name: str
    category: Optional[str]
    wind_kt: Optional[float]
    message: str
    timestamp: datetime


# ── Review ───────────────────────────────────────────────────────────────────
class ReviewRequest(BaseModel):
    cyclone_id: Optional[int] = None
    prediction_id: Optional[int] = None
    action: str = Field(..., pattern="^(accept|override|annotate)$")
    analyst_note: Optional[str] = None
    overridden_category: Optional[str] = None
    overridden_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    analyst_id: str = "analyst@imd.gov.in"


class ReviewOut(BaseModel):
    id: int
    cyclone_id: Optional[int]
    prediction_id: Optional[int]
    action: str
    analyst_note: Optional[str]
    overridden_category: Optional[str]
    timestamp: datetime

    model_config = {"from_attributes": True}


# ── Health ───────────────────────────────────────────────────────────────────
class HealthOut(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str
    model_loaded: bool
    data_source: str
    db_connected: bool
    version: str
