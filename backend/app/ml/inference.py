"""
ML Inference Pipeline — runs the CycloNexModel on a DataFrame,
returns a structured prediction dict.
Falls back to synthetic (rule-based) inference if model weights don't exist.
"""
from __future__ import annotations
import numpy as np
import torch
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from app.ml.model import CycloNexModel, build_model, load_model
from app.ml.gradcam import generate_xai_image, generate_evidence_caption
from app.data.base import DataFrame
from app.schemas.cyclone import IMD_CATEGORIES, wind_to_category
from app.config import settings

logger = logging.getLogger(__name__)

CHANNELS_ORDER = ["sst", "olr", "qpe", "cmv", "wvw", "uth"]
TRACK_HOURS = [6, 12, 18, 24]
TREND_LABELS = ["Intensifying", "Steady", "Weakening"]
CONF_THRESHOLD = 0.60  # flag for human review below this

_model: Optional[CycloNexModel] = None
_model_loaded = False


def get_model() -> Optional[CycloNexModel]:
    global _model, _model_loaded
    if _model_loaded:
        return _model
    _model_loaded = True
    path = settings.MODEL_WEIGHTS_PATH
    _model = load_model(path)
    if _model:
        logger.info(f"✅ Model loaded from {path}")
    else:
        logger.warning(f"⚠️ No model weights at {path}. Using synthetic inference fallback.")
    return _model


def frame_to_tensor(frame: DataFrame, T: int = 1) -> torch.Tensor:
    """Convert a DataFrame to (1, T, C, H, W) tensor, normalized."""
    ch_array = frame.as_tensor_channels(CHANNELS_ORDER)  # (C, H, W)
    # Per-channel normalization to [0, 1]
    for i in range(ch_array.shape[0]):
        mn, mx = ch_array[i].min(), ch_array[i].max()
        if mx > mn:
            ch_array[i] = (ch_array[i] - mn) / (mx - mn)
    tensor = torch.tensor(ch_array, dtype=torch.float32)
    # Add batch + time dims: (1, 1, C, H, W)
    return tensor.unsqueeze(0).unsqueeze(0)


def _synthetic_inference(frame: DataFrame) -> dict:
    """
    Rule-based fallback inference when model weights are unavailable.
    Uses channel statistics to produce plausible outputs.
    """
    meta = frame.metadata
    wind = float(meta.get("max_wind_kt", 40.0))
    lat = float(meta.get("center_lat", 15.0))
    lon = float(meta.get("center_lon", 85.0))

    # Simulate detection probability based on wind speed
    det_prob = min(0.95, max(0.1, (wind - 15) / 100.0))
    detected = wind >= 17.0

    cat_idx, cat_name = wind_to_category(wind)
    cat_idx = max(0, cat_idx)

    # Intensity trend from OLR gradient simulation
    olr = frame.channels.get("olr", np.zeros((1, 1)))
    olr_mean = float(olr.mean())
    if olr_mean < 220:
        trend = "Intensifying"
    elif olr_mean > 260:
        trend = "Weakening"
    else:
        trend = "Steady"

    # Track forecast: simple linear extrapolation with random walk
    rng = np.random.default_rng(42)
    track = []
    clat, clon = lat, lon
    for h in TRACK_HOURS:
        clat += rng.uniform(-0.3, 0.5) * (h / 6)
        clon += rng.uniform(-0.2, 0.8) * (h / 6)
        r = max(50, 150 - h * 3)
        track.append({"hour": h, "lat": round(clat, 2), "lon": round(clon, 2),
                       "wind_kt": max(20, wind - h * 0.8), "confidence_radius_km": r})

    conf = min(0.95, det_prob + 0.15)

    return {
        "detection_prob": round(det_prob, 3),
        "detected": detected,
        "center_lat": round(lat, 2),
        "center_lon": round(lon, 2),
        "category": cat_name if detected else None,
        "category_code": cat_idx if detected else None,
        "intensity_trend": trend,
        "max_wind_kt": round(wind, 1),
        "wind_24h_kt": round(max(20, wind - 15), 1),
        "confidence": round(conf, 3),
        "flagged_for_review": conf < CONF_THRESHOLD,
        "xai_image_b64": None,
        "xai_evidence": generate_evidence_caption({
            "category": cat_name, "intensity_trend": trend,
            "confidence": conf, "max_wind_kt": wind
        }, list(IMD_CATEGORIES.values())),
        "track_forecast": track,
    }


def run_inference(
    frames: list[DataFrame],
    use_xai: bool = True,
    xai_output_path: Optional[str] = None,
) -> dict:
    """
    Run full inference pipeline.
    Args:
        frames: list of DataFrames (sequence for temporal modeling)
        use_xai: whether to generate Grad-CAM overlay
        xai_output_path: path to save XAI PNG
    Returns:
        structured prediction dict
    """
    if not frames:
        raise ValueError("No frames provided for inference")

    model = get_model()

    # Use latest frame for metadata
    latest = frames[-1]

    if model is None:
        logger.info("Using synthetic inference fallback")
        return _synthetic_inference(latest)

    try:
        # Build tensor sequence: (1, T, C, H, W)
        tensors = [frame_to_tensor(f, T=1).squeeze(1) for f in frames]  # list of (1, C, H, W)
        # Pad/truncate to last 4 frames
        tensors = tensors[-4:]
        x = torch.cat([t.unsqueeze(1) for t in tensors], dim=1)  # (1, T, C, H, W)

        model.eval()
        with torch.set_grad_enabled(use_xai):
            out = model(x)

        det_prob = torch.sigmoid(out["detection_logit"])[0, 0].item()
        detected = det_prob >= 0.5

        loc = out["localization"][0].detach().numpy()  # (2,) — lat delta, lon delta
        meta = latest.metadata
        center_lat = float(meta.get("center_lat", 15.0)) + float(loc[0]) * 2.0
        center_lon = float(meta.get("center_lon", 85.0)) + float(loc[1]) * 2.0
        center_lat = float(np.clip(center_lat, 0, 30))
        center_lon = float(np.clip(center_lon, 60, 100))

        cls_probs = torch.softmax(out["classification"][0], dim=0).detach().numpy()
        cat_idx = int(np.argmax(cls_probs))
        cat_name = IMD_CATEGORIES[cat_idx]

        trend_probs = torch.softmax(out["intensity_trend"][0], dim=0).detach().numpy()
        trend = TREND_LABELS[int(np.argmax(trend_probs))]

        track_raw = out["track_forecast"][0].detach().numpy()  # (4, 2)
        track = []
        for i, h in enumerate(TRACK_HOURS):
            r = max(50, 200 - h * 5)
            track.append({
                "hour": h,
                "lat": round(center_lat + float(track_raw[i, 0]) * 2, 2),
                "lon": round(center_lon + float(track_raw[i, 1]) * 2, 2),
                "wind_kt": None,
                "confidence_radius_km": r,
            })

        winds = out["wind"][0].detach().numpy()
        wind_now = float(winds[0]) * 50 + 30   # scale output to realistic range
        wind_24h = float(winds[1]) * 50 + 25

        unc = torch.sigmoid(out["uncertainty_logit"])[0, 0].item()
        conf = float(1.0 - unc * 0.5)
        conf = max(0.35, min(0.99, conf))

        # Track wind
        for tp in track:
            step = TRACK_HOURS.index(tp["hour"])
            tp["wind_kt"] = round(wind_now + (wind_24h - wind_now) * step / len(TRACK_HOURS), 1)

        # XAI
        xai_b64 = None
        if use_xai:
            try:
                _, xai_b64 = generate_xai_image(
                    model, x, target_class=cat_idx, output_path=xai_output_path
                )
            except Exception as xe:
                logger.warning(f"XAI generation error: {xe}")

        evidence = generate_evidence_caption({
            "category": cat_name, "intensity_trend": trend,
            "confidence": conf, "max_wind_kt": wind_now
        }, list(IMD_CATEGORIES.values()))

        return {
            "detection_prob": round(det_prob, 3),
            "detected": detected,
            "center_lat": round(center_lat, 2),
            "center_lon": round(center_lon, 2),
            "category": cat_name if detected else None,
            "category_code": cat_idx if detected else None,
            "intensity_trend": trend,
            "max_wind_kt": round(wind_now, 1),
            "wind_24h_kt": round(wind_24h, 1),
            "confidence": round(conf, 3),
            "flagged_for_review": conf < CONF_THRESHOLD,
            "xai_image_b64": xai_b64,
            "xai_evidence": evidence,
            "track_forecast": track,
        }

    except Exception as e:
        logger.error(f"Model inference failed: {e}. Falling back to synthetic.")
        return _synthetic_inference(latest)
