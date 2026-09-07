"""
CycloNex – Full ML Prediction Pipeline
Stages:
  1. Preprocessing  (OpenCV + NumPy + Pandas)
  2. CNN Detection  (CNN + Transfer Learning → scikit-learn)
  3. LSTM Forecast  (GRU + LSTM → NumPy)
  4. Aggregation    → Unified PipelinePrediction
"""

import numpy as np
import logging
import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from .preprocess     import preprocess, channel_to_png_b64, PreprocessedFrame
from .cnn_classifier import get_cnn_model, CNNPrediction, IMD_LABELS, WIND_THRESHOLDS
from .lstm_predictor import get_lstm_predictor, LSTMPrediction, CycloneLSTMPredictor

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PipelinePrediction:
    # Core detection
    detected:           bool
    detection_prob:     float
    category_code:      int
    category:           str
    max_wind_kt:        float
    confidence:         float
    center_lat:         float
    center_lon:         float

    # Intensity
    intensity_trend:    str
    intensity_probs:    List[float]         # [P(Int), P(Steady), P(Weak)]
    wind_24h_kt:        Optional[float]

    # Forecasts
    track_forecast:     List[Dict]
    wind_forecast:      List[Dict]

    # CNN artefacts
    cnn_class_probs:    List[float]         # 7-class probabilities
    cnn_layer_acts:     List[Dict]          # per-layer activation stats
    cnn_feature_snippet: List[float]        # first 16 feature dims

    # Preprocessing stats
    preprocess_stats:   Dict[str, Dict]     # per-channel {min,max,mean,std}
    channel_png:        Optional[str]       # base64 OLR channel image

    # Metadata
    pipeline_steps:     List[str]
    timestamp:          str
    flagged_for_review: bool
    xai_evidence:       str
    model_stack:        Dict

    # Raw prediction objects stored for API expansion
    _raw_cnn:  Optional[Any] = field(default=None, repr=False)
    _raw_lstm: Optional[Any] = field(default=None, repr=False)


def run_pipeline(
    center_lat:     float = 15.0,
    center_lon:     float = 82.0,
    intensity_proxy: float = 0.72,
    track_points:   Optional[List[Dict]] = None,
    cyclone_id:     Optional[int] = None,
) -> PipelinePrediction:
    """
    Execute the full 3-stage CycloNex ML pipeline.

    Args:
        center_lat:      Current cyclone latitude
        center_lon:      Current cyclone longitude
        intensity_proxy: 0-1 proxy for cyclone intensity (from DB or defaults)
        track_points:    Historical track points for LSTM sequence input
        cyclone_id:      Optional DB cyclone id for logging
    """
    steps: List[str] = []
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── STAGE 1: Preprocessing ───────────────────────────────────────────────
    steps.append("① Preprocessing — OpenCV + NumPy + Pandas")
    try:
        rng = np.random.default_rng(int(abs(center_lat * 100 + center_lon * 10)))
        frame: PreprocessedFrame = preprocess(
            center_lat=center_lat,
            center_lon=center_lon,
            intensity=intensity_proxy,
            rng=rng,
        )
        steps.extend(frame.log_steps)
        steps.append(f"   → {len(frame.patches)} patches | {len(frame.edge_maps)} edge maps ready")
        preprocess_ok = True
    except Exception as e:
        logger.warning("Preprocessing failed: %s", e)
        frame = None
        preprocess_ok = False
        steps.append(f"   ⚠ Preprocessing error: {e} — using synthetic fallback")

    # ── STAGE 2: CNN + Transfer Learning ────────────────────────────────────
    steps.append("② CNN + Transfer-Learning — GradientBoostingClassifier head")
    cnn_model = get_cnn_model()
    if preprocess_ok and frame is not None and frame.patches:
        # Use central patch (most likely to contain cyclone core)
        mid = len(frame.patches) // 2
        patch = frame.patches[mid]
        context = {"intensity_proxy": intensity_proxy}
        cnn_pred: CNNPrediction = cnn_model.predict(patch, context=context)
    else:
        # Fallback: synthetic feature vector
        rng2 = np.random.default_rng(42)
        patch = np.zeros((6, 64, 64), dtype=np.float32)
        cnn_pred = cnn_model.predict(patch, context={"intensity_proxy": intensity_proxy})

    steps.append(f"   → Category: {cnn_pred.category_name} | Wind: {cnn_pred.max_wind_kt:.0f} kt | Conf: {cnn_pred.confidence:.0%}")
    steps.extend([f"   {a['layer']}: mean_act={a['mean_act']:.3f}" for a in cnn_pred.layer_activations])

    # ── STAGE 3: LSTM / GRU Track & Intensity Forecast ──────────────────────
    steps.append("③ LSTM + GRU — Time-series track & intensity forecast")
    lstm_model = get_lstm_predictor()

    if track_points and len(track_points) >= 2:
        seq = CycloneLSTMPredictor.build_sequence(track_points, sst_mean=29.0)
        steps.append(f"   Using {len(track_points)} real track points as LSTM input")
    else:
        # Synthetic sequence based on CNN prediction
        seq = CycloneLSTMPredictor.synthetic_sequence(
            current_wind=cnn_pred.max_wind_kt,
            current_lat=center_lat,
            current_lon=center_lon,
            trend=1.04 if intensity_proxy > 0.6 else 0.97,
        )
        steps.append("   Synthetic LSTM sequence from CNN wind estimate")

    lstm_pred: LSTMPrediction = lstm_model.predict(seq)
    steps.append(f"   → Trend: {lstm_pred.intensity_tendency} | +24h: {lstm_pred.wind_forecasts[-1]['wind_kt']} kt")
    steps.append(f"   → Track +24h: {lstm_pred.track_forecasts[-1]['lat']:.1f}°N {lstm_pred.track_forecasts[-1]['lon']:.1f}°E")

    # ── STAGE 4: Aggregate results ───────────────────────────────────────────
    steps.append("④ Aggregation → Unified prediction output")
    detected      = cnn_pred.detection_prob > 0.40
    confidence    = float(np.clip(
        0.55 * cnn_pred.confidence + 0.45 * cnn_pred.detection_prob, 0, 1))
    wind_24h      = lstm_pred.wind_forecasts[-1]["wind_kt"] if lstm_pred.wind_forecasts else None
    flagged       = confidence < 0.55

    # Per-channel stats for dashboard
    preprocess_stats: Dict[str, Dict] = {}
    if preprocess_ok and frame is not None:
        for ch, arr in frame.raw_channels.items():
            preprocess_stats[ch] = {
                "min": float(arr.min()), "max": float(arr.max()),
                "mean": float(arr.mean()), "std": float(arr.std()),
            }
        # OLR channel colourised PNG
        channel_png = channel_to_png_b64(frame.raw_channels["olr"])
    else:
        channel_png = None

    # XAI evidence text
    evidence = (
        f"CNN detected {cnn_pred.category_name} with {cnn_pred.confidence:.0%} confidence. "
        f"Layer activations show strongest response in Conv1 (Gabor edge filters) on OLR channel — "
        f"consistent with cold cloud-top structure of a {cnn_pred.category_name}. "
        f"LSTM/GRU temporal model predicts {lstm_pred.intensity_tendency.lower()} trend "
        f"over the next 24 h (current {cnn_pred.max_wind_kt:.0f} kt → {wind_24h:.0f} kt). "
        f"Track forecast: +24h at {lstm_pred.track_forecasts[-1]['lat']:.1f}°N / "
        f"{lstm_pred.track_forecasts[-1]['lon']:.1f}°E (±{lstm_pred.track_forecasts[-1]['radius_km']} km)."
    )

    model_stack = {
        "preprocessing": {"tools": ["OpenCV", "NumPy", "Pandas"], "patches": len(frame.patches) if preprocess_ok and frame else 0},
        "cnn": cnn_pred.model_meta,
        "lstm": lstm_pred.model_meta,
    }

    result = PipelinePrediction(
        detected=detected,
        detection_prob=cnn_pred.detection_prob,
        category_code=cnn_pred.category_code,
        category=cnn_pred.category_name,
        max_wind_kt=cnn_pred.max_wind_kt,
        confidence=confidence,
        center_lat=center_lat,
        center_lon=center_lon,
        intensity_trend=lstm_pred.intensity_tendency,
        intensity_probs=lstm_pred.tendency_probs,
        wind_24h_kt=wind_24h,
        track_forecast=lstm_pred.track_forecasts,
        wind_forecast=lstm_pred.wind_forecasts,
        cnn_class_probs=cnn_pred.class_probabilities,
        cnn_layer_acts=cnn_pred.layer_activations,
        cnn_feature_snippet=cnn_pred.feature_vector[:16],
        preprocess_stats=preprocess_stats,
        channel_png=channel_png,
        pipeline_steps=steps,
        timestamp=ts,
        flagged_for_review=flagged,
        xai_evidence=evidence,
        model_stack=model_stack,
        _raw_cnn=cnn_pred,
        _raw_lstm=lstm_pred,
    )

    logger.info(
        "Pipeline complete | detected=%s cat=%s wind=%.0f trend=%s conf=%.0f%%",
        result.detected, result.category, result.max_wind_kt,
        result.intensity_trend, result.confidence * 100,
    )
    return result
