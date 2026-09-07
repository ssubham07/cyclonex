"""
CycloNex – Satellite Image Preprocessing Pipeline
Tools: OpenCV, NumPy, Pandas
Purpose: Simulate preprocessing of multi-channel satellite imagery
         (SST, OLR, QPE, CMV, WVW, UTH) for cyclone detection/classification.
"""

import cv2
import numpy as np
import pandas as pd
import logging
import base64
import io
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
CHANNELS = ["sst", "olr", "qpe", "cmv", "wvw", "uth"]
GRID_H, GRID_W = 121, 161          # 0–30°N × 60–100°E at 0.25° resolution
PATCH_SIZE      = 64               # CNN input patch (pixels)
NORM_STATS = {
    # channel: (mean, std)  — based on NIO climatology
    "sst": (28.5,  3.2),
    "olr": (220.0, 40.0),
    "qpe": (3.0,   8.0),
    "cmv": (5.0,   4.0),
    "wvw": (45.0,  15.0),
    "uth": (230.0, 20.0),
}


@dataclass
class PreprocessedFrame:
    """One fully preprocessed satellite frame ready for ML inference."""
    raw_channels:       Dict[str, np.ndarray]   # H×W float32 raw
    normalized:         Dict[str, np.ndarray]   # H×W float32 normalized
    patches:            List[np.ndarray]         # list of C×64×64 patches
    edge_maps:          Dict[str, np.ndarray]   # Canny edge maps
    gradient_maps:      Dict[str, np.ndarray]   # Sobel magnitude maps
    timestamp:          str
    stats_df:           pd.DataFrame             # per-channel statistics table
    log_steps:          List[str] = field(default_factory=list)


# ── Step 1: Simulate multi-channel satellite array ────────────────────────────
def simulate_satellite_channels(
    center_lat: float = 15.0,
    center_lon: float = 82.0,
    intensity: float = 0.8,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """
    Synthesise a realistic multi-channel NIO satellite scene.
    Used when live MOSDAC/GPM data is unavailable.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    lats = np.linspace(0, 30, GRID_H)
    lons = np.linspace(60, 100, GRID_W)
    lon_g, lat_g = np.meshgrid(lons, lats)

    # Gaussian cyclone vortex centred at (center_lat, center_lon)
    r = np.sqrt((lat_g - center_lat) ** 2 + (lon_g - center_lon) ** 2)
    vortex = intensity * np.exp(-r ** 2 / (2 * 3.0 ** 2))  # sigma≈3°

    # Spiral band structure
    theta = np.arctan2(lat_g - center_lat, lon_g - center_lon)
    spiral = 0.35 * np.exp(-r / 5) * np.abs(np.sin(theta + r / 2))

    channels: Dict[str, np.ndarray] = {}

    # SST: warm core (high near centre)
    channels["sst"] = (
        28.0 + 3.5 * vortex - 0.8 * np.clip(r, 0, 8) / 8
        + 0.3 * rng.standard_normal((GRID_H, GRID_W))
    ).astype(np.float32)

    # OLR: low (cold cloud tops) near centre
    channels["olr"] = (
        240.0 - 60.0 * vortex - 20.0 * spiral
        + 5.0 * rng.standard_normal((GRID_H, GRID_W))
    ).astype(np.float32)

    # QPE: high rain near centre & bands
    channels["qpe"] = np.clip(
        20.0 * vortex + 12.0 * spiral
        + 1.5 * rng.standard_normal((GRID_H, GRID_W)),
        0, None
    ).astype(np.float32)

    # CMV: cloud-motion vector magnitude
    channels["cmv"] = np.clip(
        8.0 * vortex + 3.0 * spiral
        + 0.5 * rng.standard_normal((GRID_H, GRID_W)),
        0, None
    ).astype(np.float32)

    # WVW: water-vapour wind
    channels["wvw"] = (
        50.0 - 25.0 * vortex + 2.0 * rng.standard_normal((GRID_H, GRID_W))
    ).astype(np.float32)

    # UTH: upper-tropospheric humidity
    channels["uth"] = (
        220.0 - 30.0 * vortex - 10.0 * spiral
        + 3.0 * rng.standard_normal((GRID_H, GRID_W))
    ).astype(np.float32)

    return channels


# ── Step 2: Normalisation ─────────────────────────────────────────────────────
def normalize_channels(raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Z-score normalise each channel using pre-computed climatological stats."""
    norm: Dict[str, np.ndarray] = {}
    for ch, arr in raw.items():
        mu, sigma = NORM_STATS.get(ch, (arr.mean(), arr.std() + 1e-6))
        norm[ch] = ((arr - mu) / sigma).astype(np.float32)
    return norm


# ── Step 3: OpenCV – edge detection & gradient maps ───────────────────────────
def compute_cv_features(
    raw: Dict[str, np.ndarray]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Compute Canny edge maps and Sobel gradient magnitudes with OpenCV.
    These highlight spiral bands and the cyclone eye for the CNN.
    """
    edge_maps:    Dict[str, np.ndarray] = {}
    gradient_maps: Dict[str, np.ndarray] = {}

    for ch, arr in raw.items():
        # Scale to uint8 for OpenCV
        lo, hi = arr.min(), arr.max()
        img_u8 = ((arr - lo) / (hi - lo + 1e-9) * 255).astype(np.uint8)

        # Gaussian blur before edge detection (reduces noise)
        blurred = cv2.GaussianBlur(img_u8, (5, 5), 1.5)

        # Canny edges
        edge_maps[ch] = cv2.Canny(blurred, threshold1=30, threshold2=80)

        # Sobel gradient magnitude
        sx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_maps[ch] = np.sqrt(sx ** 2 + sy ** 2).astype(np.float32)

    return edge_maps, gradient_maps


# ── Step 4: Patch extraction (for CNN) ───────────────────────────────────────
def extract_patches(
    norm: Dict[str, np.ndarray],
    patch_size: int = PATCH_SIZE,
    stride: int = 32,
) -> List[np.ndarray]:
    """
    Extract overlapping C×patch_size×patch_size patches.
    Returns list of [6, 64, 64] arrays for CNN input.
    """
    C = len(CHANNELS)
    stack = np.stack([norm[ch] for ch in CHANNELS], axis=0)   # C × H × W
    H, W = GRID_H, GRID_W
    patches = []
    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            patch = stack[:, r:r + patch_size, c:c + patch_size]
            patches.append(patch.astype(np.float32))
    return patches


# ── Step 5: Per-channel statistics (Pandas) ──────────────────────────────────
def compute_statistics(raw: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Build a Pandas DataFrame of per-channel statistics for display/logging."""
    rows = []
    for ch, arr in raw.items():
        rows.append({
            "Channel": ch.upper(),
            "Min":     float(arr.min()),
            "Max":     float(arr.max()),
            "Mean":    float(arr.mean()),
            "Std":     float(arr.std()),
            "P25":     float(np.percentile(arr, 25)),
            "P75":     float(np.percentile(arr, 75)),
        })
    return pd.DataFrame(rows).set_index("Channel")


# ── Step 6: Channel image → base64 PNG for the API ───────────────────────────
def channel_to_png_b64(arr: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> str:
    """Encode a single-channel float array as a colourised PNG in base64."""
    lo, hi = arr.min(), arr.max()
    u8 = ((arr - lo) / (hi - lo + 1e-9) * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(u8, colormap)
    ok, buf = cv2.imencode(".png", coloured)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode()


# ── Master pipeline ───────────────────────────────────────────────────────────
def preprocess(
    center_lat: float = 15.0,
    center_lon: float = 82.0,
    intensity: float = 0.8,
    rng: Optional[np.random.Generator] = None,
) -> PreprocessedFrame:
    """
    Full preprocessing pipeline:
      1. Synthesise (or load) raw satellite channels
      2. Z-score normalise
      3. OpenCV edge + gradient maps
      4. Extract CNN patches
      5. Build stats DataFrame
    Returns a PreprocessedFrame with everything needed for ML inference.
    """
    log: List[str] = []

    log.append("STEP 1: Synthesising satellite channels (SST, OLR, QPE, CMV, WVW, UTH)")
    raw = simulate_satellite_channels(center_lat, center_lon, intensity, rng)

    log.append(f"STEP 2: Z-score normalisation using NIO climatology stats")
    norm = normalize_channels(raw)

    log.append("STEP 3: OpenCV – Canny edge detection + Sobel gradient maps")
    edge_maps, gradient_maps = compute_cv_features(raw)

    log.append(f"STEP 4: Extracting overlapping {PATCH_SIZE}×{PATCH_SIZE} patches (stride=32)")
    patches = extract_patches(norm)
    log.append(f"        → {len(patches)} patches extracted")

    log.append("STEP 5: Computing per-channel statistics with Pandas")
    stats_df = compute_statistics(raw)
    log.append(f"\n{stats_df.round(2).to_string()}")

    import datetime
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    frame = PreprocessedFrame(
        raw_channels=raw,
        normalized=norm,
        patches=patches,
        edge_maps=edge_maps,
        gradient_maps=gradient_maps,
        timestamp=ts,
        stats_df=stats_df,
        log_steps=log,
    )

    logger.info("Preprocessing complete | patches=%d ts=%s", len(patches), ts)
    return frame
