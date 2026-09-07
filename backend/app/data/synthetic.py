"""
SyntheticDataSource — Generates physically-plausible gridded cyclone fields
over the North Indian Ocean (0-30°N, 60-100°E) for development/demo use.

Channels produced (INSAT-3D/3DR proxies):
  sst   — Sea Surface Temperature (°C)
  olr   — Outgoing Longwave Radiation (W/m²)
  qpe   — Quantitative Precipitation Estimate (mm/hr)
  cmv   — Cloud Motion Vector speed (m/s)
  wvw   — Water Vapor Windowing channel (K)
  uth   — Upper Tropospheric Humidity (%)
"""
from __future__ import annotations
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional
from app.data.base import DataSource, DataFrame
import random


# NIO grid: 0-30°N, 60-100°E at 0.25° resolution
_LAT = np.linspace(0, 30, 121)   # (121,)
_LON = np.linspace(60, 100, 161)  # (161,)
_LON_G, _LAT_G = np.meshgrid(_LON, _LAT)   # (121, 161)

CHANNELS = ["sst", "olr", "qpe", "cmv", "wvw", "uth"]


def _gaussian_vortex(
    center_lat: float,
    center_lon: float,
    intensity: float = 1.0,
    radius_deg: float = 4.0,
) -> np.ndarray:
    """Smooth 2D Gaussian vortex pattern at given center, returns (H, W)."""
    r2 = (_LAT_G - center_lat) ** 2 + (_LON_G - center_lon) ** 2
    return intensity * np.exp(-r2 / (2 * radius_deg ** 2))


def _spiral_bands(
    center_lat: float,
    center_lon: float,
    n_bands: int = 4,
    intensity: float = 0.6,
) -> np.ndarray:
    """Simulate spiral rain bands as a sum of rotated Gaussians."""
    result = np.zeros_like(_LAT_G)
    for i in range(n_bands):
        angle = 2 * np.pi * i / n_bands
        bl = center_lat + 5 * np.sin(angle)
        bln = center_lon + 5 * np.cos(angle)
        result += _gaussian_vortex(bl, bln, intensity * 0.5, radius_deg=2.0)
    return result


def generate_frame(
    center_lat: float = 12.0,
    center_lon: float = 85.0,
    max_wind_kt: float = 60.0,
    timestamp: Optional[datetime] = None,
    noise_level: float = 0.1,
) -> DataFrame:
    """Generate a single synthetic multi-channel observation frame."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    norm_intensity = min(max_wind_kt / 120.0, 1.0)
    vortex = _gaussian_vortex(center_lat, center_lon, norm_intensity, radius_deg=3.5)
    spiral = _spiral_bands(center_lat, center_lon, intensity=norm_intensity * 0.8)

    rng = np.random.default_rng(seed=int(timestamp.timestamp()) % (2**31))

    sst = 28.0 - 2.0 * vortex + rng.normal(0, noise_level, vortex.shape)
    olr = 200.0 - 100.0 * vortex + rng.normal(0, 5, vortex.shape)
    qpe = 50.0 * (vortex + spiral) + rng.normal(0, noise_level * 5, vortex.shape)
    cmv = 15.0 * (vortex + 0.5 * spiral) + rng.normal(0, noise_level, vortex.shape)
    wvw = 240.0 - 40.0 * vortex + rng.normal(0, 2, vortex.shape)
    uth = 60.0 + 35.0 * (vortex + 0.5 * spiral) + rng.normal(0, noise_level * 3, vortex.shape)

    # Clip to physical ranges
    sst = np.clip(sst, 20, 35)
    olr = np.clip(olr, 100, 320)
    qpe = np.clip(qpe, 0, 200)
    cmv = np.clip(cmv, 0, 60)
    wvw = np.clip(wvw, 200, 280)
    uth = np.clip(uth, 0, 100)

    return DataFrame(
        timestamp=timestamp.isoformat(),
        grid_lat=_LAT,
        grid_lon=_LON,
        channels={"sst": sst, "olr": olr, "qpe": qpe, "cmv": cmv, "wvw": wvw, "uth": uth},
        metadata={
            "center_lat": center_lat,
            "center_lon": center_lon,
            "max_wind_kt": max_wind_kt,
            "source": "synthetic",
        },
    )


class SyntheticDataSource(DataSource):
    """Generates physically-plausible synthetic cyclone frames for dev/demo."""

    def __init__(self):
        self._center_lat = 12.0
        self._center_lon = 85.0
        self._max_wind = 65.0

    @property
    def name(self) -> str:
        return "synthetic"

    def get_channels(self) -> list[str]:
        return CHANNELS

    def get_latest_frame(self) -> DataFrame:
        return generate_frame(
            self._center_lat, self._center_lon, self._max_wind,
            timestamp=datetime.now(timezone.utc)
        )

    def get_time_series(self, start: str, end: str) -> list[DataFrame]:
        """Generate a sequence of frames along a synthetic track."""
        from datetime import datetime
        t_start = datetime.fromisoformat(start.replace("Z", "+00:00"))
        t_end = datetime.fromisoformat(end.replace("Z", "+00:00"))
        step = timedelta(hours=6)
        frames = []
        t = t_start
        lat, lon, wind = self._center_lat, self._center_lon, self._max_wind
        while t <= t_end:
            lat += random.uniform(-0.5, 0.5)
            lon += random.uniform(-0.3, 0.7)
            wind = max(20, min(130, wind + random.uniform(-5, 5)))
            frames.append(generate_frame(lat, lon, wind, timestamp=t))
            t += step
        return frames
