"""
Data Source Abstraction Layer
Abstract interface + Synthetic + IBTrACS + stubs for MOSDAC/GPM
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataFrame:
    """A single multi-channel gridded observation frame."""
    timestamp: str
    grid_lat: np.ndarray        # (H,) latitude axis
    grid_lon: np.ndarray        # (W,) longitude axis
    channels: dict              # name → (H, W) array
    metadata: dict = field(default_factory=dict)

    def as_tensor_channels(self, channel_order: list[str]) -> np.ndarray:
        """Stack channels into (C, H, W) array, filling missing with zeros."""
        arrays = []
        for ch in channel_order:
            if ch in self.channels:
                arrays.append(self.channels[ch])
            else:
                arrays.append(np.zeros_like(next(iter(self.channels.values()))))
        return np.stack(arrays, axis=0)  # (C, H, W)


class DataSource(ABC):
    """Abstract data source interface."""

    @abstractmethod
    def get_latest_frame(self) -> DataFrame:
        """Return the most recent observation frame."""
        ...

    @abstractmethod
    def get_time_series(self, start: str, end: str) -> list[DataFrame]:
        """Return a list of frames between start and end (ISO timestamps)."""
        ...

    @abstractmethod
    def get_channels(self) -> list[str]:
        """Return available channel names."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
