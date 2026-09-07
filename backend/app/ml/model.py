"""
CycloNex Multi-Task CNN + ConvLSTM Model
=========================================
Architecture:
  - CNN Spatial Encoder (ResNet-style): extracts spatial features from multi-channel input
  - ConvLSTM: models temporal evolution across a sequence of frames
  - Shared storm representation → 7 task heads:
      a. Detection (sigmoid): cyclone present/absent
      b. Localization (regression): center lat/lon
      c. Classification (softmax, 7 IMD categories)
      d. Intensity trend (3-class): Intensifying / Steady / Weakening
      e. Track forecast (regression): lat/lon at +6/+12/+18/+24h
      f. Wind regression: max sustained wind now + 24h
      g. Uncertainty (scalar): confidence 0-1

Input: (B, T, C, H, W) — batch, time steps, channels, height, width
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ── ConvLSTM Cell ────────────────────────────────────────────────────────────
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels, 4 * hidden_channels,
            kernel_size, padding=pad, bias=True
        )

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def init_state(self, batch_size: int, h: int, w: int, device):
        return (
            torch.zeros(batch_size, self.hidden_channels, h, w, device=device),
            torch.zeros(batch_size, self.hidden_channels, h, w, device=device),
        )


# ── CNN Spatial Encoder ──────────────────────────────────────────────────────
class CNNEncoder(nn.Module):
    """Lightweight ResNet-style encoder for spatial feature extraction."""
    def __init__(self, in_channels: int = 6, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.layer1 = self._res_block(c, c * 2)
        self.layer2 = self._res_block(c * 2, c * 4)
        self.layer3 = self._res_block(c * 4, c * 4)
        self.out_channels = c * 4

    def _res_block(self, in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )

    def forward(self, x):           # x: (B, C_in, H, W)
        x = self.stem(x)            # → (B, c, H/2, W/2)
        x = self.layer1(x)          # → (B, 2c, H/4, W/4)
        x = self.layer2(x)          # → (B, 4c, H/8, W/8)
        x = self.layer3(x)          # → (B, 4c, H/16, W/16)
        return x


# ── CycloNex Multi-Task Model ────────────────────────────────────────────────
class CycloNexModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        base_channels: int = 32,
        lstm_hidden: int = 64,
        num_classes: int = 7,
        num_track_steps: int = 4,  # +6, +12, +18, +24h
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_track_steps = num_track_steps

        # Spatial encoder
        self.encoder = CNNEncoder(in_channels, base_channels)
        enc_c = self.encoder.out_channels  # 4 * base_channels

        # ConvLSTM for temporal modeling
        self.conv_lstm = ConvLSTMCell(enc_c, lstm_hidden)
        self.lstm_hidden = lstm_hidden

        # Global average pooling → flat feature vector
        self.gap = nn.AdaptiveAvgPool2d(1)
        flat_dim = lstm_hidden

        # ── Task heads ──
        self.head_detection = nn.Sequential(
            nn.Linear(flat_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        self.head_localization = nn.Sequential(
            nn.Linear(flat_dim, 64), nn.ReLU(),
            nn.Linear(64, 2)   # lat, lon delta from grid center
        )
        self.head_classification = nn.Sequential(
            nn.Linear(flat_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self.head_intensity_trend = nn.Sequential(
            nn.Linear(flat_dim, 64), nn.ReLU(),
            nn.Linear(64, 3)   # Intensifying | Steady | Weakening
        )
        self.head_track = nn.Sequential(
            nn.Linear(flat_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_track_steps * 2)   # (lat, lon) × steps
        )
        self.head_wind = nn.Sequential(
            nn.Linear(flat_dim, 64), nn.ReLU(),
            nn.Linear(64, 2)   # current wind_kt, +24h wind_kt
        )
        self.head_uncertainty = nn.Sequential(
            nn.Linear(flat_dim, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Hook storage for Grad-CAM
        self.gradcam_activations = None
        self.gradcam_gradients = None

    def _register_gradcam_hooks(self):
        def fwd_hook(module, input, output):
            self.gradcam_activations = output.detach()
        def bwd_hook(module, grad_input, grad_output):
            self.gradcam_gradients = grad_output[0].detach()
        self.encoder.layer3.register_forward_hook(fwd_hook)
        self.encoder.layer3.register_full_backward_hook(bwd_hook)

    def forward(self, x, use_mc_dropout: bool = False):
        """
        Args:
            x: (B, T, C, H, W)  or  (B, C, H, W) for single-frame
        Returns:
            dict of task head outputs
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)   # treat as T=1

        B, T, C, H, W = x.shape
        device = x.device

        # Initialize ConvLSTM state
        # After encoder: spatial dims become H', W'
        dummy = self.encoder(x[:, 0])       # (B, enc_c, H', W')
        _, _, Hp, Wp = dummy.shape
        state = self.conv_lstm.init_state(B, Hp, Wp, device)

        feat_seq = []
        for t in range(T):
            spatial = self.encoder(x[:, t])   # (B, enc_c, H', W')
            h, c = self.conv_lstm(spatial, state)
            state = (h, c)
            feat_seq.append(h)

        # Use last hidden state
        feat_map = feat_seq[-1]  # (B, lstm_hidden, H', W')
        feat = self.gap(feat_map).view(B, -1)  # (B, lstm_hidden)

        # Enable dropout during MC-dropout inference for uncertainty
        if use_mc_dropout:
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.train()

        return {
            "detection_logit":   self.head_detection(feat),           # (B, 1)
            "localization":      self.head_localization(feat),         # (B, 2)
            "classification":    self.head_classification(feat),       # (B, num_classes)
            "intensity_trend":   self.head_intensity_trend(feat),      # (B, 3)
            "track_forecast":    self.head_track(feat).view(B, self.num_track_steps, 2),  # (B, 4, 2)
            "wind":              F.relu(self.head_wind(feat)),         # (B, 2)  — non-negative
            "uncertainty_logit": self.head_uncertainty(feat),          # (B, 1)
            "feat_map":          feat_map,                             # for Grad-CAM
        }

    def mc_dropout_uncertainty(self, x, n_samples: int = 10) -> float:
        """Run N forward passes with dropout enabled, return std as uncertainty."""
        self.eval()
        probs = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(x, use_mc_dropout=True)
                p = torch.sigmoid(out["detection_logit"]).item()
                probs.append(p)
        return float(np.std(probs)) if len(probs) > 1 else 0.0


# ── Factory / checkpoint loading ─────────────────────────────────────────────
def build_model(in_channels: int = 6) -> CycloNexModel:
    return CycloNexModel(in_channels=in_channels)


def load_model(path: str, device: str = "cpu") -> Optional[CycloNexModel]:
    import os
    if not os.path.exists(path):
        return None
    model = build_model()
    try:
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception as e:
        import logging
        logging.warning(f"Could not load model weights from {path}: {e}")
        return None


# Fix missing import for mc_dropout_uncertainty
import numpy as np
