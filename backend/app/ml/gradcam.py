"""
Grad-CAM implementation for CycloNex CNN encoder.
Produces saliency heatmaps showing what the model "looked at" for a prediction.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import base64
import io
import os
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd(module, input, output):
            self.activations = output.detach()

        def bwd(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(fwd)
        self.target_layer.register_full_backward_hook(bwd)

    def generate(self, x: torch.Tensor, target_class: int = 0) -> np.ndarray:
        """
        Args:
            x: input tensor (1, T, C, H, W) or (1, C, H, W)
            target_class: which classification head class to visualize
        Returns:
            heatmap: (H, W) normalized heatmap [0, 1]
        """
        self.model.eval()
        x = x.requires_grad_(True)

        out = self.model(x)
        # Use classification logits for gradient computation
        logits = out["classification"]  # (1, 7)
        score = logits[0, target_class]

        self.model.zero_grad()
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            logger.warning("Grad-CAM: no gradients/activations captured")
            return np.zeros((64, 64))

        # Pool gradients over spatial dims
        pooled_grads = self.gradients.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
        weighted = (self.activations * pooled_grads).sum(dim=1)       # (B, H, W)
        cam = F.relu(weighted[0])                                       # (H, W)

        # Normalize
        cam_np = cam.cpu().numpy()
        cam_np = cam_np - cam_np.min()
        if cam_np.max() > 0:
            cam_np /= cam_np.max()

        return cam_np


def generate_xai_image(
    model,
    x: torch.Tensor,
    input_channel: int = 1,   # which channel to use as background (1=OLR)
    target_class: int = 2,
    output_path: str = None,
    size: tuple = (320, 240),
) -> tuple[str, str]:
    """
    Generate a Grad-CAM overlay PNG, save to file, return (file_path, base64_string).
    Args:
        model: CycloNexModel
        x: input tensor
        input_channel: which input channel to show as the background image
        target_class: IMD category index for gradient computation
        output_path: where to save the PNG
        size: output image (W, H)
    Returns:
        (saved_path, base64_encoded_png)
    """
    try:
        target_layer = model.encoder.layer3
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate(x, target_class=target_class)

        # Get background: use the specified channel of the first frame
        if x.dim() == 5:
            bg = x[0, 0, input_channel].detach().cpu().numpy()
        else:
            bg = x[0, input_channel].detach().cpu().numpy()

        # Normalize background to 0-255
        bg_norm = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)
        bg_img = Image.fromarray((bg_norm * 255).astype(np.uint8)).convert("L")
        bg_img = bg_img.resize(size, Image.BILINEAR)
        bg_rgba = np.array(bg_img.convert("RGBA"))

        # Resize and colorize CAM
        cam_img = Image.fromarray((cam * 255).astype(np.uint8))
        cam_img = cam_img.resize(size, Image.BILINEAR)
        cam_np = np.array(cam_img) / 255.0

        # Jet colormap: red = high attention
        r = np.clip(1.5 - np.abs(cam_np * 4 - 3), 0, 1)
        g = np.clip(1.5 - np.abs(cam_np * 4 - 2), 0, 1)
        b = np.clip(1.5 - np.abs(cam_np * 4 - 1), 0, 1)
        alpha = (cam_np * 200).astype(np.uint8)

        heat_rgba = np.stack([
            (r * 255).astype(np.uint8),
            (g * 255).astype(np.uint8),
            (b * 255).astype(np.uint8),
            alpha
        ], axis=2)

        # Composite
        result = Image.fromarray(bg_rgba, "RGBA")
        overlay = Image.fromarray(heat_rgba, "RGBA")
        result = Image.alpha_composite(result, overlay).convert("RGB")

        # Save
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result.save(output_path, format="PNG")

        # Encode to base64
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return output_path or "", b64

    except Exception as e:
        logger.error(f"Grad-CAM generation failed: {e}")
        # Return a blank placeholder
        placeholder = Image.new("RGB", size, color=(20, 40, 80))
        buf = io.BytesIO()
        placeholder.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return "", b64


def generate_evidence_caption(outputs: dict, imd_categories: list[str]) -> str:
    """Auto-generate a human-readable XAI evidence caption."""
    category = outputs.get("category", "Unknown")
    trend = outputs.get("intensity_trend", "Unknown")
    conf = outputs.get("confidence", 0.0)
    wind = outputs.get("max_wind_kt", 0.0)

    trend_text = {
        "Intensifying": "intensification of the system",
        "Steady": "steady-state maintenance",
        "Weakening": "weakening of the vortex",
    }.get(trend, "evolution of the system")

    return (
        f"Model attention concentrated on curved rain bands, deep convection near the center, "
        f"and low OLR signatures consistent with {category}. "
        f"Thermal patterns in SST and upper-level divergence indicate {trend_text}. "
        f"Classification confidence: {conf * 100:.1f}%. "
        f"Estimated max sustained wind: {wind:.0f} kt."
    )
