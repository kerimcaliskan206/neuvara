"""
Heatmap rendering utilities.

Converts a raw Grad-CAM float tensor into a visually interpretable overlay
on the original image. Uses OpenCV for colormap application and NumPy for
blending.

Output options
--------------
- overlay: colored heatmap blended onto the original image (for display)
- heatmap_only: pure colormap, no original image
- side_by_side: original image next to the overlay (for reports)

Color conventions
-----------------
- Hot (red/yellow) = high activation → region strongly influenced prediction
- Cool (blue/green) = low activation → region had little influence
"""
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_COLORMAPS: dict[str, int] = {
    "jet":      cv2.COLORMAP_JET,
    "hot":      cv2.COLORMAP_HOT,
    "viridis":  cv2.COLORMAP_VIRIDIS,
    "plasma":   cv2.COLORMAP_PLASMA,
    "inferno":  cv2.COLORMAP_INFERNO,
    "turbo":    cv2.COLORMAP_TURBO,
}


class HeatmapRenderer:
    """
    Renders Grad-CAM heatmaps as PIL Images.

    Parameters
    ----------
    colormap : str
        One of: "jet", "hot", "viridis", "plasma", "inferno", "turbo".
    alpha : float
        Blend weight for the heatmap layer (0 = invisible, 1 = opaque).
        Original image weight is (1 - alpha).
    """

    def __init__(self, colormap: str = "jet", alpha: float = 0.4) -> None:
        if colormap not in _COLORMAPS:
            raise ValueError(
                f"Unknown colormap '{colormap}'. "
                f"Available: {list(_COLORMAPS)}"
            )
        self.colormap = _COLORMAPS[colormap]
        self.alpha = alpha

    # ── Public API ────────────────────────────────────────────────────────────

    def overlay(
        self,
        original: Image.Image,
        cam: torch.Tensor,
    ) -> Image.Image:
        """
        Blend the heatmap onto the original image.

        Parameters
        ----------
        original : PIL.Image
            Source image (any size, will be kept at original resolution).
        cam : torch.Tensor
            Grad-CAM output, shape (H, W), values in [0, 1].
            Must match original's spatial size or will be resized.

        Returns
        -------
        PIL.Image (RGB)
        """
        orig_np = self._pil_to_rgb_array(original)
        heatmap_np = self._cam_to_colormap(cam, orig_np.shape[:2])
        blended = cv2.addWeighted(
            orig_np, 1 - self.alpha,
            heatmap_np, self.alpha,
            0,
        )
        return Image.fromarray(blended)

    def heatmap_only(
        self,
        cam: torch.Tensor,
        size: tuple[int, int] | None = None,
    ) -> Image.Image:
        """
        Return only the colormap visualization (no original image).

        Parameters
        ----------
        cam : torch.Tensor
            Grad-CAM output, shape (H, W), values in [0, 1].
        size : (width, height) | None
            Target output size. If None, uses cam's native resolution.
        """
        h = size[1] if size else cam.shape[0]
        w = size[0] if size else cam.shape[1]
        heatmap_np = self._cam_to_colormap(cam, (h, w))
        return Image.fromarray(heatmap_np)

    def side_by_side(
        self,
        original: Image.Image,
        cam: torch.Tensor,
    ) -> Image.Image:
        """
        Return original image and overlay placed side by side.

        Useful for generating explainability reports.
        """
        left = self._pil_to_rgb_array(original)
        h, w = left.shape[:2]
        heatmap_np = self._cam_to_colormap(cam, (h, w))
        blended = cv2.addWeighted(left, 1 - self.alpha, heatmap_np, self.alpha, 0)
        combined = np.concatenate([left, blended], axis=1)
        return Image.fromarray(combined)

    def save(
        self,
        image: Image.Image,
        path: Path | str,
        quality: int = 90,
    ) -> Path:
        """Save a rendered heatmap to disk as JPEG."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="JPEG", quality=quality)
        logger.debug("HeatmapRenderer: saved heatmap to %s", path)
        return path

    # ── Anatomical mask overlays ─────────────────────────────────────────────

    def mask_overlay(
        self,
        original: Image.Image,
        mask: torch.Tensor,
        color: tuple[int, int, int] = (0, 255, 0),
        alpha: float | None = None,
    ) -> Image.Image:
        """
        Blend a single-channel mask onto the original image with a solid color.

        Parameters
        ----------
        mask : (H, W) tensor, values treated as soft membership in [0, 1].
        color : RGB tuple for the mask's painted color.
        alpha : optional per-call override for the global ``self.alpha``.
        """
        a = self.alpha if alpha is None else alpha
        orig_np = self._pil_to_rgb_array(original)
        mask_np = self._mask_to_alpha(mask, orig_np.shape[:2])
        color_layer = np.zeros_like(orig_np)
        color_layer[..., 0] = color[0]
        color_layer[..., 1] = color[1]
        color_layer[..., 2] = color[2]
        m3 = (mask_np[..., None].astype(np.float32) / 255.0) * a
        blended = (orig_np.astype(np.float32) * (1.0 - m3)
                   + color_layer.astype(np.float32) * m3)
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    def zone_overlay(
        self,
        original: Image.Image,
        central: torch.Tensor,
        peripheral: torch.Tensor,
        diaphragm: torch.Tensor | None = None,
        *,
        central_color:    tuple[int, int, int] = (0, 220, 0),    # green
        peripheral_color: tuple[int, int, int] = (220, 60, 60),  # red
        diaphragm_color:  tuple[int, int, int] = (60, 130, 220), # blue
        alpha: float | None = None,
    ) -> Image.Image:
        """
        Color-coded anatomical zone overlay:
        green = central (where we want focus), red = peripheral (shortcut zone),
        blue = diaphragm-adjacent (preserved for lower-lobe pathology).
        """
        a = self.alpha if alpha is None else alpha
        orig_np = self._pil_to_rgb_array(original)
        h, w = orig_np.shape[:2]

        zones = [(central, central_color), (peripheral, peripheral_color)]
        if diaphragm is not None:
            zones.append((diaphragm, diaphragm_color))

        out = orig_np.astype(np.float32)
        for mask, color in zones:
            mask_np = self._mask_to_alpha(mask, (h, w)).astype(np.float32) / 255.0
            m3 = mask_np[..., None] * a
            layer = np.zeros_like(orig_np, dtype=np.float32)
            layer[..., 0], layer[..., 1], layer[..., 2] = color
            out = out * (1.0 - m3) + layer * m3
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    def cam_zone_composite(
        self,
        original: Image.Image,
        cam: torch.Tensor,
        zones: dict[str, torch.Tensor],
    ) -> Image.Image:
        """
        2×2 composite panel for explainability reports:

          ┌─────────────────┬─────────────────┐
          │   lung mask     │   central zone  │
          ├─────────────────┼─────────────────┤
          │ peripheral zone │ GradCAM ∩ lung  │
          └─────────────────┴─────────────────┘

        ``zones`` must contain at least: ``lung_mask``, ``central``,
        ``peripheral``. ``diaphragm`` is included alongside central in the
        central panel when present.
        """
        for k in ("lung_mask", "central", "peripheral"):
            if k not in zones:
                raise ValueError(f"cam_zone_composite: missing zone '{k}'")

        lung_mask  = zones["lung_mask"]
        central    = zones["central"]
        peripheral = zones["peripheral"]
        diaphragm  = zones.get("diaphragm")

        panel_lung = self.mask_overlay(original, lung_mask, color=(255, 255, 255))
        panel_central = self.zone_overlay(
            original, central=central,
            peripheral=torch.zeros_like(central),
            diaphragm=diaphragm,
        )
        panel_peripheral = self.mask_overlay(original, peripheral, color=(220, 60, 60))

        # GradCAM ∩ lung — masks out non-lung activations entirely.
        cam_np = cam.numpy() if isinstance(cam, torch.Tensor) else cam
        mask_np = (
            lung_mask.numpy() if isinstance(lung_mask, torch.Tensor) else lung_mask
        ).astype(np.float32)
        if mask_np.shape != cam_np.shape:
            mask_np = cv2.resize(
                mask_np, (cam_np.shape[1], cam_np.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        cam_intersect = torch.from_numpy(cam_np * mask_np)
        panel_overlap = self.overlay(original, cam_intersect)

        top = np.concatenate(
            [np.array(panel_lung), np.array(panel_central)], axis=1
        )
        bot = np.concatenate(
            [np.array(panel_peripheral), np.array(panel_overlap)], axis=1
        )
        grid = np.concatenate([top, bot], axis=0)
        return Image.fromarray(grid)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _cam_to_colormap(
        self,
        cam: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> np.ndarray:
        """Convert a (H, W) float tensor → (H, W, 3) uint8 RGB colormap."""
        h, w = target_hw
        cam_np = cam.numpy() if isinstance(cam, torch.Tensor) else cam
        cam_uint8 = (cam_np * 255).clip(0, 255).astype(np.uint8)

        if cam_uint8.shape != (h, w):
            cam_uint8 = cv2.resize(cam_uint8, (w, h), interpolation=cv2.INTER_LINEAR)

        colored = cv2.applyColorMap(cam_uint8, self.colormap)  # BGR
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)         # → RGB

    @staticmethod
    def _mask_to_alpha(
        mask: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> np.ndarray:
        """Convert a (H, W) soft mask in [0, 1] → uint8 alpha at target_hw."""
        h, w = target_hw
        m_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
        m_uint8 = (m_np * 255).clip(0, 255).astype(np.uint8)
        if m_uint8.shape != (h, w):
            m_uint8 = cv2.resize(m_uint8, (w, h), interpolation=cv2.INTER_LINEAR)
        return m_uint8

    @staticmethod
    def _pil_to_rgb_array(image: Image.Image) -> np.ndarray:
        return np.array(image.convert("RGB"), dtype=np.uint8)
