"""Image I/O utilities — thin wrappers that centralise PIL/CV2 usage."""
import base64
import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def load_image(path: Path | str) -> Image.Image:
    """Load an image from disk and convert to RGB."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image = Image.open(path).convert("RGB")
    logger.debug("Loaded image: %s  size=%s", path.name, image.size)
    return image


def load_image_from_bytes(data: bytes) -> Image.Image:
    """Decode raw bytes into a PIL Image (RGB)."""
    return Image.open(io.BytesIO(data)).convert("RGB")


def save_image(image: Image.Image, path: Path | str, quality: int = 95) -> Path:
    """Save a PIL Image to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = path.suffix.lower().lstrip(".")
    fmt = "jpeg" if fmt in ("jpg", "jpeg") else fmt
    image.save(path, format=fmt.upper(), quality=quality)
    logger.debug("Saved image → %s", path)
    return path


def image_to_tensor(image: Image.Image):
    """Convert PIL Image (H×W×C) → float32 torch.Tensor (C×H×W) in [0, 1]."""
    import torch
    arr = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def tensor_to_image(tensor) -> Image.Image:
    """Convert float32 torch.Tensor (C×H×W, [0,1]) → PIL Image."""
    arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim == 3:
        arr = arr.transpose(1, 2, 0)
    return Image.fromarray((arr * 255).astype(np.uint8))


def image_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    """Encode a PIL Image as a base64 string (for embedding in JSON)."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_image_info(image: Image.Image) -> dict:
    """Return a metadata dict for a PIL Image."""
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image.format or "unknown",
        "n_channels": len(image.getbands()),
    }
