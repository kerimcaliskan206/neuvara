"""
Vision model predictor.

Loads a trained checkpoint and runs inference on single images or batches.
Does NOT expose HTTP endpoints — that is the service layer's responsibility.

Class names are loaded from the checkpoint itself (embedded by ModelCheckpoint)
or supplied explicitly. This eliminates the previous hardcoded IDX_TO_CLASS
dependency that silently produced wrong labels for 3-class models.

Usage
-----
    predictor = VisionPredictor.from_checkpoint(
        checkpoint_path=Path("checkpoints/best_0.9231_epoch012.pt"),
        device="auto",
    )
    result = predictor.predict_single(image)   # PIL Image or Path
    results = predictor.predict_batch(images)  # list of images
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

from app.modules.vision.models.registry import VisionModelRegistry
from app.modules.vision.preprocessing.pipeline import ImagePreprocessingPipeline
from app.modules.vision.training.trainer import VisionTrainer  # _resolve_device only

logger = logging.getLogger(__name__)


# ── Prediction result ─────────────────────────────────────────────────────────


class VisionPrediction:
    """
    Single-image prediction result — self-contained with class names.

    Phase-3 fields (backward-compatible, all default to safe values):
      segmentation_quality : str   — "good" | "single_lung" | "fallback" | "not_applied"
      low_trust            : bool  — True when segmentation quality was weak;
                                     confidence has been capped to LOW_TRUST_CAP.
      segmentation_telemetry : dict — full SegmentationTelemetry payload (may be {}).
    """

    __slots__ = (
        "class_index", "class_label", "confidence",
        "probabilities", "class_names", "inference_ms",
        "segmentation_quality", "low_trust", "segmentation_telemetry",
    )

    def __init__(
        self,
        class_index: int,
        class_label: str,
        confidence: float,
        probabilities: list[float],
        inference_ms: float,
        class_names: Optional[list[str]] = None,
        segmentation_quality: str = "not_applied",
        low_trust: bool = False,
        segmentation_telemetry: Optional[dict] = None,
    ) -> None:
        self.class_index = class_index
        self.class_label = class_label
        self.confidence = confidence
        self.probabilities = probabilities
        self.class_names = class_names or []
        self.inference_ms = inference_ms
        self.segmentation_quality = segmentation_quality
        self.low_trust = low_trust
        self.segmentation_telemetry = segmentation_telemetry or {}

    def as_dict(self) -> dict:
        if self.class_names:
            prob_map = {
                name: round(prob, 4)
                for name, prob in zip(self.class_names, self.probabilities)
            }
        else:
            prob_map = {str(i): round(p, 4) for i, p in enumerate(self.probabilities)}

        return {
            "class_index": self.class_index,
            "class_label": self.class_label,
            "confidence": round(self.confidence, 4),
            "probabilities": prob_map,
            "inference_ms": round(self.inference_ms, 2),
            "segmentation_quality": self.segmentation_quality,
            "low_trust": self.low_trust,
        }


# ── Predictor ─────────────────────────────────────────────────────────────────


class VisionPredictor:
    """
    Loads a checkpoint and runs forward passes.

    Prefer VisionPredictor.from_checkpoint() when loading a ModelCheckpoint
    file — it reads class_names and architecture automatically.

    Parameters
    ----------
    checkpoint_path : Path
        Path to a .pt checkpoint saved by ModelCheckpoint or VisionModelStore.
    architecture : str
        Model architecture name (overridden by checkpoint metadata if present).
    num_classes : int
        Output classes (overridden by checkpoint metadata if present).
    class_names : list[str] | None
        Class label names in index order (overridden by checkpoint metadata).
    device : str
        Compute device ("auto", "cpu", "cuda", "mps").
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        architecture: str = "efficientnet_b0",
        num_classes: int = 2,
        class_names: Optional[list[str]] = None,
        device: str = "auto",
        use_segmentation: bool = True,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.architecture = architecture
        self.num_classes = num_classes
        self._class_names: list[str] = class_names or []
        self.device = VisionTrainer._resolve_device(device)
        self._pipeline = ImagePreprocessingPipeline()
        self._model: Optional[torch.nn.Module] = None
        self._checkpoint_meta: dict = {}
        self.use_segmentation = use_segmentation
        if use_segmentation:
            from app.modules.vision.segmentation import LungSegmentationPipeline
            self._seg_pipeline: Optional[LungSegmentationPipeline] = (
                LungSegmentationPipeline(padding_frac=0.07, save_debug=False)
            )
        else:
            self._seg_pipeline = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        device: str = "auto",
    ) -> "VisionPredictor":
        """
        Load a self-contained ModelCheckpoint file.

        Reads class_names, architecture, and num_classes from the checkpoint
        dict — no extra arguments required.
        """
        ckpt = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        class_names: list[str] = ckpt.get("class_names", [])
        architecture: str = ckpt.get("architecture", "efficientnet_b0")
        num_classes: int = ckpt.get("num_classes", len(class_names) or 2)

        if not class_names:
            logger.warning(
                "VisionPredictor.from_checkpoint: checkpoint has no class_names — "
                "probabilities will use index keys in as_dict()"
            )

        predictor = cls(
            checkpoint_path=checkpoint_path,
            architecture=architecture,
            num_classes=num_classes,
            class_names=class_names,
            device=device,
        )
        predictor.load()
        return predictor

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load model weights from checkpoint into memory."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )

        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=True
        )

        # Resolve metadata from checkpoint (overrides constructor defaults)
        if "class_names" in checkpoint and checkpoint["class_names"]:
            self._class_names = checkpoint["class_names"]
        if "architecture" in checkpoint and checkpoint["architecture"]:
            self.architecture = checkpoint["architecture"]
        if "num_classes" in checkpoint and checkpoint["num_classes"]:
            self.num_classes = checkpoint["num_classes"]

        model = VisionModelRegistry.build(
            architecture=self.architecture,
            num_classes=self.num_classes,
            pretrained=False,
            freeze=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        self._model = model
        self._checkpoint_meta = {
            k: v for k, v in checkpoint.items() if k != "model_state_dict"
        }

        logger.info(
            "VisionPredictor: loaded %s | epoch=%s | score=%.4f | classes=%s",
            self.architecture,
            checkpoint.get("epoch", "?"),
            checkpoint.get("score", 0.0),
            self._class_names,
        )

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    # ── Inference ─────────────────────────────────────────────────────────────

    # Confidence ceiling when segmentation quality is weak.
    _LOW_TRUST_CAP: float = 0.75

    def predict_single(self, image: Image.Image | Path | str) -> VisionPrediction:
        """
        Run inference on a single image.

        When use_segmentation=True (default), the raw image is first passed
        through LungSegmentationPipeline to extract the lung ROI; only the
        ROI reaches the classifier.  If segmentation quality is weak, the
        returned confidence is capped at _LOW_TRUST_CAP and low_trust=True.
        """
        self._require_ready()

        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        t0 = time.perf_counter()

        seg_quality = "not_applied"
        low_trust = False
        seg_tel_dict: dict = {}

        if self._seg_pipeline is not None:
            roi_image, seg_tel = self._seg_pipeline.process(image)
            seg_quality = seg_tel.quality
            seg_tel_dict = seg_tel.as_dict()
            low_trust = seg_quality == "fallback"
            preprocessed = roi_image
        else:
            preprocessed = image

        tensor = self._pipeline.preprocess_for_inference(preprocessed).to(self.device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        class_idx = int(probs.argmax().item())
        confidence = float(probs[class_idx].item())

        if low_trust:
            confidence = min(confidence, self._LOW_TRUST_CAP)
            logger.info(
                "VisionPredictor: low_trust cap applied (quality=%s) — "
                "confidence capped at %.2f",
                seg_quality, self._LOW_TRUST_CAP,
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        class_label = (
            self._class_names[class_idx]
            if self._class_names and class_idx < len(self._class_names)
            else str(class_idx)
        )

        return VisionPrediction(
            class_index=class_idx,
            class_label=class_label,
            confidence=confidence,
            probabilities=probs.tolist(),
            inference_ms=elapsed_ms,
            class_names=self._class_names,
            segmentation_quality=seg_quality,
            low_trust=low_trust,
            segmentation_telemetry=seg_tel_dict,
        )

    def predict_batch(
        self,
        images: list[Image.Image | Path | str],
    ) -> list[VisionPrediction]:
        """Run inference on a list of images (sequential)."""
        self._require_ready()
        return [self.predict_single(img) for img in images]

    def predict_tensor(self, tensor: torch.Tensor) -> VisionPrediction:
        """Run inference on a pre-processed tensor of shape (1, C, H, W)."""
        self._require_ready()

        t0 = time.perf_counter()
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        class_idx = int(probs.argmax().item())
        confidence = float(probs[class_idx].item())
        elapsed_ms = (time.perf_counter() - t0) * 1000

        class_label = (
            self._class_names[class_idx]
            if self._class_names and class_idx < len(self._class_names)
            else str(class_idx)
        )

        return VisionPrediction(
            class_index=class_idx,
            class_label=class_label,
            confidence=confidence,
            probabilities=probs.tolist(),
            inference_ms=elapsed_ms,
            class_names=self._class_names,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _require_ready(self) -> None:
        if not self.is_ready:
            raise RuntimeError(
                "VisionPredictor is not loaded. Call .load() or use .from_checkpoint()."
            )

    @property
    def checkpoint_metadata(self) -> dict:
        return self._checkpoint_meta.copy()
