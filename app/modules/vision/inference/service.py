"""
Vision inference service.

Manages the lifecycle of a vision model — loading, reloading, and routing
inference calls. Designed to be instantiated once at application startup.

Two load paths:
  load_from_checkpoint(path, architecture, class_names)
      Training-style .pt checkpoint (dict with model_state_dict).
  load_from_store(version=None)
      Versioned model from VisionModelStore (production path).

Prediction reliability tracking
---------------------------------
call service.reliability_report() to get:
  - total predictions served
  - low-confidence prediction rate
  - mean/min/max confidence observed
"""
from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

from app.modules.vision.config import VisionConfig, vision_config
from app.modules.vision.inference.predictor import VisionPrediction, VisionPredictor
from app.modules.vision.models.registry import VisionModelRegistry
from app.modules.vision.persistence.model_store import VisionModelStore
from app.modules.vision.preprocessing.pipeline import ImagePreprocessingPipeline
from app.modules.vision.training.trainer import VisionTrainer  # _resolve_device only

logger = logging.getLogger(__name__)


# ── Reliability tracker ───────────────────────────────────────────────────────


@dataclass
class ReliabilityStats:
    """Running statistics for prediction confidence monitoring."""

    n_total: int = 0
    n_low_confidence: int = 0    # below low_confidence_threshold
    n_extreme_confidence: int = 0  # ≥ 0.99 — suspicious overconfidence
    confidence_sum: float = 0.0
    confidence_min: float = float("inf")
    confidence_max: float = float("-inf")
    low_confidence_threshold: float = 0.6
    extreme_confidence_threshold: float = 0.99
    # Coarse 10-bucket histogram (0.0-0.1, ..., 0.9-1.0)
    histogram: list[int] = field(default_factory=lambda: [0] * 10)
    # Per-class prediction counts — surfaces a runaway class bias.
    per_class_predictions: dict[str, int] = field(default_factory=dict)
    # Per-source-prefix confidence accumulator — detects systematic
    # over-prediction on uploads from the same source/folder/filename pattern.
    per_source_count: dict[str, int] = field(default_factory=dict)
    per_source_extreme: dict[str, int] = field(default_factory=dict)

    def record(
        self,
        confidence: float,
        class_label: Optional[str] = None,
        source: Optional[str] = None,
    ) -> None:
        self.n_total += 1
        self.confidence_sum += confidence
        self.confidence_min = min(self.confidence_min, confidence)
        self.confidence_max = max(self.confidence_max, confidence)
        if confidence < self.low_confidence_threshold:
            self.n_low_confidence += 1
        is_extreme = confidence >= self.extreme_confidence_threshold
        if is_extreme:
            self.n_extreme_confidence += 1
        bucket = min(9, int(confidence * 10))
        self.histogram[bucket] += 1
        if class_label:
            self.per_class_predictions[class_label] = (
                self.per_class_predictions.get(class_label, 0) + 1
            )
        if source:
            # Use a short prefix (first 16 chars before extension) — enough to
            # cluster uploads from the same batch / source / filename scheme
            # without retaining the full original filename.
            tag = source.rsplit(".", 1)[0][:16] or "?"
            self.per_source_count[tag] = self.per_source_count.get(tag, 0) + 1
            if is_extreme:
                self.per_source_extreme[tag] = (
                    self.per_source_extreme.get(tag, 0) + 1
                )

    @property
    def confidence_mean(self) -> float:
        return self.confidence_sum / max(self.n_total, 1)

    @property
    def low_confidence_rate(self) -> float:
        return self.n_low_confidence / max(self.n_total, 1)

    @property
    def extreme_confidence_rate(self) -> float:
        return self.n_extreme_confidence / max(self.n_total, 1)

    def as_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "confidence_mean": round(self.confidence_mean, 4),
            "confidence_min": round(self.confidence_min, 4) if math.isfinite(self.confidence_min) else None,
            "confidence_max": round(self.confidence_max, 4) if math.isfinite(self.confidence_max) else None,
            "low_confidence_rate": round(self.low_confidence_rate, 4),
            "low_confidence_threshold": self.low_confidence_threshold,
            "n_low_confidence": self.n_low_confidence,
            "extreme_confidence_rate": round(self.extreme_confidence_rate, 4),
            "extreme_confidence_threshold": self.extreme_confidence_threshold,
            "n_extreme_confidence": self.n_extreme_confidence,
            "histogram": list(self.histogram),
            "per_class_predictions": dict(self.per_class_predictions),
            "per_source_count": dict(self.per_source_count),
            "per_source_extreme": dict(self.per_source_extreme),
        }

    def reset(self) -> None:
        self.n_total = 0
        self.n_low_confidence = 0
        self.n_extreme_confidence = 0
        self.confidence_sum = 0.0
        self.confidence_min = float("inf")
        self.confidence_max = float("-inf")
        self.histogram = [0] * 10
        self.per_class_predictions = {}
        self.per_source_count = {}
        self.per_source_extreme = {}


# ── Service ───────────────────────────────────────────────────────────────────


class VisionInferenceService:
    """
    Wraps a loaded vision model with lifecycle management and reliability tracking.

    Parameters
    ----------
    config : VisionConfig
        Storage + image-size + device configuration.
    """

    # Confidence ceiling applied when segmentation quality is weak.
    _LOW_TRUST_CAP: float = 0.75

    def __init__(self, config: VisionConfig = vision_config) -> None:
        self.config = config
        self._device = VisionTrainer._resolve_device(config.resolve_device())
        self._pipeline = ImagePreprocessingPipeline(config)
        self._store = VisionModelStore(config.storage.models_dir)

        self._model: Optional[torch.nn.Module] = None
        self._architecture: Optional[str] = None
        self._class_names: list[str] = []
        self._version: Optional[str] = None
        self._metadata: dict = {}
        self._calibration_temperature: float = 1.0

        self._stats = ReliabilityStats()

        # Phase-3: mandatory lung segmentation — every raw image is cropped to
        # its lung ROI before reaching the classifier.
        from app.modules.vision.segmentation import LungSegmentationPipeline
        self._seg_pipeline = LungSegmentationPipeline(
            padding_frac=0.07, save_debug=False
        )

    # ── State ────────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def architecture(self) -> Optional[str]:
        return self._architecture

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    @property
    def version(self) -> Optional[str]:
        return self._version

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def model(self) -> Optional[torch.nn.Module]:
        return self._model

    @property
    def calibration_temperature(self) -> float:
        return self._calibration_temperature

    # ── Loading from VisionModelStore (production path) ──────────────────────

    def load_from_store(self, version: Optional[str] = None) -> bool:
        """
        Load a model saved by VisionModelStore.save().

        Returns True on success. Errors are logged, never raised — inspect
        .is_ready afterward.
        """
        try:
            metadata = self._store.load_metadata(version=version)
            state_dict = self._store.load_weights(version=version)
        except FileNotFoundError as exc:
            logger.warning("VisionInferenceService: load_from_store failed — %s", exc)
            return False
        except Exception:
            logger.exception("VisionInferenceService: unexpected error loading from store")
            return False

        architecture = metadata.get("architecture")
        class_names = metadata.get("class_names") or []
        if not architecture or not class_names:
            logger.error(
                "VisionInferenceService: metadata missing architecture or class_names: %s",
                metadata,
            )
            return False

        try:
            model = VisionModelRegistry.build(
                architecture=architecture,
                num_classes=len(class_names),
                pretrained=False,
                freeze=False,
            )
            model.load_state_dict(state_dict)
        except Exception:
            logger.exception(
                "VisionInferenceService: failed to build %s with %d classes",
                architecture, len(class_names),
            )
            return False

        model.to(self._device).eval()

        self._model = model
        self._architecture = architecture
        self._class_names = class_names
        self._version = metadata.get("version", version)
        self._metadata = metadata
        self._calibration_temperature = float(
            metadata.get("calibration_temperature", 1.0)
        )
        self._stats.reset()

        logger.info(
            "VisionInferenceService: loaded %s @ %s | classes=%s | T=%.4f | device=%s",
            architecture, self._version, class_names,
            self._calibration_temperature, self._device,
        )
        return True

    # ── Loading v6 calibrated checkpoint ────────────────────────────────────

    def load_v6_calibrated(
        self,
        checkpoint_path: str | Path,
        temperature_config_path: str | Path | None = None,
    ) -> bool:
        """
        Load a v6 calibrated checkpoint (stage_b_calibrated.pt format).

        Expected structure inside the .pt file:
          { "model_state_dict": <state_dict>, "v6_meta": { "architecture": ...,
            "classes": [...], "calibration_temperature": float, ... } }

        Optionally reads the authoritative temperature from a JSON sidecar
        (temperature_config.json) alongside the checkpoint.
        """
        import json as _json

        checkpoint_path = Path(checkpoint_path)
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception:
            logger.exception("load_v6_calibrated: failed to load %s", checkpoint_path)
            return False

        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt or "v6_meta" not in ckpt:
            logger.error(
                "load_v6_calibrated: %s is not a v6 checkpoint "
                "(expected keys: model_state_dict, v6_meta)",
                checkpoint_path,
            )
            return False

        meta = ckpt["v6_meta"]
        architecture = meta.get("architecture")
        class_names  = meta.get("classes")
        if not architecture or not class_names:
            logger.error(
                "load_v6_calibrated: v6_meta missing 'architecture' or 'classes': %s",
                list(meta.keys()),
            )
            return False

        # Temperature: sidecar JSON wins if present, else fall back to v6_meta
        temperature = float(meta.get("calibration_temperature", 1.0))
        if temperature_config_path is not None:
            try:
                tc = _json.loads(Path(temperature_config_path).read_text(encoding="utf-8"))
                temperature = float(tc.get("temperature", temperature))
            except Exception:
                logger.warning(
                    "load_v6_calibrated: could not read temperature_config %s — "
                    "using v6_meta value %.4f",
                    temperature_config_path, temperature,
                )

        try:
            model = VisionModelRegistry.build(
                architecture=architecture,
                num_classes=len(class_names),
                pretrained=False,
                freeze=False,
            )
            model.load_state_dict(ckpt["model_state_dict"])
        except Exception:
            logger.exception(
                "load_v6_calibrated: failed to build %s with %d classes",
                architecture, len(class_names),
            )
            return False

        model.to(self._device).eval()

        sub_stage   = meta.get("sub_stage", "stage_b_calibrated")
        self._model = model
        self._architecture = architecture
        self._class_names  = list(class_names)
        self._version      = f"v6_calibrated:{sub_stage}@T{temperature:.4f}"
        self._metadata     = {
            "architecture":            architecture,
            "class_names":             list(class_names),
            "num_classes":             len(class_names),
            "calibration_temperature": temperature,
            "version":                 self._version,
            "image_size":              [224, 224],
            "v6_phase":                meta.get("v6_phase"),
            "sub_stage":               sub_stage,
            "best_val_f1":             meta.get("best_val_f1"),
            "calibration_phase":       meta.get("calibration_phase"),
        }
        self._calibration_temperature = temperature
        self._stats.reset()

        logger.info(
            "VisionInferenceService: loaded v6 calibrated %s | T=%.4f | classes=%s | device=%s",
            architecture, temperature, class_names, self._device,
        )
        return True

    # ── Loading from a training-style checkpoint ─────────────────────────────

    def load_from_checkpoint(
        self,
        checkpoint_path: str | Path,
        architecture: str,
        class_names: list[str],
    ) -> bool:
        """Load a ModelCheckpoint-style .pt file (dev / testing workflow)."""
        try:
            predictor = VisionPredictor(
                checkpoint_path=checkpoint_path,
                architecture=architecture,
                num_classes=len(class_names),
                class_names=class_names,
                device=str(self._device),
            )
            predictor.load()
        except Exception:
            logger.exception(
                "VisionInferenceService: failed to load checkpoint %s",
                checkpoint_path,
            )
            return False

        self._model = predictor._model
        self._architecture = architecture
        self._class_names = class_names
        self._version = f"checkpoint:{Path(checkpoint_path).name}"
        self._metadata = predictor.checkpoint_metadata
        self._calibration_temperature = 1.0
        self._stats.reset()

        logger.info(
            "VisionInferenceService: loaded checkpoint %s as %s",
            Path(checkpoint_path).name, architecture,
        )
        return True

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(
        self,
        image: Image.Image | Path | str,
        *,
        source: Optional[str] = None,
    ) -> VisionPrediction:
        """Run inference on a single image with reliability tracking."""
        self._require_ready()

        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        t0 = time.perf_counter()

        # ── Phase-3: mandatory lung segmentation ──────────────────────────────
        # Every raw image is cropped to its lung ROI before reaching the model.
        # The centre-crop TTA that previously compensated for border shortcuts
        # has been removed — segmentation makes it redundant.
        roi_image, seg_tel = self._seg_pipeline.process(image)
        seg_quality = seg_tel.quality
        low_trust = seg_quality == "fallback"

        if low_trust:
            logger.info(
                "VisionInferenceService: segmentation fallback (lung_area=%.3f) "
                "— low_trust=True, confidence will be capped at %.2f",
                seg_tel.lung_area_pct, self._LOW_TRUST_CAP,
            )

        tensor = self._pipeline.preprocess_for_inference(roi_image).to(self._device)

        # Deployment-side temperature override (escape hatch; not primary calibration).
        effective_temperature = self._calibration_temperature
        temp_override = os.environ.get("MEDICAL_TEMPERATURE")
        if temp_override:
            try:
                effective_temperature = float(temp_override)
            except ValueError:
                pass

        with torch.no_grad():
            logits = self._model(tensor)
            cal_logits = logits / effective_temperature
            probs = F.softmax(cal_logits, dim=1).squeeze(0).cpu()

        class_idx = int(probs.argmax().item())
        confidence = float(probs[class_idx].item())

        # ── Segmentation failure confidence ceiling ───────────────────────────
        if low_trust and confidence > self._LOW_TRUST_CAP:
            confidence = self._LOW_TRUST_CAP

        # Deployment-side hard cap (escape hatch for calibration mismatches).
        cap_str = os.environ.get("MEDICAL_CONFIDENCE_CAP")
        if cap_str:
            try:
                cap = float(cap_str)
                if 0.0 < cap < 1.0 and confidence > cap:
                    excess = confidence - cap
                    n_other = max(1, probs.numel() - 1)
                    probs = probs.clone()
                    probs[class_idx] = cap
                    bump = excess / n_other
                    for j in range(probs.numel()):
                        if j != class_idx:
                            probs[j] = probs[j] + bump
                    confidence = float(probs[class_idx].item())
            except ValueError:
                pass

        elapsed_ms = (time.perf_counter() - t0) * 1000

        class_label_for_stats = (
            self._class_names[class_idx]
            if self._class_names and class_idx < len(self._class_names)
            else str(class_idx)
        )
        self._stats.record(
            confidence, class_label=class_label_for_stats, source=source,
        )

        # Surface persistent over-confidence as a runtime warning.  Once we have
        # at least 20 predictions and >75% of them sit in the ≥0.99 bucket the
        # calibration is clearly mismatched to deployment data.
        if (
            self._stats.n_total == 20
            or (self._stats.n_total > 20 and self._stats.n_total % 50 == 0)
        ):
            if self._stats.extreme_confidence_rate > 0.75:
                logger.warning(
                    "VisionInferenceService: %.0f%% of predictions ≥0.99 "
                    "(n=%d, T=%.4f) — calibration likely mismatched. "
                    "Set MEDICAL_TEMPERATURE>1.0 or MEDICAL_CONFIDENCE_CAP to "
                    "compensate, or recalibrate on held-out healthy data.",
                    self._stats.extreme_confidence_rate * 100,
                    self._stats.n_total,
                    self._calibration_temperature,
                )
            # Class-bias alert: if pneumonia_xray dominates >70% of predictions
            # the model is reproducing the 2.69× training imbalance at inference.
            pneu_n = self._stats.per_class_predictions.get("pneumonia_xray", 0)
            if pneu_n / max(self._stats.n_total, 1) > 0.70:
                logger.warning(
                    "VisionInferenceService: pneumonia_xray predicted on %d/%d "
                    "uploads (%.0f%%) — class bias reproducing training "
                    "imbalance. Per-class counts: %s",
                    pneu_n, self._stats.n_total,
                    100 * pneu_n / max(self._stats.n_total, 1),
                    self._stats.per_class_predictions,
                )
            # Source-correlated overconfidence: any single source prefix that
            # accounts for ≥5 uploads and produces ≥0.99 confidence on >80% of
            # them is a strong signal of dataset-source shortcut.
            for tag, count in self._stats.per_source_count.items():
                if count >= 5:
                    extreme = self._stats.per_source_extreme.get(tag, 0)
                    if extreme / count > 0.80:
                        logger.warning(
                            "VisionInferenceService: source-prefix '%s' has "
                            "%d/%d (%.0f%%) ≥0.99 predictions — likely a "
                            "source-correlated shortcut.",
                            tag, extreme, count, 100 * extreme / count,
                        )

        class_label = class_label_for_stats

        return VisionPrediction(
            class_index=class_idx,
            class_label=class_label,
            confidence=confidence,
            probabilities=probs.tolist(),
            inference_ms=elapsed_ms,
            class_names=self._class_names,
            segmentation_quality=seg_quality,
            low_trust=low_trust,
            segmentation_telemetry=seg_tel.as_dict(),
        )

    def predict_batch(
        self, images: list[Image.Image | Path | str]
    ) -> list[VisionPrediction]:
        """Run inference on a list of images sequentially."""
        self._require_ready()
        return [self.predict(img) for img in images]

    def predict_batch_fast(
        self, tensors: list[torch.Tensor]
    ) -> list[VisionPrediction]:
        """
        Batch inference on pre-processed tensors.

        Stacks tensors into a single forward pass — more efficient than
        sequential predict() calls when tensors are already preprocessed.

        Parameters
        ----------
        tensors : list of (C, H, W) or (1, C, H, W) tensors
        """
        self._require_ready()
        if not tensors:
            return []

        t0 = time.perf_counter()
        batch = torch.stack(
            [t.squeeze(0) if t.dim() == 4 else t for t in tensors]
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(batch)
            cal_logits = logits / self._calibration_temperature
            probs_batch = F.softmax(cal_logits, dim=1).cpu()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_image_ms = elapsed_ms / len(tensors)

        results = []
        for probs in probs_batch:
            class_idx = int(probs.argmax().item())
            confidence = float(probs[class_idx].item())
            self._stats.record(confidence)
            class_label = (
                self._class_names[class_idx]
                if self._class_names and class_idx < len(self._class_names)
                else str(class_idx)
            )
            results.append(VisionPrediction(
                class_index=class_idx,
                class_label=class_label,
                confidence=confidence,
                probabilities=probs.tolist(),
                inference_ms=per_image_ms,
                class_names=self._class_names,
            ))

        return results

    # ── Info & reliability ────────────────────────────────────────────────────

    def model_info(self) -> dict:
        if not self.is_ready:
            return {"is_ready": False}
        return {
            "is_ready": True,
            "architecture": self._architecture,
            "model_version": self._version,
            "class_names": self.class_names,
            "image_size": self._metadata.get("image_size"),
            "metrics": self._metadata.get("metrics", {}),
            "calibration_temperature": self._calibration_temperature,
        }

    def reliability_report(self) -> dict:
        """Return running confidence statistics since last model load."""
        return {
            "model_version": self._version,
            "architecture": self._architecture,
            **self._stats.as_dict(),
        }

    def reset_reliability_stats(self) -> None:
        """Reset the reliability tracker (e.g., at start of a new session)."""
        self._stats.reset()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _require_ready(self) -> None:
        if not self.is_ready:
            raise RuntimeError(
                "VisionInferenceService has no model loaded. "
                "Train one (scripts/train_vision.py) and call .load_from_store()."
            )
