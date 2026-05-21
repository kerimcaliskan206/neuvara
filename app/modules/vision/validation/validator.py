"""
Image validation system.

Validates images at the boundary (upload) and before inference.
Centralising all rejection logic here means routes stay thin and validation
rules are testable in isolation.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.modules.vision.config import VisionConfig, vision_config
from app.modules.vision.inference.service import VisionInferenceService
from app.modules.vision.inference.threshold import ConfidenceFilter

logger = logging.getLogger(__name__)

# Default class name used to mark images as relevant to the main task.
# Matches ImageFolderDataset.CLASSES = ["unrelated", "related"].
DEFAULT_RELEVANT_CLASS = "related"


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    image_info: dict | None = None

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def log(self) -> None:
        status = "PASSED" if self.passed else "FAILED"
        logger.info("Image validation: %s", status)
        for e in self.errors:
            logger.warning("  ✗ %s", e)
        for w in self.warnings:
            logger.info("  ⚠ %s", w)


class ImageValidator:
    """
    Validates images at upload time and before inference.

    Checks performed at upload:
      1. File extension is in the allowed set
      2. File size is within limit
      3. File is not corrupted (PIL can open it)
      4. Image dimensions are within bounds
      5. Image mode can be converted to RGB

    Checks at inference:
      1. Image is in RGB mode
      2. Dimensions meet minimum size

    Placeholder for future content validation:
      6. Image appears to be hantavirus-relevant (classifier gate)
    """

    def __init__(
        self,
        config: VisionConfig = vision_config,
        relevance_service: VisionInferenceService | None = None,
        relevance_threshold: float = 0.7,
        relevant_class: str = DEFAULT_RELEVANT_CLASS,
    ) -> None:
        """
        Parameters
        ----------
        config : VisionConfig
            Storage, upload, image-size settings.
        relevance_service : VisionInferenceService | None
            Trained related/unrelated gate. When provided,
            ``validate_content_relevance`` runs it and rejects uploads whose
            top class is not ``relevant_class`` or whose confidence falls
            below ``relevance_threshold``. When ``None``, the relevance
            check is a no-op that emits a single warning.
        relevance_threshold : float
            Minimum softmax confidence required to accept a relevant
            classification.
        relevant_class : str
            Class label that marks an image as on-topic. Defaults to "related".
        """
        self.config = config
        self.relevance_service = relevance_service
        self.relevant_class = relevant_class
        self._relevance_filter = ConfidenceFilter(threshold=relevance_threshold)

    # ── Upload boundary validation ────────────────────────────────────────────

    def validate_upload(
        self,
        file_bytes: bytes,
        original_filename: str,
    ) -> ValidationResult:
        result = ValidationResult(passed=True)
        upload_cfg = self.config.upload

        # 1. Extension check
        suffix = Path(original_filename).suffix.lower()
        if suffix not in upload_cfg.allowed_extensions:
            result.add_error(
                f"Unsupported file extension '{suffix}'. "
                f"Allowed: {sorted(upload_cfg.allowed_extensions)}"
            )
            return result  # can't proceed without a valid extension

        # 2. File size check
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > upload_cfg.max_file_size_mb:
            result.add_error(
                f"File too large: {size_mb:.1f} MB "
                f"(limit: {upload_cfg.max_file_size_mb} MB)"
            )
            return result

        if size_mb == 0:
            result.add_error("Empty file — 0 bytes.")
            return result

        # 3. Corruption / decodability check
        try:
            import io
            image = Image.open(io.BytesIO(file_bytes))
            image.verify()  # checks integrity without fully decoding
            # Re-open after verify (verify consumes the stream)
            image = Image.open(io.BytesIO(file_bytes))
            image.load()    # fully decode to catch truncated files
        except UnidentifiedImageError:
            result.add_error(
                "File cannot be identified as an image. "
                "It may be corrupted or not an image file."
            )
            return result
        except Exception as exc:
            result.add_error(f"Image is corrupted or truncated: {exc}")
            return result

        # 4. Format mismatch (extension vs actual format)
        actual_format = (image.format or "").lower()
        declared_format = suffix.lstrip(".")
        if declared_format == "jpg":
            declared_format = "jpeg"
        if actual_format and actual_format != declared_format:
            result.add_warning(
                f"Extension '{suffix}' does not match actual format '{image.format}'. "
                "File accepted but consider using the correct extension."
            )

        # 5. Dimension bounds
        w, h = image.size
        min_dim = upload_cfg.min_dimension_px
        max_dim = upload_cfg.max_dimension_px
        if w < min_dim or h < min_dim:
            result.add_error(
                f"Image too small: {w}×{h} px "
                f"(minimum: {min_dim}×{min_dim} px)"
            )
        if w > max_dim or h > max_dim:
            result.add_error(
                f"Image too large: {w}×{h} px "
                f"(maximum: {max_dim}×{max_dim} px)"
            )

        # 6. Mode convertibility
        try:
            image.convert("RGB")
        except Exception as exc:
            result.add_error(f"Cannot convert image to RGB: {exc}")

        if result.passed:
            result.image_info = {
                "width": w,
                "height": h,
                "mode": image.mode,
                "format": image.format,
                "size_mb": round(size_mb, 3),
            }

        result.log()
        return result

    # ── Pre-inference validation ──────────────────────────────────────────────

    def validate_for_inference(self, image: Image.Image) -> ValidationResult:
        result = ValidationResult(passed=True)
        min_dim = self.config.upload.min_dimension_px

        if image.mode != "RGB":
            try:
                image = image.convert("RGB")
                result.add_warning(
                    f"Image mode was '{image.mode}', converted to RGB."
                )
            except Exception as exc:
                result.add_error(f"Cannot convert to RGB: {exc}")
                return result

        w, h = image.size
        if w < min_dim or h < min_dim:
            result.add_error(
                f"Image too small for inference: {w}×{h} px "
                f"(minimum: {min_dim}×{min_dim} px)"
            )

        return result

    # ── Content relevance gate ───────────────────────────────────────────────

    def validate_content_relevance(self, image: Image.Image) -> ValidationResult:
        """
        Run the trained related/unrelated gate over an image.

        Rejects the upload when:
          - the gate predicts the "unrelated" class, or
          - the gate's top-class confidence falls below the configured threshold.

        Pass-through (with a warning) when no ``relevance_service`` is wired —
        callers can still rely on this method being safe to call.
        """
        result = ValidationResult(passed=True)

        if self.relevance_service is None or not self.relevance_service.is_ready:
            result.add_warning(
                "Content relevance gate is not loaded — image accepted by default. "
                "Train the related/unrelated classifier and pass it to ImageValidator "
                "to activate this check."
            )
            return result

        prediction = self.relevance_service.predict(image)
        filtered = self._relevance_filter.apply(prediction)

        result.image_info = {
            "gate_predicted_class": prediction.class_label,
            "gate_confidence": round(prediction.confidence, 4),
            "gate_threshold": self._relevance_filter.threshold,
        }

        if not filtered.accepted:
            result.add_error(
                f"Image rejected by relevance gate: low confidence "
                f"({prediction.confidence:.2f} < {self._relevance_filter.threshold:.2f}). "
                f"Top class was '{prediction.class_label}'."
            )
            return result

        if prediction.class_label != self.relevant_class:
            result.add_error(
                f"Image classified as '{prediction.class_label}' — "
                f"expected '{self.relevant_class}' to proceed with analysis."
            )
            return result

        logger.debug(
            "Content relevance gate accepted image (class=%s, confidence=%.4f)",
            prediction.class_label, prediction.confidence,
        )
        return result
