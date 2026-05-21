"""
Image upload handler.

Responsibilities:
  - Accept raw bytes from any source (HTTP multipart, filesystem, test fixtures)
  - Validate before persisting
  - Generate safe, collision-free filenames
  - Store under a structured uploads directory
  - Return a rich UploadResult for callers to act on
"""
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from app.modules.vision.config import VisionConfig, vision_config
from app.modules.vision.utils.io import load_image_from_bytes, save_image
from app.modules.vision.validation.validator import ImageValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class ImageMetadata:
    original_filename: str
    safe_filename: str
    width: int
    height: int
    mode: str
    format: str
    file_size_bytes: int
    uploaded_at: str
    upload_path: Path

    def to_dict(self) -> dict:
        return {
            "original_filename": self.original_filename,
            "safe_filename": self.safe_filename,
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
            "format": self.format,
            "file_size_bytes": self.file_size_bytes,
            "uploaded_at": self.uploaded_at,
            "upload_path": str(self.upload_path),
        }


@dataclass
class UploadResult:
    success: bool
    original_filename: str
    validation: ValidationResult
    safe_filename: str | None = None
    upload_path: Path | None = None
    metadata: ImageMetadata | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "original_filename": self.original_filename,
            "safe_filename": self.safe_filename,
            "upload_path": str(self.upload_path) if self.upload_path else None,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "errors": self.validation.errors,
            "warnings": self.validation.warnings,
        }


class ImageUploadHandler:
    """
    Handles the full image upload lifecycle:
    validate → generate safe name → persist → return result.

    All uploads are saved as JPEG to normalize formats and reduce
    storage variance. The original filename is preserved in metadata only.
    """

    def __init__(self, config: VisionConfig = vision_config) -> None:
        self.config = config
        self.validator = ImageValidator(config)
        self._ensure_dirs()

    # ── Public API ────────────────────────────────────────────────────────────

    def handle(self, file_bytes: bytes, original_filename: str) -> UploadResult:
        """
        Validate, save, and return an UploadResult.
        On validation failure the file is NOT written to disk.
        """
        logger.info(
            "Upload received: '%s' (%d bytes)", original_filename, len(file_bytes)
        )

        validation = self.validator.validate_upload(file_bytes, original_filename)
        if not validation.passed:
            logger.warning(
                "Upload rejected: '%s' — %s",
                original_filename, "; ".join(validation.errors),
            )
            return UploadResult(
                success=False,
                original_filename=original_filename,
                validation=validation,
                error="; ".join(validation.errors),
            )

        safe_name = self._safe_filename(original_filename)
        save_path = self.config.storage.uploads_dir / safe_name

        try:
            image = load_image_from_bytes(file_bytes)
            image_rgb = image.convert("RGB")
            save_image(image_rgb, save_path, quality=95)
        except Exception as exc:
            logger.exception("Failed to persist upload '%s'", original_filename)
            return UploadResult(
                success=False,
                original_filename=original_filename,
                validation=validation,
                error=f"Storage error: {exc}",
            )

        metadata = ImageMetadata(
            original_filename=original_filename,
            safe_filename=safe_name,
            width=image.width,
            height=image.height,
            mode=image.mode,
            format=image.format or "unknown",
            file_size_bytes=len(file_bytes),
            uploaded_at=datetime.now(timezone.utc).isoformat(),
            upload_path=save_path,
        )

        logger.info(
            "Upload saved: '%s' → '%s'  (%dx%d)",
            original_filename, safe_name, image.width, image.height,
        )

        return UploadResult(
            success=True,
            original_filename=original_filename,
            validation=validation,
            safe_filename=safe_name,
            upload_path=save_path,
            metadata=metadata,
        )

    # ── Filename generation ───────────────────────────────────────────────────

    @staticmethod
    def safe_filename(original_filename: str) -> str:
        """
        Generate a UUID-based filename that preserves the original extension.
        Prevents path traversal, special characters, and collisions.
        """
        suffix = Path(original_filename).suffix.lower()
        if suffix in (".jpg",):
            suffix = ".jpg"
        return f"{uuid.uuid4().hex}{suffix}"

    # alias used internally
    _safe_filename = safe_filename

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _ensure_dirs(self) -> None:
        self.config.storage.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.config.storage.processed_dir.mkdir(parents=True, exist_ok=True)
