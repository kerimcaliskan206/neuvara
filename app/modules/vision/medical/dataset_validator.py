"""
Medical dataset validator — Phase 7.

Per-image file-level checks used during dataset construction:
  - supported extension
  - file integrity (decodeable without corruption)
  - minimum and recommended dimension thresholds
  - grayscale mode detection
  - SHA-256 hash computation (feeds leakage detector)

Does NOT perform semantic or ML inference — pure filesystem operations.
Safe to run on any machine without GPU or model checkpoints.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
)
_MIN_DIMENSION: int = 64
_RECOMMENDED_MIN_DIMENSION: int = 224


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass
class ImageValidationResult:
    """Outcome of validating a single image file."""

    path: Path
    valid: bool
    sha256: str | None = None
    width: int | None = None
    height: int | None = None
    mode: str | None = None        # PIL mode: "RGB", "L", "RGBA", etc.
    is_grayscale: bool = False     # True when mode is "L" or "LA"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ── Hash utility ──────────────────────────────────────────────────────────────


def compute_sha256(path: Path, chunk_size: int = 65536) -> str:
    """Stream-hash a file and return its SHA-256 hex digest."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Single-image validation ───────────────────────────────────────────────────


def validate_image(path: Path) -> ImageValidationResult:
    """
    Run all file-level checks on one image.

    Returns an ImageValidationResult with errors/warnings populated.
    Checks: extension → PIL integrity → dimensions → hash.
    """
    result = ImageValidationResult(path=path, valid=False)
    errors: list[str] = []
    warnings: list[str] = []

    if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        result.errors = [f"Unsupported extension: {path.suffix!r}"]
        return result

    # PIL integrity + metadata
    try:
        from PIL import Image, UnidentifiedImageError  # lazy import — no hard dep at module level

        try:
            img = Image.open(path)
            img.verify()           # file-level integrity (does not decode pixels)
        except UnidentifiedImageError:
            result.errors = ["Unidentifiable image format — file may be corrupt."]
            return result
        except Exception as exc:
            result.errors = [f"PIL verify failed: {exc}"]
            return result

        img = Image.open(path)  # re-open after verify (verify() closes the handle)
        result.width, result.height = img.size
        result.mode = img.mode
        result.is_grayscale = img.mode in ("L", "LA")

    except ImportError:
        warnings.append("Pillow not installed — skipping image integrity check.")

    if result.width is not None and result.height is not None:
        w, h = result.width, result.height
        if w < _MIN_DIMENSION or h < _MIN_DIMENSION:
            errors.append(
                f"Image too small: {w}×{h} px "
                f"(minimum {_MIN_DIMENSION} px each side)."
            )
        elif w < _RECOMMENDED_MIN_DIMENSION or h < _RECOMMENDED_MIN_DIMENSION:
            warnings.append(
                f"Below recommended size: {w}×{h} px "
                f"(recommend ≥ {_RECOMMENDED_MIN_DIMENSION} px)."
            )

    # SHA-256 for leakage detection
    try:
        result.sha256 = compute_sha256(path)
    except OSError as exc:
        warnings.append(f"Could not compute SHA-256: {exc}")

    result.valid = len(errors) == 0
    result.errors = errors
    result.warnings = warnings
    return result


# ── Directory-level validation ────────────────────────────────────────────────


def validate_dataset_dir(
    directory: Path,
    *,
    recursive: bool = True,
) -> list[ImageValidationResult]:
    """
    Validate all images found under `directory`.

    Non-image files (by extension) are silently skipped.
    Returns one ImageValidationResult per image file discovered.
    """
    pattern = "**/*" if recursive else "*"
    paths = sorted(
        p for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
    )
    logger.info("Validating %d images in %s", len(paths), directory)
    results = []
    for p in paths:
        r = validate_image(p)
        if not r.valid:
            logger.debug("INVALID %s: %s", p.name, r.errors)
        results.append(r)
    return results


def build_hash_map(results: list[ImageValidationResult]) -> dict[Path, str]:
    """Extract {path: sha256} from validated results, skipping failed images."""
    return {r.path: r.sha256 for r in results if r.valid and r.sha256}
