#!/usr/bin/env python3
"""
Medical dataset normalization pipeline — Phase 9.

Reads manually-downloaded raw medical images from data/medical_raw/<source_id>/,
validates and normalizes each image, and writes to data/medical_ready/<v6_class>/.

What it does
------------
  1. Discovers images under the raw source directory.
  2. Validates each image (PIL integrity, dimensions, format).
  3. Computes SHA-256 to detect duplicates against existing output.
  4. Maps source label → v6 class using the source registry.
  5. Normalizes: resize to 224×224 with letterbox padding, convert to
     consistent grayscale (for CXR) or RGB (for microscopy).
  6. Saves to data/medical_ready/<v6_class>/<source_id>_<hash8>_<idx>.jpg
  7. Writes a per-run normalization log and updates the hash cache.

What it does NOT do
-------------------
  - No automatic downloading.
  - No ML inference.
  - No train/val/test splitting (done by manifest stage).
  - No writing to the production inference pipeline.

Usage
-----
# Normalize all images from a single manually-staged source:
python scripts/normalize_medical_dataset.py \\
    --source-id kermany \\
    --raw-dir data/medical_raw \\
    --out-dir data/medical_ready

# With a CSV metadata file (RSNA, NIH):
python scripts/normalize_medical_dataset.py \\
    --source-id rsna \\
    --raw-dir data/medical_raw \\
    --out-dir data/medical_ready \\
    --metadata-csv data/medical_raw/rsna/stage_2_train_labels.csv \\
    --csv-id-col patientId \\
    --csv-label-col Target

# Dry-run (validate only, do not write output):
python scripts/normalize_medical_dataset.py \\
    --source-id kermany --dry-run

Exit codes: 0 clean, 1 partial failures, 2 critical error.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("normalize")

# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_SIZE: int = 224
JPEG_QUALITY: int = 95
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"
})
# DICOM handled separately via optional pydicom import
DICOM_EXTENSION: str = ".dcm"

OUTPUT_CLASSES: tuple[str, ...] = (
    "healthy_xray",
    "pneumonia_xray",
    "uncertain_xray",
    "rejected_quality",
    "fake_medical",
)

_CACHE_FILE: str = ".normalization_cache.json"


# ── Config & result types ─────────────────────────────────────────────────────


@dataclass
class NormalizationConfig:
    target_size: int = TARGET_SIZE
    jpeg_quality: int = JPEG_QUALITY
    min_input_dimension: int = 64
    recommended_input_dimension: int = 224
    force_grayscale_for_xray: bool = True    # CXR classes → always save as L (grayscale)
    min_trainability_score: float = 0.50     # below this → rejected_quality
    letterbox_fill: int = 0                  # pixel value for letterbox padding (0 = black)
    skip_existing: bool = True               # skip if output hash already in cache


@dataclass
class NormalizedImageRecord:
    """Metadata record for one normalized image."""
    source_id: str
    source_label: str
    v6_label: str
    original_path: str
    output_path: str | None
    sha256: str
    original_width: int
    original_height: int
    output_width: int
    output_height: int
    is_grayscale: bool
    trainability_score: float
    quality_flags: list[str]
    status: str                   # "ok" | "rejected" | "duplicate" | "error"
    rejection_reason: str | None
    processed_at: str


@dataclass
class NormalizationRunSummary:
    source_id: str
    started_at: str
    finished_at: str
    total_discovered: int
    processed_ok: int
    rejected_quality: int
    skipped_duplicate: int
    skipped_existing: int
    errors: int
    output_dir: str
    records: list[NormalizedImageRecord] = field(default_factory=list)


# ── Quality gate ──────────────────────────────────────────────────────────────


def _trainability_score(
    width: int,
    height: int,
    is_grayscale: bool,
    v6_label: str,
) -> tuple[float, list[str]]:
    """
    Compute a [0, 1] trainability score and collect quality flags.

    Flags do not reject images; the score threshold does.
    """
    flags: list[str] = []
    score = 1.0

    # Dimension score: penalise images well below target
    min_dim = min(width, height)
    if min_dim < 64:
        return 0.0, ["too_small"]
    if min_dim < TARGET_SIZE:
        dim_penalty = (TARGET_SIZE - min_dim) / TARGET_SIZE * 0.30
        score -= dim_penalty
        flags.append(f"below_target_size_{min_dim}px")

    # Aspect ratio: severe distortion hurts model
    ratio = max(width, height) / max(min(width, height), 1)
    if ratio > 2.5:
        score -= 0.20
        flags.append(f"extreme_aspect_ratio_{ratio:.1f}")
    elif ratio > 1.8:
        score -= 0.08
        flags.append(f"high_aspect_ratio_{ratio:.1f}")

    # Grayscale consistency for CXR classes
    xray_classes = {"healthy_xray", "pneumonia_xray", "uncertain_xray", "opacity_pattern"}
    if v6_label in xray_classes and not is_grayscale:
        score -= 0.05
        flags.append("rgb_xray_converted")

    return round(max(0.0, min(1.0, score)), 4), flags


# ── DICOM support (optional) ──────────────────────────────────────────────────


def _load_dicom(path: Path):
    """Convert a DICOM file to a PIL Image. Requires pydicom."""
    try:
        import pydicom
        import numpy as np
        from PIL import Image

        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(float)
        # Normalise to [0, 255]
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        img = Image.fromarray(arr.astype("uint8"))
        if img.mode != "L":
            img = img.convert("L")
        return img
    except ImportError:
        raise RuntimeError(
            "pydicom is required for DICOM files: pip install pydicom"
        )


# ── Normalizer ────────────────────────────────────────────────────────────────


class ImageNormalizer:
    """
    Resize + pad + mode-convert a single PIL image.

    Output is always TARGET_SIZE × TARGET_SIZE.
    Letterbox padding (black fill) preserves aspect ratio.
    """

    def __init__(self, cfg: NormalizationConfig) -> None:
        self.cfg = cfg

    def normalize(self, image, v6_label: str):
        """Return a normalized PIL image ready for saving."""
        from PIL import Image

        # Convert RGBA → RGB (or grayscale)
        if image.mode == "RGBA":
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[3])
            image = bg

        # CXR classes → force grayscale
        xray_classes = {"healthy_xray", "pneumonia_xray", "uncertain_xray",
                        "opacity_pattern", "infiltrate_pattern", "hantavirus_candidate"}
        if self.cfg.force_grayscale_for_xray and v6_label in xray_classes:
            image = image.convert("L")
        elif image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        t = self.cfg.target_size
        w, h = image.size

        # Letterbox resize: scale so the longer side fits in t×t
        scale = t / max(w, h)
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))

        # Use LANCZOS for downscaling, BICUBIC for upscaling
        resample = Image.LANCZOS if scale < 1.0 else Image.BICUBIC
        image = image.resize((new_w, new_h), resample)

        # Pad to t×t
        fill = self.cfg.letterbox_fill
        mode = image.mode
        canvas = Image.new(mode, (t, t), fill if mode == "L" else (fill, fill, fill))
        paste_x = (t - new_w) // 2
        paste_y = (t - new_h) // 2
        canvas.paste(image, (paste_x, paste_y))
        return canvas


# ── Hash cache ────────────────────────────────────────────────────────────────


def _load_cache(out_dir: Path) -> dict[str, str]:
    """Load {sha256: output_path} cache from disk."""
    cache_path = out_dir / _CACHE_FILE
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(out_dir: Path, cache: dict[str, str]) -> None:
    (out_dir / _CACHE_FILE).write_text(json.dumps(cache, indent=2))


# ── Label discovery ───────────────────────────────────────────────────────────


def _discover_images(
    raw_source_dir: Path,
    label_mapping: dict[str, str],
    metadata_csv: Path | None,
    csv_id_col: str,
    csv_label_col: str,
) -> list[tuple[Path, str, str]]:
    """
    Return [(image_path, source_label, v6_label), ...].

    Two discovery modes:
      1. Directory-based: images in subdirs named by source label.
      2. CSV-based: flat directory + label CSV (RSNA, NIH style).
    """
    found: list[tuple[Path, str, str]] = []

    if metadata_csv and metadata_csv.exists():
        # CSV mode: read label from CSV, find image file by ID
        id_to_label: dict[str, str] = {}
        with open(metadata_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id  = row.get(csv_id_col, "").strip()
                raw_lbl = row.get(csv_label_col, "").strip()
                if img_id:
                    id_to_label[img_id] = raw_lbl

        for ext in SUPPORTED_EXTENSIONS | {DICOM_EXTENSION}:
            for img_path in raw_source_dir.rglob(f"*{ext}"):
                stem = img_path.stem
                raw_lbl = id_to_label.get(stem, id_to_label.get(img_path.name, ""))
                v6 = label_mapping.get(raw_lbl, "uncertain_xray")
                found.append((img_path, raw_lbl or "unknown", v6))
    else:
        # Directory mode: subdir name is the source label
        for ext in SUPPORTED_EXTENSIONS | {DICOM_EXTENSION}:
            for img_path in raw_source_dir.rglob(f"*{ext}"):
                # The immediate parent dir is the source label
                parts = img_path.relative_to(raw_source_dir).parts
                raw_lbl = parts[0] if len(parts) > 1 else "unknown"
                v6 = label_mapping.get(raw_lbl, "uncertain_xray")
                found.append((img_path, raw_lbl, v6))

    return found


# ── Pipeline ──────────────────────────────────────────────────────────────────


class NormalizationPipeline:

    def __init__(self, cfg: NormalizationConfig) -> None:
        self.cfg = cfg
        self.normalizer = ImageNormalizer(cfg)

    def run(
        self,
        *,
        source_id: str,
        label_mapping: dict[str, str],
        raw_source_dir: Path,
        out_dir: Path,
        metadata_csv: Path | None = None,
        csv_id_col: str = "id",
        csv_label_col: str = "label",
        dry_run: bool = False,
    ) -> NormalizationRunSummary:
        from PIL import Image
        from app.modules.vision.medical.dataset_validator import compute_sha256

        started_at = datetime.now(timezone.utc).isoformat()
        logger.info("Source : %s", source_id)
        logger.info("Raw dir: %s", raw_source_dir)
        logger.info("Out dir: %s", out_dir)

        if not dry_run:
            for cls in OUTPUT_CLASSES:
                (out_dir / cls).mkdir(parents=True, exist_ok=True)

        cache = _load_cache(out_dir) if not dry_run else {}
        images = _discover_images(
            raw_source_dir, label_mapping, metadata_csv, csv_id_col, csv_label_col
        )
        logger.info("Discovered %d images", len(images))

        records: list[NormalizedImageRecord] = []
        counters = dict(ok=0, rejected=0, duplicate=0, existing=0, error=0)
        global_idx = sum(1 for p in out_dir.rglob("*.jpg")) if not dry_run else 0

        for img_path, raw_lbl, v6_label in images:
            rec = self._process_one(
                img_path=img_path,
                raw_lbl=raw_lbl,
                v6_label=v6_label,
                source_id=source_id,
                out_dir=out_dir,
                cache=cache,
                idx=global_idx,
                dry_run=dry_run,
            )
            records.append(rec)
            counters[rec.status if rec.status in counters else "error"] += 1
            if rec.status == "ok":
                global_idx += 1
                if not dry_run and rec.sha256 and rec.output_path:
                    cache[rec.sha256] = rec.output_path

        if not dry_run:
            _save_cache(out_dir, cache)

        summary = NormalizationRunSummary(
            source_id=source_id,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
            total_discovered=len(images),
            processed_ok=counters["ok"],
            rejected_quality=counters["rejected"],
            skipped_duplicate=counters["duplicate"],
            skipped_existing=counters["existing"],
            errors=counters["error"],
            output_dir=str(out_dir),
            records=records,
        )
        logger.info(
            "Done: ok=%d rejected=%d duplicate=%d existing=%d errors=%d",
            summary.processed_ok, summary.rejected_quality,
            summary.skipped_duplicate, summary.skipped_existing, summary.errors,
        )
        return summary

    def _process_one(
        self,
        *,
        img_path: Path,
        raw_lbl: str,
        v6_label: str,
        source_id: str,
        out_dir: Path,
        cache: dict[str, str],
        idx: int,
        dry_run: bool,
    ) -> NormalizedImageRecord:
        from PIL import Image, UnidentifiedImageError
        from app.modules.vision.medical.dataset_validator import compute_sha256

        now = datetime.now(timezone.utc).isoformat()

        def _err(reason: str) -> NormalizedImageRecord:
            return NormalizedImageRecord(
                source_id=source_id, source_label=raw_lbl, v6_label=v6_label,
                original_path=str(img_path), output_path=None,
                sha256="", original_width=0, original_height=0,
                output_width=0, output_height=0, is_grayscale=False,
                trainability_score=0.0, quality_flags=[],
                status="error", rejection_reason=reason, processed_at=now,
            )

        # Hash
        try:
            sha256 = compute_sha256(img_path)
        except Exception as e:
            return _err(f"hash_failed: {e}")

        # Skip if already in cache
        if self.cfg.skip_existing and sha256 in cache:
            return NormalizedImageRecord(
                source_id=source_id, source_label=raw_lbl, v6_label=v6_label,
                original_path=str(img_path), output_path=cache[sha256],
                sha256=sha256, original_width=0, original_height=0,
                output_width=TARGET_SIZE, output_height=TARGET_SIZE,
                is_grayscale=True, trainability_score=1.0, quality_flags=[],
                status="existing", rejection_reason=None, processed_at=now,
            )

        # Open image (DICOM or PIL)
        try:
            if img_path.suffix.lower() == DICOM_EXTENSION:
                pil_img = _load_dicom(img_path)
            else:
                pil_img = Image.open(img_path)
                pil_img.verify()
                pil_img = Image.open(img_path)
        except Exception as e:
            return _err(f"open_failed: {e}")

        orig_w, orig_h = pil_img.size
        is_grayscale = pil_img.mode in ("L", "LA")

        # Quality gate
        t_score, flags = _trainability_score(orig_w, orig_h, is_grayscale, v6_label)

        if t_score < self.cfg.min_trainability_score:
            if not dry_run:
                _write_rejected(out_dir, img_path, sha256, source_id, flags)
            return NormalizedImageRecord(
                source_id=source_id, source_label=raw_lbl, v6_label="rejected_quality",
                original_path=str(img_path), output_path=None,
                sha256=sha256, original_width=orig_w, original_height=orig_h,
                output_width=0, output_height=0, is_grayscale=is_grayscale,
                trainability_score=t_score, quality_flags=flags,
                status="rejected", rejection_reason=f"score={t_score} flags={flags}",
                processed_at=now,
            )

        # Normalize
        try:
            norm_img = self.normalizer.normalize(pil_img, v6_label)
        except Exception as e:
            return _err(f"normalize_failed: {e}")

        # Determine output path
        out_class_dir = out_dir / v6_label
        out_filename  = f"{source_id}_{sha256[:8]}_{idx:05d}.jpg"
        out_path = out_class_dir / out_filename

        if not dry_run:
            norm_img.save(str(out_path), format="JPEG", quality=self.cfg.jpeg_quality)

        return NormalizedImageRecord(
            source_id=source_id, source_label=raw_lbl, v6_label=v6_label,
            original_path=str(img_path), output_path=str(out_path),
            sha256=sha256, original_width=orig_w, original_height=orig_h,
            output_width=TARGET_SIZE, output_height=TARGET_SIZE,
            is_grayscale=(norm_img.mode == "L"),
            trainability_score=t_score, quality_flags=flags,
            status="ok", rejection_reason=None, processed_at=now,
        )


def _write_rejected(
    out_dir: Path,
    img_path: Path,
    sha256: str,
    source_id: str,
    flags: list[str],
) -> None:
    """Copy rejected image to rejected_quality/ with a sidecar .reject file."""
    rej_dir = out_dir / "rejected_quality"
    rej_dir.mkdir(parents=True, exist_ok=True)
    dest = rej_dir / f"{source_id}_{sha256[:8]}{img_path.suffix}"
    try:
        shutil.copy2(img_path, dest)
        dest.with_suffix(".reject").write_text("\n".join(flags))
    except Exception:
        pass


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="normalize_medical_dataset",
        description="Normalize manually-staged medical images → data/medical_ready/.",
    )
    p.add_argument("--source-id",     required=True,
                   help="Source ID matching a KNOWN_SOURCES entry (e.g. 'kermany').")
    p.add_argument("--raw-dir",       type=Path, default=Path("data/medical_raw"),
                   help="Root of raw datasets (default: data/medical_raw).")
    p.add_argument("--out-dir",       type=Path, default=Path("data/medical_ready"),
                   help="Output directory (default: data/medical_ready).")
    p.add_argument("--metadata-csv",  type=Path, default=None,
                   help="Optional label CSV for flat-directory sources (RSNA, NIH).")
    p.add_argument("--csv-id-col",    default="patientId",
                   help="CSV column for image ID/filename stem (default: patientId).")
    p.add_argument("--csv-label-col", default="Target",
                   help="CSV column for label (default: Target).")
    p.add_argument("--target-size",   type=int, default=TARGET_SIZE,
                   help=f"Output image size (default: {TARGET_SIZE}).")
    p.add_argument("--min-score",     type=float, default=0.50,
                   help="Minimum trainability score to accept an image (default: 0.50).")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Re-process images already in the cache.")
    p.add_argument("--dry-run",       action="store_true",
                   help="Validate and score images without writing output.")
    p.add_argument("--log-output",    type=Path, default=None,
                   help="Write run summary JSON to this file.")
    return p


def main() -> int:
    args = build_parser().parse_args()

    from scripts.medical_dataset_sources import KNOWN_SOURCES, check_raw_directory

    if args.source_id not in KNOWN_SOURCES:
        logger.error("Unknown source-id: %r — run --list to see registered sources.", args.source_id)
        return 2

    src = KNOWN_SOURCES[args.source_id]
    raw_source_dir = args.raw_dir / args.source_id

    # Pre-flight check
    check = check_raw_directory(args.source_id, args.raw_dir)
    if not check.get("ok"):
        logger.error("Raw directory check failed: %s", check.get("error", "unknown"))
        logger.error("Run: python scripts/medical_dataset_sources.py --info %s", args.source_id)
        return 2

    cfg = NormalizationConfig(
        target_size=args.target_size,
        min_trainability_score=args.min_score,
        skip_existing=not args.no_skip_existing,
    )
    pipeline = NormalizationPipeline(cfg)

    t0 = time.perf_counter()
    summary = pipeline.run(
        source_id=args.source_id,
        label_mapping=src.label_mapping,
        raw_source_dir=raw_source_dir,
        out_dir=args.out_dir,
        metadata_csv=args.metadata_csv,
        csv_id_col=args.csv_id_col,
        csv_label_col=args.csv_label_col,
        dry_run=args.dry_run,
    )
    elapsed = time.perf_counter() - t0
    logger.info("Elapsed: %.1f s  (%.0f img/s)", elapsed,
                summary.total_discovered / max(elapsed, 0.001))

    if args.log_output:
        args.log_output.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v for k, v in asdict(summary).items() if k != "records"}
        data["record_count"] = len(summary.records)
        args.log_output.write_text(json.dumps(data, indent=2))
        logger.info("Run log: %s", args.log_output)

    if summary.errors > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
