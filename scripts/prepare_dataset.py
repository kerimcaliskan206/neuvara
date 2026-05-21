"""
HantaProject — Professional Dataset Preparation Pipeline
=========================================================

Transforms a raw image drop zone into a versioned, quality-validated,
deduplicated, and split dataset ready for vision model training.

Pipeline stages
---------------
  1. Discover raw images in the drop zone (data/vision/raw/<class>/)
  2. Validate image format and decodability
  3. Compute content hash (SHA-256) + perceptual hash (dhash)
  4. Exact-duplicate check — skip images already in the manifest
  5. Near-duplicate check — flag images within Hamming distance threshold
  6. Quality scoring — blur, brightness, contrast, resolution
  7. Ingest accepted images into versioned processed/ directory
  8. Record all metadata in manifest.json
  9. Assign train/val/test splits via stratified random sampling
 10. Materialize split directories (copy or hard-link)
 11. Register the version in the dataset version index
 12. Write a JSON pipeline report alongside the manifest
 13. Print a comprehensive human-readable summary

Usage
-----
  # Create version v1 from everything in data/vision/raw/
  python scripts/prepare_dataset.py

  # Named version with description
  python scripts/prepare_dataset.py \\
      --version v1_baseline \\
      --description "Initial curated set from CDC and literature"

  # Use hard links instead of file copies (saves disk on same filesystem)
  python scripts/prepare_dataset.py --no-copy

  # Skip quality filtering (accept all decodable, non-duplicate images)
  python scripts/prepare_dataset.py --no-quality-filter

  # Dry run — analyze without writing anything
  python scripts/prepare_dataset.py --dry-run

  # Validate an already-prepared version
  python scripts/prepare_dataset.py --validate-only --version v1
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from app.core.logging import setup_logging  # noqa: E402
from app.modules.vision.datasets.deduplication import (  # noqa: E402
    DuplicateDetector,
    content_hash as compute_content_hash,
    perceptual_hash as compute_perceptual_hash,
)
from app.modules.vision.datasets.manifest import DatasetManifest  # noqa: E402
from app.modules.vision.datasets.quality import ImageQualityValidator  # noqa: E402
from app.modules.vision.datasets.schema import (  # noqa: E402
    ImageClass,
    ImageRecord,
    QualityFlag,
    SourceType,
    Split,
)
from app.modules.vision.datasets.split import split_from_manifest  # noqa: E402
from app.modules.vision.datasets.versioning import DatasetVersionManager  # noqa: E402

logger = logging.getLogger(__name__)

RAW_DIR = _PROJECT_ROOT / "data" / "vision" / "raw"
DATASETS_DIR = _PROJECT_ROOT / "data" / "vision" / "datasets"

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

_DIR_TO_CLASS: dict[str, ImageClass] = {
    "related":      ImageClass.RELATED,
    "unrelated":    ImageClass.UNRELATED,
    "hard_negative": ImageClass.HARD_NEGATIVE,
}


# ── Rejection tracking ────────────────────────────────────────────────────────


class RejectionReason(str, Enum):
    UNREADABLE   = "unreadable"    # OSError reading file
    EMPTY        = "empty"         # 0-byte file
    CORRUPT      = "corrupt"       # not a valid image
    EXACT_DUP    = "exact_duplicate"
    NEAR_DUP     = "near_duplicate"
    LOW_QUALITY  = "low_quality"


@dataclass
class IngestResult:
    """Outcome of processing a single raw image."""
    record: ImageRecord | None
    rejection_reason: RejectionReason | None = None
    rejection_detail: str = ""

    @property
    def accepted(self) -> bool:
        return self.record is not None


@dataclass
class PipelineCounters:
    accepted: int = 0
    rejected_unreadable: int = 0
    rejected_empty: int = 0
    rejected_corrupt: int = 0
    rejected_exact_dup: int = 0
    rejected_near_dup: int = 0
    rejected_low_quality: int = 0

    def record(self, result: IngestResult) -> None:
        if result.accepted:
            self.accepted += 1
            return
        r = result.rejection_reason
        if r == RejectionReason.UNREADABLE:  self.rejected_unreadable += 1
        elif r == RejectionReason.EMPTY:     self.rejected_empty += 1
        elif r == RejectionReason.CORRUPT:   self.rejected_corrupt += 1
        elif r == RejectionReason.EXACT_DUP: self.rejected_exact_dup += 1
        elif r == RejectionReason.NEAR_DUP:  self.rejected_near_dup += 1
        elif r == RejectionReason.LOW_QUALITY: self.rejected_low_quality += 1

    @property
    def total_rejected(self) -> int:
        return (
            self.rejected_unreadable + self.rejected_empty + self.rejected_corrupt
            + self.rejected_exact_dup + self.rejected_near_dup + self.rejected_low_quality
        )

    def as_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "rejected_total": self.total_rejected,
            "rejected_breakdown": {
                "unreadable":      self.rejected_unreadable,
                "empty":           self.rejected_empty,
                "corrupt":         self.rejected_corrupt,
                "exact_duplicate": self.rejected_exact_dup,
                "near_duplicate":  self.rejected_near_dup,
                "low_quality":     self.rejected_low_quality,
            },
        }


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HantaProject — Prepare a versioned vision dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", default=None,
                        help="Version string (auto-increments if omitted).")
    parser.add_argument("--description", default="",
                        help="Human-readable version description.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR,
                        help="Raw image drop zone root.")
    parser.add_argument("--datasets-dir", type=Path, default=DATASETS_DIR,
                        help="Versioned datasets root.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio",   type=float, default=0.15)
    parser.add_argument("--test-ratio",  type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-copy", action="store_true",
                        help="Use hard links instead of file copies.")
    parser.add_argument("--no-quality-filter", action="store_true",
                        help="Accept all decodable, non-duplicate images.")
    parser.add_argument("--min-quality", type=float, default=0.5,
                        help="Minimum composite quality score for acceptance.")
    parser.add_argument("--hamming-threshold", type=int, default=8,
                        help="Hamming distance threshold for near-duplicate detection.")
    parser.add_argument(
        "--hard-negative-cap", type=int, default=200, metavar="N",
        help=(
            "Maximum hard_negative images to include (0 = no cap / use all). "
            "Images are sampled proportionally across subfolders to preserve "
            "category diversity. Default: 200."
        ),
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyse raw images and report without writing anything.")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate an existing version (use with --version).")
    return parser.parse_args()


# ── Image ingestion ───────────────────────────────────────────────────────────


def ingest_image(
    path: Path,
    image_class: ImageClass,
    validator: ImageQualityValidator,
    existing_content_hashes: set[str],
    existing_perceptual_hashes: list[str],
    detector: DuplicateDetector,
    apply_quality_filter: bool,
) -> IngestResult:
    """
    Process a single raw image file through the full ingestion pipeline.

    Returns an IngestResult. If accepted, result.record is a populated
    ImageRecord ready to be added to the manifest. If rejected, record is
    None and rejection_reason explains why.
    """
    from PIL import Image, UnidentifiedImageError

    # ── Stage 1: Read bytes ───────────────────────────────────────────────────
    try:
        data = path.read_bytes()
    except OSError as exc:
        return IngestResult(None, RejectionReason.UNREADABLE, str(exc))

    if len(data) == 0:
        return IngestResult(None, RejectionReason.EMPTY, "0-byte file")

    # ── Stage 2: Decode ───────────────────────────────────────────────────────
    try:
        image = Image.open(io.BytesIO(data))
        image.verify()
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, Exception) as exc:
        return IngestResult(
            None, RejectionReason.CORRUPT,
            f"PIL cannot decode: {exc}",
        )

    # ── Hard dimension guard — reject before hashing to save work ────────────
    w_img, h_img = image.size
    min_w = validator.thresholds.min_width
    min_h = validator.thresholds.min_height
    if w_img < min_w or h_img < min_h:
        return IngestResult(
            None, RejectionReason.LOW_QUALITY,
            f"Image too small: {w_img}×{h_img}px — minimum is {min_w}×{min_h}px",
        )

    # ── Stage 3: Exact duplicate ──────────────────────────────────────────────
    c_hash = compute_content_hash(data)
    if c_hash in existing_content_hashes:
        return IngestResult(
            None, RejectionReason.EXACT_DUP,
            f"content_hash={c_hash[:16]}… already in manifest",
        )

    # ── Stage 4: Perceptual near-duplicate ────────────────────────────────────
    p_hash = compute_perceptual_hash(image)
    if detector.is_duplicate_of_any(p_hash, existing_perceptual_hashes):
        return IngestResult(
            None, RejectionReason.NEAR_DUP,
            f"phash={p_hash} within Hamming {detector.hamming_threshold} of an existing image",
        )

    # ── Stage 5: Quality ──────────────────────────────────────────────────────
    quality_report = validator.validate(image)
    quality_flags = list(quality_report.flags)

    if apply_quality_filter and not quality_report.passed:
        return IngestResult(
            None, RejectionReason.LOW_QUALITY,
            f"score={quality_report.quality_score:.3f} — " + quality_report.summary(),
        )

    # ── Stage 6: Format mismatch flag (non-blocking) ──────────────────────────
    actual_fmt   = (image.format or "").lower()
    declared_ext = path.suffix.lower().lstrip(".")
    if declared_ext == "jpg":
        declared_ext = "jpeg"
    if actual_fmt and actual_fmt != declared_ext:
        quality_flags.append(QualityFlag.FORMAT_MISMATCH)

    # ── Stage 7: Build record ─────────────────────────────────────────────────
    w, h = image.size
    record = ImageRecord(
        image_id=c_hash,
        filename=path.name,
        class_name=image_class,
        content_hash=c_hash,
        perceptual_hash=p_hash,
        source_type=SourceType.COLLECTED,
        width=w,
        height=h,
        channels=3,
        format=(image.format or path.suffix.upper().lstrip(".")),
        file_size_bytes=len(data),
        quality_score=quality_report.quality_score,
        quality_flags=quality_flags,
        blur_score=quality_report.blur_score,
        brightness_mean=quality_report.brightness_mean,
        contrast_std=quality_report.contrast_std,
        validated=True,
    )
    return IngestResult(record)


# ── Hard-negative balanced sampling ──────────────────────────────────────────


def _discover_hn_subfolders(hn_dir: Path) -> dict[str, list[Path]]:
    """
    Recursively discover hard_negative images grouped by source subfolder.

    Flat files directly inside `hn_dir` are grouped under the key ``"_root"``.
    Each real sub-directory (e.g. fox/, gorilla/, wolf/) gets its own key.

    Returns an ordered dict sorted so that the largest groups come last
    (helps the remainder-distribution step in proportional sampling land on
    the rarest categories rather than the most common ones).
    """
    groups: dict[str, list[Path]] = {}

    flat = sorted(
        p for p in hn_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if flat:
        groups["_root"] = flat

    for sub in sorted(hn_dir.iterdir()):
        if not sub.is_dir():
            continue
        sub_files = sorted(
            p for p in sub.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if sub_files:
            groups[sub.name] = sub_files

    return groups


def _proportional_sample(
    groups: dict[str, list[Path]],
    target: int,
    seed: int,
) -> tuple[list[Path], dict[str, dict]]:
    """
    Sample ``target`` images from hard_negative groups proportionally.

    Each group receives a floor-allocated share based on its relative size.
    Any remaining slots are distributed one-at-a-time to the groups that
    still have the most unselected images (maximises coverage of rare groups).

    Within each group the list is shuffled with ``seed`` before slicing so
    that repeated runs with the same seed are deterministic, while different
    seeds explore different diversity pockets.

    Returns
    -------
    sampled : flat list of selected ``Path`` objects (order: group by group)
    report  : per-group dict with keys
              total_available / allocated / selected / rejected
    """
    import random as _random

    rng = _random.Random(seed)
    total_available = sum(len(v) for v in groups.values())

    if total_available == 0:
        return [], {}

    effective_target = min(target, total_available)
    report: dict[str, dict] = {}

    # Floor-allocation
    allocations: dict[str, int] = {}
    for name, paths in groups.items():
        alloc = int(len(paths) / total_available * effective_target)
        alloc = min(alloc, len(paths))
        allocations[name] = alloc

    # Distribute remaining slots to groups with the most slack (largest gap)
    remainder = effective_target - sum(allocations.values())
    slack_order = sorted(
        groups.keys(),
        key=lambda n: len(groups[n]) - allocations[n],
        reverse=True,
    )
    for name in slack_order:
        if remainder <= 0:
            break
        if allocations[name] < len(groups[name]):
            allocations[name] += 1
            remainder -= 1

    sampled: list[Path] = []
    for name, paths in groups.items():
        budget = allocations[name]
        shuffled = list(paths)
        rng.shuffle(shuffled)
        selected = shuffled[:budget]
        sampled.extend(selected)
        report[name] = {
            "total_available": len(paths),
            "allocated": budget,
            "selected": len(selected),
            "rejected": len(paths) - len(selected),
        }
        logger.info(
            "HN sampling  %-15s available=%-4d  selected=%-4d  skipped=%d",
            f"'{name}':",
            len(paths), len(selected), len(paths) - len(selected),
        )

    logger.info(
        "HN sampling  total selected=%d / %d  (cap=%d)",
        len(sampled), total_available, target,
    )
    return sampled, report


# ── Core pipeline ─────────────────────────────────────────────────────────────


def discover_raw_images(raw_dir: Path) -> dict[ImageClass, list[Path]]:
    """
    Discover flat images in each class drop zone.

    For the ``hard_negative`` class this function only reads files *directly*
    inside ``raw/hard_negative/`` — files in sub-directories (fox/, gorilla/,
    wolf/, …) are handled by the balanced-sampling path in ``run_pipeline``
    which calls ``_discover_hn_subfolders`` + ``_proportional_sample`` and
    then replaces the hard_negative list before ingestion begins.
    """
    discovered: dict[ImageClass, list[Path]] = {cls: [] for cls in ImageClass}
    for dir_name, image_class in _DIR_TO_CLASS.items():
        class_dir = raw_dir / dir_name
        if not class_dir.exists():
            logger.warning("Drop zone directory missing: %s", class_dir)
            continue
        paths = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        discovered[image_class] = paths
        logger.info("Discovered %d image(s) in %s/", len(paths), dir_name)
    return discovered


def run_pipeline(args: argparse.Namespace) -> None:
    version_manager = DatasetVersionManager(args.datasets_dir)

    if args.validate_only:
        version = args.version or version_manager.latest_version()
        if version is None:
            logger.error("No dataset versions found in %s", args.datasets_dir)
            sys.exit(1)
        _validate_existing_version(version_manager, version)
        return

    version = args.version or version_manager.next_version()

    logger.info("=" * 62)
    logger.info("  HantaProject — Dataset Preparation")
    logger.info("  Version     : %s", version)
    logger.info("  Raw dir     : %s", args.raw_dir)
    logger.info("  Quality min : %.2f  (filter=%s)", args.min_quality, not args.no_quality_filter)
    logger.info("  Dedup HT    : %d bits", args.hamming_threshold)
    logger.info("  Dry run     : %s", args.dry_run)
    logger.info("=" * 62)

    if not args.dry_run and version_manager.exists(version):
        logger.error(
            "Version %s already exists. Choose a new version or use --validate-only.",
            version,
        )
        sys.exit(1)

    raw_images = discover_raw_images(args.raw_dir)

    # ── Hard-negative balanced sampling ──────────────────────────────────────
    hn_diversity_report: dict[str, dict] = {}
    hn_dir = args.raw_dir / "hard_negative"
    if hn_dir.exists():
        hn_cap = args.hard_negative_cap
        hn_groups = _discover_hn_subfolders(hn_dir)
        hn_total_available = sum(len(v) for v in hn_groups.values())
        logger.info(
            "Hard-negative raw pool: %d images across %d categories (cap=%s)",
            hn_total_available,
            len(hn_groups),
            str(hn_cap) if hn_cap > 0 else "disabled",
        )
        if hn_cap > 0 and hn_total_available > hn_cap:
            sampled_hn, hn_diversity_report = _proportional_sample(
                hn_groups, hn_cap, args.seed
            )
            raw_images[ImageClass.HARD_NEGATIVE] = sampled_hn
        else:
            # No cap or already within budget — use all images from all groups
            all_hn: list[Path] = []
            for paths in hn_groups.values():
                all_hn.extend(paths)
            raw_images[ImageClass.HARD_NEGATIVE] = all_hn
            for name, paths in hn_groups.items():
                hn_diversity_report[name] = {
                    "total_available": len(paths),
                    "allocated": len(paths),
                    "selected": len(paths),
                    "rejected": 0,
                }

    total_raw = sum(len(v) for v in raw_images.values())
    if total_raw == 0:
        logger.error(
            "No images found in drop zone at %s\n"
            "Place images in:\n"
            "  %s/related/\n  %s/unrelated/\n  %s/hard_negative/",
            args.raw_dir, args.raw_dir, args.raw_dir, args.raw_dir,
        )
        sys.exit(1)

    logger.info("Total raw images discovered: %d", total_raw)

    if args.dry_run:
        _dry_run_report(raw_images, args, hn_diversity_report=hn_diversity_report)
        return

    dirs = version_manager.init_version_dirs(version)

    manifest = DatasetManifest(manifest_dir=dirs["metadata"], version=version)
    manifest.load()

    validator = ImageQualityValidator(min_quality_score=args.min_quality)
    detector  = DuplicateDetector(hamming_threshold=args.hamming_threshold)

    counters = PipelineCounters()
    per_class_counters: dict[str, PipelineCounters] = {
        cls.value: PipelineCounters() for cls in ImageClass
    }

    existing_content_hashes: set[str]  = manifest.content_hashes()
    existing_perceptual_hashes: list[str] = manifest.perceptual_hashes()

    rejection_log: list[dict] = []

    for image_class, paths in raw_images.items():
        for path in paths:
            result = ingest_image(
                path=path,
                image_class=image_class,
                validator=validator,
                existing_content_hashes=existing_content_hashes,
                existing_perceptual_hashes=existing_perceptual_hashes,
                detector=detector,
                apply_quality_filter=not args.no_quality_filter,
            )

            counters.record(result)
            per_class_counters[image_class.value].record(result)

            if not result.accepted:
                rejection_log.append({
                    "file": str(path.relative_to(args.raw_dir)),
                    "reason": result.rejection_reason.value if result.rejection_reason else "unknown",
                    "detail": result.rejection_detail,
                })
                logger.debug(
                    "REJECT %s | %s | %s",
                    path.name,
                    result.rejection_reason.value if result.rejection_reason else "?",
                    result.rejection_detail[:80],
                )
                continue

            rec = result.record
            dst = dirs["processed"] / image_class.value / rec.filename
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(path, dst)

            manifest.add(rec)
            existing_content_hashes.add(rec.content_hash)
            existing_perceptual_hashes.append(rec.perceptual_hash)

        cls_c = per_class_counters[image_class.value]
        logger.info(
            "Class %-15s accepted=%d  rejected=%d  "
            "(corrupt=%d  exact_dup=%d  near_dup=%d  quality=%d)",
            f"'{image_class.value}':",
            cls_c.accepted,
            cls_c.total_rejected,
            cls_c.rejected_corrupt,
            cls_c.rejected_exact_dup,
            cls_c.rejected_near_dup,
            cls_c.rejected_low_quality,
        )

    manifest.save()

    logger.info(
        "Ingestion complete — total accepted=%d  rejected=%d  "
        "(unreadable=%d  empty=%d  corrupt=%d  exact_dup=%d  near_dup=%d  quality=%d)",
        counters.accepted, counters.total_rejected,
        counters.rejected_unreadable, counters.rejected_empty,
        counters.rejected_corrupt, counters.rejected_exact_dup,
        counters.rejected_near_dup, counters.rejected_low_quality,
    )

    if counters.accepted == 0:
        logger.error("No images were accepted. Cannot create splits.")
        sys.exit(1)

    # ── Split ─────────────────────────────────────────────────────────────────
    logger.info("Assigning train/val/test splits (seed=%d)…", args.seed)
    split_stats = split_from_manifest(
        manifest=manifest,
        processed_dir=dirs["processed"],
        splits_dir=dirs["splits"],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed,
        copy=not args.no_copy,
    )
    manifest.save()

    # ── Register version ──────────────────────────────────────────────────────
    full_stats = manifest.stats()
    version_manager.register(
        version=version,
        stats=full_stats,
        description=args.description,
    )

    # ── Write pipeline report ─────────────────────────────────────────────────
    report = {
        "version": version,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "pipeline": {
            "raw_dir": str(args.raw_dir),
            "total_raw": total_raw,
            "quality_filter": not args.no_quality_filter,
            "min_quality_score": args.min_quality,
            "hamming_threshold": args.hamming_threshold,
            "hard_negative_cap": args.hard_negative_cap,
            "split_ratios": {
                "train": args.train_ratio,
                "val": args.val_ratio,
                "test": args.test_ratio,
            },
            "seed": args.seed,
        },
        "ingestion": counters.as_dict(),
        "per_class_ingestion": {
            cls: c.as_dict()
            for cls, c in per_class_counters.items()
        },
        "hard_negative_diversity": _build_hn_diversity_report(hn_diversity_report),
        "split_stats": split_stats,
        "dataset_stats": full_stats,
        "rejections": rejection_log,
    }
    report_path = dirs["metadata"] / "pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Pipeline report saved → %s", report_path)

    _print_summary(version, dirs, full_stats, split_stats, counters, hn_diversity_report)


# ── Hard-negative diversity report builder ────────────────────────────────────


def _build_hn_diversity_report(hn_report: dict[str, dict]) -> dict:
    """Aggregate the per-group sampling stats into a structured report."""
    if not hn_report:
        return {}
    total_available = sum(g["total_available"] for g in hn_report.values())
    total_selected  = sum(g["selected"] for g in hn_report.values())
    total_rejected  = sum(g["rejected"] for g in hn_report.values())
    return {
        "total_available": total_available,
        "total_selected": total_selected,
        "total_rejected": total_rejected,
        "category_breakdown": hn_report,
    }


# ── Dry run ───────────────────────────────────────────────────────────────────


def _dry_run_report(
    raw_images: dict[ImageClass, list[Path]],
    args: argparse.Namespace,
    hn_diversity_report: dict[str, dict] | None = None,
) -> None:
    validator = ImageQualityValidator(min_quality_score=args.min_quality)
    detector  = DuplicateDetector(hamming_threshold=args.hamming_threshold)

    existing_c_hashes: set[str]  = set()
    existing_p_hashes: list[str] = []

    print("\n" + "=" * 62)
    print("  DRY RUN — no files will be written")
    print("  Quality filter active:", not args.no_quality_filter)
    print("=" * 62)

    grand_total = grand_accepted = 0

    for image_class, paths in raw_images.items():
        counters = PipelineCounters()
        for path in paths:
            result = ingest_image(
                path=path,
                image_class=image_class,
                validator=validator,
                existing_content_hashes=existing_c_hashes,
                existing_perceptual_hashes=existing_p_hashes,
                detector=detector,
                apply_quality_filter=not args.no_quality_filter,
            )
            counters.record(result)
            if result.accepted and result.record is not None:
                existing_c_hashes.add(result.record.content_hash)
                existing_p_hashes.append(result.record.perceptual_hash)

        grand_total    += len(paths)
        grand_accepted += counters.accepted
        rd = counters.as_dict()["rejected_breakdown"]

        print(f"\n  Class: {image_class.value}")
        print(f"    Raw images      : {len(paths)}")
        print(f"    Would accept    : {counters.accepted}")
        print(f"    Reject (corrupt): {rd['corrupt'] + rd['empty'] + rd['unreadable']}")
        print(f"    Reject (exact dup): {rd['exact_duplicate']}")
        print(f"    Reject (near dup) : {rd['near_duplicate']}")
        print(f"    Reject (quality)  : {rd['low_quality']}")

    print(f"\n  Total raw       : {grand_total}")
    print(f"  Total accepted  : {grand_accepted}")
    print(f"  Total rejected  : {grand_total - grand_accepted}")

    if hn_diversity_report:
        print("\n  Hard-negative Sampling Diversity (before quality filter):")
        total_avail = sum(g["total_available"] for g in hn_diversity_report.values())
        total_sel   = sum(g["selected"] for g in hn_diversity_report.values())
        for name, g in hn_diversity_report.items():
            pct = 100 * g["selected"] / max(total_sel, 1)
            print(
                f"    {name:<20}: avail={g['total_available']:>4}  "
                f"selected={g['selected']:>4}  skipped={g['rejected']:>4}  "
                f"({pct:.0f}% of HN pool)"
            )
        print(f"    {'TOTAL':<20}: avail={total_avail:>4}  selected={total_sel:>4}")

    print("=" * 62 + "\n")


# ── Validate existing version ─────────────────────────────────────────────────


def _validate_existing_version(manager: DatasetVersionManager, version: str) -> None:
    manifest = DatasetManifest(
        manifest_dir=manager.metadata_dir(version),
        version=version,
    )
    manifest.load()
    result = manifest.check_integrity(manager.processed_dir(version))
    logger.info("Validation for version %s:", version)
    logger.info("  Missing files  : %d", len(result["missing"]))
    logger.info("  Orphaned files : %d", len(result["orphaned"]))
    manifest.log_stats()


# ── Summary ───────────────────────────────────────────────────────────────────


def _print_summary(
    version: str,
    dirs: dict[str, Path],
    stats: dict,
    split_stats: dict,
    counters: PipelineCounters,
    hn_diversity_report: dict[str, dict] | None = None,
) -> None:
    W = 62
    sep = "─" * W
    lines = [
        "", sep,
        f"  Dataset Preparation Complete".ljust(W), sep,
        f"  Version               : {version}",
        f"  Total raw discovered  : {counters.accepted + counters.total_rejected}",
        f"  Accepted              : {counters.accepted}",
        f"  Rejected (unreadable) : {counters.rejected_unreadable}",
        f"  Rejected (empty)      : {counters.rejected_empty}",
        f"  Rejected (corrupt)    : {counters.rejected_corrupt}",
        f"  Rejected (exact dup)  : {counters.rejected_exact_dup}",
        f"  Rejected (near dup)   : {counters.rejected_near_dup}",
        f"  Rejected (quality)    : {counters.rejected_low_quality}",
        sep, "  Class Distribution:",
    ]
    for cls, count in stats.get("class_distribution", {}).items():
        pct = 100 * count / max(stats["total"], 1)
        lines.append(f"    {cls:<20} : {count:4d}  ({pct:.0f}%)")

    lines += [sep, "  Split Distribution:"]
    for split_name, counts in split_stats.items():
        total = sum(counts.values())
        parts = "  ".join(f"{c}={n}" for c, n in counts.items() if n > 0)
        lines.append(f"    {split_name:<8} total={total:<5}  {parts}")

    if hn_diversity_report:
        total_avail = sum(g["total_available"] for g in hn_diversity_report.values())
        total_sel   = sum(g["selected"]        for g in hn_diversity_report.values())
        lines += [sep, f"  Hard-negative Diversity  (sampled {total_sel}/{total_avail}):"]
        for name, g in hn_diversity_report.items():
            pct = 100 * g["selected"] / max(total_sel, 1)
            lines.append(
                f"    {name:<20}: {g['selected']:>4} selected / "
                f"{g['total_available']:>4} available  ({pct:.0f}%)"
            )

    q = stats.get("quality", {})
    lines += [
        sep,
        f"  Quality scores : mean={q.get('mean', 0):.3f}  "
        f"min={q.get('min', 0):.3f}  max={q.get('max', 0):.3f}",
        sep,
        f"  Processed → {dirs['processed']}",
        f"  Splits    → {dirs['splits']}",
        f"  Metadata  → {dirs['metadata']}",
        sep, "",
    ]
    print("\n".join(lines))


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    setup_logging(debug=True, environment="development")
    run_pipeline(args)


if __name__ == "__main__":
    main()
