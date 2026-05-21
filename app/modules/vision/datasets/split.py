"""
Dataset splitting utilities — manifest-aware, stratified, integrity-checked.

Two modes of operation:

  1. split_from_manifest(manifest, processed_dir, splits_dir)
     Reads the manifest and assigns each validated, accepted record to a
     split, then hard-links (or copies) the file into the split directory.
     Split assignments are written back into the manifest.

  2. create_splits(source_dir, output_dir)
     Legacy file-system split for cases where no manifest exists yet.
     Kept for backward compatibility and ad-hoc use.

Guarantees provided by split_from_manifest
------------------------------------------
  - Stratified: each class maintains its proportion across splits.
  - No leakage: the test set is drawn first and frozen. val is drawn from
    the remaining pool. This prevents test contamination.
  - Reproducible: fixed random seed produces identical assignments.
  - Auditable: every assignment is recorded in the manifest.
  - No data loss: rejected or unvalidated images are skipped gracefully
    and logged.
"""
from __future__ import annotations

import logging
import math
import random
import shutil
from pathlib import Path

from app.modules.vision.datasets.manifest import DatasetManifest
from app.modules.vision.datasets.schema import ImageClass, QualityFlag, Split

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Default split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def split_from_manifest(
    manifest: DatasetManifest,
    processed_dir: Path,
    splits_dir: Path,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = 42,
    copy: bool = True,
    skip_existing: bool = True,
) -> dict[str, dict[str, int]]:
    """
    Assign train/val/test splits to manifest records and materialize them
    on disk.

    Only records that are validated and pass quality acceptance are split.
    Records with ``split != UNASSIGNED`` are skipped (preserves existing
    assignments for incremental dataset updates).

    Parameters
    ----------
    manifest : DatasetManifest
        Loaded manifest. Will be mutated (split field updated) in-place.
        Caller must call manifest.save() after this function returns.
    processed_dir : Path
        Root directory containing per-class subdirectories of processed images.
        Expected: processed_dir/<class_name>/<filename>
    splits_dir : Path
        Destination root. Structure created:
        splits_dir/<split>/<class_name>/<filename>
    train_ratio, val_ratio, test_ratio : float
        Must sum to 1.0.
    random_state : int
        Reproducibility seed.
    copy : bool
        If True, copy files. If False, create hard links (saves disk space).
    skip_existing : bool
        If True, records already assigned to a split are not re-assigned.

    Returns
    -------
    dict mapping split_name → class_name → count
    """
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = random.Random(random_state)

    stats: dict[str, dict[str, int]] = {
        s: {c.value: 0 for c in ImageClass}
        for s in ("train", "val", "test")
    }

    for cls in ImageClass:
        # Eligible: validated, accepted quality, not already split (if skip_existing)
        candidates = [
            r for r in manifest.by_class(cls)
            if r.validated
            and r.is_acceptable_quality
            and (r.split == Split.UNASSIGNED or not skip_existing)
        ]

        if not candidates:
            logger.info(
                "split_from_manifest: no eligible records for class '%s'", cls.value
            )
            continue

        rng.shuffle(candidates)
        n = len(candidates)

        # Draw test first, then val, remainder is train
        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))
        n_train = n - n_test - n_val

        if n_train < 1:
            logger.warning(
                "Class '%s': only %d eligible images — cannot form all 3 splits. "
                "Assigning all to train.", cls.value, n
            )
            assignments = [(r, Split.TRAIN) for r in candidates]
        else:
            assignments = (
                [(r, Split.TEST)  for r in candidates[:n_test]]
                + [(r, Split.VAL)   for r in candidates[n_test:n_test + n_val]]
                + [(r, Split.TRAIN) for r in candidates[n_test + n_val:]]
            )

        for record, split in assignments:
            src = processed_dir / cls.value / record.filename
            if not src.exists():
                logger.warning(
                    "split_from_manifest: source file missing: %s — skipping", src
                )
                continue

            dst_dir = splits_dir / split.value / cls.value
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / record.filename

            if not dst.exists():
                if copy:
                    shutil.copy2(src, dst)
                else:
                    try:
                        dst.hardlink_to(src)
                    except OSError:
                        shutil.copy2(src, dst)

            manifest.assign_split(record.image_id, split)
            stats[split.value][cls.value] += 1

        logger.info(
            "Class '%s': train=%d | val=%d | test=%d",
            cls.value,
            stats["train"][cls.value],
            stats["val"][cls.value],
            stats["test"][cls.value],
        )

    _log_split_stats(stats)
    return stats


def _log_split_stats(stats: dict[str, dict[str, int]]) -> None:
    logger.info("─── Split Summary ───────────────────────────────")
    for split, counts in stats.items():
        total = sum(counts.values())
        parts = "  ".join(f"{cls}={n}" for cls, n in counts.items() if n > 0)
        logger.info("  %-8s total=%-5d  %s", split, total, parts)
    logger.info("─────────────────────────────────────────────────")


# ── Legacy file-system split ──────────────────────────────────────────────────


def create_splits(
    source_dir: Path | str,
    output_dir: Path | str,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = 42,
    copy: bool = True,
    classes: list[str] | None = None,
) -> dict[str, dict[str, int]]:
    """
    Legacy split utility.  Works on a flat class-directory structure without
    a manifest.  Use ``split_from_manifest`` for production workflows.

    Parameters
    ----------
    source_dir : Path
        Root directory with class sub-directories (related/, unrelated/, ...).
    output_dir : Path
        Destination for the split directories.
    """
    from app.modules.vision.datasets.dataset import CLASSES as _DEFAULT_CLASSES

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    _classes = classes if classes is not None else _DEFAULT_CLASSES

    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = random.Random(random_state)
    stats: dict[str, dict[str, int]] = {
        s: {cls: 0 for cls in _classes}
        for s in ("train", "val", "test")
    }

    for class_name in _classes:
        class_src = source_dir / class_name
        if not class_src.exists():
            logger.warning("Class directory not found, skipping: %s", class_src)
            continue

        images = sorted(
            p for p in class_src.iterdir()
            if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not images:
            logger.warning("No images found in %s", class_src)
            continue

        rng.shuffle(images)
        n = len(images)
        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))

        splits_images = {
            "test":  images[:n_test],
            "val":   images[n_test:n_test + n_val],
            "train": images[n_test + n_val:],
        }

        for split_name, split_paths in splits_images.items():
            dest_dir = output_dir / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for src_path in split_paths:
                dest_path = dest_dir / src_path.name
                if copy:
                    shutil.copy2(src_path, dest_path)
                else:
                    if not dest_path.exists():
                        try:
                            dest_path.hardlink_to(src_path.resolve())
                        except OSError:
                            shutil.copy2(src_path, dest_path)

                stats[split_name][class_name] += 1

        logger.info(
            "Class '%s': train=%d | val=%d | test=%d",
            class_name,
            stats["train"][class_name],
            stats["val"][class_name],
            stats["test"][class_name],
        )

    _log_split_stats(stats)
    return stats
