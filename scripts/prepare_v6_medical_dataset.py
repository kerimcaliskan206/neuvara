#!/usr/bin/env python3
"""
Prepare v6 medical dataset for training — Phase 10.

Reads normalized images from data/medical_ready/ (Phase 9 output) and
manifest/split_plan from data/medical_manifest/ (Phase 9 manifest output),
then builds a training-ready split directory using symlinks:

    data/medical_v6_splits/
      train/
        healthy_xray/   ← symlinks → ../../medical_ready/healthy_xray/img.jpg
        pneumonia_xray/
        hard_negative/
        fake_medical/
      val/
        ...
      test/
        ...
      dataset_summary.json

The split directory is compatible with ImageFolderDataset (existing infra).
V5 production data and model are not touched.

Usage
-----
    # Full prepare with manifest
    python scripts/prepare_v6_medical_dataset.py \\
        --ready-dir data/medical_ready \\
        --manifest-dir data/medical_manifest \\
        --output-dir data/medical_v6_splits

    # Only target 2 classes (binary healthy vs pneumonia + OOD)
    python scripts/prepare_v6_medical_dataset.py \\
        --ready-dir data/medical_ready \\
        --manifest-dir data/medical_manifest \\
        --classes healthy_xray pneumonia_xray hard_negative fake_medical

    # Use copies instead of symlinks (for cross-filesystem compatibility)
    python scripts/prepare_v6_medical_dataset.py ... --copy

    # Dry run: validate structure without creating files
    python scripts/prepare_v6_medical_dataset.py ... --dry-run

Exit codes: 0 success, 1 error, 2 insufficient data.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("prepare_v6")

# Minimum samples per class per split for training to be viable
_MIN_TRAIN_SAMPLES: dict[str, int] = {
    "healthy_xray":   400,
    "pneumonia_xray": 400,
    "hard_negative":  200,
    "fake_medical":   100,
}
_MIN_VAL_SAMPLES: dict[str, int] = {
    "healthy_xray":   80,
    "pneumonia_xray": 80,
    "hard_negative":  40,
    "fake_medical":   20,
}

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ── Manifest loading ──────────────────────────────────────────────────────────


def _load_manifest(manifest_dir: Path) -> dict[str, dict]:
    """Return {image_id: entry_dict} from manifest.json."""
    manifest_path = manifest_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = data if isinstance(data, list) else data.get("images", data.get("entries", []))
    return {e["image_id"]: e for e in entries}


def _load_split_plan(manifest_dir: Path) -> dict[str, str]:
    """Return {image_id: 'train'|'val'|'test'} from split_plan.json."""
    split_path = manifest_dir / "split_plan.json"
    if not split_path.exists():
        raise FileNotFoundError(f"split_plan.json not found: {split_path}")
    data = json.loads(split_path.read_text(encoding="utf-8"))
    return data.get("assignments", data)


# ── Directory scanning (no-manifest fallback) ─────────────────────────────────


def _scan_ready_dir(
    ready_dir: Path, classes: list[str]
) -> dict[str, list[Path]]:
    """Return {class_name: [image_path, ...]} by scanning ready_dir/<class>/."""
    result: dict[str, list[Path]] = {}
    for cls in classes:
        cls_dir = ready_dir / cls
        if not cls_dir.exists():
            logger.warning("Class directory not found: %s", cls_dir)
            result[cls] = []
            continue
        images = sorted(
            p for p in cls_dir.iterdir()
            if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        result[cls] = images
        logger.info("  %-22s %d images", cls, len(images))
    return result


# ── Split directory builder ───────────────────────────────────────────────────


def _build_split_dirs_from_manifest(
    ready_dir: Path,
    manifest: dict[str, dict],
    split_plan: dict[str, str],
    output_dir: Path,
    classes: list[str],
    use_copy: bool,
    dry_run: bool,
) -> dict[str, dict[str, int]]:
    """
    Create output_dir/split/class/image symlinks (or copies).

    Returns counts: {split: {class: n_images}}.
    """
    target_classes = set(classes)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for image_id, entry in manifest.items():
        v6_label = entry.get("v6_label", "")
        if v6_label not in target_classes:
            continue

        split = split_plan.get(image_id)
        if split not in ("train", "val", "test"):
            continue

        src = Path(entry["path"])
        if not src.is_absolute():
            src = ready_dir / entry["path"]

        dst_dir = output_dir / split / v6_label
        dst     = dst_dir / src.name

        if dry_run:
            logger.debug("DRY-RUN  %s → %s", src.name, dst)
        else:
            dst_dir.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                if use_copy:
                    shutil.copy2(src, dst)
                else:
                    try:
                        dst.symlink_to(src.resolve())
                    except FileExistsError:
                        pass

        counts[split][v6_label] += 1

    return {s: dict(d) for s, d in counts.items()}


def _build_split_dirs_no_manifest(
    ready_dir: Path,
    output_dir: Path,
    classes: list[str],
    val_fraction: float,
    test_fraction: float,
    seed: int,
    use_copy: bool,
    dry_run: bool,
) -> dict[str, dict[str, int]]:
    """
    Fallback: no manifest. Stratified split from ready_dir/<class>/ images.
    """
    import random
    rng = random.Random(seed)

    class_images = _scan_ready_dir(ready_dir, classes)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for cls, images in class_images.items():
        shuffled = list(images)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, int(n * test_fraction))
        n_val  = max(1, int(n * val_fraction))
        n_train = n - n_val - n_test

        split_map: list[tuple[Path, str]] = (
            [(p, "train") for p in shuffled[:n_train]]
            + [(p, "val")   for p in shuffled[n_train:n_train+n_val]]
            + [(p, "test")  for p in shuffled[n_train+n_val:]]
        )

        for src, split in split_map:
            dst_dir = output_dir / split / cls
            dst     = dst_dir / src.name

            if dry_run:
                logger.debug("DRY-RUN  %s → %s/%s/%s", src.name, split, cls, src.name)
            else:
                dst_dir.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    if use_copy:
                        shutil.copy2(src, dst)
                    else:
                        try:
                            dst.symlink_to(src.resolve())
                        except FileExistsError:
                            pass

            counts[split][cls] += 1

    return {s: dict(d) for s, d in counts.items()}


# ── Validation ────────────────────────────────────────────────────────────────


def _validate_counts(
    counts: dict[str, dict[str, int]], classes: list[str]
) -> list[str]:
    """Return list of warning strings for insufficient sample counts."""
    warnings: list[str] = []
    minimums = {"train": _MIN_TRAIN_SAMPLES, "val": _MIN_VAL_SAMPLES}

    for split, mins in minimums.items():
        split_counts = counts.get(split, {})
        for cls in classes:
            n = split_counts.get(cls, 0)
            required = mins.get(cls, 0)
            if n < required:
                warnings.append(
                    f"[{split}/{cls}] {n} images — minimum {required} required"
                )

    return warnings


# ── Summary ───────────────────────────────────────────────────────────────────


def _write_summary(
    output_dir: Path,
    counts: dict[str, dict[str, int]],
    classes: list[str],
    warnings: list[str],
    dry_run: bool,
) -> None:
    summary = {
        "output_dir": str(output_dir),
        "classes": classes,
        "splits": {},
        "warnings": warnings,
        "ready_for_training": len(warnings) == 0,
    }
    for split in ("train", "val", "test"):
        split_counts = counts.get(split, {})
        total = sum(split_counts.values())
        summary["splits"][split] = {
            "total": total,
            "per_class": split_counts,
        }

    _print_table(counts, classes)

    if warnings:
        logger.warning("=== DATA WARNINGS ===")
        for w in warnings:
            logger.warning("  %s", w)

    if dry_run:
        logger.info("DRY-RUN complete — no files written.")
        return

    summary_path = output_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Summary written: %s", summary_path)


def _print_table(
    counts: dict[str, dict[str, int]], classes: list[str]
) -> None:
    splits = ["train", "val", "test"]
    header = f"  {'Class':<22}" + "".join(f"  {s:>8}" for s in splits) + "   Total"
    print("\n=== V6 Dataset Splits ===")
    print(header)
    print("  " + "-" * (22 + 12 * len(splits) + 8))
    for cls in classes:
        row = f"  {cls:<22}"
        total = 0
        for s in splits:
            n = counts.get(s, {}).get(cls, 0)
            row += f"  {n:>8}"
            total += n
        row += f"  {total:>6}"
        print(row)
    print()


# ── Entry point ───────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prepare_v6_medical_dataset",
        description="Build v6 medical training splits from normalized images.",
    )
    p.add_argument(
        "--ready-dir", type=Path, default=Path("data/medical_ready"),
        help="Root of normalized images (data/medical_ready/<class>/).",
    )
    p.add_argument(
        "--manifest-dir", type=Path, default=None,
        help="Directory containing manifest.json + split_plan.json (Phase 9 output). "
             "If omitted, splits are created from the ready_dir directly.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("data/medical_v6_splits"),
        help="Where to write train/val/test split dirs (default: data/medical_v6_splits).",
    )
    p.add_argument(
        "--classes", nargs="+",
        default=["healthy_xray", "pneumonia_xray", "hard_negative", "fake_medical"],
        help="Classes to include in the dataset.",
    )
    p.add_argument(
        "--val-fraction",  type=float, default=0.10,
        help="Fraction for validation split (no-manifest mode only).",
    )
    p.add_argument(
        "--test-fraction", type=float, default=0.15,
        help="Fraction for test split (no-manifest mode only).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for stratified split (no-manifest mode only).",
    )
    p.add_argument(
        "--copy", action="store_true",
        help="Copy images instead of symlinking (cross-filesystem safe).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Validate and print without creating any files.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    logger.info("=== Prepare V6 Medical Dataset — Phase 10 ===")
    logger.info("Ready dir  : %s", args.ready_dir)
    logger.info("Output dir : %s", args.output_dir)
    logger.info("Classes    : %s", args.classes)
    logger.info("Mode       : %s", "copy" if args.copy else "symlink")
    logger.info("Dry-run    : %s", args.dry_run)

    if not args.ready_dir.exists():
        logger.error("Ready dir does not exist: %s", args.ready_dir)
        logger.error("Run scripts/normalize_medical_dataset.py first.")
        return 1

    try:
        if args.manifest_dir is not None:
            logger.info("Manifest dir: %s", args.manifest_dir)
            manifest   = _load_manifest(args.manifest_dir)
            split_plan = _load_split_plan(args.manifest_dir)
            logger.info(
                "Manifest: %d entries | split_plan: %d entries",
                len(manifest), len(split_plan),
            )
            counts = _build_split_dirs_from_manifest(
                ready_dir=args.ready_dir,
                manifest=manifest,
                split_plan=split_plan,
                output_dir=args.output_dir,
                classes=args.classes,
                use_copy=args.copy,
                dry_run=args.dry_run,
            )
        else:
            logger.info(
                "No manifest provided — scanning ready_dir and creating stratified splits."
            )
            counts = _build_split_dirs_no_manifest(
                ready_dir=args.ready_dir,
                output_dir=args.output_dir,
                classes=args.classes,
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
                seed=args.seed,
                use_copy=args.copy,
                dry_run=args.dry_run,
            )

        warnings = _validate_counts(counts, args.classes)
        _write_summary(args.output_dir, counts, args.classes, warnings, args.dry_run)

        if warnings:
            logger.warning(
                "Dataset has %d insufficiency warning(s). "
                "Collect more data before training.",
                len(warnings),
            )
            return 2

        logger.info(
            "Dataset ready. Next step:\n"
            "  python scripts/train_v6_medical.py "
            "--dataset-dir %s --stage all",
            args.output_dir,
        )
        return 0

    except Exception:
        logger.exception("Failed to prepare dataset")
        return 1


if __name__ == "__main__":
    sys.exit(main())
