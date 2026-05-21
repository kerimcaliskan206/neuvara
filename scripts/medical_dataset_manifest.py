#!/usr/bin/env python3
"""
Medical dataset manifest generator — Phase 9.

Scans data/medical_ready/ (populated by normalize_medical_dataset.py),
reads the .normalization_cache.json, and produces a comprehensive
audit-ready manifest covering:

  - per-image metadata (hash, source, label, dimensions, trainability)
  - class distribution
  - source distribution
  - grayscale consistency per class
  - duplicate detection across the full ready directory
  - split-readiness scoring (suggests train/val/test split sizes)
  - leakage-safe manifest (SHA-256 hashes for future cross-dataset checks)

Manifest format
---------------
data/medical_ready/
  manifest.json          ← full manifest (all images)
  manifest_<class>.json  ← per-class sub-manifests
  split_plan.json        ← suggested train/val/test split (no actual copy)

Usage
-----
python scripts/medical_dataset_manifest.py \\
    --ready-dir data/medical_ready \\
    --output data/medical_ready/manifest.json

# Append new images since last manifest run:
python scripts/medical_dataset_manifest.py \\
    --ready-dir data/medical_ready \\
    --output data/medical_ready/manifest.json \\
    --incremental

# Only generate split plan (no full re-scan):
python scripts/medical_dataset_manifest.py \\
    --ready-dir data/medical_ready \\
    --split-only \\
    --val-fraction 0.10 \\
    --test-fraction 0.15
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("manifest")

_CACHE_FILE: str = ".normalization_cache.json"
_SUPPORTED: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"})
_OUTPUT_CLASSES: tuple[str, ...] = (
    "healthy_xray", "pneumonia_xray", "uncertain_xray",
    "opacity_pattern", "infiltrate_pattern", "hantavirus_candidate",
    "normal_microscopy", "infected_microscopy",
    "hard_negative", "unrelated", "fake_medical", "ai_generated_medical",
)

# Classes that should predominantly be grayscale (for shortcut risk check)
_EXPECTED_GRAYSCALE_CLASSES: frozenset[str] = frozenset({
    "healthy_xray", "pneumonia_xray", "uncertain_xray",
    "opacity_pattern", "infiltrate_pattern", "hantavirus_candidate",
})


# ── Data types ────────────────────────────────────────────────────────────────


@dataclass
class ManifestEntry:
    """Single image record in the manifest."""
    image_id: str               # <source_id>_<hash8>_<idx>
    path: str                   # relative to ready_dir
    sha256: str
    v6_label: str
    source_id: str              # derived from filename prefix
    original_label: str | None  # from normalization cache (if available)
    width: int
    height: int
    is_grayscale: bool
    trainability_score: float
    quality_flags: list[str]


@dataclass
class ClassStats:
    label: str
    count: int
    grayscale_count: int
    mean_trainability: float
    sources: dict[str, int]          # source_id → count
    grayscale_fraction: float = 0.0


@dataclass
class ManifestSummary:
    generated_at: str
    ready_dir: str
    total_images: int
    class_distribution: dict[str, int]
    source_distribution: dict[str, int]
    grayscale_shortcut_risk: str    # "none" | "low" | "medium" | "high"
    duplicate_groups: int
    duplicate_images: int
    mean_trainability: float
    min_samples_for_training: dict[str, str]   # label → "ok" | "insufficient:<N>"
    warnings: list[str]


# ── Image scanner ─────────────────────────────────────────────────────────────


def _source_id_from_filename(stem: str) -> str:
    """Extract source_id prefix from normalized filename like 'kermany_3f8a1b2c_00001'."""
    parts = stem.split("_")
    return parts[0] if parts else "unknown"


def _scan_ready_dir(
    ready_dir: Path,
    cache: dict[str, str],
) -> list[ManifestEntry]:
    """Scan ready_dir for all normalized images and build ManifestEntry list."""
    from app.modules.vision.medical.dataset_validator import compute_sha256

    # Invert cache: output_path → sha256
    path_to_sha: dict[str, str] = {v: k for k, v in cache.items()}

    entries: list[ManifestEntry] = []
    seen_hashes: dict[str, list[str]] = defaultdict(list)

    for v6_label in _OUTPUT_CLASSES:
        class_dir = ready_dir / v6_label
        if not class_dir.is_dir():
            continue

        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() not in _SUPPORTED:
                continue

            rel = str(img_path.relative_to(ready_dir))

            # Get SHA-256 from cache (fast) or compute it (slow)
            sha256 = path_to_sha.get(rel) or path_to_sha.get(str(img_path))
            if not sha256:
                try:
                    sha256 = compute_sha256(img_path)
                except Exception:
                    logger.warning("Could not hash %s — skipping", img_path.name)
                    continue

            seen_hashes[sha256].append(rel)

            # Dimensions and grayscale from PIL (lightweight — no pixel decode)
            try:
                from PIL import Image
                img = Image.open(img_path)
                w, h = img.size
                is_gray = img.mode in ("L", "LA")
            except Exception:
                w, h, is_gray = 0, 0, False

            # Trainability score
            from scripts.normalize_medical_dataset import _trainability_score
            t_score, flags = _trainability_score(w, h, is_gray, v6_label)

            source_id = _source_id_from_filename(img_path.stem)

            entries.append(ManifestEntry(
                image_id=img_path.stem,
                path=rel,
                sha256=sha256,
                v6_label=v6_label,
                source_id=source_id,
                original_label=None,
                width=w,
                height=h,
                is_grayscale=is_gray,
                trainability_score=t_score,
                quality_flags=flags,
            ))

    logger.info("Scanned %d images across %d classes", len(entries),
                len({e.v6_label for e in entries}))
    return entries, seen_hashes


# ── Statistics ────────────────────────────────────────────────────────────────


def _compute_class_stats(entries: list[ManifestEntry]) -> dict[str, ClassStats]:
    stats: dict[str, ClassStats] = {}
    groups: dict[str, list[ManifestEntry]] = defaultdict(list)
    for e in entries:
        groups[e.v6_label].append(e)

    for label, group in groups.items():
        gray_n = sum(1 for e in group if e.is_grayscale)
        mean_t = sum(e.trainability_score for e in group) / max(len(group), 1)
        src_counts: dict[str, int] = defaultdict(int)
        for e in group:
            src_counts[e.source_id] += 1

        stats[label] = ClassStats(
            label=label,
            count=len(group),
            grayscale_count=gray_n,
            mean_trainability=round(mean_t, 4),
            sources=dict(src_counts),
            grayscale_fraction=round(gray_n / max(len(group), 1), 4),
        )
    return stats


def _grayscale_shortcut_risk(class_stats: dict[str, ClassStats]) -> tuple[str, list[str]]:
    """Check if grayscale distribution creates a colorspace shortcut."""
    warnings: list[str] = []
    threshold = 0.85

    high_gray = [
        lbl for lbl, s in class_stats.items()
        if s.grayscale_fraction >= threshold
    ]
    low_gray = [
        lbl for lbl, s in class_stats.items()
        if s.grayscale_fraction <= (1 - threshold)
    ]
    at_risk = [(hg, lg) for hg in high_gray for lg in low_gray]

    if at_risk:
        risk = "high"
        for hg, lg in at_risk:
            warnings.append(
                f"Grayscale shortcut risk: '{hg}' ({class_stats[hg].grayscale_fraction:.0%} gray) "
                f"vs '{lg}' ({class_stats[lg].grayscale_fraction:.0%} gray)."
            )
    elif any(s.grayscale_fraction > 0.65 for s in class_stats.values()) and \
         any(s.grayscale_fraction < 0.35 for s in class_stats.values()):
        risk = "medium"
        warnings.append("Moderate grayscale distribution difference across classes.")
    else:
        risk = "none"

    return risk, warnings


def _min_samples_check(
    class_stats: dict[str, ClassStats],
) -> dict[str, str]:
    """Check whether each class meets the Phase 8 minimum sample requirements."""
    from app.modules.vision.medical.training_plan import BALANCING_PLANS, TrainingStage

    stage4 = BALANCING_PLANS[TrainingStage.STAGE_4_FULL_SPECIALIZATION]
    result: dict[str, str] = {}
    for label, stats in class_stats.items():
        required = stage4.target_per_class.get(label, 0)
        if required == 0:
            result[label] = "ok (no minimum set)"
        elif stats.count >= required:
            result[label] = f"ok ({stats.count}/{required})"
        else:
            result[label] = f"insufficient:{stats.count}/{required}_needed"
    return result


# ── Split planner ─────────────────────────────────────────────────────────────


def _build_split_plan(
    entries: list[ManifestEntry],
    val_fraction: float,
    test_fraction: float,
    seed: int = 42,
) -> dict:
    """
    Produce a stratified split plan without actually moving files.

    Returns a dict mapping image_id → "train" | "val" | "test".
    Stratified by v6_label to preserve class ratios.
    """
    rng = random.Random(seed)
    groups: dict[str, list[str]] = defaultdict(list)
    for e in entries:
        groups[e.v6_label].append(e.image_id)

    assignments: dict[str, str] = {}
    split_counts: dict[str, dict[str, int]] = {}

    for label, ids in groups.items():
        shuffled = list(ids)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, round(n * test_fraction))
        n_val  = max(1, round(n * val_fraction))
        n_train = n - n_val - n_test

        for iid in shuffled[:n_train]:
            assignments[iid] = "train"
        for iid in shuffled[n_train:n_train + n_val]:
            assignments[iid] = "val"
        for iid in shuffled[n_train + n_val:]:
            assignments[iid] = "test"

        split_counts[label] = {
            "train": n_train,
            "val":   n_val,
            "test":  n_test,
            "total": n,
        }

    return {
        "strategy": "stratified_by_class",
        "val_fraction":  val_fraction,
        "test_fraction": test_fraction,
        "seed": seed,
        "class_split_counts": split_counts,
        "assignments": assignments,
        "note": (
            "This is a PLAN only — no files are moved. "
            "Pass this file to the training script to honour the split boundaries."
        ),
    }


# ── Manifest writer ───────────────────────────────────────────────────────────


def build_manifest(
    ready_dir: Path,
    output_path: Path,
    val_fraction: float = 0.10,
    test_fraction: float = 0.15,
    seed: int = 42,
    split_only: bool = False,
) -> int:
    cache_file = ready_dir / _CACHE_FILE
    cache: dict[str, str] = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
        except Exception:
            logger.warning("Could not parse normalization cache — recomputing all hashes")

    if split_only:
        if not output_path.exists():
            logger.error("--split-only requires an existing manifest at %s", output_path)
            return 2
        existing = json.loads(output_path.read_text())
        entries = [ManifestEntry(**e) for e in existing.get("images", [])]
    else:
        entries, seen_hashes = _scan_ready_dir(ready_dir, cache)

    class_stats = _compute_class_stats(entries)
    source_dist: dict[str, int] = defaultdict(int)
    for e in entries:
        source_dist[e.source_id] += 1

    gray_risk, gray_warnings = _grayscale_shortcut_risk(class_stats)
    min_samples = _min_samples_check(class_stats)

    # Duplicates
    if not split_only:
        dup_groups = sum(1 for paths in seen_hashes.values() if len(paths) > 1)
        dup_images = sum(len(paths) for paths in seen_hashes.values() if len(paths) > 1)
    else:
        dup_groups = dup_images = 0

    mean_t = sum(e.trainability_score for e in entries) / max(len(entries), 1)

    all_warnings: list[str] = list(gray_warnings)
    insufficient = [lbl for lbl, s in min_samples.items() if "insufficient" in s]
    if insufficient:
        all_warnings.append(f"Insufficient samples for: {insufficient}")
    if dup_groups:
        all_warnings.append(f"{dup_groups} duplicate groups ({dup_images} images) found.")

    summary = ManifestSummary(
        generated_at=datetime.now(timezone.utc).isoformat(),
        ready_dir=str(ready_dir),
        total_images=len(entries),
        class_distribution={s.label: s.count for s in class_stats.values()},
        source_distribution=dict(source_dist),
        grayscale_shortcut_risk=gray_risk,
        duplicate_groups=dup_groups if not split_only else 0,
        duplicate_images=dup_images if not split_only else 0,
        mean_trainability=round(mean_t, 4),
        min_samples_for_training=min_samples,
        warnings=all_warnings,
    )

    # Split plan
    split_plan = _build_split_plan(entries, val_fraction, test_fraction, seed)

    # Full manifest
    manifest = {
        **asdict(summary),
        "images": [asdict(e) for e in entries],
        "class_stats": {
            lbl: asdict(s) for lbl, s in class_stats.items()
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Manifest: %s  (%d images)", output_path, len(entries))

    # Per-class sub-manifests
    for label, group_entries in defaultdict(list, {
        e.v6_label: [] for e in entries
    }).items():
        pass  # populated below

    label_groups: dict[str, list[ManifestEntry]] = defaultdict(list)
    for e in entries:
        label_groups[e.v6_label].append(e)

    for label, group in label_groups.items():
        sub_path = output_path.parent / f"manifest_{label}.json"
        sub_path.write_text(json.dumps(
            {"v6_label": label, "count": len(group), "images": [asdict(e) for e in group]},
            indent=2
        ))

    # Split plan
    split_path = output_path.parent / "split_plan.json"
    split_path.write_text(json.dumps(split_plan, indent=2))
    logger.info("Split plan: %s", split_path)

    # Print summary
    print(f"\n=== Manifest Summary ===")
    print(f"Total images : {summary.total_images:,}")
    print(f"Gray risk    : {summary.grayscale_shortcut_risk}")
    print(f"Duplicates   : {summary.duplicate_groups} groups")
    print(f"Mean quality : {summary.mean_trainability:.3f}")
    print(f"\nClass distribution:")
    for lbl, cnt in sorted(summary.class_distribution.items(), key=lambda x: -x[1]):
        status = min_samples.get(lbl, "")
        flag = " ⚠" if "insufficient" in status else ""
        print(f"  {lbl:<25} {cnt:>5}{flag}")
    if all_warnings:
        print(f"\nWarnings:")
        for w in all_warnings:
            print(f"  ! {w}")
    print()

    return 1 if all_warnings else 0


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="medical_dataset_manifest",
        description="Generate audit-ready manifest for data/medical_ready/.",
    )
    p.add_argument("--ready-dir",       type=Path, default=Path("data/medical_ready"),
                   help="Normalized dataset directory (default: data/medical_ready).")
    p.add_argument("--output",          type=Path, default=Path("data/medical_ready/manifest.json"),
                   help="Output manifest path (default: data/medical_ready/manifest.json).")
    p.add_argument("--val-fraction",    type=float, default=0.10,
                   help="Validation fraction for split plan (default: 0.10).")
    p.add_argument("--test-fraction",   type=float, default=0.15,
                   help="Test fraction for split plan (default: 0.15).")
    p.add_argument("--seed",            type=int,   default=42,
                   help="Random seed for split plan (default: 42).")
    p.add_argument("--split-only",      action="store_true",
                   help="Only regenerate split plan from existing manifest (no re-scan).")
    return p


def main() -> int:
    args = build_parser().parse_args()
    return build_manifest(
        ready_dir=args.ready_dir,
        output_path=args.output,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        split_only=args.split_only,
    )


if __name__ == "__main__":
    sys.exit(main())
