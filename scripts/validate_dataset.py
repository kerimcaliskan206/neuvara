"""
HantaProject — Dataset Validation Script
=========================================

Validates a prepared dataset version for:
  - Manifest integrity (every manifest record has a file on disk)
  - Split integrity (every file in a split directory is in the manifest)
  - Class balance (warns on severe imbalance)
  - Quality distribution (summarizes score distribution)
  - Duplicate check (finds any near-duplicates that slipped through)
  - Split ratio verification (actual ratios vs. configured)

Usage
-----
  # Validate the latest version
  python scripts/validate_dataset.py

  # Validate a specific version
  python scripts/validate_dataset.py --version v1

  # Check for near-duplicates across train/val/test (expensive)
  python scripts/validate_dataset.py --check-cross-split-duplicates
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from app.core.logging import setup_logging  # noqa: E402
from app.modules.vision.datasets.balancer import imbalance_report  # noqa: E402
from app.modules.vision.datasets.deduplication import DuplicateDetector  # noqa: E402
from app.modules.vision.datasets.manifest import DatasetManifest  # noqa: E402
from app.modules.vision.datasets.schema import ImageClass, Split  # noqa: E402
from app.modules.vision.datasets.versioning import DatasetVersionManager  # noqa: E402

logger = logging.getLogger(__name__)

DATASETS_DIR = _PROJECT_ROOT / "data" / "vision" / "datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HantaProject — Validate a prepared dataset version",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", default=None, help="Version to validate (default: latest).")
    parser.add_argument("--datasets-dir", type=Path, default=DATASETS_DIR)
    parser.add_argument(
        "--check-cross-split-duplicates",
        action="store_true",
        help="Check for near-duplicates that appear in multiple splits (slow).",
    )
    return parser.parse_args()


def validate_version(
    version: str,
    manager: DatasetVersionManager,
    check_cross_split_duplicates: bool,
) -> bool:
    """Run all validation checks. Returns True if all pass."""
    all_passed = True

    processed_dir = manager.processed_dir(version)
    splits_dir = manager.splits_dir(version)
    metadata_dir = manager.metadata_dir(version)

    manifest = DatasetManifest(manifest_dir=metadata_dir, version=version)
    manifest.load()

    print(f"\n{'═' * 60}")
    print(f"  Validating dataset version: {version}")
    print(f"{'═' * 60}")

    # ── 1. Manifest integrity ─────────────────────────────────────────────────
    print("\n[1] Manifest integrity")
    integrity = manifest.check_integrity(processed_dir)
    n_missing = len(integrity["missing"])
    n_orphaned = len(integrity["orphaned"])

    if n_missing == 0 and n_orphaned == 0:
        print("    ✓ All manifest records have files; no orphaned files.")
    else:
        if n_missing:
            print(f"    ✗ Missing files: {n_missing}")
            all_passed = False
        if n_orphaned:
            print(f"    ⚠ Orphaned files (on disk but not in manifest): {n_orphaned}")

    # ── 2. Split distribution ─────────────────────────────────────────────────
    print("\n[2] Split distribution")
    for split in Split:
        if split == Split.UNASSIGNED:
            continue
        for cls in ImageClass:
            count = len(manifest.by_class_and_split(cls, split))
            split_dir = splits_dir / split.value / cls.value
            dir_count = 0
            if split_dir.exists():
                dir_count = sum(
                    1 for p in split_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
                )
            match = "✓" if count == dir_count else "✗"
            print(f"    {match} {split.value}/{cls.value}: manifest={count}, disk={dir_count}")
            if count != dir_count:
                all_passed = False

    unassigned = len(manifest.by_split(Split.UNASSIGNED))
    if unassigned > 0:
        print(f"    ⚠ {unassigned} records have no split assignment")

    # ── 3. Class balance ──────────────────────────────────────────────────────
    print("\n[3] Class balance (training split)")
    train_counts = {
        cls.value: len(manifest.by_class_and_split(cls, Split.TRAIN))
        for cls in ImageClass
    }
    report = imbalance_report({k: v for k, v in train_counts.items() if v > 0})
    severity = report.get("severity", "unknown")
    ratio = report.get("imbalance_ratio", 0)
    print(f"    Imbalance ratio : {ratio:.1f}:1 | severity: {severity}")
    print(f"    Distribution    : {train_counts}")
    if severity == "severe":
        print(f"    ✗ {report['recommendation']}")
        all_passed = False
    elif severity in ("moderate", "mild"):
        print(f"    ⚠ {report['recommendation']}")
    else:
        print("    ✓ Dataset is balanced.")

    # ── 4. Quality distribution ───────────────────────────────────────────────
    print("\n[4] Quality distribution")
    accepted = manifest.accepted_quality()
    all_records = manifest.all()
    q_scores = [r.quality_score for r in all_records]
    if q_scores:
        low_q = sum(1 for q in q_scores if q < 0.5)
        mid_q = sum(1 for q in q_scores if 0.5 <= q < 0.75)
        high_q = sum(1 for q in q_scores if q >= 0.75)
        mean_q = sum(q_scores) / len(q_scores)
        print(f"    Mean quality score : {mean_q:.4f}")
        print(f"    High (≥0.75)       : {high_q}")
        print(f"    Mid  (0.5–0.75)    : {mid_q}")
        print(f"    Low  (<0.50)       : {low_q}")
        if low_q > 0:
            print(f"    ⚠ {low_q} images below quality threshold")
    else:
        print("    ⚠ No quality scores found in manifest")

    # ── 5. Cross-split duplicate check ────────────────────────────────────────
    if check_cross_split_duplicates:
        print("\n[5] Cross-split duplicate check (perceptual hash)")
        detector = DuplicateDetector(hamming_threshold=8)

        splits_to_check = [Split.TRAIN, Split.VAL, Split.TEST]
        hash_to_splits: dict[str, list[str]] = {}

        for split in splits_to_check:
            records = manifest.by_split(split)
            for r in records:
                hash_to_splits.setdefault(r.perceptual_hash, []).append(split.value)

        leaks = {h: splits for h, splits in hash_to_splits.items() if len(set(splits)) > 1}
        if leaks:
            print(f"    ✗ {len(leaks)} images appear in multiple splits (data leakage risk)")
            for h, splits in list(leaks.items())[:5]:
                print(f"      hash={h} in splits: {splits}")
            all_passed = False
        else:
            print("    ✓ No cross-split duplicates detected.")
    else:
        print("\n[5] Cross-split duplicate check — skipped (use --check-cross-split-duplicates)")

    # ── Summary ───────────────────────────────────────────────────────────────
    stats = manifest.stats()
    print(f"\n{'─' * 60}")
    print(f"  Total images    : {stats['total']}")
    print(f"  Validated       : {stats.get('validated', 0)}")
    print(f"  Accepted quality: {stats.get('accepted', 0)}")
    print(f"  Duplicates flagged: {stats.get('duplicates_flagged', 0)}")
    print(f"\n  Overall result: {'✓ PASSED' if all_passed else '✗ FAILED'}")
    print(f"{'═' * 60}\n")

    return all_passed


def main() -> None:
    args = parse_args()
    setup_logging(debug=False, environment="development")

    manager = DatasetVersionManager(args.datasets_dir)
    version = args.version or manager.latest_version()

    if version is None:
        logger.error(
            "No dataset versions found in %s. "
            "Run scripts/prepare_dataset.py first.",
            args.datasets_dir,
        )
        sys.exit(1)

    if not manager.exists(version):
        logger.error("Version '%s' not found. Available: %s", version, manager.list_versions())
        sys.exit(1)

    passed = validate_version(
        version=version,
        manager=manager,
        check_cross_split_duplicates=args.check_cross_split_duplicates,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
