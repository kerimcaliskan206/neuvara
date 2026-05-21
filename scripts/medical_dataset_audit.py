#!/usr/bin/env python3
"""
Medical dataset audit CLI — Phase 7.

Orchestrates leakage detection, category distribution analysis,
grayscale shortcut risk assessment, and size distribution audit
for a medical image dataset split into train / val / test directories.

Usage
-----
python scripts/medical_dataset_audit.py \\
    --train  data/medical_v5/train \\
    --val    data/medical_v5/val \\
    --test   data/medical_v5/test \\
    --output reports/audit_report.json

With optional per-image metadata (JSON array):
python scripts/medical_dataset_audit.py \\
    --train  data/medical_v5/train \\
    --val    data/medical_v5/val \\
    --test   data/medical_v5/test \\
    --metadata-file data/medical_v5/metadata.json \\
    --output reports/audit_report.json

Metadata JSON schema (one object per image):
{
  "path":         "data/medical_v5/train/img_001.jpg",
  "category":     "healthy_xray",
  "split":        "train",
  "source_id":    "NIH_ChestXray",
  "is_grayscale": true,
  "width":        1024,
  "height":       1024
}

Exit codes
----------
  0 — audit clean (contamination_risk=none, no distribution warnings)
  1 — moderate issues (risk=low/medium or distribution warnings)
  2 — critical issues (risk=high or grayscale_shortcut_risk=high)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("medical_audit")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _hash_split(directory: Path) -> dict[Path, str]:
    """Validate all images in a directory and return {path: sha256}."""
    from app.modules.vision.medical.dataset_validator import (
        validate_dataset_dir,
        build_hash_map,
    )
    results = validate_dataset_dir(directory)
    invalid = [r for r in results if not r.valid]
    if invalid:
        logger.warning(
            "  %d invalid image(s) in %s (skipped): %s",
            len(invalid), directory,
            [str(r.path.name) for r in invalid[:5]],
        )
    hashes = build_hash_map(results)
    logger.info("  %d valid images hashed in %s", len(hashes), directory)
    return hashes


def _load_metadata(metadata_file: Path | None) -> list[dict]:
    if not metadata_file or not metadata_file.exists():
        return []
    with open(metadata_file) as f:
        data = json.load(f)
    if not isinstance(data, list):
        logger.warning("--metadata-file must be a JSON array — ignoring")
        return []
    logger.info("Loaded %d metadata records from %s", len(data), metadata_file)
    return data


def _run_leakage(
    train: dict[Path, str],
    val: dict[Path, str],
    test: dict[Path, str],
    metadata: list[dict],
) -> dict:
    from app.modules.vision.medical.leakage_detector import LeakageDetector

    detector = LeakageDetector()
    all_hashes = {**train, **val, **test}

    duplicates   = detector.detect_duplicates(all_hashes)
    cross_leaks  = detector.detect_cross_split_leakage(train, val, test)
    train_meta   = [m for m in metadata if m.get("split") == "train"]
    test_meta    = [m for m in metadata if m.get("split") == "test"]
    source_leaks = detector.detect_source_leakage(train_meta, test_meta)
    report       = detector.build_report(
        duplicates, cross_leaks, source_leaks, len(all_hashes)
    )

    return {
        "contamination_risk": report.contamination_risk,
        "summary": report.summary,
        "duplicate_groups": len(report.duplicates),
        "duplicate_image_total": sum(d.count for d in report.duplicates),
        "cross_split_leaks": len(report.cross_split_leaks),
        "source_leaks": [
            {
                "source_id": s.source_id,
                "train_count": s.train_count,
                "test_count":  s.test_count,
                "risk_level":  s.risk_level,
            }
            for s in report.source_leaks
        ],
    }


def _run_distribution(metadata: list[dict]) -> dict:
    from app.modules.vision.medical.source_auditor import SourceAuditor

    result = SourceAuditor().full_audit(metadata)
    return {
        "category_counts":        result.distribution.counts,
        "total_images":           result.distribution.total,
        "imbalance_ratio":        result.distribution.imbalance_ratio,
        "dominant_category":      result.distribution.dominant_category,
        "minority_category":      result.distribution.minority_category,
        "grayscale_shortcut_risk": result.grayscale.shortcut_risk,
        "grayscale_explanation":  result.grayscale.explanation,
        "size_unique_count":      result.size.unique_size_count,
        "mixed_sources_suspected": result.size.mixed_sources_suspected,
        "common_sizes": [
            {"width": w, "height": h, "count": c}
            for w, h, c in result.size.common_sizes
        ],
        "warnings": result.audit_warnings,
    }


def _registry_summary() -> dict:
    """Embed the category registry into the report for traceability."""
    from app.modules.vision.medical.category_registry import (
        CATEGORY_REGISTRY,
        get_trainable_disease_groups,
        get_hard_negative_categories,
    )
    return {
        "categories": [
            {
                "name":        cat.value,
                "modality":    meta.modality.value,
                "pathological": meta.is_pathological,
                "real_medical": meta.is_real_medical,
                "training_priority": meta.training_priority,
                "min_recommended_samples": meta.min_recommended_samples,
            }
            for cat, meta in CATEGORY_REGISTRY.items()
        ],
        "trainable_disease_groups": [g.value for g in get_trainable_disease_groups()],
        "hard_negative_categories": [c.value for c in get_hard_negative_categories()],
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="medical_dataset_audit",
        description="Phase 7 — medical dataset leakage + distribution audit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--train",         type=Path, required=True)
    p.add_argument("--val",           type=Path, required=True)
    p.add_argument("--test",          type=Path, required=True)
    p.add_argument("--metadata-file", type=Path, default=None,
                   help="JSON array of per-image metadata records.")
    p.add_argument("--output",        type=Path, default=Path("audit_report.json"),
                   help="Output JSON report path (default: audit_report.json).")
    p.add_argument("--skip-registry", action="store_true",
                   help="Omit category registry from the report.")
    return p


def main() -> int:
    args = build_parser().parse_args()

    for split_name, split_dir in [("train", args.train), ("val", args.val), ("test", args.test)]:
        if not split_dir.is_dir():
            logger.error("%s directory not found: %s", split_name, split_dir)
            return 2

    logger.info("=== Medical Dataset Audit — Phase 7 ===")
    logger.info("Train : %s", args.train)
    logger.info("Val   : %s", args.val)
    logger.info("Test  : %s", args.test)

    # 1. Hash all images
    logger.info("[1/4] Hashing images per split...")
    train_hashes = _hash_split(args.train)
    val_hashes   = _hash_split(args.val)
    test_hashes  = _hash_split(args.test)
    total = len(train_hashes) + len(val_hashes) + len(test_hashes)
    logger.info("Total valid images: %d", total)

    # 2. Leakage detection
    logger.info("[2/4] Running leakage detection...")
    metadata = _load_metadata(args.metadata_file)
    leakage  = _run_leakage(train_hashes, val_hashes, test_hashes, metadata)
    logger.info("  Contamination risk: %s", leakage["contamination_risk"].upper())
    for line in leakage["summary"]:
        logger.info("  → %s", line)

    # 3. Distribution audit
    distribution: dict = {}
    logger.info("[3/4] Running distribution audit...")
    if metadata:
        distribution = _run_distribution(metadata)
        for w in distribution.get("warnings", []):
            logger.warning("  ! %s", w)
    else:
        logger.info("  No --metadata-file provided — distribution audit skipped.")

    # 4. Registry snapshot
    logger.info("[4/4] Embedding category registry...")
    registry = {} if args.skip_registry else _registry_summary()

    # Build report
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "splits": {
            "train": len(train_hashes),
            "val":   len(val_hashes),
            "test":  len(test_hashes),
            "total": total,
        },
        "leakage":       leakage,
        "distribution":  distribution,
        "registry":      registry,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Report written → %s", args.output)

    # Exit code
    contamination_risk = leakage.get("contamination_risk", "none")
    grayscale_risk     = distribution.get("grayscale_shortcut_risk", "none")
    dist_warnings      = distribution.get("warnings", [])

    if contamination_risk == "high" or grayscale_risk == "high":
        logger.error("CRITICAL ISSUES DETECTED — resolve before training.")
        return 2
    if contamination_risk in ("medium", "low") or dist_warnings:
        logger.warning("Moderate issues detected — review report before training.")
        return 1

    logger.info("Audit CLEAN — no critical issues found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
