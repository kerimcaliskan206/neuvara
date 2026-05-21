#!/usr/bin/env python3
"""
Build medical dataset directory structure — Phase 8.

Creates the staged directory tree for v6 medical multiclass training,
writes per-stage manifest templates, and emits a migration checklist
that maps each v5 class to its v6 relabeling requirements.

Directories created
-------------------
<root>/
  stage_1_baseline/
    train/ val/ test/  ← 3 classes: related, unrelated, hard_negative
  stage_2_binary_medical/
    train/ val/ test/  ← 4 classes: healthy_xray, pneumonia_xray, unrelated, hard_negative
  stage_3_subtle_classes/
    train/ val/ test/  ← 6 classes: + opacity_pattern, infiltrate_pattern
  stage_4_full_specialization/
    train/ val/ test/  ← 11 classes: all medical + OOD expansion
  metadata/
    class_registry.json    ← CATEGORY_REGISTRY dump
    migration_plan.json    ← V5_TO_V6_MIGRATION dump
    stage_plan.json        ← STAGE_PLAN dump
    migration_checklist.md ← human-readable annotation task list

Usage
-----
python scripts/build_medical_dataset_structure.py \\
    --root data/medical_v6 \\
    [--dry-run]            # print paths without creating

Exit codes: 0 success, 1 error.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("build_medical")

SPLITS = ("train", "val", "test")


# ── Stage class sets ──────────────────────────────────────────────────────────


STAGE_CLASSES: dict[str, tuple[str, ...]] = {
    "stage_1_baseline": (
        "related",
        "unrelated",
        "hard_negative",
    ),
    "stage_2_binary_medical": (
        "healthy_xray",
        "pneumonia_xray",
        "unrelated",
        "hard_negative",
    ),
    "stage_3_subtle_classes": (
        "healthy_xray",
        "pneumonia_xray",
        "opacity_pattern",
        "infiltrate_pattern",
        "unrelated",
        "hard_negative",
    ),
    "stage_4_full_specialization": (
        "healthy_xray",
        "pneumonia_xray",
        "opacity_pattern",
        "infiltrate_pattern",
        "hantavirus_candidate",
        "normal_microscopy",
        "infected_microscopy",
        "unrelated",
        "hard_negative",
        "fake_medical",
        "ai_generated_medical",
    ),
}


# ── Directory builder ─────────────────────────────────────────────────────────


def _create_dirs(root: Path, dry_run: bool) -> list[Path]:
    created: list[Path] = []

    for stage_name, classes in STAGE_CLASSES.items():
        for split in SPLITS:
            for cls in classes:
                target = root / stage_name / split / cls
                if dry_run:
                    logger.info("DRY-RUN  mkdir -p %s", target)
                else:
                    target.mkdir(parents=True, exist_ok=True)
                created.append(target)

    metadata_dir = root / "metadata"
    if not dry_run:
        metadata_dir.mkdir(parents=True, exist_ok=True)

    return created


# ── Manifest templates ────────────────────────────────────────────────────────


def _stage_manifest(stage_name: str, classes: tuple[str, ...]) -> dict:
    return {
        "stage": stage_name,
        "splits": list(SPLITS),
        "classes": list(classes),
        "target_classes": [
            c for c in classes
            if c not in ("unrelated", "hard_negative", "fake_medical", "ai_generated_medical")
        ],
        "ood_classes": [
            c for c in classes
            if c in ("unrelated", "hard_negative", "fake_medical", "ai_generated_medical")
        ],
        "sample_counts": {split: {cls: 0 for cls in classes} for split in SPLITS},
        "status": "empty",
        "notes": "",
    }


def _write_metadata(root: Path, dry_run: bool) -> None:
    from app.modules.vision.medical.category_registry import (
        CATEGORY_REGISTRY, DISEASE_MAPPINGS,
        get_trainable_disease_groups, get_hard_negative_categories,
    )
    from app.modules.vision.medical.class_migration import (
        V5_TO_V6_MIGRATION, ACCEPTANCE_POLICY_BY_STAGE,
        POSITIVE_CLASSES_BY_STAGE, FORGETTING_PREVENTION,
    )
    from app.modules.vision.medical.training_plan import (
        STAGE_PLAN, OOD_PRESERVATION_POLICY, BALANCING_PLANS,
        get_all_classes_at_stage,
    )

    metadata_dir = root / "metadata"

    # class_registry.json
    registry_data = {
        cat.value: {
            "modality":           meta.modality.value,
            "is_pathological":    meta.is_pathological,
            "is_real_medical":    meta.is_real_medical,
            "semantic_group":     meta.semantic_group,
            "training_priority":  meta.training_priority,
            "min_recommended_samples": meta.min_recommended_samples,
            "hard_negative_compatible": meta.hard_negative_compatible,
            "description":        meta.description,
        }
        for cat, meta in CATEGORY_REGISTRY.items()
    }
    _write_json(metadata_dir / "class_registry.json", registry_data, dry_run)

    # migration_plan.json
    migration_data = {
        v5.value: {
            "v6_labels":           [v.value for v in mapping.v6_labels],
            "splitting_strategy":  mapping.splitting_strategy.value,
            "requires_relabeling": mapping.requires_relabeling,
            "requires_new_data":   mapping.requires_new_data,
            "migration_risk":      mapping.migration_risk.value,
            "minimum_v6_samples":  mapping.minimum_v6_samples,
            "public_dataset_sources": list(mapping.public_dataset_sources),
            "notes":               mapping.notes,
        }
        for v5, mapping in V5_TO_V6_MIGRATION.items()
    }
    _write_json(metadata_dir / "migration_plan.json", migration_data, dry_run)

    # stage_plan.json
    stage_data = {
        desc.stage.value: {
            "display_name":      desc.display_name,
            "description":       desc.description,
            "all_classes":       list(get_all_classes_at_stage(desc.stage)),
            "target_classes":    list(desc.target_classes),
            "ood_classes":       list(desc.ood_classes),
            "backbone_strategy": desc.backbone_strategy.value,
            "unfrozen_blocks":   desc.unfrozen_blocks,
            "max_epochs":        desc.max_epochs,
            "base_lr":           desc.base_lr,
            "augmentation":      desc.augmentation.value,
            "prerequisite":      desc.prerequisite.value if desc.prerequisite else None,
            "notes":             desc.notes,
            "ood_policy": {
                "hard_negative_min_fraction":       OOD_PRESERVATION_POLICY.hard_negative_min_fraction,
                "min_hard_negative_rejection_rate": OOD_PRESERVATION_POLICY.min_hard_negative_rejection_rate,
                "semantic_gate_always_active":      OOD_PRESERVATION_POLICY.semantic_gate_always_active,
            },
            "balancing": {
                "strategy":        BALANCING_PLANS[desc.stage].strategy.value,
                "target_per_class": BALANCING_PLANS[desc.stage].target_per_class,
                "hard_negative_cap": BALANCING_PLANS[desc.stage].hard_negative_cap,
            } if desc.stage in BALANCING_PLANS else {},
        }
        for desc in STAGE_PLAN
    }
    _write_json(metadata_dir / "stage_plan.json", stage_data, dry_run)

    # Per-stage manifest templates
    for stage_name, classes in STAGE_CLASSES.items():
        manifest = _stage_manifest(stage_name, classes)
        _write_json(metadata_dir / f"manifest_{stage_name}.json", manifest, dry_run)

    # Migration checklist markdown
    _write_migration_checklist(metadata_dir / "migration_checklist.md", dry_run)


def _write_migration_checklist(path: Path, dry_run: bool) -> None:
    from app.modules.vision.medical.class_migration import V5_TO_V6_MIGRATION, V5Label

    lines = [
        "# V5 → V6 Class Migration Checklist",
        "",
        "Generated by `scripts/build_medical_dataset_structure.py`.",
        "Complete each item before starting the corresponding training stage.",
        "",
    ]

    for v5_label in [V5Label.RELATED, V5Label.HARD_NEGATIVE, V5Label.UNRELATED]:
        mapping = V5_TO_V6_MIGRATION[v5_label]
        lines += [
            f"## `{v5_label.value}` → {[v.value for v in mapping.v6_labels]}",
            f"- **Risk**: {mapping.migration_risk.value}",
            f"- **Strategy**: {mapping.splitting_strategy.value}",
            f"- **Requires relabeling**: {mapping.requires_relabeling}",
            f"- **Requires new data**: {mapping.requires_new_data}",
            "",
        ]
        if mapping.minimum_v6_samples:
            lines.append("**Minimum samples needed:**")
            for cls, n in mapping.minimum_v6_samples.items():
                lines.append(f"- [ ] `{cls}`: {n} images")
            lines.append("")

        if mapping.public_dataset_sources:
            lines.append("**Recommended public sources:**")
            for src in mapping.public_dataset_sources:
                lines.append(f"- {src}")
            lines.append("")

        if mapping.notes:
            lines += [f"**Notes:** {mapping.notes}", ""]
        lines.append("---")
        lines.append("")

    content = "\n".join(lines)
    if dry_run:
        logger.info("DRY-RUN  write %s (%d chars)", path, len(content))
    else:
        path.write_text(content, encoding="utf-8")
        logger.info("Written %s", path)


def _write_json(path: Path, data: dict, dry_run: bool) -> None:
    content = json.dumps(data, indent=2, default=str)
    if dry_run:
        logger.info("DRY-RUN  write %s (%d bytes)", path, len(content))
    else:
        path.write_text(content, encoding="utf-8")
        logger.info("Written %s", path)


# ── Summary ───────────────────────────────────────────────────────────────────


def _print_summary(root: Path) -> None:
    from app.modules.vision.medical.training_plan import STAGE_PLAN
    from app.modules.vision.medical.class_migration import migration_critical_path

    print("\n=== Medical Dataset Structure ===")
    print(f"Root: {root}\n")

    for stage_name, classes in STAGE_CLASSES.items():
        n_dirs = len(SPLITS) * len(classes)
        print(f"  {stage_name:<35} {len(classes):2d} classes × {len(SPLITS)} splits = {n_dirs:3d} dirs")

    print("\n=== Migration Critical Path ===")
    for v5_label in migration_critical_path():
        from app.modules.vision.medical.class_migration import V5_TO_V6_MIGRATION
        m = V5_TO_V6_MIGRATION[v5_label]
        print(f"  [{m.migration_risk.value.upper():8s}] {v5_label.value} "
              f"→ {len(m.v6_labels)} v6 classes  "
              f"(relabel={m.requires_relabeling}, new_data={m.requires_new_data})")

    print("\n=== Stage Prerequisites ===")
    for desc in STAGE_PLAN:
        prereq = desc.prerequisite.value if desc.prerequisite else "none"
        print(f"  {desc.stage.value:<35} requires: {prereq}")

    print()


# ── Entry point ───────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_medical_dataset_structure",
        description="Create v6 medical dataset directory tree + metadata.",
    )
    p.add_argument("--root",    type=Path, default=Path("data/medical_v6"),
                   help="Root directory for the dataset (default: data/medical_v6).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print paths without creating files or directories.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    logger.info("=== Build Medical Dataset Structure — Phase 8 ===")
    logger.info("Root    : %s", args.root)
    logger.info("Dry-run : %s", args.dry_run)

    try:
        dirs = _create_dirs(args.root, args.dry_run)
        logger.info("Directories: %d planned", len(dirs))

        _write_metadata(args.root, args.dry_run)

        _print_summary(args.root)

        if args.dry_run:
            logger.info("DRY-RUN complete — no files written.")
        else:
            logger.info("Done. Populate class directories with images, then run:")
            logger.info("  python scripts/medical_dataset_audit.py --train %s/stage_2_binary_medical/train ...",
                        args.root)

        return 0
    except Exception:
        logger.exception("Failed to build dataset structure")
        return 1


if __name__ == "__main__":
    sys.exit(main())
