#!/usr/bin/env python3
"""
Stage medical dataset — Phase 11.

Master orchestrator for the first real medical dataset ingestion:
healthy_xray vs pneumonia_xray (Kermany + optional RSNA).

Pipeline steps
--------------
  1. Validate raw source directories
  2. Normalize images per source with label discovery
  3. Build SHA-256 manifest + stratified split plan
  4. Hash-based leakage detection (cross-split + duplicates)
  5. Build v6 training split directories (symlinks)
  6. Export 4 structured reports

SAFETY
------
  - No training is performed.
  - V5 production model and inference pipeline are not touched.
  - All output is isolated under data/medical_ready/ and data/medical_v6_splits/.

Expected raw sources
--------------------
  data/medical_raw/
    kermany/             ← Kaggle chest X-ray 2017 (NORMAL/ PNEUMONIA/ dirs)
    rsna/                ← RSNA Pneumonia Detection (optional, .dcm + CSV)
    nih/                 ← NIH ChestX-ray14 (optional, images/ + labels CSV)
    hard_negative/       ← v5 OOD samples for replay buffer (optional)

Usage
-----
    # Full ingestion from Kermany only
    python scripts/stage_medical_dataset.py \\
        --raw-dir data/medical_raw \\
        --ready-dir data/medical_ready \\
        --splits-dir data/medical_v6_splits \\
        --reports-dir data/medical_reports \\
        --sources kermany

    # Include RSNA (requires stage_2_train_labels.csv)
    python scripts/stage_medical_dataset.py ... --sources kermany rsna

    # Dry run — validate structure without writing
    python scripts/stage_medical_dataset.py ... --dry-run

Exit codes: 0 clean, 1 warnings, 2 critical (leakage or missing data).
"""
from __future__ import annotations

import argparse
import csv as csv_module
import importlib.util
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("stage_medical")

_SUPPORTED_EXT   = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"})
_DICOM_EXT       = ".dcm"
_ALL_IMG_EXT     = _SUPPORTED_EXT | {_DICOM_EXT}


# ── Source configs ────────────────────────────────────────────────────────────


SOURCE_CONFIGS: dict[str, dict] = {
    "kermany": {
        "display_name": "Kermany Chest X-Ray 2017 (Kaggle)",
        "label_mapping": {"NORMAL": "healthy_xray", "PNEUMONIA": "pneumonia_xray"},
        "mode": "directory",       # discovers by ancestor directory name
        "min_images": {"healthy_xray": 400, "pneumonia_xray": 400},
    },
    "rsna": {
        "display_name": "RSNA Pneumonia Detection Challenge",
        "label_mapping": {"0": "healthy_xray", "1": "pneumonia_xray"},
        "mode": "csv",
        "csv_filename": "stage_2_train_labels.csv",
        "csv_id_col": "patientId",
        "csv_label_col": "Target",
        "min_images": {"healthy_xray": 400, "pneumonia_xray": 400},
    },
    "nih": {
        "display_name": "NIH ChestX-ray14",
        "label_mapping": {"No Finding": "healthy_xray", "Pneumonia": "pneumonia_xray"},
        "mode": "csv",
        "csv_filename": "Data_Entry_2017.csv",
        "csv_id_col": "Image Index",
        "csv_label_col": "Finding Labels",
        "min_images": {"healthy_xray": 400, "pneumonia_xray": 400},
    },
    "hard_negative": {
        "display_name": "Hard Negative OOD Pool (v5 replay buffer)",
        "label_mapping": {"hard_negative": "hard_negative"},
        "mode": "directory",
        "min_images": {"hard_negative": 100},
    },
    "fake_medical": {
        "display_name": "Fake Medical OOD Pool (non-medical images that could fool a naive classifier)",
        "label_mapping": {"fake_medical": "fake_medical"},
        "mode": "directory",
        "min_images": {"fake_medical": 50},
    },
}


# ── Script imports (normalize + manifest are scripts, not installable modules) ─


def _import_script(name: str, script_path: Path):
    spec = importlib.util.spec_from_file_location(name, script_path)
    mod  = importlib.util.module_from_spec(spec)
    # Register in sys.modules before exec so @dataclass can resolve __module__.
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


def _load_normalize_mod():
    return _import_script(
        "normalize_medical_dataset",
        _PROJECT_ROOT / "scripts" / "normalize_medical_dataset.py",
    )


def _load_manifest_mod():
    return _import_script(
        "medical_dataset_manifest",
        _PROJECT_ROOT / "scripts" / "medical_dataset_manifest.py",
    )


# ── Image discovery ───────────────────────────────────────────────────────────


def _discover_by_ancestor(
    raw_dir: Path,
    label_mapping: dict[str, str],
) -> list[tuple[Path, str, str]]:
    """
    Discover images by finding the nearest ancestor directory whose name is in label_mapping.

    Handles both:
      Flat   : raw_dir/NORMAL/img.jpg
      Nested : raw_dir/chest_xray/train/NORMAL/img.jpg
    """
    label_dirs = set(label_mapping.keys())
    found: list[tuple[Path, str, str]] = []
    seen: set[Path] = set()

    for img_path in sorted(raw_dir.rglob("*")):
        if not img_path.is_file() or img_path in seen:
            continue
        if img_path.suffix.lower() not in _ALL_IMG_EXT:
            continue
        for parent in img_path.parents:
            if parent == raw_dir:
                break
            if parent.name in label_dirs:
                raw_lbl = parent.name
                found.append((img_path, raw_lbl, label_mapping[raw_lbl]))
                seen.add(img_path)
                break

    return found


def _discover_rsna_csv(
    raw_dir: Path,
    csv_path: Path,
    id_col: str,
    label_col: str,
    label_mapping: dict[str, str],
) -> list[tuple[Path, str, str]]:
    """
    Discover RSNA DICOM images using stage_2_train_labels.csv.

    Takes max(Target) per patientId (binary: 0=normal, 1=pneumonia).
    """
    if not csv_path.exists():
        logger.warning("RSNA CSV not found: %s", csv_path)
        return []

    patient_labels: dict[str, int] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv_module.DictReader(f):
            pid = row.get(id_col, "").strip()
            try:
                target = int(float(row.get(label_col, "0") or "0"))
            except (ValueError, TypeError):
                target = 0
            if pid:
                patient_labels[pid] = max(patient_labels.get(pid, 0), target)

    logger.info("RSNA CSV: %d patients labeled (healthy=%d, pneumonia=%d)",
                len(patient_labels),
                sum(1 for v in patient_labels.values() if v == 0),
                sum(1 for v in patient_labels.values() if v == 1))

    found: list[tuple[Path, str, str]] = []
    for ext in (_DICOM_EXT, ".jpg", ".jpeg"):
        for img_path in sorted(raw_dir.rglob(f"*{ext}")):
            pid = img_path.stem
            if pid not in patient_labels:
                continue
            raw_lbl = str(patient_labels[pid])
            v6      = label_mapping.get(raw_lbl, "uncertain_xray")
            found.append((img_path, raw_lbl, v6))

    return found


def _discover_nih_csv(
    raw_dir: Path,
    csv_path: Path,
    id_col: str,
    label_col: str,
    label_mapping: dict[str, str],
) -> list[tuple[Path, str, str]]:
    """
    Discover NIH ChestX-ray14 images.

    Finding Labels is pipe-separated: "Pneumonia|Effusion". We use only images
    where the finding is a single label matching the label_mapping.
    """
    if not csv_path.exists():
        logger.warning("NIH CSV not found: %s", csv_path)
        return []

    img_to_label: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv_module.DictReader(f):
            img_name = row.get(id_col, "").strip()
            findings = row.get(label_col, "").strip()
            # Only use images with a single finding that we can map
            if "|" not in findings and findings in label_mapping:
                img_to_label[img_name] = findings

    logger.info("NIH CSV: %d usable single-label images", len(img_to_label))

    images_dir = raw_dir / "images"
    search_dir = images_dir if images_dir.is_dir() else raw_dir

    found: list[tuple[Path, str, str]] = []
    for img_path in sorted(search_dir.rglob("*.png")):
        raw_lbl = img_to_label.get(img_path.name)
        if raw_lbl is None:
            continue
        v6 = label_mapping.get(raw_lbl, "uncertain_xray")
        found.append((img_path, raw_lbl, v6))

    return found


# ── Normalization ─────────────────────────────────────────────────────────────


def _run_source_normalization(
    source_id: str,
    source_cfg: dict,
    raw_source_dir: Path,
    out_dir: Path,
    dry_run: bool,
) -> dict:
    """
    Normalize all images for one source. Returns a per-source stats dict.

    Uses NormalizationPipeline._process_one() with custom discovery so that
    nested source structures (Kermany chest_xray/train/NORMAL/) are handled.
    """
    norm_mod = _load_normalize_mod()
    NormalizationConfig  = norm_mod.NormalizationConfig
    NormalizationPipeline = norm_mod.NormalizationPipeline
    _load_cache = norm_mod._load_cache
    _save_cache = norm_mod._save_cache

    mode          = source_cfg.get("mode", "directory")
    label_mapping = source_cfg["label_mapping"]

    # Discover images
    if mode == "csv" and source_id == "rsna":
        csv_path = raw_source_dir / source_cfg["csv_filename"]
        images   = _discover_rsna_csv(
            raw_source_dir, csv_path,
            source_cfg["csv_id_col"], source_cfg["csv_label_col"],
            label_mapping,
        )
    elif mode == "csv" and source_id == "nih":
        csv_path = raw_source_dir / source_cfg["csv_filename"]
        images   = _discover_nih_csv(
            raw_source_dir, csv_path,
            source_cfg["csv_id_col"], source_cfg["csv_label_col"],
            label_mapping,
        )
    else:
        images = _discover_by_ancestor(raw_source_dir, label_mapping)

    if not images:
        logger.warning("No images discovered for source '%s' in %s", source_id, raw_source_dir)
        return {
            "source_id": source_id,
            "raw_dir": str(raw_source_dir),
            "total_discovered": 0,
            "processed_ok": 0,
            "skipped_existing": 0,
            "rejected_quality": 0,
            "skipped_duplicate": 0,
            "errors": 0,
            "per_label": {},
        }

    logger.info("Source '%s': %d images discovered", source_id, len(images))

    cfg      = NormalizationConfig()
    pipeline = NormalizationPipeline(cfg)

    if not dry_run:
        v6_classes = {v6 for _, _, v6 in images}
        for cls in v6_classes:
            (out_dir / cls).mkdir(parents=True, exist_ok=True)

    cache      = norm_mod._load_cache(out_dir) if not dry_run else {}
    counters   = defaultdict(int)
    global_idx = sum(1 for p in out_dir.rglob("*.jpg")) if not dry_run else 0
    per_label: dict[str, dict] = defaultdict(lambda: {"count": 0, "v6_label": ""})

    for img_path, raw_lbl, v6_label in images:
        try:
            rec = pipeline._process_one(
                img_path=img_path,
                raw_lbl=raw_lbl,
                v6_label=v6_label,
                source_id=source_id,
                out_dir=out_dir,
                cache=cache,
                idx=global_idx,
                dry_run=dry_run,
            )
        except Exception as exc:
            logger.debug("Error processing %s: %s", img_path, exc)
            counters["error"] += 1
            continue

        status = rec.status
        counters[status] += 1

        if status in ("ok", "existing"):
            per_label[raw_lbl]["count"]    += 1
            per_label[raw_lbl]["v6_label"]  = v6_label
            global_idx += 1
            if not dry_run and rec.sha256 and rec.output_path:
                cache[rec.sha256] = rec.output_path

    if not dry_run:
        norm_mod._save_cache(out_dir, cache)

    logger.info(
        "Source '%s' done: ok=%d existing=%d rejected=%d dup=%d err=%d",
        source_id,
        counters["ok"], counters["existing"],
        counters["rejected"], counters["duplicate"], counters["error"],
    )

    return {
        "source_id": source_id,
        "raw_dir": str(raw_source_dir),
        "total_discovered": len(images),
        "processed_ok": counters["ok"],
        "skipped_existing": counters["existing"],
        "rejected_quality": counters["rejected"],
        "skipped_duplicate": counters["duplicate"],
        "errors": counters["error"],
        "per_label": {k: dict(v) for k, v in per_label.items()},
    }


# ── Manifest + split ──────────────────────────────────────────────────────────


def _build_manifest(
    ready_dir: Path,
    manifest_dir: Path,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    dry_run: bool,
) -> tuple[dict, dict[str, str]]:
    """
    Build manifest.json + split_plan.json. Returns (manifest_dict, split_plan).
    """
    if dry_run:
        logger.info("DRY-RUN: skipping manifest build")
        return {}, {}

    manifest_mod  = _load_manifest_mod()
    manifest_path = manifest_dir / "manifest.json"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    rc = manifest_mod.build_manifest(
        ready_dir=ready_dir,
        output_path=manifest_path,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    if rc not in (0, 1):
        logger.warning("build_manifest returned rc=%d", rc)

    manifest_data: dict = {}
    split_plan:    dict = {}

    if manifest_path.exists():
        manifest_data = json.loads(manifest_path.read_text())

    split_path = manifest_dir / "split_plan.json"
    if split_path.exists():
        split_plan = json.loads(split_path.read_text())

    return manifest_data, split_plan


# ── V6 split directories ──────────────────────────────────────────────────────


def _build_v6_splits(
    ready_dir: Path,
    manifest_data: dict,
    split_plan: dict[str, str],
    splits_dir: Path,
    target_classes: set[str],
    dry_run: bool,
) -> dict[str, dict[str, int]]:
    """
    Create train/val/test symlink tree under splits_dir.
    Returns {split: {class: n_images}}.
    """
    if dry_run or not manifest_data:
        return {}

    images       = manifest_data.get("images", [])
    # split_plan.json nests assignments under the "assignments" key.
    assignments  = split_plan.get("assignments", split_plan)
    split_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for entry in images:
        v6_label = entry.get("v6_label", "")
        if v6_label not in target_classes:
            continue

        image_id = entry.get("image_id", "")
        split    = assignments.get(image_id)
        if split not in ("train", "val", "test"):
            continue

        rel_path = entry.get("path", "")
        src = ready_dir / rel_path
        if not src.exists():
            continue

        dst_dir = splits_dir / split / v6_label
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name

        if not dst.exists():
            try:
                dst.symlink_to(src.resolve())
            except (FileExistsError, OSError):
                pass

        split_counts[split][v6_label] += 1

    logger.info(
        "V6 splits built at %s | "
        "train=%d val=%d test=%d",
        splits_dir,
        sum(split_counts.get("train", {}).values()),
        sum(split_counts.get("val", {}).values()),
        sum(split_counts.get("test", {}).values()),
    )
    return {s: dict(d) for s, d in split_counts.items()}


# ── Leakage detection ─────────────────────────────────────────────────────────


def _run_leakage_detection(
    manifest_data: dict,
    split_plan: dict[str, str],
) -> dict:
    """
    Hash-based leakage detection using existing LeakageDetector.
    Returns a serializable leakage report dict.
    """
    from app.modules.vision.medical.leakage_detector import LeakageDetector

    images       = manifest_data.get("images", [])
    assignments  = split_plan.get("assignments", split_plan)

    split_hashes: dict[str, dict[Path, str]] = defaultdict(dict)
    all_hashes:   dict[Path, str]             = {}

    for entry in images:
        sha256   = entry.get("sha256", "")
        image_id = entry.get("image_id", "")
        rel_path = entry.get("path", image_id)
        split    = assignments.get(image_id, "unknown")
        path     = Path(rel_path)

        if not sha256:
            continue
        all_hashes[path] = sha256
        if split in ("train", "val", "test"):
            split_hashes[split][path] = sha256

    detector    = LeakageDetector()
    dup_groups  = detector.detect_duplicates(all_hashes)
    cross_leaks = detector.detect_cross_split_leakage(
        train_hashes=split_hashes.get("train", {}),
        val_hashes=split_hashes.get("val"),
        test_hashes=split_hashes.get("test"),
    )

    n_dup_imgs  = sum(len(g.paths) for g in dup_groups)
    n_leaks     = len(cross_leaks)

    if n_leaks > 10:
        contamination_risk = "high"
    elif n_leaks > 0:
        contamination_risk = "low"
    else:
        contamination_risk = "none"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_images_checked": len(all_hashes),
        "within_class_duplicates": {
            "total_duplicate_groups": len(dup_groups),
            "total_duplicate_images": n_dup_imgs,
            "groups": [
                {
                    "sha256": g.sha256,
                    "paths":  [str(p) for p in g.paths[:5]],
                    "count":  len(g.paths),
                }
                for g in dup_groups[:20]
            ],
        },
        "cross_split_leakage": {
            "total_leaked_images": n_leaks,
            "train_val_leaks":  sum(1 for l in cross_leaks if l.contaminated_split == "val"),
            "train_test_leaks": sum(1 for l in cross_leaks if l.contaminated_split == "test"),
            "leaks": [
                {
                    "sha256":             l.sha256,
                    "train_path":         str(l.train_path),
                    "contaminated_path":  str(l.contaminated_path),
                    "contaminated_split": l.contaminated_split,
                }
                for l in cross_leaks[:20]
            ],
        },
        "contamination_risk": contamination_risk,
        "safe_for_training":  n_leaks == 0,
    }


# ── Report builders ───────────────────────────────────────────────────────────


def _build_ingestion_report(
    source_summaries: list[dict],
    generated_at: str,
) -> dict:
    total_ok  = sum(s.get("processed_ok", 0) + s.get("skipped_existing", 0) for s in source_summaries)
    total_rej = sum(s.get("rejected_quality", 0) for s in source_summaries)
    total_dup = sum(s.get("skipped_duplicate", 0) for s in source_summaries)
    total_err = sum(s.get("errors", 0) for s in source_summaries)

    per_v6: dict[str, int] = defaultdict(int)
    rejection_breakdown: dict[str, int] = defaultdict(int)

    for s in source_summaries:
        for lbl_info in s.get("per_label", {}).values():
            v6  = lbl_info.get("v6_label", "unknown")
            per_v6[v6] += lbl_info.get("count", 0)
        rejection_breakdown[s["source_id"]] = s.get("rejected_quality", 0)

    return {
        "generated_at":      generated_at,
        "sources":           {s["source_id"]: s for s in source_summaries},
        "total": {
            "processed_ok":      total_ok,
            "rejected_quality":  total_rej,
            "skipped_duplicate": total_dup,
            "errors":            total_err,
            "per_v6_class":      dict(per_v6),
        },
        "rejection_by_source": dict(rejection_breakdown),
    }


def _build_grayscale_report(
    manifest_data: dict,
    generated_at: str,
) -> dict:
    images   = manifest_data.get("images", [])
    per_cls: dict[str, dict] = defaultdict(lambda: {"total": 0, "grayscale": 0, "rgb": 0})

    for entry in images:
        cls = entry.get("v6_label", "unknown")
        per_cls[cls]["total"] += 1
        if entry.get("is_grayscale", False):
            per_cls[cls]["grayscale"] += 1
        else:
            per_cls[cls]["rgb"] += 1

    result: dict[str, dict] = {}
    for cls, counts in per_cls.items():
        total     = counts["total"]
        gray_frac = counts["grayscale"] / total if total > 0 else 0.0
        result[cls] = {
            "total":              total,
            "grayscale":          counts["grayscale"],
            "rgb":                counts["rgb"],
            "grayscale_fraction": round(gray_frac, 4),
        }

    # Shortcut risk: severe if one xray class all-gray and another all-RGB
    xray_fracs = {
        k: v["grayscale_fraction"]
        for k, v in result.items()
        if k in ("healthy_xray", "pneumonia_xray")
    }
    shortcut_risk = "none"
    details: list[str] = []

    if len(xray_fracs) >= 2:
        fracs = list(xray_fracs.values())
        spread = max(fracs) - min(fracs)
        if any(f >= 0.85 for f in fracs) and any(f <= 0.15 for f in fracs):
            shortcut_risk = "high"
            details.append(
                "Critical: one xray class is nearly all-grayscale, another nearly all-RGB. "
                "Model may learn the color channel as a diagnostic shortcut."
            )
        elif spread > 0.30:
            shortcut_risk = "medium"
            details.append(
                f"Grayscale imbalance between xray classes (spread={spread:.2f}). "
                "Ensure NormalizationPipeline force_grayscale_for_xray=True."
            )

    return {
        "generated_at":         generated_at,
        "per_class":            result,
        "shortcut_risk":        shortcut_risk,
        "shortcut_risk_details": details,
        "recommendation": (
            "Grayscale normalization forced for all CXR classes by NormalizationPipeline."
            if shortcut_risk == "none"
            else "Re-run normalization with force_grayscale_for_xray=True (already the default)."
        ),
    }


def _build_balance_report(
    manifest_data: dict,
    split_counts: dict[str, dict[str, int]],
    classes: list[str],
    generated_at: str,
) -> dict:
    images   = manifest_data.get("images", [])
    per_cls: dict[str, int] = defaultdict(int)
    for entry in images:
        per_cls[entry.get("v6_label", "unknown")] += 1

    total  = sum(per_cls.values())
    counts = [per_cls.get(c, 0) for c in classes]
    max_c  = max(counts) if counts else 0
    min_c  = min(c for c in counts if c > 0) if any(c > 0 for c in counts) else 0
    imbalance_ratio = round(max_c / max(min_c, 1), 2)

    # Minimum sample requirements (from Phase 8 BALANCING_PLANS)
    min_required = {
        "healthy_xray":   400,
        "pneumonia_xray": 400,
        "hard_negative":  200,
        "fake_medical":   100,
    }
    min_samples_met = {
        c: per_cls.get(c, 0) >= min_required.get(c, 0)
        for c in classes
    }

    warnings: list[str] = []
    if imbalance_ratio > 5.0:
        warnings.append(
            f"Severe class imbalance: {imbalance_ratio}x. "
            "Use oversampling or focal loss during training."
        )
    elif imbalance_ratio > 2.0:
        warnings.append(
            f"Class imbalance: {imbalance_ratio}x. "
            "WeightedRandomSampler is enabled by default in train_v6_medical.py."
        )

    for cls in classes:
        n   = per_cls.get(cls, 0)
        req = min_required.get(cls, 0)
        if n < req:
            warnings.append(f"{cls}: {n} images < {req} required. Collect more data.")

    ready = all(min_samples_met.values()) and bool(split_counts.get("train"))

    return {
        "generated_at":        generated_at,
        "total_images":        total,
        "per_class":           {c: per_cls.get(c, 0) for c in classes},
        "imbalance_ratio":     imbalance_ratio,
        "splits":              split_counts,
        "minimum_samples_met": min_samples_met,
        "ready_for_training":  ready,
        "next_step": (
            "python scripts/train_v6_medical.py --dataset-dir data/medical_v6_splits --stage all"
            if ready else
            "Collect more data, then re-run stage_medical_dataset.py"
        ),
        "warnings":            warnings,
    }


# ── Summary printer ───────────────────────────────────────────────────────────


def _print_summary(
    source_summaries: list[dict],
    split_counts: dict[str, dict[str, int]],
    leakage: dict,
    reports_dir: Path,
    elapsed: float,
) -> None:
    print("\n" + "=" * 60)
    print("Phase 11 — Medical Dataset Staging Summary")
    print("=" * 60)

    print("\nIngestion:")
    print(f"  {'Source':<16}  {'Discovered':>12}  {'OK':>8}  {'Rejected':>9}  {'Dupes':>6}  {'Errors':>7}")
    print("  " + "-" * 65)
    for s in source_summaries:
        sid  = s["source_id"]
        disc = s.get("total_discovered", 0)
        ok   = s.get("processed_ok", 0) + s.get("skipped_existing", 0)
        rej  = s.get("rejected_quality", 0)
        dup  = s.get("skipped_duplicate", 0)
        err  = s.get("errors", 0)
        print(f"  {sid:<16}  {disc:>12,}  {ok:>8,}  {rej:>9,}  {dup:>6,}  {err:>7,}")

    print("\nClass distribution (data/medical_ready/):")
    all_v6: dict[str, int] = defaultdict(int)
    for s in source_summaries:
        for info in s.get("per_label", {}).values():
            all_v6[info.get("v6_label", "unknown")] += info.get("count", 0)
    for cls in sorted(all_v6):
        print(f"  {cls:<24}  {all_v6[cls]:>6,}")

    if split_counts:
        print("\nSplit distribution (data/medical_v6_splits/):")
        all_classes = sorted({c for split in split_counts.values() for c in split})
        header = f"  {'Class':<24}" + "".join(f"  {s:>8}" for s in ["train", "val", "test"]) + "   Total"
        print(header)
        print("  " + "-" * (24 + 30 + 8))
        for cls in all_classes:
            row   = f"  {cls:<24}"
            total = 0
            for split in ["train", "val", "test"]:
                n      = split_counts.get(split, {}).get(cls, 0)
                row   += f"  {n:>8,}"
                total += n
            row += f"  {total:>6,}"
            print(row)

    print("\nLeakage audit:")
    dup_groups  = leakage.get("within_class_duplicates", {}).get("total_duplicate_groups", 0)
    cross_leaks = leakage.get("cross_split_leakage", {}).get("total_leaked_images", 0)
    risk        = leakage.get("contamination_risk", "unknown")
    safe        = leakage.get("safe_for_training", False)
    print(f"  Duplicate groups    : {dup_groups}")
    print(f"  Cross-split leaks   : {cross_leaks}")
    print(f"  Contamination risk  : {risk}")
    print(f"  Safe for training   : {'YES' if safe else 'NO — review leakage_report.json'}")

    print(f"\nReports written to  : {reports_dir}")
    print(f"Elapsed             : {elapsed:.1f}s")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stage_medical_dataset",
        description="Ingest, normalize, and split medical dataset for v6 training.",
    )
    p.add_argument(
        "--raw-dir", type=Path, default=Path("data/medical_raw"),
        help="Root of manually-downloaded raw images (default: data/medical_raw).",
    )
    p.add_argument(
        "--ready-dir", type=Path, default=Path("data/medical_ready"),
        help="Normalized output root (default: data/medical_ready).",
    )
    p.add_argument(
        "--splits-dir", type=Path, default=Path("data/medical_v6_splits"),
        help="V6 train/val/test split dirs (default: data/medical_v6_splits).",
    )
    p.add_argument(
        "--reports-dir", type=Path, default=Path("data/medical_reports"),
        help="Where to write JSON reports (default: data/medical_reports).",
    )
    p.add_argument(
        "--sources", nargs="+",
        default=["kermany"],
        choices=list(SOURCE_CONFIGS.keys()),
        help="Source IDs to ingest (default: kermany).",
    )
    p.add_argument(
        "--classes", nargs="+",
        default=["healthy_xray", "pneumonia_xray", "hard_negative", "fake_medical"],
        help="V6 target classes for split building.",
    )
    p.add_argument(
        "--val-fraction",  type=float, default=0.10,
        help="Validation fraction (default: 0.10).",
    )
    p.add_argument(
        "--test-fraction", type=float, default=0.15,
        help="Test fraction (default: 0.15).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for stratified splitting (default: 42).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Validate and report without writing any files.",
    )
    return p


def _write_report(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info("Report: %s", path)


def main() -> int:
    args = build_parser().parse_args()

    logger.info("=== Stage Medical Dataset — Phase 11 ===")
    logger.info("Raw dir      : %s", args.raw_dir)
    logger.info("Ready dir    : %s", args.ready_dir)
    logger.info("Splits dir   : %s", args.splits_dir)
    logger.info("Reports dir  : %s", args.reports_dir)
    logger.info("Sources      : %s", args.sources)
    logger.info("Classes      : %s", args.classes)
    logger.info("Dry-run      : %s", args.dry_run)

    t0           = time.time()
    generated_at = datetime.now(timezone.utc).isoformat()

    # ── Step 1: validate raw dirs ─────────────────────────────────────────────

    missing_sources = []
    for source_id in args.sources:
        src_dir = args.raw_dir / source_id
        if not src_dir.exists():
            logger.warning("Raw source dir not found: %s", src_dir)
            missing_sources.append(source_id)
        else:
            logger.info("Source dir OK: %s", src_dir)

    if missing_sources and not args.dry_run:
        logger.error(
            "Missing source dirs: %s\n"
            "Download the dataset(s) manually and place under %s/<source_id>/",
            missing_sources, args.raw_dir,
        )
        if len(missing_sources) == len(args.sources):
            return 2

    # ── Step 2: normalize each source ────────────────────────────────────────

    source_summaries: list[dict] = []

    for source_id in args.sources:
        src_dir    = args.raw_dir / source_id
        source_cfg = SOURCE_CONFIGS.get(source_id)

        if source_cfg is None:
            logger.warning("Unknown source '%s' — skipping.", source_id)
            continue
        if not src_dir.exists():
            logger.warning("Skipping '%s': directory not found.", source_id)
            source_summaries.append({"source_id": source_id, "total_discovered": 0,
                                     "processed_ok": 0, "skipped_existing": 0,
                                     "rejected_quality": 0, "skipped_duplicate": 0,
                                     "errors": 0, "per_label": {}})
            continue

        logger.info("=== Normalizing source: %s ===", source_id)
        summary = _run_source_normalization(
            source_id=source_id,
            source_cfg=source_cfg,
            raw_source_dir=src_dir,
            out_dir=args.ready_dir,
            dry_run=args.dry_run,
        )
        source_summaries.append(summary)

    # ── Step 3: build manifest + split plan ───────────────────────────────────

    logger.info("=== Building manifest + split plan ===")
    manifest_dir = args.reports_dir / "manifest"
    manifest_data, split_plan = _build_manifest(
        ready_dir=args.ready_dir,
        manifest_dir=manifest_dir,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    # ── Step 4: leakage detection ─────────────────────────────────────────────

    logger.info("=== Running leakage detection ===")
    leakage_data: dict = {}
    if manifest_data and split_plan:
        leakage_data = _run_leakage_detection(manifest_data, split_plan)
    else:
        leakage_data = {
            "generated_at": generated_at,
            "note": "No manifest available — leakage check skipped (dry_run or no data).",
            "contamination_risk": "unknown",
            "safe_for_training": False,
        }

    # ── Step 5: build v6 split directories ───────────────────────────────────

    logger.info("=== Building v6 split directories ===")
    split_counts = _build_v6_splits(
        ready_dir=args.ready_dir,
        manifest_data=manifest_data,
        split_plan=split_plan,
        splits_dir=args.splits_dir,
        target_classes=set(args.classes),
        dry_run=args.dry_run,
    )

    # ── Step 6: write reports ─────────────────────────────────────────────────

    logger.info("=== Writing reports ===")

    ingestion_report = _build_ingestion_report(source_summaries, generated_at)
    grayscale_report = _build_grayscale_report(manifest_data, generated_at)
    balance_report   = _build_balance_report(manifest_data, split_counts, args.classes, generated_at)

    if not args.dry_run:
        _write_report(args.reports_dir / "ingestion_report.json",    ingestion_report)
        _write_report(args.reports_dir / "leakage_report.json",      leakage_data)
        _write_report(args.reports_dir / "grayscale_audit.json",     grayscale_report)
        _write_report(args.reports_dir / "class_balance_report.json", balance_report)
    else:
        logger.info("DRY-RUN: reports not written")

    # ── Summary + exit code ───────────────────────────────────────────────────

    elapsed = time.time() - t0
    _print_summary(source_summaries, split_counts, leakage_data, args.reports_dir, elapsed)

    safe_for_training = leakage_data.get("safe_for_training", False)
    ready_for_training = balance_report.get("ready_for_training", False)

    if not safe_for_training and not args.dry_run:
        logger.error(
            "Cross-split leakage detected. Review %s before training.",
            args.reports_dir / "leakage_report.json",
        )
        return 2

    if not ready_for_training:
        logger.warning(
            "Dataset not yet ready for training. "
            "See class_balance_report.json for missing sample counts."
        )
        return 1

    logger.info(
        "Staging complete. Ready for training:\n"
        "  python scripts/train_v6_medical.py "
        "--dataset-dir %s --stage all",
        args.splits_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
