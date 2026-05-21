#!/usr/bin/env python3
"""
Medical dataset source registry — Phase 9.

Defines known public medical imaging datasets that can be manually
downloaded and fed into normalize_medical_dataset.py.

NO automatic downloading.  Each source entry contains:
  - human-readable download instructions
  - expected raw directory layout
  - source label → v6 class mapping
  - license and attribution

Usage
-----
# List all registered sources
python scripts/medical_dataset_sources.py --list

# Show detailed download instructions for one source
python scripts/medical_dataset_sources.py --info kermany

# Validate that a manually-downloaded source directory looks correct
python scripts/medical_dataset_sources.py --check kermany --raw-dir data/medical_raw
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("dataset_sources")


# ── Source metadata schema ────────────────────────────────────────────────────


@dataclass(frozen=True)
class DatasetSource:
    """Registry entry for one publicly available medical imaging dataset."""

    source_id: str                           # directory name under data/medical_raw/
    display_name: str
    description: str
    url: str                                 # canonical dataset URL (Kaggle / official)
    license: str
    citation: str

    # Expected structure under data/medical_raw/<source_id>/
    expected_subdirs: tuple[str, ...]        # raw subdirectory names

    # Maps raw subdir/label name → v6 class label
    label_mapping: dict[str, str]

    # Approximate totals (for pre-flight sanity check)
    expected_min_images: int
    expected_max_images: int
    approximate_size_gb: float

    # Manual download steps (one step per list item)
    download_steps: tuple[str, ...]

    # Additional notes for the annotator
    notes: str = ""
    extra_metadata_keys: tuple[str, ...] = ()   # metadata CSV columns to capture


@dataclass
class SourceRegistration:
    """
    Runtime record of a source that has been staged in data/medical_raw/.
    Written to data/medical_raw/<source_id>/source_info.json after --register.
    """
    source_id: str
    registered_at: str               # ISO-8601
    raw_dir: str
    image_count: int
    label_counts: dict[str, int]
    sha256_sample: str | None = None  # SHA-256 of first image as spot-check anchor
    notes: str = ""


# ── Known sources ─────────────────────────────────────────────────────────────


KNOWN_SOURCES: dict[str, DatasetSource] = {

    "kermany": DatasetSource(
        source_id="kermany",
        display_name="Kermany Chest X-Ray (Pneumonia)",
        description=(
            "5,863 chest X-ray images (JPEG) organized into Normal and Pneumonia. "
            "Widely used for binary CXR classification. "
            "Pediatric patients, 1–5 years old, Guangzhou Women and Children's Medical Center."
        ),
        url="https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia",
        license="CC BY 4.0",
        citation=(
            "Kermany, D. et al. (2018). Identifying Medical Diagnoses and Treatable "
            "Diseases by Image-Based Deep Learning. Cell, 172(5), 1122–1131."
        ),
        expected_subdirs=("train/NORMAL", "train/PNEUMONIA", "val/NORMAL", "val/PNEUMONIA",
                          "test/NORMAL", "test/PNEUMONIA"),
        label_mapping={
            "NORMAL":    "healthy_xray",
            "PNEUMONIA": "pneumonia_xray",
        },
        expected_min_images=5000,
        expected_max_images=6500,
        approximate_size_gb=1.2,
        download_steps=(
            "1. Create a Kaggle account and accept dataset terms at the URL above.",
            "2. Install Kaggle CLI: pip install kaggle",
            "3. Place kaggle.json API key in ~/.kaggle/kaggle.json (chmod 600).",
            "4. Run: kaggle datasets download -d paultimothymooney/chest-xray-pneumonia",
            "5. Unzip into data/medical_raw/kermany/ — the top-level must contain train/, val/, test/.",
            "6. Run: python scripts/medical_dataset_sources.py --check kermany --raw-dir data/medical_raw",
        ),
        notes=(
            "Pre-split into train/val/test by the original authors. "
            "DO NOT merge splits — keep the original split boundaries to allow "
            "future cross-dataset contamination checks."
        ),
    ),

    "rsna": DatasetSource(
        source_id="rsna",
        display_name="RSNA Pneumonia Detection Challenge",
        description=(
            "26,684 frontal-view chest X-rays from the NIH Clinical Center, "
            "labelled by board-certified radiologists for the 2018 RSNA challenge. "
            "Labels: Normal, Lung Opacity, Not Normal / No Lung Opacity."
        ),
        url="https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data",
        license="Custom (Kaggle competition — non-commercial research)",
        citation=(
            "RSNA Pneumonia Detection Challenge. Radiological Society of North America, 2018. "
            "https://www.kaggle.com/c/rsna-pneumonia-detection-challenge"
        ),
        expected_subdirs=("stage_2_train_images",),
        label_mapping={
            "Normal":             "healthy_xray",
            "Lung Opacity":       "pneumonia_xray",
            "No Lung Opacity/Not Normal": "uncertain_xray",
        },
        expected_min_images=20000,
        expected_max_images=30000,
        approximate_size_gb=8.0,
        download_steps=(
            "1. Join the RSNA Pneumonia Detection Challenge on Kaggle and accept terms.",
            "2. Run: kaggle competitions download -c rsna-pneumonia-detection-challenge",
            "3. Unzip into data/medical_raw/rsna/.",
            "4. The label CSV (stage_2_train_labels.csv) must be in data/medical_raw/rsna/.",
            "5. Images are DICOM (.dcm) — the normalizer will convert to JPEG.",
            "6. Run: python scripts/medical_dataset_sources.py --check rsna --raw-dir data/medical_raw",
        ),
        extra_metadata_keys=("patientId", "x", "y", "width", "height", "Target"),
        notes=(
            "Images are DICOM format. The normalizer handles conversion. "
            "Labels come from stage_2_train_labels.csv, NOT from directory names. "
            "Use --metadata-csv data/medical_raw/rsna/stage_2_train_labels.csv "
            "when running the normalizer."
        ),
    ),

    "nih_cxr14": DatasetSource(
        source_id="nih_cxr14",
        display_name="NIH ChestX-ray14",
        description=(
            "112,120 frontal-view chest X-rays from 30,805 unique patients, "
            "14 disease labels extracted from radiology reports via NLP. "
            "Labels relevant here: No Finding (→ healthy_xray), "
            "Pneumonia, Infiltration, Consolidation (→ pneumonia_xray / opacity_pattern)."
        ),
        url="https://nihcc.app.box.com/v/ChestXray-NIHCC",
        license="CC0 (Public Domain)",
        citation=(
            "Wang, X. et al. (2017). ChestX-ray8: Hospital-scale Chest X-ray Database "
            "and Benchmarks. CVPR 2017."
        ),
        expected_subdirs=("images",),
        label_mapping={
            "No Finding":    "healthy_xray",
            "Pneumonia":     "pneumonia_xray",
            "Infiltration":  "opacity_pattern",
            "Consolidation": "opacity_pattern",
            "Effusion":      "uncertain_xray",
        },
        expected_min_images=80000,
        expected_max_images=120000,
        approximate_size_gb=45.0,
        download_steps=(
            "1. Accept the data use agreement at the URL above.",
            "2. Download all 12 zip files (images_001.tar.gz through images_012.tar.gz).",
            "3. Extract all into data/medical_raw/nih_cxr14/images/.",
            "4. Download Data_Entry_2017.csv into data/medical_raw/nih_cxr14/.",
            "5. NOTE: 112k images ≈ 45 GB. Download only what is needed for the target classes.",
            "6. Run: python scripts/medical_dataset_sources.py --check nih_cxr14 --raw-dir data/medical_raw",
        ),
        extra_metadata_keys=("Image Index", "Finding Labels", "Patient Age", "Patient Gender",
                             "View Position"),
        notes=(
            "CAUTION: 45 GB total. Download selectively — use Data_Entry_2017.csv to "
            "identify only 'No Finding' and 'Pneumonia' images first (~30k images, ~12 GB). "
            "Labels are pipe-separated multi-labels; filter by primary label only. "
            "PA (posteroanterior) view only — exclude AP views for consistency."
        ),
    ),

    "pneumoniamnist": DatasetSource(
        source_id="pneumoniamnist",
        display_name="PneumoniaMNIST (MedMNIST v2)",
        description=(
            "5,856 chest X-ray images from Kermany et al., resized to 28×28 (or 224×224). "
            "Binary classification: Normal vs Pneumonia. "
            "Part of the MedMNIST v2 benchmark suite."
        ),
        url="https://medmnist.com/",
        license="CC BY 4.0",
        citation=(
            "Yang, J. et al. (2023). MedMNIST v2 — A Large-Scale Lightweight Benchmark "
            "for 2D and 3D Biomedical Image Classification. Scientific Data, 10, 41."
        ),
        expected_subdirs=(),
        label_mapping={
            "0": "healthy_xray",
            "1": "pneumonia_xray",
        },
        expected_min_images=5000,
        expected_max_images=6000,
        approximate_size_gb=0.1,
        download_steps=(
            "1. pip install medmnist",
            "2. python -c \"import medmnist; medmnist.PneumoniaMNIST(split='train', download=True, size=224)\"",
            "3. The dataset is stored as a NumPy .npz file in ~/.medmnist/.",
            "4. Export to JPEG: see medmnist documentation for as_pil().",
            "5. Place exported images under data/medical_raw/pneumoniamnist/.",
            "6. CAUTION: 28×28 version is too small for EfficientNet — use size=224.",
        ),
        notes=(
            "Derived from Kermany — high overlap with the kermany source. "
            "Run duplicate detection (scripts/medical_dataset_audit.py) "
            "BEFORE merging with kermany to avoid train/test contamination."
        ),
    ),

    "custom": DatasetSource(
        source_id="custom",
        display_name="Custom / Institutional Dataset",
        description=(
            "Placeholder for institutional, proprietary, or custom-collected datasets. "
            "Images must be manually placed in subdirectories named by their v6 class label."
        ),
        url="",
        license="Custom — check with your institution",
        citation="As required by your institution.",
        expected_subdirs=("healthy_xray", "pneumonia_xray", "uncertain_xray"),
        label_mapping={
            "healthy_xray":   "healthy_xray",
            "pneumonia_xray": "pneumonia_xray",
            "uncertain_xray": "uncertain_xray",
        },
        expected_min_images=1,
        expected_max_images=999999,
        approximate_size_gb=0.0,
        download_steps=(
            "1. Organize images into subdirectories by v6 class label.",
            "2. Place under data/medical_raw/custom/<class_name>/.",
            "3. Ensure patient consent and IRB approval before use.",
            "4. Add a source_info.json describing provenance (institution, date, protocol).",
        ),
        notes=(
            "Directory structure must match label_mapping keys exactly. "
            "All images require de-identification before placement here. "
            "Document IRB approval number in source_info.json."
        ),
    ),
}


# ── Validation helpers ────────────────────────────────────────────────────────


def check_raw_directory(source_id: str, raw_root: Path) -> dict:
    """
    Validate that a manually-downloaded source directory looks structurally correct.

    Returns a dict with: found_images, missing_subdirs, label_coverage, ok.
    Does NOT decode or hash images — just counts and checks structure.
    """
    if source_id not in KNOWN_SOURCES:
        return {"ok": False, "error": f"Unknown source_id: {source_id!r}"}

    src = KNOWN_SOURCES[source_id]
    raw_dir = raw_root / source_id

    if not raw_dir.is_dir():
        return {
            "ok": False,
            "error": f"Directory not found: {raw_dir}",
            "hint": f"Download {src.display_name} and place under {raw_dir}",
        }

    _SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".dcm"}
    image_files = [
        p for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _SUPPORTED
    ]
    n = len(image_files)

    missing_subdirs: list[str] = []
    for subdir in src.expected_subdirs:
        if not (raw_dir / subdir).exists():
            missing_subdirs.append(subdir)

    ok = (
        src.expected_min_images <= n <= src.expected_max_images
        and len(missing_subdirs) == 0
    )

    return {
        "ok": ok,
        "source_id": source_id,
        "raw_dir": str(raw_dir),
        "found_images": n,
        "expected_range": [src.expected_min_images, src.expected_max_images],
        "missing_subdirs": missing_subdirs,
        "label_mapping": src.label_mapping,
    }


def list_sources() -> None:
    print(f"\n{'ID':<18} {'Images':>8}  {'Size':>6}  {'License':<18}  Name")
    print("-" * 80)
    for sid, src in KNOWN_SOURCES.items():
        print(f"{sid:<18} {src.expected_min_images:>5}–{src.expected_max_images:<6}  "
              f"{src.approximate_size_gb:>4.0f}GB  {src.license:<18}  {src.display_name}")
    print()


def show_source_info(source_id: str) -> None:
    if source_id not in KNOWN_SOURCES:
        print(f"Unknown source: {source_id!r}. Run --list to see available sources.")
        return
    src = KNOWN_SOURCES[source_id]
    print(f"\n{'='*60}")
    print(f"  {src.display_name}")
    print(f"{'='*60}")
    print(f"  URL      : {src.url}")
    print(f"  License  : {src.license}")
    print(f"  Images   : {src.expected_min_images:,} – {src.expected_max_images:,}")
    print(f"  Size     : ~{src.approximate_size_gb:.0f} GB")
    print(f"\n  Label mapping:")
    for raw_label, v6 in src.label_mapping.items():
        print(f"    '{raw_label}'  →  '{v6}'")
    print(f"\n  Download steps:")
    for step in src.download_steps:
        print(f"    {step}")
    if src.notes:
        print(f"\n  Notes: {src.notes}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="medical_dataset_sources",
        description="Medical dataset source registry — list, inspect, and validate sources.",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--list",    action="store_true",
                   help="List all registered dataset sources.")
    g.add_argument("--info",    metavar="SOURCE_ID",
                   help="Show detailed info and download instructions for a source.")
    g.add_argument("--check",   metavar="SOURCE_ID",
                   help="Validate a manually-downloaded source directory.")
    g.add_argument("--dump",    metavar="SOURCE_ID",
                   help="Dump source registry entry as JSON.")
    p.add_argument("--raw-dir", type=Path, default=Path("data/medical_raw"),
                   help="Root of manually-downloaded raw datasets (default: data/medical_raw).")
    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.list:
        list_sources()
        return 0

    if args.info:
        show_source_info(args.info)
        return 0

    if args.dump:
        if args.dump not in KNOWN_SOURCES:
            logger.error("Unknown source: %s", args.dump)
            return 1
        src = KNOWN_SOURCES[args.dump]
        print(json.dumps({
            "source_id": src.source_id,
            "display_name": src.display_name,
            "url": src.url,
            "license": src.license,
            "label_mapping": src.label_mapping,
            "expected_min_images": src.expected_min_images,
            "expected_max_images": src.expected_max_images,
            "approximate_size_gb": src.approximate_size_gb,
        }, indent=2))
        return 0

    if args.check:
        result = check_raw_directory(args.check, args.raw_dir)
        status = "OK" if result.get("ok") else "FAIL"
        print(json.dumps(result, indent=2))
        if not result.get("ok"):
            error = result.get("error", "Structural check failed")
            hint  = result.get("hint", "")
            logger.error("%s: %s", status, error)
            if hint:
                logger.info("Hint: %s", hint)
            return 1
        logger.info("CHECK %s: found %d images", status, result.get("found_images", 0))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
