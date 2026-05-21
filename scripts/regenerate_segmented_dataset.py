"""
HantaProject — Offline Lung ROI Dataset Regeneration
=====================================================

Processes a source dataset directory, runs lung segmentation on every image,
and writes a new dataset of segmented ROI crops.

Input layout (standard split/class layout):
    <source_dir>/
    ├── train/healthy_xray/*.jpg
    ├── train/pneumonia_xray/*.jpg
    ├── val/healthy_xray/*.jpg
    └── test/...

Output layout (identical structure, new root):
    <output_dir>/
    ├── train/healthy_xray/img001.jpg
    ├── train/healthy_xray/img001_telemetry.json
    └── quarantine/
        ├── train/healthy_xray/bad_001.jpg
        └── train/healthy_xray/bad_001_telemetry.json

Each accepted image is saved as a JPEG ROI crop alongside a
`<stem>_telemetry.json` file recording:

    lung_area_pct, roi_width, roi_height, crop_ratio, border_removed,
    quality, n_components, roi_x1, roi_y1, roi_x2, roi_y2,
    roi_center_x, roi_center_y, original_width, original_height,
    class_name, split, source_path, rejection_reason (quarantine only)

Quality filtering
-----------------
A sample is quarantined (not training-eligible) if any of these hold:
  - lung_area_pct < MIN_LUNG_PCT  (too little lung visible)
  - lung_area_pct > MAX_LUNG_PCT  (oversegmentation)
  - n_components == 0             (no lung found at all)
  - ROI aspect ratio outside [MIN_ASPECT, MAX_ASPECT]
  - quality == "fallback" AND border_removed on all 4 edges

QA Visualisation
----------------
A random sample of --qa-samples images gets a 4-panel debug image saved to
<output_dir>/qa_visualisations/: original | mask | ROI | overlay.

Usage
-----
    python scripts/regenerate_segmented_dataset.py \\
        --source-dir data/medical_v6_splits \\
        --output-dir data/segmented_dataset \\
        --qa-samples 40

    # Strict mode: quarantine fallback segmentations too
    python scripts/regenerate_segmented_dataset.py \\
        --source-dir data/medical_v6_splits \\
        --output-dir data/segmented_dataset \\
        --quarantine-fallback \\
        --min-lung-pct 0.10 \\
        --max-lung-pct 0.65
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from app.core.logging import setup_logging  # noqa: E402
from app.modules.vision.segmentation import LungSegmenter, ROIExtractor  # noqa: E402
from app.modules.vision.segmentation.mask_utils import detect_black_border  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ── Quality thresholds (overridable via CLI) ───────────────────────────────────

DEFAULT_MIN_LUNG_PCT = 0.08
DEFAULT_MAX_LUNG_PCT = 0.68
DEFAULT_MIN_ASPECT = 0.25    # roi_width / roi_height
DEFAULT_MAX_ASPECT = 4.0


# ── Result containers ─────────────────────────────────────────────────────────


class ProcessResult:
    """Outcome of processing one image."""

    __slots__ = (
        "source_path", "class_name", "split", "accepted",
        "rejection_reason", "telemetry", "roi_image", "seg_mask",
    )

    def __init__(
        self,
        source_path: Path,
        class_name: str,
        split: str,
        accepted: bool,
        rejection_reason: str,
        telemetry: dict,
        roi_image: Optional[Image.Image],
        seg_mask: Optional[np.ndarray],
    ) -> None:
        self.source_path = source_path
        self.class_name = class_name
        self.split = split
        self.accepted = accepted
        self.rejection_reason = rejection_reason
        self.telemetry = telemetry
        self.roi_image = roi_image
        self.seg_mask = seg_mask


# ── Per-image processing ──────────────────────────────────────────────────────


def process_image(
    img_path: Path,
    class_name: str,
    split: str,
    segmenter: LungSegmenter,
    extractor: ROIExtractor,
    min_lung_pct: float,
    max_lung_pct: float,
    min_aspect: float,
    max_aspect: float,
    quarantine_fallback: bool,
) -> ProcessResult:
    """
    Segment a single image and decide accept / quarantine.

    Returns a ProcessResult with the ROI crop and telemetry dict.
    """
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as exc:
        return ProcessResult(
            source_path=img_path, class_name=class_name, split=split,
            accepted=False, rejection_reason=f"load_error:{exc}",
            telemetry={}, roi_image=None, seg_mask=None,
        )

    orig_w, orig_h = image.size
    gray_arr = np.array(image.convert("L"))

    seg = segmenter.segment(image)
    roi = extractor.extract(image, seg.mask)

    x1, y1, x2, y2 = roi.bbox
    roi_cx = (x1 + x2) / 2.0 / max(orig_w, 1)  # normalised 0-1
    roi_cy = (y1 + y2) / 2.0 / max(orig_h, 1)
    aspect = roi.roi_width / max(roi.roi_height, 1)

    # Check all 4 edges for black borders
    border_all_sides = (
        detect_black_border(gray_arr, margin=15, threshold=15)
        and detect_black_border(gray_arr[:, ::-1], margin=15, threshold=15)
    )

    telemetry = {
        "lung_area_pct": seg.lung_area_pct,
        "roi_width": roi.roi_width,
        "roi_height": roi.roi_height,
        "crop_ratio": roi.crop_ratio,
        "border_removed": roi.border_removed,
        "quality": seg.quality,
        "n_components": seg.n_components,
        "segmentation_ms": round(seg.segmentation_ms, 2),
        "roi_x1": x1, "roi_y1": y1, "roi_x2": x2, "roi_y2": y2,
        "roi_center_x": round(roi_cx, 4),
        "roi_center_y": round(roi_cy, 4),
        "aspect_ratio": round(aspect, 4),
        "original_width": orig_w,
        "original_height": orig_h,
        "class_name": class_name,
        "split": split,
        "source_path": str(img_path),
    }

    # ── Quality checks ────────────────────────────────────────────────────────

    rejection_reason = ""

    if seg.n_components == 0:
        rejection_reason = "no_lung_found"
    elif seg.lung_area_pct < min_lung_pct:
        rejection_reason = f"lung_area_too_small:{seg.lung_area_pct:.3f}<{min_lung_pct}"
    elif seg.lung_area_pct > max_lung_pct:
        rejection_reason = f"lung_area_too_large:{seg.lung_area_pct:.3f}>{max_lung_pct}"
    elif aspect < min_aspect or aspect > max_aspect:
        rejection_reason = f"aspect_ratio_invalid:{aspect:.2f} not in [{min_aspect},{max_aspect}]"
    elif quarantine_fallback and seg.quality == "fallback" and border_all_sides:
        rejection_reason = "fallback_with_all_border_artifacts"

    accepted = rejection_reason == ""
    if not accepted:
        telemetry["rejection_reason"] = rejection_reason

    return ProcessResult(
        source_path=img_path,
        class_name=class_name,
        split=split,
        accepted=accepted,
        rejection_reason=rejection_reason,
        telemetry=telemetry,
        roi_image=roi.roi_image if accepted else roi.roi_image,
        seg_mask=seg.mask,
    )


# ── Saving helpers ─────────────────────────────────────────────────────────────


def save_result(
    result: ProcessResult,
    accepted_base: Path,
    quarantine_base: Path,
    jpeg_quality: int,
) -> Path:
    """Write the ROI crop + telemetry sidecar to the appropriate directory."""
    base = accepted_base if result.accepted else quarantine_base
    out_dir = base / result.split / result.class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = result.source_path.stem
    out_img = out_dir / f"{stem}.jpg"
    out_tel = out_dir / f"{stem}_telemetry.json"

    if result.roi_image is not None:
        result.roi_image.convert("RGB").save(out_img, format="JPEG", quality=jpeg_quality)
    else:
        # Quarantine the original image so it can be inspected
        shutil_copy_safe(result.source_path, out_img)

    out_tel.write_text(json.dumps(result.telemetry, indent=2), encoding="utf-8")
    return out_img


def shutil_copy_safe(src: Path, dst: Path) -> None:
    import shutil
    shutil.copy2(src, dst)


# ── QA visualisation ──────────────────────────────────────────────────────────


def save_qa_visualisation(
    result: ProcessResult,
    qa_dir: Path,
    segmenter: LungSegmenter,
) -> None:
    """
    Save a 4-panel QA image: original | mask | ROI | overlay.

    Skips silently if the result has no roi_image.
    """
    if result.roi_image is None or result.seg_mask is None:
        return

    from PIL import ImageDraw, ImageFont

    try:
        original = Image.open(result.source_path).convert("RGB")
        mask_img = Image.fromarray(result.seg_mask).convert("RGB")
        roi_img = result.roi_image.convert("RGB")

        # Overlay: green mask + red bounding box
        overlay = original.copy().convert("RGBA")
        h, w = result.seg_mask.shape
        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask_rgba[result.seg_mask > 127] = [0, 200, 0, 100]
        overlay = Image.alpha_composite(overlay, Image.fromarray(mask_rgba, "RGBA"))
        draw = ImageDraw.Draw(overlay)
        x1, y1, x2, y2 = (
            result.telemetry.get("roi_x1", 0),
            result.telemetry.get("roi_y1", 0),
            result.telemetry.get("roi_x2", w),
            result.telemetry.get("roi_y2", h),
        )
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 220), width=3)
        overlay = overlay.convert("RGB")

        # Compose 4-panel 2×2 grid (all resized to 256×256)
        panel_size = 256
        panels = [original, mask_img, roi_img, overlay]
        resized = [p.resize((panel_size, panel_size), Image.LANCZOS) for p in panels]
        grid = Image.new("RGB", (panel_size * 2, panel_size * 2), (40, 40, 40))
        grid.paste(resized[0], (0, 0))
        grid.paste(resized[1], (panel_size, 0))
        grid.paste(resized[2], (0, panel_size))
        grid.paste(resized[3], (panel_size, panel_size))

        tag = "OK" if result.accepted else "QUAR"
        fname = f"{result.split}_{result.class_name}_{result.source_path.stem}_{tag}.jpg"
        qa_dir.mkdir(parents=True, exist_ok=True)
        grid.save(qa_dir / fname, format="JPEG", quality=88)

    except Exception:
        logger.debug(
            "QA visualisation failed for %s", result.source_path, exc_info=True
        )


# ── Dataset report ─────────────────────────────────────────────────────────────


def generate_report(
    all_results: list[ProcessResult],
    output_dir: Path,
    args: argparse.Namespace,
    elapsed_s: float,
) -> dict:
    """Build and return the full dataset report dict; also writes JSON."""
    accepted = [r for r in all_results if r.accepted]
    quarantined = [r for r in all_results if not r.accepted]

    def _class_dist(results: list[ProcessResult]) -> dict[str, int]:
        d: dict[str, int] = {}
        for r in results:
            d[r.class_name] = d.get(r.class_name, 0) + 1
        return d

    def _split_dist(results: list[ProcessResult]) -> dict[str, int]:
        d: dict[str, int] = {}
        for r in results:
            d[r.split] = d.get(r.split, 0) + 1
        return d

    def _quality_dist(results: list[ProcessResult]) -> dict[str, int]:
        d: dict[str, int] = {}
        for r in results:
            q = r.telemetry.get("quality", "unknown")
            d[q] = d.get(q, 0) + 1
        return d

    def _rejection_dist(results: list[ProcessResult]) -> dict[str, int]:
        d: dict[str, int] = {}
        for r in results:
            reason = r.rejection_reason.split(":")[0]
            d[reason] = d.get(reason, 0) + 1
        return d

    # ROI statistics from accepted samples
    lung_areas = [r.telemetry.get("lung_area_pct", 0) for r in accepted]
    crop_ratios = [r.telemetry.get("crop_ratio", 0) for r in accepted]
    roi_widths  = [r.telemetry.get("roi_width", 0)  for r in accepted]
    roi_heights = [r.telemetry.get("roi_height", 0) for r in accepted]

    def _stats(vals: list) -> dict:
        if not vals:
            return {"mean": 0, "min": 0, "max": 0}
        import statistics as _s
        return {
            "mean": round(_s.mean(vals), 4),
            "std": round(_s.stdev(vals) if len(vals) > 1 else 0.0, 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_dir": str(args.source_dir),
        "output_dir": str(output_dir),
        "elapsed_s": round(elapsed_s, 1),
        "parameters": {
            "min_lung_pct": args.min_lung_pct,
            "max_lung_pct": args.max_lung_pct,
            "min_aspect_ratio": args.min_aspect,
            "max_aspect_ratio": args.max_aspect,
            "quarantine_fallback": args.quarantine_fallback,
            "jpeg_quality": args.jpeg_quality,
        },
        "counts": {
            "total": len(all_results),
            "accepted": len(accepted),
            "quarantined": len(quarantined),
            "segmentation_success_rate_pct": round(
                100 * len(accepted) / max(len(all_results), 1), 2
            ),
        },
        "class_balance": {
            "accepted": _class_dist(accepted),
            "quarantined": _class_dist(quarantined),
        },
        "split_counts": {
            "accepted": _split_dist(accepted),
            "quarantined": _split_dist(quarantined),
        },
        "segmentation_quality": _quality_dist(accepted),
        "rejection_reasons": _rejection_dist(quarantined),
        "roi_statistics": {
            "lung_area_pct": _stats(lung_areas),
            "crop_ratio": _stats(crop_ratios),
            "roi_width_px": _stats(roi_widths),
            "roi_height_px": _stats(roi_heights),
        },
    }

    report_path = output_dir / "segmentation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ── Summary printer ────────────────────────────────────────────────────────────


def print_summary(report: dict, output_dir: Path) -> None:
    sep = "─" * 64
    lines = [
        sep,
        "  Lung ROI Dataset Regeneration — Complete",
        sep,
        f"  Total images processed  : {report['counts']['total']}",
        f"  Accepted (training-ready): {report['counts']['accepted']}",
        f"  Quarantined             : {report['counts']['quarantined']}",
        f"  Segmentation success    : {report['counts']['segmentation_success_rate_pct']}%",
        f"  Elapsed                 : {report['elapsed_s']:.1f}s",
        sep,
        "  Class distribution (accepted):",
    ]
    for cls, cnt in report["class_balance"]["accepted"].items():
        lines.append(f"    {cls:<24} {cnt}")
    lines += [
        sep,
        "  ROI statistics:",
        f"    lung_area_pct  mean={report['roi_statistics']['lung_area_pct']['mean']:.3f}",
        f"    crop_ratio     mean={report['roi_statistics']['crop_ratio']['mean']:.3f}",
        sep,
        f"  Report → {output_dir / 'segmentation_report.json'}",
        sep,
    ]
    print("\n".join(lines))


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate a segmented lung-ROI dataset from source images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-dir", required=True, type=Path,
        help="Source dataset root (split/class/image.* layout).",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Output directory for segmented ROI crops.",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        help="Dataset splits to process.",
    )
    parser.add_argument("--min-lung-pct", type=float, default=DEFAULT_MIN_LUNG_PCT)
    parser.add_argument("--max-lung-pct", type=float, default=DEFAULT_MAX_LUNG_PCT)
    parser.add_argument("--min-aspect", type=float, default=DEFAULT_MIN_ASPECT,
                        help="Minimum ROI width/height aspect ratio.")
    parser.add_argument("--max-aspect", type=float, default=DEFAULT_MAX_ASPECT,
                        help="Maximum ROI width/height aspect ratio.")
    parser.add_argument(
        "--quarantine-fallback", action="store_true",
        help="Also quarantine images where segmentation fell back to the center mask "
             "AND all 4 image edges are dark (strong artifact indicator).",
    )
    parser.add_argument(
        "--qa-samples", type=int, default=20,
        help="Number of random QA visualisations to generate.",
    )
    parser.add_argument(
        "--padding-frac", type=float, default=0.07,
        help="ROI padding fraction (7%% per side).",
    )
    parser.add_argument("--jpeg-quality", type=int, default=92,
                        help="JPEG quality for saved ROI crops.")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for QA sample selection.",
    )
    parser.add_argument("--log-interval", type=int, default=100,
                        help="Log progress every N images.")
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    setup_logging(debug=False, environment="development")

    t_start = time.perf_counter()

    if not args.source_dir.exists():
        logger.error("Source directory not found: %s", args.source_dir)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    quarantine_base = args.output_dir / "quarantine"
    qa_dir = args.output_dir / "qa_visualisations"

    # ── Shared segmentation components ────────────────────────────────────────

    segmenter = LungSegmenter()
    extractor = ROIExtractor(padding_frac=args.padding_frac)

    # ── Discover all images ───────────────────────────────────────────────────

    tasks: list[tuple[Path, str, str]] = []   # (path, class_name, split)
    for split in args.splits:
        split_dir = args.source_dir / split
        if not split_dir.exists():
            logger.warning("Split directory not found: %s — skipping", split_dir)
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in _IMAGE_EXTENSIONS:
                    tasks.append((img_path, class_name, split))

    total = len(tasks)
    logger.info(
        "Regeneration starting: %d images across splits=%s", total, args.splits
    )

    if total == 0:
        logger.error("No images found in %s", args.source_dir)
        sys.exit(1)

    # ── Select QA samples before processing ───────────────────────────────────

    rng = random.Random(args.seed)
    qa_indices = set(
        rng.sample(range(total), min(args.qa_samples, total))
    )

    # ── Process images ────────────────────────────────────────────────────────

    all_results: list[ProcessResult] = []
    n_accepted = 0
    n_quarantined = 0

    for i, (img_path, class_name, split) in enumerate(tasks):
        result = process_image(
            img_path=img_path,
            class_name=class_name,
            split=split,
            segmenter=segmenter,
            extractor=extractor,
            min_lung_pct=args.min_lung_pct,
            max_lung_pct=args.max_lung_pct,
            min_aspect=args.min_aspect,
            max_aspect=args.max_aspect,
            quarantine_fallback=args.quarantine_fallback,
        )

        save_result(result, args.output_dir, quarantine_base, args.jpeg_quality)

        if i in qa_indices:
            save_qa_visualisation(result, qa_dir, segmenter)

        all_results.append(result)
        if result.accepted:
            n_accepted += 1
        else:
            n_quarantined += 1

        if (i + 1) % args.log_interval == 0 or i + 1 == total:
            logger.info(
                "Progress: %d/%d | accepted=%d quarantined=%d "
                "(%.1f%% success)",
                i + 1, total, n_accepted, n_quarantined,
                100 * n_accepted / max(i + 1, 1),
            )

    elapsed = time.perf_counter() - t_start

    # ── Report ────────────────────────────────────────────────────────────────

    report = generate_report(all_results, args.output_dir, args, elapsed)
    print_summary(report, args.output_dir)

    logger.info(
        "Segmentation report written to %s",
        args.output_dir / "segmentation_report.json",
    )
    if args.qa_samples > 0:
        logger.info("QA visualisations → %s", qa_dir)


if __name__ == "__main__":
    main()
