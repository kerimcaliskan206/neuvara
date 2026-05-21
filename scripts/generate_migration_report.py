"""
Migration report generator — Phase 3 segmented lung ROI architecture.

Reads benchmark results (if available) and produces a comprehensive Markdown
report documenting:
  - Architecture changes (what was removed, what was added)
  - Removed bias mechanisms
  - Remaining safety layers
  - Segmentation pipeline statistics
  - GradCAM audit summary
  - Stress suite summary
  - Model performance comparison

Usage
-----
    # Full report with benchmark data:
    python scripts/generate_migration_report.py \
        --benchmark results/benchmark.json \
        --seg-report data/segmented_dataset/segmentation_report.json \
        --output    results/migration_report.md

    # Architecture-only report (no benchmark data required):
    python scripts/generate_migration_report.py \
        --output results/migration_report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Report sections ───────────────────────────────────────────────────────────


ARCHITECTURE_SECTION = """\
## Architecture Overview

### Phase 3 Segmented-Lung-First Pipeline

```
Raw X-ray Image
       │
       ▼
┌─────────────────────────────────────────┐
│         LungSegmentationPipeline        │
│  CLAHE → Otsu × 2 → Morphology →       │
│  Hole-fill → Top-2 Components → ROI    │
└─────────────────────────────────────────┘
       │
       │  (fallback: center 80% if plausibility fails)
       │  (telemetry: lung_area_pct, crop_ratio, quality)
       ▼
┌─────────────────────────────────────────┐
│          Lung ROI Image                 │
│  Both lungs, +7% padding, aspect kept  │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│       ImagePreprocessingPipeline        │
│  Resize(224) → Normalize(ImageNet)      │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│     EfficientNet-B0 Classifier          │
│  Trained exclusively on ROI crops       │
└─────────────────────────────────────────┘
       │
       ▼
   VisionPrediction
   + segmentation_quality
   + low_trust
   + segmentation_telemetry
```
"""

REMOVED_MECHANISMS = """\
## Removed Bias Mechanisms

| Mechanism | Reason Removed |
|-----------|----------------|
| Centre-crop TTA (50% crop inference) | Redundant — segmentation already removes borders |
| Hardcoded IDX_TO_CLASS dependency | Replaced by checkpoint-embedded class_names |
| Full-image fallback in inference | Segmentation is now mandatory; fallback only to center-mask |
| Border-dependency heuristics (TTA diff) | Eliminated at source — ROI crops cannot contain borders |
| Lateral penalty default 0.35 | Reduced to 0.10 — weaker suppression, trust the model more |
| min_attenuation default 0.55 | Raised to 0.82 — permits confident predictions on clean ROIs |
| peripheral_dominance_threshold 35.0 | Raised to 40.0 — less aggressive GradCAM attenuation |
"""

REMAINING_SAFETY = """\
## Remaining Safety Layers

| Layer | Description |
|-------|-------------|
| `low_trust` flag | Set when segmentation quality == "fallback"; confidence capped at 0.75 |
| `MEDICAL_CONFIDENCE_CAP` env var | Deployment-side escape hatch for calibration mismatches |
| `MEDICAL_TEMPERATURE` env var | Temperature scaling override at inference time |
| `ReliabilityStats` monitoring | Tracks extreme-confidence rate, per-class bias, source-prefix shortcuts |
| GradCAM `compute_score_attenuation` | Minimal residual attenuation (min=0.82, threshold=40.0) |
| Calibration temperature (T) | Trained with temperature scaling; loaded from checkpoint metadata |
"""

SEGMENTATION_PIPELINE = """\
## Segmentation Pipeline

### Algorithm
1. **CLAHE** (contrast-limited adaptive histogram equalization) on grayscale
2. **Gaussian blur** (5×5) to suppress noise
3. **Dual Otsu**: threshold normal image, then threshold inverted image; pick the
   candidate whose lung area fraction is closest to 0.38 (midpoint of [0.08, 0.68])
4. **Morphological closing** (MORPH_CLOSE, ellipse kernel 15×15) to fill small gaps
5. **Hole-fill** via flood-fill from all four image boundaries, then invert
6. **Keep top-2 connected components** by area (left + right lung)
7. **Plausibility check**: area fraction must be in [0.08, 0.68]; if not → fallback
8. **Fallback**: center 80% mask; `quality = "fallback"`, `low_trust = True`

### Telemetry per image
- `lung_area_pct`: fraction of original image occupied by lungs
- `crop_ratio`: ROI crop area / original image area
- `roi_width`, `roi_height`: ROI pixel dimensions
- `border_removed`: whether black border pixels were detected
- `quality`: `"good"` | `"single_lung"` | `"fallback"`
- `n_components`: number of connected components found
"""


def _load_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"_error": str(e)}


def _seg_section(seg: dict) -> str:
    if not seg:
        return "_No segmentation report found._\n"
    lines = ["## Segmentation Dataset Statistics\n"]
    counts = seg.get("counts", {})
    if counts:
        lines.append(f"- Total processed: {counts.get('total', '—')}")
        lines.append(f"- Accepted: {counts.get('accepted', '—')}")
        lines.append(f"- Rejected: {counts.get('rejected', '—')}")
        lines.append(f"- Quarantined: {counts.get('quarantined', '—')}")
    quality = seg.get("segmentation_quality", {})
    if quality:
        lines.append("\n**Segmentation quality distribution:**")
        for q, n in quality.items():
            lines.append(f"  - `{q}`: {n}")
    roi_stats = seg.get("roi_statistics", {})
    if roi_stats:
        lines.append("\n**ROI statistics (accepted images):**")
        for k, v in roi_stats.items():
            lines.append(f"  - {k}: {v}")
    return "\n".join(lines) + "\n"


def _benchmark_section(bench: dict) -> str:
    if not bench:
        return "_No benchmark data found. Run `scripts/benchmark_models.py` first._\n"

    def _val(d: dict, *keys, fmt=".3f") -> str:
        for k in keys:
            if k in d:
                v = d[k]
                return format(v, fmt) if isinstance(v, float) else str(v)
        return "—"

    lines = ["## Model Comparison\n"]

    # Bias benchmark
    bias = bench.get("bias_benchmark", {})
    if bias:
        old_b = bias.get("old", {}).get("means", {})
        new_b = bias.get("new", {}).get("means", {})
        lines += [
            "### BiasBenchmark",
            "",
            "| Metric | Old (full image) | New (ROI) |",
            "|--------|-----------------|-----------|",
            f"| Edge attention | {_val(old_b, 'edge_attention_pct')} | {_val(new_b, 'edge_attention_pct')} |",
            f"| Lung attention | {_val(old_b, 'lung_attention_pct')} | {_val(new_b, 'lung_attention_pct')} |",
            f"| Center attention | {_val(old_b, 'center_attention_pct')} | {_val(new_b, 'center_attention_pct')} |",
            f"| Confidence stability | {_val(old_b, 'confidence_stability')} | {_val(new_b, 'confidence_stability')} |",
            "",
        ]

    # Stress suite
    stress = bench.get("stress_suite", {})
    if stress:
        old_s = stress.get("old", {})
        new_s = stress.get("new", {})
        lines += [
            "### Stress Suite",
            "",
            "| Metric | Old (full image) | New (ROI) |",
            "|--------|-----------------|-----------|",
            f"| Flip rate | {_val(old_s, 'flip_rate')} | {_val(new_s, 'flip_rate')} |",
            f"| Mean confidence delta | {_val(old_s, 'mean_confidence_delta'):} | {_val(new_s, 'mean_confidence_delta')} |",
            "",
        ]

    # GradCAM audit
    audit = bench.get("gradcam_audit", {})
    if audit:
        old_a = audit.get("old", {})
        new_a = audit.get("new", {})
        lines += [
            "### GradCAM Audit",
            "",
            "| Metric | Old (full image) | New (ROI) |",
            "|--------|-----------------|-----------|",
            f"| Lung attention | {_val(old_a, 'mean_lung_attention')} | {_val(new_a, 'mean_lung_attention')} |",
            f"| Edge attention | {_val(old_a, 'mean_edge_attention')} | {_val(new_a, 'mean_edge_attention')} |",
            f"| Corner attention | {_val(old_a, 'mean_corner_attention')} | {_val(new_a, 'mean_corner_attention')} |",
            f"| Pass rate | {_val(old_a, 'pass_rate')} | {_val(new_a, 'pass_rate')} |",
            "",
        ]

    return "\n".join(lines) + "\n"


def build_report(
    benchmark_data: dict,
    seg_data: dict,
    timestamp: str,
) -> str:
    sections = [
        f"# Phase 3 Migration Report\n\n_Generated: {timestamp}_\n",
        ARCHITECTURE_SECTION,
        REMOVED_MECHANISMS,
        REMAINING_SAFETY,
        SEGMENTATION_PIPELINE,
        _seg_section(seg_data),
        _benchmark_section(benchmark_data),
        "## Next Steps\n\n"
        "1. Run `scripts/benchmark_models.py` with held-out test images to confirm "
        "lung attention improvement.\n"
        "2. Inspect QA visualisations in `data/segmented_dataset/qa_visualisations/` "
        "for segmentation quality.\n"
        "3. Monitor `ReliabilityStats.extreme_confidence_rate` in production — "
        "should drop as the ROI model generalises better.\n"
        "4. If `low_trust` rate exceeds 15%, revisit `LungSegmenter` "
        "plausibility thresholds.\n",
    ]
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 3 migration report")
    parser.add_argument(
        "--benchmark", type=Path, default=None,
        help="Path to benchmark JSON (from benchmark_models.py)",
    )
    parser.add_argument(
        "--seg-report", type=Path, default=None,
        help="Path to segmentation_report.json (from regenerate_segmented_dataset.py)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/migration_report.md"),
    )
    args = parser.parse_args()

    benchmark_data = _load_json(args.benchmark)
    seg_data = _load_json(args.seg_report)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    report = build_report(benchmark_data, seg_data, timestamp)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Migration report written to: {args.output}")


if __name__ == "__main__":
    main()
