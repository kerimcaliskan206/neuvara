"""
Training dataset for pre-segmented lung ROI crops.

This is the Phase-2 replacement for ImageFolderDataset in the lung-training
pipeline.  It loads crops that were written by the offline regeneration script
(scripts/regenerate_segmented_dataset.py) and enforces a hard contract:

    The classifier sees ONLY segmented lung ROI crops.
    No fallback to original full images.

If a sidecar telemetry JSON is present, its data drives train-time telemetry
reporting.  If it is absent the sample is still valid (some datasets may not
have sidecars), but will not contribute to telemetry stats.

Expected directory layout
--------------------------
    <root>/
    ├── train/
    │   ├── healthy_xray/
    │   │   ├── img001.jpg
    │   │   ├── img001_telemetry.json    ← written by regeneration script
    │   │   └── ...
    │   └── pneumonia_xray/
    │       └── ...
    └── val/
        └── ...

Telemetry summary
-----------------
Call `dataset.telemetry_summary()` to get aggregate stats over the split.
These are computed from on-disk sidecars at construction time and are safe
to call even with multi-worker DataLoaders.
"""
from __future__ import annotations

import json
import logging
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ── Telemetry summary ─────────────────────────────────────────────────────────


@dataclass
class DatasetTelemetrySummary:
    """Aggregate segmentation metrics for one dataset split."""

    n_samples: int = 0
    n_with_telemetry: int = 0

    # lung_area_pct distribution
    lung_area_pct_mean: float = 0.0
    lung_area_pct_std: float = 0.0
    lung_area_pct_min: float = 0.0
    lung_area_pct_max: float = 0.0

    # crop_ratio distribution
    crop_ratio_mean: float = 0.0
    crop_ratio_std: float = 0.0

    # ROI dimensions
    roi_width_mean: float = 0.0
    roi_height_mean: float = 0.0

    # Boolean counts
    border_removed_count: int = 0
    border_removed_pct: float = 0.0

    # Segmentation quality distribution
    quality_counts: dict[str, int] = field(default_factory=dict)

    # Left/right ROI balance: mean normalized ROI center X (0=left, 1=right)
    roi_center_x_mean: float = 0.5
    roi_center_x_std: float = 0.0

    # Class distribution from the split
    class_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "n_with_telemetry": self.n_with_telemetry,
            "telemetry_coverage_pct": round(
                100 * self.n_with_telemetry / max(self.n_samples, 1), 1
            ),
            "lung_area_pct": {
                "mean": round(self.lung_area_pct_mean, 4),
                "std": round(self.lung_area_pct_std, 4),
                "min": round(self.lung_area_pct_min, 4),
                "max": round(self.lung_area_pct_max, 4),
            },
            "crop_ratio": {
                "mean": round(self.crop_ratio_mean, 4),
                "std": round(self.crop_ratio_std, 4),
            },
            "roi_size": {
                "width_mean": round(self.roi_width_mean, 1),
                "height_mean": round(self.roi_height_mean, 1),
            },
            "border_removed_pct": round(self.border_removed_pct, 1),
            "quality_distribution": self.quality_counts,
            "left_right_balance": {
                "roi_center_x_mean": round(self.roi_center_x_mean, 4),
                "roi_center_x_std": round(self.roi_center_x_std, 4),
            },
            "class_counts": self.class_counts,
        }

    def log(self, split: str = "") -> None:
        tag = f"[{split}] " if split else ""
        logger.info(
            "%sTelemetry: n=%d | lung_area=%.3f±%.3f | "
            "crop_ratio=%.3f | border_removed=%.1f%% | "
            "quality=%s | roi_balance_x=%.3f±%.3f",
            tag,
            self.n_samples,
            self.lung_area_pct_mean, self.lung_area_pct_std,
            self.crop_ratio_mean,
            self.border_removed_pct,
            self.quality_counts,
            self.roi_center_x_mean, self.roi_center_x_std,
        )


# ── Segmented dataset ─────────────────────────────────────────────────────────


class SegmentedROIDataset(Dataset):
    """
    Dataset of pre-segmented lung ROI crops for training the chest-X-ray
    classifier.

    Parameters
    ----------
    root_dir : Path
        Root directory containing split sub-directories (train/, val/, test/).
    split : str
        One of "train", "val", "test".
    transform : callable, optional
        torchvision transform pipeline applied to each ROI image.
    classes : list[str] | None
        Class names in label-index order.  Auto-detected from disk if None.
    require_telemetry : bool
        When True, images without a sidecar telemetry JSON are excluded with
        a warning.  When False (default), they are included without telemetry.
    """

    def __init__(
        self,
        root_dir: Path | str,
        split: str = "train",
        transform: Optional[Callable] = None,
        classes: Optional[list[str]] = None,
        require_telemetry: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.require_telemetry = require_telemetry

        self.split_dir = self.root_dir / split
        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Segmented dataset split not found: {self.split_dir}\n"
                f"Run: python scripts/regenerate_segmented_dataset.py first."
            )

        self.classes: list[str] = classes if classes is not None else self._detect_classes()
        self.class_to_idx: dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class: dict[int, str] = {i: c for i, c in enumerate(self.classes)}

        # Parallel lists: paths and labels
        self._paths: list[Path] = []
        self._labels: list[int] = []
        self._telemetry: list[Optional[dict]] = []  # None when sidecar absent

        self._discover_samples()
        self._telemetry_summary: Optional[DatasetTelemetrySummary] = None

        if not self._paths:
            logger.warning(
                "SegmentedROIDataset [%s]: no images found in %s",
                split, self.split_dir,
            )
        else:
            logger.info(
                "SegmentedROIDataset [%s]: %d images | classes=%s | %s",
                split, len(self._paths), self.classes, self._distribution_str(),
            )

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Returns (tensor, label).

        The tensor has already had lung segmentation applied (it IS the ROI
        crop). No additional segmentation is performed here.
        """
        path = self._paths[idx]
        label = self._labels[idx]

        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    # ── Class distribution ────────────────────────────────────────────────────

    def class_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {cls: 0 for cls in self.classes}
        for label in self._labels:
            counts[self.idx_to_class[label]] += 1
        return counts

    def is_balanced(self, tolerance: float = 0.2) -> bool:
        dist = self.class_distribution()
        total = sum(dist.values())
        if total == 0:
            return True
        return min(dist.values()) / max(dist.values()) >= (1 - tolerance)

    # ── Telemetry ─────────────────────────────────────────────────────────────

    def telemetry_summary(self) -> DatasetTelemetrySummary:
        """
        Compute and return aggregate segmentation metrics for this split.

        Computed once and cached; subsequent calls return the cached result.
        """
        if self._telemetry_summary is not None:
            return self._telemetry_summary
        self._telemetry_summary = self._compute_telemetry_summary()
        return self._telemetry_summary

    def _compute_telemetry_summary(self) -> DatasetTelemetrySummary:
        summaries_with_data = [t for t in self._telemetry if t is not None]
        n = len(self._paths)
        nw = len(summaries_with_data)

        if not summaries_with_data:
            return DatasetTelemetrySummary(
                n_samples=n,
                class_counts=self.class_distribution(),
            )

        lung_areas = [t["lung_area_pct"] for t in summaries_with_data if "lung_area_pct" in t]
        crop_ratios = [t["crop_ratio"] for t in summaries_with_data if "crop_ratio" in t]
        roi_widths = [t["roi_width"] for t in summaries_with_data if "roi_width" in t]
        roi_heights = [t["roi_height"] for t in summaries_with_data if "roi_height" in t]
        border_removed = [t["border_removed"] for t in summaries_with_data if "border_removed" in t]
        qualities = [t["quality"] for t in summaries_with_data if "quality" in t]

        # Left/right balance: roi_center_x = (x1 + x2) / 2 / original_width
        roi_centers_x = [
            t["roi_center_x"] for t in summaries_with_data
            if "roi_center_x" in t
        ]

        def _safe_stats(vals: list) -> tuple[float, float, float, float]:
            if not vals:
                return 0.0, 0.0, 0.0, 0.0
            return (
                statistics.mean(vals),
                statistics.stdev(vals) if len(vals) > 1 else 0.0,
                min(vals),
                max(vals),
            )

        lap_mean, lap_std, lap_min, lap_max = _safe_stats(lung_areas)
        cr_mean, cr_std, _, _ = _safe_stats(crop_ratios)
        cx_mean, cx_std, _, _ = _safe_stats(roi_centers_x)

        quality_counter: Counter[str] = Counter(qualities)

        return DatasetTelemetrySummary(
            n_samples=n,
            n_with_telemetry=nw,
            lung_area_pct_mean=lap_mean,
            lung_area_pct_std=lap_std,
            lung_area_pct_min=lap_min,
            lung_area_pct_max=lap_max,
            crop_ratio_mean=cr_mean,
            crop_ratio_std=cr_std,
            roi_width_mean=statistics.mean(roi_widths) if roi_widths else 0.0,
            roi_height_mean=statistics.mean(roi_heights) if roi_heights else 0.0,
            border_removed_count=sum(1 for b in border_removed if b),
            border_removed_pct=100.0 * sum(1 for b in border_removed if b) / max(len(border_removed), 1),
            quality_counts=dict(quality_counter),
            roi_center_x_mean=cx_mean,
            roi_center_x_std=cx_std,
            class_counts=self.class_distribution(),
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _detect_classes(self) -> list[str]:
        """Infer class names from subdirectory names under the split directory."""
        subdirs = sorted(
            d.name for d in self.split_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        if not subdirs:
            raise RuntimeError(
                f"No class subdirectories found in {self.split_dir}"
            )
        return subdirs

    def _discover_samples(self) -> None:
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                logger.warning(
                    "SegmentedROIDataset: class directory missing: %s", class_dir
                )
                continue

            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                    continue

                tel = self._load_sidecar(img_path)
                if self.require_telemetry and tel is None:
                    logger.warning(
                        "SegmentedROIDataset: telemetry required but missing for %s — skipping",
                        img_path.name,
                    )
                    continue

                self._paths.append(img_path)
                self._labels.append(class_idx)
                self._telemetry.append(tel)

    @staticmethod
    def _load_sidecar(img_path: Path) -> Optional[dict]:
        """Load <stem>_telemetry.json if present, else return None."""
        sidecar = img_path.with_name(img_path.stem + "_telemetry.json")
        if not sidecar.exists():
            return None
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.debug("SegmentedROIDataset: could not read sidecar %s", sidecar)
            return None

    def _distribution_str(self) -> str:
        dist = self.class_distribution()
        total = max(len(self._paths), 1)
        return " | ".join(
            f"{cls}={n} ({100 * n // total}%)" for cls, n in dist.items()
        )
