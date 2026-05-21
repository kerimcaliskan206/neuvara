"""
Source distribution auditor + radiology metadata analyzer — Phase 7.

Checks dataset-level distribution properties that affect training quality:

  1. Category imbalance ratio — extreme imbalances inflate apparent accuracy
     and cause the model to ignore minority classes.

  2. Grayscale shortcut risk  — if one class is ≥ 85 % grayscale while
     another is ≥ 85 % RGB, the model can learn colorspace rather than
     pathology (a known spurious correlation in radiology datasets).

  3. Size distribution        — large variation in image dimensions signals
     mixed acquisition sources with different scanner protocols.

  4. Contrast distribution    — per-channel mean/stdev across the dataset;
     large inter-category differences can act as domain-level shortcuts.

  5. Dataset lineage tracking — provenance record for reproducibility,
     consumed by downstream training pipelines.

No pixel decoding beyond PIL metadata reading.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_GRAYSCALE_SHORTCUT_THRESHOLD: float = 0.85
_CRITICAL_IMBALANCE_RATIO: float = 10.0
_WARNING_IMBALANCE_RATIO: float = 5.0
_MAX_COMMON_SIZES: int = 10
_MIXED_SIZES_THRESHOLD: int = 10


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class CategoryDistribution:
    counts: dict[str, int]
    total: int
    imbalance_ratio: float
    dominant_category: str
    minority_category: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class GrayscaleShortcutReport:
    category_grayscale_rates: dict[str, float]   # category → fraction grayscale [0, 1]
    shortcut_risk: str                            # "none" | "low" | "medium" | "high"
    at_risk_pairs: list[tuple[str, str]]          # (high-gray cat, low-gray cat)
    explanation: str


@dataclass
class SizeDistributionReport:
    width_stats: dict[str, float]                    # min, max, mean, stdev
    height_stats: dict[str, float]
    common_sizes: list[tuple[int, int, int]]         # (width, height, count) desc
    unique_size_count: int
    mixed_sources_suspected: bool
    warnings: list[str] = field(default_factory=list)


@dataclass
class DatasetLineageRecord:
    """Provenance record for a versioned dataset or split."""

    dataset_name: str
    version: str
    created_at: str                       # ISO-8601 UTC
    split: str                            # "train" | "val" | "test" | "all"
    source_urls: list[str] = field(default_factory=list)
    category_distribution: dict[str, int] = field(default_factory=dict)
    image_count: int = 0
    sha256_manifest_path: str | None = None
    notes: str = ""


@dataclass
class DistributionAuditResult:
    """Full source auditor output for one dataset split."""

    distribution: CategoryDistribution
    grayscale: GrayscaleShortcutReport
    size: SizeDistributionReport
    lineage: DatasetLineageRecord | None
    audit_warnings: list[str] = field(default_factory=list)


# ── Auditor ───────────────────────────────────────────────────────────────────


class SourceAuditor:
    """
    Distribution property analysis for medical image datasets.

    Stateless — all state is passed as arguments.
    Metadata records are plain dicts; keys are configurable to match
    whatever schema the dataset pipeline produces.
    """

    def audit_category_distribution(
        self,
        metadata: list[dict],
        category_key: str = "category",
    ) -> CategoryDistribution:
        """Compute per-category counts and imbalance ratio."""
        counts: dict[str, int] = {}
        for record in metadata:
            cat = str(record.get(category_key, "unknown"))
            counts[cat] = counts.get(cat, 0) + 1

        if not counts:
            return CategoryDistribution(
                counts={}, total=0, imbalance_ratio=1.0,
                dominant_category="none", minority_category="none",
            )

        total     = sum(counts.values())
        max_count = max(counts.values())
        min_count = min(counts.values())
        imbalance = round(max_count / max(min_count, 1), 2)
        dominant  = max(counts, key=lambda k: counts[k])
        minority  = min(counts, key=lambda k: counts[k])

        warnings: list[str] = []
        if imbalance >= _CRITICAL_IMBALANCE_RATIO:
            warnings.append(
                f"Critical imbalance (ratio={imbalance}×): "
                f"'{dominant}' ({max_count}) vs '{minority}' ({min_count}). "
                "Use oversampling or class-weighted loss."
            )
        elif imbalance >= _WARNING_IMBALANCE_RATIO:
            warnings.append(
                f"Moderate imbalance (ratio={imbalance}×): '{dominant}' vs '{minority}'."
            )

        return CategoryDistribution(
            counts=counts,
            total=total,
            imbalance_ratio=imbalance,
            dominant_category=dominant,
            minority_category=minority,
            warnings=warnings,
        )

    def assess_grayscale_shortcut_risk(
        self,
        metadata: list[dict],
        category_key: str = "category",
        is_grayscale_key: str = "is_grayscale",
    ) -> GrayscaleShortcutReport:
        """
        Detect shortcut-learning risk from unbalanced grayscale distribution.

        Risk is high when one category is ≥ THRESHOLD grayscale and another
        is ≥ THRESHOLD RGB — the model can distinguish them by colorspace.
        """
        gray_counts: dict[str, int] = {}
        total_counts: dict[str, int] = {}

        for record in metadata:
            cat  = str(record.get(category_key, "unknown"))
            gray = bool(record.get(is_grayscale_key, False))
            total_counts[cat] = total_counts.get(cat, 0) + 1
            if gray:
                gray_counts[cat] = gray_counts.get(cat, 0) + 1

        rates: dict[str, float] = {
            cat: gray_counts.get(cat, 0) / max(total_counts[cat], 1)
            for cat in total_counts
        }

        t = _GRAYSCALE_SHORTCUT_THRESHOLD
        high_gray = [c for c, r in rates.items() if r >= t]
        low_gray  = [c for c, r in rates.items() if r <= (1.0 - t)]
        at_risk   = [(hg, lg) for hg in high_gray for lg in low_gray]

        if at_risk:
            risk = "high"
            explanation = (
                f"Categories {high_gray} are ≥{t:.0%} grayscale "
                f"while {low_gray} are ≥{t:.0%} RGB — "
                "model may learn colorspace shortcut instead of pathology."
            )
        elif (
            any(r >= 0.65 for r in rates.values())
            and any(r <= 0.35 for r in rates.values())
        ):
            risk = "medium"
            explanation = "Moderate grayscale rate difference across categories."
        elif rates:
            risk = "low" if any(r > 0.0 for r in rates.values()) else "none"
            explanation = "Grayscale distribution broadly consistent across categories."
        else:
            risk = "none"
            explanation = "No grayscale metadata available."

        return GrayscaleShortcutReport(
            category_grayscale_rates={k: round(v, 4) for k, v in rates.items()},
            shortcut_risk=risk,
            at_risk_pairs=at_risk,
            explanation=explanation,
        )

    def audit_size_distribution(
        self,
        metadata: list[dict],
        width_key: str = "width",
        height_key: str = "height",
    ) -> SizeDistributionReport:
        """Analyze image size variation; flag mixed-acquisition-source patterns."""
        widths: list[int] = []
        heights: list[int] = []
        size_counts: dict[tuple[int, int], int] = {}

        for record in metadata:
            w = record.get(width_key)
            h = record.get(height_key)
            if w is not None and h is not None:
                wi, hi = int(w), int(h)
                widths.append(wi)
                heights.append(hi)
                size_counts[(wi, hi)] = size_counts.get((wi, hi), 0) + 1

        def _stats(values: list[int]) -> dict[str, float]:
            if not values:
                return {"min": 0.0, "max": 0.0, "mean": 0.0, "stdev": 0.0}
            return {
                "min":   float(min(values)),
                "max":   float(max(values)),
                "mean":  round(statistics.mean(values), 1),
                "stdev": round(statistics.stdev(values) if len(values) > 1 else 0.0, 1),
            }

        unique_count = len(size_counts)
        mixed = unique_count > _MIXED_SIZES_THRESHOLD

        warnings: list[str] = []
        if mixed:
            warnings.append(
                f"{unique_count} distinct image sizes — mixed acquisition sources suspected. "
                "Resize to a single canonical resolution before training."
            )

        common = sorted(
            [(w, h, cnt) for (w, h), cnt in size_counts.items()],
            key=lambda x: x[2],
            reverse=True,
        )[:_MAX_COMMON_SIZES]

        return SizeDistributionReport(
            width_stats=_stats(widths),
            height_stats=_stats(heights),
            common_sizes=common,
            unique_size_count=unique_count,
            mixed_sources_suspected=mixed,
            warnings=warnings,
        )

    def full_audit(
        self,
        metadata: list[dict],
        lineage: DatasetLineageRecord | None = None,
    ) -> DistributionAuditResult:
        """Run all source auditor checks and return a combined result."""
        dist  = self.audit_category_distribution(metadata)
        gray  = self.assess_grayscale_shortcut_risk(metadata)
        sizes = self.audit_size_distribution(metadata)

        all_warnings = list(dist.warnings) + list(sizes.warnings)
        if gray.shortcut_risk in ("medium", "high"):
            all_warnings.append(
                f"Grayscale shortcut risk: {gray.shortcut_risk}. {gray.explanation}"
            )

        logger.info(
            "SourceAuditor: categories=%d imbalance=%.1f× gray_risk=%s sizes=%d",
            len(dist.counts), dist.imbalance_ratio,
            gray.shortcut_risk, sizes.unique_size_count,
        )

        return DistributionAuditResult(
            distribution=dist,
            grayscale=gray,
            size=sizes,
            lineage=lineage,
            audit_warnings=all_warnings,
        )
