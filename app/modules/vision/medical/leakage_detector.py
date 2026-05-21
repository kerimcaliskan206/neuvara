"""
Leakage detector + cross-split contamination checker — Phase 7.

Identifies data integrity problems that inflate evaluation metrics:

  1. Duplicate hashes     — exact-duplicate images within or across splits
                            (same pixel content re-used in train and test).
  2. Cross-split leakage  — images present in both train and val/test,
                            detected by SHA-256 hash equality.
  3. Source leakage       — same hospital / acquisition source in both
                            train and test, enabling domain-adaptation
                            shortcuts that don't generalise.

All checks are hash / metadata based — no pixel decoding, no ML inference.
Input: {Path: sha256} maps from dataset_validator.build_hash_map().
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_HIGH_SOURCE_OVERLAP: float = 0.40
_MEDIUM_SOURCE_OVERLAP: float = 0.20


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class DuplicateGroup:
    """A set of image files that share the same SHA-256 hash."""

    sha256: str
    paths: list[Path]

    @property
    def count(self) -> int:
        return len(self.paths)


@dataclass
class CrossSplitLeak:
    """An image that appears in both the train split and val/test."""

    sha256: str
    train_path: Path
    contaminated_path: Path
    contaminated_split: str    # "val" | "test"


@dataclass
class SourceLeak:
    """A hospital / source that contributes images to both train and test."""

    source_id: str
    train_count: int
    test_count: int
    risk_level: str            # "low" | "medium" | "high"


@dataclass
class LeakageReport:
    """Aggregated leakage analysis across all splits."""

    duplicates: list[DuplicateGroup]
    cross_split_leaks: list[CrossSplitLeak]
    source_leaks: list[SourceLeak]
    contamination_risk: str           # "none" | "low" | "medium" | "high"
    total_images_checked: int
    summary: list[str] = field(default_factory=list)


# ── Detector ──────────────────────────────────────────────────────────────────


class LeakageDetector:
    """
    Hash-based leakage detection for medical image datasets.

    Stateless — all inputs passed per-call.
    """

    def detect_duplicates(
        self,
        image_hashes: dict[Path, str],
    ) -> list[DuplicateGroup]:
        """Find groups of images with identical SHA-256 hashes."""
        hash_to_paths: dict[str, list[Path]] = defaultdict(list)
        for path, sha256 in image_hashes.items():
            hash_to_paths[sha256].append(path)

        groups = [
            DuplicateGroup(sha256=h, paths=sorted(paths))
            for h, paths in hash_to_paths.items()
            if len(paths) > 1
        ]
        logger.info("Duplicate groups found: %d", len(groups))
        return groups

    def detect_cross_split_leakage(
        self,
        train_hashes: dict[Path, str],
        val_hashes: dict[Path, str] | None = None,
        test_hashes: dict[Path, str] | None = None,
    ) -> list[CrossSplitLeak]:
        """
        Detect images in train that also appear in val or test (by hash).

        Builds an inverse map of train hashes and probes each non-train split.
        """
        train_hash_to_path: dict[str, Path] = {v: k for k, v in train_hashes.items()}
        leaks: list[CrossSplitLeak] = []

        for split_name, split_hashes in [("val", val_hashes), ("test", test_hashes)]:
            if not split_hashes:
                continue
            for path, sha256 in split_hashes.items():
                if sha256 in train_hash_to_path:
                    leaks.append(CrossSplitLeak(
                        sha256=sha256,
                        train_path=train_hash_to_path[sha256],
                        contaminated_path=path,
                        contaminated_split=split_name,
                    ))

        logger.info("Cross-split leaks: %d", len(leaks))
        return leaks

    def detect_source_leakage(
        self,
        train_metadata: list[dict],
        test_metadata: list[dict],
        source_key: str = "source_id",
    ) -> list[SourceLeak]:
        """
        Detect acquisition sources (hospital, scanner, dataset) present in
        both train and test.

        Overlap ratio = test_count / (train_count + test_count):
          ≥ 0.40 → high risk
          ≥ 0.20 → medium risk
          <  0.20 → low risk
        """
        train_sources: dict[str, int] = defaultdict(int)
        test_sources: dict[str, int] = defaultdict(int)

        for record in train_metadata:
            sid = record.get(source_key)
            if sid:
                train_sources[sid] += 1
        for record in test_metadata:
            sid = record.get(source_key)
            if sid:
                test_sources[sid] += 1

        leaks: list[SourceLeak] = []
        for source_id in set(train_sources) & set(test_sources):
            train_n = train_sources[source_id]
            test_n  = test_sources[source_id]
            overlap = test_n / max(train_n + test_n, 1)
            risk = (
                "high"   if overlap >= _HIGH_SOURCE_OVERLAP   else
                "medium" if overlap >= _MEDIUM_SOURCE_OVERLAP else
                "low"
            )
            leaks.append(SourceLeak(
                source_id=source_id,
                train_count=train_n,
                test_count=test_n,
                risk_level=risk,
            ))

        logger.info("Source leaks: %d sources in common", len(leaks))
        return leaks

    def build_report(
        self,
        duplicates: list[DuplicateGroup],
        cross_split_leaks: list[CrossSplitLeak],
        source_leaks: list[SourceLeak],
        total_images: int,
    ) -> LeakageReport:
        """Aggregate findings into a LeakageReport with overall risk level."""
        high_source_risks = sum(1 for s in source_leaks if s.risk_level == "high")

        if cross_split_leaks or high_source_risks > 0:
            risk = "high"
        elif duplicates or source_leaks:
            risk = "medium"
        else:
            risk = "none"

        summary: list[str] = []
        if duplicates:
            dup_images = sum(d.count for d in duplicates)
            summary.append(
                f"{len(duplicates)} duplicate group(s) spanning {dup_images} images."
            )
        if cross_split_leaks:
            summary.append(
                f"{len(cross_split_leaks)} image(s) present in both train and val/test."
            )
        if source_leaks:
            high_count = sum(1 for s in source_leaks if s.risk_level == "high")
            summary.append(
                f"{len(source_leaks)} source(s) in train+test overlap "
                f"({high_count} high-risk)."
            )
        if not summary:
            summary.append("No leakage detected.")

        return LeakageReport(
            duplicates=duplicates,
            cross_split_leaks=cross_split_leaks,
            source_leaks=source_leaks,
            contamination_risk=risk,
            total_images_checked=total_images,
            summary=summary,
        )
