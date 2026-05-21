"""
Dataset manifest — the single source of truth for all dataset images.

The manifest is a JSON file with this structure:
    {
        "meta": DatasetManifestMeta,
        "images": { image_id: ImageRecord, ... }
    }

The manifest is the authoritative record for:
  - which images are in the dataset
  - their class assignment
  - their quality scores and flags
  - their split (train/val/test) assignment
  - their provenance

All dataset-modifying operations (add, remove, update split) go through
this class so the manifest stays consistent.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from app.modules.vision.datasets.schema import (
    DatasetManifestMeta,
    ImageClass,
    ImageRecord,
    QualityFlag,
    Split,
)

logger = logging.getLogger(__name__)

_MANIFEST_FILENAME = "manifest.json"


class DatasetManifest:
    """
    Manages the dataset manifest file for a single dataset version.

    Parameters
    ----------
    manifest_dir : Path
        Directory that contains (or will contain) ``manifest.json``.
    version : str
        Dataset version string (e.g. "v1").
    """

    def __init__(self, manifest_dir: Path, version: str) -> None:
        self.manifest_dir = Path(manifest_dir)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.version = version
        self._path = self.manifest_dir / _MANIFEST_FILENAME
        self._meta: DatasetManifestMeta | None = None
        self._images: dict[str, ImageRecord] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load manifest from disk. Idempotent."""
        if not self._path.exists():
            logger.info(
                "DatasetManifest: no manifest at %s — starting fresh.", self._path
            )
            self._meta = DatasetManifestMeta(version=self.version)
            return

        raw = json.loads(self._path.read_text(encoding="utf-8"))
        self._meta = DatasetManifestMeta.model_validate(raw["meta"])
        self._images = {
            k: ImageRecord.model_validate(v)
            for k, v in raw.get("images", {}).items()
        }
        logger.info(
            "DatasetManifest: loaded %d images from %s", len(self._images), self._path
        )

    def save(self) -> None:
        """Persist manifest to disk."""
        if self._meta is None:
            self._meta = DatasetManifestMeta(version=self.version)
        self._meta.updated_at = datetime.now(tz=timezone.utc)

        payload = {
            "meta": json.loads(self._meta.model_dump_json()),
            "images": {
                k: json.loads(v.model_dump_json())
                for k, v in self._images.items()
            },
        }
        self._path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.debug("DatasetManifest: saved %d images → %s", len(self._images), self._path)

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def add(self, record: ImageRecord, overwrite: bool = False) -> bool:
        """
        Add an ImageRecord. Returns True if added, False if already present.

        Parameters
        ----------
        overwrite : bool
            If True, replace any existing record with the same image_id.
        """
        if record.image_id in self._images and not overwrite:
            logger.debug(
                "DatasetManifest.add: skipped %s (already exists)", record.image_id[:12]
            )
            return False
        self._images[record.image_id] = record
        return True

    def remove(self, image_id: str) -> bool:
        """Remove a record by image_id. Returns True if it existed."""
        if image_id not in self._images:
            return False
        del self._images[image_id]
        return True

    def get(self, image_id: str) -> ImageRecord | None:
        return self._images.get(image_id)

    def update(self, image_id: str, **kwargs) -> bool:
        """Patch specific fields on an existing record."""
        record = self._images.get(image_id)
        if record is None:
            return False
        updated = record.model_copy(update=kwargs)
        self._images[image_id] = updated
        return True

    def assign_split(self, image_id: str, split: Split) -> bool:
        return self.update(image_id, split=split)

    # ── Queries ───────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._images)

    def __contains__(self, image_id: str) -> bool:
        return image_id in self._images

    def __iter__(self) -> Iterator[ImageRecord]:
        return iter(self._images.values())

    def all(self) -> list[ImageRecord]:
        return list(self._images.values())

    def by_class(self, class_name: ImageClass) -> list[ImageRecord]:
        return [r for r in self._images.values() if r.class_name == class_name]

    def by_split(self, split: Split) -> list[ImageRecord]:
        return [r for r in self._images.values() if r.split == split]

    def by_class_and_split(
        self, class_name: ImageClass, split: Split
    ) -> list[ImageRecord]:
        return [
            r for r in self._images.values()
            if r.class_name == class_name and r.split == split
        ]

    def filter(self, predicate: Callable[[ImageRecord], bool]) -> list[ImageRecord]:
        return [r for r in self._images.values() if predicate(r)]

    def accepted_quality(self) -> list[ImageRecord]:
        """Records that passed quality validation and are not marked as duplicates."""
        return self.filter(lambda r: r.is_acceptable_quality and r.validated)

    def content_hashes(self) -> set[str]:
        return {r.content_hash for r in self._images.values()}

    def perceptual_hashes(self) -> list[str]:
        return [r.perceptual_hash for r in self._images.values()]

    # ── Statistics ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """
        Compute class distribution, split distribution, quality summary,
        and flag frequency counts.
        """
        records = list(self._images.values())
        n = len(records)
        if n == 0:
            return {"total": 0}

        class_counts = Counter(r.class_name.value for r in records)
        split_counts = Counter(r.split.value for r in records)
        flag_counts: Counter[str] = Counter()
        for r in records:
            for f in r.quality_flags:
                flag_counts[f.value] += 1

        quality_scores = [r.quality_score for r in records]
        validated = sum(1 for r in records if r.validated)
        accepted = sum(1 for r in records if r.is_acceptable_quality)
        duplicates = sum(
            1 for r in records if QualityFlag.POSSIBLE_DUPLICATE in r.quality_flags
        )

        # Per-class per-split breakdown
        breakdown: dict[str, dict[str, int]] = {}
        for cls in ImageClass:
            breakdown[cls.value] = {}
            for spl in Split:
                count = len(self.by_class_and_split(cls, spl))
                if count > 0:
                    breakdown[cls.value][spl.value] = count

        return {
            "total": n,
            "validated": validated,
            "accepted": accepted,
            "duplicates_flagged": duplicates,
            "class_distribution": dict(class_counts),
            "split_distribution": dict(split_counts),
            "class_split_breakdown": breakdown,
            "quality": {
                "mean": round(sum(quality_scores) / n, 4),
                "min": round(min(quality_scores), 4),
                "max": round(max(quality_scores), 4),
            },
            "flag_frequency": dict(flag_counts),
        }

    def log_stats(self) -> None:
        s = self.stats()
        logger.info("─── Dataset Manifest Stats ──────────────────────")
        logger.info("  Version   : %s", self.version)
        logger.info("  Total     : %d", s["total"])
        logger.info("  Validated : %d", s.get("validated", 0))
        logger.info("  Accepted  : %d", s.get("accepted", 0))
        logger.info("  Duplicates: %d", s.get("duplicates_flagged", 0))
        for cls, count in s.get("class_distribution", {}).items():
            logger.info("  %-15s : %d", cls, count)
        logger.info("─────────────────────────────────────────────────")

    # ── Integrity ─────────────────────────────────────────────────────────────

    def check_integrity(self, images_root: Path) -> dict:
        """
        Verify that every record in the manifest has a corresponding file on disk.

        Parameters
        ----------
        images_root : Path
            Root directory under which class subdirectories are found.

        Returns
        -------
        dict with keys "missing" (list of image_ids whose files are absent)
        and "orphaned" (files on disk not present in the manifest).
        """
        missing: list[str] = []
        manifest_filenames = set()

        for record in self._images.values():
            expected_path = images_root / record.class_name.value / record.filename
            if not expected_path.exists():
                missing.append(record.image_id)
                logger.warning(
                    "DatasetManifest: file missing for %s: %s",
                    record.image_id[:12],
                    expected_path,
                )
            manifest_filenames.add(record.filename)

        orphaned: list[str] = []
        _IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        for cls in ImageClass:
            cls_dir = images_root / cls.value
            if not cls_dir.exists():
                continue
            for p in cls_dir.iterdir():
                if p.suffix.lower() in _IMG_EXT and p.name not in manifest_filenames:
                    orphaned.append(str(p))
                    logger.debug("DatasetManifest: orphaned file: %s", p)

        result = {"missing": missing, "orphaned": orphaned}
        if missing or orphaned:
            logger.warning(
                "DatasetManifest integrity: %d missing, %d orphaned",
                len(missing),
                len(orphaned),
            )
        else:
            logger.info("DatasetManifest integrity: OK")

        return result
