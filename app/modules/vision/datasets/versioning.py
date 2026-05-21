"""
Dataset version management.

Dataset versions are immutable once created. The workflow is:

  1. Collect raw images into ``data/vision/raw/<class>/``
  2. Run ``prepare_dataset.py`` — validates, deduplicates, splits, saves to
     ``data/vision/datasets/v<N>/``
  3. Register the version with DatasetVersionManager.
  4. Train on a specific version (never the "latest" implicitly — always
     record the version string alongside the trained model).

Version naming
--------------
  v1, v2, v3, ...  — sequential integers, zero-padded to no minimum width
  Custom strings are allowed for named milestones (e.g., "v1_baseline").

Immutability guarantee
----------------------
Once a version is registered, its manifest and processed images should
not be modified. To add images, create a new version that references the
parent. This guarantees reproducibility: a model trained on v2 can always
be re-trained from scratch using the same data.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_INDEX_FILENAME = "versions.json"


class DatasetVersionManager:
    """
    Tracks all dataset versions in a root directory.

    Parameters
    ----------
    base_dir : Path
        Root directory containing versioned dataset sub-directories.
        Typically ``data/vision/datasets/``.
    """

    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.base_dir / _INDEX_FILENAME
        self._index: dict = self._load_index()

    # ── Index management ──────────────────────────────────────────────────────

    def _load_index(self) -> dict:
        if self._index_path.exists():
            return json.loads(self._index_path.read_text(encoding="utf-8"))
        return {"versions": {}}

    def _save_index(self) -> None:
        self._index_path.write_text(
            json.dumps(self._index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Version lifecycle ─────────────────────────────────────────────────────

    def next_version(self) -> str:
        """Return the next sequential version string (e.g. 'v3' after 'v2')."""
        numeric = [
            int(v[1:]) for v in self._index["versions"]
            if v.startswith("v") and v[1:].isdigit()
        ]
        n = max(numeric, default=0) + 1
        return f"v{n}"

    def register(
        self,
        version: str,
        stats: dict,
        parent_version: str | None = None,
        description: str = "",
        git_commit: str | None = None,
    ) -> None:
        """
        Register a new dataset version in the index.

        Parameters
        ----------
        version : str
            Version string (e.g. "v1").
        stats : dict
            Output of DatasetManifest.stats() for this version.
        parent_version : str | None
            The version this was derived from (for incremental datasets).
        description : str
            Human-readable description of what changed in this version.
        git_commit : str | None
            Git commit SHA at the time of dataset creation.
        """
        self._index["versions"][version] = {
            "version": version,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "parent_version": parent_version,
            "description": description,
            "git_commit": git_commit,
            "stats": stats,
            "path": str(self.version_dir(version)),
        }
        self._save_index()
        logger.info("DatasetVersionManager: registered version %s", version)

    def version_dir(self, version: str) -> Path:
        return self.base_dir / version

    def processed_dir(self, version: str) -> Path:
        return self.version_dir(version) / "processed"

    def splits_dir(self, version: str) -> Path:
        return self.version_dir(version) / "splits"

    def metadata_dir(self, version: str) -> Path:
        return self.version_dir(version) / "metadata"

    def list_versions(self) -> list[str]:
        """All registered versions, newest first."""
        versions = list(self._index["versions"].keys())
        return sorted(versions, reverse=True)

    def latest_version(self) -> str | None:
        versions = self.list_versions()
        return versions[0] if versions else None

    def get_version_info(self, version: str) -> dict | None:
        return self._index["versions"].get(version)

    def exists(self, version: str) -> bool:
        return version in self._index["versions"]

    # ── Directory initialization ──────────────────────────────────────────────

    def init_version_dirs(self, version: str) -> dict[str, Path]:
        """
        Create the directory tree for a new version.

        Returns a dict mapping logical names to their Paths.
        """
        v_dir = self.version_dir(version)
        dirs = {
            "root":      v_dir,
            "raw":       v_dir / "raw",
            "processed": v_dir / "processed",
            "splits":    v_dir / "splits",
            "metadata":  v_dir / "metadata",
        }

        for cls in ("related", "unrelated", "hard_negative"):
            dirs[f"processed_{cls}"] = dirs["processed"] / cls
            for split in ("train", "val", "test"):
                dirs[f"splits_{split}_{cls}"] = dirs["splits"] / split / cls

        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)

        logger.info("DatasetVersionManager: initialized dirs for %s at %s", version, v_dir)
        return dirs

    # ── Comparison ────────────────────────────────────────────────────────────

    def compare(self, version_a: str, version_b: str) -> dict:
        """
        Produce a side-by-side comparison of two versions' stats.

        Useful for validating that a new version improves on the previous.
        """
        info_a = self.get_version_info(version_a)
        info_b = self.get_version_info(version_b)

        if not info_a or not info_b:
            return {"error": f"One or both versions not found: {version_a}, {version_b}"}

        stats_a = info_a.get("stats", {})
        stats_b = info_b.get("stats", {})

        def _delta(a, b):
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return round(b - a, 4)
            return "N/A"

        return {
            "version_a": version_a,
            "version_b": version_b,
            "total_images": {
                version_a: stats_a.get("total"),
                version_b: stats_b.get("total"),
                "delta": _delta(stats_a.get("total", 0), stats_b.get("total", 0)),
            },
            "class_distribution": {
                "a": stats_a.get("class_distribution", {}),
                "b": stats_b.get("class_distribution", {}),
            },
            "quality_mean": {
                version_a: stats_a.get("quality", {}).get("mean"),
                version_b: stats_b.get("quality", {}).get("mean"),
            },
        }
