import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib

from app.modules.ml.config import MLConfig, ml_config

logger = logging.getLogger(__name__)


class ModelStore:
    """
    Versioned model and artifact storage.

    Directory layout:
        models/
        └── v{version}/
            ├── {name}.joblib          ← fitted model / pipeline
            └── {name}.meta.json       ← optional metadata sidecar
    """

    def __init__(self, config: MLConfig = ml_config) -> None:
        self.base_dir = config.storage.models_dir

    # ── Save ─────────────────────────────────────────────────────────────────

    def save(self, artifact, name: str, version: str | None = None) -> Path:
        version = version or self._new_version()
        path = self._artifact_path(name, version)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, path)
        logger.info("Saved artifact → %s (v%s)", name, version)
        return path

    def save_metadata(self, name: str, version: str, metadata: dict) -> Path:
        """Save a JSON sidecar alongside the joblib artifact."""
        path = self._meta_path(name, version)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2))
        logger.info("Saved metadata → %s.meta.json (v%s)", name, version)
        return path

    # ── Load ─────────────────────────────────────────────────────────────────

    def load(self, name: str, version: str) -> object:
        path = self._artifact_path(name, version)
        if not path.exists():
            raise FileNotFoundError(
                f"Artifact not found: {path}\n"
                f"Available versions: {self.list_versions(name)}"
            )
        artifact = joblib.load(path)
        logger.info("Loaded artifact → %s (v%s)", name, version)
        return artifact

    def load_metadata(self, name: str, version: str) -> dict:
        path = self._meta_path(name, version)
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def load_latest(self, name: str) -> tuple[object, str]:
        """Load the most recent version of an artifact. Returns (artifact, version)."""
        version = self.latest_version(name)
        if version is None:
            raise FileNotFoundError(
                f"No saved versions found for artifact '{name}' in {self.base_dir}"
            )
        return self.load(name, version), version

    # ── Discovery ─────────────────────────────────────────────────────────────

    def list_versions(self, name: str) -> list[str]:
        """Return all version directory names that contain `name`.joblib."""
        if not self.base_dir.exists():
            return []
        return sorted(
            d.name
            for d in self.base_dir.iterdir()
            if d.is_dir() and self._artifact_path(name, d.name).exists()
        )

    def latest_version(self, name: str) -> str | None:
        versions = self.list_versions(name)
        return versions[-1] if versions else None

    def list_artifacts(self, version: str) -> list[str]:
        """Return all artifact names (without .joblib) inside a version directory."""
        version_dir = self._version_dir(version)
        if not version_dir.exists():
            return []
        return sorted(p.stem for p in version_dir.glob("*.joblib"))

    def list_all_versions(self) -> list[str]:
        """Return every version directory present under base_dir."""
        if not self.base_dir.exists():
            return []
        return sorted(d.name for d in self.base_dir.iterdir() if d.is_dir())

    # ── Internal ──────────────────────────────────────────────────────────────

    def _version_dir(self, version: str) -> Path:
        # Accept both "20260514_171807" and "v20260514_171807" — always store with "v" prefix
        v = version if version.startswith("v") else f"v{version}"
        return self.base_dir / v

    def _artifact_path(self, name: str, version: str) -> Path:
        return self._version_dir(version) / f"{name}.joblib"

    def _meta_path(self, name: str, version: str) -> Path:
        return self._version_dir(version) / f"{name}.meta.json"

    @staticmethod
    def _new_version() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
