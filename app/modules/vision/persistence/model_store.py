"""
Vision model persistence.

Saves and loads trained model weights alongside a JSON metadata sidecar.
Transforms are not pickled — the metadata is sufficient to reconstruct
them deterministically.

Directory layout
----------------
    vision_models/
    └── v20260514_120000/
        ├── weights.pt           ← torch.save(model.state_dict())
        ├── metadata.json        ← architecture, classes, metrics, config
        └── calibration.json     ← temperature, ECE, threshold (optional)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class VisionModelStore:
    """
    Saves and loads trained vision models with structured versioning.

    Parameters
    ----------
    base_dir : Path
        Root directory where versioned sub-directories are created.
    """

    _WEIGHTS_FILENAME = "weights.pt"
    _METADATA_FILENAME = "metadata.json"
    _CALIBRATION_FILENAME = "calibration.json"

    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ── Saving ────────────────────────────────────────────────────────────────

    def save(
        self,
        model: torch.nn.Module,
        architecture: str,
        class_names: list[str],
        image_size: tuple[int, int],
        metrics: Optional[dict] = None,
        training_config: Optional[dict] = None,
        version: Optional[str] = None,
        calibration_temperature: float = 1.0,
        confidence_threshold: float = 0.6,
        dataset_version: Optional[str] = None,
    ) -> str:
        """
        Persist model weights and metadata.

        Parameters
        ----------
        model : nn.Module
            Trained model (state dict is extracted — no pickling of the class).
        architecture : str
            Registry name (e.g. "efficientnet_b0").
        class_names : list[str]
            Class labels in index order — must match num_classes.
        image_size : (width, height)
            Input resolution used during training.
        metrics : dict | None
            Evaluation metrics (accuracy, f1, ECE, etc.).
        training_config : dict | None
            Serialisable VisionTrainingConfig.
        version : str | None
            Override the auto-generated version string.
        calibration_temperature : float
            Temperature from TemperatureScaler.fit(). 1.0 = uncalibrated.
        confidence_threshold : float
            Recommended confidence threshold for inference rejection.
        dataset_version : str | None
            Dataset version used for training (for audit trail).

        Returns
        -------
        str — the version string (e.g. "v20260514_120000").
        """
        if version is None:
            ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
            version = f"v{ts}"
        elif not version.startswith("v"):
            version = f"v{version}"

        version_dir = self.base_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Weights
        torch.save(model.state_dict(), version_dir / self._WEIGHTS_FILENAME)

        # Metadata
        metadata = {
            "version": version,
            "architecture": architecture,
            "class_names": class_names,
            "num_classes": len(class_names),
            "image_size": list(image_size),
            "metrics": metrics or {},
            "training_config": training_config or {},
            "calibration_temperature": calibration_temperature,
            "confidence_threshold": confidence_threshold,
            "dataset_version": dataset_version,
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        (version_dir / self._METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        logger.info(
            "VisionModelStore: saved %s → %s | f1=%.4f | T=%.4f | classes=%s",
            architecture, version_dir,
            (metrics or {}).get("f1", 0.0),
            calibration_temperature,
            class_names,
        )
        return version

    def save_calibration(
        self,
        version: str,
        temperature: float,
        ece_before: Optional[float] = None,
        ece_after: Optional[float] = None,
        nll_before: Optional[float] = None,
        nll_after: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
    ) -> None:
        """
        Write calibration data alongside an existing model version.

        Also updates calibration_temperature in metadata.json.
        """
        version_dir = self._resolve_version_dir(version)

        cal = {
            "temperature": temperature,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "nll_before": nll_before,
            "nll_after": nll_after,
            "confidence_threshold": confidence_threshold,
            "calibrated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        (version_dir / self._CALIBRATION_FILENAME).write_text(
            json.dumps(cal, indent=2), encoding="utf-8"
        )

        # Patch metadata.json with the new temperature
        meta_path = version_dir / self._METADATA_FILENAME
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["calibration_temperature"] = temperature
            if confidence_threshold is not None:
                meta["confidence_threshold"] = confidence_threshold
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        logger.info(
            "VisionModelStore: calibration saved for %s | T=%.4f | ECE %.4f → %.4f",
            version, temperature,
            ece_before or 0.0, ece_after or 0.0,
        )

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_weights(self, version: Optional[str] = None) -> dict:
        """Return the state dict for the given (or latest) version."""
        version_dir = self._resolve_version_dir(version)
        weights_path = version_dir / self._WEIGHTS_FILENAME
        if not weights_path.exists():
            raise FileNotFoundError(f"No weights file at {weights_path}")
        return torch.load(weights_path, map_location="cpu", weights_only=True)

    def load_metadata(self, version: Optional[str] = None) -> dict:
        """Return the metadata dict for the given (or latest) version."""
        version_dir = self._resolve_version_dir(version)
        meta_path = version_dir / self._METADATA_FILENAME
        if not meta_path.exists():
            raise FileNotFoundError(f"No metadata file at {meta_path}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def load_calibration(self, version: Optional[str] = None) -> Optional[dict]:
        """Return calibration data, or None if not calibrated."""
        version_dir = self._resolve_version_dir(version)
        cal_path = version_dir / self._CALIBRATION_FILENAME
        if not cal_path.exists():
            return None
        return json.loads(cal_path.read_text(encoding="utf-8"))

    # ── Version management ────────────────────────────────────────────────────

    def list_versions(self) -> list[str]:
        """Return all saved versions, newest first.

        Only directories that contain a metadata.json sidecar are considered
        valid store versions — this excludes non-standard directories such as
        v6_medical/ that use a different checkpoint layout.
        """
        return sorted(
            (d.name for d in self.base_dir.iterdir()
             if d.is_dir()
             and d.name.startswith("v")
             and (d / self._METADATA_FILENAME).exists()),
            reverse=True,
        )

    def latest_version(self) -> Optional[str]:
        """Return the most recent version string, or None if empty."""
        versions = self.list_versions()
        return versions[0] if versions else None

    def version_dir(self, version: str) -> Path:
        v = version if version.startswith("v") else f"v{version}"
        return self.base_dir / v

    # ── Internal ──────────────────────────────────────────────────────────────

    def _resolve_version_dir(self, version: Optional[str]) -> Path:
        if version is None:
            latest = self.latest_version()
            if latest is None:
                raise FileNotFoundError(f"No versions found in {self.base_dir}")
            version = latest
        v = version if version.startswith("v") else f"v{version}"
        d = self.base_dir / v
        if not d.exists():
            raise FileNotFoundError(f"Version directory not found: {d}")
        return d
