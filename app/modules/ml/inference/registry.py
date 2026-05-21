import logging

from app.modules.ml.config import MLConfig, ml_config
from app.modules.ml.persistence.model_store import ModelStore

logger = logging.getLogger(__name__)

# Checked in order — first match wins
_PRIORITY_ORDER = [
    "weighted_voting_tuned",
    "xgboost_tuned",
    "lightgbm_tuned",
    "random_forest_tuned",
    "voting",
    "xgboost",
    "lightgbm",
    "random_forest",
]


class ModelRegistry:
    """
    Scans model storage and resolves which model to serve.

    Priority prefers tuned ensembles over single base models.
    All discovery is lazy — no models are loaded into memory here.
    """

    def __init__(self, config: MLConfig = ml_config) -> None:
        self.store = ModelStore(config)

    def best_available(self) -> tuple[str, str] | None:
        """
        Return (model_name, version) for the highest-priority available model.
        Returns None if no model exists on disk.
        """
        for model_name in _PRIORITY_ORDER:
            version = self.store.latest_version(model_name)
            if version is not None:
                logger.info(
                    "ModelRegistry: selected '%s' @ %s", model_name, version
                )
                return model_name, version
        logger.warning("ModelRegistry: no models found on disk.")
        return None

    def resolve(self, model_name: str, version: str | None = None) -> tuple[str, str] | None:
        """
        Resolve a specific model name to (model_name, version).
        Falls back to latest version if version is None.
        Returns None if the model does not exist.
        """
        resolved = version or self.store.latest_version(model_name)
        if resolved is None:
            return None
        # Verify the artifact file actually exists
        versions = self.store.list_versions(model_name)
        if resolved not in versions:
            return None
        return model_name, resolved

    def get_metadata(self, model_name: str, version: str) -> dict:
        return self.store.load_metadata(model_name, version)

    def list_available(self) -> list[dict]:
        """
        Return info on every known model type that has at least one saved version.
        """
        results = []
        for model_name in _PRIORITY_ORDER:
            version = self.store.latest_version(model_name)
            if version is not None:
                meta = self.store.load_metadata(model_name, version)
                results.append(
                    {
                        "model_name": model_name,
                        "version": version,
                        "metadata": meta,
                    }
                )
        return results
