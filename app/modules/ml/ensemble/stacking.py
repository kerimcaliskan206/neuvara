"""
Stacking ensemble (stacked generalization).

Architecture
────────────
Level 0 — base models (RF, XGBoost, LightGBM)
    Trained on K-fold cross-fitting to produce out-of-fold (OOF) probability
    predictions on the training set.  Avoids leaking label information into
    the meta-learner's training data.

Level 1 — meta-learner (LogisticRegression)
    Trained on the OOF probability matrix [n_samples × n_base_models].
    Makes final predictions by combining base-model probabilities on
    unseen data.

Why stacking can beat voting
────────────────────────────
VotingClassifier averages probabilities with fixed (possibly learned) weights.
The meta-learner learns *when* to trust each model — it can give XGBoost
more weight on one region of feature space and RF more weight on another.
"""
import logging
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


@dataclass
class StackingConfig:
    cv_folds: int = 5
    meta_C: float = 1.0
    meta_max_iter: int = 1000
    random_state: int = 42
    use_probabilities: bool = True  # use predict_proba for meta-features


class StackingEnsemble:
    """
    Two-level stacking classifier.

    Parameters
    ----------
    base_models : list of (name, estimator) pairs
        Base classifiers — must support predict_proba.
    config : StackingConfig, optional
    """

    def __init__(
        self,
        base_models: list[tuple[str, object]],
        config: StackingConfig | None = None,
    ) -> None:
        self.base_models = base_models
        self.config = config or StackingConfig()
        self.meta_learner = LogisticRegression(
            C=self.config.meta_C,
            max_iter=self.config.meta_max_iter,
            random_state=self.config.random_state,
        )
        self._fitted = False

    # ── sklearn-compatible API ────────────────────────────────────────────────

    def fit(self, X, y) -> "StackingEnsemble":
        X = np.asarray(X)
        y = np.asarray(y)

        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        n_models = len(self.base_models)
        oof_preds = np.zeros((len(X), n_models))

        logger.info(
            "StackingEnsemble: OOF cross-fitting %d base models (%d folds)",
            n_models, self.config.cv_folds,
        )

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr = y[train_idx]

            for m_idx, (name, model) in enumerate(self.base_models):
                model.fit(X_tr, y_tr)
                if self.config.use_probabilities and hasattr(model, "predict_proba"):
                    oof_preds[val_idx, m_idx] = model.predict_proba(X_val)[:, 1]
                else:
                    oof_preds[val_idx, m_idx] = model.predict(X_val)

            logger.debug(
                "StackingEnsemble: fold %d/%d complete",
                fold_idx + 1, self.config.cv_folds,
            )

        # Refit base models on the full training set
        for name, model in self.base_models:
            model.fit(X, y)
            logger.info("StackingEnsemble: [%s] refitted on full train set", name)

        # Fit meta-learner on OOF predictions
        self.meta_learner.fit(oof_preds, y)
        self._fitted = True
        logger.info("StackingEnsemble: meta-learner fitted")
        return self

    def predict(self, X) -> np.ndarray:
        self._check_fitted()
        return self.meta_learner.predict(self._meta_features(X))

    def predict_proba(self, X) -> np.ndarray:
        self._check_fitted()
        return self.meta_learner.predict_proba(self._meta_features(X))

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # ── Internal ─────────────────────────────────────────────────────────────

    def _meta_features(self, X) -> np.ndarray:
        X = np.asarray(X)
        meta = np.zeros((len(X), len(self.base_models)))
        for i, (name, model) in enumerate(self.base_models):
            if self.config.use_probabilities and hasattr(model, "predict_proba"):
                meta[:, i] = model.predict_proba(X)[:, 1]
            else:
                meta[:, i] = model.predict(X)
        return meta

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "StackingEnsemble is not fitted. Call fit() before predict()."
            )
