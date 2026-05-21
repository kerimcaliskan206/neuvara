"""
Lightweight clinical stabilizer for hantavirus ML predictions.

Applies small bounded adjustments to the calibrated ML probability in the
direction of evidence-consistent clinical signals. The ML ensemble remains
the primary decision engine; this layer only smooths boundary behaviour.

Hard constraints (enforced, not aspirational):
  - Operates ONLY on post-Platt calibrated probability — never on raw logits
  - Per-feature contribution cap: ±0.020 (probability units)
  - Total adjustment cap: ±0.060 (probability units)
  - All adjustments are scaled by U(p) = 4p(1-p), which approaches zero as
    the ML probability approaches 0 or 1 (high-confidence predictions are
    nearly immune to stabilization)
  - No knowledge of risk thresholds, confidence labels, or output-side logic
  - Only features present in the training dataset are referenced
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Caps ──────────────────────────────────────────────────────────────────────

_PER_FEATURE_CAP: float = 0.020
_TOTAL_CAP: float = 0.060

# ── Signal table (binary features) ───────────────────────────────────────────
#
# Applied as a positive delta when the feature value is truthy (non-zero).
# Weights are chosen to be within the calibration uncertainty band for
# borderline predictions (≈ ±0.05–0.08).  No weight exceeds the per-feature
# cap, so the cap only activates through the U(p) scaling path.
#
# Features are ordered by epidemiological evidence strength for HPS.
# Only features present in the training dataset are listed.

_BINARY_SIGNALS: dict[str, float] = {
    "thrombocytopenia": 0.020,  # HPS-specific clinical marker; low platelets
    "rodent_contact":   0.018,  # Primary transmission pathway
    "fever":            0.015,  # Universal early HPS symptom
    "outdoor_work":     0.012,  # Exposure proxy (field / forest work)
    "myalgia":          0.010,  # Common early HPS symptom
    "headache":         0.008,  # Common early HPS symptom
}

# ── Signal table (continuous features) ───────────────────────────────────────
#
# (training_median, training_scale, max_delta)
#
# delta_raw = max_delta × clamp((value − median) / scale, −1, +1)
#
# Baseline values approximate the training dataset statistics.
# max_delta is the contribution at one full standard deviation from the median.

_CONTINUOUS_SIGNALS: dict[str, tuple[float, float, float]] = {
    "rodent_density": (5.0, 3.0, 0.012),
    "humidity_pct":   (60.0, 20.0, 0.007),
}


# ── Math helpers ──────────────────────────────────────────────────────────────

def _uncertainty_weight(p: float) -> float:
    """
    U(p) = 4p(1-p).

    Equals 1.0 at p = 0.5 (maximum ML uncertainty).
    Approaches 0.0 as p → 0 or p → 1 (high ML confidence → tiny adjustment).

    Selected values:
      p=0.50 → U=1.000   p=0.62 → U=0.942   p=0.80 → U=0.640
      p=0.85 → U=0.510   p=0.90 → U=0.360   p=0.95 → U=0.190
    """
    return 4.0 * p * (1.0 - p)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class FeatureContribution:
    """Contribution of a single feature to the probability adjustment."""
    feature: str
    value: float            # raw input value
    raw_delta: float        # per-feature delta before uncertainty scaling
    effective_delta: float  # after U(p) scaling and per-feature clamping


@dataclass
class StabilizationResult:
    """Complete output of one stabilizer call."""
    raw_probability: float        # ML calibrated probability — before adjustment
    adjusted_probability: float   # after bounded adjustment
    total_delta: float            # net signed adjustment applied
    uncertainty_weight: float     # U(p) used in this call
    contributions: list[FeatureContribution] = field(default_factory=list)
    stabilization_applied: bool = False


# ── Stabilizer ────────────────────────────────────────────────────────────────

class ClinicalStabilizer:
    """
    Post-calibration probability stabilizer.

    Stateless — safe to instantiate once and reuse across all requests.
    Has no fit/train step; all signal weights are fixed constants.

    The stabilizer has no access to threshold values, risk labels, or
    confidence categories — it operates purely in [0, 1] probability space.
    """

    def adjust(
        self,
        probability: float,
        raw_features: dict,
    ) -> StabilizationResult:
        """
        Compute a bounded clinical adjustment to the calibrated probability.

        Parameters
        ----------
        probability:
            Calibrated ML probability (post-Platt scaling). Must be in [0, 1].
        raw_features:
            Original input feature dict, before any preprocessing.
            Missing keys and None values are silently skipped.

        Returns
        -------
        StabilizationResult
            Contains the adjusted probability, net delta, and a full
            per-feature contribution breakdown for explainability.
        """
        u = _uncertainty_weight(probability)
        contributions: list[FeatureContribution] = []

        # ── Binary features ──────────────────────────────────────────────────
        for feat, signal_weight in _BINARY_SIGNALS.items():
            raw_val = raw_features.get(feat)
            if raw_val is None:
                continue
            try:
                val = float(raw_val)
            except (TypeError, ValueError):
                continue
            if val <= 0.0:
                continue

            raw_delta = signal_weight
            effective = _clamp(raw_delta * u, -_PER_FEATURE_CAP, _PER_FEATURE_CAP)
            contributions.append(FeatureContribution(
                feature=feat,
                value=val,
                raw_delta=raw_delta,
                effective_delta=effective,
            ))

        # ── Continuous features ───────────────────────────────────────────────
        for feat, (median, scale, max_delta) in _CONTINUOUS_SIGNALS.items():
            raw_val = raw_features.get(feat)
            if raw_val is None:
                continue
            try:
                val = float(raw_val)
            except (TypeError, ValueError):
                continue

            normalised = _clamp((val - median) / scale, -1.0, 1.0)
            raw_delta = max_delta * normalised
            effective = _clamp(raw_delta * u, -_PER_FEATURE_CAP, _PER_FEATURE_CAP)
            if abs(effective) < 1e-9:
                continue
            contributions.append(FeatureContribution(
                feature=feat,
                value=val,
                raw_delta=round(raw_delta, 6),
                effective_delta=effective,
            ))

        # ── Aggregate and apply total cap ────────────────────────────────────
        total_raw = sum(c.effective_delta for c in contributions)
        total_delta = _clamp(total_raw, -_TOTAL_CAP, _TOTAL_CAP)
        adjusted = _clamp(probability + total_delta, 0.0, 1.0)
        applied = abs(total_delta) > 1e-9

        logger.debug(
            "Stabilizer: p_raw=%.4f → p_adj=%.4f  Δ=%.4f  U(p)=%.4f  active_features=%d",
            probability, adjusted, total_delta, u, len(contributions),
        )

        return StabilizationResult(
            raw_probability=round(probability, 4),
            adjusted_probability=round(adjusted, 4),
            total_delta=round(total_delta, 4),
            uncertainty_weight=round(u, 4),
            contributions=[
                FeatureContribution(
                    feature=c.feature,
                    value=c.value,
                    raw_delta=round(c.raw_delta, 4),
                    effective_delta=round(c.effective_delta, 4),
                )
                for c in contributions
            ],
            stabilization_applied=applied,
        )


# Module-level singleton — stateless, safe to share across all requests.
clinical_stabilizer = ClinicalStabilizer()
