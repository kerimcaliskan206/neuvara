"""
Semantic gate configuration.

Thresholds are calibrated against CLIP ViT-B-32/openai on the HantaProject
dataset (11-category softmax, 52 prompts).  Observable score ranges:

  Category          med_avg   ood_avg   top_label
  ─────────────────────────────────────────────────
  gorilla           0.179     0.651     wildlife / human (mixed)
  wolf              0.177     0.653     wildlife (consistent)
  fox               0.179     0.652     wildlife (consistent)
  chest_xray        0.193     0.641     medical_xray (consistent)
  unrelated_xray    0.193     0.641     medical_xray (consistent)
  related_samples   0.190     0.642     medical_xray / rodent

Two complementary rejection signals:

  1. Label gate  — top_semantic_label in REJECT_SEMANTIC_LABELS
                   Reliable: CLIP's relative ranking is robust.
                   Catches: gorilla, wolf, fox, food, vehicles, furniture.

  2. Score gate  — medical_relevance < MEDICAL_RELEVANCE_THRESHOLD
                   OR ood_score > OOD_REJECTION_THRESHOLD
                   Secondary: catches label edge cases (e.g. gorilla face
                   misclassified as 'human' with low medical_relevance).

Environment-variable overrides (all optional):

  SEMANTIC_GATE_ENABLED               "true" | "false"   (default: true)
  SEMANTIC_MEDICAL_RELEVANCE_THRESHOLD  float             (default: 0.184)
  SEMANTIC_OOD_REJECTION_THRESHOLD      float             (default: 0.648)

IMPORTANT: do NOT set the user-example thresholds (0.35, 0.60) without
recalibrating — those values would incorrectly reject X-ray images at
CLIP ViT-B-32/openai output scale.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


# ── Label-based rejection ─────────────────────────────────────────────────────
#
# Categories whose top predicted label triggers immediate rejection.
# Derived from semantic_analyzer._CATEGORIES names.
# 'human' and 'rodent' are intentionally excluded (ambiguous; let classifier
# decide). 'indoor_scene' and 'outdoor_scene' are also excluded (could be
# a lab or field environment).

REJECT_SEMANTIC_LABELS: frozenset[str] = frozenset({
    "wildlife",       # gorillas, wolves, foxes, bears, monkeys
    "food",           # food photographs
    "furniture",      # chairs, tables, sofas
    "vehicle",        # cars, trucks, motorcycles
    "random_object",  # miscellaneous everyday objects
})

# Machine-readable code per top label (for semantic.rejection_reason)
LABEL_TO_REJECTION_CODE: dict[str, str] = {
    "wildlife":       "wildlife_detected",
    "food":           "food_detected",
    "furniture":      "non_medical_scene",
    "vehicle":        "vehicle_detected",
    "random_object":  "random_object_detected",
    # score-gate fallbacks (not triggered via label gate)
    "human":          "human_portrait_detected",
    "indoor_scene":   "non_medical_scene",
    "outdoor_scene":  "non_medical_scene",
    "rodent":         "non_medical_scene",
}


# ── Score-based thresholds ────────────────────────────────────────────────────
#
# Calibrated to ViT-B-32/openai + 11-category softmax (52 prompts).
# Clean separation observed between wildlife (max med=0.181) and
# medical images (min med=0.192).

_DEFAULT_MEDICAL_RELEVANCE_THRESHOLD: float = 0.184
_DEFAULT_OOD_REJECTION_THRESHOLD: float = 0.648


# ── Config dataclass ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SemanticGateConfig:
    """
    Runtime configuration for the semantic gate.

    Instances are immutable after construction.  Build via
    ``SemanticGateConfig.from_env()`` to respect environment-variable overrides.
    """

    enabled: bool = True

    # minimum medical_relevance_score to pass (reject if below)
    medical_relevance_threshold: float = _DEFAULT_MEDICAL_RELEVANCE_THRESHOLD

    # maximum ood_score to pass (reject if above)
    ood_rejection_threshold: float = _DEFAULT_OOD_REJECTION_THRESHOLD

    # labels that trigger immediate rejection regardless of scores
    reject_labels: frozenset[str] = field(
        default_factory=lambda: REJECT_SEMANTIC_LABELS,
    )

    # number of top CLIP matches to include in the API response
    top_k_in_response: int = 3

    @classmethod
    def from_env(cls) -> "SemanticGateConfig":
        """
        Build config, applying environment-variable overrides where set.
        Unrecognised or malformed values fall back to the compiled defaults.
        """
        enabled_raw = os.environ.get("SEMANTIC_GATE_ENABLED", "true").lower()
        enabled = enabled_raw not in {"false", "0", "no", "off"}

        try:
            med_thr = float(
                os.environ.get(
                    "SEMANTIC_MEDICAL_RELEVANCE_THRESHOLD",
                    str(_DEFAULT_MEDICAL_RELEVANCE_THRESHOLD),
                )
            )
        except ValueError:
            med_thr = _DEFAULT_MEDICAL_RELEVANCE_THRESHOLD

        try:
            ood_thr = float(
                os.environ.get(
                    "SEMANTIC_OOD_REJECTION_THRESHOLD",
                    str(_DEFAULT_OOD_REJECTION_THRESHOLD),
                )
            )
        except ValueError:
            ood_thr = _DEFAULT_OOD_REJECTION_THRESHOLD

        return cls(
            enabled=enabled,
            medical_relevance_threshold=med_thr,
            ood_rejection_threshold=ood_thr,
        )


# Module-level singleton — loaded once, respects env vars at import time.
semantic_gate_config = SemanticGateConfig.from_env()
