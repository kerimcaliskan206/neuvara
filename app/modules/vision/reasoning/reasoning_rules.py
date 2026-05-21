"""
Semantic reasoning rules.

Each rule is a named scorer function (ReasoningEvidence → float [0, 1]) that
represents one hypothesis about the image's semantic content.  The reasoner
evaluates all rules, selects the highest-scoring one (with margin penalty),
and derives a final decision.

Calibration principles
-----------------------
CLIP ViT-B-32/openai over 11 categories produces an inherently FLAT softmax
(entropy ≈ 0.88–0.95 normalised).  Naïve group-mass amplifiers break because:

  • Consumer group (4 sub-labels) accumulates ~0.36 probability mass even for
    medical images where each consumer label gets ~0.09 by chance.
  • Shannon entropy is high for ALL real images, so "uncertain_semantic" beats
    specific rules when scored as entropy × constant.

Correct scoring strategy — three-tier design:
  Tier 1  top_label exactly matches this rule's category  → HIGH base (0.55–0.90)
           amplified by how far above the uniform baseline (1/11 ≈ 0.091) the
           top-label probability is.
  Tier 2  rule's category appears in top-3 labels         → MEDIUM base (0.35–0.45)
  Tier 3  group probability mass only                     → LOW score (< 0.25)
           for multi-label groups (consumer: 4, scene: 2) only fire if mass
           meaningfully exceeds the expected fair share (n_labels / 11).

uncertain_semantic is capped at 0.70 and further suppressed when the top-1
label is clearly above the uniform baseline (top_label_score × 11 > 1.5),
so specific tier-1 rules always beat the uncertain fallback for clear images.

Adding a new CLIP category
--------------------------
1. Add it to LABEL_GROUPS below.
2. It appears in LABEL_TO_GROUP automatically.
3. If a new semantic group is needed, also add a field to LabelGroupScores
   (reasoning_types.py) and a branch in compute_label_group_scores
   (reasoning_utils.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.modules.vision.reasoning.reasoning_types import (
    REASONING_TYPE_DECISIONS,
    ReasoningEvidence,
    SemanticDecision,
)

# ── Label → group mapping ─────────────────────────────────────────────────────

LABEL_GROUPS: dict[str, list[str]] = {
    "medical":   ["medical_xray", "medical_microscopy"],
    "wildlife":  ["wildlife"],
    "rodent":    ["rodent"],
    "human":     ["human"],
    "consumer":  ["food", "furniture", "vehicle", "random_object"],
    "scene":     ["indoor_scene", "outdoor_scene"],
}

LABEL_TO_GROUP: dict[str, str] = {
    label: group
    for group, labels in LABEL_GROUPS.items()
    for label in labels
}

_CONSUMER_LABELS: frozenset[str] = frozenset({"food", "furniture", "vehicle", "random_object"})
_SCENE_LABELS:    frozenset[str] = frozenset({"indoor_scene", "outdoor_scene"})
_MEDICAL_LABELS:  frozenset[str] = frozenset({"medical_xray", "medical_microscopy"})
_NON_MEDICAL_LABELS: frozenset[str] = frozenset(LABEL_TO_GROUP.keys()) - _MEDICAL_LABELS

# Number of CLIP categories — drives baseline probability calculation.
_N_CATS: int = 11
_BASELINE_PROB: float = 1.0 / _N_CATS   # ≈ 0.0909 (uniform baseline per category)


# ── Rule dataclass ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReasoningRule:
    """
    A named, evidence-to-confidence scoring function.

    name        : key in REASONING_TYPE_DECISIONS
    scorer      : ReasoningEvidence → float [0, 1]
    description : one-line human description used in debug logging
    """

    name: str
    scorer: Callable[[ReasoningEvidence], float]
    description: str

    @property
    def decision(self) -> SemanticDecision:
        return REASONING_TYPE_DECISIONS[self.name]


# ── Shared scoring primitives ─────────────────────────────────────────────────


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _certainty_factor(ev: ReasoningEvidence, penalty: float = 0.15) -> float:
    """
    Scale factor in [1-penalty, 1] that shrinks with high semantic uncertainty.

    Lower penalty here than in the old design — uncertainty is high for ALL
    images (CLIP is inherently soft), so we use it only as a gentle modifier,
    not as a primary decision driver.
    """
    return 1.0 - ev.semantic_uncertainty * penalty


def _top_strength(ev: ReasoningEvidence) -> float:
    """
    Normalised excess probability of the top label above the uniform baseline.

    Returns 0.0 when the top label is at or below the baseline (uniform
    distribution), positive when the top label clearly dominates.

    Examples (ViT-B-32/openai, 11 categories):
      wolf image:       top=wildlife(0.19) → (0.19/0.091 - 1) = 1.09
      chest X-ray:      top=medical_xray(0.145) → 0.59
      flat/ambiguous:   top=any(0.091)    → max(0, 0) = 0.0
      strong consumer:  top=food(0.32)    → 2.52
    """
    return max(0.0, ev.top_label_score / _BASELINE_PROB - 1.0)


def _tier1(ev: ReasoningEvidence, label: str, base: float = 0.55, slope: float = 0.28) -> float:
    """
    Tier-1 score when top_label exactly matches a specific label.

    Returns 0 immediately when top_label does not match — callers handle
    Tier-2/3 fallback themselves.
    """
    if ev.top_label != label:
        return 0.0
    return _clamp(base + _top_strength(ev) * slope)


def _tier1_group(
    ev: ReasoningEvidence,
    labels: frozenset[str],
    base: float = 0.50,
    slope: float = 0.25,
) -> float:
    """
    Tier-1 score when top_label is any member of a label set (e.g. consumer).

    Returns 0 when top_label is not in the set.
    """
    if ev.top_label not in labels:
        return 0.0
    return _clamp(base + _top_strength(ev) * slope)


def _top3_has(ev: ReasoningEvidence, labels: frozenset[str]) -> bool:
    return any(l in labels for l in ev.top3_labels)


def _group_excess_score(
    group_prob: float,
    n_group_labels: int,
    multiplier: float = 2.5,
) -> float:
    """
    Tier-3 score for multi-label groups based on excess above fair share.

    Fair share = n_group_labels / _N_CATS.  Only scores meaningfully when
    the group is notably over-represented (more than 8 pp above fair share).
    This prevents multi-label groups (consumer: 4 labels → fair_share=0.364)
    from scoring high when each sub-label merely gets ~0.09 by chance.
    """
    fair_share = n_group_labels / _N_CATS
    excess = max(0.0, group_prob - fair_share - 0.08)
    return _clamp(excess * multiplier)


# ── Scorer functions ──────────────────────────────────────────────────────────
#
# Each scorer is a pure function of ReasoningEvidence → float [0, 1].
#
# Expected score ranges (ViT-B-32/openai typical output):
#   Clear match (tier-1):   0.55–0.85 (before certainty factor)
#   Moderate (tier-2):      0.35–0.50
#   Weak (tier-3 / group):  0.05–0.25
#   uncertain_semantic:     0.30–0.70  (capped, suppressed for clear top labels)


def _score_radiology_candidate(ev: ReasoningEvidence) -> float:
    """
    Top label is medical_xray, amplified by how far above baseline.

    Example: chest X-ray → top=medical_xray(0.145), strength=0.59
    tier1=0.55+0.17=0.72, consistency_bonus=0.49×0.25=0.12, certainty=0.87
    → (0.72+0.12)×0.87 = 0.73  (wins against uncertain_semantic at 0.35)
    """
    t1 = _tier1(ev, "medical_xray")
    if t1 == 0.0:
        if "medical_xray" not in ev.top3_labels:
            return 0.0
        t1 = 0.35   # Tier 2: medical_xray in top-3 but not top-1

    consistency_bonus = ev.semantic_consistency * 0.25
    return _clamp((t1 + consistency_bonus) * _certainty_factor(ev, 0.15))


def _score_microscopy_candidate(ev: ReasoningEvidence) -> float:
    """Top label is medical_microscopy with high consistency."""
    t1 = _tier1(ev, "medical_microscopy")
    if t1 == 0.0:
        if "medical_microscopy" not in ev.top3_labels:
            return 0.0
        t1 = 0.35

    consistency_bonus = ev.semantic_consistency * 0.25
    return _clamp((t1 + consistency_bonus) * _certainty_factor(ev, 0.15))


def _score_likely_medical(ev: ReasoningEvidence) -> float:
    """
    Medical signal present but not top-1.

    Does not fire when radiology/microscopy covers the same signal.
    """
    if ev.top_label in _MEDICAL_LABELS:
        return 0.0
    if _top3_has(ev, _MEDICAL_LABELS):
        base = 0.40
    else:
        base = ev.group_scores.medical * 2.5
    return _clamp(base * _certainty_factor(ev, 0.12))


def _score_ambiguous_medical(ev: ReasoningEvidence) -> float:
    """
    Both medical and non-medical signals are present at comparable strength.
    Fires when uncertainty is high AND some medical signal exists.
    """
    medical = ev.group_scores.medical
    non_medical = ev.group_scores.non_medical
    balance = 1.0 - abs(medical - non_medical)       # 1 when equal, 0 at extremes
    medical_floor = _clamp(ev.medical_relevance * 3.0)
    uncertainty_factor = ev.semantic_uncertainty * 0.45
    return _clamp(balance * uncertainty_factor * medical_floor)


def _score_wildlife_scene(ev: ReasoningEvidence) -> float:
    """
    Wildlife label dominant.

    Tier-1: top_label=wildlife AND score clearly above baseline → strong signal.
    Tier-2: wildlife in top-3 → moderate signal.
    Tier-3: group mass only → weak (wildlife has only 1 label, so no multi-label inflation).

    Decision path:
      wolf(top=wildlife, 0.19):   strength=1.09, tier1=0.86, score=0.86×0.78 ≈ 0.67  (wins)
      gorilla-as-human(wildlife in top3): tier2=0.38+0.13=0.51, score≈0.40
      X-ray(wildlife=0.07):               tier3=0.07×2.5×0.82 ≈ 0.14
    """
    if ev.top_label == "wildlife":
        t1 = _tier1(ev, "wildlife")
    elif "wildlife" in ev.top3_labels:
        t1 = 0.38 + ev.group_scores.wildlife * 1.0   # Tier 2
    else:
        t1 = ev.group_scores.wildlife * 2.5          # Tier 3 (safe: 1 label)
    return _clamp(t1 * _certainty_factor(ev, 0.18))


def _score_portrait_scene(ev: ReasoningEvidence) -> float:
    """
    Human portrait dominant.

    Handles the gorilla-classified-as-human edge case:
      gorilla-as-human(top=human, 0.15): strength=0.65, tier1=0.73, score=0.73×0.82=0.60
      → portrait_scene REJECT fires → reasoning override possible if confidence ≥ 0.80
    """
    if ev.top_label == "human":
        t1 = _tier1(ev, "human")
    elif "human" in ev.top3_labels:
        t1 = 0.38 + ev.group_scores.human * 1.0      # Tier 2
    else:
        t1 = ev.group_scores.human * 2.5             # Tier 3 (safe: 1 label)
    return _clamp(t1 * _certainty_factor(ev, 0.18))


def _score_consumer_object(ev: ReasoningEvidence) -> float:
    """
    Food, vehicle, furniture, or miscellaneous object.

    Consumer has 4 sub-labels → natural fair share ≈ 0.364.  We MUST use
    the three-tier design to avoid scoring 1.0 for medical images where each
    consumer sub-label merely gets ~0.09 by chance.

    Tier-1: top_label is any consumer label       → strong (strength-based)
    Tier-2: any consumer label in top-3           → moderate
    Tier-3: group mass far above fair-share only  → weak (excess above 0.44)
    """
    t1 = _tier1_group(ev, _CONSUMER_LABELS)
    if t1 > 0:
        return _clamp(t1 * _certainty_factor(ev, 0.15))

    if _top3_has(ev, _CONSUMER_LABELS):
        base = 0.35 + ev.group_scores.consumer * 0.40
        return _clamp(base * _certainty_factor(ev, 0.15))

    # Tier 3: group mass only — suppress multi-label inflation
    return _clamp(_group_excess_score(ev.group_scores.consumer, 4) * _certainty_factor(ev, 0.10))


def _score_natural_scene(ev: ReasoningEvidence) -> float:
    """
    Indoor or outdoor scene dominant.

    Scene has 2 sub-labels → fair share ≈ 0.182.  Same three-tier approach.
    """
    t1 = _tier1_group(ev, _SCENE_LABELS)
    if t1 > 0:
        return _clamp(t1 * _certainty_factor(ev, 0.15))

    if _top3_has(ev, _SCENE_LABELS):
        base = 0.35 + ev.group_scores.scene * 0.40
        return _clamp(base * _certainty_factor(ev, 0.15))

    return _clamp(_group_excess_score(ev.group_scores.scene, 2) * _certainty_factor(ev, 0.10))


def _score_clear_non_medical(ev: ReasoningEvidence) -> float:
    """
    Catch-all for non-medical images that don't fit a specific scene type.

    Returns 0 immediately for:
    - Medical top labels (medical_xray, medical_microscopy) — prevent false rejects.
    - Any known non-medical top label — specific rules (wildlife, portrait, consumer,
      scene) already cover those cases; don't double-count.

    Only fires for edge cases where the top label is unknown or is a boundary
    category (e.g., rodent) and non-medical group mass is dominant.
    """
    if ev.top_label in _MEDICAL_LABELS:
        return 0.0   # medical image — let radiology/microscopy/likely_medical handle it
    if ev.top_label in _NON_MEDICAL_LABELS:
        return 0.0   # specific rule covers this top label
    # Edge case: boundary category or future unknown category as top label
    return _clamp(ev.group_scores.non_medical * ev.semantic_consistency * 1.5)


def _score_uncertain_semantic(ev: ReasoningEvidence) -> float:
    """
    Fallback: fires when no specific semantic category can be identified.

    Capped at 0.70 so that tier-1 specific rules (which score 0.55–0.90)
    always win for images with a clear top label.

    Further suppressed when top_label_score is clearly above the uniform
    baseline (strength > 0.5, meaning top_label is 1.5× the baseline).
    In this case, the top label IS informative — uncertainty is about the
    margins, not the category winner.

    Example outputs:
      flat/uniform(strength≈0, uncertainty≈0.99):  0.99×0.70 = 0.69  (wins)
      wolf(strength=1.09, uncertainty=0.92):        0.92×0.70×0.55=0.35  (loses to wildlife)
      X-ray(strength=0.59, uncertainty=0.90):       0.90×0.70×0.55=0.35  (loses to radiology)
    """
    base = ev.semantic_uncertainty * 0.70    # hard cap at 0.70
    if _top_strength(ev) > 0.40:
        # Suppress when top label is notably above baseline (1.4× or more).
        # Threshold 0.40 corresponds to top_label_score ≈ 0.128 (vs. baseline 0.091).
        # At that level CLIP's relative ranking is reliable — treat top label as signal.
        base *= 0.55
    return _clamp(base)


# ── Rule registry ─────────────────────────────────────────────────────────────

REASONING_RULES: tuple[ReasoningRule, ...] = (
    ReasoningRule(
        name="radiology_candidate",
        scorer=_score_radiology_candidate,
        description="top-1 is medical_xray, strength-amplified + consistency bonus",
    ),
    ReasoningRule(
        name="microscopy_candidate",
        scorer=_score_microscopy_candidate,
        description="top-1 is medical_microscopy, strength-amplified + consistency bonus",
    ),
    ReasoningRule(
        name="likely_medical",
        scorer=_score_likely_medical,
        description="medical label in top-3 but not top-1",
    ),
    ReasoningRule(
        name="wildlife_scene",
        scorer=_score_wildlife_scene,
        description="wildlife label dominant, three-tier confidence-weighted",
    ),
    ReasoningRule(
        name="portrait_scene",
        scorer=_score_portrait_scene,
        description="human portrait dominant, handles misclassified gorilla/ape",
    ),
    ReasoningRule(
        name="consumer_object",
        scorer=_score_consumer_object,
        description="food/vehicle/furniture/random_object — inflation-safe three-tier",
    ),
    ReasoningRule(
        name="natural_scene",
        scorer=_score_natural_scene,
        description="indoor or outdoor scene — inflation-safe three-tier",
    ),
    ReasoningRule(
        name="ambiguous_medical",
        scorer=_score_ambiguous_medical,
        description="medical and non-medical signals comparably strong with high uncertainty",
    ),
    ReasoningRule(
        name="clear_non_medical",
        scorer=_score_clear_non_medical,
        description="catch-all: clearly non-medical without a specific scene type",
    ),
    ReasoningRule(
        name="uncertain_semantic",
        scorer=_score_uncertain_semantic,
        description="flat distribution fallback — capped at 0.70, suppressed for clear top labels",
    ),
)
