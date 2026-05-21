"""
Unified Medical Reasoning Engine — Phase 17.

Architecture contract
---------------------
  IMAGING IS ALWAYS THE PRIMARY SIGNAL.
  Clinical data is a bounded modifier only — it cannot flip risk tiers,
  cannot override severe imaging findings, and cannot elevate a normal
  image to critical risk.

Signal hierarchy
----------------
  1. Imaging score     (EfficientNet, calibrated T*=0.4585)  — primary
  2. Semantic guard    (CLIP + semantic gate)                 — OOD veto
  3. Fusion intel      (alignment, uncertainty)              — advisory weight
  4. Clinical modifier (bounded ±MAX_CLINICAL_DELTA=0.15)    — advisory shift
  5. Trust report      (calibration V2)                      — confidence tier

Risk tier engine
----------------
  FINAL_SCORE = clamp(IMAGING_SCORE + CLINICAL_DELTA, 0, 1)
  But:
    OOD detected → FINAL_SCORE capped at 0.15 regardless
    CRITICAL image (≥0.80) + mitigating clinical → cannot drop below 0.65
    LOW image (<0.35) + severe clinical (alarm≥0.60) → lifted into lower
      MODERATE band [0.40, 0.55] (clinical lift, Phase 25). Suppressed when
      the image is high-confidence-healthy with near-zero bilateral burden.

  Tier thresholds:
    [0.00, 0.35)  → LOW
    [0.35, 0.60)  → MODERATE
    [0.60, 0.80)  → HIGH_DIFFERENTIAL_RISK
    [0.80, 1.00]  → CRITICAL_PULMONARY_RISK

System behavior examples
------------------------
  1. Clean image (healthy) + severe clinical symptoms:
       imaging_score=0.10, clinical_alarm=0.72, additive_final≈0.18
       → clinical lift engages → final=0.40 → MODERATE
       (symptoms amplify risk into the lower MODERATE band, but never HIGH)

  2. Severe image (pneumonia, high confidence) + no clinical data:
       imaging_score=0.91, clinical_delta=0.0   → 0.91 → CRITICAL_PULMONARY_RISK

  3. Severe image + contradicting clinical ("routine checkup, no symptoms"):
       imaging_score=0.84, raw_clinical=-0.12 → contradiction detected
       weight_factor=0.20 → applied_delta=-0.024 → 0.816 → still CRITICAL
       contradiction_severity="severe"

  4. Uncertain imaging (near threshold) + mild symptoms:
       imaging_score=0.51, trust_tier="uncertain", clinical_delta=+0.07 → 0.58 → MODERATE
       near_boundary=True

  5. OOD / non-medical image:
       semantic_gate → OOD → ood_guard_applied=True
       imaging_score → capped to 0.10 → LOW
       clinical ignored (OOD images produce no clinical modifier)
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_CLINICAL_DELTA: float = 0.15            # maximum clinical influence (up or down)
HEALTHY_BILATERAL_LOW_CLINICAL_CAP: float = 0.05    # tightened cap: healthy + moderate bilateral
HEALTHY_VERY_LOW_CLINICAL_CAP: float = 0.03         # ultra-tight cap: healthy + high confidence + very low bilateral
OOD_SCORE_CAP: float = 0.15            # imaging score hard cap when OOD detected
CONTRADICTION_FLOOR: float = 0.65      # CRITICAL images cannot drop below this

# Phase 26 — Three-factor imaging trust. Replaces the prior 9-step attenuation
# cascade + multibranch_cap + refiner_healthy_veto with a single product of
# three named [0.30, 1.0] trust scalars. When the product falls below
# LOW_TRUST_THRESHOLD, the soft floor LOW_TRUST_IMAGING_CAP applies.
LOW_TRUST_THRESHOLD:  float = 0.20
LOW_TRUST_IMAGING_CAP: float = 0.30
# Symmetric healthy-doubt: when classifier predicts healthy but enough
# weighted dissent fires, pin imaging_score to the LOW/MODERATE boundary.
# Phase 29 calmness: votes are weighted by signal strength —
#   refiner_top=pneumonia → 2  (strong, independent classifier)
#   semantic_misaligned   → 2  (strong, independent reasoner)
#   bilateral_burden≥0.55 → 1  (weaker, spatial-only signal)
# Threshold ≥ 4 means bilateral alone or bilateral + one strong is NOT
# enough — at least two STRONG dissents (refiner + misaligned), or all
# three signals together, are required to override a healthy classification.
HEALTHY_DOUBT_REFINER_WEIGHT:   int = 2
HEALTHY_DOUBT_MISALIGNED_WEIGHT: int = 2
HEALTHY_DOUBT_BILATERAL_WEIGHT:  int = 1
HEALTHY_DOUBT_REQUIRED_VOTES:    int = 4   # weighted threshold
HEALTHY_DOUBT_PINNED_SCORE:      float = 0.40  # LOW/MODERATE boundary (cautious)

# Phase 25 — Clinical lift. Severe symptoms on an otherwise unremarkable image
# must not be dismissed entirely. When clinical_alarm crosses the severe band
# AND the image is in LOW range AND the system is NOT high-confidence-healthy,
# floor the final score into the lower MODERATE band. Capped at mid-MODERATE
# so symptoms alone can never reach HIGH from a clean image.
CLINICAL_LIFT_THRESHOLD:  float = 0.60   # alarm level required to trigger lift
CLINICAL_LIFT_FLOOR:      float = 0.40   # MODERATE entry (0.35) + small margin
CLINICAL_LIFT_CEILING:    float = 0.55   # mid-MODERATE — lift can never cross into HIGH

# Risk tier boundaries
_TIER_LOW_UPPER:          float = 0.35
_TIER_MODERATE_UPPER:     float = 0.60
_TIER_HIGH_UPPER:         float = 0.80

# Contradiction gap thresholds (image_alarm vs clinical_alarm)
_CONTRADICTION_MILD:      float = 0.20
_CONTRADICTION_MODERATE:  float = 0.35
_CONTRADICTION_SEVERE:    float = 0.55

# Symptom → alarm contribution weights (additive, capped at 1.0)
_SYMPTOM_WEIGHTS: dict[str, float] = {
    "fever":              0.20,
    "cough":              0.15,
    "dyspnea":            0.30,
    "shortness_of_breath": 0.30,
    "chest_pain":         0.20,
    "hemoptysis":         0.25,
    "tachypnea":          0.20,
    "hypoxia":            0.35,
    "fatigue":            0.10,
    "myalgia":            0.10,
    "night_sweats":       0.10,
    "weight_loss":        0.10,
    "wheezing":           0.12,
    "productive_cough":   0.18,
}

# Exposure → additional alarm contribution
_EXPOSURE_WEIGHTS: dict[str, float] = {
    "rodent_contact":     0.20,
    "hospital":           0.15,
    "sick_contact":       0.15,
    "travel":             0.10,
    "healthcare_worker":  0.12,
    "immunocompromised":  0.20,
}

# ── Phase 22 — Structured clinical signal weights ─────────────────────────────
# All weights produce 0.0 for None (unknown = neutral, NOT treated as normal).

_RESPIRATORY_WEIGHTS: dict[str, float] = {
    "normal": 0.0,
    "mild":   0.15,
    "severe": 0.30,
}

_OXYGENATION_WEIGHTS: dict[str, float] = {
    "normal":      0.0,
    "mild_drop":   0.12,
    "severe_drop": 0.28,
}

_FEVER_SEVERITY_WEIGHTS: dict[str, float] = {
    "none":     0.0,
    "mild":     0.05,
    "moderate": 0.10,
    "high":     0.18,
}

_WORSENING_WEIGHTS: dict[str, float] = {
    "none":       0.0,
    "some":       0.10,
    "rapid_48h":  0.25,  # raised: HIGH priority signal
}

_RODENT_EXPOSURE_WEIGHTS: dict[str, float] = {
    "none":             0.0,
    "unsure":           0.02,  # reduced: LOW priority (differential-shaping only)
    "rural_env":        0.04,
    "possible_contact": 0.08,
}

_DURATION_TIER_WEIGHTS: dict[str, float] = {
    "1_2_days":    0.02,
    "3_7_days":    0.05,
    "over_1_week": 0.10,
}

# Anti-stacking cap: structured signals (resp + oxy + fever + worsening) ≤ 0.50
_STRUCTURED_SIGNALS_CAP: float = 0.50

# Age → age_group mapping
def _map_age_to_group(age: Optional[int]) -> Optional[str]:
    if age is None:
        return None
    if age < 18:  return "adolescent"
    if age < 30:  return "young_adult"
    if age < 50:  return "adult"
    if age < 65:  return "older_adult"
    return "elderly"


# ── Enums ─────────────────────────────────────────────────────────────────────


class MedicalRiskTier(str, Enum):
    LOW                   = "LOW"
    MODERATE              = "MODERATE"
    HIGH_DIFFERENTIAL_RISK = "HIGH_DIFFERENTIAL_RISK"
    CRITICAL_PULMONARY_RISK = "CRITICAL_PULMONARY_RISK"


# ── Input dataclasses ─────────────────────────────────────────────────────────


@dataclass
class ClinicalContext:
    """
    Bounded clinical input — all fields are optional.

    None = not provided = neutral contribution (Phase 22 design contract).
    Clinical context NEVER overrides imaging findings — it is a bounded modifier.
    """
    # Existing fields (kept for backward compatibility)
    symptoms:             list[str] = field(default_factory=list)
    exposure_history:     Optional[str] = None   # hospital | sick_contact | travel | healthcare_worker | immunocompromised
    duration_days:        Optional[int] = None
    severity:             Optional[str] = None   # legacy: "mild" | "moderate" | "severe"
    immunocompromised:    bool = False
    age_group:            Optional[str] = None   # adolescent | young_adult | adult | older_adult | elderly
    notes:                Optional[str] = None

    # Phase 22 — structured clinical signals
    age_numeric:              Optional[int] = None
    sex:                      Optional[str] = None   # "male" | "female"
    respiratory_severity:     Optional[str] = None   # "normal" | "mild" | "severe"
    oxygenation_context:      Optional[str] = None   # "normal" | "mild_drop" | "severe_drop"
    fever_severity:           Optional[str] = None   # "none" | "mild" | "moderate" | "high"
    recent_worsening:         Optional[str] = None   # "none" | "some" | "rapid_48h"
    rodent_exposure_level:    Optional[str] = None   # "none" | "unsure" | "rural_env" | "possible_contact"
    symptom_duration_tier:    Optional[str] = None   # "1_2_days" | "3_7_days" | "over_1_week"

    def __post_init__(self) -> None:
        # Derive age_group from age_numeric if not explicitly set
        if self.age_group is None and self.age_numeric is not None:
            self.age_group = _map_age_to_group(self.age_numeric)

    @property
    def is_empty(self) -> bool:
        return (
            not self.symptoms
            and self.exposure_history is None
            and self.duration_days is None
            and self.severity is None
            and not self.immunocompromised
            and self.respiratory_severity is None
            and self.oxygenation_context is None
            and self.fever_severity is None
            and self.recent_worsening is None
            and self.rodent_exposure_level is None
            and self.symptom_duration_tier is None
            and self.age_numeric is None
            and self.sex is None
        )


# ── Internal result types ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ContradictionResult:
    detected:           bool
    severity:           Optional[str]     # None | "mild" | "moderate" | "severe"
    weight_factor:      float             # [0, 1] — 1.0 = no reduction, 0.0 = fully suppressed
    image_alarm:        float             # imaging alarm level used
    clinical_alarm:     float             # clinical alarm level used
    gap:                float             # |image_alarm - clinical_alarm|
    note:               Optional[str]


@dataclass(frozen=True)
class ClinicalModifierResult:
    raw_delta:          float             # clinical delta before contradiction adjustment
    applied_delta:      float             # final delta ∈ [-MAX_CLINICAL_DELTA, +MAX_CLINICAL_DELTA]
    direction:          str               # "upward" | "downward" | "neutral"
    symptom_score:      float             # [0, 1] symptom contribution
    exposure_score:     float             # [0, 1] exposure contribution
    weight_factor:      float             # contradiction weight factor applied
    contradiction:      ContradictionResult


@dataclass(frozen=True)
class UnifiedReasoningResult:
    session_id:         str
    timestamp:          str

    # Scores
    imaging_score:      float             # primary signal (may include bilateral boost)
    clinical_modifier:  float             # bounded delta applied
    final_score:        float             # imaging_score + clinical_modifier (clamped)
    bilateral_burden:   Optional[float]   # spatial bilateral severity [0,1] from GradCAM

    # Risk tier
    risk_tier:          MedicalRiskTier
    near_boundary:      bool
    boundary_proximity: float             # distance to nearest tier boundary

    # Signals
    ood_detected:       bool
    ood_guard_applied:  bool
    trust_tier:         str               # from CalibrationV2
    trust_score:        float
    calibration_state:  str
    semantic_alignment: str               # from FusionIntelligence
    agreement_score:    float
    uncertainty_score:  float

    # Clinical
    clinical_modifier_result: Optional[ClinicalModifierResult]
    clinical_provided:  bool

    # Flags
    requires_immediate_action: bool       # True only for CRITICAL_PULMONARY_RISK
    clinical_override_attempted: bool     # True if clinical would have crossed tier boundary
    pipeline_warnings:  list[str]

    # Explainability
    differential_classes: list[str]
    reasoning_chain:    list[str]
    imaging_findings:   str
    clinical_summary:   Optional[str]
    contradiction_note: Optional[str]
    final_summary:      str

    # Phase 29 — calmness telemetry. All have defaults so existing callers
    # that build UnifiedReasoningResult without these still work.
    escalation_reason_count: int = 0   # # of independent escalation hooks that fired
    weak_signal_count:       int = 0   # # of near-threshold "almost fired" signals
    disagreement_strength:   float = 0.0  # 1 - evidence_trust (pneumonia path only; else 0)

    # Phase 30 — focal pathology calibration telemetry.
    localization_confidence: float = 0.0   # CAM-derived focal pathology confidence [0,1]
    focal_boost_applied:     bool  = False  # whether the focal pathology boost fired


# ── Bounded clinical modifier ─────────────────────────────────────────────────


class BoundedClinicalModifier:
    """
    Translates clinical context into a bounded score modifier.

    Design rules:
      - delta ∈ [-MAX_CLINICAL_DELTA, +MAX_CLINICAL_DELTA]
      - Empty clinical context → delta = 0.0
      - Severe contradiction detected → weight_factor applied to raw delta
      - OOD image → clinical modifier not applied (returns zero delta)
    """

    def compute(
        self,
        ctx: Optional[ClinicalContext],
        image_score: float,
        is_ood: bool,
    ) -> ClinicalModifierResult:
        if is_ood or ctx is None or ctx.is_empty:
            neutral = ContradictionResult(
                detected=False, severity=None, weight_factor=1.0,
                image_alarm=image_score, clinical_alarm=0.0, gap=0.0, note=None,
            )
            return ClinicalModifierResult(
                raw_delta=0.0, applied_delta=0.0, direction="neutral",
                symptom_score=0.0, exposure_score=0.0,
                weight_factor=1.0, contradiction=neutral,
            )

        symptom_score  = self._symptom_score(ctx)
        exposure_score = self._exposure_score(ctx)

        # Severity multiplier: respiratory_severity (Phase 22) takes priority over legacy severity
        if ctx.respiratory_severity is not None:
            severity_mult = {"normal": 0.35, "mild": 0.70, "severe": 1.0}.get(ctx.respiratory_severity, 0.55)
        else:
            severity_mult = {"mild": 0.40, "moderate": 0.70, "severe": 1.0}.get(ctx.severity or "", 0.55)

        # Phase 22 structured signals — None → 0.0 (unknown = neutral)
        resp_score    = _RESPIRATORY_WEIGHTS.get(ctx.respiratory_severity or "",    0.0)
        oxy_score     = _OXYGENATION_WEIGHTS.get(ctx.oxygenation_context or "",    0.0)
        fever_score   = _FEVER_SEVERITY_WEIGHTS.get(ctx.fever_severity or "",      0.0)
        worsen_score  = _WORSENING_WEIGHTS.get(ctx.recent_worsening or "",         0.0)
        rodent_score  = _RODENT_EXPOSURE_WEIGHTS.get(ctx.rodent_exposure_level or "", 0.0)

        # Anti-stacking cap: prevent four strong signals from exploding the alarm
        structured_sum = min(_STRUCTURED_SIGNALS_CAP, resp_score + oxy_score + fever_score + worsen_score)

        # Duration boost: prefer tier-based (Phase 22), fall back to numeric days (legacy)
        duration_boost = self._duration_boost(ctx)

        # Sex modifier: very low weight, contextual only
        sex_modifier = 0.02 if ctx.sex == "male" else 0.0

        # Full clinical alarm [0, 1]
        combined_alarm = min(1.0,
            symptom_score * severity_mult
            + exposure_score * 0.50
            + rodent_score
            + structured_sum
            + sex_modifier
        )
        clinical_alarm = min(1.0, combined_alarm + duration_boost)

        # Contradiction analysis
        contradiction = self._analyze_contradiction(image_score, clinical_alarm, ctx)

        # Raw delta: map [0,1] clinical_alarm → [-0.05, +MAX_CLINICAL_DELTA]
        # clinical_alarm=0   → delta≈-0.05 (no symptoms = slight mitigating)
        # clinical_alarm=0.5 → delta≈+0.05 (moderate symptoms = moderate push)
        # clinical_alarm=1.0 → delta=+MAX_CLINICAL_DELTA (severe = full push)
        raw_delta = (clinical_alarm - 0.25) * (MAX_CLINICAL_DELTA / 0.75)
        raw_delta = max(-MAX_CLINICAL_DELTA, min(MAX_CLINICAL_DELTA, raw_delta))

        # Apply contradiction weight factor
        applied_delta = raw_delta * contradiction.weight_factor
        applied_delta = max(-MAX_CLINICAL_DELTA, min(MAX_CLINICAL_DELTA, applied_delta))

        direction = "neutral" if abs(applied_delta) < 0.01 else (
            "upward" if applied_delta > 0 else "downward"
        )

        return ClinicalModifierResult(
            raw_delta=round(raw_delta, 4),
            applied_delta=round(applied_delta, 4),
            direction=direction,
            symptom_score=round(symptom_score, 4),
            exposure_score=round(exposure_score, 4),
            weight_factor=round(contradiction.weight_factor, 4),
            contradiction=contradiction,
        )

    @staticmethod
    def _symptom_score(ctx: ClinicalContext) -> float:
        score = sum(_SYMPTOM_WEIGHTS.get(s.lower(), 0.05) for s in ctx.symptoms)
        if ctx.immunocompromised:
            score += 0.15
        return min(1.0, score)

    @staticmethod
    def _exposure_score(ctx: ClinicalContext) -> float:
        base = _EXPOSURE_WEIGHTS.get(ctx.exposure_history or "", 0.0)
        if ctx.immunocompromised:
            base = max(base, _EXPOSURE_WEIGHTS["immunocompromised"])
        return min(1.0, base)

    @staticmethod
    def _duration_boost(ctx: ClinicalContext) -> float:
        if ctx.symptom_duration_tier is not None:
            return _DURATION_TIER_WEIGHTS.get(ctx.symptom_duration_tier, 0.0)
        return min(0.25, (ctx.duration_days or 0) / 28.0)

    @staticmethod
    def _analyze_contradiction(
        image_score: float,
        clinical_alarm: float,
        ctx: ClinicalContext,
    ) -> ContradictionResult:
        gap = abs(image_score - clinical_alarm)

        if gap < _CONTRADICTION_MILD:
            return ContradictionResult(
                detected=False, severity=None, weight_factor=1.0,
                image_alarm=round(image_score, 4),
                clinical_alarm=round(clinical_alarm, 4),
                gap=round(gap, 4), note=None,
            )

        if gap < _CONTRADICTION_MODERATE:
            severity, weight_factor = "mild", 0.80
            note = "Görüntü ve klinik bulgular arasında hafif uyumsuzluk tespit edildi; klinik ağırlık azaltıldı."
        elif gap < _CONTRADICTION_SEVERE:
            severity, weight_factor = "moderate", 0.50
            note = "Görüntü ile klinik bağlam arasında belirgin uyumsuzluk — görüntü bulguları baskın kabul edilmektedir."
        else:
            severity, weight_factor = "severe", 0.20
            note = (
                "Görüntü ve klinik bulgular ciddi biçimde çelişmektedir. "
                "Risk düzeyi görüntü bulguları tarafından belirlenmektedir; klinik katkı baskılandı."
            )

        return ContradictionResult(
            detected=True, severity=severity, weight_factor=weight_factor,
            image_alarm=round(image_score, 4),
            clinical_alarm=round(clinical_alarm, 4),
            gap=round(gap, 4), note=note,
        )


# ── Clinical lift helper (Phase 25) ───────────────────────────────────────────


def _apply_clinical_lift(
    final_score: float,
    clinical_alarm: float,
    is_ood: bool,
    high_conf_healthy_suppression: bool,
) -> tuple[float, bool]:
    """
    Lift a LOW final score into the lower MODERATE band when clinical signals
    are severe and the image is not high-confidence-healthy.

    Returns (possibly_lifted_score, lift_applied).

    Pure function — used by the engine after fusion, and directly by tests.
    """
    if is_ood or high_conf_healthy_suppression:
        return final_score, False
    if final_score >= _TIER_LOW_UPPER:
        return final_score, False
    if clinical_alarm < CLINICAL_LIFT_THRESHOLD:
        return final_score, False
    lifted = min(CLINICAL_LIFT_CEILING, max(final_score, CLINICAL_LIFT_FLOOR))
    if lifted <= final_score:
        return final_score, False
    return round(lifted, 4), True


# ── Imaging-first fusion policy ───────────────────────────────────────────────


class ImagingFirstFusionPolicy:
    """
    Computes the final risk score under imaging-first constraints.

    Guarantees:
      - OOD images → score ≤ OOD_SCORE_CAP = 0.15
      - CRITICAL imaging (≥0.80) + mitigating clinical → score ≥ CONTRADICTION_FLOOR = 0.65
      - Clinical delta is always applied AFTER the OOD guard
    """

    def fuse(
        self,
        imaging_score: float,
        clinical_delta: float,
        is_ood: bool,
        predicted_class: str,
    ) -> tuple[float, bool, bool]:
        """
        Returns (final_score, ood_guard_applied, clinical_override_attempted).
        """
        if is_ood:
            return OOD_SCORE_CAP, True, False

        raw_final = imaging_score + clinical_delta
        clamped   = max(0.0, min(1.0, raw_final))

        # Safety floor: CRITICAL images cannot drop to MODERATE from clinical alone
        if imaging_score >= _TIER_HIGH_UPPER and clamped < CONTRADICTION_FLOOR:
            clamped = CONTRADICTION_FLOOR
            override_attempted = True
        else:
            override_attempted = clinical_delta < 0 and (
                _risk_tier_from_score(raw_final) != _risk_tier_from_score(imaging_score)
            )

        return round(clamped, 4), False, override_attempted


# ── Risk tier engine ──────────────────────────────────────────────────────────


def _risk_tier_from_score(score: float) -> MedicalRiskTier:
    if score < _TIER_LOW_UPPER:
        return MedicalRiskTier.LOW
    if score < _TIER_MODERATE_UPPER:
        return MedicalRiskTier.MODERATE
    if score < _TIER_HIGH_UPPER:
        return MedicalRiskTier.HIGH_DIFFERENTIAL_RISK
    return MedicalRiskTier.CRITICAL_PULMONARY_RISK


def _boundary_proximity(score: float, tier: MedicalRiskTier) -> tuple[bool, float]:
    """Return (near_boundary, proximity) where proximity is distance to nearest edge."""
    boundaries = [0.0, _TIER_LOW_UPPER, _TIER_MODERATE_UPPER, _TIER_HIGH_UPPER, 1.0]
    min_dist = min(abs(score - b) for b in boundaries)
    near = min_dist < 0.05
    return near, round(min_dist, 4)


# ── Phase 26 — Three-factor imaging trust ────────────────────────────────────
# Each helper returns (trust_scalar, reasons). Trust is in [0.30, 1.0].
# The product semantic_trust × refiner_trust × spatial_trust replaces the
# previous 9-step attenuation cascade + multibranch_cap + refiner_healthy_veto.
# Every per-signal multiplier below is copy-pasted verbatim from the old
# cascade — this is a structural rename, not a re-tuning.

_TRUST_FLOOR: float = 0.30


def _semantic_trust(
    *,
    semantic_alignment: str,
    medical_relevance_score: Optional[float],
    medical_plausibility: Optional[float],
    fake_medical_score: Optional[float],
) -> tuple[float, list[str]]:
    """Folds CLIP semantic-alignment, medical relevance, plausibility, and the
    fake-medical detector into a single [0.30, 1.0] trust scalar.
    1.0 = upstream semantic stack fully backs the classifier."""
    trust = 1.0
    reasons: list[str] = []
    if semantic_alignment == "misaligned":
        trust *= 0.40
        reasons.append("semantic_misaligned×0.40")
    elif semantic_alignment == "uncertain":
        # Phase 29 calmness: "uncertain" is genuinely weaker than positive
        # disagreement. Softened 0.55 → 0.65 so the gap from "misaligned"
        # is meaningful and weak semantic noise no longer drives MODERATE.
        trust *= 0.65
        reasons.append("semantic_uncertain×0.65")
    if medical_relevance_score is not None and medical_relevance_score < 0.22:
        trust *= 0.65
        reasons.append(f"low_medical_rel({medical_relevance_score:.2f})×0.65")
    if medical_plausibility is not None and medical_plausibility < 0.55:
        trust *= 0.70
        reasons.append(f"low_plausibility({medical_plausibility:.2f})×0.70")
    if fake_medical_score is not None and fake_medical_score > 0.35:
        trust *= 0.55
        reasons.append(f"fake_suspect({fake_medical_score:.2f})×0.55")
    return max(_TRUST_FLOOR, min(1.0, round(trust, 4))), reasons


def _refiner_trust(
    *,
    refiner_top_type: Optional[str],
    refiner_group_scores: Optional[dict[str, float]],
    semantic_margin: Optional[float],
    predicted_class: str,
) -> tuple[float, list[str]]:
    """Folds CLIP medical-refiner sub-group disagreement, top-type mismatch
    (the old refiner_healthy_veto, kept at ×0.30), and refiner margin tightness
    into one trust scalar."""
    trust = 1.0
    reasons: list[str] = []
    # Old refiner_healthy_veto — preserved as a single multiplicative factor
    # so the floor (×0.30) matches the previous hard cap behavior numerically.
    if predicted_class == "pneumonia_xray" and refiner_top_type == "healthy_xray":
        trust *= 0.30
        reasons.append("refiner_healthy_veto×0.30")
    # Refiner sub-group agreement: 0.0 → ×0.35 (full contradict), 0.5 → ×1.0
    if refiner_group_scores:
        rg_pneu = float(refiner_group_scores.get("pneumonia_xray", 0.0))
        rg_heal = float(refiner_group_scores.get("healthy_xray", 0.0))
        if rg_pneu + rg_heal > 0:
            refiner_agree = rg_pneu / (rg_pneu + rg_heal)
            if refiner_agree < 0.5:
                factor = 0.35 + refiner_agree * 1.30
                trust *= factor
                reasons.append(
                    f"refiner_disagrees(pneu={rg_pneu:.2f} heal={rg_heal:.2f})×{factor:.2f}"
                )
    # Refiner is ambiguous between healthy and pneumonia sub-groups.
    if semantic_margin is not None and semantic_margin < 0.05:
        trust *= 0.75
        reasons.append(f"tight_refiner_margin({semantic_margin:.3f})×0.75")
    return max(_TRUST_FLOOR, min(1.0, round(trust, 4))), reasons


def _spatial_trust(
    *,
    bilateral_burden: Optional[float],
    uncertainty_score: float,
    fusion_delta: float,
) -> tuple[float, list[str]]:
    """Folds physical/spatial evidence (GradCAM bilateral burden — attenuator
    legs only; the boost leg stays additive elsewhere), fusion-layer
    uncertainty, and the advisory fusion_delta into one trust scalar."""
    trust = 1.0
    reasons: list[str] = []
    if uncertainty_score >= 0.55:
        trust *= 0.65
        reasons.append(f"high_uncertainty({uncertainty_score:.2f})×0.65")
    if fusion_delta:
        # Preserve the non-multiplicative shape of the old cascade verbatim.
        factor = max(0.70, 1.0 + fusion_delta * 2.0)
        trust *= factor
        reasons.append(f"fusion_delta({fusion_delta:+.3f})×{factor:.2f}")
    if bilateral_burden is not None:
        if bilateral_burden < 0.25:
            trust *= 0.55
            reasons.append(f"low_bilateral_burden({bilateral_burden:.2f})×0.55")
        elif bilateral_burden < 0.40:
            trust *= 0.75
            reasons.append(f"weak_bilateral_burden({bilateral_burden:.2f})×0.75")
    return max(_TRUST_FLOOR, min(1.0, round(trust, 4))), reasons


def _doubt_healthy_prediction(
    *,
    predicted_class: str,
    is_ood: bool,
    refiner_top_type: Optional[str],
    bilateral_burden: Optional[float],
    semantic_alignment: str,
) -> tuple[bool, list[str], int]:
    """Symmetric scrutiny for healthy_xray predictions (Phase 29 calmness).

    Returns (triggered, vote_reasons, weight_total).

    Triggered when weighted votes >= HEALTHY_DOUBT_REQUIRED_VOTES:
      - CLIP refiner top sub-group = pneumonia_xray   (weight 2, strong)
      - Semantic alignment = misaligned               (weight 2, strong)
      - GradCAM bilateral burden >= 0.55              (weight 1, weak)

    Bilateral burden alone, or bilateral + ONE strong signal, is NOT
    enough — the rule requires either both strong signals together (4)
    or all three signals (5). This prevents healthy lungs from being
    escalated by spatial noise + a single dissent.

    Does NOT modify predicted_class — caller pins imaging_score.
    """
    if is_ood or predicted_class != "healthy_xray":
        return False, [], 0
    votes: list[str] = []
    weight = 0
    if refiner_top_type == "pneumonia_xray":
        votes.append(f"refiner_top=pneumonia_xray(w={HEALTHY_DOUBT_REFINER_WEIGHT})")
        weight += HEALTHY_DOUBT_REFINER_WEIGHT
    if semantic_alignment == "misaligned":
        votes.append(f"semantic_misaligned(w={HEALTHY_DOUBT_MISALIGNED_WEIGHT})")
        weight += HEALTHY_DOUBT_MISALIGNED_WEIGHT
    if bilateral_burden is not None and bilateral_burden >= 0.55:
        votes.append(f"bilateral_burden={bilateral_burden:.2f}≥0.55(w={HEALTHY_DOUBT_BILATERAL_WEIGHT})")
        weight += HEALTHY_DOUBT_BILATERAL_WEIGHT
    return weight >= HEALTHY_DOUBT_REQUIRED_VOTES, votes, weight


# ── Explainability builders ───────────────────────────────────────────────────


_V6_CLASSES = {
    "healthy_xray":    "No significant pulmonary pathology detected",
    "pneumonia_xray":  "Pulmonary consolidation / infiltrate pattern consistent with pneumonia",
    "hard_negative":   "Non-medical or unrelated image content",
    "fake_medical":    "Synthetic or non-genuine medical image",
}

_DIFFERENTIAL_MAP: dict[str, list[str]] = {
    MedicalRiskTier.LOW.value: [],
    MedicalRiskTier.MODERATE.value: [
        "Atypical pneumonia (early)",
        "Viral URI with mild lower respiratory involvement",
        "Bronchitis",
    ],
    MedicalRiskTier.HIGH_DIFFERENTIAL_RISK.value: [
        "Bacterial pneumonia",
        "Viral pneumonia (influenza, RSV, COVID-19)",
        "Pulmonary edema (early)",
        "Pulmonary tuberculosis",
    ],
    MedicalRiskTier.CRITICAL_PULMONARY_RISK.value: [
        "Severe bacterial pneumonia",
        "ARDS",
        "Bilateral viral pneumonia",
        "Hantavirus Pulmonary Syndrome (if exposure history)",
        "Pulmonary edema (cardiogenic or non-cardiogenic)",
    ],
}


def _build_imaging_findings(
    predicted_class: str,
    confidence: float,
    probabilities: dict[str, float],
    healthy_doubt_applied: bool = False,
    risk_tier: Optional["MedicalRiskTier"] = None,
) -> str:
    if healthy_doubt_applied:
        # Phase 26: classifier said healthy but independent evidence dissents.
        # Do not render the false "no significant pathology" sentence.
        conf_pct = round(confidence * 100)
        return (
            "Sınıflandırıcı belirgin patoloji bulgulamadı; ancak bağımsız "
            "kanıtlar (refiner, bilateral aktivasyon veya semantik uyumsuzluk) "
            f"bu yargıyı desteklemediği için bulgular sınırlı şekilde rapor "
            f"edilmektedir (%{conf_pct} güven)."
        )
    # Soft phrasing when pneumonia is predicted but evidence_trust attenuation
    # pushed the final risk score into the LOW tier.  Avoids the alarming
    # "konsolidasyon/infiltrat saptandı" line when the system itself has
    # already discounted that signal to LOW risk.
    if predicted_class == "pneumonia_xray" and risk_tier == MedicalRiskTier.LOW:
        conf_pct = round(confidence * 100)
        return (
            "Görüntüde kesin patolojik anlam taşımayan hafif sinyal değişimleri "
            "mevcut olabilir; belirgin pulmoner konsolidasyon veya infiltrat "
            f"izlenmemektedir (düşük güven: %{conf_pct})."
        )
    _BASE_TR = {
        "healthy_xray":   "Görüntüde belirgin pulmoner patoloji bulgusu izlenmedi",
        "pneumonia_xray": "Görüntüde pulmoner konsolidasyon veya infiltrat paterni saptandı",
        "hard_negative":  "Görüntü tıbbi radyoloji içeriği taşımamaktadır",
        "fake_medical":   "Görüntünün tıbbi niteliği doğrulanamadı",
    }
    base = _BASE_TR.get(predicted_class, predicted_class)
    conf_label = (
        "yüksek güven" if confidence >= 0.80 else
        "orta güven"   if confidence >= 0.60 else
        "düşük güven"
    )
    conf_pct = round(confidence * 100)
    return f"{base} ({conf_label}: %{conf_pct})."


def _build_clinical_summary(ctx: Optional[ClinicalContext]) -> Optional[str]:
    if ctx is None or ctx.is_empty:
        return None
    _SYM_TR = {
        "fever": "ateş", "cough": "öksürük", "dyspnea": "dispne",
        "shortness_of_breath": "nefes darlığı", "chest_pain": "göğüs ağrısı",
        "hemoptysis": "hemoptizi", "tachypnea": "takipne", "hypoxia": "hipoksi",
        "fatigue": "yorgunluk", "myalgia": "miyalji", "night_sweats": "gece terlemesi",
        "weight_loss": "kilo kaybı", "wheezing": "hışıltı", "productive_cough": "balgamlı öksürük",
    }
    _RESP_TR  = {"mild": "hafif nefes güçlüğü", "severe": "ciddi nefes güçlüğü"}
    _SEV_TR   = {"mild": "hafif", "moderate": "orta", "severe": "ağır"}
    _OXY_TR   = {"mild_drop": "azalmış nefes kapasitesi", "severe_drop": "ağır oksijen yetersizliği"}
    _FEV_TR   = {"mild": "hafif ateş", "moderate": "orta ateş", "high": "yüksek ateş"}
    _WOR_TR   = {"some": "semptom kötüleşmesi", "rapid_48h": "son 48 saatte hızlı kötüleşme"}
    _DUR_TR   = {"1_2_days": "1–2 günlük semptom", "3_7_days": "3–7 günlük semptom", "over_1_week": "1 haftayı aşan semptom"}
    _ROD_TR   = {"rural_env": "kırsal/depo ortamı maruziyeti", "possible_contact": "olası kemirgen teması"}
    _EXP_TR   = {
        "rodent_contact": "kemirgen teması", "hospital": "hastane maruziyeti",
        "sick_contact": "hasta ile temas", "travel": "seyahat öyküsü",
        "healthcare_worker": "sağlık çalışanı", "immunocompromised": "immün yetmezlik",
    }
    parts: list[str] = []
    if ctx.symptoms:
        labeled = [_SYM_TR.get(s, s) for s in ctx.symptoms]
        parts.append(f"Semptomlar: {', '.join(labeled)}")
    if ctx.respiratory_severity and ctx.respiratory_severity != "normal":
        parts.append(_RESP_TR.get(ctx.respiratory_severity, ctx.respiratory_severity))
    elif ctx.severity:
        parts.append(f"{_SEV_TR.get(ctx.severity, ctx.severity)} şiddet")
    if ctx.oxygenation_context and ctx.oxygenation_context != "normal":
        parts.append(_OXY_TR.get(ctx.oxygenation_context, ctx.oxygenation_context))
    if ctx.fever_severity and ctx.fever_severity != "none":
        parts.append(_FEV_TR.get(ctx.fever_severity, ctx.fever_severity))
    if ctx.recent_worsening and ctx.recent_worsening != "none":
        parts.append(_WOR_TR.get(ctx.recent_worsening, ctx.recent_worsening))
    if ctx.symptom_duration_tier:
        parts.append(_DUR_TR.get(ctx.symptom_duration_tier, ctx.symptom_duration_tier))
    elif ctx.duration_days:
        parts.append(f"{ctx.duration_days} günlük semptom")
    if ctx.rodent_exposure_level and ctx.rodent_exposure_level not in ("none", "unsure"):
        parts.append(_ROD_TR.get(ctx.rodent_exposure_level, ctx.rodent_exposure_level))
    if ctx.exposure_history:
        parts.append(_EXP_TR.get(ctx.exposure_history, ctx.exposure_history))
    if ctx.immunocompromised:
        parts.append("immün yetmezlik")
    return "; ".join(parts) + "." if parts else None


def _build_reasoning_chain(
    imaging_score: float,
    clinical_modifier: ClinicalModifierResult | None,
    is_ood: bool,
    ood_guard_applied: bool,
    trust_tier: str,
    calibration_state: str,
    final_score: float,
    risk_tier: MedicalRiskTier,
    override_attempted: bool,
    clinical_lift_applied: bool = False,
    trust_summary: Optional[tuple[float, float, float]] = None,
    healthy_doubt_applied: bool = False,
    escalation_reason_count: int = 0,
    weak_signal_count: int = 0,
    disagreement_strength: float = 0.0,
) -> list[str]:
    chain = []
    chain.append(
        f"[1/IMAGING] EfficientNet classifier: score={imaging_score:.4f}, "
        f"calibrated with T*=0.4585 (ECE=0.039)"
    )
    if trust_summary is not None:
        sem, ref, spa = trust_summary
        chain.append(
            f"[1a/TRUST] semantic_trust={sem:.2f}, refiner_trust={ref:.2f}, "
            f"spatial_trust={spa:.2f} → evidence_trust={sem*ref*spa:.2f}"
        )
    if healthy_doubt_applied:
        chain.append(
            "[1c/HEALTHY_DOUBT] Independent evidence contradicts healthy "
            f"classification — imaging_score pinned at {HEALTHY_DOUBT_PINNED_SCORE:.2f} "
            "pending clinician review"
        )
    # Phase 29: surface escalation count + weak-signal count so the demo can
    # show "we considered escalating but the evidence wasn't strong enough."
    if escalation_reason_count or weak_signal_count:
        chain.append(
            f"[1d/CALMNESS] escalations={escalation_reason_count}, "
            f"weak_signals={weak_signal_count}, "
            f"disagreement_strength={disagreement_strength:.2f}"
        )
    if is_ood:
        chain.append("[2/OOD_GUARD] Semantic gate detected OOD — score capped at 0.15, clinical ignored")
    else:
        chain.append(f"[2/OOD_GUARD] Semantic gate: medical image confirmed")

    if clinical_modifier and not is_ood:
        chain.append(
            f"[3/CLINICAL] Clinical modifier: raw_delta={clinical_modifier.raw_delta:+.4f}, "
            f"applied_delta={clinical_modifier.applied_delta:+.4f} "
            f"(contradiction_weight={clinical_modifier.weight_factor:.2f})"
        )
        if clinical_modifier.contradiction.detected:
            chain.append(
                f"[3/CLINICAL] Contradiction detected: severity={clinical_modifier.contradiction.severity}, "
                f"gap={clinical_modifier.contradiction.gap:.4f}"
            )
    else:
        chain.append("[3/CLINICAL] No clinical context provided — zero clinical delta")

    if override_attempted:
        chain.append(
            "[4/SAFETY] Clinical override attempt blocked — "
            "CRITICAL imaging cannot be downgraded below HIGH tier"
        )

    if clinical_lift_applied:
        chain.append(
            "[4/LIFT] Clinical lift engaged — severe symptom burden on an "
            "unremarkable image raised final score into the lower MODERATE band "
            "(capped below HIGH; imaging remains primary)"
        )

    chain.append(
        f"[5/TRUST] Calibration V2: trust_tier={trust_tier}, "
        f"calibration_state={calibration_state}"
    )
    chain.append(
        f"[6/VERDICT] Final score={final_score:.4f} → risk_tier={risk_tier.value}"
    )
    return chain


def _build_final_summary(
    risk_tier: MedicalRiskTier,
    predicted_class: str,
    final_score: float,
    is_ood: bool,
    trust_tier: str,
    contradiction_note: Optional[str],
    bilateral_burden: Optional[float] = None,
    clinical_lift_applied: bool = False,
    healthy_doubt_applied: bool = False,
) -> str:
    if is_ood:
        return (
            "Yüklenen görüntü tıbbi görüntüleme içeriği olarak tanımlanamadı. "
            "Klinik risk değerlendirmesi yapılmadı."
        )
    # Phase 26: healthy-doubt MODERATE — classifier said healthy, but
    # independent evidence (refiner, bilateral, semantic) dissents. Avoid
    # rendering the false "hafif pulmoner değişiklik" line.
    if healthy_doubt_applied and risk_tier == MedicalRiskTier.MODERATE:
        base = (
            "Sınıflandırıcı bulgu saptamadı; ancak bağımsız değerlendirme "
            "(CLIP refiner, bilateral aktivasyon veya semantik gerekçelendirme) "
            "bu yargıyı desteklemiyor. Risk düzeyi orta olarak işaretlendi; "
            "klinisyen değerlendirmesi önerilir."
        )
        if trust_tier in ("uncertain", "suspicious"):
            base += (
                " Görüntü güvenilirliği sınırlı olduğundan klinisyen "
                "değerlendirmesi ayrıca önerilir."
            )
        if contradiction_note:
            base += f" {contradiction_note}"
        return base
    # Clinical-lift MODERATE: image is unremarkable, severity comes from symptoms.
    # Do NOT claim imaging changes that aren't there.
    if clinical_lift_applied and risk_tier == MedicalRiskTier.MODERATE:
        base = (
            "Görüntü bulguları belirgin pulmoner patoloji düşündürmemektedir; "
            "ancak klinik tabloda belirgin semptom yükü mevcuttur. "
            "Bu nedenle risk düzeyi orta olarak değerlendirilmiştir — "
            "ekstrapulmoner veya erken-evre nedenler dışlanmalıdır. "
            "Klinik korelasyon ve yakın takip önerilmektedir."
        )
        if trust_tier in ("uncertain", "suspicious"):
            base += (
                " Görüntü güvenilirliği sınırlı olduğundan klinisyen "
                "değerlendirmesi ayrıca önerilir."
            )
        if contradiction_note:
            base += f" {contradiction_note}"
        return base
    tier_messages: dict[MedicalRiskTier, str] = {
        MedicalRiskTier.LOW: (
            "Görüntü bulguları belirgin pulmoner patoloji düşündürmemektedir. "
            "Pulmoner risk düzeyi düşük olarak değerlendirilmektedir."
        ),
        MedicalRiskTier.MODERATE: (
            "Görüntüde hafif pulmoner değişiklik izlenmektedir. "
            "Klinik korelasyon ve yakın takip önerilmektedir."
        ),
        MedicalRiskTier.HIGH_DIFFERENTIAL_RISK: (
            "Görüntüde belirgin pulmoner anormallik saptandı. "
            "Geniş ayırıcı tanı ve klinisyen değerlendirmesi gerekmektedir."
        ),
        MedicalRiskTier.CRITICAL_PULMONARY_RISK: (
            "Görüntüde ciddi pulmoner tutulum paterni izlenmektedir. "
            "Acil klinik değerlendirme gereklidir. "
            "Bu sistem tanı koymaz; bulgu destekleyici niteliktedir."
        ),
    }
    base = tier_messages[risk_tier]
    # Healthy class + elevated bilateral → cautious addendum
    if (
        predicted_class == "healthy_xray"
        and bilateral_burden is not None
        and bilateral_burden >= 0.55
    ):
        base += (
            " Bununla birlikte görüntüde bilateral pulmoner aktivasyon saptandı; "
            "klinik korelasyon önerilir."
        )
    if trust_tier in ("uncertain", "suspicious"):
        base += (
            " Görüntü kalitesi veya içerik belirsizliği nedeniyle güvenilirlik sınırlıdır; "
            "klinisyen değerlendirmesi önerilir."
        )
    if contradiction_note:
        base += f" {contradiction_note}"
    return base


# ── Main engine ───────────────────────────────────────────────────────────────


_clinical_modifier = BoundedClinicalModifier()
_fusion_policy     = ImagingFirstFusionPolicy()


class UnifiedMedicalReasoningEngine:
    """
    Orchestrates the imaging-first, clinically-assisted reasoning pipeline.

    Accepts structured outputs from all upstream components and returns a
    single UnifiedReasoningResult that drives the dashboard response.

    This engine never modifies upstream predictions — it only synthesises them.
    """

    def analyze(
        self,
        *,
        predicted_class: str,
        calibrated_confidence: float,
        probabilities: dict[str, float],
        is_ood: bool,
        ood_class: Optional[str],
        trust_tier: str,
        trust_score: float,
        calibration_state: str,
        uncertainty_reason: Optional[str],
        semantic_warning: Optional[str],
        semantic_alignment: str,
        agreement_score: float,
        uncertainty_score: float,
        fusion_delta: float,
        clinical_context: Optional[ClinicalContext],
        bilateral_score: Optional[object] = None,  # BilateralSpatialScore | None
        model_version: str = "v6_calibrated",
        # Additional advisory signals used to dampen overconfident pneumonia
        # predictions when upstream evidence does not support them.
        medical_relevance_score: Optional[float] = None,
        medical_plausibility: Optional[float] = None,
        fake_medical_score: Optional[float] = None,
        semantic_margin: Optional[float] = None,
        # Refiner sub-class signals — used to detect when the CLIP medical
        # refiner disagrees with the EfficientNet classifier at the
        # healthy_xray vs pneumonia_xray level.
        refiner_top_type: Optional[str] = None,
        refiner_group_scores: Optional[dict[str, float]] = None,
        source_filename: Optional[str] = None,
        # Phase 30 — CAM-derived focal pathology confidence [0,1].
        # Composite of cam_trust_gain, entropy (focality), and coherence.
        # Enables a small positive imaging boost when all conditions are met.
        localization_confidence: Optional[float] = None,
    ) -> UnifiedReasoningResult:
        session_id = uuid.uuid4().hex[:12]
        timestamp  = datetime.now(timezone.utc).isoformat()
        warnings: list[str] = []

        # ── Imaging score ──────────────────────────────────────────────────────
        # Risk proxy: use pneumonia_xray probability as the imaging risk signal.
        # This correctly maps:
        #   healthy images  → low pneumonia prob → LOW imaging_score → LOW tier
        #   pneumonia images → high pneumonia prob → HIGH imaging_score → HIGH/CRITICAL tier
        # Fallback to calibrated_confidence only when the class key is absent.
        raw_pneumonia_score = round(
            float(probabilities.get("pneumonia_xray", calibrated_confidence)),
            4,
        )
        raw_healthy_score = float(probabilities.get("healthy_xray", 0.0))

        # ── Training-prior correction ──────────────────────────────────────────
        # The training set carried 3184 pneumonia vs 1184 healthy samples
        # (ratio 2.69).  An uninformed posterior on a balanced deployment
        # population should be ~0.5; instead the model defaults to ~0.73
        # pneumonia even before any image evidence is considered.  Divide the
        # raw pneumonia probability by the prior ratio to recover an
        # approximately calibrated posterior under a balanced deployment
        # prior, then re-normalise against the healthy mass that exists.
        _PNEUMONIA_TRAINING_PRIOR: float = 0.6450   # 3184/4937 train healthy+pneumonia
        _BALANCED_DEPLOY_PRIOR:    float = 0.5000
        prior_correction = _BALANCED_DEPLOY_PRIOR / max(_PNEUMONIA_TRAINING_PRIOR, 1e-6)
        prior_corrected_pneumonia = min(1.0, raw_pneumonia_score * prior_correction)

        # ── Bilateral burden (Phase 26: single early read) ────────────────────
        # Read once here so both the spatial-trust factor and the
        # healthy-doubt check below see the same value. The bilateral *boost*
        # leg later in the function reuses the same variable.
        bilateral_burden: Optional[float] = None
        if bilateral_score is not None:
            try:
                bilateral_burden = float(bilateral_score.bilateral_burden)
            except Exception:
                logger.debug("Bilateral score extraction failed — skipping", exc_info=True)

        # ── Phase 26 — Three-factor imaging trust ──────────────────────────────
        # Each factor is in [0.30, 1.0]; their product is the evidence_trust
        # scalar that scales raw_pneumonia_score. Only applies to pneumonia
        # predictions — healthy predictions are scrutinized below by
        # _doubt_healthy_prediction.
        semantic_trust, sem_reasons = _semantic_trust(
            semantic_alignment=semantic_alignment,
            medical_relevance_score=medical_relevance_score,
            medical_plausibility=medical_plausibility,
            fake_medical_score=fake_medical_score,
        )
        refiner_trust, ref_reasons = _refiner_trust(
            refiner_top_type=refiner_top_type,
            refiner_group_scores=refiner_group_scores,
            semantic_margin=semantic_margin,
            predicted_class=predicted_class,
        )
        spatial_trust, spa_reasons = _spatial_trust(
            bilateral_burden=bilateral_burden,
            uncertainty_score=uncertainty_score,
            fusion_delta=fusion_delta,
        )

        if not is_ood and predicted_class == "pneumonia_xray":
            evidence_trust = round(
                semantic_trust * refiner_trust * spatial_trust, 4
            )
        else:
            evidence_trust = 1.0

        imaging_score = round(
            min(prior_corrected_pneumonia, raw_pneumonia_score * evidence_trust),
            4,
        )

        # Soft floor replaces the old multibranch_cap. When evidence collapses,
        # the imaging score cannot exceed LOW_TRUST_IMAGING_CAP.
        low_trust_cap_applied = False
        if (
            not is_ood
            and predicted_class == "pneumonia_xray"
            and evidence_trust <= LOW_TRUST_THRESHOLD
        ):
            imaging_score = round(min(imaging_score, LOW_TRUST_IMAGING_CAP), 4)
            low_trust_cap_applied = True

        if evidence_trust < 1.0 or prior_correction < 1.0 or low_trust_cap_applied:
            trust_reasons = (
                [f"semantic_trust={semantic_trust:.2f}({','.join(sem_reasons) or 'ok'})"]
                + [f"refiner_trust={refiner_trust:.2f}({','.join(ref_reasons) or 'ok'})"]
                + [f"spatial_trust={spatial_trust:.2f}({','.join(spa_reasons) or 'ok'})"]
                + [f"evidence_trust={evidence_trust:.2f}"]
                + [f"prior_corrected={prior_corrected_pneumonia:.3f}"]
            )
            if low_trust_cap_applied:
                trust_reasons.append(
                    f"low_trust_cap→imaging_score≤{LOW_TRUST_IMAGING_CAP}"
                )
            logger.info(
                "UnifiedReasoning: imaging_score=%.4f sem=%.2f ref=%.2f spa=%.2f "
                "evidence_trust=%.2f prior=%.3f raw_pneu=%.3f raw_heal=%.3f file=%s",
                imaging_score, semantic_trust, refiner_trust, spatial_trust,
                evidence_trust, prior_correction,
                raw_pneumonia_score, raw_healthy_score,
                source_filename or "?",
            )
            warnings.append(
                "Imaging score attenuated: " + "; ".join(trust_reasons)
            )

        # ── Focal pathology reinforcement (Phase 30 calibration) ────────────────
        # A small positive imaging adjustment when GradCAM telemetry confirms
        # that the activation is well-localized, focal, and anatomically coherent.
        # All six guards must pass before any boost is applied:
        #   (1) pneumonia_xray class predicted
        #   (2) not OOD
        #   (3) low-trust cap was NOT applied (reliable evidence required)
        #   (4) localization_confidence >= 0.65 (composite CAM quality gate)
        #   (5) imaging_score >= 0.25 (existing signal must be meaningful)
        # Max boost = 0.06; linear in [0.65, 1.00].
        focal_boost: float = 0.0
        focal_boost_applied = False
        if (
            not is_ood
            and predicted_class == "pneumonia_xray"
            and not low_trust_cap_applied
            and localization_confidence is not None
            and localization_confidence >= 0.65
            and imaging_score >= 0.25
        ):
            focal_boost = round(
                min(0.06, (localization_confidence - 0.65) / 0.35 * 0.06), 4
            )
            if focal_boost > 0:
                imaging_score = round(min(1.0, imaging_score + focal_boost), 4)
                focal_boost_applied = True
                logger.info(
                    "UnifiedReasoning: focal pathology boost — "
                    "localization_confidence=%.3f boost=+%.4f → imaging_score=%.4f",
                    localization_confidence, focal_boost, imaging_score,
                )

        # ── Bilateral spatial boost (Phase 21) ─────────────────────────────────
        # Pneumonia + strong bilateral burden → additive severity boost.
        # The attenuator legs of the same signal were folded into
        # _spatial_trust above; only the boost leg is preserved here.
        bilateral_boost: float = 0.0
        if (
            not is_ood
            and bilateral_burden is not None
            and predicted_class == "pneumonia_xray"
            and bilateral_burden >= 0.55
            and imaging_score >= 0.35
        ):
            bilateral_boost = round(min(0.12, (bilateral_burden - 0.55) * 0.27), 4)
            imaging_score = round(min(1.0, imaging_score + bilateral_boost), 4)

        # Bilateral activation on healthy-classified image → soft warning
        # (complements the harder _doubt_healthy_prediction pin below).
        if (
            not is_ood
            and bilateral_burden is not None
            and predicted_class == "healthy_xray"
            and bilateral_burden >= 0.55
        ):
            warnings.append(
                f"Bilateral GradCAM activation on healthy-class prediction "
                f"(burden={bilateral_burden:.2f}) — consider clinical correlation"
            )

        # ── Phase 26 — Symmetric healthy-doubt pin ─────────────────────────────
        # When the classifier predicts healthy but enough independent evidence
        # dissents, pin imaging_score to lower MODERATE. predicted_class is
        # NOT mutated — the engine's "never modifies upstream predictions"
        # contract is preserved.
        healthy_doubt_applied, doubt_votes, doubt_weight = _doubt_healthy_prediction(
            predicted_class=predicted_class,
            is_ood=is_ood,
            refiner_top_type=refiner_top_type,
            bilateral_burden=bilateral_burden,
            semantic_alignment=semantic_alignment,
        )
        if healthy_doubt_applied:
            previous_score = imaging_score
            imaging_score = round(max(imaging_score, HEALTHY_DOUBT_PINNED_SCORE), 4)
            logger.info(
                "UnifiedReasoning: healthy-doubt pin applied — votes=%s, "
                "imaging_score %.4f → %.4f",
                doubt_votes, previous_score, imaging_score,
            )
            warnings.append(
                "Healthy prediction contradicted by independent evidence ("
                + ", ".join(doubt_votes)
                + f") — imaging_score pinned at {HEALTHY_DOUBT_PINNED_SCORE:.2f} (lower MODERATE)"
            )

        # ── Clinical modifier ──────────────────────────────────────────────────
        clin_result = _clinical_modifier.compute(clinical_context, imaging_score, is_ood)

        # Healthy + low bilateral burden → tighten clinical influence cap
        # Phase 24: broader activation (burden<0.40, conf>=0.55) + ultra-tight tier (burden<0.15, conf>=0.80)
        effective_clinical_delta = clin_result.applied_delta
        if (
            bilateral_score is not None
            and not is_ood
            and predicted_class == "healthy_xray"
            and bilateral_burden is not None
        ):
            if bilateral_burden < 0.15 and calibrated_confidence >= 0.80:
                # Ultra-tight: high-confidence healthy with near-zero bilateral — clinical almost irrelevant
                effective_clinical_delta = max(
                    -HEALTHY_VERY_LOW_CLINICAL_CAP,
                    min(HEALTHY_VERY_LOW_CLINICAL_CAP, effective_clinical_delta),
                )
                if abs(effective_clinical_delta) < abs(clin_result.applied_delta):
                    warnings.append(
                        "High-confidence healthy image + minimal bilateral: clinical modifier capped at ±0.03"
                    )
            elif bilateral_burden < 0.40 and calibrated_confidence >= 0.55:
                # Standard healthy suppression
                effective_clinical_delta = max(
                    -HEALTHY_BILATERAL_LOW_CLINICAL_CAP,
                    min(HEALTHY_BILATERAL_LOW_CLINICAL_CAP, effective_clinical_delta),
                )
                if abs(effective_clinical_delta) < abs(clin_result.applied_delta):
                    warnings.append(
                        "Healthy imaging + low bilateral burden: clinical modifier capped at ±0.05"
                    )

        # ── Imaging-first fusion ───────────────────────────────────────────────
        final_score, ood_guard_applied, override_attempted = _fusion_policy.fuse(
            imaging_score=imaging_score,
            clinical_delta=effective_clinical_delta,
            is_ood=is_ood,
            predicted_class=predicted_class,
        )

        # ── Clinical lift (Phase 25) ───────────────────────────────────────────
        # Severe clinical burden on a clean image must not be silently dropped.
        # See _apply_clinical_lift docstring for the rule.
        clinical_alarm_for_lift = (
            clin_result.contradiction.clinical_alarm if clin_result is not None else 0.0
        )
        high_conf_healthy_suppression = (
            not is_ood
            and predicted_class == "healthy_xray"
            and bilateral_burden is not None
            and bilateral_burden < 0.15
            and calibrated_confidence >= 0.80
        )
        # Phase 29 calmness: also suppress lift when the imaging evidence
        # was caught by the low-trust cap. Low-trust pneumonia + clinical
        # alarm previously compounded into MODERATE (e.g. 0.30 + 0.15 lift
        # = 0.40); that pattern looked "too risky" given how little we
        # actually trust the image evidence.
        suppress_lift_for_low_trust = (
            not is_ood
            and predicted_class == "pneumonia_xray"
            and low_trust_cap_applied
        )
        pre_lift_score = final_score
        final_score, clinical_lift_applied = _apply_clinical_lift(
            final_score=final_score,
            clinical_alarm=clinical_alarm_for_lift,
            is_ood=is_ood,
            high_conf_healthy_suppression=(
                high_conf_healthy_suppression or suppress_lift_for_low_trust
            ),
        )
        if clinical_lift_applied:
            logger.info(
                "UnifiedReasoning: clinical lift applied — alarm=%.2f, "
                "imaging=%.4f, final %.4f → %.4f (capped at MODERATE)",
                clinical_alarm_for_lift, imaging_score, pre_lift_score, final_score,
            )
            warnings.append(
                f"Clinical lift applied — severe symptom burden "
                f"(alarm={clinical_alarm_for_lift:.2f}) on unremarkable image: "
                f"{pre_lift_score:.2f} → {final_score:.2f} (capped at MODERATE)"
            )

        # ── Risk tier ──────────────────────────────────────────────────────────
        risk_tier     = _risk_tier_from_score(final_score)
        near_boundary, boundary_proximity = _boundary_proximity(final_score, risk_tier)

        # ── CRITICAL tier safeguard ─────────────────────────────────────────────
        # CRITICAL must not be reachable on the strength of a single overconfident
        # classifier alone.  Demote CRITICAL → HIGH when independent corroborating
        # evidence is absent (semantic alignment, medical plausibility, trust).
        if (
            risk_tier == MedicalRiskTier.CRITICAL_PULMONARY_RISK
            and not is_ood
        ):
            critical_supporters = 0
            if semantic_alignment == "aligned":
                critical_supporters += 1
            if (medical_plausibility is not None and medical_plausibility >= 0.60
                    and (fake_medical_score is None or fake_medical_score <= 0.30)):
                critical_supporters += 1
            if trust_tier in ("high_trust", "very_high_trust"):
                critical_supporters += 1
            if (bilateral_burden is not None and bilateral_burden >= 0.55
                    and predicted_class == "pneumonia_xray"):
                critical_supporters += 1

            if critical_supporters < 2:
                # Downgrade to HIGH_DIFFERENTIAL_RISK and cap the final score
                # just below the CRITICAL boundary.
                demoted_score = min(final_score, _TIER_HIGH_UPPER - 0.005)
                logger.info(
                    "UnifiedReasoning: CRITICAL demoted to HIGH "
                    "(supporters=%d/4 align=%s plausibility=%s fake=%s trust=%s "
                    "bilateral=%s) score %.4f → %.4f",
                    critical_supporters, semantic_alignment,
                    medical_plausibility, fake_medical_score, trust_tier,
                    bilateral_burden, final_score, demoted_score,
                )
                warnings.append(
                    f"CRITICAL risk demoted to HIGH — only {critical_supporters}/4 "
                    "independent branches corroborate."
                )
                final_score = round(demoted_score, 4)
                risk_tier   = _risk_tier_from_score(final_score)
                near_boundary, boundary_proximity = _boundary_proximity(final_score, risk_tier)

        # ── Differential classes ───────────────────────────────────────────────
        differential_classes = list(_DIFFERENTIAL_MAP.get(risk_tier.value, []))

        if clinical_context:
            ctx = clinical_context

            # Rodent exposure — HPS differential (binary contact OR new level-based)
            is_rodent_risk = (
                ctx.exposure_history == "rodent_contact"
                or ctx.rodent_exposure_level in ("rural_env", "possible_contact")
            )
            if is_rodent_risk:
                hanta_flag = (
                    "Hantavirus Pulmonary Sendromu ⚑ kemirgen teması"
                    if ctx.exposure_history == "rodent_contact" or ctx.rodent_exposure_level == "possible_contact"
                    else "Hantavirus Pulmoner Sendromu (olası — kırsal ortam)"
                )
                differential_classes = [d for d in differential_classes if "Hantavirus" not in d]
                if risk_tier in (MedicalRiskTier.HIGH_DIFFERENTIAL_RISK, MedicalRiskTier.CRITICAL_PULMONARY_RISK):
                    differential_classes = [hanta_flag] + differential_classes

            # Age-based differentials
            age_group = ctx.age_group
            if age_group in ("older_adult", "elderly") and risk_tier != MedicalRiskTier.LOW:
                if "Aspiration pneumonia" not in " ".join(differential_classes):
                    differential_classes.append("Aspiration pneumonia / Atypical presentation")
            if age_group == "adolescent" and risk_tier != MedicalRiskTier.LOW:
                if "Mycoplasma" not in " ".join(differential_classes):
                    differential_classes.append("Mycoplasma pneumonia (walking pneumonia)")

            # Immunocompromised
            if ctx.immunocompromised and risk_tier not in (MedicalRiskTier.LOW,):
                differential_classes.append("Opportunistic pulmonary infection (PCP / Fungal)")

        differential_classes = differential_classes[:5]

        # ── Warnings ───────────────────────────────────────────────────────────
        if trust_tier in ("uncertain", "suspicious"):
            warnings.append(f"Low model confidence: trust_tier={trust_tier}")
        if calibration_state == "near_threshold":
            warnings.append("Prediction near decision boundary — interpret with caution")
        if near_boundary:
            warnings.append(f"Score {final_score:.3f} is near a risk tier boundary (±0.05)")
        if override_attempted:
            warnings.append(
                "Clinical context attempted to reduce CRITICAL risk — blocked by safety floor"
            )
        if clin_result.contradiction.detected and clin_result.contradiction.severity == "severe":
            warnings.append("Severe imaging/clinical contradiction — clinical modifier suppressed")
        if semantic_warning:
            warnings.append(f"Semantic: {semantic_warning}")

        # ── Explainability ─────────────────────────────────────────────────────
        imaging_findings = _build_imaging_findings(
            predicted_class,
            calibrated_confidence,
            probabilities,
            healthy_doubt_applied=healthy_doubt_applied,
            risk_tier=risk_tier,
        )
        clinical_summary = _build_clinical_summary(clinical_context)
        contradiction_note = clin_result.contradiction.note if clin_result.contradiction.detected else None

        # Phase 26: surface the three trust factors only for pneumonia
        # predictions (healthy predictions skip the cascade).
        trust_summary_for_chain: Optional[tuple[float, float, float]] = (
            (semantic_trust, refiner_trust, spatial_trust)
            if not is_ood and predicted_class == "pneumonia_xray"
            else None
        )

        # ── Phase 29 — Calmness telemetry ──────────────────────────────────────
        # Count escalation hooks that actually fired and weak signals that
        # nearly fired. Gives the demo a way to show "we considered escalating
        # but the evidence wasn't strong enough."
        escalation_reasons: list[str] = []
        if clinical_lift_applied:
            escalation_reasons.append("clinical_lift")
        if healthy_doubt_applied:
            escalation_reasons.append("healthy_doubt")
        if low_trust_cap_applied:
            escalation_reasons.append("low_trust_cap")
        if override_attempted:
            escalation_reasons.append("clinical_override_blocked")
        escalation_reason_count = len(escalation_reasons)

        weak_signals: list[str] = []
        # Near-miss healthy-doubt: at least one dissent vote present but not enough.
        if (
            predicted_class == "healthy_xray"
            and not healthy_doubt_applied
            and 0 < doubt_weight < HEALTHY_DOUBT_REQUIRED_VOTES
        ):
            weak_signals.append(f"healthy_doubt_near_miss(w={doubt_weight})")
        # Soft semantic warning that did NOT trigger a hard policy.
        if not is_ood and semantic_alignment == "uncertain":
            weak_signals.append("semantic_uncertain")
        # Moderate-but-subthreshold clinical alarm.
        if clin_result is not None and not clinical_lift_applied:
            ca = clin_result.contradiction.clinical_alarm
            if 0.40 <= ca < CLINICAL_LIFT_THRESHOLD:
                weak_signals.append(f"clinical_alarm_subthreshold({ca:.2f})")
        # Borderline evidence_trust (pneumonia path).
        if (
            not is_ood
            and predicted_class == "pneumonia_xray"
            and not low_trust_cap_applied
            and LOW_TRUST_THRESHOLD < evidence_trust < 0.50
        ):
            weak_signals.append(f"borderline_evidence_trust({evidence_trust:.2f})")
        weak_signal_count = len(weak_signals)

        # Disagreement strength: 1 - evidence_trust for pneumonia, weighted
        # doubt fraction for healthy, 0 otherwise. Capped at 1.0.
        if not is_ood and predicted_class == "pneumonia_xray":
            disagreement_strength = round(max(0.0, 1.0 - evidence_trust), 4)
        elif not is_ood and predicted_class == "healthy_xray":
            max_doubt = (
                HEALTHY_DOUBT_REFINER_WEIGHT
                + HEALTHY_DOUBT_MISALIGNED_WEIGHT
                + HEALTHY_DOUBT_BILATERAL_WEIGHT
            )
            disagreement_strength = round(min(1.0, doubt_weight / max(1, max_doubt)), 4)
        else:
            disagreement_strength = 0.0

        if escalation_reason_count or weak_signal_count:
            logger.debug(
                "UnifiedReasoning calmness: escalations=%s weak_signals=%s "
                "disagreement=%.2f",
                escalation_reasons, weak_signals, disagreement_strength,
            )

        reasoning_chain = _build_reasoning_chain(
            imaging_score=imaging_score,
            clinical_modifier=clin_result if not is_ood else None,
            is_ood=is_ood,
            ood_guard_applied=ood_guard_applied,
            trust_tier=trust_tier,
            calibration_state=calibration_state,
            final_score=final_score,
            risk_tier=risk_tier,
            override_attempted=override_attempted,
            clinical_lift_applied=clinical_lift_applied,
            trust_summary=trust_summary_for_chain,
            healthy_doubt_applied=healthy_doubt_applied,
            escalation_reason_count=escalation_reason_count,
            weak_signal_count=weak_signal_count,
            disagreement_strength=disagreement_strength,
        )
        if bilateral_boost > 0.0:
            reasoning_chain = list(reasoning_chain)
            reasoning_chain.insert(
                1,
                f"[1b/BILATERAL] Bilateral spatial burden={bilateral_burden:.2f} "
                f"(pneumonia pattern) → imaging_score boosted by +{bilateral_boost:.4f}",
            )
        elif bilateral_burden is not None:
            reasoning_chain = list(reasoning_chain)
            reasoning_chain.insert(
                1,
                f"[1b/BILATERAL] Bilateral spatial burden={bilateral_burden:.2f} "
                f"(no boost applied for class={predicted_class})",
            )
        if focal_boost > 0.0:
            reasoning_chain = list(reasoning_chain)
            reasoning_chain.insert(
                2,
                f"[1e/FOCAL] Focal pathology reinforcement — "
                f"localization_confidence={localization_confidence:.3f} "
                f"→ imaging_score boosted by +{focal_boost:.4f}",
            )

        final_summary = _build_final_summary(
            risk_tier=risk_tier,
            predicted_class=predicted_class,
            final_score=final_score,
            is_ood=is_ood,
            trust_tier=trust_tier,
            contradiction_note=contradiction_note,
            bilateral_burden=bilateral_burden,
            clinical_lift_applied=clinical_lift_applied,
            healthy_doubt_applied=healthy_doubt_applied,
        )

        logger.info(
            "UnifiedReasoning[%s]: class=%s img=%.4f cli_delta=%+.4f "
            "final=%.4f tier=%s trust=%s",
            session_id, predicted_class, imaging_score,
            clin_result.applied_delta, final_score,
            risk_tier.value, trust_tier,
        )

        return UnifiedReasoningResult(
            session_id=session_id,
            timestamp=timestamp,
            imaging_score=imaging_score,
            clinical_modifier=round(effective_clinical_delta, 4),
            final_score=final_score,
            bilateral_burden=bilateral_burden,
            risk_tier=risk_tier,
            near_boundary=near_boundary,
            boundary_proximity=boundary_proximity,
            ood_detected=is_ood,
            ood_guard_applied=ood_guard_applied,
            trust_tier=trust_tier,
            trust_score=round(trust_score, 4),
            calibration_state=calibration_state,
            semantic_alignment=semantic_alignment,
            agreement_score=round(agreement_score, 4),
            uncertainty_score=round(uncertainty_score, 4),
            clinical_modifier_result=clin_result if not clin_result.applied_delta == 0.0 or (clinical_context and not clinical_context.is_empty) else None,
            clinical_provided=bool(clinical_context and not clinical_context.is_empty),
            requires_immediate_action=risk_tier == MedicalRiskTier.CRITICAL_PULMONARY_RISK,
            clinical_override_attempted=override_attempted,
            pipeline_warnings=warnings,
            differential_classes=differential_classes,
            reasoning_chain=reasoning_chain,
            imaging_findings=imaging_findings,
            clinical_summary=clinical_summary,
            contradiction_note=contradiction_note,
            final_summary=final_summary,
            escalation_reason_count=escalation_reason_count,
            weak_signal_count=weak_signal_count,
            disagreement_strength=disagreement_strength,
            localization_confidence=localization_confidence if localization_confidence is not None else 0.0,
            focal_boost_applied=focal_boost_applied,
        )


# Module-level singleton
unified_reasoning_engine = UnifiedMedicalReasoningEngine()
