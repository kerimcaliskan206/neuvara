"""
Phase 30 — End-to-end smoke validation.

Drives 10 scenarios through the real segmentation engine + real reasoning
engine + real GradCAM metric helpers, with synthesized model outputs
representing what the EfficientNet classifier + CLIP semantic gate would
emit for each scenario. Validates the calmness/stability claims accumulated
across Phases 25-29.

Runs without conftest because the repo's conftest pulls in app.main
(missing xgboost in this env). All imports here are module-local.

Each scenario prints a compact report; the test asserts on the
behavioural pattern (LOW vs MODERATE vs HIGH, calm vs localized CAM, etc.)
rather than exact numbers. The full report is also written to stdout so
a human can scan it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest
import torch
from PIL import Image

from app.modules.vision.explainability.anatomical_constraint import (
    compute_border_activation_ratio,
    compute_cam_entropy,
    compute_central_bias_score,
    compute_lung_overlap_score,
)
from app.modules.vision.medical.unified_reasoning import (
    ClinicalContext,
    MedicalRiskTier,
    UnifiedMedicalReasoningEngine,
)
from app.modules.vision.segmentation.lung_segmenter import LungSegmenter


# ─────────────────────────────── reporting ────────────────────────────────


@dataclass
class ScenarioReport:
    name: str
    expected_tier: str
    # Segmentation
    seg_quality: str = ""
    mask_confidence: float = 0.0
    lung_area_ratio: float = 0.0
    fallback_reason: Optional[str] = None
    bilateral_balance: float = 0.0
    contour_count: int = 0
    # Reasoning
    predicted_class: str = ""
    imaging_score: float = 0.0
    final_score: float = 0.0
    risk_tier: str = ""
    near_boundary: bool = False
    # Trust / disagreement
    disagreement_strength: float = 0.0
    escalation_reason_count: int = 0
    weak_signal_count: int = 0
    # CAM metrics (from synthetic CAM that mirrors each scenario)
    cam_lung_overlap: float = 0.0
    cam_border_ratio: float = 0.0
    cam_entropy: float = 0.0
    cam_summary: str = ""
    # Verdict
    passed: bool = False
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> str:
        v = "PASS" if self.passed else "FAIL"
        return (
            f"{self.name:32s} | {v} | tier={self.risk_tier:25s} "
            f"img={self.imaging_score:5.2f} fin={self.final_score:5.2f} "
            f"seg={self.seg_quality:10s} maskC={self.mask_confidence:4.2f} "
            f"esc={self.escalation_reason_count} weak={self.weak_signal_count} "
            f"dis={self.disagreement_strength:4.2f} "
            f"CAM[lung={self.cam_lung_overlap:.2f} bord={self.cam_border_ratio:.2f} "
            f"ent={self.cam_entropy:.2f}]"
        )


# ────────────────────────── image / CAM synthesis ─────────────────────────
# Real CXRs would flow through EfficientNet + CLIP. Here we hand-craft the
# images for segmentation and the simulated model/gate outputs for reasoning.


def _img_clean_healthy(size: int = 384) -> Image.Image:
    img = np.full((size, size), 30, dtype=np.uint8)
    img[60:300,  60:170] = 195
    img[60:300, 214:324] = 195
    img[50:310, 175:209] = 75
    return Image.fromarray(img, mode="L").convert("RGB")


def _img_pneumonia(size: int = 384) -> Image.Image:
    img = np.full((size, size), 30, dtype=np.uint8)
    img[60:300,  60:170] = 195
    img[60:300, 214:324] = 195
    img[50:310, 175:209] = 75
    # Bilateral pneumonic infiltrates
    img[140:230,  80:160] = 230
    img[140:230, 224:304] = 230
    return Image.fromarray(img, mode="L").convert("RGB")


def _img_low_contrast(size: int = 384) -> Image.Image:
    img = np.full((size, size), 130, dtype=np.uint8)
    img[60:300,  60:170] = 112
    img[60:300, 214:324] = 112
    return Image.fromarray(img, mode="L").convert("RGB")


def _img_rotated(size: int = 384) -> Image.Image:
    base = _img_clean_healthy(size)
    return base.rotate(12, fillcolor=(0, 0, 0))


def _img_tiny(size: int = 128) -> Image.Image:
    img = np.full((size, size), 25, dtype=np.uint8)
    img[22:104,  22:56]  = 195
    img[22:104,  72:106] = 195
    return Image.fromarray(img, mode="L").convert("RGB")


def _img_border_padded(size: int = 384, border: int = 70) -> Image.Image:
    inner = _img_clean_healthy(size - 2 * border)
    arr = np.array(inner.convert("L"))
    framed = np.zeros((size, size), dtype=np.uint8)
    framed[border:size - border, border:size - border] = arr
    return Image.fromarray(framed, mode="L").convert("RGB")


def _img_fake_medical(size: int = 384) -> Image.Image:
    """Smooth gradient — looks vaguely medical but has no lung structure."""
    yy, xx = np.mgrid[:size, :size]
    img = ((xx + yy) % 256).astype(np.uint8)
    return Image.fromarray(img, mode="L").convert("RGB")


def _img_random_photo(size: int = 384) -> Image.Image:
    """Random color blocks — meme/photo proxy."""
    rng = np.random.default_rng(7)
    img = rng.integers(40, 220, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(img, mode="RGB")


def _img_blank_noise(size: int = 384) -> Image.Image:
    rng = np.random.default_rng(11)
    img = rng.integers(0, 256, (size, size), dtype=np.uint8)
    return Image.fromarray(img, mode="L").convert("RGB")


# Synthetic CAMs that mirror each scenario's expected CAM behaviour. We
# don't run GradCAM (needs model weights) — we craft the CAM the trust-gain
# would have produced and run the metric helpers on it.


def _cam_calm(H: int, W: int) -> torch.Tensor:
    cam = torch.zeros(H, W)
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    d = ((yy - cy) ** 2 + (xx - cx) ** 2).float()
    cam = 0.10 * torch.exp(-d / (0.4 * H * W))
    return cam


def _cam_localized(H: int, W: int) -> torch.Tensor:
    cam = torch.zeros(H, W)
    # Two focal hotspots inside the lung fields
    for cy, cx in [(int(H * 0.45), int(W * 0.30)), (int(H * 0.45), int(W * 0.70))]:
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        d = ((yy - cy) ** 2 + (xx - cx) ** 2).float()
        cam = torch.maximum(cam, 0.90 * torch.exp(-d / (0.012 * H * W)))
    return cam


def _cam_border_heavy(H: int, W: int) -> torch.Tensor:
    cam = torch.rand(H, W) * 0.15
    bh, bw = int(H * 0.10), int(W * 0.10)
    cam[:bh, :]      += 0.7
    cam[-bh:, :]     += 0.7
    cam[:, :bw]      += 0.7
    cam[:, -bw:]     += 0.7
    return cam.clamp(0.0, 1.0)


def _cam_diffuse_noise(H: int, W: int) -> torch.Tensor:
    return torch.rand(H, W) * 0.6


def _synthetic_lung_mask(H: int, W: int) -> torch.Tensor:
    """A reasonable lung mask matching where _img_clean_healthy puts lungs."""
    m = torch.zeros(H, W)
    m[int(H * 0.15):int(H * 0.78), int(W * 0.15):int(W * 0.45)] = 1.0
    m[int(H * 0.15):int(H * 0.78), int(W * 0.55):int(W * 0.85)] = 1.0
    return m


# ────────────────────────── simulated model output ────────────────────────


@dataclass
class ModelInputs:
    """What the classifier + semantic gate + refiner + bilateral scorer
    would produce for each scenario. Drives the real reasoning engine."""
    predicted_class: str
    calibrated_confidence: float
    probabilities: dict
    is_ood: bool
    semantic_alignment: str  # "aligned" | "uncertain" | "misaligned"
    medical_relevance_score: float
    medical_plausibility: float
    fake_medical_score: float
    semantic_margin: float
    refiner_top_type: str
    refiner_group_scores: dict
    uncertainty_score: float
    fusion_delta: float
    trust_tier: str
    trust_score: float
    bilateral_burden: float


def _run_scenario(
    name: str,
    expected_tier: str,
    image: Image.Image,
    inputs: ModelInputs,
    cam: torch.Tensor,
    cam_summary: str,
    clinical_context: Optional[ClinicalContext] = None,
) -> ScenarioReport:
    rpt = ScenarioReport(name=name, expected_tier=expected_tier)

    # 1. Real segmentation.
    seg = LungSegmenter().segment(image)
    rpt.seg_quality      = seg.quality
    rpt.mask_confidence  = seg.mask_confidence
    rpt.lung_area_ratio  = seg.lung_area_ratio
    rpt.fallback_reason  = seg.fallback_reason
    rpt.bilateral_balance = seg.bilateral_balance
    rpt.contour_count    = seg.contour_count

    # 2. Real reasoning engine with simulated upstream outputs.
    engine = UnifiedMedicalReasoningEngine()
    result = engine.analyze(
        predicted_class=inputs.predicted_class,
        calibrated_confidence=inputs.calibrated_confidence,
        probabilities=inputs.probabilities,
        is_ood=inputs.is_ood,
        ood_class=None if not inputs.is_ood else "non_medical",
        trust_tier=inputs.trust_tier,
        trust_score=inputs.trust_score,
        calibration_state="stable",
        uncertainty_reason=None,
        semantic_warning=None,
        semantic_alignment=inputs.semantic_alignment,
        agreement_score=1.0 - inputs.uncertainty_score,
        uncertainty_score=inputs.uncertainty_score,
        fusion_delta=inputs.fusion_delta,
        clinical_context=clinical_context,
        bilateral_score=SimpleNamespace(bilateral_burden=inputs.bilateral_burden),
        medical_relevance_score=inputs.medical_relevance_score,
        medical_plausibility=inputs.medical_plausibility,
        fake_medical_score=inputs.fake_medical_score,
        semantic_margin=inputs.semantic_margin,
        refiner_top_type=inputs.refiner_top_type,
        refiner_group_scores=inputs.refiner_group_scores,
        source_filename=f"{name}.jpg",
    )
    rpt.predicted_class         = inputs.predicted_class
    rpt.imaging_score           = result.imaging_score
    rpt.final_score             = result.final_score
    rpt.risk_tier               = result.risk_tier.value
    rpt.near_boundary           = result.near_boundary
    rpt.disagreement_strength   = result.disagreement_strength
    rpt.escalation_reason_count = result.escalation_reason_count
    rpt.weak_signal_count       = result.weak_signal_count

    # 3. Real CAM metrics on the synthetic CAM. The CAM was hand-crafted to
    #    mirror what the trust gain would output for the scenario.
    H, W = cam.shape
    lung = _synthetic_lung_mask(H, W)
    rpt.cam_lung_overlap = compute_lung_overlap_score(cam, lung)
    rpt.cam_border_ratio = compute_border_activation_ratio(cam)
    rpt.cam_entropy      = compute_cam_entropy(cam)
    rpt.cam_summary      = cam_summary

    return rpt


# ─────────────────────────────── scenarios ────────────────────────────────


@pytest.fixture(scope="module")
def reports() -> list[ScenarioReport]:
    """Accumulator so we can print a summary at the end of the run."""
    return []


def _print_report_footer(reports: list[ScenarioReport]) -> None:
    print("\n" + "=" * 140)
    print("PHASE 30 — END-TO-END SMOKE VALIDATION REPORT")
    print("=" * 140)
    for r in reports:
        print(r.as_row())
    print("=" * 140)
    passed = sum(r.passed for r in reports)
    print(f"Summary: {passed}/{len(reports)} scenarios behave as expected")
    print("=" * 140 + "\n")


# 1. Clean healthy CXR ─────────────────────────────────────────────────────


def test_smoke_clean_healthy(reports):
    H = W = 224
    rpt = _run_scenario(
        name="1) clean_healthy_cxr",
        expected_tier="LOW",
        image=_img_clean_healthy(),
        inputs=ModelInputs(
            predicted_class="healthy_xray",
            calibrated_confidence=0.92,
            probabilities={"healthy_xray": 0.92, "pneumonia_xray": 0.05},
            is_ood=False,
            semantic_alignment="aligned",
            medical_relevance_score=0.85,
            medical_plausibility=0.85,
            fake_medical_score=0.05,
            semantic_margin=0.40,
            refiner_top_type="healthy_xray",
            refiner_group_scores={"pneumonia_xray": 0.05, "healthy_xray": 0.90},
            uncertainty_score=0.15,
            fusion_delta=0.0,
            trust_tier="high_trust",
            trust_score=0.85,
            bilateral_burden=0.08,
        ),
        cam=_cam_calm(H, W),
        cam_summary="calm, low-magnitude central blob",
    )
    rpt.passed = (
        rpt.risk_tier == "LOW"
        and rpt.escalation_reason_count == 0
        and rpt.cam_entropy > 0.85
    )
    if not rpt.passed:
        rpt.notes.append("expected LOW + calm CAM + zero escalations")
    reports.append(rpt)
    assert rpt.risk_tier == "LOW", rpt.as_row()


# 2. Strong pneumonia ──────────────────────────────────────────────────────


def test_smoke_strong_pneumonia(reports):
    H = W = 224
    rpt = _run_scenario(
        name="2) strong_pneumonia",
        expected_tier="HIGH_DIFFERENTIAL_RISK or CRITICAL",
        image=_img_pneumonia(),
        inputs=ModelInputs(
            predicted_class="pneumonia_xray",
            calibrated_confidence=0.88,
            probabilities={"healthy_xray": 0.08, "pneumonia_xray": 0.88},
            is_ood=False,
            semantic_alignment="aligned",
            medical_relevance_score=0.80,
            medical_plausibility=0.82,
            fake_medical_score=0.05,
            semantic_margin=0.45,
            refiner_top_type="pneumonia_xray",
            refiner_group_scores={"pneumonia_xray": 0.85, "healthy_xray": 0.10},
            uncertainty_score=0.20,
            fusion_delta=0.0,
            trust_tier="high_trust",
            trust_score=0.85,
            bilateral_burden=0.70,
        ),
        cam=_cam_localized(H, W),
        cam_summary="two focal hotspots inside lung fields",
    )
    # A localized multi-hotspot CAM still has moderate Shannon entropy (the
    # Gaussian tails have non-zero mass). The decisive localization signal
    # is "mostly inside lungs, not at borders".
    rpt.passed = (
        rpt.risk_tier in ("HIGH_DIFFERENTIAL_RISK", "CRITICAL_PULMONARY_RISK")
        and rpt.cam_lung_overlap > 0.50
        and rpt.cam_border_ratio < 0.10
    )
    if not rpt.passed:
        rpt.notes.append("expected HIGH or CRITICAL + lung-bound, non-border-dominated CAM")
    reports.append(rpt)
    assert rpt.risk_tier in ("HIGH_DIFFERENTIAL_RISK", "CRITICAL_PULMONARY_RISK"), rpt.as_row()


# 3. Mild / ambiguous abnormality ──────────────────────────────────────────


def test_smoke_mild_ambiguous(reports):
    H = W = 224
    rpt = _run_scenario(
        name="3) mild_ambiguous_finding",
        expected_tier="MODERATE",
        image=_img_pneumonia(),
        inputs=ModelInputs(
            predicted_class="pneumonia_xray",
            calibrated_confidence=0.55,
            probabilities={"healthy_xray": 0.40, "pneumonia_xray": 0.55},
            is_ood=False,
            semantic_alignment="uncertain",
            medical_relevance_score=0.60,
            medical_plausibility=0.65,
            fake_medical_score=0.10,
            semantic_margin=0.15,
            refiner_top_type="pneumonia_xray",
            refiner_group_scores={"pneumonia_xray": 0.55, "healthy_xray": 0.35},
            uncertainty_score=0.50,
            fusion_delta=0.0,
            trust_tier="medium_trust",
            trust_score=0.55,
            bilateral_burden=0.45,
        ),
        cam=_cam_localized(H, W),
        cam_summary="moderate localized activation",
    )
    rpt.passed = rpt.risk_tier in ("MODERATE", "HIGH_DIFFERENTIAL_RISK")
    if not rpt.passed:
        rpt.notes.append("expected MODERATE or upper-LOW for ambiguous finding")
    reports.append(rpt)
    assert rpt.risk_tier != "CRITICAL_PULMONARY_RISK", (
        f"ambiguous case must not reach CRITICAL: {rpt.as_row()}"
    )


# 4. Low contrast X-ray ─────────────────────────────────────────────────────


def test_smoke_low_contrast(reports):
    H = W = 224
    rpt = _run_scenario(
        name="4) low_contrast_xray",
        expected_tier="LOW",
        image=_img_low_contrast(),
        inputs=ModelInputs(
            predicted_class="healthy_xray",
            calibrated_confidence=0.70,
            probabilities={"healthy_xray": 0.70, "pneumonia_xray": 0.25},
            is_ood=False,
            semantic_alignment="uncertain",
            medical_relevance_score=0.55,
            medical_plausibility=0.60,
            fake_medical_score=0.15,
            semantic_margin=0.10,
            refiner_top_type="healthy_xray",
            refiner_group_scores={"pneumonia_xray": 0.20, "healthy_xray": 0.70},
            uncertainty_score=0.55,
            fusion_delta=0.0,
            trust_tier="uncertain",
            trust_score=0.45,
            bilateral_burden=0.20,
        ),
        cam=_cam_calm(H, W),
        cam_summary="low-magnitude diffuse",
    )
    # Adaptive threshold should rescue segmentation, NOT fallback.
    seg_ok = rpt.seg_quality != "fallback"
    rpt.passed = rpt.risk_tier == "LOW" and seg_ok
    if not rpt.passed:
        rpt.notes.append(f"expected LOW + non-fallback segmentation (seg={rpt.seg_quality})")
    reports.append(rpt)
    assert rpt.risk_tier == "LOW", rpt.as_row()


# 5. Rotated image ──────────────────────────────────────────────────────────


def test_smoke_rotated(reports):
    H = W = 224
    rpt = _run_scenario(
        name="5) rotated_12deg_healthy",
        expected_tier="LOW",
        image=_img_rotated(),
        inputs=ModelInputs(
            predicted_class="healthy_xray",
            calibrated_confidence=0.80,
            probabilities={"healthy_xray": 0.80, "pneumonia_xray": 0.15},
            is_ood=False,
            semantic_alignment="aligned",
            medical_relevance_score=0.70,
            medical_plausibility=0.75,
            fake_medical_score=0.08,
            semantic_margin=0.30,
            refiner_top_type="healthy_xray",
            refiner_group_scores={"pneumonia_xray": 0.15, "healthy_xray": 0.75},
            uncertainty_score=0.30,
            fusion_delta=0.0,
            trust_tier="medium_trust",
            trust_score=0.65,
            bilateral_burden=0.12,
        ),
        cam=_cam_calm(H, W),
        cam_summary="calm, rotated context",
    )
    rpt.passed = rpt.risk_tier == "LOW"
    reports.append(rpt)
    assert rpt.risk_tier == "LOW", rpt.as_row()


# 6. Tiny / low-res image ───────────────────────────────────────────────────


def test_smoke_tiny_lowres(reports):
    H = W = 128
    rpt = _run_scenario(
        name="6) tiny_128px_healthy",
        expected_tier="LOW",
        image=_img_tiny(),
        inputs=ModelInputs(
            predicted_class="healthy_xray",
            calibrated_confidence=0.78,
            probabilities={"healthy_xray": 0.78, "pneumonia_xray": 0.18},
            is_ood=False,
            semantic_alignment="aligned",
            medical_relevance_score=0.65,
            medical_plausibility=0.70,
            fake_medical_score=0.10,
            semantic_margin=0.25,
            refiner_top_type="healthy_xray",
            refiner_group_scores={"pneumonia_xray": 0.18, "healthy_xray": 0.72},
            uncertainty_score=0.40,
            fusion_delta=0.0,
            trust_tier="medium_trust",
            trust_score=0.60,
            bilateral_burden=0.18,
        ),
        cam=_cam_calm(H, W),
        cam_summary="small-image calm CAM (smoothing sigma 2.0)",
    )
    rpt.passed = rpt.risk_tier == "LOW"
    reports.append(rpt)
    assert rpt.risk_tier == "LOW", rpt.as_row()


# 7. Border-padded image ────────────────────────────────────────────────────


def test_smoke_border_padded(reports):
    H = W = 224
    rpt = _run_scenario(
        name="7) border_padded_healthy",
        expected_tier="LOW",
        image=_img_border_padded(),
        inputs=ModelInputs(
            predicted_class="healthy_xray",
            calibrated_confidence=0.85,
            probabilities={"healthy_xray": 0.85, "pneumonia_xray": 0.10},
            is_ood=False,
            semantic_alignment="aligned",
            medical_relevance_score=0.75,
            medical_plausibility=0.78,
            fake_medical_score=0.08,
            semantic_margin=0.30,
            refiner_top_type="healthy_xray",
            refiner_group_scores={"pneumonia_xray": 0.10, "healthy_xray": 0.80},
            uncertainty_score=0.25,
            fusion_delta=0.0,
            trust_tier="high_trust",
            trust_score=0.75,
            bilateral_burden=0.10,
        ),
        cam=_cam_calm(H, W),
        cam_summary="calm, border-cropped before segmentation",
    )
    # Border must have been cropped — segmentation should NOT fall back.
    seg_ok = rpt.seg_quality != "fallback"
    rpt.passed = rpt.risk_tier == "LOW" and seg_ok
    if not rpt.passed:
        rpt.notes.append(f"expected LOW + non-fallback seg (seg={rpt.seg_quality})")
    reports.append(rpt)
    assert rpt.risk_tier == "LOW", rpt.as_row()


# 8. Fake / non-medical image ───────────────────────────────────────────────


def test_smoke_fake_nonmedical(reports):
    H = W = 224
    rpt = _run_scenario(
        name="8) fake_nonmedical_gradient",
        expected_tier="LOW",
        image=_img_fake_medical(),
        inputs=ModelInputs(
            # Classifier might say pneumonia with moderate conf on garbage,
            # but every gate signal should reject it.
            predicted_class="pneumonia_xray",
            calibrated_confidence=0.60,
            probabilities={"healthy_xray": 0.30, "pneumonia_xray": 0.60},
            is_ood=False,
            semantic_alignment="misaligned",
            medical_relevance_score=0.10,
            medical_plausibility=0.15,
            fake_medical_score=0.85,
            semantic_margin=0.02,
            refiner_top_type="fake_medical",
            refiner_group_scores={"pneumonia_xray": 0.10, "healthy_xray": 0.05},
            uncertainty_score=0.75,
            fusion_delta=-0.06,
            trust_tier="suspicious",
            trust_score=0.20,
            bilateral_burden=0.20,
        ),
        cam=_cam_border_heavy(H, W),
        cam_summary="border-heavy, low lung overlap",
    )
    rpt.passed = (
        rpt.risk_tier == "LOW"
        and rpt.imaging_score <= 0.30 + 1e-6   # low-trust cap or worse
        and rpt.disagreement_strength > 0.5
        and rpt.cam_lung_overlap < 0.30
    )
    if not rpt.passed:
        rpt.notes.append(
            "fake image expected LOW + low_trust_cap + high disagreement + low lung CAM"
        )
    reports.append(rpt)
    assert rpt.risk_tier == "LOW", rpt.as_row()


# 9. Random meme/photo (OOD path) ───────────────────────────────────────────


def test_smoke_random_photo(reports):
    H = W = 224
    rpt = _run_scenario(
        name="9) random_photo_ood",
        expected_tier="LOW (OOD cap)",
        image=_img_random_photo(),
        inputs=ModelInputs(
            predicted_class="hard_negative",
            calibrated_confidence=0.40,
            probabilities={"healthy_xray": 0.20, "pneumonia_xray": 0.20, "hard_negative": 0.55},
            is_ood=True,
            semantic_alignment="misaligned",
            medical_relevance_score=0.05,
            medical_plausibility=0.10,
            fake_medical_score=0.30,
            semantic_margin=0.01,
            refiner_top_type="hard_negative",
            refiner_group_scores={"pneumonia_xray": 0.10, "healthy_xray": 0.10},
            uncertainty_score=0.80,
            fusion_delta=-0.08,
            trust_tier="suspicious",
            trust_score=0.15,
            bilateral_burden=0.15,
        ),
        cam=_cam_diffuse_noise(H, W),
        cam_summary="diffuse noise, high entropy",
    )
    rpt.passed = (
        rpt.risk_tier == "LOW"
        and rpt.final_score <= 0.16   # OOD cap is 0.15
        and rpt.cam_entropy > 0.90
    )
    if not rpt.passed:
        rpt.notes.append("OOD expected LOW with final_score ≤ 0.15")
    reports.append(rpt)
    assert rpt.risk_tier == "LOW", rpt.as_row()
    assert rpt.final_score <= 0.16, rpt.as_row()


# 10. Blank / noisy image ───────────────────────────────────────────────────


def test_smoke_blank_noise(reports):
    H = W = 224
    rpt = _run_scenario(
        name="10) blank_noise_input",
        expected_tier="LOW",
        image=_img_blank_noise(),
        inputs=ModelInputs(
            predicted_class="healthy_xray",
            calibrated_confidence=0.50,
            probabilities={"healthy_xray": 0.50, "pneumonia_xray": 0.45},
            is_ood=False,
            semantic_alignment="uncertain",
            medical_relevance_score=0.20,
            medical_plausibility=0.25,
            fake_medical_score=0.45,
            semantic_margin=0.03,
            refiner_top_type="healthy_xray",
            refiner_group_scores={"pneumonia_xray": 0.45, "healthy_xray": 0.50},
            uncertainty_score=0.80,
            fusion_delta=-0.04,
            trust_tier="uncertain",
            trust_score=0.30,
            bilateral_burden=0.25,
        ),
        cam=_cam_diffuse_noise(H, W),
        cam_summary="high-entropy diffuse, no localization",
    )
    # Segmentation should fall back (synthetic noise has no lungs).
    seg_fallback = rpt.seg_quality == "fallback"
    rpt.passed = (
        rpt.risk_tier == "LOW"
        and seg_fallback
        and rpt.fallback_reason is not None
    )
    if not rpt.passed:
        rpt.notes.append("blank noise expected LOW + fallback with explicit reason")
    reports.append(rpt)
    assert rpt.risk_tier == "LOW", rpt.as_row()


# ────────────────────────────── final report ──────────────────────────────


def test_zzz_print_summary_table(reports):
    """Tail test (alphabetical: 'zzz') so it runs last and prints the table."""
    _print_report_footer(reports)
    failed = [r for r in reports if not r.passed]
    if failed:
        msg = "\nScenarios that did not match expected pattern:\n" + "\n".join(
            f"  - {r.name}: {'; '.join(r.notes) or 'see row above'}" for r in failed
        )
        pytest.fail(msg)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
