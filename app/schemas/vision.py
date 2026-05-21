"""
Pydantic schemas for the vision API.

Responses are intentionally flat and self-describing so they are easy to
consume from any client (CLI, web frontend, third-party integration).
"""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ── Sub-objects ──────────────────────────────────────────────────────────────


class SemanticMatch(BaseModel):
    """One ranked CLIP category match."""

    label: str
    score: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)


class MedicalRefinement(BaseModel):
    """
    Advisory output of the medical semantic refinement layer (Phase 4).

    Evaluates whether an image that passed the semantic gate resembles
    genuine medical imagery vs. fake/generated/generic content.
    ADVISORY ONLY — does not override the EfficientNet classifier.
    """

    semantic_medical_type: str = Field(
        description="Dominant medical sub-group: 'healthy_xray', 'pneumonia_xray', "
                    "'lung_opacity', 'radiology_scan', 'medical_microscopy', "
                    "'fake_medical_texture', 'ai_generated_medical', "
                    "'generic_grayscale', 'non_medical_grayscale'.",
    )
    medical_plausibility: float = Field(
        ge=0.0,
        le=1.0,
        description="Sum of real-medical sub-group probabilities [0, 1]. "
                    "High = image closely resembles genuine medical imaging.",
    )
    fake_medical_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Sum of suspicious sub-group probabilities [0, 1]. "
                    "High = image may be AI-generated, synthetic, or generic grayscale.",
    )
    semantic_margin: float = Field(
        ge=0.0,
        description="Probability gap between top-1 and top-2 sub-groups. "
                    "High = clear identification. Low = ambiguous signals.",
    )
    refinement_reason: str = Field(
        description="Turkish advisory note explaining the refinement outcome.",
    )
    inference_ms: float = Field(description="Refiner inference time in milliseconds.")


class SemanticInfo(BaseModel):
    """
    CLIP semantic analysis + reasoning result included in each prediction response.

    Present whenever the semantic gate ran (even when it passed).
    Absent only when the gate is disabled via SEMANTIC_GATE_ENABLED=false
    or when CLIP failed to load.

    Gate fields  (threshold-based — layers 1 & 2)
    -----------------------------------------------
    label, medical_relevance_score, ood_score, rejection_code,
    rejection_reason, triggered_by, top_matches, inference_ms

    Reasoning fields  (evidence-weighted — layer 3)
    -------------------------------------------------
    reasoning_type, reasoning_confidence, reasoning_decision,
    semantic_uncertainty, semantic_consistency, explanation, group_scores
    """

    # ── Gate fields ───────────────────────────────────────────────────────────
    label: str = Field(description="Top CLIP category predicted for this image.")
    medical_relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Sum of softmax probabilities across medical categories.",
    )
    ood_score: float = Field(
        ge=0.0,
        description="Weighted OOD score (higher = more out-of-distribution).",
    )
    rejection_code: str | None = Field(
        default=None,
        description="Machine-readable rejection code when the gate fired. None if passed.",
    )
    rejection_reason: str | None = Field(
        default=None,
        description="Human-readable rejection reason. None if the gate passed.",
    )
    triggered_by: str | None = Field(
        default=None,
        description=(
            "Which gate signal fired: 'label', 'score', 'reasoning' (reasoning override), "
            "'none' (passed), or 'disabled'."
        ),
    )
    top_matches: list[SemanticMatch] = Field(
        default_factory=list,
        description="Top-K CLIP category matches (label, score, rank).",
    )
    inference_ms: float = Field(description="CLIP inference time in milliseconds.")

    # ── Reasoning fields ──────────────────────────────────────────────────────
    reasoning_type: str | None = Field(
        default=None,
        description=(
            "Dominant semantic scene type from the reasoning engine. "
            "Examples: 'radiology_candidate', 'wildlife_scene', 'ambiguous_medical'."
        ),
    )
    reasoning_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Effective reasoning confidence after competing-hypothesis penalty.",
    )
    reasoning_decision: str | None = Field(
        default=None,
        description="Reasoning engine decision: 'allow', 'reject', or 'uncertain'.",
    )
    semantic_uncertainty: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Shannon entropy of the CLIP distribution, normalised [0, 1]. "
            "High = flat/undecided. Low = peaked/confident."
        ),
    )
    semantic_consistency: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Group coherence of the top-K CLIP matches [0, 1]. "
            "High = all top matches from one group. Low = spread across groups."
        ),
    )
    explanation: str | None = Field(
        default=None,
        description="Human-readable Turkish explanation from the reasoning engine.",
    )
    group_scores: dict[str, float] | None = Field(
        default=None,
        description=(
            "Aggregated softmax probability per semantic group: "
            "'medical', 'wildlife', 'human', 'consumer', 'scene', 'rodent'."
        ),
    )
    medical_refinement: MedicalRefinement | None = Field(
        default=None,
        description=(
            "Medical semantic refinement result (Phase 4). Advisory only — "
            "does not override the EfficientNet classifier. "
            "Present only when the semantic gate passed and the refiner ran."
        ),
    )


class GateInfo(BaseModel):
    """Outcome of the related/unrelated content gate."""

    enabled: bool = Field(description="Whether the gate was applied to this image.")
    predicted_class: str | None = Field(
        default=None, description="Class the gate predicted (None if disabled)."
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Top-class softmax confidence from the gate.",
    )
    threshold: float | None = Field(
        default=None, description="Confidence threshold required to pass the gate."
    )
    relevant_class: str | None = Field(
        default=None, description="Class label the gate treats as on-topic."
    )


class ImageInfo(BaseModel):
    """Metadata about the uploaded image."""

    original_filename: str
    width: int
    height: int
    format: str
    size_bytes: int


class UploadInfo(BaseModel):
    """Information about where the upload was persisted (if at all)."""

    stored: bool = Field(description="Whether the upload was persisted to disk.")
    safe_filename: str | None = None
    storage_path: str | None = None


# ── Explainability + Calibration V2 ──────────────────────────────────────────


class VisionExplainabilityResult(BaseModel):
    """Advisory output of the Calibration V2 / Explainability layer (Phase 6)."""

    model_config = ConfigDict(protected_namespaces=())

    trust_tier: str = Field(
        description=(
            "5-level trust label: 'very_high_trust', 'high_trust', "
            "'moderate_trust', 'uncertain', or 'suspicious'."
        ),
    )
    trust_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Holistic [0, 1] trustworthiness score combining fusion confidence, "
            "uncertainty, alignment, medical plausibility, and fake-medical penalty."
        ),
    )
    calibration_state: str = Field(
        description=(
            "Confidence stability label: 'stable', 'near_threshold', "
            "'softened' (semantic mismatch detected), or 'suspicious'."
        ),
    )
    explanation_summary: str = Field(
        description="Single Turkish sentence summarising why the system produced this result.",
    )
    uncertainty_reason: str | None = Field(
        default=None,
        description=(
            "Turkish description of uncertainty sources. "
            "None when uncertainty is negligible."
        ),
    )
    semantic_warning: str | None = Field(
        default=None,
        description=(
            "Turkish semantic-conflict warning. "
            "None when no actionable conflict is detected."
        ),
    )


# ── Fusion Intelligence ───────────────────────────────────────────────────────


class VisionFusionResult(BaseModel):
    """Advisory output of the Fusion Intelligence Layer (Phase 5)."""

    model_config = ConfigDict(protected_namespaces=())

    fusion_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Classifier confidence after semantic-aware adjustment (bounded ±0.08). "
            "For display and explainability only — acceptance decisions still use "
            "the raw EfficientNet confidence."
        ),
    )
    fusion_delta: float = Field(
        description=(
            "Signed delta applied to classifier confidence (bounded ±0.08). "
            "Positive = boost (aligned signals). Negative = dampening (mismatch or uncertainty)."
        ),
    )
    agreement_score: float = Field(
        ge=0.0,
        le=1.0,
        description="[0, 1] alignment between semantic reasoning and the classifier.",
    )
    uncertainty_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Combined uncertainty signal [0, 1] derived from semantic uncertainty, "
            "OOD tendency, classifier confidence, and fake-medical suspicion."
        ),
    )
    semantic_alignment: str = Field(
        description=(
            "Alignment state: 'aligned' (sources agree on medical interpretation), "
            "'misaligned' (strong disagreement), or 'uncertain' (ambiguous signals)."
        ),
    )
    fusion_reason: str = Field(
        description="Human-readable Turkish explanation of the fusion decision.",
    )


# ── Responses ────────────────────────────────────────────────────────────────


class VisionPredictionResponse(BaseModel):
    """Structured response for a single image prediction."""

    model_config = ConfigDict(protected_namespaces=())

    accepted: bool = Field(
        description=(
            "True when the prediction passed the gate and confidence filter. "
            "False when the image was rejected for any reason.  Stays true "
            "for `acceptance_level == 'accepted_low_confidence'` to keep "
            "boolean callers working."
        )
    )
    acceptance_level: Literal["accepted", "accepted_low_confidence", "rejected"] = (
        Field(
            description=(
                "Tri-state acceptance outcome.  `accepted` = strong confidence; "
                "`accepted_low_confidence` = borderline, frontends should warn; "
                "`rejected` = unrelated / below threshold / sanity-blocked."
            ),
        )
    )
    low_confidence_reason: str | None = Field(
        default=None,
        description=(
            "Human-readable note when acceptance_level == 'accepted_low_confidence'. "
            "None for both fully-accepted and rejected outcomes."
        ),
    )
    predicted_class: str | None = Field(
        default=None,
        description=(
            "Class label predicted by the main classifier. Always present when the "
            "main classifier ran (even for rejected predictions). None only when the "
            "content-relevance gate rejected the image before main inference."
        ),
    )
    predicted_class_index: int | None = Field(
        default=None,
        description=(
            "Integer class index. Always present when the main classifier ran. "
            "None only when the gate rejected before main inference."
        ),
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Softmax confidence of the predicted class.",
    )
    probabilities: dict[str, float] | None = Field(
        default=None,
        description="Per-class softmax probabilities (always present when not rejected).",
    )
    threshold: float = Field(
        description="Confidence threshold used to decide accept vs reject."
    )
    rejection_reason: str | None = Field(
        default=None,
        description=(
            "Human-readable reason the image was rejected. None when accepted."
        ),
    )
    semantic: SemanticInfo | None = Field(
        default=None,
        description=(
            "CLIP semantic analysis result. Present when the semantic gate ran "
            "(regardless of outcome). None only when SEMANTIC_GATE_ENABLED=false "
            "or when CLIP failed to load."
        ),
    )
    fusion: VisionFusionResult | None = Field(
        default=None,
        description=(
            "Advisory fusion intelligence result. Present when the full semantic "
            "pipeline ran (gate + reasoner + refiner) and the image reached the "
            "main classifier. For display/explainability only — does not override "
            "acceptance decisions."
        ),
    )
    explainability: VisionExplainabilityResult | None = Field(
        default=None,
        description=(
            "Calibration V2 explainability result: trust tier, trust score, "
            "calibration state, and Turkish explanation strings. "
            "Present when fusion ran successfully. Advisory only."
        ),
    )
    gate: GateInfo = Field(description="Result of the content-relevance gate.")
    image: ImageInfo = Field(description="Metadata about the uploaded image.")
    upload: UploadInfo = Field(description="Where the upload was persisted, if any.")
    model_name: str = Field(description="Architecture used (e.g. 'efficientnet_b0').")
    model_version: str = Field(description="VisionModelStore version (e.g. 'v20260514_120000').")
    inference_duration_ms: float = Field(description="End-to-end inference time in ms.")
    gradcam_base64: str | None = Field(
        default=None,
        description=(
            "Base64-encoded JPEG of the Grad-CAM overlay for the predicted class. "
            "Present for any predicted class (including hard_negative and unrelated) "
            "when ?gradcam=true is requested. Enables shortcut-learning analysis and "
            "explainability auditing of rejected predictions."
        ),
    )
    timestamp: str = Field(description="ISO-8601 UTC timestamp of the response.")


class VisionModelInfoResponse(BaseModel):
    """Metadata about the currently loaded vision model."""

    model_config = ConfigDict(protected_namespaces=())

    is_ready: bool
    architecture: str | None = None
    model_version: str | None = None
    class_names: list[str] = Field(default_factory=list)
    image_size: list[int] | None = Field(
        default=None, description="[width, height] used during training."
    )
    metrics: dict[str, Any] = Field(default_factory=dict)
    gate_loaded: bool = Field(
        default=False,
        description="Whether the related/unrelated gate is loaded alongside the main model.",
    )