/**
 * Wire types — kept in lock-step with the FastAPI Pydantic schemas under
 * app/schemas/*.  Each type mirrors a backend schema 1:1 so the API layer
 * stays a thin pass-through.
 */

// ── Auth ─────────────────────────────────────────────────────────────────────

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface UserResponse {
  id: number;
  username: string;
  email: string;
  is_active: boolean;
  created_at: string;
}

// ── Patient input (mirrors app/schemas/predict.py PatientInput) ───────────────

export interface PatientInput {
  age?: number | null;
  gender?: string | null;
  region?: string | null;
  season?: string | null;
  rodent_contact?: number | null;   // 0 | 1
  outdoor_work?: number | null;     // 0 | 1
  fever?: number | null;            // 0 | 1
  myalgia?: number | null;          // 0 | 1
  headache?: number | null;         // 0 | 1
  thrombocytopenia?: number | null; // 0 | 1
  rodent_density?: number | null;
  precipitation_mm?: number | null;
  humidity_pct?: number | null;
}

// ── ML prediction ─────────────────────────────────────────────────────────────

export interface MLPredictionResponse {
  prediction: number;
  label: string;
  probability: number | null;
  confidence: string;
  model_name: string;
  model_version: string;
  inference_duration_ms: number;
  timestamp: string;
}

// ── Vision — Explainability + Calibration V2 ─────────────────────────────────

export interface VisionExplainabilityResult {
  trust_tier: string;          // "very_high_trust" | "high_trust" | "moderate_trust" | "uncertain" | "suspicious"
  trust_score: number;         // [0, 1]
  calibration_state: string;   // "stable" | "near_threshold" | "softened" | "suspicious"
  explanation_summary: string; // Turkish explanation sentence
  uncertainty_reason: string | null;
  semantic_warning: string | null;
}

// ── Vision — Fusion Intelligence ─────────────────────────────────────────────

export interface VisionFusionResult {
  fusion_confidence: number;   // adjusted confidence (bounded ±0.08 from classifier)
  fusion_delta: number;        // signed delta applied
  agreement_score: number;     // [0, 1] semantic ↔ classifier alignment
  uncertainty_score: number;   // [0, 1] combined uncertainty signal
  semantic_alignment: string;  // "aligned" | "misaligned" | "uncertain"
  fusion_reason: string;       // Turkish explanation
}

// ── Vision ───────────────────────────────────────────────────────────────────

export interface SemanticMatch {
  label: string;
  score: number;
  rank: number;
}

export interface MedicalRefinement {
  semantic_medical_type: string;
  medical_plausibility: number;
  fake_medical_score: number;
  semantic_margin: number;
  refinement_reason: string;
  inference_ms: number;
}

export interface SemanticInfo {
  // Gate fields (threshold-based)
  label: string;
  medical_relevance_score: number;
  ood_score: number;
  rejection_code: string | null;
  rejection_reason: string | null;
  triggered_by: string | null;
  top_matches: SemanticMatch[];
  inference_ms: number;
  // Reasoning fields (evidence-weighted, layer 3)
  reasoning_type?: string | null;
  reasoning_confidence?: number | null;
  reasoning_decision?: string | null;
  semantic_uncertainty?: number | null;
  semantic_consistency?: number | null;
  explanation?: string | null;
  group_scores?: Record<string, number> | null;
  medical_refinement?: MedicalRefinement | null;
}

export interface GateInfo {
  enabled: boolean;
  predicted_class: string | null;
  confidence: number | null;
}

export interface ImageInfo {
  width: number;
  height: number;
  mode?: string;
  format: string;
  size_bytes: number;
}

export interface UploadInfo {
  stored: boolean;
  safe_filename: string | null;
  storage_path: string | null;
}

export interface VisionPredictionResponse {
  accepted: boolean;
  predicted_class: string | null;
  predicted_class_index: number | null;
  confidence: number | null;
  probabilities: Record<string, number> | null;
  threshold: number | null;
  rejection_reason: string | null;
  semantic?: SemanticInfo | null;
  fusion?: VisionFusionResult | null;
  explainability?: VisionExplainabilityResult | null;
  gate: GateInfo;
  image: ImageInfo | null;
  upload: UploadInfo | null;
  model_name: string;
  model_version: string;
  inference_duration_ms: number;
  gradcam_base64: string | null;
  timestamp: string;
}

export interface VisionModelInfoResponse {
  is_ready: boolean;
  architecture: string;
  model_version: string;
  class_names: string[];
  image_size: [number, number];
  gate_loaded: boolean;
}

// ── Fusion ───────────────────────────────────────────────────────────────────

export interface FusionVisionInput {
  accepted: boolean;
  predicted_class: string | null;
  predicted_class_index: number | null;
  confidence: number | null;
  probabilities: Record<string, number> | null;
  rejection_reason: string | null;
  model_name: string | null;
  model_version: string | null;
  gradcam_base64: string | null;
}

export interface FusionRequest {
  patient: PatientInput;
  vision?: FusionVisionInput | null;
  ml_model_name?: string | null;
}

export interface FusionWeightsUsed {
  ml_weight: number;
  vision_weight: number;
  vision_status: string;
  reason: string;
}

export interface FusionExplanationPayload {
  risk_level: string;
  final_risk_score: number;
  ml_probability: number;
  ml_label: string;
  ml_confidence: string;
  vision_used: boolean;
  vision_class: string | null;
  vision_confidence: number | null;
  vision_status: string;
  vision_rejection_reason: string | null;
  uncertainty_flags: string[];
  dominant_signal: string;
}

export interface FusionResponse {
  final_risk_score: number;
  risk_level: "high" | "medium" | "low";
  fusion_confidence: "high" | "medium" | "low";
  ml_risk_score: number;
  ml_contribution: number;
  vision_contribution: number;
  vision_status: string;
  vision_rejection_reason: string | null;
  uncertainty_flags: string[];
  weights_used: FusionWeightsUsed;
  explanation_payload: FusionExplanationPayload;
  ml_model_name: string | null;
  ml_model_version: string | null;
  vision_model_name: string | null;
  vision_model_version: string | null;
  ml_raw: Record<string, unknown>;
}

// ── AI assistant ──────────────────────────────────────────────────────────────

export interface ChatRequest {
  message: string;
  session_id?: string;
}

export interface ChatResponse {
  content: string;
  intent: string;
  refused: boolean;
  refusal_reason: string | null;
  model: string;
  duration_ms: number;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  timestamp: string;
}

export interface MLInterpretationRequest {
  prediction: number;
  label: string;
  probability?: number | null;
  confidence?: string | null;
  model_name?: string | null;
  model_version?: string | null;
  feature_summary?: string | null;
}

export interface VisionInterpretationGate {
  enabled: boolean;
  predicted_class?: string | null;
  confidence?: number | null;
}

export interface VisionInterpretationRequest {
  accepted: boolean;
  predicted_class?: string | null;
  confidence?: number | null;
  threshold?: number | null;
  rejection_reason?: string | null;
  gate: VisionInterpretationGate;
  model_name?: string | null;
  model_version?: string | null;
}

export interface FusionInterpretationRequest {
  fusion_confidence: string;
  explanation_payload: FusionExplanationPayload;
}

export interface InterpretationResponse {
  content: string;
  model: string;
  duration_ms: number;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  timestamp: string;
}

export interface AIHealthResponse {
  ok: boolean;
  enabled: boolean;
  base_url?: string | null;
  model?: string | null;
  model_loaded: boolean;
  available_models: string[];
  reason?: string | null;
  error?: string | null;
}

// ── Generic error envelope (matches app/core/exception_handlers.py) ──────────

export interface ValidationErrorItem {
  field: string;
  message: string;
}

export interface ApiErrorBody {
  error: string;
  detail?: ValidationErrorItem[] | string | Record<string, unknown> | null;
  status_code?: number;
}

// ── Unified Medical Reasoning — Phase 17/18 ──────────────────────────────────

export type MedicalRiskTier =
  | 'LOW'
  | 'MODERATE'
  | 'HIGH_DIFFERENTIAL_RISK'
  | 'CRITICAL_PULMONARY_RISK';

export interface MedicalImagingSignal {
  predicted_class: string;
  calibrated_confidence: number;
  raw_confidence: number;
  temperature_applied: number;
  class_probabilities: Record<string, number>;
  ood_detected: boolean;
  ood_class: string | null;
  imaging_score: number;
  model_version: string;
  inference_ms: number;
}

export interface MedicalClinicalModifier {
  provided: boolean;
  clinical_delta: number;
  delta_direction: 'upward' | 'downward' | 'neutral';
  symptoms_flagged: string[];
  exposure_flagged: string | null;
  symptom_score: number;
  exposure_score: number;
  contradiction_detected: boolean;
  contradiction_severity: 'mild' | 'moderate' | 'severe' | null;
  contradiction_note: string | null;
  weight_applied: number;
}

export interface MedicalFusionReasoning {
  imaging_weight: number;
  clinical_weight: number;
  semantic_alignment: 'aligned' | 'misaligned' | 'uncertain';
  agreement_score: number;
  uncertainty_score: number;
  fusion_delta: number;
  ood_guard_applied: boolean;
}

export interface MedicalTrustReport {
  trust_tier: 'very_high_trust' | 'high_trust' | 'moderate_trust' | 'uncertain' | 'suspicious';
  trust_score: number;
  calibration_state: 'stable' | 'near_threshold' | 'softened' | 'suspicious';
  ece_at_training: number;
  temperature_used: number;
  uncertainty_reason: string | null;
  semantic_warning: string | null;
}

export interface MedicalRiskAssessment {
  risk_tier: MedicalRiskTier;
  final_score: number;
  imaging_score: number;
  clinical_modifier: number;
  near_boundary: boolean;
  boundary_proximity: number;
  requires_immediate_action: boolean;
  differential_classes: string[];
  tier_thresholds: {
    LOW_upper: number;
    MODERATE_upper: number;
    HIGH_DIFFERENTIAL_RISK_upper: number;
  };
}

export interface MedicalSemanticSignal {
  label: string;
  medical_relevance_score: number;
  ood_score: number;
  gate_passed: boolean;
  rejection_code: string | null;
  reasoning_decision: string | null;
  reasoning_confidence: number | null;
  top_matches: Array<{ label: string; score: number; rank: number }>;
}

export interface MedicalExplainability {
  summary: string;
  imaging_findings: string;
  clinical_context_applied: string | null;
  contradiction_note: string | null;
  gradcam_base64: string | null;
  gradcam_target_class: string | null;
  reasoning_chain: string[];
  pipeline_warnings: string[];
}

export interface UnifiedAnalysisSession {
  session_id: string;
  timestamp: string;
  imaging: MedicalImagingSignal;
  semantic: MedicalSemanticSignal | null;
  clinical: MedicalClinicalModifier;
  fusion: MedicalFusionReasoning;
  trust: MedicalTrustReport;
  risk: MedicalRiskAssessment;
  explainability: MedicalExplainability;
  ood_guard_applied: boolean;
  clinical_override_attempted: boolean;
  model_version: string;
  pipeline_version: string;
}

// ── Medical AI Assistant (Phase 26) ──────────────────────────────────────────

export interface MedicalAnalysisContext {
  risk_tier: "LOW" | "MODERATE" | "HIGH_DIFFERENTIAL_RISK" | "CRITICAL_PULMONARY_RISK";
  final_score: number;
  requires_immediate_action: boolean;
  near_boundary: boolean;
  has_image: boolean;
  predicted_class?: string | null;
  imaging_score?: number | null;
  bilateral_burden?: number | null;
  ood_detected: boolean;
  ood_label?: string | null;
  has_clinical: boolean;
  symptoms_flagged?: string[];
  respiratory_severity?: string | null;
  oxygenation_context?: string | null;
  fever_severity?: string | null;
  recent_worsening?: string | null;
  rodent_exposure_level?: string | null;
  symptom_duration_tier?: string | null;
  exposure_history?: string | null;
  age?: number | null;
  sex?: string | null;
  summary: string;
  imaging_findings?: string | null;
}

export interface MedicalAssistantRequest {
  message: string;
  session_id?: string;
  analysis_context: MedicalAnalysisContext;
}

export interface MedicalAssistantResponse {
  content: string;
  refused: boolean;
  refusal_reason?: string | null;
  model: string;
  duration_ms: number;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  timestamp: string;
  session_id: string;
}

export interface ClinicalContextRequest {
  // Legacy fields (backward-compatible)
  symptoms?: string[];
  exposure_history?: string | null;  // hospital | sick_contact | travel | healthcare_worker | immunocompromised
  duration_days?: number | null;
  severity?: 'mild' | 'moderate' | 'severe' | null;
  immunocompromised?: boolean;
  age_group?: 'adolescent' | 'young_adult' | 'adult' | 'older_adult' | 'elderly' | null;
  notes?: string | null;

  // Phase 22 — structured clinical signals (None = neutral, not treated as normal)
  age?: number | null;
  sex?: 'male' | 'female' | null;
  respiratory_severity?: 'normal' | 'mild' | 'severe' | null;
  oxygenation_context?: 'normal' | 'mild_drop' | 'severe_drop' | null;
  fever_severity?: 'none' | 'mild' | 'moderate' | 'high' | null;
  recent_worsening?: 'none' | 'some' | 'rapid_48h' | null;
  rodent_exposure_level?: 'none' | 'unsure' | 'rural_env' | 'possible_contact' | null;
  symptom_duration_tier?: '1_2_days' | '3_7_days' | 'over_1_week' | null;
}
