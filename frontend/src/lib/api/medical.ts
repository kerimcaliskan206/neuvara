import { api, getAuthToken } from "@/lib/api/client";
import { config } from "@/lib/config";
import type {
  ClinicalContextRequest,
  MedicalAnalysisContext,
  MedicalAssistantRequest,
  MedicalAssistantResponse,
  UnifiedAnalysisSession,
} from "@/lib/api/types";

export interface MedicalAnalyzeOptions {
  gradcam?: boolean;
  clinicalContext?: ClinicalContextRequest | null;
  onProgress?: (percent: number) => void;
  signal?: AbortSignal;
}

export const medicalApi = {
  async analyze(
    file: File,
    options: MedicalAnalyzeOptions = {},
  ): Promise<UnifiedAnalysisSession> {
    const form = new FormData();
    form.append("file", file);
    form.append("gradcam", String(options.gradcam ?? true));

    if (options.clinicalContext) {
      const serialized = JSON.stringify(options.clinicalContext);
      form.append("clinical_context", serialized);
    }

    const { data } = await api.post<UnifiedAnalysisSession>(
      "/medical/analyze",
      form,
      {
        headers: { "Content-Type": "multipart/form-data" },
        signal: options.signal,
        onUploadProgress: (event) => {
          if (!options.onProgress || !event.total) return;
          const pct = Math.round((event.loaded / event.total) * 100);
          options.onProgress(Math.max(0, Math.min(100, pct)));
        },
      },
    );
    return data;
  },

  async persistClinical(payload: {
    session_id: string;
    risk_tier: string;
    final_score: number;
    summary?: string;
    duration_ms?: number;
  }): Promise<void> {
    await api.post("/medical/persist-clinical", payload);
  },

  async assistant(payload: MedicalAssistantRequest): Promise<MedicalAssistantResponse> {
    const { data } = await api.post<MedicalAssistantResponse>(
      "/medical/assistant",
      payload,
    );
    return data;
  },
};

/**
 * Stream AI assistant response tokens via SSE.
 * Yields each token as it arrives; the caller accumulates them into the message.
 */
export async function* assistantStream(
  payload: MedicalAssistantRequest,
  signal?: AbortSignal,
): AsyncGenerator<string> {
  const token = getAuthToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "text/event-stream",
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const response = await fetch(`${config.api.v1}/medical/assistant/stream`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`AI assistant stream error: HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const raw = line.slice(6).trim();
        if (raw === "[DONE]") return;
        let parsed: { token?: string; error?: string };
        try {
          parsed = JSON.parse(raw) as { token?: string; error?: string };
        } catch {
          continue; // skip malformed JSON
        }
        if (parsed.error) throw new Error(parsed.error);
        if (parsed.token) yield parsed.token;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Build the curated MedicalAnalysisContext from a full UnifiedAnalysisSession.
 * Strips all engineering internals — only exposes clinically-relevant fields.
 */
export function buildAssistantContext(
  session: UnifiedAnalysisSession,
  clinicalCtx?: ClinicalContextRequest | null,
): MedicalAnalysisContext {
  const { risk, imaging, clinical, semantic, explainability } = session;

  return {
    risk_tier: risk.risk_tier,
    final_score: risk.final_score,
    requires_immediate_action: risk.requires_immediate_action,
    near_boundary: risk.near_boundary,

    has_image: true,
    predicted_class: imaging.predicted_class,
    imaging_score: imaging.imaging_score,
    bilateral_burden: null,  // not exposed in UnifiedAnalysisSession — backend uses it internally

    ood_detected: !!semantic?.ood_score && !semantic.gate_passed,
    ood_label: semantic?.label ?? null,

    has_clinical: clinical.provided,
    symptoms_flagged: clinical.symptoms_flagged ?? [],

    // Phase 22 fields from original clinical input (not stored in session — passed via clinicalCtx)
    respiratory_severity: clinicalCtx?.respiratory_severity ?? null,
    oxygenation_context: clinicalCtx?.oxygenation_context ?? null,
    fever_severity: clinicalCtx?.fever_severity ?? null,
    recent_worsening: clinicalCtx?.recent_worsening ?? null,
    rodent_exposure_level: clinicalCtx?.rodent_exposure_level ?? null,
    symptom_duration_tier: clinicalCtx?.symptom_duration_tier ?? null,
    exposure_history: clinicalCtx?.exposure_history ?? null,
    age: clinicalCtx?.age ?? null,
    sex: clinicalCtx?.sex ?? null,

    summary: explainability.summary,
    imaging_findings: explainability.imaging_findings ?? null,
  };
}
