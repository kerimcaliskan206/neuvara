import { create } from "zustand";

import type {
  FusionResponse,
  PatientInput,
  VisionPredictionResponse,
} from "@/lib/api/types";

// ── Step index ─────────────────────────────────────────────────────────────────

export const WIZARD_STEPS = [
  "Semptomlar",
  "Risk Faktörleri",
  "Görüntü",
  "İnceleme",
  "Sonuç",
] as const;

export type WizardStep = 0 | 1 | 2 | 3 | 4;

// ── Typed sub-slices ──────────────────────────────────────────────────────────

export interface SymptomData {
  fever: boolean;
  myalgia: boolean;
  headache: boolean;
  thrombocytopenia: boolean;
}

export interface RiskFactorData {
  age: number | "";
  gender: "M" | "F" | "";
  region: string;
  season: string;
  rodent_contact: boolean;
  outdoor_work: boolean;
  rodent_density: number | "";
  precipitation_mm: number | "";
  humidity_pct: number | "";
}

export const DEFAULT_SYMPTOMS: SymptomData = {
  fever: false,
  myalgia: false,
  headache: false,
  thrombocytopenia: false,
};

export const DEFAULT_RISK_FACTORS: RiskFactorData = {
  age: "",
  gender: "",
  region: "",
  season: "",
  rodent_contact: false,
  outdoor_work: false,
  rodent_density: "",
  precipitation_mm: "",
  humidity_pct: "",
};

// ── Store ──────────────────────────────────────────────────────────────────────

interface FusionStore {
  step: WizardStep;
  symptoms: SymptomData;
  riskFactors: RiskFactorData;
  visionResult: VisionPredictionResponse | null;
  visionPreviewUrl: string | null;
  fusionResult: FusionResponse | null;
  isAnalyzing: boolean;
  analysisError: string | null;

  // Actions
  setStep: (step: WizardStep) => void;
  nextStep: () => void;
  prevStep: () => void;
  setSymptoms: (data: Partial<SymptomData>) => void;
  setRiskFactors: (data: Partial<RiskFactorData>) => void;
  setVisionResult: (result: VisionPredictionResponse | null, previewUrl?: string | null) => void;
  setFusionResult: (result: FusionResponse) => void;
  setAnalyzing: (state: boolean) => void;
  setAnalysisError: (error: string | null) => void;
  reset: () => void;

  // Derived helper — build PatientInput from wizard data
  buildPatientInput: () => PatientInput;
}

export const useFusionStore = create<FusionStore>((set, get) => ({
  step: 0,
  symptoms: { ...DEFAULT_SYMPTOMS },
  riskFactors: { ...DEFAULT_RISK_FACTORS },
  visionResult: null,
  visionPreviewUrl: null,
  fusionResult: null,
  isAnalyzing: false,
  analysisError: null,

  setStep: (step) => set({ step }),
  nextStep: () =>
    set((s) => ({ step: Math.min(4, s.step + 1) as WizardStep })),
  prevStep: () =>
    set((s) => ({ step: Math.max(0, s.step - 1) as WizardStep })),

  setSymptoms: (data) =>
    set((s) => ({ symptoms: { ...s.symptoms, ...data } })),
  setRiskFactors: (data) =>
    set((s) => ({ riskFactors: { ...s.riskFactors, ...data } })),

  setVisionResult: (result, previewUrl = null) =>
    set({ visionResult: result, visionPreviewUrl: previewUrl }),

  setFusionResult: (result) => set({ fusionResult: result }),
  setAnalyzing: (state) => set({ isAnalyzing: state }),
  setAnalysisError: (error) => set({ analysisError: error }),

  reset: () =>
    set({
      step: 0,
      symptoms: { ...DEFAULT_SYMPTOMS },
      riskFactors: { ...DEFAULT_RISK_FACTORS },
      visionResult: null,
      visionPreviewUrl: null,
      fusionResult: null,
      isAnalyzing: false,
      analysisError: null,
    }),

  buildPatientInput: (): PatientInput => {
    const { symptoms, riskFactors } = get();
    return {
      fever: symptoms.fever ? 1 : 0,
      myalgia: symptoms.myalgia ? 1 : 0,
      headache: symptoms.headache ? 1 : 0,
      thrombocytopenia: symptoms.thrombocytopenia ? 1 : 0,
      age: riskFactors.age !== "" ? Number(riskFactors.age) : null,
      gender: riskFactors.gender || null,
      region: riskFactors.region || null,
      season: riskFactors.season || null,
      rodent_contact: riskFactors.rodent_contact ? 1 : 0,
      outdoor_work: riskFactors.outdoor_work ? 1 : 0,
      rodent_density:
        riskFactors.rodent_density !== ""
          ? Number(riskFactors.rodent_density)
          : null,
      precipitation_mm:
        riskFactors.precipitation_mm !== ""
          ? Number(riskFactors.precipitation_mm)
          : null,
      humidity_pct:
        riskFactors.humidity_pct !== ""
          ? Number(riskFactors.humidity_pct)
          : null,
    };
  },
}));
