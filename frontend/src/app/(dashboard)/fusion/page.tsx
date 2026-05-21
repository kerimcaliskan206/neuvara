"use client";

import { useCallback } from "react";

import { FusionResult } from "@/components/analysis/fusion-result";
import { StepImage } from "@/components/analysis/step-image";
import { StepReview } from "@/components/analysis/step-review";
import { StepRiskFactors } from "@/components/analysis/step-risk-factors";
import { StepSymptoms } from "@/components/analysis/step-symptoms";
import { WizardProgress } from "@/components/analysis/wizard-progress";
import { Alert } from "@/components/ui/alert";
import { useExplainFusion } from "@/hooks/use-ai-explanation";
import { useFusionPredict } from "@/hooks/use-fusion";
import { useFusionStore, type WizardStep } from "@/stores/fusion-store";
import type { VisionPredictionResponse } from "@/lib/api/types";

export default function FusionPage() {
  const store = useFusionStore();

  const fusion = useFusionPredict();
  const explain = useExplainFusion();

  // ── Navigation helpers ─────────────────────────────────────────────────────

  const goTo = useCallback(
    (step: number) => store.setStep(step as WizardStep),
    [store],
  );

  // ── Step 4 → trigger fusion on "Analizi Başlat" ──────────────────────────

  function handleAnalyze() {
    const patient = store.buildPatientInput();
    const vision = store.visionResult
      ? {
          accepted: store.visionResult.accepted,
          predicted_class: store.visionResult.predicted_class,
          predicted_class_index: store.visionResult.predicted_class_index,
          confidence: store.visionResult.confidence,
          probabilities: store.visionResult.probabilities,
          rejection_reason: store.visionResult.rejection_reason,
          model_name: store.visionResult.model_name,
          model_version: store.visionResult.model_version,
          gradcam_base64: store.visionResult.gradcam_base64,
        }
      : null;

    fusion.mutate(
      { patient, vision },
      {
        onSuccess: (data) => {
          store.setFusionResult(data);
          store.nextStep();
        },
      },
    );
  }

  // ── Step 5 → AI explanation ───────────────────────────────────────────────

  function handleExplain() {
    if (!store.fusionResult) return;
    explain.mutate(store.fusionResult);
  }

  // ── Restart ───────────────────────────────────────────────────────────────

  function handleRestart() {
    fusion.reset();
    explain.reset();
    store.reset();
  }

  // ── Handle vision result from step-image ─────────────────────────────────

  function handleVisionResult(
    result: VisionPredictionResponse | null,
    previewUrl: string | null,
  ) {
    store.setVisionResult(result, previewUrl);
  }

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="mx-auto max-w-2xl space-y-6 pb-12">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-white">
          Çok Modlu Füzyon Analizi
        </h1>
        <p className="mt-1 text-sm text-white/65">
          Semptom verisi ve görsel kanıtı birleştiren bütünleşik risk değerlendirmesi.
        </p>
      </div>

      {/* Wizard progress bar */}
      <WizardProgress
        currentStep={store.step}
        onStepClick={goTo}
      />

      {/* Fusion API error (step 4 → 5 transition) */}
      {fusion.isError && (
        <Alert variant="danger" title="Analiz başarısız">
          {fusion.error?.message ?? "Sunucu bağlantı hatası. Lütfen tekrar deneyin."}
        </Alert>
      )}

      {/* Step panels — keyed wrapper so each step re-animates on transition */}
      <div className="rounded-2xl glass-card-light p-5 sm:p-6">
        <div key={store.step} className="step-enter">
        {store.step === 0 && (
          <StepSymptoms
            data={store.symptoms}
            onChange={store.setSymptoms}
            onNext={() => goTo(1)}
          />
        )}

        {store.step === 1 && (
          <StepRiskFactors
            data={store.riskFactors}
            onChange={store.setRiskFactors}
            onNext={() => goTo(2)}
            onBack={() => goTo(0)}
          />
        )}

        {store.step === 2 && (
          <StepImage
            visionResult={store.visionResult}
            visionPreviewUrl={store.visionPreviewUrl}
            onVisionResult={handleVisionResult}
            onNext={() => goTo(3)}
            onBack={() => goTo(1)}
            onSkip={() => goTo(3)}
          />
        )}

        {store.step === 3 && (
          <StepReview
            symptoms={store.symptoms}
            riskFactors={store.riskFactors}
            visionResult={store.visionResult}
            visionPreviewUrl={store.visionPreviewUrl}
            onNext={handleAnalyze}
            onBack={() => goTo(2)}
            onGoToStep={goTo}
            isAnalyzing={fusion.isPending}
          />
        )}

        {store.step === 4 && store.fusionResult && (
          <FusionResult
            result={store.fusionResult}
            visionResult={store.visionResult}
            visionPreviewUrl={store.visionPreviewUrl}
            explanation={explain.data ?? null}
            isExplaining={explain.isPending}
            isExplainError={explain.isError}
            explainError={explain.error}
            onExplain={handleExplain}
            onRestart={handleRestart}
          />
        )}
        </div>
      </div>
    </div>
  );
}
