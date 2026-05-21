"use client";

import { AlertTriangle, RefreshCw, RotateCcw, ShieldAlert, ShieldCheck, ShieldQuestion } from "lucide-react";

import { AiExplanationPanel } from "@/components/analysis/ai-explanation-panel";
import { ContributionBar } from "@/components/analysis/contribution-bar";
import { GradCamViewer } from "@/components/analysis/gradcam-viewer";
import { RiskMeter } from "@/components/analysis/risk-meter";
import { Alert } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { FusionResponse, InterpretationResponse, VisionPredictionResponse } from "@/lib/api/types";

const RISK_CONFIG = {
  high: {
    label: "Yüksek Risk",
    description: "HPS açısından yüksek risk tespit edildi. Klinik değerlendirme önerilir.",
    icon: ShieldAlert,
    headerClass:
      "glass-card border-danger-200/50 shadow-[0_8px_24px_-12px_hsl(0_72%_51%/0.28)]",
    iconBg: "bg-danger-100/60",
    iconClass: "text-danger-500",
    labelClass: "text-danger-700",
    descClass: "text-danger-600",
    glowHue: "0 72% 51%",
  },
  medium: {
    label: "Orta Risk",
    description: "Orta düzeyde risk faktörleri tespit edildi. Yakın takip önerilir.",
    icon: ShieldQuestion,
    headerClass:
      "glass-card border-warning-200/50 shadow-[0_8px_24px_-12px_hsl(38_90%_48%/0.25)]",
    iconBg: "bg-warning-100/60",
    iconClass: "text-warning-500",
    labelClass: "text-warning-700",
    descClass: "text-warning-600",
    glowHue: "38 90% 48%",
  },
  low: {
    label: "Düşük Risk",
    description: "Mevcut veriler düşük risk profiline işaret ediyor.",
    icon: ShieldCheck,
    headerClass:
      "glass-card border-success-200/50 shadow-[0_8px_24px_-12px_hsl(152_65%_48%/0.25)]",
    iconBg: "bg-success-100/60",
    iconClass: "text-success-500",
    labelClass: "text-success-700",
    descClass: "text-success-600",
    glowHue: "152 65% 48%",
  },
} as const;

const CONFIDENCE_LABEL: Record<string, string> = {
  high: "Yüksek",
  medium: "Orta",
  low: "Düşük",
};

const VISION_STATUS_LABEL: Record<string, string> = {
  USED: "Kullanıldı",
  REJECTED: "Reddedildi",
  UNAVAILABLE: "Mevcut değil",
  UNRELATED: "İlgisiz görüntü",
  LOW_CONFIDENCE: "Düşük güven — yok sayıldı",
};

interface FusionResultProps {
  result: FusionResponse;
  visionResult: VisionPredictionResponse | null;
  visionPreviewUrl: string | null;
  explanation: InterpretationResponse | null;
  isExplaining: boolean;
  isExplainError: boolean;
  explainError?: Error | null;
  onExplain: () => void;
  onRestart: () => void;
}

export function FusionResult({
  result,
  visionResult,
  visionPreviewUrl,
  explanation,
  isExplaining,
  isExplainError,
  explainError,
  onExplain,
  onRestart,
}: FusionResultProps) {
  const riskKey = result.risk_level as keyof typeof RISK_CONFIG;
  const config = RISK_CONFIG[riskKey] ?? RISK_CONFIG.low;
  const Icon = config.icon;

  const uncertaintyMessages: Record<string, string> = {
    ml_near_decision_boundary: "ML skoru karar sınırına yakın — belirsizlik yüksek",
    ml_low_confidence: "ML güveni düşük",
    vision_low_confidence_ignored: "Görüntü düşük güven nedeniyle yok sayıldı",
  };

  const hasUncertainty = result.uncertainty_flags.length > 0;
  const showGradCam =
    visionPreviewUrl != null &&
    visionResult?.gradcam_base64 != null;

  return (
    <div className="space-y-6">
      {/* Risk header — premium gradient surface with subtle glow */}
      <div
        className={cn(
          "relative overflow-hidden rounded-2xl border p-5 flex items-start gap-4 animate-fade-up",
          config.headerClass,
        )}
      >
        {/* Soft glow orb in the corner */}
        <div
          aria-hidden
          className="atmosphere-orb pointer-events-none"
          style={{
            top: "-40%",
            right: "-15%",
            width: 220,
            height: 220,
            background: `radial-gradient(circle, hsl(${config.glowHue} / 0.18) 0%, transparent 70%)`,
            opacity: 0.7,
            animationDuration: "30s",
          }}
        />
        <div
          className={cn(
            "relative z-10 flex h-11 w-11 shrink-0 items-center justify-center rounded-xl shadow-sm ring-1 ring-white/60",
            config.iconBg,
          )}
        >
          <Icon className={cn("h-5 w-5", config.iconClass)} />
        </div>
        <div className="relative z-10 min-w-0 flex-1">
          <p className={cn("text-base font-bold", config.labelClass)}>{config.label}</p>
          <p className={cn("mt-0.5 text-sm", config.descClass)}>{config.description}</p>
          <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-foreground-secondary">
            <span>Füzyon güveni: <strong>{CONFIDENCE_LABEL[result.fusion_confidence] ?? result.fusion_confidence}</strong></span>
            <span>·</span>
            <span>Ham skor: <strong>{(result.final_risk_score * 100).toFixed(1)}%</strong></span>
          </div>
        </div>
      </div>

      {/* Uncertainty */}
      {hasUncertainty && (
        <div className="animate-fade-up animate-delay-75">
        <Alert variant="warning" title="Belirsizlik Uyarıları">
          <ul className="mt-1 space-y-0.5">
            {result.uncertainty_flags.map((flag) => (
              <li key={flag} className="text-xs">
                {uncertaintyMessages[flag] ?? flag}
              </li>
            ))}
          </ul>
        </Alert>
        </div>
      )}

      {/* Risk meter + contributions */}
      <div className="grid gap-4 sm:grid-cols-2 animate-fade-up animate-delay-150">
        <div className="rounded-xl border border-border bg-surface p-4 flex flex-col items-center">
          <p className="text-xs font-semibold uppercase tracking-wider text-foreground-muted mb-3">
            Risk Skoru
          </p>
          <RiskMeter
            score={result.final_risk_score}
            level={result.risk_level}
            confidence={result.fusion_confidence}
          />
        </div>

        <div className="rounded-xl border border-border bg-surface p-4 space-y-4">
          <p className="text-xs font-semibold uppercase tracking-wider text-foreground-muted">
            Sinyal Katkıları
          </p>
          <ContributionBar
            mlContribution={result.ml_contribution}
            visionContribution={result.vision_contribution}
            mlWeight={result.weights_used.ml_weight}
            visionWeight={result.weights_used.vision_weight}
            visionStatus={result.vision_status.toLowerCase()}
          />
        </div>
      </div>

      {/* Weights detail */}
      <div className="rounded-xl border border-border bg-canvas p-4 animate-fade-up animate-delay-200">
        <p className="text-xs font-semibold uppercase tracking-wider text-foreground-muted mb-3">
          Ağırlık Politikası
        </p>
        <div className="grid gap-3 sm:grid-cols-3 text-center">
          <div>
            <p className="text-lg font-bold text-brand-600">
              {(result.weights_used.ml_weight * 100).toFixed(0)}%
            </p>
            <p className="text-xs text-foreground-muted">ML Ağırlığı (α)</p>
          </div>
          <div>
            <p className="text-lg font-bold text-success-600">
              {(result.weights_used.vision_weight * 100).toFixed(0)}%
            </p>
            <p className="text-xs text-foreground-muted">Görüntü Ağırlığı (β)</p>
          </div>
          <div>
            <p
              className={cn(
                "text-xs font-semibold px-2 py-1 rounded-md inline-block",
                result.vision_status === "USED"
                  ? "bg-success-50 text-success-700"
                  : "bg-warning-50 text-warning-700",
              )}
            >
              {VISION_STATUS_LABEL[result.vision_status] ?? result.vision_status}
            </p>
            <p className="text-xs text-foreground-muted mt-1">Görüntü Durumu</p>
          </div>
        </div>
        {result.weights_used.reason && (
          <p className="mt-3 text-xs text-foreground-muted border-t border-border pt-3">
            {result.weights_used.reason}
          </p>
        )}
      </div>

      {/* Grad-CAM */}
      {showGradCam && (
        <div className="rounded-xl border border-border bg-surface p-4 space-y-3 animate-fade-up animate-delay-300">
          <p className="text-xs font-semibold uppercase tracking-wider text-foreground-muted">
            Tahmin Edilen Sınıf — Model Dikkat Görselleştirmesi
          </p>
          <GradCamViewer
            originalImageUrl={visionPreviewUrl!}
            gradcamBase64={visionResult!.gradcam_base64}
          />
        </div>
      )}

      {/* AI Explanation */}
      <div className="rounded-xl border border-border bg-surface p-4 animate-fade-up animate-delay-450">
        <AiExplanationPanel
          isLoading={isExplaining}
          isError={isExplainError}
          error={explainError}
          data={explanation}
          onRequest={onExplain}
        />
      </div>

      {/* Disclaimer */}
      <div className="flex items-start gap-3 rounded-xl border border-border bg-canvas p-4 animate-fade-up animate-delay-600">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-foreground-muted" />
        <p className="text-xs text-foreground-muted leading-relaxed">
          Bu sistem bir karar destek aracıdır ve tıbbi teşhis koymaz. Sonuçlar
          yalnızca klinisyen değerlendirmesini desteklemek amacıyla üretilmektedir.
          Klinik kararlar daima uzman hekim tarafından verilmelidir.
        </p>
      </div>

      {/* Actions */}
      <div className="flex items-center justify-between gap-3">
        <Button variant="secondary" onClick={onRestart} size="sm">
          <RotateCcw className="h-3.5 w-3.5" />
          Yeni Analiz
        </Button>
        {!explanation && !isExplaining && (
          <Button onClick={onExplain} variant="primary" size="sm">
            <RefreshCw className="h-3.5 w-3.5" />
            AI ile Açıkla
          </Button>
        )}
      </div>
    </div>
  );
}
