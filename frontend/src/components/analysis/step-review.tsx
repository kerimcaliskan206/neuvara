import { CheckCircle2, Edit2, ImageOff } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { RiskFactorData, SymptomData } from "@/stores/fusion-store";
import type { VisionPredictionResponse } from "@/lib/api/types";

const SYMPTOM_LABELS: Record<string, string> = {
  fever: "Ateş",
  myalgia: "Miyalji",
  headache: "Baş Ağrısı",
  thrombocytopenia: "Trombositopeni",
};

const REGION_TR: Record<string, string> = {
  north: "Kuzey",
  south: "Güney",
  east: "Doğu",
  west: "Batı",
  central: "Orta",
};

const SEASON_TR: Record<string, string> = {
  spring: "İlkbahar",
  summer: "Yaz",
  fall: "Sonbahar",
  winter: "Kış",
};

interface ReviewSectionProps {
  title: string;
  onEdit: () => void;
  children: React.ReactNode;
}

function ReviewSection({ title, onEdit, children }: ReviewSectionProps) {
  return (
    <div className="rounded-xl border border-border bg-surface p-4 space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold uppercase tracking-wider text-foreground-muted">
          {title}
        </p>
        <button
          type="button"
          onClick={onEdit}
          className="flex items-center gap-1.5 text-xs text-brand-600 hover:text-brand-700"
        >
          <Edit2 className="h-3 w-3" />
          Düzenle
        </button>
      </div>
      {children}
    </div>
  );
}

interface TagProps {
  children: React.ReactNode;
  variant?: "active" | "neutral";
}

function Tag({ children, variant = "neutral" }: TagProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md px-2.5 py-0.5 text-xs font-medium",
        variant === "active"
          ? "bg-brand-50 text-brand-700"
          : "bg-canvas text-foreground-secondary border border-border",
      )}
    >
      {children}
    </span>
  );
}

interface StepReviewProps {
  symptoms: SymptomData;
  riskFactors: RiskFactorData;
  visionResult: VisionPredictionResponse | null;
  visionPreviewUrl: string | null;
  onNext: () => void;
  onBack: () => void;
  onGoToStep: (step: number) => void;
  isAnalyzing?: boolean;
}

export function StepReview({
  symptoms,
  riskFactors,
  visionResult,
  visionPreviewUrl,
  onNext,
  onBack,
  onGoToStep,
  isAnalyzing,
}: StepReviewProps) {
  const activeSymptoms = Object.entries(symptoms)
    .filter(([, v]) => v)
    .map(([k]) => SYMPTOM_LABELS[k] ?? k);

  const hasAnySymptom = activeSymptoms.length > 0;
  const hasAnyRiskFactor =
    riskFactors.age !== "" ||
    riskFactors.gender !== "" ||
    riskFactors.region !== "" ||
    riskFactors.season !== "" ||
    riskFactors.rodent_contact ||
    riskFactors.outdoor_work ||
    riskFactors.rodent_density !== "" ||
    riskFactors.precipitation_mm !== "" ||
    riskFactors.humidity_pct !== "";

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-bold text-foreground">İnceleme ve Onay</h2>
        <p className="mt-1 text-sm text-foreground-secondary">
          Girilen bilgileri kontrol edin. Analizi başlatmak için Devam edin.
        </p>
      </div>

      {/* Symptoms */}
      <ReviewSection title="Semptomlar" onEdit={() => onGoToStep(0)}>
        {hasAnySymptom ? (
          <div className="flex flex-wrap gap-2">
            {activeSymptoms.map((s) => (
              <Tag key={s} variant="active">
                <CheckCircle2 className="mr-1 h-3 w-3" />
                {s}
              </Tag>
            ))}
          </div>
        ) : (
          <p className="text-xs text-foreground-muted italic">
            Semptom seçilmedi — eksik veriler ML tarafından doldurulacak
          </p>
        )}
      </ReviewSection>

      {/* Risk Factors */}
      <ReviewSection title="Risk Faktörleri" onEdit={() => onGoToStep(1)}>
        {hasAnyRiskFactor ? (
          <div className="flex flex-wrap gap-2">
            {riskFactors.age !== "" && (
              <Tag>Yaş: {riskFactors.age}</Tag>
            )}
            {riskFactors.gender !== "" && (
              <Tag>{riskFactors.gender === "M" ? "Erkek" : "Kadın"}</Tag>
            )}
            {riskFactors.region !== "" && (
              <Tag>{REGION_TR[riskFactors.region] ?? riskFactors.region}</Tag>
            )}
            {riskFactors.season !== "" && (
              <Tag>{SEASON_TR[riskFactors.season] ?? riskFactors.season}</Tag>
            )}
            {riskFactors.rodent_contact && <Tag>Kemirici Teması</Tag>}
            {riskFactors.outdoor_work && <Tag>Dış Ortam Çalışması</Tag>}
            {riskFactors.rodent_density !== "" && (
              <Tag>Kemirici Yoğunluğu: {riskFactors.rodent_density}</Tag>
            )}
            {riskFactors.precipitation_mm !== "" && (
              <Tag>Yağış: {riskFactors.precipitation_mm} mm</Tag>
            )}
            {riskFactors.humidity_pct !== "" && (
              <Tag>Nem: {riskFactors.humidity_pct}%</Tag>
            )}
          </div>
        ) : (
          <p className="text-xs text-foreground-muted italic">
            Risk faktörü girilmedi — tüm alanlar isteğe bağlıdır
          </p>
        )}
      </ReviewSection>

      {/* Vision */}
      <ReviewSection title="Görüntü Analizi" onEdit={() => onGoToStep(2)}>
        {visionResult ? (
          <div className="flex items-center gap-3">
            {visionPreviewUrl && (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={visionPreviewUrl}
                alt="Seçilen görüntü"
                className="h-14 w-14 rounded-lg object-cover border border-border shrink-0"
              />
            )}
            <div>
              <p
                className={cn(
                  "text-sm font-semibold",
                  visionResult.accepted ? "text-success-600" : "text-warning-600",
                )}
              >
                {visionResult.accepted ? "✓ Kabul edildi" : "⚠ Reddedildi"}
              </p>
              <p className="text-xs text-foreground-secondary mt-0.5">
                {visionResult.accepted
                  ? `${visionResult.predicted_class ?? "—"} · ${((visionResult.confidence ?? 0) * 100).toFixed(1)}% güven · β=25% katkı`
                  : `${visionResult.rejection_reason ?? "Reddedildi"} · β=0%`}
              </p>
            </div>
          </div>
        ) : (
          <div className="flex items-center gap-2 text-foreground-muted">
            <ImageOff className="h-4 w-4 shrink-0" />
            <p className="text-xs italic">
              Görüntü eklenmedi — ML analizi tek başına çalışacak (β=0%)
            </p>
          </div>
        )}
      </ReviewSection>

      {/* Info */}
      <div className="rounded-xl bg-brand-50 border border-brand-100 p-4">
        <p className="text-xs text-brand-800 leading-relaxed">
          <strong>Füzyon motoru:</strong> ML semptom skoru ağırlık α=75% ile birincil sinyal olarak kullanılır.
          {visionResult?.accepted
            ? " Görüntü β=25% katkı sağlayacak."
            : " Görüntü mevcut olmadığından tüm ağırlık ML'e ayrılacak."}
        </p>
      </div>

      {/* Navigation */}
      <div className="flex items-center justify-between gap-3">
        <Button variant="secondary" onClick={onBack}>
          Geri
        </Button>
        <Button onClick={onNext} size="lg" isLoading={isAnalyzing} disabled={isAnalyzing}>
          {isAnalyzing ? "Analiz ediliyor…" : "Analizi Başlat"}
        </Button>
      </div>
    </div>
  );
}
