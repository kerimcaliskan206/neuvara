"use client";

import { Activity } from "lucide-react";

import { cn } from "@/lib/utils";
import type { MedicalImagingSignal } from "@/lib/api/types";

const CLASS_LABELS: Record<string, string> = {
  healthy_xray:    "Normal Akciğer Grafisi",
  pneumonia_xray:  "Pnömoni Paterni",
  hard_negative:   "İlgisiz İçerik",
  fake_medical:    "Sahte Tıbbi Görüntü",
};

const CLASS_DESCRIPTION: Record<string, string> = {
  healthy_xray:    "Belirgin pulmoner patoloji bulgusu izlenmemektedir.",
  pneumonia_xray:  "Konsolidasyon veya infiltrat paterni saptanmaktadır.",
  hard_negative:   "Görüntü tıbbi radyoloji içeriği taşımamaktadır.",
  fake_medical:    "Görüntünün orijinalliği sorgulanmaktadır; gerçek tıbbi görüntü olmayabilir.",
};

interface ImagingAnalysisPanelProps {
  imaging: MedicalImagingSignal;
}

export function ImagingAnalysisPanel({ imaging }: ImagingAnalysisPanelProps) {
  const sortedProbs = Object.entries(imaging.class_probabilities)
    .sort(([, a], [, b]) => b - a);

  const displayLabel = CLASS_LABELS[imaging.predicted_class] ?? imaging.predicted_class;
  const displayDesc  = CLASS_DESCRIPTION[imaging.predicted_class] ?? "";

  return (
    <div className="rounded-2xl glass-card-light p-5 space-y-4 animate-fade-up animate-delay-150">
      <div className="flex items-center gap-2">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-brand-50">
          <Activity className="h-3.5 w-3.5 text-brand-600" />
        </div>
        <p className="text-sm font-semibold text-foreground">Görüntü Analizi</p>
        <span className="ml-auto text-xs text-foreground-muted">
          {imaging.inference_ms.toFixed(0)} ms
        </span>
      </div>

      {/* Primary finding */}
      <div className="rounded-xl border border-border bg-surface/60 p-4 space-y-1">
        <p className="text-xs text-foreground-muted uppercase tracking-wider font-semibold">
          Birincil Bulgu
        </p>
        <p className="text-base font-bold text-foreground">{displayLabel}</p>
        {displayDesc && (
          <p className="text-xs text-foreground-secondary leading-relaxed">{displayDesc}</p>
        )}
        <div className="mt-2 flex flex-wrap gap-3 text-xs text-foreground-muted">
          <span>
            Güven:{" "}
            <strong className="text-foreground">
              {(imaging.calibrated_confidence * 100).toFixed(1)}%
            </strong>
          </span>
          <span>·</span>
          <span>
            Ham güven:{" "}
            <strong className="text-foreground">
              {(imaging.raw_confidence * 100).toFixed(1)}%
            </strong>
          </span>
          <span>·</span>
          <span>
            T*={" "}
            <strong className="text-foreground font-mono">
              {imaging.temperature_applied.toFixed(4)}
            </strong>
          </span>
        </div>
      </div>

      {/* Class probabilities */}
      <div className="space-y-2.5">
        <p className="text-xs font-semibold uppercase tracking-wider text-foreground-muted">
          Sınıf Olasılıkları
        </p>
        {sortedProbs.map(([cls, prob]) => {
          const isPredicted = cls === imaging.predicted_class;
          const pct = prob * 100;
          return (
            <div key={cls} className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span
                  className={cn(
                    isPredicted
                      ? "font-semibold text-foreground"
                      : "text-foreground-secondary",
                  )}
                >
                  {CLASS_LABELS[cls] ?? cls}
                </span>
                <span
                  className={cn(
                    "tabular-nums font-semibold",
                    isPredicted ? "text-brand-600" : "text-foreground-muted",
                  )}
                >
                  {pct.toFixed(1)}%
                </span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-canvas border border-border-subtle">
                <div
                  className={cn(
                    "h-full rounded-full transition-[width] duration-700 ease-out",
                    isPredicted ? "bg-brand-500" : "bg-foreground-muted/30",
                  )}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <p className="text-2xs text-foreground-muted">
        Model: {imaging.model_version} · Kalibrasyon uygulandı (T*={imaging.temperature_applied.toFixed(4)})
      </p>
    </div>
  );
}
