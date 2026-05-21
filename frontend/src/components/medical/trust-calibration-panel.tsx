"use client";

import { ShieldCheck } from "lucide-react";

import { cn } from "@/lib/utils";
import type { MedicalTrustReport } from "@/lib/api/types";

const TRUST_TIER_CONFIG: Record<string, { label: string; color: string; bg: string }> = {
  very_high_trust: { label: "Çok Yüksek Güven",  color: "text-success-700",  bg: "bg-success-50 border-success-100" },
  high_trust:      { label: "Yüksek Güven",       color: "text-success-600",  bg: "bg-success-50/70 border-success-100" },
  moderate_trust:  { label: "Orta Düzey Güven",   color: "text-warning-700",  bg: "bg-warning-50 border-warning-100" },
  uncertain:       { label: "Belirsiz",            color: "text-orange-600",   bg: "bg-orange-50 border-orange-100" },
  suspicious:      { label: "Şüpheli",             color: "text-danger-600",   bg: "bg-danger-50 border-danger-100" },
};

const CALIBRATION_STATE_LABELS: Record<string, string> = {
  stable:          "Kararlı",
  near_threshold:  "Eşik Yakını",
  softened:        "Dengelenmiş",
  suspicious:      "Şüpheli",
};

interface TrustCalibrationPanelProps {
  trust: MedicalTrustReport;
}

export function TrustCalibrationPanel({ trust }: TrustCalibrationPanelProps) {
  const tierCfg = (TRUST_TIER_CONFIG[trust.trust_tier] ?? TRUST_TIER_CONFIG.moderate_trust)!;
  const trustPct = Math.round(trust.trust_score * 100);
  const calLabel = CALIBRATION_STATE_LABELS[trust.calibration_state] ?? trust.calibration_state;

  return (
    <div className="rounded-2xl glass-card-light p-5 space-y-4 animate-fade-up animate-delay-350">
      <div className="flex items-center gap-2">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-brand-50">
          <ShieldCheck className="h-3.5 w-3.5 text-brand-600" />
        </div>
        <p className="text-sm font-semibold text-foreground">Güven & Kalibrasyon</p>
      </div>

      {/* Trust tier badge */}
      <div className={cn("rounded-xl border px-4 py-3", tierCfg.bg)}>
        <div className="flex items-center justify-between">
          <p className={cn("text-sm font-bold", tierCfg.color)}>{tierCfg.label}</p>
          <p className={cn("text-lg font-bold tabular-nums", tierCfg.color)}>
            {trustPct}%
          </p>
        </div>
        {/* Trust bar */}
        <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-black/8">
          <div
            className={cn(
              "h-full rounded-full transition-[width] duration-700 ease-out",
              trust.trust_score >= 0.72 ? "bg-success-500"
              : trust.trust_score >= 0.55 ? "bg-warning-400"
              : "bg-danger-400",
            )}
            style={{ width: `${trustPct}%` }}
          />
        </div>
      </div>

      {/* Calibration metrics */}
      <div className="grid grid-cols-3 gap-3 text-center">
        <div className="rounded-lg border border-border bg-canvas p-2.5">
          <p className="text-xs text-foreground-muted">Kalibrasyon Durumu</p>
          <p className="mt-0.5 text-sm font-bold text-foreground">{calLabel}</p>
        </div>
        <div className="rounded-lg border border-border bg-canvas p-2.5">
          <p className="text-xs text-foreground-muted">ECE (Eğitim)</p>
          <p className="mt-0.5 text-sm font-bold text-brand-600">
            {(trust.ece_at_training * 100).toFixed(2)}%
          </p>
        </div>
        <div className="rounded-lg border border-border bg-canvas p-2.5">
          <p className="text-xs text-foreground-muted">Sıcaklık (T*)</p>
          <p className="mt-0.5 text-sm font-bold font-mono text-foreground">
            {trust.temperature_used.toFixed(4)}
          </p>
        </div>
      </div>

      {/* Advisory notes */}
      {(trust.uncertainty_reason || trust.semantic_warning) && (
        <div className="space-y-2">
          {trust.uncertainty_reason && (
            <p className="rounded-lg bg-warning-50 border border-warning-100 px-3 py-2 text-xs text-warning-700 leading-relaxed">
              {trust.uncertainty_reason}
            </p>
          )}
          {trust.semantic_warning && (
            <p className="rounded-lg bg-canvas border border-border px-3 py-2 text-xs text-foreground-secondary leading-relaxed">
              {trust.semantic_warning}
            </p>
          )}
        </div>
      )}

      <p className="text-2xs text-foreground-muted">
        Güven skoru: füzyon güveni × belirsizlik × semantik hizalama × plausibilite çarpımıdır.
        T* kalibrasyon ölçeği görüntü skorunu etkilemez, yalnızca olasılık çıktılarını kalibre eder.
      </p>
    </div>
  );
}
