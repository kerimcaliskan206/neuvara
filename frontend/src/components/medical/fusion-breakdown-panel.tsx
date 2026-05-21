"use client";

import { GitMerge } from "lucide-react";

import { cn } from "@/lib/utils";
import type { MedicalFusionReasoning } from "@/lib/api/types";

const ALIGNMENT_CONFIG = {
  aligned:    { label: "Hizalı",       color: "text-success-700 bg-success-50 border-success-100" },
  misaligned: { label: "Uyumsuz",      color: "text-danger-700 bg-danger-50 border-danger-100" },
  uncertain:  { label: "Belirsiz",     color: "text-warning-700 bg-warning-50 border-warning-100" },
};

interface FusionBreakdownPanelProps {
  fusion: MedicalFusionReasoning;
}

export function FusionBreakdownPanel({ fusion }: FusionBreakdownPanelProps) {
  const alignCfg = ALIGNMENT_CONFIG[fusion.semantic_alignment] ?? ALIGNMENT_CONFIG.uncertain;

  const imagingPct  = (fusion.imaging_weight  * 100).toFixed(0);
  const clinicalPct = (fusion.clinical_weight * 100).toFixed(0);
  const agreementPct  = (fusion.agreement_score   * 100).toFixed(1);
  const uncertaintyPct = (fusion.uncertainty_score * 100).toFixed(1);

  const fusionDeltaSign = fusion.fusion_delta >= 0 ? "+" : "";
  const fusionDeltaPct  = (fusion.fusion_delta * 100).toFixed(1);

  return (
    <div className="rounded-2xl glass-card-light p-5 space-y-4 animate-fade-up animate-delay-400">
      <div className="flex items-center gap-2">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-brand-50">
          <GitMerge className="h-3.5 w-3.5 text-brand-600" />
        </div>
        <p className="text-sm font-semibold text-foreground">Füzyon Dağılımı</p>
        <span className={cn("ml-auto rounded-full border px-2.5 py-0.5 text-xs font-semibold", alignCfg.color)}>
          {alignCfg.label}
        </span>
      </div>

      {/* Weight bars */}
      <div className="space-y-3">
        <p className="text-xs font-semibold uppercase tracking-wider text-foreground-muted">
          Ağırlık Dağılımı
        </p>

        <div className="space-y-1.5">
          <div className="flex justify-between text-xs">
            <span className="text-foreground-secondary font-medium">Görüntü Sinyali</span>
            <span className="tabular-nums font-bold text-brand-600">{imagingPct}%</span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-black/8">
            <div
              className="h-full rounded-full bg-brand-500 transition-[width] duration-700 ease-out"
              style={{ width: `${imagingPct}%` }}
            />
          </div>
        </div>

        <div className="space-y-1.5">
          <div className="flex justify-between text-xs">
            <span className="text-foreground-secondary font-medium">Klinik Bağlam</span>
            <span className="tabular-nums font-bold text-foreground-muted">{clinicalPct}%</span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-black/8">
            <div
              className="h-full rounded-full bg-foreground-muted/40 transition-[width] duration-700 ease-out"
              style={{ width: `${clinicalPct}%` }}
            />
          </div>
        </div>
      </div>

      {/* Metrics grid */}
      <div className="grid grid-cols-3 gap-3 text-center">
        <div className="rounded-lg border border-border bg-canvas p-2.5">
          <p className="text-xs text-foreground-muted">Uyum Skoru</p>
          <p className={cn(
            "mt-0.5 text-sm font-bold tabular-nums",
            fusion.agreement_score >= 0.70 ? "text-success-600"
            : fusion.agreement_score >= 0.45 ? "text-warning-600"
            : "text-danger-600",
          )}>
            {agreementPct}%
          </p>
        </div>
        <div className="rounded-lg border border-border bg-canvas p-2.5">
          <p className="text-xs text-foreground-muted">Belirsizlik</p>
          <p className={cn(
            "mt-0.5 text-sm font-bold tabular-nums",
            fusion.uncertainty_score <= 0.20 ? "text-success-600"
            : fusion.uncertainty_score <= 0.40 ? "text-warning-600"
            : "text-danger-600",
          )}>
            {uncertaintyPct}%
          </p>
        </div>
        <div className="rounded-lg border border-border bg-canvas p-2.5">
          <p className="text-xs text-foreground-muted">Füzyon Δ</p>
          <p className={cn(
            "mt-0.5 text-sm font-bold tabular-nums",
            fusion.fusion_delta > 0.03 ? "text-warning-600"
            : fusion.fusion_delta < -0.03 ? "text-success-600"
            : "text-foreground",
          )}>
            {fusionDeltaSign}{fusionDeltaPct}%
          </p>
        </div>
      </div>

      {/* OOD guard notice */}
      {fusion.ood_guard_applied && (
        <p className="rounded-lg border border-warning-100 bg-warning-50 px-3 py-2 text-xs text-warning-700 leading-relaxed">
          OOD koruyucusu devreye girdi. Risk skoru {15}% üst sınırına sabitlendi.
        </p>
      )}

      <p className="text-2xs text-foreground-muted">
        Füzyon: görüntü birincil sinyal, klinik katkı ±%15 sınırlıdır.
      </p>
    </div>
  );
}
