"use client";

import { Brain } from "lucide-react";

import { cn } from "@/lib/utils";
import type { MedicalSemanticSignal } from "@/lib/api/types";

const DECISION_LABELS: Record<string, string> = {
  accepted:          "Kabul Edildi",
  rejected_ood:      "OOD Reddedildi",
  rejected_score:    "Skor Altı Reddedildi",
  overridden_accept: "Geçersiz Kılındı (Kabul)",
  overridden_reject: "Geçersiz Kılındı (Red)",
};

interface SemanticReasoningPanelProps {
  semantic: MedicalSemanticSignal | null;
}

export function SemanticReasoningPanel({ semantic }: SemanticReasoningPanelProps) {
  if (!semantic) {
    return (
      <div className="rounded-2xl glass-card-light p-5 animate-fade-up animate-delay-300">
        <div className="flex items-center gap-2 mb-3">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-canvas">
            <Brain className="h-3.5 w-3.5 text-foreground-muted" />
          </div>
          <p className="text-sm font-semibold text-foreground">Semantik Analiz</p>
        </div>
        <p className="text-sm text-foreground-muted">Semantik sinyal mevcut değil.</p>
      </div>
    );
  }

  const gatePassed = semantic.gate_passed;
  const relevancePct = (semantic.medical_relevance_score * 100).toFixed(1);
  const decisionLabel = semantic.reasoning_decision
    ? (DECISION_LABELS[semantic.reasoning_decision] ?? semantic.reasoning_decision)
    : null;

  return (
    <div className="rounded-2xl glass-card-light p-5 space-y-4 animate-fade-up animate-delay-300">
      <div className="flex items-center gap-2">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-brand-50">
          <Brain className="h-3.5 w-3.5 text-brand-600" />
        </div>
        <p className="text-sm font-semibold text-foreground">Semantik Analiz</p>
        <span
          className={cn(
            "ml-auto rounded-full border px-2.5 py-0.5 text-xs font-semibold",
            gatePassed
              ? "bg-success-50 text-success-700 border-success-100"
              : "bg-danger-50 text-danger-700 border-danger-100",
          )}
        >
          {gatePassed ? "Geçti" : "Reddedildi"}
        </span>
      </div>

      {/* Primary label + relevance */}
      <div className="rounded-xl border border-border bg-surface/60 p-3.5 space-y-2">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-xs text-foreground-muted">Tespit Edilen Kategori</p>
            <p className="mt-0.5 text-sm font-bold text-foreground">{semantic.label}</p>
          </div>
          <div className="text-right shrink-0">
            <p className="text-xs text-foreground-muted">Tıbbi İlgililik</p>
            <p className={cn(
              "mt-0.5 text-lg font-bold tabular-nums",
              semantic.medical_relevance_score >= 0.70 ? "text-success-600"
              : semantic.medical_relevance_score >= 0.45 ? "text-warning-600"
              : "text-danger-600",
            )}>
              {relevancePct}%
            </p>
          </div>
        </div>

        {/* Relevance bar */}
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-black/8">
          <div
            className={cn(
              "h-full rounded-full transition-[width] duration-700 ease-out",
              semantic.medical_relevance_score >= 0.70 ? "bg-success-400"
              : semantic.medical_relevance_score >= 0.45 ? "bg-warning-400"
              : "bg-danger-400",
            )}
            style={{ width: `${relevancePct}%` }}
          />
        </div>
      </div>

      {/* Reasoning decision */}
      {(decisionLabel || semantic.reasoning_confidence != null) && (
        <div className="flex items-center gap-3 flex-wrap text-xs text-foreground-muted">
          {decisionLabel && (
            <span className="rounded-md border border-border bg-canvas px-2.5 py-1 font-medium text-foreground-secondary">
              {decisionLabel}
            </span>
          )}
          {semantic.reasoning_confidence != null && (
            <span>
              CLIP güveni:{" "}
              <strong className="text-foreground tabular-nums">
                {(semantic.reasoning_confidence * 100).toFixed(1)}%
              </strong>
            </span>
          )}
        </div>
      )}

      {/* Top CLIP matches */}
      {semantic.top_matches.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-wider text-foreground-muted">
            En İyi Eşleşmeler
          </p>
          {semantic.top_matches.slice(0, 3).map((m) => (
            <div key={m.rank} className="flex items-center gap-2 text-xs">
              <span className="w-4 shrink-0 text-foreground-muted tabular-nums">{m.rank}.</span>
              <span className="flex-1 truncate text-foreground-secondary">{m.label}</span>
              <span className="tabular-nums font-semibold text-brand-600">
                {(m.score * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Rejection code */}
      {semantic.rejection_code && (
        <p className="rounded-lg border border-danger-100 bg-danger-50 px-3 py-2 text-xs text-danger-700 font-mono">
          Red kodu: {semantic.rejection_code}
        </p>
      )}
    </div>
  );
}
