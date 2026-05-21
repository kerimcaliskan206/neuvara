"use client";

import { motion } from "framer-motion";
import {
  AlertTriangle,
  ShieldAlert,
  ShieldCheck,
  ShieldQuestion,
  ShieldX,
} from "lucide-react";

import { useCountUp } from "@/hooks/use-count-up";
import { cn } from "@/lib/utils";
import type { MedicalRiskAssessment, MedicalRiskTier } from "@/lib/api/types";

// ── Tier config ───────────────────────────────────────────────────────────────

const TIER_CONFIG = {
  LOW: {
    label: "Düşük Risk",
    sublabel: "Pulmoner patoloji bulgusu saptanmadı.",
    icon: ShieldCheck,
    headerClass:
      "glass-card border-success-200/50 shadow-[0_8px_32px_-12px_hsl(152_65%_48%/0.25)]",
    iconBg: "bg-success-100/60",
    iconClass: "text-success-500",
    labelClass: "text-success-700",
    sublabelClass: "text-success-600",
    scoreClass: "text-success-500",
    barClass: "bg-success-500",
    glowHue: "152 65% 48%",
    badgeClass: "bg-success-50/80 text-success-700 border-success-200/60",
  },
  MODERATE: {
    label: "Orta Düzey Risk",
    sublabel: "Hafif pulmoner değişiklik izlenmektedir; klinik takip önerilir.",
    icon: ShieldQuestion,
    headerClass:
      "glass-card border-warning-200/50 shadow-[0_8px_32px_-12px_hsl(38_90%_48%/0.25)]",
    iconBg: "bg-warning-100/60",
    iconClass: "text-warning-500",
    labelClass: "text-warning-700",
    sublabelClass: "text-warning-600",
    scoreClass: "text-warning-500",
    barClass: "bg-warning-400",
    glowHue: "38 90% 48%",
    badgeClass: "bg-warning-50/80 text-warning-700 border-warning-200/60",
  },
  HIGH_DIFFERENTIAL_RISK: {
    label: "Yüksek Diferansiyel Risk",
    sublabel: "Belirgin pulmoner anormallik tespit edildi; geniş ayırıcı tanı gereklidir.",
    icon: ShieldAlert,
    headerClass:
      "glass-card border-danger-200/50 shadow-[0_8px_32px_-12px_hsl(0_72%_51%/0.28)]",
    iconBg: "bg-danger-100/60",
    iconClass: "text-danger-500",
    labelClass: "text-danger-700",
    sublabelClass: "text-danger-600",
    scoreClass: "text-danger-500",
    barClass: "bg-danger-400",
    glowHue: "0 72% 51%",
    badgeClass: "bg-danger-50/80 text-danger-700 border-danger-200/60",
  },
  CRITICAL_PULMONARY_RISK: {
    label: "Kritik Pulmoner Risk",
    sublabel: "Ciddi bilateral pulmoner tutulum paterni. Acil klinik değerlendirme gereklidir.",
    icon: ShieldX,
    headerClass:
      "glass-card border-danger-300/60 shadow-[0_12px_40px_-12px_hsl(0_72%_51%/0.45)]",
    iconBg: "bg-danger-200/60",
    iconClass: "text-danger-500",
    labelClass: "text-danger-700",
    sublabelClass: "text-danger-600",
    scoreClass: "text-danger-500",
    barClass: "bg-danger-500",
    glowHue: "0 72% 51%",
    badgeClass: "bg-danger-100/80 text-danger-700 border-danger-300/60",
  },
} as const satisfies Record<MedicalRiskTier, object>;

// ── Score bar ─────────────────────────────────────────────────────────────────

function ScoreBar({
  score,
  barClass,
  thresholds,
}: {
  score: number;
  barClass: string;
  thresholds: MedicalRiskAssessment["tier_thresholds"];
}) {
  const pct = Math.round(score * 100);
  return (
    <div className="space-y-1">
      <div className="relative h-2 w-full overflow-hidden rounded-full bg-white/8">
        {/* Tier boundary markers */}
        {[thresholds.LOW_upper, thresholds.MODERATE_upper, thresholds.HIGH_DIFFERENTIAL_RISK_upper].map((t) => (
          <div
            key={t}
            className="absolute top-0 h-full w-px bg-white/30"
            style={{ left: `${t * 100}%` }}
          />
        ))}
        <motion.div
          className={cn("h-full rounded-full", barClass)}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.9, delay: 0.15, ease: [0.2, 0, 0, 1] }}
        />
      </div>
      <div className="flex justify-between text-2xs text-foreground-muted/50">
        <span>0</span>
        <span>35</span>
        <span>60</span>
        <span>80</span>
        <span>100</span>
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

interface FinalRiskCardProps {
  risk: MedicalRiskAssessment;
  sessionId: string;
  requiresImmediateAction: boolean;
  hideBreakdown?: boolean;
}

export function FinalRiskCard({ risk, sessionId, requiresImmediateAction, hideBreakdown = false }: FinalRiskCardProps) {
  const cfg = TIER_CONFIG[risk.risk_tier];
  const Icon = cfg.icon;
  const animatedScore = useCountUp(risk.final_score * 100, { duration: 800, decimals: 1 });

  return (
    <motion.div
      initial={{ opacity: 0, y: 16, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.5, ease: [0.2, 0, 0, 1] }}
      className={cn(
        "relative overflow-hidden rounded-2xl border p-6",
        cfg.headerClass,
      )}
    >
      {/* Ambient glow orb */}
      <div
        aria-hidden
        className="atmosphere-orb pointer-events-none"
        style={{
          top: "-50%",
          right: "-10%",
          width: 300,
          height: 300,
          background: `radial-gradient(circle, hsl(${cfg.glowHue} / 0.18) 0%, transparent 70%)`,
          opacity: 0.9,
          animationDuration: "28s",
        }}
      />
      {/* Inset severity glow */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 rounded-2xl"
        style={{ boxShadow: `inset 0 0 80px -30px hsl(${cfg.glowHue} / 0.18)` }}
      />

      <div className="relative z-10 flex items-start gap-5">
        {/* Icon */}
        <div
          className={cn(
            "flex h-16 w-16 shrink-0 items-center justify-center rounded-2xl ring-1 ring-white/8",
            cfg.iconBg,
          )}
          style={{ boxShadow: `0 4px 24px -8px hsl(${cfg.glowHue} / 0.3)` }}
        >
          <Icon className={cn("h-8 w-8", cfg.iconClass)} />
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <p className={cn("text-2xl font-bold tracking-tight", cfg.labelClass)}>
              {cfg.label}
            </p>
            {risk.near_boundary && (
              <span className="text-xs bg-canvas/60 text-foreground-muted rounded-full px-2 py-0.5 border border-border-subtle">
                Sınır bölgesi
              </span>
            )}
          </div>

          <p className={cn("mt-1 text-sm leading-relaxed", cfg.sublabelClass)}>
            {cfg.sublabel}
          </p>

          {/* Score display */}
          <div className="mt-4 space-y-2">
            <div className="flex items-baseline gap-2">
              <span className={cn("text-5xl font-bold tabular-nums leading-none tracking-tight", cfg.scoreClass)}>
                {animatedScore.toFixed(1)}
              </span>
              <span className={cn("text-base font-medium", cfg.scoreClass, "opacity-60")}>
                / 100
              </span>
              {!hideBreakdown && (
                <div className="ml-auto flex items-baseline gap-1 text-xs text-foreground-muted">
                  <span>Görüntü:</span>
                  <span className="font-semibold tabular-nums">{(risk.imaging_score * 100).toFixed(1)}</span>
                  {risk.clinical_modifier !== 0 && (
                    <>
                      <span>· Klinik:</span>
                      <span className={cn("font-semibold tabular-nums", risk.clinical_modifier > 0 ? cfg.scoreClass : "text-foreground-muted")}>
                        {risk.clinical_modifier > 0 ? "+" : ""}{(risk.clinical_modifier * 100).toFixed(1)}
                      </span>
                    </>
                  )}
                </div>
              )}
            </div>
            <ScoreBar score={risk.final_score} barClass={cfg.barClass} thresholds={risk.tier_thresholds} />
          </div>

          {/* Differential classes */}
          {risk.differential_classes.length > 0 && (
            <div className="mt-4 space-y-1.5">
              <p className="text-2xs font-semibold uppercase tracking-wider text-current/50">
                Ayırıcı Tanı
              </p>
              <div className="flex flex-wrap gap-1.5">
                {risk.differential_classes.slice(0, 4).map((cls) => (
                  <span
                    key={cls}
                    className={cn(
                      "rounded-full border px-2.5 py-0.5 text-xs font-medium",
                      cfg.badgeClass,
                    )}
                  >
                    {cls}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Immediate action banner */}
      {requiresImmediateAction && (
        <div className="relative z-10 mt-5 flex items-start gap-2 rounded-xl bg-danger-500/20 border border-danger-500/30 px-4 py-3">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-danger-500" />
          <p className="text-xs font-medium text-danger-700 leading-relaxed">
            Acil klinik değerlendirme gereklidir. Bu sistem tanı koymaz; bulgu destekleyici niteliktedir.
          </p>
        </div>
      )}

      {/* Session ID footer */}
      <div className="relative z-10 mt-4 border-t border-current/10 pt-3">
        <p className="text-2xs text-current/40 font-mono">
          Oturum: {sessionId}
        </p>
      </div>
    </motion.div>
  );
}
