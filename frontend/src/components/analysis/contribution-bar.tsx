"use client";

import { useCountUp } from "@/hooks/use-count-up";
import { cn } from "@/lib/utils";

interface ContributionBarProps {
  mlContribution: number;       // 0..1 weighted contribution
  visionContribution: number;   // 0..1 weighted contribution
  mlWeight: number;             // α applied
  visionWeight: number;         // β applied
  visionStatus: string;
  className?: string;
}

interface SignalRowProps {
  label: string;
  sublabel: string;
  value: number;      // 0..1
  barColor: string;
  valueColor: string;
  delay?: number;
}

function SignalRow({ label, sublabel, value, barColor, valueColor, delay = 0 }: SignalRowProps) {
  const animated = useCountUp(value, { duration: 800, decimals: 4 });
  const pct = Math.max(0, Math.min(1, animated));
  const animatedPctDisplay = useCountUp(value * 100, { duration: 800, decimals: 1 });

  return (
    <div className="space-y-1.5 animate-fade-up" style={{ animationDelay: `${delay}ms` }}>
      <div className="flex items-end justify-between gap-2">
        <div>
          <p className="text-sm font-semibold text-foreground">{label}</p>
          <p className="text-xs text-foreground-muted">{sublabel}</p>
        </div>
        <span className={cn("text-sm font-bold tabular-nums", valueColor)}>
          {animatedPctDisplay.toFixed(1)}%
        </span>
      </div>
      <div className="h-2.5 w-full overflow-hidden rounded-full bg-canvas border border-border-subtle">
        <div
          className={cn("h-full rounded-full", barColor)}
          style={{
            width: `${pct * 100}%`,
            transition: "width 600ms cubic-bezier(0.2, 0, 0, 1)",
          }}
        />
      </div>
    </div>
  );
}

const visionStatusLabel: Record<string, string> = {
  used: "Görüntü kullanıldı",
  rejected: "Görüntü reddedildi",
  unrelated: "Görüntü ilgisiz",
  unavailable: "Görüntü yok",
  low_confidence: "Güven çok düşük",
};

export function ContributionBar({
  mlContribution,
  visionContribution,
  mlWeight,
  visionWeight,
  visionStatus,
  className,
}: ContributionBarProps) {
  const visionUsed = visionStatus === "used";

  return (
    <div className={cn("space-y-4", className)}>
      <SignalRow
        label="ML Semptom Analizi"
        sublabel={`Birincil sinyal · ağırlık α=${(mlWeight * 100).toFixed(0)}%`}
        value={mlContribution}
        barColor="bg-brand-500"
        valueColor="text-brand-700"
        delay={0}
      />
      <SignalRow
        label="Görüntü Analizi"
        sublabel={`${visionStatusLabel[visionStatus] ?? visionStatus} · ağırlık β=${(visionWeight * 100).toFixed(0)}%`}
        value={visionContribution}
        barColor={visionUsed ? "bg-success-500" : "bg-border"}
        valueColor={visionUsed ? "text-success-600" : "text-foreground-muted"}
        delay={120}
      />
    </div>
  );
}
