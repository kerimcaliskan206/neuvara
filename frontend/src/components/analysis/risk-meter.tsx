"use client";

import { useCountUp } from "@/hooks/use-count-up";
import { cn } from "@/lib/utils";

type RiskLevel = "high" | "medium" | "low";

interface RiskMeterProps {
  score: number;        // 0..1
  level: RiskLevel;
  confidence: "high" | "medium" | "low";
  className?: string;
}

const levelConfig: Record<RiskLevel, { label: string; color: string; ringColor: string; bgColor: string }> = {
  high: {
    label: "Yüksek Risk",
    color: "text-danger-600",
    ringColor: "stroke-danger-500",
    bgColor: "bg-danger-50",
  },
  medium: {
    label: "Orta Risk",
    color: "text-warning-600",
    ringColor: "stroke-warning-500",
    bgColor: "bg-warning-50",
  },
  low: {
    label: "Düşük Risk",
    color: "text-success-500",
    ringColor: "stroke-success-500",
    bgColor: "bg-success-50",
  },
};

const confidenceLabel: Record<string, string> = {
  high: "Yüksek güven",
  medium: "Orta güven",
  low: "Düşük güven",
};

export function RiskMeter({ score, level, confidence, className }: RiskMeterProps) {
  const { label, color, ringColor, bgColor } = levelConfig[level] ?? levelConfig.low;

  // Animated value (0 → score) used for both the arc and the numeric counter
  const animatedScore = useCountUp(score, { duration: 900, decimals: 4 });
  const displayPct = useCountUp(score * 100, { duration: 900, decimals: 0 });

  // SVG arc parameters
  const size = 180;
  const center = size / 2;
  const radius = 70;
  const strokeWidth = 11;
  const circumference = Math.PI * radius; // half circle
  const filled = circumference * Math.max(0, Math.min(1, animatedScore));
  const gap = circumference - filled;

  return (
    <div className={cn("flex flex-col items-center gap-3", className)}>
      {/* SVG half-ring meter */}
      <div className="relative">
        <svg
          width={size}
          height={size / 2 + strokeWidth}
          viewBox={`0 0 ${size} ${size / 2 + strokeWidth}`}
          aria-hidden
        >
          <defs>
            {/* Subtle inner shadow on the track */}
            <linearGradient id="track-grad" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="hsl(214 20% 90%)" />
              <stop offset="100%" stopColor="hsl(214 20% 94%)" />
            </linearGradient>
          </defs>

          {/* Track */}
          <path
            d={`M ${strokeWidth / 2} ${center} A ${radius} ${radius} 0 0 1 ${size - strokeWidth / 2} ${center}`}
            fill="none"
            stroke="url(#track-grad)"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />
          {/* Filled arc */}
          <path
            d={`M ${strokeWidth / 2} ${center} A ${radius} ${radius} 0 0 1 ${size - strokeWidth / 2} ${center}`}
            fill="none"
            className={ringColor}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={`${filled} ${gap}`}
          />
        </svg>

        {/* Score in center */}
        <div className="absolute inset-x-0 bottom-0 flex flex-col items-center pb-1">
          <span className={cn("text-4xl font-bold tabular-nums leading-none", color)}>
            {displayPct}
            <span className="text-lg font-normal align-top">%</span>
          </span>
        </div>
      </div>

      {/* Risk label pill */}
      <span
        className={cn(
          "rounded-full px-4 py-1.5 text-sm font-bold animate-scale-in",
          bgColor,
          color,
        )}
      >
        {label}
      </span>

      {/* Confidence */}
      <p className="text-xs text-foreground-muted">
        {confidenceLabel[confidence] ?? "Güven bilinmiyor"}
      </p>
    </div>
  );
}
