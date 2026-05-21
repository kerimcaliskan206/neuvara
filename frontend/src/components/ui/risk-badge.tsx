import { AlertTriangle, CheckCircle2, ShieldAlert } from "lucide-react";

import { cn } from "@/lib/utils";

type RiskLevel = "high" | "medium" | "low" | "unknown";

const config: Record<
  RiskLevel,
  { label: string; icon: React.ElementType; classes: string }
> = {
  high: {
    label: "Yüksek Risk",
    icon: ShieldAlert,
    classes: "bg-danger-50 text-danger-600 border-danger-100",
  },
  medium: {
    label: "Orta Risk",
    icon: AlertTriangle,
    classes: "bg-warning-50 text-warning-600 border-warning-100",
  },
  low: {
    label: "Düşük Risk",
    icon: CheckCircle2,
    classes: "bg-success-50 text-success-500 border-success-100",
  },
  unknown: {
    label: "Belirsiz",
    icon: AlertTriangle,
    classes: "bg-muted text-foreground-muted border-border",
  },
};

interface RiskBadgeProps {
  level: RiskLevel;
  score?: number;
  className?: string;
}

export function RiskBadge({ level, score, className }: RiskBadgeProps) {
  const { label, icon: Icon, classes } = config[level] ?? config.unknown;

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold",
        classes,
        className,
      )}
    >
      <Icon className="h-3.5 w-3.5" />
      {label}
      {score !== undefined ? (
        <span className="opacity-70">({(score * 100).toFixed(0)}%)</span>
      ) : null}
    </span>
  );
}
