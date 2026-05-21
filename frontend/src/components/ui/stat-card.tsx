import { type LucideIcon } from "lucide-react";

import { cn } from "@/lib/utils";

type Trend = "up" | "down" | "neutral";

interface StatCardProps {
  label: string;
  value: string | number;
  description?: string;
  icon: LucideIcon;
  iconColor?: string;
  iconBg?: string;
  trend?: Trend;
  trendLabel?: string;
  className?: string;
}

export function StatCard({
  label,
  value,
  description,
  icon: Icon,
  iconColor = "text-brand-600",
  iconBg = "bg-brand-50",
  trend,
  trendLabel,
  className,
}: StatCardProps) {
  return (
    <div
      className={cn(
        "group relative overflow-hidden rounded-xl border border-border bg-surface/90 backdrop-blur-md p-5 shadow-card lift-on-hover",
        className,
      )}
    >
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/80 to-transparent"
      />
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <p className="truncate text-xs font-medium uppercase tracking-wider text-foreground-muted">
            {label}
          </p>
          <p className="mt-2 text-2xl font-bold tracking-tight text-foreground">
            {value}
          </p>
          {description ? (
            <p className="mt-1 truncate text-xs text-foreground-muted">{description}</p>
          ) : null}
          {trend && trendLabel ? (
            <p
              className={cn(
                "mt-1.5 text-xs font-medium",
                trend === "up" && "text-success-500",
                trend === "down" && "text-danger-500",
                trend === "neutral" && "text-foreground-muted",
              )}
            >
              {trendLabel}
            </p>
          ) : null}
        </div>
        <div
          className={cn(
            "flex h-10 w-10 shrink-0 items-center justify-center rounded-xl",
            "transition-transform duration-300 ease-swift-out group-hover:scale-110",
            iconBg,
          )}
        >
          <Icon className={cn("h-5 w-5", iconColor)} />
        </div>
      </div>
    </div>
  );
}
