import type { LucideIcon } from "lucide-react";

import { cn } from "@/lib/utils";

interface EmptyStateProps {
  icon?: LucideIcon;
  title: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
  tone?: "neutral" | "brand" | "muted";
}

const toneStyles = {
  neutral: {
    iconBg: "bg-canvas",
    iconColor: "text-foreground-muted",
  },
  brand: {
    iconBg: "bg-brand-50",
    iconColor: "text-brand-500",
  },
  muted: {
    iconBg: "bg-border-subtle",
    iconColor: "text-foreground-placeholder",
  },
} as const;

export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
  className,
  tone = "brand",
}: EmptyStateProps) {
  const t = toneStyles[tone];
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center rounded-2xl border border-dashed border-border bg-canvas/50",
        "px-6 py-10 text-center animate-fade-up",
        className,
      )}
    >
      {Icon ? (
        <div
          className={cn(
            "mb-3 flex h-12 w-12 items-center justify-center rounded-2xl",
            t.iconBg,
          )}
        >
          <Icon className={cn("h-6 w-6", t.iconColor)} />
        </div>
      ) : null}
      <p className="text-sm font-semibold text-foreground">{title}</p>
      {description ? (
        <p className="mt-1.5 max-w-xs text-xs leading-relaxed text-foreground-muted">
          {description}
        </p>
      ) : null}
      {action ? <div className="mt-4">{action}</div> : null}
    </div>
  );
}
