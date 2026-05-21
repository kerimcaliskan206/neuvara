"use client";

import { useBackendHealth } from "@/hooks/use-backend-health";
import { cn } from "@/lib/utils";

export function BackendStatusPill() {
  const { isOnline, isLoading } = useBackendHealth();

  const label = isLoading
    ? "Kontrol ediliyor…"
    : isOnline
      ? "Sunucu çevrim içi"
      : "Sunucu çevrim dışı";

  return (
    <div
      className="inline-flex items-center gap-2 rounded-full border border-border bg-canvas px-3 py-1 text-xs font-medium text-foreground-secondary"
      title={label}
    >
      <span
        aria-hidden
        className={cn(
          "inline-block h-1.5 w-1.5 rounded-full",
          isLoading && "animate-pulse bg-foreground-muted",
          !isLoading && isOnline && "bg-success-500",
          !isLoading && !isOnline && "bg-danger-500",
        )}
      />
      <span>{label}</span>
    </div>
  );
}
