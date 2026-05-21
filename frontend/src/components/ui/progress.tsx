import { cn } from "@/lib/utils";

export interface ProgressProps {
  /** 0..100. If null/undefined → indeterminate shimmer */
  value?: number | null;
  label?: string;
  className?: string;
  indeterminate?: boolean;
}

export function Progress({ value, label, className, indeterminate }: ProgressProps) {
  const determinate = !indeterminate && typeof value === "number";
  const clamped = determinate ? Math.max(0, Math.min(100, value!)) : 0;

  return (
    <div className={cn("space-y-1.5", className)}>
      <div
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={determinate ? Math.round(clamped) : undefined}
        className="h-2 w-full overflow-hidden rounded-full bg-canvas border border-border-subtle"
      >
        {determinate ? (
          <div
            className="h-full rounded-full bg-brand-600 transition-[width] duration-500 ease-swift-out"
            style={{ width: `${clamped}%` }}
          />
        ) : (
          <div className="h-full w-full progress-shimmer rounded-full" />
        )}
      </div>
      {label ? (
        <div className="flex justify-between text-xs text-foreground-muted">
          <span>{label}</span>
          {determinate ? (
            <span className="tabular-nums font-medium">{Math.round(clamped)}%</span>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
