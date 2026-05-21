import { cn } from "@/lib/utils";

interface LoadingDotsProps {
  className?: string;
  /** Tailwind color utility for the dots, e.g. "bg-brand-500" */
  color?: string;
  /** Tailwind size utility for each dot, e.g. "h-1.5 w-1.5" */
  size?: string;
  label?: string;
}

/**
 * Three softly bouncing dots — used for AI thinking/loading states.
 */
export function LoadingDots({
  className,
  color = "bg-brand-500",
  size = "h-1.5 w-1.5",
  label,
}: LoadingDotsProps) {
  return (
    <span
      className={cn("inline-flex items-center gap-1", className)}
      role="status"
      aria-label={label ?? "Yükleniyor"}
    >
      <span className={cn("rounded-full animate-blink-dot", color, size)} />
      <span
        className={cn("rounded-full animate-blink-dot animate-delay-150", color, size)}
      />
      <span
        className={cn("rounded-full animate-blink-dot animate-delay-300", color, size)}
      />
      {label ? <span className="ml-1.5 text-xs text-foreground-muted">{label}</span> : null}
    </span>
  );
}
