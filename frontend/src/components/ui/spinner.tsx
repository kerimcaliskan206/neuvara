import { cn } from "@/lib/utils";

export interface SpinnerProps {
  size?: "sm" | "md" | "lg";
  label?: string;
  className?: string;
}

const sizes = {
  sm: "h-4 w-4 border-2",
  md: "h-6 w-6 border-2",
  lg: "h-10 w-10 border-[3px]",
};

export function Spinner({ size = "md", label, className }: SpinnerProps) {
  return (
    <div
      role="status"
      aria-live="polite"
      className={cn("inline-flex items-center gap-2 text-muted-foreground", className)}
    >
      <span
        aria-hidden
        className={cn(
          "inline-block animate-spin rounded-full border-primary border-r-transparent",
          sizes[size],
        )}
      />
      {label ? <span className="text-sm">{label}</span> : null}
    </div>
  );
}
