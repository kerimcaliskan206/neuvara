import { cn } from "@/lib/utils";

/**
 * Skeleton placeholder with shimmer surface.
 * Falls back to a static muted bar under `prefers-reduced-motion`.
 */
export function Skeleton({
  className,
  ...rest
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "shimmer-surface rounded-md",
        className,
      )}
      aria-hidden
      {...rest}
    />
  );
}
