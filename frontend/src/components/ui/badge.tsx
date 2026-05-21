import { cn } from "@/lib/utils";

type BadgeVariant = "neutral" | "primary" | "success" | "warning" | "danger" | "outline";

const variantClass: Record<BadgeVariant, string> = {
  neutral: "bg-canvas text-foreground-secondary border border-border",
  primary: "bg-brand-50 text-brand-700 border border-brand-100",
  success: "bg-success-50 text-success-500 border border-success-100",
  warning: "bg-warning-50 text-warning-500 border border-warning-100",
  danger: "bg-danger-50 text-danger-500 border border-danger-100",
  outline: "bg-transparent text-foreground-secondary border border-border",
};

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
}

export function Badge({ variant = "neutral", className, ...rest }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium",
        variantClass[variant],
        className,
      )}
      {...rest}
    />
  );
}
