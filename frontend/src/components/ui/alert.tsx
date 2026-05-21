import { AlertCircle, AlertTriangle, CheckCircle2, Info } from "lucide-react";

import { cn } from "@/lib/utils";

type AlertVariant = "info" | "success" | "warning" | "danger";

const variantConfig: Record<
  AlertVariant,
  { classes: string; icon: React.ElementType }
> = {
  info: {
    classes: "border-brand-200 bg-brand-50 text-brand-800",
    icon: Info,
  },
  success: {
    classes: "border-success-100 bg-success-50 text-success-600",
    icon: CheckCircle2,
  },
  warning: {
    classes: "border-warning-100 bg-warning-50 text-warning-600",
    icon: AlertTriangle,
  },
  danger: {
    classes: "border-danger-100 bg-danger-50 text-danger-600",
    icon: AlertCircle,
  },
};

export interface AlertProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: AlertVariant;
  title?: string;
}

export function Alert({
  variant = "info",
  title,
  className,
  children,
  ...rest
}: AlertProps) {
  const { classes, icon: Icon } = variantConfig[variant];

  return (
    <div
      role="alert"
      className={cn(
        "flex gap-3 rounded-xl border p-4",
        classes,
        className,
      )}
      {...rest}
    >
      <Icon className="mt-0.5 h-4 w-4 shrink-0" />
      <div className="min-w-0 flex-1 text-sm">
        {title ? <p className="mb-0.5 font-semibold">{title}</p> : null}
        <div className="leading-relaxed">{children}</div>
      </div>
    </div>
  );
}
