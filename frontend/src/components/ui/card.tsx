import { cn } from "@/lib/utils";

type CardVariant = "default" | "raised" | "flat" | "accent";

const variantClass: Record<CardVariant, string> = {
  default: "border border-border bg-surface/90 shadow-card",
  raised: "border border-border bg-surface/95 shadow-card-raised",
  flat: "border border-border-subtle bg-surface/85",
  accent: "border border-brand-200 bg-brand-50/90",
};

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: CardVariant;
  hoverable?: boolean;
}

export function Card({
  className,
  variant = "default",
  hoverable = false,
  ...rest
}: CardProps) {
  return (
    <div
      className={cn(
        "rounded-xl backdrop-blur-md",
        variantClass[variant],
        hoverable && "lift-on-hover cursor-pointer",
        !hoverable && "transition-shadow duration-200",
        className,
      )}
      {...rest}
    />
  );
}

export function CardHeader({
  className,
  ...rest
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("flex flex-col gap-1 px-6 pb-3 pt-5", className)}
      {...rest}
    />
  );
}

export function CardTitle({
  className,
  ...rest
}: React.HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3
      className={cn(
        "text-base font-semibold leading-tight tracking-tight text-foreground",
        className,
      )}
      {...rest}
    />
  );
}

export function CardDescription({
  className,
  ...rest
}: React.HTMLAttributes<HTMLParagraphElement>) {
  return (
    <p
      className={cn("text-sm leading-relaxed text-foreground-secondary", className)}
      {...rest}
    />
  );
}

export function CardContent({
  className,
  ...rest
}: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("px-6 pb-5 pt-2", className)} {...rest} />;
}

export function CardFooter({
  className,
  ...rest
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "flex items-center gap-3 border-t border-border-subtle px-6 py-4",
        className,
      )}
      {...rest}
    />
  );
}
