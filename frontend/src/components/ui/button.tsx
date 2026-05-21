import { forwardRef } from "react";

import { cn } from "@/lib/utils";

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger" | "outline";
type ButtonSize = "sm" | "md" | "lg" | "icon";

const variantClass: Record<ButtonVariant, string> = {
  primary: [
    "bg-brand-600 text-white shadow-sm",
    "hover:bg-brand-700 active:bg-brand-800",
    "focus-visible:ring-brand-500",
  ].join(" "),
  secondary: [
    "bg-canvas text-foreground border border-border shadow-sm",
    "hover:bg-border-subtle hover:border-border-strong",
    "focus-visible:ring-brand-500",
  ].join(" "),
  outline: [
    "bg-transparent text-brand-700 border border-brand-200",
    "hover:bg-brand-50 hover:border-brand-300",
    "focus-visible:ring-brand-500",
  ].join(" "),
  ghost: [
    "bg-transparent text-foreground-secondary",
    "hover:bg-canvas hover:text-foreground",
    "focus-visible:ring-brand-500",
  ].join(" "),
  danger: [
    "bg-danger-500 text-white shadow-sm",
    "hover:bg-danger-600",
    "focus-visible:ring-danger-500",
  ].join(" "),
};

const sizeClass: Record<ButtonSize, string> = {
  sm: "h-8 px-3 text-xs gap-1.5",
  md: "h-9 px-4 text-sm gap-2",
  lg: "h-11 px-5 text-sm gap-2",
  icon: "h-9 w-9 p-0",
};

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  isLoading?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "primary", size = "md", isLoading, className, children, disabled, ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      disabled={disabled || isLoading}
      className={cn(
        "inline-flex items-center justify-center rounded-lg font-medium",
        "transition-[background-color,color,box-shadow,border-color,transform] duration-200 ease-swift-out",
        "active:scale-[0.97]",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2",
        "disabled:cursor-not-allowed disabled:opacity-55 disabled:active:scale-100",
        variantClass[variant],
        sizeClass[size],
        className,
      )}
      {...rest}
    >
      {isLoading ? (
        <span
          aria-hidden
          className="h-4 w-4 animate-spin rounded-full border-2 border-current border-r-transparent"
        />
      ) : null}
      {children}
    </button>
  );
});
