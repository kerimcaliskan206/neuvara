import { forwardRef } from "react";

import { cn } from "@/lib/utils";

export const Input = forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  function Input({ className, ...rest }, ref) {
    return (
      <input
        ref={ref}
        className={cn(
          "h-10 w-full rounded-lg border border-border bg-surface px-3.5 text-sm text-foreground",
          "placeholder:text-foreground-placeholder",
          "transition-colors duration-150",
          "hover:border-border-strong",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 focus-visible:ring-offset-1 focus-visible:border-brand-400",
          "disabled:cursor-not-allowed disabled:bg-canvas disabled:opacity-60",
          className,
        )}
        {...rest}
      />
    );
  },
);
