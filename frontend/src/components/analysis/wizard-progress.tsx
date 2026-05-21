import { Check } from "lucide-react";

import { cn } from "@/lib/utils";
import { WIZARD_STEPS, type WizardStep } from "@/stores/fusion-store";

interface WizardProgressProps {
  currentStep: WizardStep;
  onStepClick?: (step: WizardStep) => void;
}

export function WizardProgress({ currentStep, onStepClick }: WizardProgressProps) {
  return (
    <nav aria-label="Analiz adımları" className="w-full">
      <ol className="flex items-center">
        {WIZARD_STEPS.map((label, idx) => {
          const step = idx as WizardStep;
          const done = idx < currentStep;
          const active = idx === currentStep;
          const clickable = onStepClick && idx < currentStep;

          return (
            <li key={label} className="flex flex-1 items-center">
              {/* Circle */}
              <button
                type="button"
                disabled={!clickable}
                onClick={() => clickable && onStepClick(step)}
                className={cn(
                  "relative flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-xs font-bold",
                  "transition-all duration-300 ease-swift-out",
                  done && "cursor-pointer bg-gradient-to-br from-brand-500 to-brand-700 text-white shadow-[0_0_18px_-2px_hsl(221_90%_60%/0.7)] hover:from-brand-400 hover:to-brand-600 active:scale-95",
                  active && "bg-gradient-to-br from-brand-500 to-brand-700 text-white ring-4 ring-brand-400/30 shadow-[0_0_24px_-2px_hsl(221_90%_60%/0.85)] scale-110",
                  !done && !active && "border border-white/15 bg-white/[0.06] backdrop-blur-md text-white/55",
                )}
                aria-current={active ? "step" : undefined}
              >
                {done ? (
                  <Check className="h-4 w-4 animate-scale-in" />
                ) : (
                  <span className={cn(active && "animate-scale-in")}>{idx + 1}</span>
                )}
                {active && (
                  <span
                    aria-hidden
                    className="absolute -inset-1 rounded-full bg-brand-500/20 animate-pulse-slow"
                  />
                )}
              </button>

              {/* Label — hidden on small screens */}
              <span
                className={cn(
                  "ml-2 hidden text-xs font-medium sm:block transition-colors duration-300",
                  active ? "text-white" : done ? "text-white/75" : "text-white/45",
                )}
              >
                {label}
              </span>

              {/* Connector */}
              {idx < WIZARD_STEPS.length - 1 && (
                <div className="mx-2 sm:mx-3 flex-1">
                  <div className="h-0.5 w-full overflow-hidden rounded-full bg-white/15">
                    <div
                      className={cn(
                        "h-full rounded-full bg-brand-400 shadow-[0_0_8px_0_hsl(221_90%_60%/0.6)]",
                        "transition-all duration-500 ease-swift-out",
                      )}
                      style={{ width: done ? "100%" : "0%" }}
                    />
                  </div>
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
