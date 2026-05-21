"use client";

import { Eye, Layers, Thermometer } from "lucide-react";
import { useState } from "react";

import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

type Tab = "original" | "heatmap" | "overlay";

const tabs: { id: Tab; label: string; icon: React.ElementType }[] = [
  { id: "original", label: "Orijinal", icon: Eye },
  { id: "heatmap", label: "Isı Haritası", icon: Thermometer },
  { id: "overlay", label: "Katman", icon: Layers },
];

export interface GradCamViewerProps {
  originalImageUrl: string;
  gradcamBase64: string | null;
  isLoading?: boolean;
  className?: string;
}

export function GradCamViewer({
  originalImageUrl,
  gradcamBase64,
  isLoading,
  className,
}: GradCamViewerProps) {
  const [activeTab, setActiveTab] = useState<Tab>(gradcamBase64 ? "overlay" : "original");
  const overlaySrc = gradcamBase64 ? `data:image/jpeg;base64,${gradcamBase64}` : null;

  const tabIndex = tabs.findIndex((t) => t.id === activeTab);

  return (
    <div className={cn("space-y-3", className)}>
      {/* Tab bar with sliding indicator */}
      <div className="relative flex gap-1 rounded-xl border border-border bg-canvas p-1">
        {/* Sliding background pill */}
        <div
          aria-hidden
          className="absolute top-1 bottom-1 rounded-lg bg-surface shadow-sm transition-all duration-300 ease-swift-out"
          style={{
            left: `calc(${tabIndex} * (100% / ${tabs.length}) + 0.25rem)`,
            width: `calc(100% / ${tabs.length} - 0.5rem)`,
          }}
        />
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const disabled = tab.id !== "original" && !overlaySrc && !isLoading;
          const active = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              type="button"
              disabled={disabled}
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                "relative z-10 flex flex-1 items-center justify-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium",
                "transition-colors duration-200",
                active ? "text-brand-700" : "text-foreground-muted hover:text-foreground",
                disabled && "cursor-not-allowed opacity-40 hover:text-foreground-muted",
              )}
            >
              <Icon className="h-3.5 w-3.5" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Viewer */}
      <div className="relative overflow-hidden rounded-xl border border-border bg-canvas">
        {isLoading ? (
          <Skeleton className="aspect-video w-full" />
        ) : (
          <div className="relative inline-block w-full">
            {/* Original always present as base */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={originalImageUrl}
              alt="Yüklenen görüntü"
              className={cn(
                "block h-auto w-full object-contain transition-opacity duration-300",
                activeTab === "heatmap" && "opacity-0",
              )}
            />

            {/* Heatmap only */}
            {overlaySrc && activeTab === "heatmap" ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={overlaySrc}
                alt="Grad-CAM ısı haritası"
                className="block h-auto w-full object-contain animate-fade-in"
              />
            ) : null}

            {/* Overlay (composite) */}
            {overlaySrc && activeTab === "overlay" ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={overlaySrc}
                alt="Grad-CAM katman görünümü"
                className="absolute inset-0 h-full w-full object-contain mix-blend-multiply opacity-75 animate-fade-in"
              />
            ) : null}
          </div>
        )}
      </div>

      {/* Caption */}
      {!isLoading && !overlaySrc ? (
        <p className="text-center text-xs text-foreground-muted">
          Bu tahmin için Grad-CAM çıktısı bulunmuyor. Tahmin sırasında Grad-CAM&apos;i etkinleştirin.
        </p>
      ) : null}

      {!isLoading && overlaySrc ? (
        <p className="text-center text-xs text-foreground-muted">
          Tahmin edilen sınıf için model dikkat görselleştirmesi
        </p>
      ) : null}
    </div>
  );
}
