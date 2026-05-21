"use client";

import { motion, AnimatePresence } from "framer-motion";
import { Eye, EyeOff, Microscope } from "lucide-react";
import { useState } from "react";

import { cn } from "@/lib/utils";

interface MedicalGradCAMViewerProps {
  originalImageUrl: string;
  gradcamBase64: string | null;
  targetClass?: string | null;
}

const CLASS_LABELS: Record<string, string> = {
  healthy_xray:   "Normal Akciğer Grafisi",
  pneumonia_xray: "Pnömoni Paterni",
  hard_negative:  "İlgisiz İçerik",
  fake_medical:   "Sahte Görüntü",
};

export function MedicalGradCAMViewer({
  originalImageUrl,
  gradcamBase64,
  targetClass,
}: MedicalGradCAMViewerProps) {
  const [showOverlay, setShowOverlay] = useState(true);

  const gradcamSrc = gradcamBase64
    ? `data:image/jpeg;base64,${gradcamBase64}`
    : null;

  const targetLabel = targetClass
    ? (CLASS_LABELS[targetClass] ?? targetClass)
    : null;

  const activeSrc = showOverlay && gradcamSrc ? gradcamSrc : originalImageUrl;

  return (
    <div className="space-y-3">
      {/* Header row */}
      <div className="flex items-start justify-between gap-3">
        <p className="text-xs text-foreground-muted leading-relaxed">
          Model dikkat haritası — aktivasyon yoğunluğu yüksek alanlar, sinyal katkısı en güçlü bölgeleri gösterir.
          {targetLabel && (
            <> Hedef sınıf: <span className="font-semibold text-foreground-secondary">{targetLabel}</span>.</>
          )}
        </p>
        {gradcamSrc && (
          <button
            type="button"
            onClick={() => setShowOverlay((v) => !v)}
            className={cn(
              "ml-3 shrink-0 flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-xs font-medium transition-all duration-200",
              showOverlay
                ? "border-brand-200/60 bg-brand-50/60 text-brand-600 hover:bg-brand-50"
                : "border-border bg-canvas text-foreground-secondary hover:border-brand-200/50 hover:text-foreground",
            )}
          >
            {showOverlay ? (
              <><EyeOff className="h-3 w-3" /> Orijinal</>
            ) : (
              <><Eye className="h-3 w-3" /> Isı Haritası</>
            )}
          </button>
        )}
      </div>

      {/* Cinematic image frame */}
      <div
        className={cn(
          "relative overflow-hidden rounded-xl border bg-canvas",
          gradcamSrc && showOverlay
            ? "border-brand-200/40 shadow-[0_0_40px_-12px_hsl(221_83%_53%/0.20)]"
            : "border-border/60",
        )}
      >
        {/* Corner scan brackets */}
        <div className="pointer-events-none absolute inset-3 z-10 rounded-lg">
          <div className="absolute left-0 top-0 h-4 w-4 border-l border-t border-brand-400/30" />
          <div className="absolute right-0 top-0 h-4 w-4 border-r border-t border-brand-400/30" />
          <div className="absolute bottom-0 left-0 h-4 w-4 border-b border-l border-brand-400/30" />
          <div className="absolute bottom-0 right-0 h-4 w-4 border-b border-r border-brand-400/30" />
        </div>

        {/* Image with reveal animation */}
        <AnimatePresence mode="wait">
          <motion.img
            key={activeSrc}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.4, ease: [0.2, 0, 0, 1] }}
            src={activeSrc}
            alt={showOverlay && gradcamSrc ? "GradCAM ısı haritası" : "Orijinal görüntü"}
            className="max-h-72 w-full object-contain"
          />
        </AnimatePresence>

        {/* Mode badge */}
        {gradcamSrc && (
          <div className="absolute bottom-2.5 left-2.5 flex items-center gap-1.5 rounded-full bg-black/55 px-2.5 py-1 backdrop-blur-sm">
            <Microscope className="h-3 w-3 text-brand-400" />
            <span className="text-2xs font-medium text-white/90">
              {showOverlay ? "Grad-CAM" : "Orijinal"}
            </span>
          </div>
        )}

        {/* No gradcam notice */}
        {!gradcamSrc && (
          <div className="absolute inset-0 flex items-end justify-center pb-4">
            <span className="rounded-full bg-black/40 px-3 py-1 text-xs text-white/70 backdrop-blur-sm">
              Isı haritası mevcut değil
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
