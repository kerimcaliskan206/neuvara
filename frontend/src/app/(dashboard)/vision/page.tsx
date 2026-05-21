"use client";

import { ImageIcon, ImageOff, Sparkles, Zap } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { AiExplanationPanel } from "@/components/analysis/ai-explanation-panel";
import { GradCamViewer } from "@/components/analysis/gradcam-viewer";
import { Alert } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/empty-state";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import {
  ImageUploader,
  type SelectedFile,
} from "@/components/vision/image-uploader";
import { useExplainVision } from "@/hooks/use-ai-explanation";
import { useVisionUpload } from "@/hooks/use-vision-upload";
import { config } from "@/lib/config";
import { cn } from "@/lib/utils";

export default function VisionPage() {
  const [selected, setSelected] = useState<SelectedFile | null>(null);
  const [enableGradcam, setEnableGradcam] = useState(config.features.gradcam);
  const previewUrlRef = useRef<string | null>(null);

  const upload = useVisionUpload();
  const explain = useExplainVision();

  // Cleanup preview URL on unmount
  useEffect(() => {
    return () => {
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    };
  }, []);

  // Reset explanation whenever a new prediction comes back
  useEffect(() => {
    explain.reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [upload.data]);

  const prediction = upload.data;

  function handleSelect(file: SelectedFile) {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = file.previewUrl;
    setSelected(file);
    upload.reset();
    explain.reset();
  }

  function handleClear() {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = null;
    setSelected(null);
    upload.reset();
    explain.reset();
  }

  function handlePredict() {
    if (!selected) return;
    upload.mutate({ file: selected.file, gradcam: enableGradcam });
  }

  return (
    <div className="space-y-6 pb-12">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-white">
          Görüntü Analizi
        </h1>
        <p className="mt-1 text-sm text-white/65">
          CNN tabanlı sınıflandırıcı görüntüyü değerlendirir ve isteğe bağlı olarak
          Grad-CAM ısı haritası üretir.
        </p>
      </div>

      {/* Supporting evidence callout — frosted glass */}
      <div className="flex items-start gap-3 rounded-xl border border-warning-200/70 bg-warning-50/90 backdrop-blur-md p-4">
        <Zap className="mt-0.5 h-4 w-4 shrink-0 text-warning-500" />
        <p className="text-xs text-warning-700 leading-relaxed">
          <strong>Bilgi:</strong> Görüntü analizi tek başına teşhis aracı değildir.
          Çok modlu füzyon analizinde destekleyici kanıt olarak kullanılır
          (β=25% ağırlık). Reddedilen veya ilgisiz görüntüler otomatik olarak
          yok sayılır.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* LEFT — Upload */}
        <div className="space-y-4">
          <div className="rounded-2xl glass-card-light p-5 space-y-4">
            <div className="flex items-center gap-2">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-brand-50">
                <ImageIcon className="h-3.5 w-3.5 text-brand-600" />
              </div>
              <p className="text-sm font-semibold text-foreground">
                Görüntü Seçin
              </p>
            </div>

            {!selected ? (
              <ImageUploader onSelect={handleSelect} disabled={upload.isPending} />
            ) : (
              <div className="relative overflow-hidden rounded-xl border border-border">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={selected.previewUrl}
                  alt="Seçilen görüntü"
                  className="max-h-72 w-full object-contain"
                />
                <button
                  type="button"
                  onClick={handleClear}
                  className="absolute right-2 top-2 flex h-8 w-8 items-center justify-center rounded-full bg-surface/95 shadow-sm hover:bg-danger-50"
                  aria-label="Görüntüyü kaldır"
                  disabled={upload.isPending}
                >
                  <ImageOff className="h-4 w-4 text-danger-500" />
                </button>
              </div>
            )}

            {/* Grad-CAM checkbox */}
            {config.features.gradcam && selected && !prediction ? (
              <label className="flex cursor-pointer items-center gap-2.5 text-sm">
                <input
                  type="checkbox"
                  checked={enableGradcam}
                  onChange={(e) => setEnableGradcam(e.target.checked)}
                  className="h-4 w-4 rounded border-border text-brand-600"
                  disabled={upload.isPending}
                />
                <span className="text-foreground-secondary">
                  Grad-CAM ısı haritası iste
                </span>
              </label>
            ) : null}

            {/* Action buttons */}
            <div className="flex items-center gap-2">
              <Button
                type="button"
                disabled={!selected || upload.isPending}
                isLoading={upload.isPending}
                onClick={handlePredict}
              >
                <Sparkles className="h-3.5 w-3.5" />
                Tahmin Et
              </Button>
              {upload.isPending && (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => upload.cancel()}
                >
                  İptal
                </Button>
              )}
              {upload.isError && !upload.isPending && selected && (
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  onClick={handlePredict}
                >
                  Tekrar dene
                </Button>
              )}
            </div>

            {/* Progress */}
            {upload.isPending && (
              <div className="space-y-2">
                {upload.progress !== null ? (
                  <Progress value={upload.progress} label="Yükleniyor" />
                ) : (
                  <Progress value={100} label="Analiz ediliyor…" />
                )}
              </div>
            )}

            {/* Error */}
            {upload.isError && (
              <Alert variant="danger" title="İstek başarısız">
                {upload.error.message}
              </Alert>
            )}
          </div>
        </div>

        {/* RIGHT — Results */}
        <div className="space-y-4">
          {/* Loading skeleton */}
          {upload.isPending && upload.progress === null && (
            <div className="rounded-2xl glass-card-light p-5 space-y-4">
              <Skeleton className="h-6 w-1/2" />
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-2.5 w-full rounded-full" />
              <Skeleton className="h-2.5 w-full rounded-full" />
              <Skeleton className="h-2.5 w-2/3 rounded-full" />
            </div>
          )}

          {/* Prediction result */}
          {prediction && (
            <div
              className={cn(
                "rounded-2xl border p-5 space-y-4 shadow-card animate-fade-up",
                prediction.accepted
                  ? "border-success-100 bg-success-50"
                  : "border-warning-100 bg-warning-50",
              )}
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p
                    className={cn(
                      "text-base font-bold",
                      prediction.accepted
                        ? "text-success-700"
                        : "text-warning-700",
                    )}
                  >
                    {prediction.accepted
                      ? "✓ Görüntü Kabul Edildi"
                      : "⚠ Görüntü Reddedildi"}
                  </p>
                  <p className="text-xs text-foreground-secondary mt-0.5">
                    Sınıf: <strong>{prediction.predicted_class ?? "—"}</strong>
                  </p>
                  {!prediction.accepted && prediction.rejection_reason && (
                    <p className="text-xs text-warning-600 mt-0.5">
                      {prediction.rejection_reason}
                    </p>
                  )}
                </div>
                {prediction.confidence !== null && (
                  <span
                    className={cn(
                      "rounded-full px-3 py-1 text-xs font-bold",
                      prediction.accepted
                        ? "bg-success-100 text-success-700"
                        : "bg-warning-100 text-warning-700",
                    )}
                  >
                    {(prediction.confidence * 100).toFixed(1)}%
                  </span>
                )}
              </div>

              {/* Class probabilities */}
              {prediction.probabilities && Object.keys(prediction.probabilities).length > 0 && (
                <div className="space-y-2 pt-2 border-t border-white/50">
                  <p className="text-xs font-semibold text-foreground-secondary">
                    Sınıf Olasılıkları
                  </p>
                  {Object.entries(prediction.probabilities)
                    .sort(([, a], [, b]) => b - a)
                    .map(([name, prob]) => (
                      <div key={name} className="space-y-1">
                        <div className="flex justify-between text-xs text-foreground-secondary">
                          <span>{name}</span>
                          <span className="font-semibold tabular-nums">
                            {(prob * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/70">
                          <div
                            className={cn(
                              "h-full rounded-full",
                              name === prediction.predicted_class
                                ? prediction.accepted
                                  ? "bg-success-500"
                                  : "bg-warning-500"
                                : "bg-foreground-muted/40",
                            )}
                            style={{ width: `${prob * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                </div>
              )}

              <div className="flex flex-wrap gap-3 text-2xs text-foreground-muted pt-1 border-t border-white/50">
                <span>
                  {prediction.model_name} · v{prediction.model_version}
                </span>
                <span>·</span>
                <span>{prediction.inference_duration_ms.toFixed(0)} ms</span>
                {prediction.threshold !== null && (
                  <>
                    <span>·</span>
                    <span>
                      Eşik: {(prediction.threshold * 100).toFixed(0)}%
                    </span>
                  </>
                )}
              </div>
            </div>
          )}

          {/* Grad-CAM viewer — shown for any predicted class */}
          {selected &&
            prediction &&
            (prediction.gradcam_base64 || enableGradcam) && (
              <div className="rounded-2xl glass-card-light p-5 space-y-3 animate-fade-up animate-delay-150">
                <p className="text-xs font-semibold uppercase tracking-wider text-foreground-muted">
                  Model Dikkat Görselleştirmesi
                </p>
                <GradCamViewer
                  originalImageUrl={selected.previewUrl}
                  gradcamBase64={prediction.gradcam_base64}
                />
              </div>
            )}

          {/* AI explanation */}
          {prediction && (
            <div className="rounded-2xl glass-card-light p-5 animate-fade-up animate-delay-300">
              <AiExplanationPanel
                isLoading={explain.isPending}
                isError={explain.isError}
                error={explain.error}
                data={explain.data}
                disabled={!prediction}
                onRequest={() => explain.mutate(prediction)}
              />
            </div>
          )}

          {/* Empty state */}
          {!prediction && !upload.isPending && (
            <EmptyState
              icon={ImageIcon}
              title="Henüz tahmin yok"
              description="Bir görüntü seçin ve “Tahmin Et” düğmesine basın. Sonuçlar bu alanda görüntülenecek."
              tone="brand"
            />
          )}
        </div>
      </div>
    </div>
  );
}
