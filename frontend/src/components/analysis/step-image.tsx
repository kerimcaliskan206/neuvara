"use client";

import { ImageOff, Zap } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { Alert } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { ImageUploader, type SelectedFile } from "@/components/vision/image-uploader";
import { useVisionUpload } from "@/hooks/use-vision-upload";
import { config } from "@/lib/config";
import { cn } from "@/lib/utils";
import type { VisionPredictionResponse } from "@/lib/api/types";

interface StepImageProps {
  visionResult: VisionPredictionResponse | null;
  visionPreviewUrl: string | null;
  onVisionResult: (result: VisionPredictionResponse | null, previewUrl: string | null) => void;
  onNext: () => void;
  onBack: () => void;
  onSkip: () => void;
}

export function StepImage({
  visionResult,
  visionPreviewUrl,
  onVisionResult,
  onNext,
  onBack,
  onSkip,
}: StepImageProps) {
  const [selected, setSelected] = useState<SelectedFile | null>(null);
  const [enableGradcam, setEnableGradcam] = useState(config.features.gradcam);
  const previewUrlRef = useRef<string | null>(null);
  const upload = useVisionUpload();

  // Sync selected preview with parent on mount (if result already exists from prev visit)
  useEffect(() => {
    return () => {
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    };
  }, []);

  function handleSelect(file: SelectedFile) {
    // Revoke old preview URL
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = file.previewUrl;
    setSelected(file);
    upload.reset();
    onVisionResult(null, null);
  }

  function handleUpload() {
    if (!selected) return;
    upload.mutate(
      { file: selected.file, gradcam: enableGradcam },
      {
        onSuccess: (data) => {
          onVisionResult(data, selected.previewUrl);
        },
      },
    );
  }

  function handleClear() {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = null;
    setSelected(null);
    upload.reset();
    onVisionResult(null, null);
  }

  const prediction = visionResult ?? upload.data;
  const previewSrc = selected?.previewUrl ?? visionPreviewUrl;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-bold text-foreground">Görüntü Analizi</h2>
        <p className="mt-1 text-sm text-foreground-secondary">
          Görüntü isteğe bağlıdır ve yalnızca destekleyici kanıt olarak kullanılır.
          ML risk analizi her durumda çalışır.
        </p>
      </div>

      {/* Supporting evidence callout */}
      <div className="flex items-start gap-3 rounded-xl border border-warning-100 bg-warning-50 p-4">
        <Zap className="mt-0.5 h-4 w-4 shrink-0 text-warning-500" />
        <p className="text-xs text-warning-700 leading-relaxed">
          <strong>Destekleyici kanıt:</strong> Görüntü analizi ML semptom skorunu
          doğrulayan bir destektir. Füzyon motoru görüntüye β=25% ağırlık verir.
          Kabul edilmeyen veya ilgisiz görüntüler otomatik olarak görmezden gelinir.
        </p>
      </div>

      {!selected && !visionResult ? (
        <ImageUploader onSelect={handleSelect} disabled={upload.isPending} />
      ) : null}

      {/* Image preview */}
      {previewSrc && !prediction ? (
        <div className="relative overflow-hidden rounded-xl border border-border">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={previewSrc}
            alt="Seçilen görüntü önizleme"
            className="max-h-56 w-full object-contain"
          />
          <button
            type="button"
            onClick={handleClear}
            className="absolute right-2 top-2 flex h-7 w-7 items-center justify-center rounded-full bg-surface/90 shadow-sm hover:bg-danger-50"
            aria-label="Görüntüyü kaldır"
          >
            <ImageOff className="h-3.5 w-3.5 text-danger-500" />
          </button>
        </div>
      ) : null}

      {/* Prediction result preview (compact) */}
      {prediction ? (
        <div
          className={cn(
            "flex items-start justify-between gap-4 rounded-xl border p-4",
            prediction.accepted
              ? "border-success-100 bg-success-50"
              : "border-warning-100 bg-warning-50",
          )}
        >
          <div className="min-w-0">
            <p
              className={cn(
                "text-sm font-semibold",
                prediction.accepted ? "text-success-600" : "text-warning-600",
              )}
            >
              {prediction.accepted ? "✓ Görüntü kabul edildi" : "⚠ Görüntü reddedildi"}
            </p>
            <p className="mt-0.5 text-xs text-foreground-secondary">
              {prediction.accepted
                ? `Sınıf: ${prediction.predicted_class ?? "—"} · Güven: ${((prediction.confidence ?? 0) * 100).toFixed(1)}%`
                : prediction.rejection_reason ?? "Reddedilme nedeni bilinmiyor"}
            </p>
          </div>
          <button
            type="button"
            onClick={handleClear}
            className="shrink-0 text-xs text-foreground-muted underline hover:text-foreground"
          >
            Değiştir
          </button>
        </div>
      ) : null}

      {/* Actions */}
      {selected && !prediction && !upload.isPending ? (
        <div className="space-y-3">
          {config.features.gradcam ? (
            <label className="flex cursor-pointer items-center gap-2.5 text-sm">
              <input
                type="checkbox"
                checked={enableGradcam}
                onChange={(e) => setEnableGradcam(e.target.checked)}
                className="h-4 w-4 rounded border-border text-brand-600"
              />
              <span className="text-foreground-secondary">
                Grad-CAM ısı haritası iste
              </span>
            </label>
          ) : null}
          <Button onClick={handleUpload} isLoading={upload.isPending}>
            Görüntüyü analiz et
          </Button>
        </div>
      ) : null}

      {upload.isPending ? (
        <div className="space-y-2">
          {upload.progress !== null ? (
            <Progress value={upload.progress} label="Yükleniyor" />
          ) : (
            <Progress value={100} label="Analiz ediliyor…" />
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => { upload.cancel(); handleClear(); }}
          >
            İptal
          </Button>
        </div>
      ) : null}

      {upload.isError ? (
        <Alert variant="danger" title="Yükleme başarısız">
          {upload.error?.message ?? "Bilinmeyen hata."}
        </Alert>
      ) : null}

      {/* Navigation */}
      <div className="flex items-center justify-between gap-3">
        <Button variant="secondary" onClick={onBack}>
          Geri
        </Button>
        <div className="flex gap-2">
          {!prediction ? (
            <Button variant="ghost" onClick={onSkip}>
              Görüntü olmadan devam et
            </Button>
          ) : null}
          <Button onClick={onNext} size="lg" disabled={upload.isPending}>
            {prediction ? "Devam: İnceleme" : "Görüntüsüz devam"}
          </Button>
        </div>
      </div>
    </div>
  );
}
