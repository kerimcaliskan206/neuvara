"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

export interface GradCamOverlayProps {
  /** Browser-side preview URL of the original upload (object URL). */
  originalImageUrl: string;
  /** Base64-encoded Grad-CAM overlay returned by the backend. */
  gradcamBase64: string | null;
  /** Whether the backend was asked for Grad-CAM but the response is still loading. */
  isLoading?: boolean;
}

export function GradCamOverlay({
  originalImageUrl,
  gradcamBase64,
  isLoading,
}: GradCamOverlayProps) {
  const [showHeatmap, setShowHeatmap] = useState(true);

  const overlaySrc = gradcamBase64
    ? `data:image/png;base64,${gradcamBase64}`
    : null;

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <CardTitle>Grad-CAM</CardTitle>
        {overlaySrc ? (
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={() => setShowHeatmap((v) => !v)}
          >
            {showHeatmap ? "Isı haritasını gizle" : "Isı haritasını göster"}
          </Button>
        ) : null}
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <Skeleton className="aspect-square w-full max-w-md" />
        ) : (
          <div className="relative inline-block max-w-md overflow-hidden rounded-lg border border-border">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={originalImageUrl}
              alt="Yüklenen görüntü"
              className="block h-auto w-full"
            />
            {overlaySrc && showHeatmap ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={overlaySrc}
                alt="Grad-CAM ısı haritası"
                className="absolute inset-0 h-full w-full mix-blend-multiply opacity-80"
              />
            ) : null}
          </div>
        )}
        {!isLoading && !overlaySrc ? (
          <p className="mt-3 text-sm text-muted-foreground">
            Bu tahmin için Grad-CAM çıktısı bulunmuyor.
          </p>
        ) : null}
      </CardContent>
    </Card>
  );
}
