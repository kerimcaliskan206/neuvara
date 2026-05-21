"use client";

import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { formatPercent, formatTimestamp } from "@/lib/utils";
import type { VisionPredictionResponse } from "@/lib/api/types";

export function PredictionResultSkeleton() {
  return (
    <Card>
      <CardHeader>
        <Skeleton className="h-6 w-40" />
      </CardHeader>
      <CardContent className="space-y-3">
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-2/3" />
        <Skeleton className="h-4 w-1/2" />
      </CardContent>
    </Card>
  );
}

export interface PredictionResultProps {
  prediction: VisionPredictionResponse;
}

export function PredictionResult({ prediction }: PredictionResultProps) {
  const accepted = prediction.accepted;

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-3 space-y-0">
        <CardTitle>Tahmin Sonucu</CardTitle>
        <Badge variant={accepted ? "success" : "danger"}>
          {accepted ? "Kabul edildi" : "Reddedildi"}
        </Badge>
      </CardHeader>
      <CardContent className="space-y-4">
        {!accepted && prediction.rejection_reason ? (
          <Alert variant="warning" title="Reddedilme nedeni">
            {prediction.rejection_reason}
          </Alert>
        ) : null}

        <dl className="grid grid-cols-1 gap-3 text-sm sm:grid-cols-2">
          <div>
            <dt className="text-muted-foreground">Tahmin sınıfı</dt>
            <dd className="font-medium">{prediction.predicted_class ?? "—"}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground">Güven</dt>
            <dd className="font-medium">{formatPercent(prediction.confidence)}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground">Eşik</dt>
            <dd className="font-medium">{formatPercent(prediction.threshold)}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground">Model</dt>
            <dd className="font-medium">
              {prediction.model_name}
              <span className="ml-1 text-xs text-muted-foreground">
                {prediction.model_version}
              </span>
            </dd>
          </div>
          <div>
            <dt className="text-muted-foreground">İlgililik kapısı</dt>
            <dd className="font-medium">
              {prediction.gate.enabled
                ? `${prediction.gate.predicted_class ?? "—"} (${formatPercent(prediction.gate.confidence)})`
                : "Devre dışı"}
            </dd>
          </div>
          <div>
            <dt className="text-muted-foreground">Çıkarım süresi</dt>
            <dd className="font-medium">
              {prediction.inference_duration_ms.toFixed(1)} ms
            </dd>
          </div>
          <div className="sm:col-span-2">
            <dt className="text-muted-foreground">Zaman damgası</dt>
            <dd className="font-medium">{formatTimestamp(prediction.timestamp)}</dd>
          </div>
        </dl>

        {prediction.probabilities && Object.keys(prediction.probabilities).length > 0 ? (
          <div>
            <p className="mb-2 text-sm font-medium">Sınıf olasılıkları</p>
            <ul className="space-y-1.5">
              {Object.entries(prediction.probabilities).map(([name, value]) => (
                <li key={name} className="flex items-center gap-2 text-xs">
                  <span className="w-24 shrink-0 truncate text-muted-foreground">
                    {name}
                  </span>
                  <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                    <div
                      className="h-full bg-primary"
                      style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
                    />
                  </div>
                  <span className="w-12 shrink-0 text-right font-medium">
                    {formatPercent(value)}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
