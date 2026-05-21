"use client";

import { Alert } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Spinner } from "@/components/ui/spinner";
import { formatTimestamp } from "@/lib/utils";
import type { InterpretationResponse } from "@/lib/api/types";

export interface ExplanationCardProps {
  isLoading: boolean;
  isError: boolean;
  error?: Error | null;
  data: InterpretationResponse | undefined;
  onRequest: () => void;
  /** Disable the request button when no prediction is available. */
  disabled?: boolean;
}

export function ExplanationCard({
  isLoading,
  isError,
  error,
  data,
  onRequest,
  disabled,
}: ExplanationCardProps) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <div>
          <CardTitle>AI Açıklama</CardTitle>
          <p className="text-xs text-muted-foreground">
            Türkçe, kısa açıklama — tıbbi teşhis yerine geçmez.
          </p>
        </div>
        <Button
          size="sm"
          onClick={onRequest}
          disabled={disabled || isLoading}
          isLoading={isLoading}
        >
          {data ? "Tekrar açıkla" : "AI ile açıkla"}
        </Button>
      </CardHeader>
      <CardContent className="space-y-3">
        {isLoading ? (
          <div className="space-y-2">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-5/6" />
            <Skeleton className="h-4 w-2/3" />
            <Spinner size="sm" label="Açıklama hazırlanıyor..." />
          </div>
        ) : null}
        {isError ? (
          <Alert variant="danger" title="Açıklama alınamadı">
            {error?.message ?? "Bilinmeyen hata."}
          </Alert>
        ) : null}
        {data ? (
          <div>
            <p className="whitespace-pre-wrap text-sm leading-relaxed">
              {data.content}
            </p>
            <p className="mt-3 text-[11px] text-muted-foreground">
              {data.model} · {data.duration_ms.toFixed(0)} ms ·{" "}
              {formatTimestamp(data.timestamp)}
            </p>
          </div>
        ) : null}
        {!data && !isLoading && !isError ? (
          <p className="text-sm text-muted-foreground">
            Tahmin sonucunu AI&rsquo;a açıklatmak için yukarıdaki düğmeyi kullanın.
          </p>
        ) : null}
      </CardContent>
    </Card>
  );
}
