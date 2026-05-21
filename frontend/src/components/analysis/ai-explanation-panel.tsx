"use client";

import { BotMessageSquare, RefreshCw } from "lucide-react";

import { Alert } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { LoadingDots } from "@/components/ui/loading-dots";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import type { InterpretationResponse } from "@/lib/api/types";

interface AiExplanationPanelProps {
  isLoading: boolean;
  isError: boolean;
  error?: Error | null;
  data?: InterpretationResponse | null;
  onRequest: () => void;
  disabled?: boolean;
  className?: string;
}

export function AiExplanationPanel({
  isLoading,
  isError,
  error,
  data,
  onRequest,
  disabled,
  className,
}: AiExplanationPanelProps) {
  return (
    <div className={cn("space-y-4", className)}>
      {/* Header row */}
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-50">
            <BotMessageSquare className="h-4 w-4 text-brand-600" />
          </div>
          <div>
            <p className="text-sm font-semibold text-foreground">AI Açıklama</p>
            <p className="text-xs text-foreground-muted">
              Türkçe, sade dil · Tıbbi teşhis değildir
            </p>
          </div>
        </div>
        <Button
          size="sm"
          variant={data ? "secondary" : "primary"}
          onClick={onRequest}
          disabled={disabled || isLoading}
          isLoading={isLoading}
          className="shrink-0"
        >
          {!isLoading && <RefreshCw className={cn("h-3.5 w-3.5", data && "opacity-70")} />}
          {data ? "Tekrar açıkla" : "AI ile açıkla"}
        </Button>
      </div>

      {/* Content states */}
      {isLoading ? (
        <div className="space-y-3 rounded-xl border border-border bg-canvas p-4 animate-fade-in">
          <LoadingDots label="Açıklama hazırlanıyor…" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-5/6" />
          <Skeleton className="h-4 w-4/5" />
          <Skeleton className="h-4 w-2/3" />
        </div>
      ) : null}

      {isError && !isLoading ? (
        <Alert variant="danger" title="Açıklama alınamadı">
          {error?.message ?? "AI servisi yanıt vermedi. Lütfen tekrar deneyin."}
        </Alert>
      ) : null}

      {data && !isLoading ? (
        <div className="rounded-xl border border-brand-100 bg-brand-50 p-5 animate-fade-up">
          <p className="whitespace-pre-wrap text-sm leading-relaxed text-foreground">
            {data.content}
          </p>
          <p className="mt-4 text-2xs text-foreground-muted">
            {data.model} · {data.duration_ms.toFixed(0)} ms
          </p>
        </div>
      ) : null}

      {!data && !isLoading && !isError ? (
        <div className="rounded-xl border border-dashed border-border bg-canvas p-5 text-center">
          <BotMessageSquare className="mx-auto mb-2 h-6 w-6 text-foreground-muted" />
          <p className="text-sm text-foreground-muted">
            Sonucu AI&rsquo;a Türkçe açıklatmak için &ldquo;AI ile açıkla&rdquo; düğmesini kullanın.
          </p>
        </div>
      ) : null}
    </div>
  );
}
