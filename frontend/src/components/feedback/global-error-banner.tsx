"use client";

import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { onAppEvent } from "@/lib/events";
import { cn } from "@/lib/utils";

type Banner =
  | { kind: "offline"; title: string; message: string }
  | { kind: "timeout"; title: string; message: string }
  | { kind: "server"; title: string; message: string };

const TITLES: Record<Banner["kind"], string> = {
  offline: "Sunucuya ulaşılamıyor",
  timeout: "İstek zaman aşımına uğradı",
  server: "Sunucu hatası",
};

const STYLES: Record<Banner["kind"], string> = {
  offline: "bg-danger text-danger-foreground",
  timeout: "bg-warning text-warning-foreground",
  server: "bg-danger text-danger-foreground",
};

/**
 * One mount point at the root.  Listens for transport events from the
 * axios layer and renders a dismissible banner.  Banners auto-dismiss
 * after a short window so transient failures don't pin the UI.
 */
export function GlobalErrorBanner() {
  const [banner, setBanner] = useState<Banner | null>(null);

  useEffect(() => {
    const offline = onAppEvent("api:offline", (detail) =>
      setBanner({ kind: "offline", title: TITLES.offline, message: detail.message }),
    );
    const timeout = onAppEvent("api:timeout", (detail) =>
      setBanner({ kind: "timeout", title: TITLES.timeout, message: detail.message }),
    );
    const server = onAppEvent("api:error", (detail) =>
      setBanner({
        kind: "server",
        title: `${TITLES.server} (${detail.status})`,
        message: detail.message,
      }),
    );
    return () => {
      offline();
      timeout();
      server();
    };
  }, []);

  useEffect(() => {
    if (!banner) return;
    const id = window.setTimeout(() => setBanner(null), 6_000);
    return () => window.clearTimeout(id);
  }, [banner]);

  if (!banner) return null;

  return (
    <div
      role="status"
      aria-live="polite"
      className={cn(
        "fixed inset-x-0 top-0 z-[60] flex items-center justify-center px-4 py-2 text-sm shadow-md",
        STYLES[banner.kind],
      )}
    >
      <div className="flex w-full max-w-3xl items-center justify-between gap-3">
        <div>
          <span className="font-semibold">{banner.title}.</span>{" "}
          <span className="opacity-90">{banner.message}</span>
        </div>
        <Button
          size="sm"
          variant="ghost"
          className="text-current hover:bg-white/10"
          onClick={() => setBanner(null)}
        >
          Kapat
        </Button>
      </div>
    </div>
  );
}
