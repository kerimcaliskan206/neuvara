"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { GlobalErrorBanner } from "@/components/feedback/global-error-banner";
import { onAppEvent } from "@/lib/events";

// Importing the auth store eagerly here registers the axios auth accessor
// so every subsequent request carries the bearer token automatically.
import "@/stores/auth-store";

export function Providers({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            retry: (failureCount, error) => {
              // Don't retry auth failures — push to /login instead.
              const status = (error as { status?: number } | undefined)?.status;
              if (status === 401 || status === 403) return false;
              return failureCount < 1;
            },
            refetchOnWindowFocus: false,
            staleTime: 30_000,
          },
          mutations: {
            retry: false,
          },
        },
      }),
  );

  // Listen for auth:unauthorized — one place to know how to redirect.
  useEffect(() => {
    return onAppEvent("auth:unauthorized", () => {
      router.replace("/login");
    });
  }, [router]);

  return (
    <QueryClientProvider client={client}>
      <GlobalErrorBanner />
      {children}
    </QueryClientProvider>
  );
}
