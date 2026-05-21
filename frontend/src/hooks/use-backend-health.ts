"use client";

import { useQuery } from "@tanstack/react-query";

import { systemApi, type BackendHealthResponse } from "@/lib/api/system";

export interface BackendHealthState {
  data: BackendHealthResponse | undefined;
  isOnline: boolean;
  isLoading: boolean;
  lastCheckedAt: number | null;
}

/**
 * Polls the backend /health endpoint every 30s.
 *
 * The query is `silent` (the axios interceptor skips event dispatch) so a
 * dropped connection doesn't spam api:offline events.  Consumers gate UI
 * on `isOnline` and render their own indicator.
 */
export function useBackendHealth(): BackendHealthState {
  const query = useQuery({
    queryKey: ["backend-health"],
    queryFn: systemApi.health,
    refetchInterval: 30_000,
    refetchOnWindowFocus: true,
    retry: 1,
    staleTime: 15_000,
  });

  return {
    data: query.data,
    isOnline: !query.isError && query.data?.status === "ok",
    isLoading: query.isLoading,
    lastCheckedAt: query.dataUpdatedAt || null,
  };
}
