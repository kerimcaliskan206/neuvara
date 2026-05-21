"use client";

import { useQuery } from "@tanstack/react-query";

import {
  dashboardApi,
  EMPTY_DASHBOARD_SUMMARY,
  type DashboardSummary,
} from "@/lib/api/dashboard";

export function useDashboardSummary(): DashboardSummary & { isLoading: boolean } {
  const query = useQuery({
    queryKey: ["dashboard-summary"],
    queryFn: dashboardApi.getSummary,
    refetchInterval: 60_000,
    staleTime: 30_000,
    retry: 1,
  });

  return {
    ...(query.data ?? EMPTY_DASHBOARD_SUMMARY),
    isLoading: query.isLoading,
  };
}
