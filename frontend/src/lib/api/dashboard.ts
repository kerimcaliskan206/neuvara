import { api } from "@/lib/api/client";

export interface WeeklyTrendItem {
  date: string;
  count: number;
}

export interface RecentAnalysisItem {
  id: string;
  created_at: string;
  risk_score: number;
  risk_tier: string;
  confidence: number | null;
}

export interface RiskDistribution {
  low: number;
  moderate: number;
  high: number;
  critical: number;
}

export interface SystemStatus {
  online: boolean;
  version: string;
  model: string;
  last_check: string;
}

export interface DashboardSummary {
  total_analyses: number;
  high_risk_count: number;
  average_confidence: number | null;
  average_duration_seconds: number | null;
  weekly_trend: WeeklyTrendItem[];
  risk_distribution: RiskDistribution;
  recent_analyses: RecentAnalysisItem[];
  system_status: SystemStatus;
}

export const EMPTY_DASHBOARD_SUMMARY: DashboardSummary = {
  total_analyses: 0,
  high_risk_count: 0,
  average_confidence: null,
  average_duration_seconds: null,
  weekly_trend: [],
  risk_distribution: { low: 0, moderate: 0, high: 0, critical: 0 },
  recent_analyses: [],
  system_status: { online: false, version: "—", model: "—", last_check: "" },
};

export const dashboardApi = {
  async getSummary(): Promise<DashboardSummary> {
    const { data } = await api.get<DashboardSummary>("/dashboard/summary", {
      headers: { "X-Silent-Errors": "1" },
    });
    return data;
  },
};
