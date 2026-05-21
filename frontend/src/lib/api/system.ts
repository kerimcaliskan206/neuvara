import { api } from "@/lib/api/client";

export interface BackendHealthResponse {
  status: string;
  app: string;
  version: string;
  environment: string;
  uptime_seconds: number;
}

export const systemApi = {
  async health(): Promise<BackendHealthResponse> {
    // Tagged silent so failures don't toast — the consuming hook surfaces
    // its own offline banner.
    const { data } = await api.get<BackendHealthResponse>("/health", {
      headers: { "X-Silent-Errors": "1" },
      timeout: 5_000,
    });
    return data;
  },
};
