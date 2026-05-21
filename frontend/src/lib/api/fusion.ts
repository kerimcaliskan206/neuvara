import { api } from "@/lib/api/client";
import type { FusionRequest, FusionResponse } from "@/lib/api/types";

export const fusionApi = {
  async predict(payload: FusionRequest): Promise<FusionResponse> {
    const { data } = await api.post<FusionResponse>("/fusion/predict", payload);
    return data;
  },

  async health(): Promise<Record<string, unknown>> {
    const { data } = await api.get<Record<string, unknown>>("/fusion/health");
    return data;
  },
};
