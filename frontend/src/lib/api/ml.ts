import { api } from "@/lib/api/client";
import type { MLPredictionResponse, PatientInput } from "@/lib/api/types";

export const mlApi = {
  async predict(
    patient: PatientInput,
    modelName?: string | null,
  ): Promise<MLPredictionResponse> {
    const { data } = await api.post<MLPredictionResponse>("/ml/predict", {
      patient,
      ...(modelName ? { model_name: modelName } : {}),
    });
    return data;
  },

  async modelInfo(): Promise<Record<string, unknown>> {
    const { data } = await api.get<Record<string, unknown>>("/ml/models/current");
    return data;
  },
};
