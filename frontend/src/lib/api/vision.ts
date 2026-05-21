import { api } from "@/lib/api/client";
import type {
  VisionModelInfoResponse,
  VisionPredictionResponse,
} from "@/lib/api/types";

export interface VisionPredictOptions {
  gradcam?: boolean;
  threshold?: number;
  /** Receives a 0..100 percent on each upload progress event. */
  onProgress?: (percent: number) => void;
  /** AbortController.signal so the caller can cancel mid-upload. */
  signal?: AbortSignal;
}

export const visionApi = {
  async predict(
    file: File,
    options: VisionPredictOptions = {},
  ): Promise<VisionPredictionResponse> {
    const form = new FormData();
    form.append("file", file);

    const params: Record<string, string> = {};
    if (options.gradcam) params.gradcam = "true";
    if (typeof options.threshold === "number") {
      params.threshold = String(options.threshold);
    }

    const { data } = await api.post<VisionPredictionResponse>(
      "/vision/predict",
      form,
      {
        params,
        headers: { "Content-Type": "multipart/form-data" },
        signal: options.signal,
        onUploadProgress: (event) => {
          if (!options.onProgress) return;
          // Axios v1 sometimes reports `total` as 0 (chunked encoding etc).
          if (!event.total) return;
          const percent = Math.round((event.loaded / event.total) * 100);
          options.onProgress(Math.max(0, Math.min(100, percent)));
        },
      },
    );
    return data;
  },

  async modelInfo(): Promise<VisionModelInfoResponse> {
    const { data } = await api.get<VisionModelInfoResponse>("/vision/models/current");
    return data;
  },
};
