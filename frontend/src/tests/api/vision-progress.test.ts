import { describe, expect, it, vi } from "vitest";

import { api } from "@/lib/api/client";
import { visionApi } from "@/lib/api/vision";
import type { VisionPredictionResponse } from "@/lib/api/types";

describe("visionApi.predict — progress + cancellation", () => {
  it("invokes onProgress with percentages, then resolves", async () => {
    type Adapter = NonNullable<typeof api.defaults.adapter>;
    const onProgress = vi.fn();
    const adapter: Adapter = (cfg) => {
      // Simulate two progress events before resolving.  AxiosProgressEvent
      // requires `bytes` + `lengthComputable` in addition to loaded/total.
      cfg.onUploadProgress?.({
        loaded: 50,
        total: 100,
        bytes: 50,
        lengthComputable: true,
      });
      cfg.onUploadProgress?.({
        loaded: 100,
        total: 100,
        bytes: 50,
        lengthComputable: true,
      });

      const body: VisionPredictionResponse = {
        accepted: true,
        predicted_class: "related",
        predicted_class_index: 1,
        confidence: 0.9,
        probabilities: { related: 0.9, unrelated: 0.1 },
        threshold: 0.5,
        rejection_reason: null,
        gate: { enabled: false, predicted_class: null, confidence: null },
        image: null,
        upload: null,
        model_name: "efficientnet_b0",
        model_version: "v20260514_120000",
        inference_duration_ms: 12.3,
        gradcam_base64: null,
        timestamp: "2026-05-14T00:00:00Z",
      };
      return Promise.resolve({
        data: body,
        status: 200,
        statusText: "OK",
        headers: {},
        config: cfg,
      });
    };
    const previous = api.defaults.adapter;
    api.defaults.adapter = adapter;
    try {
      const file = new File([new Uint8Array(64)], "x.jpg", { type: "image/jpeg" });
      const result = await visionApi.predict(file, { onProgress });
      expect(result.accepted).toBe(true);
      expect(onProgress).toHaveBeenCalledWith(50);
      expect(onProgress).toHaveBeenCalledWith(100);
    } finally {
      api.defaults.adapter = previous;
    }
  });

  it("forwards AbortSignal to axios config", async () => {
    type Adapter = NonNullable<typeof api.defaults.adapter>;
    let sawSignal = false;
    const adapter: Adapter = (cfg) => {
      sawSignal = cfg.signal instanceof AbortSignal;
      return Promise.resolve({
        data: {} as VisionPredictionResponse,
        status: 200,
        statusText: "OK",
        headers: {},
        config: cfg,
      });
    };
    const previous = api.defaults.adapter;
    api.defaults.adapter = adapter;
    const controller = new AbortController();
    try {
      const file = new File([new Uint8Array(8)], "x.jpg", { type: "image/jpeg" });
      await visionApi.predict(file, { signal: controller.signal });
      expect(sawSignal).toBe(true);
    } finally {
      api.defaults.adapter = previous;
    }
  });
});
