import { describe, expect, it } from "vitest";

import { aiApi } from "@/lib/api/client";
import { aiAssistant } from "@/lib/api/ai";
import type {
  InterpretationResponse,
  VisionInterpretationRequest,
} from "@/lib/api/types";

describe("aiAssistant.explainVision", () => {
  it("posts the prediction body to /explain/vision", async () => {
    type Adapter = NonNullable<typeof aiApi.defaults.adapter>;
    const captured: { value: VisionInterpretationRequest | null } = { value: null };
    const adapter: Adapter = (cfg) => {
      captured.value = (typeof cfg.data === "string"
        ? JSON.parse(cfg.data)
        : cfg.data) as VisionInterpretationRequest;
      const body: InterpretationResponse = {
        content: "Kabul edilen sonuç; güven yüksek.",
        model: "fake-model",
        duration_ms: 4.2,
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
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
    const previous = aiApi.defaults.adapter;
    aiApi.defaults.adapter = adapter;
    try {
      const reply = await aiAssistant.explainVision({
        accepted: true,
        predicted_class: "related",
        confidence: 0.92,
        threshold: 0.5,
        rejection_reason: null,
        gate: { enabled: true, predicted_class: "related", confidence: 0.95 },
        model_name: "efficientnet_b0",
        model_version: "v20260514_120000",
      });
      expect(reply.content).toContain("Kabul");
      expect(captured.value?.accepted).toBe(true);
      expect(captured.value?.gate.enabled).toBe(true);
    } finally {
      aiApi.defaults.adapter = previous;
    }
  });
});
