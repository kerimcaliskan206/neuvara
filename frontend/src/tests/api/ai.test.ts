import { describe, expect, it } from "vitest";

import { aiApi } from "@/lib/api/client";
import { aiAssistant } from "@/lib/api/ai";

describe("aiAssistant.chat", () => {
  it("posts to /chat and returns the response body", async () => {
    type Adapter = NonNullable<typeof aiApi.defaults.adapter>;
    const adapter: Adapter = (cfg) =>
      Promise.resolve({
        data: {
          content: "Merhaba, ben HantaProject asistanı.",
          intent: "general_domain",
          refused: false,
          refusal_reason: null,
          model: "fake-model",
          duration_ms: 12.3,
          prompt_tokens: 5,
          completion_tokens: 9,
          total_tokens: 14,
          timestamp: "2026-05-14T00:00:00Z",
        },
        status: 200,
        statusText: "OK",
        headers: {},
        config: cfg,
      });
    const previous = aiApi.defaults.adapter;
    aiApi.defaults.adapter = adapter;
    try {
      const reply = await aiAssistant.chat({ message: "Hantavirüs nedir?" });
      expect(reply.refused).toBe(false);
      expect(reply.content).toContain("HantaProject");
      expect(reply.intent).toBe("general_domain");
    } finally {
      aiApi.defaults.adapter = previous;
    }
  });
});
