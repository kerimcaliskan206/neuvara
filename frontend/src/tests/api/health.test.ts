import { describe, expect, it } from "vitest";

import { api } from "@/lib/api/client";
import { systemApi } from "@/lib/api/system";

describe("systemApi.health", () => {
  it("returns parsed health payload", async () => {
    type Adapter = NonNullable<typeof api.defaults.adapter>;
    const adapter: Adapter = (cfg) =>
      Promise.resolve({
        data: {
          status: "ok",
          app: "HantaProject",
          version: "0.1.0",
          environment: "development",
          uptime_seconds: 12.3,
        },
        status: 200,
        statusText: "OK",
        headers: {},
        config: cfg,
      });
    const previous = api.defaults.adapter;
    api.defaults.adapter = adapter;
    try {
      const health = await systemApi.health();
      expect(health.status).toBe("ok");
      expect(health.version).toBe("0.1.0");
    } finally {
      api.defaults.adapter = previous;
    }
  });

  it("propagates network failures as ApiError", async () => {
    type Adapter = NonNullable<typeof api.defaults.adapter>;
    const adapter: Adapter = () =>
      Promise.reject(
        Object.assign(new Error("Network Error"), {
          isAxiosError: true,
          response: undefined,
          code: "ECONNREFUSED",
        }),
      );
    const previous = api.defaults.adapter;
    api.defaults.adapter = adapter;
    try {
      await expect(systemApi.health()).rejects.toMatchObject({
        name: "ApiError",
        status: 0,
      });
    } finally {
      api.defaults.adapter = previous;
    }
  });
});
