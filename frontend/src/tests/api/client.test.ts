import { describe, expect, it, vi } from "vitest";
import type { AxiosError } from "axios";

import { ApiError } from "@/lib/api/client";

describe("ApiError.fromAxios", () => {
  it("uses response body 'error' field when present", () => {
    const err = {
      response: { status: 422, data: { error: "Geçersiz girdi" } },
      message: "Request failed",
    } as unknown as AxiosError;

    const apiError = ApiError.fromAxios(err);
    expect(apiError).toBeInstanceOf(ApiError);
    expect(apiError.status).toBe(422);
    expect(apiError.message).toBe("Geçersiz girdi");
  });

  it("falls back to axios message when body is opaque", () => {
    const err = {
      response: { status: 500, data: "boom" },
      message: "Internal Server Error",
    } as unknown as AxiosError;

    const apiError = ApiError.fromAxios(err);
    expect(apiError.status).toBe(500);
    expect(apiError.body).toBe("boom");
    expect(apiError.message).toBe("Internal Server Error");
  });

  it("handles network errors with no response", () => {
    const err = {
      response: undefined,
      message: "Network Error",
    } as unknown as AxiosError;

    const apiError = ApiError.fromAxios(err);
    expect(apiError.status).toBe(0);
    expect(apiError.message).toBe("Network Error");
  });
});

describe("api module exports", () => {
  it("exposes both api + aiApi axios instances", async () => {
    const mod = await import("@/lib/api/client");
    expect(mod.api).toBeDefined();
    expect(mod.aiApi).toBeDefined();
    expect(typeof mod.api.get).toBe("function");
    expect(typeof mod.aiApi.post).toBe("function");
  });

  it("registerAuthAccessor swaps the token getter", async () => {
    const { registerAuthAccessor, api } = await import("@/lib/api/client");
    const getToken = vi.fn().mockReturnValue("test-token-123");
    registerAuthAccessor({ getToken, onUnauthorized: vi.fn() });

    // Issue a request through the interceptor chain without firing the
    // actual network: install a request adapter that returns synthetic data.
    type AdapterFn = NonNullable<typeof api.defaults.adapter>;
    const seenAuth: string[] = [];
    const adapter: AdapterFn = (cfg) => {
      const value = cfg.headers?.get?.("Authorization");
      if (typeof value === "string") seenAuth.push(value);
      return Promise.resolve({
        data: {},
        status: 200,
        statusText: "OK",
        headers: {},
        config: cfg,
      });
    };
    const previous = api.defaults.adapter;
    api.defaults.adapter = adapter;
    try {
      await api.get("/_test");
    } finally {
      api.defaults.adapter = previous;
    }

    expect(getToken).toHaveBeenCalled();
    expect(seenAuth).toContain("Bearer test-token-123");
  });
});
