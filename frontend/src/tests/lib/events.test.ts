import { afterEach, describe, expect, it, vi } from "vitest";

import { dispatchAppEvent, onAppEvent } from "@/lib/events";

afterEach(() => vi.restoreAllMocks());

describe("events bus", () => {
  it("delivers a payload to subscribers and supports unsubscription", () => {
    const seen: string[] = [];
    const off = onAppEvent("api:offline", (detail) => {
      seen.push(detail.message);
    });
    dispatchAppEvent("api:offline", { message: "network down" });
    expect(seen).toEqual(["network down"]);

    off();
    dispatchAppEvent("api:offline", { message: "second" });
    expect(seen).toEqual(["network down"]); // not delivered after off()
  });

  it("isolates subscribers by event name", () => {
    const offline = vi.fn();
    const timeout = vi.fn();
    onAppEvent("api:offline", offline);
    onAppEvent("api:timeout", timeout);

    dispatchAppEvent("api:timeout", { message: "slow" });

    expect(offline).not.toHaveBeenCalled();
    expect(timeout).toHaveBeenCalledWith({ message: "slow" });
  });
});
