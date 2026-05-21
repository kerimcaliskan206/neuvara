import "@testing-library/jest-dom/vitest";

import { afterEach, beforeEach } from "vitest";

// happy-dom provides localStorage, but we reset it between tests so the
// persisted auth store can't leak state across cases.
beforeEach(() => {
  if (typeof window !== "undefined") {
    window.localStorage.clear();
    window.sessionStorage.clear();
  }
});

afterEach(() => {
  // Pulled into its own hook so we can extend it later (e.g. unmount RTL).
});
