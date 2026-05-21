import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useAuthStore } from "@/stores/auth-store";
import { authApi } from "@/lib/api/auth";

const fakeUser = {
  id: 1,
  username: "testuser",
  email: "test@example.com",
  is_active: true,
  created_at: "2026-05-14T00:00:00Z",
};

describe("useAuthStore", () => {
  beforeEach(() => {
    useAuthStore.setState({
      token: null,
      user: null,
      status: "idle",
      error: null,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("stores token and user on successful login", async () => {
    vi.spyOn(authApi, "login").mockResolvedValue({
      access_token: "abc.def.ghi",
      token_type: "bearer",
    });
    vi.spyOn(authApi, "me").mockResolvedValue(fakeUser);

    await useAuthStore.getState().login("test@example.com", "secret123");

    const state = useAuthStore.getState();
    expect(state.token).toBe("abc.def.ghi");
    expect(state.user?.email).toBe("test@example.com");
    expect(state.status).toBe("authenticated");
  });

  it("records error and clears token on failed login", async () => {
    vi.spyOn(authApi, "login").mockRejectedValue(new Error("Invalid credentials"));

    await expect(
      useAuthStore.getState().login("test@example.com", "wrong"),
    ).rejects.toThrow("Invalid credentials");

    const state = useAuthStore.getState();
    expect(state.token).toBeNull();
    expect(state.user).toBeNull();
    expect(state.status).toBe("unauthenticated");
    expect(state.error).toContain("Invalid credentials");
  });

  it("logout wipes token and user", () => {
    useAuthStore.setState({
      token: "abc",
      user: fakeUser,
      status: "authenticated",
    });
    useAuthStore.getState().logout();
    const state = useAuthStore.getState();
    expect(state.token).toBeNull();
    expect(state.user).toBeNull();
    expect(state.status).toBe("unauthenticated");
  });
});
