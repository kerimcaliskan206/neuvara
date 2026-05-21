/**
 * Zustand auth store.
 *
 * Holds the access token + the current user.  Persists the token to
 * localStorage so a page reload keeps the user signed in.  We deliberately
 * do NOT persist the user record — it's re-fetched from /auth/me on hydrate
 * so a stale user can't outlive a server-side revocation.
 */
"use client";

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

import { authApi } from "@/lib/api/auth";
import { registerAuthAccessor } from "@/lib/api/client";
import { logger } from "@/lib/logger";
import type { UserResponse } from "@/lib/api/types";

interface AuthState {
  token: string | null;
  user: UserResponse | null;
  status: "idle" | "loading" | "authenticated" | "unauthenticated";
  error: string | null;
  hasHydrated: boolean;

  login: (email: string, password: string) => Promise<void>;
  register: (username: string, email: string, password: string) => Promise<UserResponse>;
  forgotPassword: (email: string) => Promise<void>;
  logout: () => void;
  refreshUser: () => Promise<void>;
  setHasHydrated: (value: boolean) => void;
  clearError: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      token: null,
      user: null,
      status: "idle",
      error: null,
      hasHydrated: false,

      async login(email, password) {
        set({ status: "loading", error: null });
        try {
          const { access_token } = await authApi.login({ email, password });
          set({ token: access_token });
          const user = await authApi.me();
          set({ user, status: "authenticated" });
        } catch (err) {
          const message = err instanceof Error ? err.message : "Giriş başarısız.";
          set({ token: null, user: null, status: "unauthenticated", error: message });
          throw err;
        }
      },

      async register(username, email, password) {
        set({ status: "loading", error: null });
        try {
          const user = await authApi.register({ username, email, password });
          set({ status: "unauthenticated" });
          return user;
        } catch (err) {
          const message = err instanceof Error ? err.message : "Kayıt başarısız.";
          set({ status: "unauthenticated", error: message });
          throw err;
        }
      },

      async forgotPassword(email) {
        try {
          await authApi.forgotPassword(email);
        } catch (err) {
          const message = err instanceof Error ? err.message : "İstek gönderilemedi.";
          throw new Error(message);
        }
      },

      logout() {
        set({ token: null, user: null, status: "unauthenticated", error: null });
      },

      async refreshUser() {
        if (!get().token) {
          set({ status: "unauthenticated" });
          return;
        }
        try {
          const user = await authApi.me();
          set({ user, status: "authenticated" });
        } catch (err) {
          logger.warn("auth refreshUser failed", err);
          set({ token: null, user: null, status: "unauthenticated" });
        }
      },

      setHasHydrated(value) {
        set({ hasHydrated: value });
      },

      clearError() {
        set({ error: null });
      },
    }),
    {
      name: "hanta-auth",
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({ token: state.token }),
      onRehydrateStorage: () => (state) => {
        state?.setHasHydrated(true);
      },
    },
  ),
);

// ── Bridge the store into the axios layer ────────────────────────────────────
// The api client reads the token from `getToken()` on each request, so we
// only have to register once.  401 responses clear the store.
registerAuthAccessor({
  getToken: () => useAuthStore.getState().token,
  onUnauthorized: () => {
    useAuthStore.getState().logout();
    if (typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("auth:unauthorized"));
    }
  },
});
