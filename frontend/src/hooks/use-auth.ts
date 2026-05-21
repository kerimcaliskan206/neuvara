"use client";

import { useEffect } from "react";

import { useAuthStore } from "@/stores/auth-store";

/**
 * Thin selector hook around the auth store.
 *
 * Triggers a `/auth/me` refresh once the persisted token rehydrates so the
 * dashboard always reflects the server's view of the user.
 */
export function useAuth() {
  const token = useAuthStore((s) => s.token);
  const user = useAuthStore((s) => s.user);
  const status = useAuthStore((s) => s.status);
  const error = useAuthStore((s) => s.error);
  const hasHydrated = useAuthStore((s) => s.hasHydrated);
  const refreshUser = useAuthStore((s) => s.refreshUser);
  const logout = useAuthStore((s) => s.logout);

  useEffect(() => {
    if (!hasHydrated) return;
    if (token && !user) {
      void refreshUser();
    }
  }, [hasHydrated, token, user, refreshUser]);

  return {
    token,
    user,
    status,
    error,
    hasHydrated,
    isAuthenticated: Boolean(token && user),
    logout,
  };
}
