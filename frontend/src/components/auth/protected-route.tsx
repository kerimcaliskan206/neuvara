"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

import { Spinner } from "@/components/ui/spinner";
import { useAuth } from "@/hooks/use-auth";

interface ProtectedRouteProps {
  children: React.ReactNode;
}

/**
 * Client-side gate.
 *
 * Wait for hydration → if no token, push to /login.  Otherwise render the
 * children once a user record exists (useAuth will lazily fetch it).
 *
 * NB: This is a UX guard, not a security boundary — the backend remains
 * the authority via Authorization: Bearer headers on every request.
 */
export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const router = useRouter();
  const { hasHydrated, token, user } = useAuth();

  useEffect(() => {
    if (!hasHydrated) return;
    if (!token) {
      router.replace("/login");
    }
  }, [hasHydrated, token, router]);

  if (!hasHydrated || (token && !user)) {
    return (
      <div className="flex min-h-[40vh] items-center justify-center">
        <Spinner label="Yükleniyor..." />
      </div>
    );
  }

  if (!token) return null;

  return <>{children}</>;
}
