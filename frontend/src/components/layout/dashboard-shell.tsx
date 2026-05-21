"use client";

import { usePathname } from "next/navigation";
import { useState } from "react";

import { ProtectedRoute } from "@/components/auth/protected-route";
import {
  PageAtmosphere,
  atmosphereVariantForPath,
} from "@/components/layout/page-atmosphere";
import { Sidebar } from "@/components/layout/sidebar";
import { TopNav } from "@/components/layout/topnav";

export function DashboardShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [drawerOpen, setDrawerOpen] = useState(false);

  const variant = atmosphereVariantForPath(pathname);

  return (
    <ProtectedRoute>
      <div className="relative flex min-h-screen">
        {/* Ambient atmospheric backdrop (fixed, behind everything) */}
        <PageAtmosphere variant={variant} />

        {/* Foreground stack */}
        <Sidebar
          mobileOpen={drawerOpen}
          onMobileClose={() => setDrawerOpen(false)}
        />
        <div className="relative z-10 flex min-w-0 flex-1 flex-col">
          <TopNav onMenuClick={() => setDrawerOpen(true)} />
          <main className="flex-1 overflow-x-hidden p-4 sm:p-5 md:p-7">
            <div
              key={pathname}
              className="mx-auto w-full max-w-[1400px] animate-fade-up"
            >
              {children}
            </div>
          </main>
        </div>
      </div>
    </ProtectedRoute>
  );
}
