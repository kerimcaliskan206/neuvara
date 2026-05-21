"use client";

import { Bell, LogOut, Menu, Plus, ShieldCheck } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";

import { Button } from "@/components/ui/button";
import { useAuth } from "@/hooks/use-auth";
import { cn } from "@/lib/utils";

interface TopNavProps {
  onMenuClick?: () => void;
}

export function TopNav({ onMenuClick }: TopNavProps) {
  const router = useRouter();
  const { user, logout } = useAuth();

  const initials = user?.username
    ? user.username.slice(0, 2).toUpperCase()
    : user?.email?.slice(0, 2).toUpperCase() ?? "??";

  return (
    <header
      className="sticky top-0 z-30 flex h-14 shrink-0 items-center justify-between px-4 md:px-6"
      style={{
        background: "hsl(222 40% 5% / 0.85)",
        backdropFilter: "blur(20px)",
        WebkitBackdropFilter: "blur(20px)",
        borderBottom: "1px solid hsl(222 22% 12%)",
      }}
    >
      {/* Mobile hamburger */}
      <div className="flex items-center gap-2 md:hidden">
        <button
          type="button"
          aria-label="Menüyü aç"
          onClick={onMenuClick}
          className={cn(
            "flex h-9 w-9 items-center justify-center rounded-lg",
            "text-foreground-secondary hover:bg-white/5 hover:text-foreground",
            "transition-all duration-150 ease-swift-out",
          )}
        >
          <Menu className="h-4 w-4" />
        </button>
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg brand-gradient">
            <ShieldCheck className="h-3.5 w-3.5 text-white" />
          </div>
          <span className="text-sm font-bold tracking-tight text-foreground">NEURAVA</span>
        </div>
      </div>

      {/* Desktop: page context + AI status */}
      <div className="hidden md:flex md:items-center gap-2.5">
        <span className="text-xs font-medium text-foreground-muted">Karar Destek Paneli</span>
        <span className="text-foreground-muted/30">·</span>
        <span className="text-xs text-foreground-muted">Pulmoner Risk Analizi</span>
        <span className="text-foreground-muted/30">·</span>
        <span className="flex items-center gap-1.5">
          <span className="h-1.5 w-1.5 rounded-full bg-success-500 animate-pulse-slow" />
          <span className="text-xs font-medium text-success-500">AI Aktif</span>
        </span>
      </div>

      {/* Right actions */}
      <div className="flex items-center gap-1.5 sm:gap-2">
        {/* Quick CTA — desktop only */}
        <Link
          href="/medical"
          className={cn(
            "hidden md:flex items-center gap-1.5 rounded-lg px-3 py-1.5 mr-1",
            "bg-brand-600/80 hover:bg-brand-600 border border-brand-500/40",
            "text-xs font-semibold text-white",
            "shadow-[0_2px_8px_-2px_hsl(221_83%_53%/0.4)]",
            "transition-colors duration-150",
          )}
        >
          <Plus className="h-3 w-3" />
          Yeni Analiz
        </Link>

        <button
          aria-label="Bildirimler"
          className={cn(
            "flex h-8 w-8 items-center justify-center rounded-lg text-foreground-muted",
            "hover:bg-white/5 hover:text-foreground active:scale-95",
            "transition-all duration-150 ease-swift-out",
          )}
        >
          <Bell className="h-4 w-4" />
        </button>

        {user && (
          <div className="hidden items-center gap-2.5 pl-2 sm:flex">
            <div
              className={cn(
                "flex h-8 w-8 items-center justify-center rounded-full",
                "brand-gradient text-xs font-bold text-white",
                "ring-1 ring-brand-600/40",
              )}
            >
              {initials}
            </div>
            <div className="text-right">
              <p className="text-xs font-semibold leading-none text-foreground">
                {user.username}
              </p>
              <p className="mt-0.5 text-2xs leading-none text-foreground-muted">
                {user.email}
              </p>
            </div>
          </div>
        )}

        <Button
          variant="ghost"
          size="sm"
          className="gap-1.5 text-foreground-muted hover:text-foreground hover:bg-white/5"
          onClick={() => {
            logout();
            router.replace("/login");
          }}
        >
          <LogOut className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">Çıkış</span>
        </Button>
      </div>
    </header>
  );
}
