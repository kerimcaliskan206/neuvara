"use client";

import { motion } from "framer-motion";
import {
  Bot,
  ChevronRight,
  Clock,
  LayoutDashboard,
  Plus,
  Settings,
  ShieldCheck,
  Stethoscope,
  X,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { cn } from "@/lib/utils";

// ── Nav items ─────────────────────────────────────────────────────────────────

interface NavItem {
  href: string;
  label: string;
  icon: React.ElementType;
  enabled: boolean;
  badge?: string;
  group: "primary" | "secondary";
}

const PRIMARY_ITEMS: NavItem[] = [
  { href: "/dashboard", label: "Genel Bakış", icon: LayoutDashboard, enabled: true, group: "primary" },
  { href: "/medical",   label: "Yeni Analiz", icon: Plus,            enabled: true, group: "primary" },
];

const SECONDARY_ITEMS: NavItem[] = [
  { href: "/history",    label: "Geçmiş Analizler",  icon: Clock,           enabled: false, badge: "Yakında", group: "secondary" },
  { href: "/reports",    label: "Raporlar",           icon: Stethoscope,     enabled: false, badge: "Yakında", group: "secondary" },
  { href: "/settings",   label: "Ayarlar",            icon: Settings,        enabled: false, badge: "Yakında", group: "secondary" },
];

// ── Sidebar content ───────────────────────────────────────────────────────────

interface SidebarContentProps {
  pathname: string;
  hash: string;
  onItemClick?: () => void;
}

function NavLink({ item, active, onItemClick }: { item: NavItem; active: boolean; onItemClick?: () => void }) {
  const Icon = item.icon;

  const inner = (
    <span
      className={cn(
        "group relative flex items-center gap-3 rounded-xl px-3 py-3",
        "transition-all duration-200 ease-swift-out",
        !item.enabled && "opacity-40 cursor-not-allowed",
        item.enabled && active
          ? "bg-brand-600/15 text-brand-400"
          : item.enabled
          ? "text-foreground-secondary hover:bg-white/[0.055] hover:text-foreground"
          : "text-foreground-muted",
      )}
    >
      {/* Active indicator bar */}
      {item.enabled && active && (
        <motion.span
          layoutId="sidebar-active-bar"
          aria-hidden
          className="absolute left-0 top-1/2 -translate-y-1/2 h-5 w-0.5 rounded-r-full bg-brand-400"
          transition={{ type: "spring", stiffness: 500, damping: 40 }}
        />
      )}

      <Icon
        className={cn(
          "h-4 w-4 shrink-0 transition-colors duration-200",
          active && item.enabled ? "text-brand-400" : "text-foreground-muted group-hover:text-foreground-secondary",
        )}
      />

      <span className={cn("flex-1 truncate text-sm", active && item.enabled ? "font-semibold" : "font-medium")}>
        {item.label}
      </span>

      {item.badge && (
        <span className="rounded-full bg-surface-raised px-2 py-0.5 text-2xs font-medium text-foreground-muted border border-border-subtle">
          {item.badge}
        </span>
      )}

      {item.enabled && !item.badge && active && (
        <ChevronRight className="h-3 w-3 shrink-0 text-brand-500" />
      )}
    </span>
  );

  if (!item.enabled) return inner;

  return (
    <Link href={item.href} onClick={onItemClick}>
      {inner}
    </Link>
  );
}

// Returns the single nav item that should be marked active.
// Hash-bearing items (/medical#assistant) only match when the URL hash matches.
// Plain items (/medical) only match when no hash-bearing item already matched.
// First-match priority ensures the layoutId bar has exactly one owner.
function resolveActiveItem(pathname: string, hash: string): NavItem | null {
  const all = [...PRIMARY_ITEMS, ...SECONDARY_ITEMS];

  // Pass 1: exact path + hash match (for items like /medical#assistant)
  const hashMatch = all.find((item) => {
    if (!item.enabled) return false;
    const sep = item.href.indexOf("#");
    if (sep === -1) return false;
    const itemPath = item.href.slice(0, sep);
    const itemHash = "#" + item.href.slice(sep + 1);
    return (pathname === itemPath || pathname.startsWith(itemPath + "/")) && hash === itemHash;
  });
  if (hashMatch) return hashMatch;

  // Pass 2: first plain-path match (items without a # in href)
  return all.find((item) => {
    if (!item.enabled) return false;
    if (item.href.includes("#")) return false;
    return pathname === item.href || pathname.startsWith(item.href + "/");
  }) ?? null;
}

function SidebarContent({ pathname, hash, onItemClick }: SidebarContentProps) {
  const activeItem = resolveActiveItem(pathname, hash);

  return (
    <div className="flex h-full flex-col">
      {/* Logo */}
      <div className="relative flex items-center gap-3 px-4 py-6">
        {/* Ambient radial glow behind logo */}
        <div
          aria-hidden
          className="pointer-events-none absolute -left-2 -top-2 h-24 w-24 rounded-full blur-2xl"
          style={{ background: "hsl(221 83% 53%)", opacity: 0.10 }}
        />
        <div className="relative flex h-9 w-9 shrink-0 items-center justify-center rounded-xl overflow-hidden">
          <div className="absolute inset-0 brand-gradient" />
          <ShieldCheck className="relative z-10 h-5 w-5 text-white" />
          <div className="absolute inset-0 noise-overlay" />
        </div>
        <div className="min-w-0">
          <p className="truncate text-sm font-bold tracking-tight text-foreground">
            NEURAVA
          </p>
          <p className="text-2xs font-medium uppercase tracking-widest text-foreground-muted">
            Hanta AI Platform
          </p>
        </div>
      </div>

      <div className="mx-4 section-divider" />

      {/* Primary nav */}
      <nav className="flex-1 space-y-0.5 px-3 py-3">
        <p className="mb-2 px-3 text-2xs font-semibold uppercase tracking-widest text-foreground-muted">
          Analiz
        </p>
        {PRIMARY_ITEMS.map((item) => (
          <NavLink
            key={`${item.href}-${item.label}`}
            item={item}
            active={activeItem === item}
            onItemClick={onItemClick}
          />
        ))}

        <div className="py-2">
          <div className="mx-0 section-divider" />
        </div>

        <p className="mb-2 px-3 text-2xs font-semibold uppercase tracking-widest text-foreground-muted">
          Platform
        </p>
        {SECONDARY_ITEMS.map((item) => (
          <NavLink
            key={`${item.href}-${item.label}`}
            item={item}
            active={activeItem === item}
            onItemClick={onItemClick}
          />
        ))}
      </nav>

      {/* Footer */}
      <div className="space-y-3 px-4 py-4">
        <div className="section-divider" />
        <div className="relative overflow-hidden rounded-xl bg-surface-raised border border-border-subtle px-3.5 py-3.5 space-y-2.5">
          {/* Top-edge accent */}
          <div
            className="absolute inset-x-0 top-0 h-px"
            style={{ background: "linear-gradient(90deg, transparent 0%, hsl(221 83% 53% / 0.5) 50%, transparent 100%)" }}
          />
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <div className="h-1.5 w-1.5 rounded-full bg-success-500 animate-pulse-slow" />
              <span className="text-xs font-semibold text-foreground-secondary">Sistem aktif</span>
            </div>
            <span className="text-2xs font-medium text-success-500 bg-success-50/50 border border-success-200/40 rounded-full px-2 py-0.5">Canlı</span>
          </div>
          <div className="space-y-1 pt-0.5">
            <div className="flex items-center justify-between">
              <span className="text-2xs text-foreground-muted">Model</span>
              <span className="text-2xs font-medium text-foreground-secondary tabular-nums">EfficientNet-B0 v6</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-2xs text-foreground-muted">Stage</span>
              <span className="text-2xs font-medium text-foreground-secondary">C · ECE 0.036</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-2xs text-foreground-muted">Recall</span>
              <span className="text-2xs font-medium text-foreground-secondary tabular-nums">98.8%</span>
            </div>
          </div>
          <div className="pt-1 border-t border-border-subtle flex items-center gap-1.5">
            <Bot className="h-3 w-3 text-brand-400" />
            <span className="text-2xs text-foreground-muted">AI Asistan</span>
            <div className="ml-auto flex items-center gap-1">
              <div className="h-1 w-1 rounded-full bg-success-400" />
              <span className="text-2xs text-success-500">Hazır</span>
            </div>
          </div>
        </div>
        <p className="px-1 text-2xs text-foreground-muted">
          Tıbbi teşhis niteliği taşımaz
        </p>
      </div>
    </div>
  );
}

// ── Sidebar shell ─────────────────────────────────────────────────────────────

interface SidebarProps {
  mobileOpen?: boolean;
  onMobileClose?: () => void;
}

export function Sidebar({ mobileOpen = false, onMobileClose }: SidebarProps) {
  const pathname = usePathname();
  const [hash, setHash] = useState("");

  // Sync hash on mount, on pathname change, and on browser hash changes
  useEffect(() => {
    setHash(window.location.hash);
    const onHashChange = () => setHash(window.location.hash);
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, [pathname]);

  useEffect(() => {
    if (typeof document === "undefined") return;
    document.body.style.overflow = mobileOpen ? "hidden" : "";
    return () => { document.body.style.overflow = ""; };
  }, [mobileOpen]);

  return (
    <>
      {/* Desktop sidebar */}
      <aside className="relative z-10 hidden w-[240px] shrink-0 flex-col border-r border-border-subtle md:flex"
        style={{ background: "hsl(222 40% 5%)" }}>
        <SidebarContent pathname={pathname} hash={hash} />
      </aside>

      {/* Mobile drawer */}
      <div
        className={cn(
          "fixed inset-0 z-50 md:hidden",
          mobileOpen ? "pointer-events-auto" : "pointer-events-none",
        )}
        aria-hidden={!mobileOpen}
      >
        <div
          onClick={onMobileClose}
          className={cn(
            "absolute inset-0 bg-black/60 backdrop-blur-sm transition-opacity duration-300",
            mobileOpen ? "opacity-100" : "opacity-0",
          )}
        />
        <aside
          className={cn(
            "absolute left-0 top-0 flex h-full w-[260px] flex-col border-r border-border-subtle",
            "transition-transform duration-300 ease-swift-out shadow-card-raised",
            mobileOpen ? "translate-x-0" : "-translate-x-full",
          )}
          style={{ background: "hsl(222 40% 5%)" }}
        >
          <button
            type="button"
            onClick={onMobileClose}
            aria-label="Menüyü kapat"
            className="absolute right-3 top-3 flex h-8 w-8 items-center justify-center rounded-lg text-foreground-muted hover:bg-white/[0.055] hover:text-foreground transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
          <SidebarContent pathname={pathname} hash={hash} onItemClick={onMobileClose} />
        </aside>
      </div>
    </>
  );
}
