"use client";

import { motion } from "framer-motion";
import {
  Activity,
  ArrowRight,
  Bot,
  Brain,
  CheckCircle2,
  ExternalLink,
  Server,
  ShieldCheck,
  Zap,
} from "lucide-react";
import Link from "next/link";

import { DashboardWidgets } from "@/components/dashboard/widgets";
import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { useAuth } from "@/hooks/use-auth";
import { useBackendHealth } from "@/hooks/use-backend-health";
import { useDashboardSummary } from "@/hooks/use-dashboard-summary";
import { cn } from "@/lib/utils";

// ── Types ─────────────────────────────────────────────────────────────────────

type RiskTier = "LOW" | "MODERATE" | "HIGH_DIFFERENTIAL_RISK" | "CRITICAL_PULMONARY_RISK";

const TIER_META: Record<RiskTier, { label: string; badgeClass: string; scoreClass: string }> = {
  LOW: {
    label: "Düşük Risk",
    badgeClass: "bg-success-50/80 text-success-500 border-success-200/60",
    scoreClass: "text-success-500",
  },
  MODERATE: {
    label: "Orta Risk",
    badgeClass: "bg-warning-50/80 text-warning-500 border-warning-200/60",
    scoreClass: "text-warning-500",
  },
  HIGH_DIFFERENTIAL_RISK: {
    label: "Yüksek Risk",
    badgeClass: "bg-danger-50/80 text-danger-500 border-danger-200/60",
    scoreClass: "text-danger-500",
  },
  CRITICAL_PULMONARY_RISK: {
    label: "Kritik Risk",
    badgeClass: "bg-danger-100/80 text-danger-600 border-danger-300/60",
    scoreClass: "text-danger-500",
  },
};

const KNOWN_TIERS = new Set<string>(Object.keys(TIER_META));

function isRiskTier(tier: string): tier is RiskTier {
  return KNOWN_TIERS.has(tier);
}

// ── Relative time formatting (Turkish) ───────────────────────────────────────

function formatRelativeTr(isoString: string): string {
  const diffMs = Date.now() - new Date(isoString).getTime();
  const mins   = Math.floor(diffMs / 60_000);
  if (mins < 1)   return "az önce";
  if (mins < 60)  return `${mins} dakika önce`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours} saat önce`;
  const days  = Math.floor(hours / 24);
  if (days === 1) return "Dün";
  return `${days} gün önce`;
}

// ── MetricTile ─────────────────────────────────────────────────────────────────

function MetricTile({
  label,
  value,
  sub,
  good = false,
}: {
  label: string;
  value: string;
  sub: string;
  good?: boolean;
}) {
  return (
    <div className="rounded-xl border border-border-subtle bg-canvas/60 px-3.5 py-3">
      <p className="text-2xs text-foreground-muted">{label}</p>
      <p className={cn(
        "mt-0.5 text-[0.9375rem] font-bold tabular-nums leading-tight",
        good ? "text-success-400" : "text-foreground",
      )}>
        {value}
      </p>
      <p className="mt-0.5 text-2xs text-foreground-muted/60">{sub}</p>
    </div>
  );
}

// ── Live AI intelligence panel (hero right) ───────────────────────────────────

function LiveAIPanel({ health }: { health: ReturnType<typeof useBackendHealth> }) {
  const rows = [
    {
      label: "Backend API",
      value: health.isLoading ? "Kontrol…" : health.isOnline ? "Sağlıklı" : "Çevrimdışı",
      ok: !health.isLoading && health.isOnline,
    },
    { label: "AI Asistan",    value: "Hazır",     ok: true  },
    { label: "Analiz Motoru", value: "Beklemede", ok: false },
  ] as const;

  return (
    <motion.div
      initial={{ opacity: 0, x: 14, scale: 0.98 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      transition={{ duration: 0.62, delay: 0.16, ease: [0.2, 0, 0, 1] }}
      className="relative flex flex-col gap-5 overflow-hidden rounded-2xl border border-border-subtle glass-elevated p-6"
    >
      {/* Top accent line */}
      <div
        className="absolute inset-x-0 top-0 h-px"
        style={{ background: "linear-gradient(90deg, transparent, hsl(221 83% 53% / 0.55), transparent)" }}
      />
      {/* Ambient glow */}
      <div
        aria-hidden
        className="pointer-events-none absolute -right-10 -top-10 h-36 w-36 rounded-full blur-3xl"
        style={{ background: "hsl(221 83% 53%)", opacity: 0.13 }}
      />

      {/* Header */}
      <div className="relative flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-brand-100/40">
            <Brain className="h-3.5 w-3.5 text-brand-400" />
          </div>
          <span className="text-xs font-bold uppercase tracking-wider text-foreground-secondary">
            AI Altyapı
          </span>
        </div>
        <span className="flex items-center gap-1.5 rounded-full border border-success-200/40 bg-success-50/20 px-2.5 py-1 text-2xs font-semibold text-success-400">
          <span className="h-1.5 w-1.5 rounded-full bg-success-500 animate-pulse-slow" />
          Canlı
        </span>
      </div>

      {/* 2×2 metrics grid */}
      <div className="relative grid grid-cols-2 gap-2">
        <MetricTile label="Model"       value="EfficientNet-B0" sub="v6 · Stage C" />
        <MetricTile label="ECE Skoru"   value="0.036"           sub="Kalibrasyon"  good />
        <MetricTile label="Recall"      value="98.8%"           sub="Pnömoni"      good />
        <MetricTile label="Ort. Güven"  value="97.1%"           sub="Precision"    good />
      </div>

      {/* Service status rows */}
      <div className="relative space-y-2.5 border-t border-border-subtle pt-3">
        {rows.map(({ label, value, ok }) => (
          <div key={label} className="flex items-center justify-between">
            <span className="text-2xs text-foreground-muted">{label}</span>
            <span className={cn(
              "text-2xs font-semibold",
              ok ? "text-success-400" : "text-foreground-secondary",
            )}>
              {value}
            </span>
          </div>
        ))}
      </div>

      {/* Micro CTA */}
      <Link
        href="/medical"
        className="relative flex items-center justify-between rounded-xl border border-brand-200/30 bg-brand-50/10 px-4 py-2.5 transition-colors duration-200 hover:border-brand-300/45 hover:bg-brand-50/20"
      >
        <span className="text-xs font-semibold text-brand-400">Yeni analiz başlat</span>
        <ArrowRight className="h-3.5 w-3.5 text-brand-400" />
      </Link>
    </motion.div>
  );
}

// ── Status chip ───────────────────────────────────────────────────────────────

function StatusChip({
  icon: Icon,
  label,
  value,
  active = false,
  delay = 0,
}: {
  icon: React.ElementType;
  label: string;
  value: string;
  active?: boolean;
  delay?: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 6, scale: 0.94 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.38, delay, ease: [0.2, 0, 0, 1] }}
      className={cn(
        "flex items-center gap-2 rounded-full border px-3.5 py-1.5 glass-card",
        active
          ? "border-success-200/40 bg-success-50/10"
          : "border-border-subtle bg-canvas/40",
      )}
    >
      {active && <span className="h-1.5 w-1.5 rounded-full bg-success-500 animate-pulse-slow" />}
      <Icon className={cn("h-3 w-3", active ? "text-success-400" : "text-foreground-muted")} />
      <span className="text-2xs text-foreground-muted">{label}</span>
      <span className={cn(
        "text-2xs font-semibold",
        active ? "text-success-400" : "text-foreground-secondary",
      )}>
        {value}
      </span>
    </motion.div>
  );
}

// ── Analysis row ──────────────────────────────────────────────────────────────

function AnalysisRow({
  item,
  index,
}: {
  item: { id: string; risk_tier: string; risk_score: number; created_at: string };
  index: number;
}) {
  const tier = isRiskTier(item.risk_tier) ? item.risk_tier : "LOW";
  const meta = TIER_META[tier];

  return (
    <motion.div
      initial={{ opacity: 0, x: -6 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3, delay: 0.06 + index * 0.05, ease: [0.2, 0, 0, 1] }}
      className="group flex items-center gap-4 rounded-xl px-4 py-4 transition-colors duration-200 hover:bg-white/[0.05]"
    >
      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-border-subtle bg-surface">
        <Activity className="h-4 w-4 text-brand-400" />
      </div>

      <div className="min-w-0 flex-1">
        <p className="truncate font-mono text-xs font-medium text-foreground-secondary">
          {item.id.slice(0, 8)}
        </p>
        <p className="mt-0.5 text-2xs text-foreground-muted">
          {formatRelativeTr(item.created_at)}
        </p>
      </div>

      <span className={cn(
        "hidden shrink-0 rounded-full border px-2.5 py-0.5 text-2xs font-semibold sm:inline-flex",
        meta.badgeClass,
      )}>
        {meta.label}
      </span>

      <div className="w-12 shrink-0 text-right">
        <p className={cn("text-sm font-bold tabular-nums", meta.scoreClass)}>
          {(item.risk_score * 100).toFixed(1)}
        </p>
        <p className="text-2xs text-foreground-muted/60">/ 100</p>
      </div>

      <Link
        href="/medical"
        className={cn(
          "flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border",
          "border-border-subtle bg-surface text-foreground-muted",
          "opacity-0 transition-all duration-150 group-hover:opacity-100",
          "hover:border-brand-300/50 hover:text-brand-400",
        )}
      >
        <ExternalLink className="h-3.5 w-3.5" />
      </Link>
    </motion.div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const { user }   = useAuth();
  const health     = useBackendHealth();
  const summary    = useDashboardSummary();

  const displayName = user?.username || user?.email?.split("@")[0] || "Kullanıcı";
  const recentCount = summary.recent_analyses.length;

  return (
    <div className="space-y-8 pb-24">

      {/* ── Hero — two-column on lg+ ──────────────────────────────────────── */}
      <div className="grid grid-cols-1 items-start gap-8 pt-7 lg:grid-cols-[1fr_340px]">

        {/* LEFT: identity + title + chips */}
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.65, ease: [0.2, 0, 0, 1] }}
          className="space-y-7"
        >
          <div className="flex items-center gap-2.5">
            <div className="h-0.5 w-6 rounded-full bg-brand-500/60" />
            <span className="text-xs font-semibold uppercase tracking-widest text-brand-400">
              Hoş geldiniz, {displayName}
            </span>
          </div>

          <div className="space-y-4">
            <h1 className="text-[2.75rem] font-bold leading-[1.12] tracking-tight gradient-text-brand sm:text-5xl lg:text-[3.5rem]">
              NEURAVA Pulmoner<br className="hidden sm:block" /> Risk Analizi
            </h1>
            <p className="max-w-[52ch] text-[0.9375rem] leading-loose text-foreground-secondary sm:text-base">
              EfficientNet-B0 tabanlı çok modlu AI sistemi. Görüntü ve klinik veriyi
              birleştirerek anlık pulmoner risk değerlendirmesi üretir.
            </p>
          </div>

          <div className="flex flex-wrap gap-2">
            <StatusChip icon={Zap}          label="AI Motoru"   value="Aktif"              active delay={0.12} />
            <StatusChip icon={CheckCircle2} label="Model"       value="EfficientNet-B0 v6"        delay={0.19} />
            <StatusChip icon={CheckCircle2} label="Kalibrasyon" value="ECE 0.036"                 delay={0.26} />
            <StatusChip
              icon={Server}
              label="Backend"
              value={health.isLoading ? "—" : health.isOnline ? "Sağlıklı" : "Çevrimdışı"}
              active={!health.isLoading && health.isOnline}
              delay={0.33}
            />
          </div>
        </motion.div>

        {/* RIGHT: live AI intelligence panel */}
        <LiveAIPanel health={health} />
      </div>

      {/* ── Offline alert ─────────────────────────────────────────────────── */}
      {!health.isOnline && !health.isLoading && (
        <Alert variant="danger" title="Sunucu erişilemiyor">
          Arka uç servisine ulaşılamıyor. Lütfen birkaç saniye sonra tekrar deneyin.
        </Alert>
      )}

      {/* ── Primary CTA ───────────────────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.52, delay: 0.12, ease: [0.2, 0, 0, 1] }}
      >
        <Link
          href="/medical"
          className={cn(
            "group relative block overflow-hidden rounded-2xl border border-brand-200/40",
            "px-8 py-8 sm:py-11 glass-card",
            "shadow-[0_4px_32px_-8px_hsl(221_83%_53%/0.24)]",
            "hover:border-brand-300/65 hover:shadow-[0_12px_48px_-8px_hsl(221_83%_53%/0.42)]",
            "transition-all duration-300",
          )}
        >
          {/* Primary top-right glow */}
          <div aria-hidden className="pointer-events-none absolute -right-12 -top-12 h-80 w-80 rounded-full blur-3xl"
            style={{ background: "hsl(221 83% 53%)", opacity: 0.30 }} />
          {/* Secondary bottom-left glow */}
          <div aria-hidden className="pointer-events-none absolute -bottom-10 -left-10 h-48 w-48 rounded-full blur-3xl"
            style={{ background: "hsl(221 83% 53%)", opacity: 0.10 }} />
          {/* Tertiary center pulse — very subtle */}
          <div aria-hidden className="pointer-events-none absolute left-1/2 top-1/2 h-24 w-24 -translate-x-1/2 -translate-y-1/2 rounded-full blur-2xl"
            style={{ background: "hsl(221 83% 65%)", opacity: 0.06 }} />

          <div className="relative z-10 flex items-center gap-7">
            <div className={cn(
              "flex h-[4.5rem] w-[4.5rem] shrink-0 items-center justify-center rounded-2xl",
              "bg-brand-100/60 ring-1 ring-brand-200/50",
              "group-hover:bg-brand-100/80 group-hover:ring-brand-300/65",
              "shadow-[0_6px_24px_-6px_hsl(221_83%_53%/0.45)] transition-all duration-300",
            )}>
              <ShieldCheck className="h-9 w-9 text-brand-500" />
            </div>

            <div className="min-w-0 flex-1">
              <div className="mb-2 flex flex-wrap items-center gap-2">
                <p className="text-xl font-bold text-foreground">Yeni Analiz Başlat</p>
                <span className="rounded-full border border-brand-200/40 bg-brand-50/50 px-2.5 py-0.5 text-2xs font-semibold text-brand-400">
                  Görüntü + Klinik
                </span>
              </div>
              <p className="text-sm leading-relaxed text-foreground-secondary">
                Akciğer grafisi yükleyin veya klinik bulgular girin — AI sistemi anında risk değerlendirmesi üretir.
              </p>
            </div>

            <ArrowRight className="h-5 w-5 shrink-0 text-brand-400 transition-transform duration-300 group-hover:translate-x-2" />
          </div>
        </Link>
      </motion.div>

      {/* ── KPI cards + charts ────────────────────────────────────────────── */}
      <DashboardWidgets />

      {/* ── Bottom enterprise grid ─────────────────────────────────────────── */}
      <div className="grid grid-cols-1 gap-7 xl:grid-cols-[1fr_380px]">

        {/* Recent analyses — activity feed */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.52, delay: 0.22, ease: [0.2, 0, 0, 1] }}
          className="overflow-hidden rounded-2xl border border-border-subtle glass-card-light"
        >
          <div className="flex items-center justify-between border-b border-border-subtle px-5 py-4">
            <div>
              <p className="text-sm font-semibold text-foreground">Son Analizler</p>
              <p className="mt-0.5 text-2xs text-foreground-muted">
                {summary.isLoading ? "Yükleniyor…" : "Son 5 kayıt · Gerçek veri"}
              </p>
            </div>
            <span className="rounded-full border border-border-subtle px-2.5 py-0.5 text-2xs font-medium text-foreground-muted">
              {recentCount} kayıt
            </span>
          </div>

          <div className="divide-y divide-border-subtle/40 px-1 py-1">
            {summary.recent_analyses.length === 0 ? (
              <p className="px-5 py-8 text-center text-sm text-foreground-muted">
                Henüz analiz bulunmuyor.
              </p>
            ) : (
              summary.recent_analyses.map((item, i) => (
                <AnalysisRow key={item.id} item={item} index={i} />
              ))
            )}
          </div>

          <div className="flex items-center justify-between border-t border-border-subtle bg-canvas/30 px-5 py-3">
            <p className="text-2xs text-foreground-muted">
              {summary.total_analyses > 5
                ? `${summary.total_analyses} toplam analiz`
                : "Tüm analizler gösteriliyor"}
            </p>
            <Link
              href="/medical"
              className="text-2xs font-semibold text-brand-400 transition-colors hover:text-brand-300"
            >
              Yeni analiz başlat →
            </Link>
          </div>
        </motion.div>

        {/* System status — enterprise panel */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.52, delay: 0.32, ease: [0.2, 0, 0, 1] }}
          className="relative overflow-hidden rounded-2xl border border-border-subtle glass-card-light p-7"
        >
          {/* Top-edge accent */}
          <div
            className="absolute inset-x-0 top-0 h-px"
            style={{ background: "linear-gradient(90deg, transparent, hsl(221 83% 53% / 0.35), transparent)" }}
          />

          <div className="flex items-center gap-2">
            <Server className="h-4 w-4 text-foreground-muted" />
            <p className="text-sm font-semibold text-foreground">Sistem Durumu</p>
            <Badge
              variant={health.isLoading ? "neutral" : health.isOnline ? "success" : "danger"}
              className="ml-auto"
            >
              <span className={cn(
                "h-1.5 w-1.5 rounded-full",
                health.isLoading
                  ? "bg-foreground-muted"
                  : health.isOnline
                  ? "bg-success-500 animate-pulse-slow"
                  : "bg-danger-500",
              )} />
              {health.isLoading ? "Kontrol ediliyor…" : health.isOnline ? "Çevrim içi" : "Çevrim dışı"}
            </Badge>
          </div>

          <dl className="mt-5 grid grid-cols-1 gap-3.5">
            <div className="rounded-lg bg-canvas px-3.5 py-2.5">
              <dt className="text-xs font-medium text-foreground-muted">Sürüm</dt>
              <dd className="mt-1 font-semibold text-foreground">
                {summary.system_status.version !== "—"
                  ? `v${summary.system_status.version}`
                  : health.data?.version
                    ? `v${health.data.version}`
                    : "—"}
              </dd>
            </div>
            <div className="rounded-lg bg-canvas px-3.5 py-2.5">
              <dt className="text-xs font-medium text-foreground-muted">Son Kontrol</dt>
              <dd className="mt-1 font-semibold text-foreground">
                {health.lastCheckedAt
                  ? new Date(health.lastCheckedAt).toLocaleTimeString("tr-TR")
                  : "—"}
              </dd>
            </div>
            <div className="rounded-lg bg-canvas px-3.5 py-2.5">
              <dt className="text-xs font-medium text-foreground-muted">AI Modeli</dt>
              <dd className="mt-1 font-semibold text-foreground">
                {summary.system_status.model !== "—"
                  ? summary.system_status.model
                  : "EfficientNet-B0 v6"}
              </dd>
            </div>
          </dl>

          {/* AI assistant quick-link */}
          <Link
            href="/medical#assistant"
            className="mt-6 flex items-center justify-between rounded-xl border border-border-subtle bg-surface px-4 py-3.5 transition-colors duration-200 hover:border-brand-300/30 hover:bg-white/[0.05]"
          >
            <div className="flex items-center gap-2.5">
              <Bot className="h-4 w-4 text-brand-400" />
              <div>
                <p className="text-xs font-semibold text-foreground-secondary">AI Asistan</p>
                <p className="text-2xs text-foreground-muted">Klinisyen sorguları için</p>
              </div>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-success-400" />
              <ArrowRight className="h-3.5 w-3.5 text-foreground-muted" />
            </div>
          </Link>
        </motion.div>
      </div>

      {/* ── Medical disclaimer ────────────────────────────────────────────── */}
      <Alert variant="warning" title="Klinisyen Uyarısı">
        Bu platform bir karar destek aracıdır. Üretilen sonuçlar tıbbi teşhis yerine geçmez;
        mutlaka bir hekim tarafından değerlendirilmelidir.
      </Alert>

    </div>
  );
}
