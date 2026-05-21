"use client";

import { motion } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Clock,
  TrendingUp,
} from "lucide-react";
import {
  Area,
  AreaChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
} from "recharts";

import { useDashboardSummary } from "@/hooks/use-dashboard-summary";
import { useCountUp } from "@/hooks/use-count-up";
import { cn } from "@/lib/utils";

// ── Helpers ───────────────────────────────────────────────────────────────────

const TR_DAYS = ["Paz", "Pzt", "Sal", "Çar", "Per", "Cum", "Cmt"] as const;

function isoToTrDay(dateStr: string): string {
  // dateStr is "YYYY-MM-DD" — parse at noon to avoid UTC-shift edge cases
  return TR_DAYS[new Date(dateStr + "T12:00:00").getDay()] ?? dateStr;
}

// ── KPI card ──────────────────────────────────────────────────────────────────

interface KpiDef {
  icon: React.ElementType;
  label: string;
  sub: string;
  value: number;
  suffix: string;
  decimals: number;
  glowHue: string;
  iconColor: string;
  delay: number;
  primary?: boolean;
}

function KpiCard({ def }: { def: KpiDef }) {
  const count   = useCountUp(def.value, { duration: 1200, decimals: def.decimals });
  const display = def.decimals > 0
    ? count.toFixed(def.decimals)
    : String(Math.round(count));

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: def.delay, ease: [0.2, 0, 0, 1] }}
      className={cn(
        "relative overflow-hidden rounded-2xl border",
        def.primary
          ? "glass-elevated p-6 border-brand-500/25 hover:border-brand-500/45 hover:shadow-[0_14px_48px_-8px_hsl(221_83%_53%/0.28)]"
          : "glass-card-light p-5 border-border-subtle hover:border-border hover:shadow-[0_8px_32px_-8px_rgba(0,0,0,0.5)]",
        "transition-all duration-300 cursor-default",
      )}
    >
      {/* Top-edge accent line */}
      <div
        className="absolute inset-x-0 top-0 h-px"
        style={{
          background: `linear-gradient(90deg, transparent 0%, hsl(${def.glowHue} / ${def.primary ? "0.85" : "0.65"}) 50%, transparent 100%)`,
        }}
      />

      {/* Ambient corner glow */}
      <div
        aria-hidden
        className="pointer-events-none absolute -right-4 -top-4 h-20 w-20 rounded-full blur-xl"
        style={{ background: `hsl(${def.glowHue})`, opacity: def.primary ? 0.18 : 0.12 }}
      />

      <div className="relative flex items-start justify-between">
        <div
          className={cn(
            "flex items-center justify-center rounded-xl",
            def.primary ? "h-10 w-10" : "h-9 w-9",
          )}
          style={{ background: `hsl(${def.glowHue} / ${def.primary ? "0.20" : "0.15"})` }}
        >
          <def.icon
            className={def.primary ? "h-5 w-5" : "h-4 w-4"}
            style={{ color: def.iconColor }}
          />
        </div>
        <TrendingUp className="h-3.5 w-3.5 text-foreground-muted/40" />
      </div>

      <div className="relative mt-3">
        <div className="flex items-baseline gap-1">
          <p className={cn(
            "font-bold tabular-nums tracking-tight text-foreground",
            def.primary ? "text-4xl" : "text-[1.75rem]",
          )}>
            {display}
          </p>
          {def.suffix && (
            <span className="text-sm font-medium text-foreground-muted">{def.suffix}</span>
          )}
        </div>
        <p className="mt-0.5 text-xs font-medium text-foreground-secondary">{def.label}</p>
        <p className="mt-0.5 text-2xs text-foreground-muted">{def.sub}</p>
      </div>
    </motion.div>
  );
}

// ── Recharts tooltip ──────────────────────────────────────────────────────────

function DarkTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ value: number }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl border border-border-subtle bg-surface-raised px-3.5 py-2.5 shadow-[0_8px_32px_-4px_rgba(0,0,0,0.6)]">
      <p className="text-2xs text-foreground-muted">{label}</p>
      <p className="text-sm font-bold text-foreground">{payload[0]?.value} analiz</p>
    </div>
  );
}

// ── Dashboard widgets ─────────────────────────────────────────────────────────

export function DashboardWidgets() {
  const summary = useDashboardSummary();

  // ── KPI values from real data ────────────────────────────────────────────
  const avgConfPct  = summary.average_confidence != null
    ? Math.round(summary.average_confidence * 1000) / 10   // 0.9710 → 97.1
    : 0;
  const avgDurSec   = summary.average_duration_seconds ?? 0;

  const kpiData: KpiDef[] = [
    {
      icon: Activity,
      label: "Toplam Analiz",
      sub: "Tüm zamanlar",
      value: summary.total_analyses,
      suffix: "",
      decimals: 0,
      glowHue: "221 83% 53%",
      iconColor: "hsl(217 88% 65%)",
      delay: 0,
      primary: true,
    },
    {
      icon: AlertTriangle,
      label: "Yüksek Risk",
      sub: "Son 30 gün",
      value: summary.high_risk_count,
      suffix: "",
      decimals: 0,
      glowHue: "0 75% 55%",
      iconColor: "hsl(0 80% 70%)",
      delay: 0.06,
    },
    {
      icon: CheckCircle2,
      label: "Ort. Güven",
      sub: "Kalibrasyon",
      value: avgConfPct,
      suffix: "%",
      decimals: 1,
      glowHue: "152 60% 40%",
      iconColor: "hsl(152 65% 55%)",
      delay: 0.12,
    },
    {
      icon: Clock,
      label: "Ort. Süre",
      sub: "API yanıt",
      value: avgDurSec,
      suffix: "s",
      decimals: 1,
      glowHue: "38 90% 55%",
      iconColor: "hsl(38 90% 70%)",
      delay: 0.18,
    },
  ];

  // ── Weekly trend ─────────────────────────────────────────────────────────
  const trendData = summary.weekly_trend.map(({ date, count }) => ({
    day: isoToTrDay(date),
    count,
  }));
  const weekTotal = trendData.reduce((s, d) => s + d.count, 0);

  // ── Risk distribution ────────────────────────────────────────────────────
  const { low, moderate, high, critical } = summary.risk_distribution;
  const distTotal = low + moderate + high + critical;
  const pct = (n: number) =>
    distTotal > 0 ? Math.round((n / distTotal) * 100) : 0;

  const riskDistFull = [
    { name: "Düşük",  value: pct(low),      color: "hsl(152 65% 48%)" },
    { name: "Orta",   value: pct(moderate),  color: "hsl(38 90% 58%)"  },
    { name: "Yüksek", value: pct(high),      color: "hsl(0 80% 62%)"   },
    { name: "Kritik", value: pct(critical),  color: "hsl(0 78% 45%)"   },
  ];

  // Use a single muted segment when there is no data so the donut renders
  const riskDist = distTotal > 0
    ? riskDistFull
    : [{ name: "", value: 1, color: "hsl(220 14% 22%)" }];

  return (
    <div className="space-y-4">

      {/* KPI row */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        {kpiData.map((def) => (
          <KpiCard key={def.label} def={def} />
        ))}
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">

        {/* Trend chart */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.24, ease: [0.2, 0, 0, 1] }}
          className="sm:col-span-2 relative overflow-hidden rounded-2xl glass-card-light border border-border-subtle p-5"
        >
          {/* Bottom ambient glow */}
          <div
            aria-hidden
            className="pointer-events-none absolute bottom-0 left-1/2 -translate-x-1/2 h-16 w-3/4 rounded-full blur-2xl"
            style={{ background: "hsl(221 83% 53%)", opacity: 0.11 }}
          />

          <div className="mb-4 flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-foreground">Haftalık Analiz Trendi</p>
              <p className="text-2xs text-foreground-muted mt-0.5">Son 7 gün</p>
            </div>
            <div className="flex items-center gap-1.5 rounded-full border border-border-subtle bg-surface px-2.5 py-1">
              <BarChart3 className="h-3 w-3 text-foreground-muted" />
              <span className="text-2xs font-semibold text-foreground-secondary">
                {weekTotal} analiz
              </span>
            </div>
          </div>

          <div className="h-44">
            {trendData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={trendData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="trendGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%"   stopColor="hsl(221 83% 53%)" stopOpacity={0.48} />
                      <stop offset="50%"  stopColor="hsl(221 83% 53%)" stopOpacity={0.10} />
                      <stop offset="100%" stopColor="hsl(221 83% 53%)" stopOpacity={0}    />
                    </linearGradient>
                  </defs>
                  <XAxis
                    dataKey="day"
                    tick={{ fontSize: 10, fill: "hsl(var(--foreground-muted))" }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip
                    content={<DarkTooltip />}
                    cursor={{ stroke: "rgba(99,130,255,0.18)", strokeWidth: 1 }}
                  />
                  <Area
                    type="monotone"
                    dataKey="count"
                    stroke="hsl(221 83% 62%)"
                    strokeWidth={2.5}
                    fill="url(#trendGrad)"
                    dot={false}
                    activeDot={{ r: 4.5, fill: "hsl(217 88% 70%)", strokeWidth: 0 }}
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full items-center justify-center">
                <p className="text-2xs text-foreground-muted">Henüz analiz verisi yok</p>
              </div>
            )}
          </div>
        </motion.div>

        {/* Risk distribution */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.30, ease: [0.2, 0, 0, 1] }}
          className="rounded-2xl glass-card-light border border-border-subtle p-5"
        >
          <p className="text-sm font-semibold text-foreground">Risk Dağılımı</p>
          <p className="text-2xs text-foreground-muted mb-3 mt-0.5">Tüm zamanlar</p>

          <div className="flex items-center justify-between gap-2">
            <div className="h-28 w-28 shrink-0">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={riskDist}
                    cx="50%"
                    cy="50%"
                    innerRadius={28}
                    outerRadius={44}
                    paddingAngle={distTotal > 0 ? 2 : 0}
                    dataKey="value"
                    strokeWidth={0}
                  >
                    {riskDist.map((entry, index) => (
                      <Cell key={index} fill={entry.color} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="flex-1 space-y-2">
              {distTotal > 0 ? (
                riskDistFull.map((entry) => (
                  <div key={entry.name} className="flex items-center justify-between gap-2">
                    <div className="flex min-w-0 items-center gap-1.5">
                      <div
                        className="h-1.5 w-1.5 shrink-0 rounded-full"
                        style={{ background: entry.color }}
                      />
                      <span className="truncate text-2xs text-foreground-muted">{entry.name}</span>
                    </div>
                    <span className="shrink-0 text-2xs font-bold tabular-nums text-foreground-secondary">
                      {entry.value}%
                    </span>
                  </div>
                ))
              ) : (
                <p className="text-2xs text-foreground-muted">Veri yok</p>
              )}
            </div>
          </div>
        </motion.div>

      </div>
    </div>
  );
}
