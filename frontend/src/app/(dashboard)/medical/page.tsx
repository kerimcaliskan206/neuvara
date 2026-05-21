"use client";

import { useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion } from "framer-motion";
import { Activity, AlertTriangle, Check, ChevronDown, ChevronUp, RefreshCw, Scan, Shield, Stethoscope } from "lucide-react";
import { Fragment, ReactNode, useEffect, useRef, useState } from "react";

import {
  ClinicalModifiersPanel,
  FusionBreakdownPanel,
  ImagingAnalysisPanel,
  MedicalAssistantPanel,
  OodRejectionPanel,
  SemanticReasoningPanel,
  TrustCalibrationPanel,
  UploadZone,
} from "@/components/medical";
import type { ClinicalContextRequest, MedicalAnalysisContext, MedicalRiskTier, UnifiedAnalysisSession } from "@/lib/api/types";
import { useCountUp } from "@/hooks/use-count-up";
import { useMedicalAnalysis } from "@/hooks/use-medical-analysis";
import { buildAssistantContext, medicalApi } from "@/lib/api/medical";
import { computeClinicalRisk, type ClinicalAnalysisResult } from "@/lib/clinical-analysis";
import { cn } from "@/lib/utils";

// ── Analysis pipeline stages ──────────────────────────────────────────────────

const IMAGING_STAGES = [
  "Akciğer segmentasyonu analiz ediliyor...",
  "Patolojik bölgeler değerlendiriliyor...",
  "GradCAM odak haritası oluşturuluyor...",
  "Klinik risk skoru hesaplanıyor...",
  "AI reasoning tamamlanıyor...",
];

// Minimum visible analysis duration (ms). Backend may finish earlier;
// the result is held until this wall-clock time has elapsed.
const MIN_ANALYSIS_MS = 5500;

const CLINICAL_STAGES = [
  "Hasta verileri işleniyor",
  "Semptom profili oluşturuluyor",
  "Klinik risk faktörleri analiz ediliyor",
  "Risk skoru hesaplanıyor",
  "Sonuçlar sentezleniyor",
];

// ── Symptom / exposure label maps ─────────────────────────────────────────────

const SYMPTOM_LABELS: Record<string, string> = {
  fever:               "Ateş",
  cough:               "Öksürük",
  dyspnea:             "Dispne",
  shortness_of_breath: "Nefes darlığı",
  chest_pain:          "Göğüs ağrısı",
  hemoptysis:          "Hemoptizi",
  tachypnea:           "Takipne",
  hypoxia:             "Hipoksi",
  fatigue:             "Yorgunluk",
  myalgia:             "Miyalji",
  night_sweats:        "Gece terlemesi",
  weight_loss:         "Kilo kaybı",
  wheezing:            "Hışıltı",
  productive_cough:    "Balgamlı öksürük",
};

const EXPOSURE_LABELS: Record<string, string> = {
  rodent_contact:    "Kemirgen teması",
  hospital:          "Hastane maruziyeti",
  sick_contact:      "Hasta ile temas",
  travel:            "Seyahat öyküsü",
  healthcare_worker: "Sağlık çalışanı",
  immunocompromised: "İmmün yetmezlik",
};

// ── Phase 22 display label maps (clinical-only summary) ───────────────────────

const RESPIRATORY_LABELS: Record<string, string> = {
  normal: "Normal nefes alabiliyorum",
  mild:   "Nefes almak biraz zorlaştı",
  severe: "Nefes almak ciddi şekilde zor",
};
const OXYGENATION_LABELS: Record<string, string> = {
  normal:      "Günlük nefesim normal",
  mild_drop:   "Nefes kapasitem azaldı",
  severe_drop: "Dinlenirken bile nefes almak zor",
};
const FEVER_LABELS: Record<string, string> = {
  none: "Ateş yok", mild: "Hafif ateş", moderate: "Orta ateş", high: "Yüksek ateş",
};
const WORSENING_LABELS: Record<string, string> = {
  none: "Kötüleşme yok", some: "Son günlerde kötüleşme", rapid_48h: "Son 48 saatte hızlı kötüleşme",
};
const DURATION_LABELS: Record<string, string> = {
  "1_2_days": "1–2 gün", "3_7_days": "3–7 gün", over_1_week: "1 haftadan uzun",
};
const RODENT_LABELS: Record<string, string> = {
  none: "Bilinen maruziyet yok", unsure: "Emin değilim",
  rural_env: "Kırsal/depo ortamı", possible_contact: "Kemirgen teması olabilir",
};
const SEX_LABELS: Record<string, string> = { male: "Erkek", female: "Kadın" };

// ── Hero risk tier config ─────────────────────────────────────────────────────

const HERO_TIER: Record<MedicalRiskTier, {
  label: string; sublabel: string;
  scoreColor: string; labelColor: string;
  glowHue: string; borderStyle: string; barBg: string;
  pulse: boolean;
}> = {
  LOW: {
    label: "DÜŞÜK RİSK",
    sublabel: "Görüntüde belirgin patolojik bulgu saptanmadı.",
    scoreColor: "hsl(152 65% 55%)", labelColor: "hsl(152 65% 60%)",
    glowHue: "152 65% 48%", borderStyle: "hsl(152 65% 48% / 0.22)",
    barBg: "hsl(152 65% 48%)", pulse: false,
  },
  MODERATE: {
    label: "ORTA RİSK",
    sublabel: "Hafif pulmoner değişiklik izleniyor; klinik değerlendirme önerilir.",
    scoreColor: "hsl(38 90% 60%)", labelColor: "hsl(38 90% 64%)",
    glowHue: "38 90% 55%", borderStyle: "hsl(38 90% 55% / 0.24)",
    barBg: "hsl(38 90% 55%)", pulse: false,
  },
  HIGH_DIFFERENTIAL_RISK: {
    label: "YÜKSEK RİSK",
    sublabel: "Belirgin pulmoner anormallik tespit edildi; uzman değerlendirmesi önerilir.",
    scoreColor: "hsl(0 78% 65%)", labelColor: "hsl(0 78% 68%)",
    glowHue: "0 78% 58%", borderStyle: "hsl(0 78% 58% / 0.28)",
    barBg: "hsl(0 78% 58%)", pulse: false,
  },
  CRITICAL_PULMONARY_RISK: {
    label: "KRİTİK RİSK",
    sublabel: "Ciddi bilateral pulmoner tutulum; acil klinik değerlendirme gereklidir.",
    scoreColor: "hsl(0 78% 65%)", labelColor: "hsl(0 78% 68%)",
    glowHue: "0 78% 52%", borderStyle: "hsl(0 78% 52% / 0.38)",
    barBg: "hsl(0 78% 52%)", pulse: true,
  },
};

// ── Human bullet generation (API path) ───────────────────────────────────────

const IMAGING_BULLETS: Partial<Record<string, string>> = {
  pneumonia_xray: "Görüntüde pulmoner konsolidasyon veya infiltrat paterni saptandı.",
  healthy_xray:   "Görüntüde belirgin pulmoner anormallik bulgusu izlenmedi.",
  hard_negative:  "Görüntü pulmoner radyoloji içeriği taşımamaktadır.",
  fake_medical:   "Görüntünün tıbbi niteliği doğrulanamadı.",
};

const EXPOSURE_BULLETS: Record<string, string> = {
  rodent_contact:    "Kemirgen teması öyküsü dikkat çekmektedir.",
  hospital:          "Hastane ortamında maruziyet öyküsü mevcut.",
  sick_contact:      "Hasta bireyle yakın temas öyküsü bildirildi.",
  immunocompromised: "İmmün yetmezlik durumu klinik riski artırmaktadır.",
  healthcare_worker: "Sağlık çalışanı olduğu belirtildi.",
  travel:            "Seyahat öyküsü klinik bağlamda değerlendirildi.",
};

// eslint-disable-next-line @typescript-eslint/no-unused-vars
function buildHumanBullets(session: UnifiedAnalysisSession): string[] {
  const bullets: string[] = [];
  const { imaging, clinical, risk, trust } = session;

  // Imaging finding
  const imgBullet = IMAGING_BULLETS[imaging.predicted_class];
  if (imgBullet) bullets.push(imgBullet);

  // CRITICAL tier → bilateral note
  if (risk.risk_tier === "CRITICAL_PULMONARY_RISK") {
    bullets.push("Her iki akciğerde yaygın pulmoner tutulum paterni saptandı.");
  }

  // Clinical signals
  if (clinical.provided) {
    const flags = new Set(clinical.symptoms_flagged);

    if (flags.has("shortness_of_breath") || flags.has("dyspnea")) {
      bullets.push("Solunum sıkıntısı bildirildi.");
    } else if (flags.has("tachypnea")) {
      bullets.push("Hızlı solunum (takipne) bildirildi.");
    }
    if (flags.has("hypoxia")) {
      bullets.push("Düşük oksijen saturasyonu bildirildi.");
    }
    if (flags.has("hemoptysis")) {
      bullets.push("Hemoptizi (kanlı balgam) öyküsü mevcut.");
    }
    if (flags.has("chest_pain")) {
      bullets.push("Göğüs ağrısı semptomu bildirildi.");
    }
    if (flags.has("fever")) {
      bullets.push("Ateş semptomu bildirildi.");
    }
    if (flags.has("weight_loss") || flags.has("night_sweats")) {
      bullets.push("Sistemik semptomlar (gece terlemesi veya kilo kaybı) bildirildi.");
    }

    if (clinical.exposure_flagged) {
      const expBullet = EXPOSURE_BULLETS[clinical.exposure_flagged];
      if (expBullet) bullets.push(expBullet);
    }

    if (clinical.delta_direction === "upward" && clinical.clinical_delta > 0.03) {
      bullets.push("Klinik bulgular görüntü bulgularıyla uyumlu biçimde pulmoner riski desteklemektedir.");
    } else if (clinical.delta_direction === "downward" && clinical.clinical_delta < -0.03) {
      bullets.push("Klinik bulgular mevcut risk düzeyini kısmen sınırlamaktadır.");
    }
  }

  // Near boundary
  if (risk.near_boundary) {
    bullets.push("Risk düzeyi iki kategori sınırına yakın; klinik takip önerilmektedir.");
  }

  // Low trust → human note (no calibration terms)
  if (trust.trust_tier === "uncertain" || trust.trust_tier === "suspicious") {
    bullets.push("Görüntü kalitesi veya içerik belirsizliği nedeniyle güvenilirlik sınırlıdır; klinisyen değerlendirmesi önerilir.");
  }

  return bullets;
}

// ── API path: Premium result components ──────────────────────────────────────

function HeroRiskSection({ session }: { session: UnifiedAnalysisSession }) {
  const cfg = HERO_TIER[session.risk.risk_tier];
  const scoreVal = useCountUp(session.risk.final_score * 100, { duration: 1100, decimals: 1 });

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.55, ease: [0.2, 0, 0, 1] }}
      className="relative overflow-hidden rounded-2xl border"
      style={{
        borderColor: cfg.borderStyle,
        background: "hsl(222 45% 5%)",
        boxShadow: `0 0 80px -20px hsl(${cfg.glowHue} / 0.30), inset 0 0 0 1px hsl(${cfg.glowHue} / 0.07)`,
      }}
    >
      {/* Ambient glow orb */}
      <motion.div
        className="pointer-events-none absolute left-1/2 top-0 -translate-x-1/2"
        style={{
          width: "500px",
          height: "260px",
          background: `radial-gradient(ellipse at 50% 0%, hsl(${cfg.glowHue} / 0.18), transparent 65%)`,
        }}
        animate={cfg.pulse ? { opacity: [0.8, 1.15, 0.8] } : {}}
        transition={cfg.pulse ? { duration: 2.4, repeat: Infinity, ease: "easeInOut" } : {}}
      />

      <div className="relative z-10 flex flex-col gap-5 px-8 py-7 sm:flex-row sm:items-center sm:justify-between">
        {/* Left: tier label + sublabel */}
        <div className="space-y-2">
          <motion.p
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.12 }}
            className="text-[2.8rem] font-bold leading-none tracking-tight"
            style={{ color: cfg.labelColor }}
          >
            {cfg.label}
          </motion.p>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.28 }}
            className="max-w-sm text-sm leading-relaxed text-foreground-secondary"
          >
            {cfg.sublabel}
          </motion.p>

          {session.risk.near_boundary && (
            <motion.span
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3, delay: 0.4 }}
              className="inline-flex items-center gap-1.5 rounded-full border border-warning-200/50 bg-warning-50/15 px-3 py-1 text-xs font-semibold text-warning-500"
            >
              <span className="h-1.5 w-1.5 rounded-full bg-warning-500 animate-pulse" />
              Sınır değerine yakın
            </motion.span>
          )}
        </div>

        {/* Right: animated score */}
        <div className="flex flex-col items-end gap-2">
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.08, ease: [0.2, 0, 0.4, 1.1] }}
            className="flex items-end leading-none"
            style={{ color: cfg.scoreColor }}
          >
            <span className="text-[5.5rem] font-bold tabular-nums tracking-tight leading-none">
              {scoreVal.toFixed(1)}
            </span>
            <span className="mb-3 ml-1 text-2xl font-semibold opacity-70">/100</span>
          </motion.div>

          {/* Score bar */}
          <div className="relative h-2 w-52 overflow-hidden rounded-full bg-white/5">
            {[35, 60, 80].map((pct) => (
              <div
                key={pct}
                className="absolute top-0 h-full w-px bg-white/15"
                style={{ left: `${pct}%` }}
              />
            ))}
            <motion.div
              className="absolute inset-y-0 left-0 rounded-full"
              style={{ background: cfg.barBg, boxShadow: `0 0 12px 3px hsl(${cfg.glowHue} / 0.4)` }}
              initial={{ width: "0%" }}
              animate={{ width: `${session.risk.final_score * 100}%` }}
              transition={{ duration: 1.1, delay: 0.2, ease: [0.4, 0, 0.2, 1] }}
            />
          </div>
          <div className="flex w-52 justify-between text-[9.5px] text-foreground-muted/40">
            <span>0</span><span>35</span><span>60</span><span>80</span><span>100</span>
          </div>
        </div>
      </div>

      {/* Immediate action banner */}
      {session.risk.requires_immediate_action && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          transition={{ duration: 0.4, delay: 0.5 }}
          className="relative z-10 flex items-center gap-3 border-t border-danger-300/20 bg-danger-500/10 px-8 py-3.5"
        >
          <AlertTriangle className="h-4 w-4 shrink-0 text-danger-400 animate-pulse" />
          <p className="text-sm font-semibold text-danger-400">
            Acil klinik değerlendirme gereklidir — lütfen bir sağlık kuruluşuna başvurun.
          </p>
        </motion.div>
      )}
    </motion.div>
  );
}

function PremiumImageViewer({
  previewUrl,
  gradcamBase64,
  targetClass,
}: {
  previewUrl: string;
  gradcamBase64?: string | null;
  targetClass?: string | null;
}) {
  const [mode, setMode] = useState<"original" | "gradcam">("gradcam");
  const hasGradcam = !!gradcamBase64;
  const activeUrl = mode === "gradcam" && hasGradcam
    ? `data:image/png;base64,${gradcamBase64}`
    : previewUrl;

  return (
    <div
      className="relative overflow-hidden rounded-2xl border border-brand-500/18 animate-fade-up"
      style={{ background: "hsl(222 45% 4%)", boxShadow: "0 0 60px -16px hsl(221 83% 53% / 0.2)" }}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3.5">
        <div className="flex items-center gap-2">
          <div className="h-1.5 w-1.5 rounded-full bg-brand-400/70" />
          <p className="text-sm font-semibold text-foreground">Görüntü Kanıtı</p>
          {targetClass && (
            <span className="rounded-md border border-brand-200/25 bg-brand-50/8 px-2 py-0.5 text-2xs font-mono text-brand-400/70">
              {targetClass}
            </span>
          )}
        </div>

        {hasGradcam && (
          <div className="flex rounded-lg border border-border/50 bg-surface p-0.5 text-xs font-medium">
            {(["original", "gradcam"] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className={cn(
                  "rounded-md px-3 py-1 transition-all duration-200",
                  mode === m
                    ? "bg-brand-500 text-white shadow-sm"
                    : "text-foreground-secondary hover:text-foreground",
                )}
              >
                {m === "original" ? "Orijinal" : "AI Odak"}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Image area */}
      <div
        className="relative mx-5 mb-5 overflow-hidden rounded-xl"
        style={{ background: "hsl(0 0% 0% / 0.4)" }}
      >
        <div className="pointer-events-none absolute inset-3 z-10">
          {LOADING_SCAN_CORNERS.map((pos, i) => (
            <div key={i} className={cn("absolute h-6 w-6 border-brand-400/30", pos)} />
          ))}
        </div>

        <AnimatePresence mode="wait">
          <motion.img
            key={mode}
            src={activeUrl}
            alt={mode === "gradcam" ? "GradCAM AI Odak Haritası" : "Orijinal Akciğer Grafisi"}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.35, ease: "easeInOut" }}
            className="w-full object-contain"
            style={{ maxHeight: "460px", minHeight: "260px" }}
          />
        </AnimatePresence>

        <div className="absolute bottom-3 left-3 z-10 flex items-center gap-1.5 rounded-lg border border-brand-400/20 bg-canvas/75 px-2.5 py-1 backdrop-blur-sm">
          {mode === "gradcam" && (
            <span className="h-1.5 w-1.5 rounded-full bg-brand-400 animate-pulse" />
          )}
          <span className="text-2xs font-medium text-foreground-secondary">
            {mode === "gradcam" ? "GradCAM · AI Aktivasyon Haritası" : "Orijinal Görüntü"}
          </span>
        </div>
      </div>
    </div>
  );
}

function InsightCard({
  label,
  value,
  sub,
  icon,
  accent,
  delay = 0,
}: {
  label: string;
  value: string;
  sub?: string;
  icon: ReactNode;
  accent: "success" | "warning" | "danger";
  delay?: number;
}) {
  const accentStyles = {
    success: { border: "border-success-400/25", bg: "bg-success-500/8", icon: "text-success-400" },
    warning: { border: "border-warning-400/25", bg: "bg-warning-500/8", icon: "text-warning-400" },
    danger:  { border: "border-danger-400/25",  bg: "bg-danger-500/8",  icon: "text-danger-400"  },
  }[accent];

  return (
    <motion.div
      initial={{ opacity: 0, x: 10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4, delay, ease: [0.2, 0, 0, 1] }}
      className={cn(
        "flex items-start gap-3 rounded-xl border p-4",
        accentStyles.border, accentStyles.bg,
      )}
    >
      <div className={cn("mt-0.5 shrink-0", accentStyles.icon)}>{icon}</div>
      <div className="min-w-0">
        <p className="text-2xs font-semibold uppercase tracking-wider text-foreground-muted">{label}</p>
        <p className="mt-0.5 text-sm font-semibold text-foreground leading-snug">{value}</p>
        {sub && <p className="mt-0.5 text-xs text-foreground-secondary leading-snug">{sub}</p>}
      </div>
    </motion.div>
  );
}

const PREDICTED_CLASS_LABELS: Record<string, string> = {
  pneumonia_xray: "Pulmoner Konsolidasyon",
  healthy_xray:   "Normal Akciğer Grafisi",
  hard_negative:  "Patolojik Bulgu Yok",
  fake_medical:   "Tıbbi Görüntü Değil",
};

const TRUST_TIER_LABELS: Record<string, string> = {
  very_high_trust: "Çok Yüksek Güven",
  high_trust:      "Yüksek Güven",
  moderate_trust:  "Orta Güven",
  uncertain:       "Düşük Güven",
  suspicious:      "Şüpheli",
};

function InsightPanel({ session }: { session: UnifiedAnalysisSession }) {
  const { imaging, trust, explainability, risk } = session;
  const hasGradcam = !!explainability.gradcam_base64;
  const tier = risk.risk_tier;

  const clinicalRec =
    tier === "CRITICAL_PULMONARY_RISK" ? "Acil klinik değerlendirme gereklidir." :
    tier === "HIGH_DIFFERENTIAL_RISK"  ? "Uzman radyolog/pnömolog değerlendirmesi önerilir." :
    tier === "MODERATE"                ? "Klinik takip ve kontrol görüntüleme önerilir." :
                                         "Rutin klinik izleme yeterlidir.";

  const insightAccent = (t: MedicalRiskTier): "success" | "warning" | "danger" =>
    t === "LOW" ? "success" : t === "MODERATE" ? "warning" : "danger";

  return (
    <div className="flex h-full flex-col gap-3">
      <InsightCard
        label="Tespit Edilen Bölge"
        value={PREDICTED_CLASS_LABELS[imaging.predicted_class] ?? imaging.predicted_class}
        icon={<Scan className="h-4 w-4" />}
        accent={insightAccent(tier)}
        delay={0.1}
      />
      <InsightCard
        label="AI Güveni"
        value={`%${Math.round(trust.trust_score * 100)}`}
        sub={TRUST_TIER_LABELS[trust.trust_tier] ?? trust.trust_tier}
        icon={<Activity className="h-4 w-4" />}
        accent={
          trust.trust_tier === "very_high_trust" || trust.trust_tier === "high_trust"
            ? "success"
            : trust.trust_tier === "moderate_trust"
            ? "warning"
            : "danger"
        }
        delay={0.18}
      />
      <InsightCard
        label="Lokalizasyon"
        value={hasGradcam ? "GradCAM Aktivasyon Haritası" : "Sınıflandırma Tabanlı"}
        sub={hasGradcam ? "Anatomik odak bölgesi belirlendi" : "Görsel lokalizasyon mevcut değil"}
        icon={<Shield className="h-4 w-4" />}
        accent={hasGradcam ? "success" : "warning"}
        delay={0.26}
      />
      <InsightCard
        label="Klinik Öneri"
        value={clinicalRec}
        icon={<Stethoscope className="h-4 w-4" />}
        accent={insightAccent(tier)}
        delay={0.34}
      />
    </div>
  );
}

// ── Structured insight items for the reasoning timeline ──────────────────────

interface InsightItem {
  primary: string;
  secondary: string;
  category: "imaging" | "clinical" | "correlation" | "trust" | "boundary";
}

function buildInsightItems(session: UnifiedAnalysisSession): InsightItem[] {
  const items: InsightItem[] = [];
  const { imaging, clinical, risk, trust } = session;

  const imgBullet = IMAGING_BULLETS[imaging.predicted_class];
  if (imgBullet) {
    items.push({
      primary: imgBullet,
      secondary: "Görüntü sınıflandırma modelinin birincil tespiti.",
      category: "imaging",
    });
  }

  if (risk.risk_tier === "CRITICAL_PULMONARY_RISK") {
    items.push({
      primary: "Her iki akciğerde yaygın pulmoner tutulum paterni saptandı.",
      secondary: "Bilateral tutulum kritik risk katmanını destekler.",
      category: "imaging",
    });
  }

  if (clinical.provided) {
    const flags = new Set(clinical.symptoms_flagged);

    if (flags.has("shortness_of_breath") || flags.has("dyspnea")) {
      items.push({
        primary: "Solunum sıkıntısı bildirildi.",
        secondary: "Dispne varlığı görüntü bulgularıyla birlikte değerlendirildi.",
        category: "clinical",
      });
    } else if (flags.has("tachypnea")) {
      items.push({
        primary: "Hızlı solunum (takipne) bildirildi.",
        secondary: "Takipne klinik ağırlık faktörü olarak işlendi.",
        category: "clinical",
      });
    }
    if (flags.has("hypoxia")) {
      items.push({
        primary: "Düşük oksijen saturasyonu bildirildi.",
        secondary: "Hipoksi varlığı risk skorunu olumsuz etkilemektedir.",
        category: "clinical",
      });
    }
    if (flags.has("hemoptysis")) {
      items.push({
        primary: "Hemoptizi (kanlı balgam) öyküsü mevcut.",
        secondary: "Hemoptizi ciddi pulmoner patoloji için güçlü bir göstergedir.",
        category: "clinical",
      });
    }
    if (flags.has("chest_pain")) {
      items.push({
        primary: "Göğüs ağrısı semptomu bildirildi.",
        secondary: "Pulmoner ve kardiyak köken açısından bağlamsal değerlendirme yapıldı.",
        category: "clinical",
      });
    }
    if (flags.has("fever")) {
      items.push({
        primary: "Ateş semptomu bildirildi.",
        secondary: "Ateş enfeksiyöz patoloji açısından anlamlı bir klinik bulgudur.",
        category: "clinical",
      });
    }
    if (flags.has("weight_loss") || flags.has("night_sweats")) {
      items.push({
        primary: "Sistemik semptomlar (gece terlemesi veya kilo kaybı) bildirildi.",
        secondary: "Sistemik bulgular kronik patoloji veya malignite açısından değerlendirildi.",
        category: "clinical",
      });
    }
    if (clinical.exposure_flagged) {
      const expBullet = EXPOSURE_BULLETS[clinical.exposure_flagged];
      if (expBullet) {
        items.push({
          primary: expBullet,
          secondary: "Maruziyet öyküsü klinik risk bağlamında işlendi.",
          category: "clinical",
        });
      }
    }
    if (clinical.delta_direction === "upward" && clinical.clinical_delta > 0.03) {
      items.push({
        primary: "Klinik bulgular görüntü bulgularıyla uyumlu biçimde pulmoner riski desteklemektedir.",
        secondary: "Görüntü–klinik korelasyonu pozitif yönde; risk skoru yukarı güncellendi.",
        category: "correlation",
      });
    } else if (clinical.delta_direction === "downward" && clinical.clinical_delta < -0.03) {
      items.push({
        primary: "Klinik bulgular mevcut risk düzeyini kısmen sınırlamaktadır.",
        secondary: "Görüntü–klinik uyumsuzluğu negatif düzeltme uyguladı.",
        category: "correlation",
      });
    }
  }

  if (risk.near_boundary) {
    items.push({
      primary: "Risk düzeyi iki kategori sınırına yakın.",
      secondary: "Sınır bölgesi tespiti; klinik takip ve kontrol görüntüleme önerilmektedir.",
      category: "boundary",
    });
  }

  if (trust.trust_tier === "uncertain" || trust.trust_tier === "suspicious") {
    items.push({
      primary: "Model güvenilirliği sınırlı.",
      secondary: "Görüntü kalitesi veya içerik belirsizliği nedeniyle klinisyen değerlendirmesi önerilir.",
      category: "trust",
    });
  }

  return items;
}

// ── AI Clinical Decision Layer ────────────────────────────────────────────────

function AIReasoningSection({ session }: { session: UnifiedAnalysisSession }) {
  const exp    = session.explainability;
  const tier   = session.risk.risk_tier;
  const cfg    = HERO_TIER[tier];
  const items  = buildInsightItems(session);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.15, ease: [0.2, 0, 0, 1] }}
      className="relative overflow-hidden rounded-2xl border"
      style={{
        background: "hsl(222 45% 5%)",
        borderColor: cfg.borderStyle,
        boxShadow: `0 0 70px -22px hsl(${cfg.glowHue} / 0.24), inset 0 0 0 1px hsl(${cfg.glowHue} / 0.05)`,
      }}
    >
      {/* Ambient top glow */}
      <div
        className="pointer-events-none absolute inset-x-0 top-0 h-36"
        style={{
          background: `radial-gradient(ellipse at 28% 0%, hsl(${cfg.glowHue} / 0.14), transparent 60%)`,
        }}
      />

      {/* Animated left accent bar */}
      <motion.div
        className="absolute left-0 top-8 bottom-8 w-[2px] rounded-full"
        style={{
          background: `linear-gradient(180deg, transparent, hsl(${cfg.glowHue}), transparent)`,
        }}
        initial={{ scaleY: 0, opacity: 0 }}
        animate={{ scaleY: 1, opacity: 1 }}
        transition={{ duration: 0.75, delay: 0.3, ease: [0.4, 0, 0.2, 1] }}
      />

      <div className="relative z-10 space-y-7 px-8 py-7">

        {/* Header */}
        <div className="space-y-1.5">
          <motion.p
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.38, delay: 0.1 }}
            className="text-[10px] font-bold uppercase tracking-[0.15em]"
            style={{ color: cfg.labelColor }}
          >
            AI KLİNİK DEĞERLENDİRME
          </motion.p>
          <motion.h3
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.16 }}
            className="text-xl font-bold tracking-tight text-foreground"
          >
            Klinik Karar Katmanı
          </motion.h3>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.4, delay: 0.24 }}
            className="text-xs leading-relaxed text-foreground-muted"
          >
            Yapay zeka görüntü bulgularını klinik bağlamda yorumladı.
          </motion.p>
        </div>

        {/* Summary callout — left-border accent */}
        <motion.div
          initial={{ opacity: 0, x: -6 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.45, delay: 0.28, ease: [0.2, 0, 0, 1] }}
          className="relative rounded-xl border bg-white/[0.025] p-4 pl-5"
          style={{
            borderColor: `hsl(${cfg.glowHue} / 0.2)`,
            boxShadow: `inset 3px 0 0 hsl(${cfg.glowHue})`,
          }}
        >
          <p className="text-sm font-medium leading-relaxed text-foreground">{exp.summary}</p>
        </motion.div>

        {/* Reasoning timeline */}
        {items.length > 0 && (
          <div>
            <p
              className="mb-4 text-[10px] font-semibold uppercase tracking-[0.13em]"
              style={{ color: cfg.labelColor, opacity: 0.6 }}
            >
              Akıl Yürütme Zinciri
            </p>

            <div className="relative space-y-1 pl-7">
              {/* Vertical connecting line */}
              <motion.div
                className="absolute left-[10px] top-4 bottom-4 w-px"
                style={{
                  background: `linear-gradient(180deg, hsl(${cfg.glowHue} / 0.40), hsl(${cfg.glowHue} / 0.04))`,
                }}
                initial={{ scaleY: 0 }}
                animate={{ scaleY: 1 }}
                transition={{ duration: 0.65, delay: 0.42, ease: [0.4, 0, 0.2, 1] }}
              />

              {items.map((item, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.38, delay: 0.4 + i * 0.10, ease: [0.2, 0, 0, 1] }}
                  className="group relative rounded-xl px-4 py-3 transition-colors duration-200 hover:bg-white/[0.03]"
                >
                  {/* Timeline node */}
                  <div
                    className="absolute -left-7 top-[13px] flex h-[20px] w-[20px] items-center justify-center rounded-full border"
                    style={{
                      borderColor: `hsl(${cfg.glowHue} / 0.45)`,
                      background:  `hsl(${cfg.glowHue} / 0.12)`,
                    }}
                  >
                    <Check
                      className="h-[11px] w-[11px]"
                      style={{ color: cfg.labelColor }}
                    />
                  </div>

                  <p className="text-sm font-semibold leading-snug text-foreground">
                    {item.primary}
                  </p>
                  <p className="mt-0.5 text-xs leading-relaxed text-foreground-muted">
                    {item.secondary}
                  </p>
                </motion.div>
              ))}
            </div>
          </div>
        )}

        {/* AI Tespit Özeti */}
        {exp.imaging_findings && (
          <motion.div
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
              duration: 0.4,
              delay: 0.5 + items.length * 0.06,
              ease: [0.2, 0, 0, 1],
            }}
            className="overflow-hidden rounded-xl border border-border/50"
            style={{ background: "hsl(222 45% 4%)" }}
          >
            <div
              className="flex items-center gap-2.5 border-b border-border/40 px-4 py-2.5"
              style={{ background: `hsl(${cfg.glowHue} / 0.06)` }}
            >
              <div
                className="h-[3px] w-5 rounded-full"
                style={{ background: `hsl(${cfg.glowHue})` }}
              />
              <p className="text-[10px] font-bold uppercase tracking-[0.13em] text-foreground-muted">
                AI Tespit Özeti
              </p>
            </div>
            <p className="px-4 py-4 text-sm leading-relaxed text-foreground-secondary">
              {exp.imaging_findings}
            </p>
          </motion.div>
        )}

        {/* Contradiction note */}
        {exp.contradiction_note && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.35, delay: 0.6 }}
            className="flex items-start gap-3 rounded-xl border border-warning-300/20 bg-warning-500/8 px-4 py-3.5"
          >
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning-400" />
            <div>
              <p className="mb-0.5 text-xs font-semibold text-warning-400">Dikkat</p>
              <p className="text-xs leading-relaxed text-warning-500/80">
                {exp.contradiction_note}
              </p>
            </div>
          </motion.div>
        )}

        {/* Disclaimer */}
        <div className="border-t border-white/[0.06] pt-4">
          <p className="text-2xs leading-relaxed text-foreground-muted/45">
            Bu sistem tıbbi teşhis koymaz. Bulgular klinisyen değerlendirmesini destekleme amacıyla üretilmiştir.
          </p>
        </div>
      </div>
    </motion.div>
  );
}

// ── Clinical-only path: premium components ───────────────────────────────────

function ClinicalHeroSection({ result }: { result: ClinicalAnalysisResult }) {
  const cfg = HERO_TIER[result.risk.risk_tier];
  const scoreVal = useCountUp(result.risk.final_score * 100, { duration: 1100, decimals: 1 });

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.55, ease: [0.2, 0, 0, 1] }}
      className="relative overflow-hidden rounded-2xl border"
      style={{
        borderColor: cfg.borderStyle,
        background: "hsl(222 45% 5%)",
        boxShadow: `0 0 80px -20px hsl(${cfg.glowHue} / 0.30), inset 0 0 0 1px hsl(${cfg.glowHue} / 0.07)`,
      }}
    >
      <motion.div
        className="pointer-events-none absolute left-1/2 top-0 -translate-x-1/2"
        style={{
          width: "500px", height: "260px",
          background: `radial-gradient(ellipse at 50% 0%, hsl(${cfg.glowHue} / 0.18), transparent 65%)`,
        }}
        animate={cfg.pulse ? { opacity: [0.8, 1.15, 0.8] } : {}}
        transition={cfg.pulse ? { duration: 2.4, repeat: Infinity, ease: "easeInOut" } : {}}
      />

      {/* Clinical mode badge */}
      <div
        className="relative z-10 flex items-center gap-2 border-b px-8 py-2.5"
        style={{ borderColor: `hsl(${cfg.glowHue} / 0.15)`, background: `hsl(${cfg.glowHue} / 0.06)` }}
      >
        <motion.div
          className="h-1.5 w-1.5 rounded-full"
          style={{ background: cfg.barBg }}
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
        />
        <span className="text-[10px] font-bold uppercase tracking-[0.14em] text-foreground-muted/70">
          Görüntüsüz Klinik Analiz
        </span>
        <span className="ml-auto rounded-md border border-border/40 px-2 py-0.5 text-[10px] font-medium text-foreground-muted/40">
          Akciğer grafisi yok
        </span>
      </div>

      <div className="relative z-10 flex flex-col gap-5 px-8 py-7 sm:flex-row sm:items-center sm:justify-between">
        <div className="space-y-2">
          <motion.p
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.12 }}
            className="text-[2.8rem] font-bold leading-none tracking-tight"
            style={{ color: cfg.labelColor }}
          >
            {cfg.label}
          </motion.p>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.28 }}
            className="max-w-sm text-sm leading-relaxed text-foreground-secondary"
          >
            {cfg.sublabel}
          </motion.p>
          {result.risk.near_boundary && (
            <motion.span
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3, delay: 0.4 }}
              className="inline-flex items-center gap-1.5 rounded-full border border-warning-200/50 bg-warning-50/15 px-3 py-1 text-xs font-semibold text-warning-500"
            >
              <span className="h-1.5 w-1.5 rounded-full bg-warning-500 animate-pulse" />
              Sınır değerine yakın
            </motion.span>
          )}
        </div>

        <div className="flex flex-col items-end gap-2">
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.08, ease: [0.2, 0, 0.4, 1.1] }}
            className="flex items-end leading-none"
            style={{ color: cfg.scoreColor }}
          >
            <span className="text-[5.5rem] font-bold tabular-nums tracking-tight leading-none">
              {scoreVal.toFixed(1)}
            </span>
            <span className="mb-3 ml-1 text-2xl font-semibold opacity-70">/100</span>
          </motion.div>
          <div className="relative h-2 w-52 overflow-hidden rounded-full bg-white/5">
            {[35, 60, 80].map((pct) => (
              <div key={pct} className="absolute top-0 h-full w-px bg-white/15" style={{ left: `${pct}%` }} />
            ))}
            <motion.div
              className="absolute inset-y-0 left-0 rounded-full"
              style={{ background: cfg.barBg, boxShadow: `0 0 12px 3px hsl(${cfg.glowHue} / 0.4)` }}
              initial={{ width: "0%" }}
              animate={{ width: `${result.risk.final_score * 100}%` }}
              transition={{ duration: 1.1, delay: 0.2, ease: [0.4, 0, 0.2, 1] }}
            />
          </div>
          <div className="flex w-52 justify-between text-[9.5px] text-foreground-muted/40">
            <span>0</span><span>35</span><span>60</span><span>80</span><span>100</span>
          </div>
        </div>
      </div>

      {result.requires_immediate_action && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          transition={{ duration: 0.4, delay: 0.5 }}
          className="relative z-10 flex items-center gap-3 border-t border-danger-300/20 bg-danger-500/10 px-8 py-3.5"
        >
          <AlertTriangle className="h-4 w-4 shrink-0 text-danger-400 animate-pulse" />
          <p className="text-sm font-semibold text-danger-400">
            Acil klinik değerlendirme gereklidir — lütfen bir sağlık kuruluşuna başvurun.
          </p>
        </motion.div>
      )}
    </motion.div>
  );
}

function ClinicalVisualization({
  result,
}: {
  result: ClinicalAnalysisResult;
  ctx: ClinicalContextRequest | null;
}) {
  const cfg      = HERO_TIER[result.risk.risk_tier];
  const symptoms = result.symptoms_provided.slice(0, 8);

  // Distribute symptom nodes evenly around a circle
  const nodes = symptoms.map((sym, i, arr) => {
    const angle = (i / arr.length) * 2 * Math.PI - Math.PI / 2;
    const r     = 36 + (i % 2) * 10;
    return { sym, cx: 50 + r * Math.cos(angle), cy: 50 + r * Math.sin(angle), delay: i * 0.13 };
  });

  return (
    <div
      className="relative overflow-hidden rounded-2xl border border-brand-500/18"
      style={{
        background: "hsl(222 45% 4%)",
        boxShadow: `0 0 60px -18px hsl(${cfg.glowHue} / 0.24)`,
        minHeight: "340px",
      }}
    >
      {/* Ambient radial glow */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{ background: `radial-gradient(ellipse at 50% 45%, hsl(${cfg.glowHue} / 0.10), transparent 60%)` }}
      />

      {/* Dot grid */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          backgroundImage: `radial-gradient(circle, hsl(${cfg.glowHue} / 0.06) 1px, transparent 1px)`,
          backgroundSize: "32px 32px",
        }}
      />

      {/* Header */}
      <div className="relative z-10 flex items-center justify-between px-5 pt-4">
        <div className="flex items-center gap-2">
          <motion.div
            className="h-1.5 w-1.5 rounded-full"
            style={{ background: cfg.barBg }}
            animate={{ opacity: [0.45, 1, 0.45] }}
            transition={{ duration: 1.9, repeat: Infinity, ease: "easeInOut" }}
          />
          <span className="text-[10px] font-bold uppercase tracking-[0.14em]" style={{ color: cfg.labelColor }}>
            AI Klinik Analiz Modu
          </span>
        </div>
        <span className="rounded-md border border-border/40 px-2.5 py-1 text-[10px] font-medium text-foreground-muted/45">
          Görüntüsüz
        </span>
      </div>

      {/* Symptom constellation SVG */}
      <div className="relative z-10 flex items-center justify-center" style={{ height: "234px" }}>
        <svg
          viewBox="0 0 100 100"
          className="absolute inset-0 h-full w-full"
          preserveAspectRatio="xMidYMid meet"
        >
          {/* Static concentric rings */}
          {[14, 22, 30, 38].map((r, i) => (
            <circle
              key={r}
              cx="50" cy="50" r={r}
              fill="none"
              stroke={`hsl(${cfg.glowHue})`}
              strokeWidth="0.35"
              strokeOpacity={0.05 + i * 0.04}
            />
          ))}

          {/* Rotating dashed outer ring */}
          <motion.circle
            cx="50" cy="50" r="34"
            fill="none"
            stroke={`hsl(${cfg.glowHue})`}
            strokeWidth="0.55"
            strokeOpacity="0.20"
            strokeDasharray="5 18"
            style={{ transformOrigin: "50px 50px" }}
            animate={{ rotate: 360 }}
            transition={{ duration: 24, repeat: Infinity, ease: "linear" }}
          />

          {/* Counter-rotating inner ring */}
          <motion.circle
            cx="50" cy="50" r="20"
            fill="none"
            stroke={`hsl(${cfg.glowHue})`}
            strokeWidth="0.4"
            strokeOpacity="0.13"
            strokeDasharray="3 12"
            style={{ transformOrigin: "50px 50px" }}
            animate={{ rotate: -360 }}
            transition={{ duration: 34, repeat: Infinity, ease: "linear" }}
          />

          {/* Connector lines from center to symptom nodes */}
          {nodes.map((n, i) => (
            <motion.line
              key={`line-${i}`}
              x1="50" y1="50" x2={n.cx} y2={n.cy}
              stroke={`hsl(${cfg.glowHue})`}
              strokeWidth="0.3"
              strokeOpacity="0.22"
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.55 + n.delay }}
            />
          ))}

          {/* Symptom nodes */}
          {nodes.map((n, i) => (
            <motion.circle
              key={`node-${i}`}
              cx={n.cx} cy={n.cy} r="2.8"
              fill={`hsl(${cfg.glowHue})`}
              fillOpacity="0.65"
              style={{ transformOrigin: `${n.cx}px ${n.cy}px` }}
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: [1, 1.25, 1], opacity: [0.5, 0.9, 0.5] }}
              transition={{
                scale:   { duration: 2.2 + i * 0.18, repeat: Infinity, ease: "easeInOut", delay: 0.6 + n.delay },
                opacity: { duration: 2.2 + i * 0.18, repeat: Infinity, ease: "easeInOut", delay: 0.6 + n.delay },
              }}
            />
          ))}
        </svg>

        {/* Center activity icon */}
        <div className="relative z-10 flex flex-col items-center gap-1.5 text-center">
          <motion.div
            className="mb-1 flex h-11 w-11 items-center justify-center rounded-full border"
            style={{
              borderColor: `hsl(${cfg.glowHue} / 0.45)`,
              background:   `hsl(${cfg.glowHue} / 0.12)`,
            }}
            animate={{
              boxShadow: [
                `0 0 0 0px hsl(${cfg.glowHue} / 0.35)`,
                `0 0 0 10px hsl(${cfg.glowHue} / 0)`,
              ],
            }}
            transition={{ duration: 2.2, repeat: Infinity, ease: "easeOut" }}
          >
            <Activity className="h-5 w-5" style={{ color: cfg.labelColor }} />
          </motion.div>
          <p className="text-[11px] font-bold tabular-nums tracking-wide" style={{ color: cfg.labelColor }}>
            {symptoms.length} Semptom
          </p>
          <p className="text-[9.5px] text-foreground-muted/45 tracking-wide">analiz edildi</p>
        </div>
      </div>

      {/* EKG waveform */}
      <div className="relative z-10 px-5">
        <svg width="100%" height="34" viewBox="0 0 300 34" preserveAspectRatio="none">
          <motion.path
            d="M0,17 L18,17 L28,7 L36,27 L44,3 L52,31 L58,17 L80,17 L88,11 L96,23 L102,17 L128,17 L138,7 L146,27 L154,3 L162,31 L168,17 L198,17 L208,11 L216,23 L222,17 L258,17 L268,7 L276,27 L284,3 L292,31 L300,17"
            fill="none"
            stroke={`hsl(${cfg.glowHue})`}
            strokeWidth="1.3"
            strokeOpacity="0.40"
            strokeLinecap="round"
            strokeLinejoin="round"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 1 }}
            transition={{ duration: 2.0, delay: 0.5, ease: "easeInOut" }}
          />
        </svg>
      </div>

      {/* Upgrade prompt footer */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.4, delay: 0.9 }}
        className="relative z-10 mt-2 flex items-center gap-2.5 border-t border-white/[0.05] px-5 py-3"
        style={{ background: `hsl(${cfg.glowHue} / 0.04)` }}
      >
        <div className="h-1 w-1 rounded-full" style={{ background: cfg.barBg }} />
        <p className="text-xs text-foreground-muted/55">
          Akciğer grafisi eklenerek görüntü analizi ile derinleştirilebilir.
        </p>
      </motion.div>
    </div>
  );
}

function ClinicalInsightPanel({
  result,
  ctx,
}: {
  result: ClinicalAnalysisResult;
  ctx: ClinicalContextRequest | null;
}) {
  const tier = result.risk.risk_tier;
  const cfg  = HERO_TIER[tier];

  const severityLabel =
    result.severity_provided
      ? ({ mild: "Hafif", moderate: "Orta", severe: "Ağır" } as Record<string, string>)[result.severity_provided] ?? result.severity_provided
      : ctx?.respiratory_severity
      ? RESPIRATORY_LABELS[ctx.respiratory_severity] ?? ctx.respiratory_severity
      : "Belirtilmedi";

  const imagingNeed =
    tier === "CRITICAL_PULMONARY_RISK" ? "Acil görüntüleme gerekli" :
    tier === "HIGH_DIFFERENTIAL_RISK"  ? "Görüntüleme şiddetle önerilir" :
    tier === "MODERATE"                ? "Görüntüleme önerilir" :
                                         "Rutin takip yeterli";

  const accent = (t: MedicalRiskTier): "success" | "warning" | "danger" =>
    t === "LOW" ? "success" : t === "MODERATE" ? "warning" : "danger";

  return (
    <div className="flex h-full flex-col gap-3">
      <InsightCard
        label="Değerlendirilen Semptom"
        value={`${result.symptoms_provided.length} semptom`}
        sub={result.exposure_provided
          ? EXPOSURE_LABELS[result.exposure_provided] ?? result.exposure_provided
          : "Maruziyet öyküsü yok"}
        icon={<Stethoscope className="h-4 w-4" />}
        accent={accent(tier)}
        delay={0.10}
      />
      <InsightCard
        label="Klinik Ağırlık"
        value={severityLabel}
        sub={result.duration_days != null ? `${result.duration_days} gündür süren semptomlar` : undefined}
        icon={<Activity className="h-4 w-4" />}
        accent={accent(tier)}
        delay={0.18}
      />
      <InsightCard
        label="Görüntüleme İhtiyacı"
        value={imagingNeed}
        sub="Klinik risk düzeyine göre belirlendi"
        icon={<Scan className="h-4 w-4" />}
        accent={tier === "LOW" ? "success" : tier === "MODERATE" ? "warning" : "danger"}
        delay={0.26}
      />
      <InsightCard
        label="Risk Kategorisi"
        value={cfg.label}
        sub={cfg.sublabel}
        icon={<Shield className="h-4 w-4" />}
        accent={accent(tier)}
        delay={0.34}
      />
    </div>
  );
}

function buildClinicalInsightItems(
  result: ClinicalAnalysisResult,
): InsightItem[] {
  return result.reasoning_bullets.map((bullet): InsightItem => {
    const b = bullet.toLowerCase();
    let secondary = "AI klinik değerlendirme çıktısı.";
    if (b.includes("nefes") || b.includes("dispne") || b.includes("takipne"))
      secondary = "Solunum fonksiyonu klinik ağırlık hesaplamasında dikkate alındı.";
    else if (b.includes("ateş"))
      secondary = "Enfeksiyöz etiyoloji açısından anlamlı klinik bulgu.";
    else if (b.includes("oksijen") || b.includes("hipoksi"))
      secondary = "Oksijen saturasyonu düşüklüğü ciddi patoloji göstergesidir.";
    else if (b.includes("göğüs"))
      secondary = "Kardiyak ve pulmoner nedenler bağlamsal olarak değerlendirildi.";
    else if (b.includes("hemoptizi"))
      secondary = "Pulmoner tutulum için güçlü klinik işaret.";
    else if (b.includes("maruziyet") || b.includes("temas"))
      secondary = "Epidemiyolojik risk faktörü olarak işlendi.";
    else if (b.includes("sınır") || b.includes("yakın"))
      secondary = "Klinik takip ve kontrol değerlendirmesi önerilmektedir.";
    else if (b.includes("görüntüleme") || b.includes("bt ") || b.includes("grafi"))
      secondary = "Tanı kesinliği için ileri görüntüleme önerilebilir.";
    else if (b.includes("sistemik"))
      secondary = "Sistemik hastalık olasılığı bağlamsal olarak değerlendirildi.";
    return { primary: bullet, secondary, category: "clinical" };
  });
}

function ClinicalReasoningSection({
  result,
  ctx,
}: {
  result: ClinicalAnalysisResult;
  ctx: ClinicalContextRequest | null;
}) {
  const tier  = result.risk.risk_tier;
  const cfg   = HERO_TIER[tier];
  const items = buildClinicalInsightItems(result);

  const p22Items: { label: string; value: string }[] = [];
  if (ctx) {
    if (ctx.age != null)                  p22Items.push({ label: "Yaş",                value: `${ctx.age}` });
    if (ctx.sex)                          p22Items.push({ label: "Cinsiyet",            value: SEX_LABELS[ctx.sex] ?? ctx.sex });
    if (ctx.respiratory_severity)         p22Items.push({ label: "Nefes durumu",        value: RESPIRATORY_LABELS[ctx.respiratory_severity] ?? ctx.respiratory_severity });
    if (ctx.oxygenation_context)          p22Items.push({ label: "Oksijen kapasitesi",  value: OXYGENATION_LABELS[ctx.oxygenation_context] ?? ctx.oxygenation_context });
    if (ctx.fever_severity && ctx.fever_severity !== "none")
                                          p22Items.push({ label: "Ateş durumu",         value: FEVER_LABELS[ctx.fever_severity] ?? ctx.fever_severity });
    if (ctx.recent_worsening && ctx.recent_worsening !== "none")
                                          p22Items.push({ label: "Son günlerdeki seyir", value: WORSENING_LABELS[ctx.recent_worsening] ?? ctx.recent_worsening });
    if (ctx.symptom_duration_tier)        p22Items.push({ label: "Şikayet süresi",      value: DURATION_LABELS[ctx.symptom_duration_tier] ?? ctx.symptom_duration_tier });
    if (ctx.rodent_exposure_level && ctx.rodent_exposure_level !== "none")
                                          p22Items.push({ label: "Kemirgen maruziyeti", value: RODENT_LABELS[ctx.rodent_exposure_level] ?? ctx.rodent_exposure_level });
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.15, ease: [0.2, 0, 0, 1] }}
      className="relative overflow-hidden rounded-2xl border"
      style={{
        background: "hsl(222 45% 5%)",
        borderColor: cfg.borderStyle,
        boxShadow: `0 0 70px -22px hsl(${cfg.glowHue} / 0.22), inset 0 0 0 1px hsl(${cfg.glowHue} / 0.05)`,
      }}
    >
      <div
        className="pointer-events-none absolute inset-x-0 top-0 h-36"
        style={{ background: `radial-gradient(ellipse at 28% 0%, hsl(${cfg.glowHue} / 0.14), transparent 60%)` }}
      />
      <motion.div
        className="absolute left-0 top-8 bottom-8 w-[2px] rounded-full"
        style={{ background: `linear-gradient(180deg, transparent, hsl(${cfg.glowHue}), transparent)` }}
        initial={{ scaleY: 0, opacity: 0 }}
        animate={{ scaleY: 1, opacity: 1 }}
        transition={{ duration: 0.75, delay: 0.3, ease: [0.4, 0, 0.2, 1] }}
      />

      <div className="relative z-10 space-y-7 px-8 py-7">

        {/* Header */}
        <div className="space-y-1.5">
          <motion.p
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.38, delay: 0.1 }}
            className="text-[10px] font-bold uppercase tracking-[0.15em]"
            style={{ color: cfg.labelColor }}
          >
            AI KLİNİK AKIL YÜRÜTME
          </motion.p>
          <motion.h3
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.16 }}
            className="text-xl font-bold tracking-tight text-foreground"
          >
            Klinik Değerlendirme Akışı
          </motion.h3>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.4, delay: 0.24 }}
            className="text-xs leading-relaxed text-foreground-muted"
          >
            Yapay zeka semptom ve risk faktörlerini klinik bağlamda yorumladı.
          </motion.p>
        </div>

        {/* Summary callout */}
        <motion.div
          initial={{ opacity: 0, x: -6 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.45, delay: 0.28, ease: [0.2, 0, 0, 1] }}
          className="relative rounded-xl border bg-white/[0.025] p-4 pl-5"
          style={{
            borderColor: `hsl(${cfg.glowHue} / 0.2)`,
            boxShadow:   `inset 3px 0 0 hsl(${cfg.glowHue})`,
          }}
        >
          <p className="text-sm font-medium leading-relaxed text-foreground">{result.summary}</p>
        </motion.div>

        {/* Reasoning timeline */}
        {items.length > 0 && (
          <div>
            <p
              className="mb-4 text-[10px] font-semibold uppercase tracking-[0.13em]"
              style={{ color: cfg.labelColor, opacity: 0.6 }}
            >
              Klinik Akıl Yürütme Zinciri
            </p>

            <div className="relative space-y-1 pl-7">
              <motion.div
                className="absolute left-[10px] top-4 bottom-4 w-px"
                style={{ background: `linear-gradient(180deg, hsl(${cfg.glowHue} / 0.40), hsl(${cfg.glowHue} / 0.04))` }}
                initial={{ scaleY: 0 }}
                animate={{ scaleY: 1 }}
                transition={{ duration: 0.65, delay: 0.42, ease: [0.4, 0, 0.2, 1] }}
              />

              {items.map((item, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.38, delay: 0.4 + i * 0.10, ease: [0.2, 0, 0, 1] }}
                  className="group relative rounded-xl px-4 py-3 transition-colors duration-200 hover:bg-white/[0.03]"
                >
                  <div
                    className="absolute -left-7 top-[13px] flex h-[20px] w-[20px] items-center justify-center rounded-full border"
                    style={{
                      borderColor: `hsl(${cfg.glowHue} / 0.45)`,
                      background:  `hsl(${cfg.glowHue} / 0.12)`,
                    }}
                  >
                    <Check className="h-[11px] w-[11px]" style={{ color: cfg.labelColor }} />
                  </div>
                  <p className="text-sm font-semibold leading-snug text-foreground">{item.primary}</p>
                  <p className="mt-0.5 text-xs leading-relaxed text-foreground-muted">{item.secondary}</p>
                </motion.div>
              ))}
            </div>
          </div>
        )}

        {/* Evaluated clinical data */}
        {(result.symptoms_provided.length > 0 || p22Items.length > 0) && (
          <motion.div
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.5 + items.length * 0.06, ease: [0.2, 0, 0, 1] }}
            className="overflow-hidden rounded-xl border border-border/50"
            style={{ background: "hsl(222 45% 4%)" }}
          >
            <div
              className="flex items-center gap-2.5 border-b border-border/40 px-4 py-2.5"
              style={{ background: `hsl(${cfg.glowHue} / 0.06)` }}
            >
              <div className="h-[3px] w-5 rounded-full" style={{ background: `hsl(${cfg.glowHue})` }} />
              <p className="text-[10px] font-bold uppercase tracking-[0.13em] text-foreground-muted">
                Değerlendirilen Klinik Veriler
              </p>
            </div>

            <div className="space-y-3 p-4">
              {result.symptoms_provided.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {result.symptoms_provided.map((s) => (
                    <span
                      key={s}
                      className="rounded-full border px-2.5 py-0.5 text-xs font-medium"
                      style={{
                        borderColor: `hsl(${cfg.glowHue} / 0.32)`,
                        background:  `hsl(${cfg.glowHue} / 0.09)`,
                        color: cfg.labelColor,
                      }}
                    >
                      {SYMPTOM_LABELS[s] ?? s}
                    </span>
                  ))}
                </div>
              )}

              {result.exposure_provided && (
                <span className="inline-flex rounded-full border border-warning-300/30 bg-warning-500/8 px-3 py-0.5 text-xs font-semibold text-warning-400">
                  {EXPOSURE_LABELS[result.exposure_provided] ?? result.exposure_provided}
                </span>
              )}

              {p22Items.length > 0 && (
                <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 pt-1">
                  {p22Items.map((item) => (
                    <div key={item.label} className="flex items-baseline gap-1.5 text-xs">
                      <span className="shrink-0 text-foreground-muted/55">{item.label}:</span>
                      <span className="font-medium text-foreground-secondary">{item.value}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}

        {/* Disclaimer */}
        <div className="border-t border-white/[0.06] pt-4">
          <p className="text-2xs leading-relaxed text-foreground-muted/45">
            Bu sistem tıbbi teşhis koymaz. Bulgular klinisyen değerlendirmesini destekleme amacıyla üretilmiştir.
          </p>
        </div>
      </div>
    </motion.div>
  );
}

// ── Developer details (collapsed) ─────────────────────────────────────────────

function DeveloperDetails({ session }: { session: UnifiedAnalysisSession }) {
  const [open, setOpen] = useState(false);
  const exp = session.explainability;

  return (
    <div className="rounded-2xl border border-border overflow-hidden animate-fade-up animate-delay-500">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-5 py-3.5 bg-canvas hover:bg-surface transition-colors text-left"
      >
        <span className="text-xs font-semibold uppercase tracking-wider text-foreground-muted">
          Geliştirici Detayları
        </span>
        {open
          ? <ChevronUp className="h-4 w-4 text-foreground-muted" />
          : <ChevronDown className="h-4 w-4 text-foreground-muted" />}
      </button>

      {open && (
        <div className="px-5 pb-5 space-y-4 bg-canvas/50">
          <p className="text-xs text-foreground-muted pt-3 pb-1 border-b border-border-subtle">
            Teknik pipeline metrikleri — bu bilgiler normal kullanımda gizlenir.
          </p>

          {/* Pipeline warnings */}
          {exp.pipeline_warnings.length > 0 && (
            <div className="space-y-1.5">
              <p className="text-xs font-semibold uppercase tracking-wider text-foreground-muted">
                Pipeline Uyarıları
              </p>
              {exp.pipeline_warnings.map((w, i) => (
                <div
                  key={i}
                  className="flex gap-2 items-start rounded-lg bg-warning-50 border border-warning-100 px-3 py-2"
                >
                  <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-warning-600" />
                  <p className="text-xs text-warning-700 leading-relaxed font-mono">{w}</p>
                </div>
              ))}
            </div>
          )}

          <ImagingAnalysisPanel imaging={session.imaging} />
          {session.clinical.provided && (
            <ClinicalModifiersPanel clinical={session.clinical} />
          )}
          <SemanticReasoningPanel semantic={session.semantic} />
          <FusionBreakdownPanel fusion={session.fusion} />
          <TrustCalibrationPanel trust={session.trust} />
        </div>
      )}
    </div>
  );
}

// ── Results: API path ─────────────────────────────────────────────────────────

function APIResults({
  session,
  previewUrl,
  clinicalCtx,
  assistantSessionId,
}: {
  session: UnifiedAnalysisSession;
  previewUrl: string | null;
  clinicalCtx: ClinicalContextRequest | null;
  assistantSessionId: string;
}) {
  if (session.ood_guard_applied) return <OodRejectionPanel semantic={session.semantic} />;

  const assistantContext = buildAssistantContext(session, clinicalCtx);

  return (
    <div className="space-y-4">
      {/* Hero risk verdict with count-up score */}
      <HeroRiskSection session={session} />

      {/* X-ray viewer + insight panel side-by-side */}
      {previewUrl && (
        <div className="grid gap-4 lg:grid-cols-5">
          <div className="lg:col-span-3">
            <PremiumImageViewer
              previewUrl={previewUrl}
              gradcamBase64={session.explainability.gradcam_base64}
              targetClass={session.explainability.gradcam_target_class}
            />
          </div>
          <div className="lg:col-span-2">
            <InsightPanel session={session} />
          </div>
        </div>
      )}

      {/* AI explanation with checkmark bullets */}
      <AIReasoningSection session={session} />

      {/* AI clinical assistant */}
      <div id="assistant" className="scroll-mt-24">
        <MedicalAssistantPanel
          analysisContext={assistantContext}
          sessionId={assistantSessionId}
        />
      </div>

      {/* Technical details (collapsed) */}
      <DeveloperDetails session={session} />
    </div>
  );
}

// ── Results: Clinical-only path ───────────────────────────────────────────────

function buildClinicalAssistantContext(
  result: ClinicalAnalysisResult,
  ctx: ClinicalContextRequest | null,
): MedicalAnalysisContext {
  return {
    risk_tier: result.risk.risk_tier as MedicalAnalysisContext["risk_tier"],
    final_score: result.risk.final_score,
    requires_immediate_action: result.requires_immediate_action,
    near_boundary: result.risk.near_boundary ?? false,
    has_image: false,
    predicted_class: null,
    imaging_score: null,
    bilateral_burden: null,
    ood_detected: false,
    ood_label: null,
    has_clinical: true,
    symptoms_flagged: result.symptoms_provided,
    respiratory_severity: ctx?.respiratory_severity ?? null,
    oxygenation_context: ctx?.oxygenation_context ?? null,
    fever_severity: ctx?.fever_severity ?? null,
    recent_worsening: ctx?.recent_worsening ?? null,
    rodent_exposure_level: ctx?.rodent_exposure_level ?? null,
    symptom_duration_tier: ctx?.symptom_duration_tier ?? null,
    exposure_history: ctx?.exposure_history ?? null,
    age: ctx?.age ?? null,
    sex: ctx?.sex ?? null,
    summary: result.summary,
    imaging_findings: null,
  };
}

function ClinicalResults({
  result,
  ctx,
  assistantSessionId,
}: {
  result: ClinicalAnalysisResult;
  ctx: ClinicalContextRequest | null;
  assistantSessionId: string;
}) {
  const assistantContext = buildClinicalAssistantContext(result, ctx);
  return (
    <div className="space-y-4">
      {/* Clinical hero: animated score + tier */}
      <ClinicalHeroSection result={result} />

      {/* Symptom constellation + insight cards */}
      <div className="grid gap-4 lg:grid-cols-5">
        <div className="lg:col-span-3">
          <ClinicalVisualization result={result} ctx={ctx} />
        </div>
        <div className="lg:col-span-2">
          <ClinicalInsightPanel result={result} ctx={ctx} />
        </div>
      </div>

      {/* AI clinical reasoning timeline */}
      <ClinicalReasoningSection result={result} ctx={ctx} />

      {/* AI clinical assistant */}
      <div id="assistant" className="scroll-mt-24">
        <MedicalAssistantPanel
          analysisContext={assistantContext}
          sessionId={assistantSessionId}
        />
      </div>
    </div>
  );
}

// ── Loading ───────────────────────────────────────────────────────────────────

const LOADING_SCAN_CORNERS = [
  "left-0 top-0 border-l border-t",
  "right-0 top-0 border-r border-t",
  "bottom-0 left-0 border-b border-l",
  "bottom-0 right-0 border-b border-r",
];

const TELEMETRY_TICKS = [
  "Akciğer lobları sınırlandırılıyor...",
  "Doku yoğunluk haritası oluşturuluyor...",
  "Bilateral pulmoner simetri değerlendiriliyor...",
  "Patoloji koherans skoru hesaplanıyor...",
  "Segmentasyon sınırları doğrulanıyor...",
  "Odaksal aktivasyon analizi yapılıyor...",
  "Klinik ağırlık faktörleri uygulanıyor...",
  "Güven skoru kalibre ediliyor...",
  "Anormal bölge lokalizasyonu kontrol ediliyor...",
  "AI karar modeli tamamlanıyor...",
];

const IMAGING_SHORT_LABELS = [
  "Segmentasyon",
  "Patoloji",
  "GradCAM",
  "Risk Skoru",
  "Sonuç",
];

interface AnalysisLoadingProps {
  isClinical: boolean;
  previewUrl: string | null;
  pendingClinical: ClinicalContextRequest | null;
  onCancel: () => void;
}

function AnalysisLoading({ isClinical, previewUrl, pendingClinical, onCancel }: AnalysisLoadingProps) {
  const [clinicalStageIdx, setClinicalStageIdx] = useState(0);
  const [imagingStageIdx,  setImagingStageIdx]  = useState(0);
  const [telemetryIdx,     setTelemetryIdx]     = useState(0);

  useEffect(() => {
    if (!isClinical) return;
    const delays = [0, 110, 240, 390, 520];
    const ids = delays.map((d, i) => setTimeout(() => setClinicalStageIdx(i), d));
    return () => ids.forEach(clearTimeout);
  }, [isClinical]);

  useEffect(() => {
    if (isClinical) return;
    const jitter = () => Math.floor(Math.random() * 120);
    const base   = [0, 950, 2050, 3200, 4450];
    const ids    = base.map((d, i) =>
      setTimeout(() => setImagingStageIdx(i), i === 0 ? 0 : d + jitter()),
    );
    return () => ids.forEach(clearTimeout);
  }, [isClinical]);

  useEffect(() => {
    if (isClinical) return;
    let id: ReturnType<typeof setTimeout>;
    const tick = () => {
      setTelemetryIdx(prev => (prev + 1) % TELEMETRY_TICKS.length);
      id = setTimeout(tick, 1800 + Math.floor(Math.random() * 350));
    };
    id = setTimeout(tick, 1900 + Math.floor(Math.random() * 350));
    return () => clearTimeout(id);
  }, [isClinical]);

  const stages    = isClinical ? CLINICAL_STAGES : IMAGING_STAGES;
  const activeIdx = isClinical ? clinicalStageIdx : imagingStageIdx;
  const symptoms  = pendingClinical?.symptoms ?? [];

  const arcR      = 75;
  const arcCircum = 2 * Math.PI * arcR;
  const arcPct    = !isClinical ? (imagingStageIdx + 1) / IMAGING_STAGES.length : 0;
  const arcOffset = arcCircum * (1 - arcPct);

  // ── Clinical path ─────────────────────────────────────────────────────────
  if (isClinical) {
    return (
      <div
        className="relative overflow-hidden rounded-2xl border border-brand-500/30 glass-elevated"
        style={{ boxShadow: "0 0 60px -16px hsl(221 83% 53% / 0.25)" }}
      >
        <div
          className="absolute inset-x-0 top-0 h-px"
          style={{ background: "linear-gradient(90deg, transparent, hsl(221 83% 53% / 0.9), transparent)" }}
        />
        <div
          className="pointer-events-none absolute inset-0"
          style={{ background: "radial-gradient(ellipse at 50% 0%, hsl(221 83% 53% / 0.11), transparent 65%)" }}
        />
        <div className="relative z-10 space-y-5 p-6">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <div className="h-1.5 w-1.5 rounded-full bg-brand-400 animate-pulse-slow" />
              <span className="text-xs font-semibold uppercase tracking-[0.08em] text-brand-400">
                Klinik AI Analizi
              </span>
            </div>
            <h2 className="text-2xl font-bold tracking-tight text-foreground">
              Klinik veriler değerlendiriliyor
            </h2>
            <p className="text-xs text-foreground-muted">
              Çok modlu AI sistemi klinik bulgularınızı işliyor
            </p>
          </div>

          <div className="space-y-0.5">
            {stages.map((stage, i) => {
              const isDone    = i < activeIdx;
              const isActive  = i === activeIdx;
              const isPending = i > activeIdx;
              return (
                <motion.div
                  key={stage}
                  initial={{ opacity: 0, x: -6 }}
                  animate={{ opacity: isPending ? 0.35 : 1, x: 0 }}
                  transition={{ duration: 0.3, delay: i * 0.06, ease: [0.2, 0, 0, 1] }}
                  className={cn(
                    "flex items-center gap-3 rounded-xl px-3 py-2.5",
                    isActive && "bg-brand-50/12 border border-brand-500/18",
                  )}
                >
                  <div className="flex h-5 w-5 shrink-0 items-center justify-center">
                    {isDone ? (
                      <div className="h-1.5 w-1.5 rounded-full bg-brand-400/60" />
                    ) : isActive ? (
                      <div className="relative flex items-center justify-center">
                        <div
                          className="absolute h-4 w-4 rounded-full border border-brand-400/35 animate-ping"
                          style={{ animationDuration: "1.8s" }}
                        />
                        <div className="h-2 w-2 rounded-full bg-brand-400 animate-pulse-slow" />
                      </div>
                    ) : (
                      <div className="h-1.5 w-1.5 rounded-full border border-border bg-transparent" />
                    )}
                  </div>
                  <span className={cn(
                    "flex-1 text-xs transition-colors duration-300",
                    isDone    && "text-foreground-muted/65",
                    isActive  && "font-semibold text-foreground",
                    isPending && "text-foreground-muted/40",
                  )}>
                    {stage}
                  </span>
                  {isActive && (
                    <motion.span
                      animate={{ opacity: [0.45, 1, 0.45] }}
                      transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
                      className="text-2xs font-medium text-brand-400"
                    >
                      aktif
                    </motion.span>
                  )}
                </motion.div>
              );
            })}
          </div>

          {symptoms.length > 0 && (
            <div className="space-y-2 pt-1">
              <p className="text-2xs font-semibold uppercase tracking-[0.06em] text-foreground-muted">
                İşlenen Semptomlar
              </p>
              <div className="flex flex-wrap gap-1.5">
                {symptoms.map((sym, i) => (
                  <motion.span
                    key={sym}
                    initial={{ opacity: 0, scale: 0.88 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.28, delay: 0.25 + i * 0.05, ease: [0.2, 0, 0, 1] }}
                    className="rounded-full border border-brand-200/50 bg-brand-50/50 px-2.5 py-0.5 text-2xs font-medium text-brand-600"
                  >
                    {SYMPTOM_LABELS[sym] ?? sym}
                  </motion.span>
                ))}
              </div>
            </div>
          )}

          <div className="flex justify-center pt-1">
            <button
              type="button"
              onClick={onCancel}
              className="rounded-xl border border-border/50 px-5 py-2 text-xs font-medium text-foreground-muted transition-all duration-200 hover:border-danger-200/60 hover:text-danger-500"
            >
              Analizi İptal Et
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ── Cinematic imaging path ─────────────────────────────────────────────────
  return (
    <div
      className="relative overflow-hidden rounded-2xl border border-brand-500/20"
      style={{
        minHeight: "700px",
        background: "hsl(222 45% 4%)",
        boxShadow: "0 0 100px -20px hsl(221 83% 53% / 0.28), inset 0 0 0 1px hsl(221 83% 53% / 0.06)",
      }}
    >
      {/* Layer 0 — X-ray background */}
      {previewUrl && (
        <div className="pointer-events-none absolute inset-0">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={previewUrl}
            alt=""
            className="h-full w-full object-cover"
            style={{ opacity: 0.085, filter: "grayscale(1) blur(3px)", transform: "scale(1.06)" }}
          />
          <div
            className="absolute inset-0"
            style={{
              background:
                "linear-gradient(180deg, hsl(222 45% 4% / 0.65) 0%, hsl(222 45% 4% / 0.28) 38%, hsl(222 45% 4% / 0.52) 68%, hsl(222 45% 4% / 0.94) 100%)",
            }}
          />
        </div>
      )}

      {/* Layer 1 — Dot grid */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          backgroundImage: "radial-gradient(circle, hsl(221 83% 53% / 0.08) 1px, transparent 1px)",
          backgroundSize: "36px 36px",
        }}
      />

      {/* Layer 2 — Ambient glows */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{ background: "radial-gradient(ellipse at 50% 0%, hsl(221 83% 53% / 0.15), transparent 52%)" }}
      />
      <div
        className="pointer-events-none absolute inset-0"
        style={{ background: "radial-gradient(ellipse at 50% 110%, hsl(221 83% 53% / 0.07), transparent 46%)" }}
      />

      {/* Layer 3 — Horizontal scan sweep */}
      <motion.div
        className="pointer-events-none absolute inset-x-0 h-[2px]"
        style={{
          background:
            "linear-gradient(90deg, transparent 0%, hsl(221 83% 53% / 0.28) 14%, hsl(221 83% 53% / 0.78) 50%, hsl(221 83% 53% / 0.28) 86%, transparent 100%)",
          boxShadow: "0 0 28px 14px hsl(221 83% 53% / 0.09)",
        }}
        animate={{ top: ["4%", "96%"] }}
        transition={{ duration: 5.4, repeat: Infinity, ease: "linear", repeatDelay: 1.8 }}
      />

      {/* Layer 4 — Top edge accent */}
      <div
        className="absolute inset-x-0 top-0 h-px"
        style={{ background: "linear-gradient(90deg, transparent, hsl(221 83% 53% / 0.68), transparent)" }}
      />

      {/* Layer 5 — Corner brackets */}
      <div className="pointer-events-none absolute inset-6">
        {LOADING_SCAN_CORNERS.map((pos, i) => (
          <div key={i} className={cn("absolute h-8 w-8 border-brand-400/20", pos)} />
        ))}
      </div>

      {/* ── Main content ── */}
      <div className="relative z-10 flex flex-col" style={{ minHeight: "700px" }}>

        {/* Header */}
        <div className="flex items-center justify-between px-8 pt-7">
          <div className="flex items-center gap-2">
            <div className="h-1.5 w-1.5 rounded-full bg-brand-400 animate-pulse-slow" />
            <span className="text-xs font-semibold uppercase tracking-[0.12em] text-brand-400">
              Pulmoner AI Analizi
            </span>
            <span className="mx-0.5 font-mono text-xs text-brand-400/22">[</span>
            <span className="text-2xs text-foreground-muted/40">EfficientNet-B0 v6</span>
            <span className="font-mono text-xs text-brand-400/22">]</span>
          </div>
          <button
            type="button"
            onClick={onCancel}
            className="rounded-lg border border-border/40 px-4 py-1.5 text-xs font-medium text-foreground-muted/60 transition-all duration-200 hover:border-danger-200/50 hover:text-danger-500"
          >
            İptal Et
          </button>
        </div>

        {/* Central focus area */}
        <div className="flex flex-1 flex-col items-center justify-center gap-7 px-8 py-10">

          {/* Circular indicator */}
          <div className="relative flex h-52 w-52 items-center justify-center">

            {/* Expanding pulse rings */}
            {[0, 1].map((i) => (
              <motion.div
                key={i}
                className="absolute rounded-full border border-brand-400/10"
                style={{ inset: `${-(i + 1) * 14}px` }}
                animate={{ scale: [1, 1.18], opacity: [0.45, 0] }}
                transition={{ duration: 2.7, delay: i * 1.0, repeat: Infinity, ease: "easeOut" }}
              />
            ))}

            {/* SVG rings + progress arc */}
            <svg
              width="208"
              height="208"
              viewBox="0 0 200 200"
              className="absolute inset-0"
              style={{ overflow: "visible" }}
            >
              <defs>
                <linearGradient id="arcGrad" x1="0%" y1="100%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="hsl(221 83% 53%)" stopOpacity="0.65" />
                  <stop offset="100%" stopColor="hsl(213 78% 66%)" stopOpacity="1" />
                </linearGradient>
              </defs>

              {/* Static background rings */}
              <circle cx="100" cy="100" r="92" fill="none" stroke="hsl(221 83% 53% / 0.06)" strokeWidth="1" />
              <circle cx="100" cy="100" r="86" fill="none" stroke="hsl(221 83% 53% / 0.04)" strokeWidth="0.5" />

              {/* Clockwise rotating dashed ring */}
              <motion.circle
                cx="100" cy="100" r="82"
                fill="none"
                stroke="hsl(221 83% 53% / 0.16)"
                strokeWidth="1.5"
                strokeDasharray="5 17"
                style={{ transformOrigin: "100px 100px" }}
                animate={{ rotate: 360 }}
                transition={{ duration: 26, repeat: Infinity, ease: "linear" }}
              />

              {/* Counter-clockwise rotating dashed ring */}
              <motion.circle
                cx="100" cy="100" r="66"
                fill="none"
                stroke="hsl(221 83% 53% / 0.09)"
                strokeWidth="1"
                strokeDasharray="3 22"
                style={{ transformOrigin: "100px 100px" }}
                animate={{ rotate: -360 }}
                transition={{ duration: 38, repeat: Infinity, ease: "linear" }}
              />

              {/* Inner soft fills */}
              <circle cx="100" cy="100" r="58" fill="hsl(221 83% 53% / 0.05)" />
              <circle cx="100" cy="100" r="36" fill="hsl(221 83% 53% / 0.045)" />

              {/* Progress arc — starts at 12 o'clock */}
              <g transform="rotate(-90 100 100)">
                <motion.circle
                  cx="100" cy="100" r={arcR}
                  fill="none"
                  stroke="url(#arcGrad)"
                  strokeWidth="3.5"
                  strokeLinecap="round"
                  strokeDasharray={arcCircum}
                  animate={{ strokeDashoffset: arcOffset }}
                  transition={{ duration: 1.0, ease: [0.4, 0, 0.2, 1] }}
                />
              </g>
            </svg>

            {/* Center percentage */}
            <div className="relative z-10 flex flex-col items-center">
              <AnimatePresence mode="wait">
                <motion.div
                  key={imagingStageIdx}
                  initial={{ opacity: 0, scale: 0.82 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 1.1 }}
                  transition={{ duration: 0.38, ease: [0.2, 0, 0, 1] }}
                  className="flex items-end leading-none"
                >
                  <span className="text-[2.6rem] font-bold tabular-nums tracking-tight text-foreground">
                    {Math.round(arcPct * 100)}
                  </span>
                  <span className="mb-1.5 ml-0.5 text-xl font-semibold text-brand-400">%</span>
                </motion.div>
              </AnimatePresence>
              <span className="mt-1 text-2xs font-medium uppercase tracking-[0.12em] text-foreground-muted/38">
                analiz
              </span>
            </div>
          </div>

          {/* Active stage label */}
          <div className="flex flex-col items-center gap-3 text-center">
            <AnimatePresence mode="wait">
              <motion.h2
                key={`stage-${imagingStageIdx}`}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.44, ease: [0.2, 0, 0, 1] }}
                className="text-xl font-semibold tracking-tight text-foreground"
              >
                {IMAGING_STAGES[imagingStageIdx]}
              </motion.h2>
            </AnimatePresence>

            {/* Telemetry ticker */}
            <div className="flex h-5 items-center">
              <AnimatePresence mode="wait">
                <motion.span
                  key={`telem-${telemetryIdx}`}
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -4 }}
                  transition={{ duration: 0.42, ease: "easeInOut" }}
                  className="font-mono text-xs text-foreground-muted/42"
                >
                  ↳ {TELEMETRY_TICKS[telemetryIdx]}
                </motion.span>
              </AnimatePresence>
            </div>
          </div>
        </div>

        {/* Bottom — stage timeline + progress bar */}
        <div className="space-y-5 px-8 pb-9">

          {/* Horizontal stage timeline */}
          <div className="flex items-start">
            {IMAGING_STAGES.map((_, i) => (
              <Fragment key={i}>
                <div className="flex flex-col items-center gap-2">
                  <div className="relative flex h-2.5 w-2.5 items-center justify-center">
                    {i === activeIdx && (
                      <motion.div
                        className="absolute h-2.5 w-2.5 rounded-full bg-brand-500"
                        animate={{ scale: [1, 1.9], opacity: [0.6, 0] }}
                        transition={{ duration: 1.5, repeat: Infinity, ease: "easeOut" }}
                      />
                    )}
                    <div className={cn(
                      "h-2.5 w-2.5 rounded-full transition-all duration-500",
                      i < activeIdx  && "bg-brand-400/65",
                      i === activeIdx && "bg-brand-500",
                      i > activeIdx  && "border border-brand-400/20 bg-transparent",
                    )} />
                  </div>
                  <span className={cn(
                    "max-w-[68px] text-center text-[9.5px] leading-snug transition-colors duration-300",
                    i < activeIdx  && "text-foreground-muted/35",
                    i === activeIdx && "font-medium text-brand-400",
                    i > activeIdx  && "text-foreground-muted/20",
                  )}>
                    {IMAGING_SHORT_LABELS[i]}
                  </span>
                </div>

                {i < IMAGING_STAGES.length - 1 && (
                  <div className="relative mx-1 mt-[5px] h-px flex-1 overflow-hidden">
                    <div className="absolute inset-0 bg-brand-400/10" />
                    <motion.div
                      className="absolute inset-y-0 left-0"
                      style={{
                        background: "linear-gradient(90deg, hsl(221 83% 53% / 0.50), hsl(213 78% 65% / 0.50))",
                      }}
                      animate={{ width: i < activeIdx ? "100%" : "0%" }}
                      transition={{ duration: 0.7, ease: [0.4, 0, 0.2, 1] }}
                    />
                  </div>
                )}
              </Fragment>
            ))}
          </div>

          {/* Slim progress bar */}
          <div className="relative h-[3px] w-full overflow-hidden rounded-full bg-brand-100/10">
            <motion.div
              className="absolute inset-y-0 left-0 rounded-full"
              style={{
                background: "linear-gradient(90deg, hsl(221 83% 53%), hsl(213 78% 65%))",
                boxShadow: "0 0 10px 3px hsl(221 83% 53% / 0.32)",
              }}
              animate={{ width: `${Math.round(arcPct * 100)}%` }}
              transition={{ duration: 0.9, ease: [0.4, 0, 0.2, 1] }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function MedicalPage() {
  const queryClient = useQueryClient();
  const { mutate, isPending, isError, error, data, progress, cancel, reset } = useMedicalAnalysis();
  const [previewUrl, setPreviewUrl]           = useState<string | null>(null);
  const [clinicalResult, setClinicalResult]   = useState<ClinicalAnalysisResult | null>(null);
  const [clinicalCtx, setClinicalCtx]         = useState<ClinicalContextRequest | null>(null);
  const [clinicalLoading, setClinicalLoading] = useState(false);
  const [pendingClinical, setPendingClinical] = useState<ClinicalContextRequest | null>(null);
  const [assistantSessionId, setAssistantSessionId] = useState(() => crypto.randomUUID());

  // Holds the backend result until MIN_ANALYSIS_MS has elapsed from the start
  // of the request, so the loading animation always plays for its full duration.
  const [displayData, setDisplayData] = useState<UnifiedAnalysisSession | null>(null);
  const analysisStartRef = useRef<number>(0);

  useEffect(() => {
    if (!data) return;
    const elapsed   = Date.now() - analysisStartRef.current;
    const remaining = Math.max(0, MIN_ANALYSIS_MS - elapsed);
    const show = () => {
      setDisplayData(data);
      queryClient.invalidateQueries({ queryKey: ["dashboard-summary"] });
    };
    if (remaining <= 0) { show(); return; }
    const id = setTimeout(show, remaining);
    return () => clearTimeout(id);
  }, [data, queryClient]);

  const isLoading = isPending || clinicalLoading || (!!data && !displayData);
  const hasResult = (displayData != null) || clinicalResult != null;

  const handleSubmit = (file: File | null, clinical: ClinicalContextRequest | null) => {
    // Guard against duplicate calls that can arrive during AnimatePresence
    // exit animations or before React flushes the pending state update.
    if (isPending || clinicalLoading) return;

    setClinicalResult(null);
    setClinicalCtx(null);
    setPendingClinical(clinical);
    setDisplayData(null);
    reset();

    if (file) {
      analysisStartRef.current = Date.now();
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl(URL.createObjectURL(file));
      mutate({ file, gradcam: true, clinicalContext: clinical });
    } else {
      setClinicalLoading(true);
      const t0 = Date.now();
      setTimeout(() => {
        const result = computeClinicalRisk(clinical ?? {});
        setClinicalCtx(clinical);
        setClinicalResult(result);
        setClinicalLoading(false);
        // Persist to backend so dashboard stats reflect clinical-only analyses.
        medicalApi.persistClinical({
          session_id:  result.session_id,
          risk_tier:   result.risk.risk_tier,
          final_score: result.risk.final_score,
          summary:     result.summary,
          duration_ms: Date.now() - t0,
        })
          .then(() => queryClient.invalidateQueries({ queryKey: ["dashboard-summary"] }))
          .catch(() => { /* silent — UI result is already shown */ });
      }, 600);
    }
  };

  const handleRestart = () => {
    cancel();
    reset();
    setDisplayData(null);
    setClinicalResult(null);
    setClinicalCtx(null);
    setClinicalLoading(false);
    setPendingClinical(null);
    setAssistantSessionId(crypto.randomUUID()); // fresh conversation on new analysis
  };

  return (
    <div className="mx-auto max-w-5xl space-y-7 pb-16">
      {/* Page header */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: [0.2, 0, 0, 1] }}
        className="space-y-2"
      >
        <div className="flex items-center gap-2.5">
          <div className="h-0.5 w-5 rounded-full bg-brand-500/60" />
          <span className="text-xs font-semibold uppercase tracking-widest text-brand-400">
            AI Karar Destek Sistemi
          </span>
        </div>
        <div className="relative">
          <div className="pointer-events-none absolute -left-8 -top-10 h-52 w-96 rounded-full bg-brand-500/8 blur-3xl" />
          <h1 className="relative text-[2.5rem] font-bold tracking-tight gradient-text-brand leading-[1.1] sm:text-5xl lg:text-[3.25rem]">
            Pulmoner Risk Analizi
          </h1>
        </div>
        <p className="text-sm text-foreground-secondary leading-relaxed max-w-lg">
          Akciğer grafisi ve klinik bulguları birlikte değerlendiren çok modlu AI sistemi.
        </p>
        {/* Live status chips */}
        <div className="flex flex-wrap items-center gap-2 pt-2">
          {([
            { label: "AI Aktif",              dot: true,  delay: 0.15, accent: true  },
            { label: "EfficientNet-B0 v6",    dot: false, delay: 0.22, accent: false },
            { label: "Stage C · ECE 0.036",   dot: false, delay: 0.29, accent: false },
            { label: "Recall 98.8%",          dot: false, delay: 0.36, accent: false },
          ]).map((chip) => (
            <motion.span
              key={chip.label}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.32, delay: chip.delay, ease: [0.2, 0, 0, 1] }}
              className={cn(
                "flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-semibold",
                chip.accent
                  ? "border-success-400/40 bg-success-500/10 text-success-400"
                  : "border-border-subtle/80 bg-surface-raised text-foreground-secondary",
              )}
            >
              {chip.dot && <span className="h-1.5 w-1.5 rounded-full bg-success-500 animate-pulse-slow" />}
              {chip.label}
            </motion.span>
          ))}
        </div>
      </motion.div>

      {/* Upload workspace ↔ Loading panel (cinematic swap) */}
      <AnimatePresence mode="wait">
        {isLoading ? (
          <motion.div
            key="loading-panel"
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.52, ease: [0.2, 0, 0, 1] }}
          >
            <AnalysisLoading
              isClinical={clinicalLoading}
              previewUrl={previewUrl}
              pendingClinical={pendingClinical}
              onCancel={handleRestart}
            />
          </motion.div>
        ) : !hasResult ? (
          <motion.div
            key="upload-zone"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.99 }}
            transition={{ duration: 0.4, ease: [0.2, 0, 0, 1] }}
          >
            <UploadZone
              onSubmit={handleSubmit}
              isLoading={isLoading}
              progress={progress}
              onCancel={handleRestart}
            />
          </motion.div>
        ) : null}
      </AnimatePresence>

      {/* Error */}
      <AnimatePresence>
        {isError && !isPending && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.3 }}
            className="rounded-2xl glass-card border-danger-200/50 p-5 flex items-start gap-3"
          >
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-danger-500" />
            <div className="flex-1">
              <p className="text-sm font-semibold text-danger-700">Değerlendirme tamamlanamadı</p>
              <p className="mt-0.5 text-xs text-danger-600 leading-relaxed">
                {error?.message ?? "Sunucu bağlantı hatası. Lütfen tekrar deneyin."}
              </p>
            </div>
            <button
              type="button"
              onClick={handleRestart}
              className="flex items-center gap-1.5 rounded-lg border border-danger-200/60 bg-canvas px-3 py-1.5 text-xs font-medium text-danger-700 hover:bg-danger-50 transition-colors shrink-0"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              Tekrar dene
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Results */}
      <AnimatePresence>
        {hasResult && !isLoading && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: [0.2, 0, 0, 1] }}
            className="space-y-5"
          >
            <div className="flex justify-end">
              <button
                type="button"
                onClick={handleRestart}
                className={cn(
                  "flex items-center gap-2 rounded-xl border border-border/60 glass-card px-5 py-2.5 text-sm font-semibold",
                  "text-foreground-secondary hover:border-brand-300/60 hover:text-foreground hover:shadow-[0_2px_16px_-4px_hsl(221_83%_53%/0.3)] transition-all duration-200",
                )}
              >
                <RefreshCw className="h-4 w-4" />
                Yeni Analiz
              </button>
            </div>

            {clinicalResult && (
              <ClinicalResults
                result={clinicalResult}
                ctx={clinicalCtx}
                assistantSessionId={assistantSessionId}
              />
            )}

            {displayData && !clinicalResult && (
              <APIResults
                session={displayData}
                previewUrl={previewUrl}
                clinicalCtx={clinicalCtx}
                assistantSessionId={assistantSessionId}
              />
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
