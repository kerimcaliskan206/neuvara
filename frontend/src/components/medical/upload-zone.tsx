"use client";

import { Activity, AlertTriangle, Camera, ImagePlus, Thermometer, Upload, User, X, Zap } from "lucide-react";
import { useCallback, useRef, useState } from "react";

import { cn, formatBytes } from "@/lib/utils";
import { config } from "@/lib/config";
import type { ClinicalContextRequest } from "@/lib/api/types";

// ── Symptom data ──────────────────────────────────────────────────────────────

const SYMPTOM_LABEL: Record<string, string> = {
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
  productive_cough:    "Balgamlı öksürük",
};

interface SymptomCategory {
  key: string;
  label: string;
  icon: React.ElementType;
  colorClass: string;
  borderActiveClass: string;
  activeChipClass: string;
  symptoms: string[];
}

const SYMPTOM_CATEGORIES: SymptomCategory[] = [
  {
    key: "respiratory",
    label: "Solunum",
    icon: Activity,
    colorClass: "text-brand-400",
    borderActiveClass: "border-brand-500/50 shadow-[0_0_24px_-8px_hsl(221_83%_53%/0.28)]",
    activeChipClass: "border-brand-300/80 bg-brand-50 text-brand-700 shadow-[0_1px_6px_-2px_hsl(221_83%_53%/0.3)]",
    symptoms: ["dyspnea", "shortness_of_breath", "cough", "productive_cough", "tachypnea", "chest_pain"],
  },
  {
    key: "systemic",
    label: "Sistemik",
    icon: Thermometer,
    colorClass: "text-warning-500",
    borderActiveClass: "border-warning-300/60 shadow-[0_0_24px_-8px_hsl(38_96%_54%/0.22)]",
    activeChipClass: "border-warning-200/80 bg-warning-50/90 text-warning-600",
    symptoms: ["fever", "fatigue", "myalgia", "night_sweats", "weight_loss"],
  },
  {
    key: "critical",
    label: "Kritik Bulgular",
    icon: AlertTriangle,
    colorClass: "text-danger-500",
    borderActiveClass: "border-danger-300/60 shadow-[0_0_24px_-8px_hsl(0_90%_51%/0.22)]",
    activeChipClass: "border-danger-200/80 bg-danger-50/90 text-danger-600",
    symptoms: ["hemoptysis", "hypoxia"],
  },
];

// ── Clinical options ───────────────────────────────────────────────────────────

const SEX_OPTIONS = [
  { value: "male",   label: "Erkek" },
  { value: "female", label: "Kadın" },
] as const;

const RESPIRATORY_OPTIONS = [
  { value: "normal", label: "Normal nefes alabiliyorum" },
  { value: "mild",   label: "Nefes almak biraz zorlaştı" },
  { value: "severe", label: "Nefes almak ciddi şekilde zor" },
] as const;

const OXYGENATION_OPTIONS = [
  { value: "normal",      label: "Günlük nefesim normal" },
  { value: "mild_drop",   label: "Nefes kapasitem azaldı" },
  { value: "severe_drop", label: "Dinlenirken bile nefes almak zor" },
] as const;

const FEVER_OPTIONS = [
  { value: "none",     label: "Ateşim yok" },
  { value: "mild",     label: "Hafif ateş" },
  { value: "moderate", label: "Orta ateş" },
  { value: "high",     label: "Yüksek ateş" },
] as const;

const WORSENING_OPTIONS = [
  { value: "none",      label: "Son günlerde belirgin kötüleşme yok" },
  { value: "some",      label: "Son günlerde kötüleşme hissediyorum" },
  { value: "rapid_48h", label: "Son 48 saatte hızlı kötüleşme oldu" },
] as const;

const DURATION_OPTIONS = [
  { value: "1_2_days",    label: "1–2 gün" },
  { value: "3_7_days",    label: "3–7 gün" },
  { value: "over_1_week", label: "1 haftadan uzun" },
] as const;

const RODENT_OPTIONS = [
  { value: "none",             label: "Bilinen maruziyet yok" },
  { value: "unsure",           label: "Emin değilim" },
  { value: "rural_env",        label: "Kırsal/depo ortamı" },
  { value: "possible_contact", label: "Kemirgen teması olabilir" },
] as const;

const EXPOSURE_OPTIONS = [
  { value: "",                  label: "Belirtilmedi" },
  { value: "hospital",          label: "Hastane maruziyeti" },
  { value: "sick_contact",      label: "Hasta ile temas" },
  { value: "travel",            label: "Seyahat öyküsü" },
  { value: "healthcare_worker", label: "Sağlık çalışanı" },
  { value: "immunocompromised", label: "İmmün yetmezlik" },
] as const;

// ── Shared styles ─────────────────────────────────────────────────────────────

const CHIP      = "rounded-full border px-3.5 py-1.5 text-xs font-semibold transition-all duration-200 select-none";
const CHIP_OFF  = "border-border-subtle bg-canvas/80 text-foreground-secondary hover:border-brand-400/60 hover:bg-brand-50/60 hover:text-foreground hover:shadow-[0_2px_10px_-3px_hsl(221_83%_53%/0.22)]";
const BLOCK     = "rounded-xl border px-3.5 py-2.5 text-xs font-medium transition-all duration-200 text-left w-full";
const BLOCK_OFF = "border-border-subtle bg-canvas/60 text-foreground-secondary hover:border-brand-400/50 hover:bg-brand-50/40 hover:text-foreground hover:shadow-[0_2px_10px_-3px_hsl(221_83%_53%/0.18)]";
const BLOCK_ON  = "border-brand-400/80 bg-brand-50 text-brand-600 shadow-[inset_0_1px_0_hsl(221_83%_53%/0.25),0_2px_10px_-3px_hsl(221_83%_53%/0.28)]";
const FL        = "text-2xs font-semibold uppercase tracking-[0.06em] text-foreground-muted";

// ── Section label ─────────────────────────────────────────────────────────────

function SectionLabel({
  icon: Icon,
  label,
  colorClass,
  right,
}: {
  icon?: React.ElementType;
  label: string;
  colorClass?: string;
  right?: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-2.5">
      {Icon && <Icon className={cn("h-4 w-4 shrink-0", colorClass ?? "text-foreground-muted")} />}
      <span className="whitespace-nowrap text-xs font-bold uppercase tracking-[0.1em] text-foreground">
        {label}
      </span>
      <div className="h-px flex-1 bg-border-subtle/80" />
      {right}
    </div>
  );
}

// ── Corner scan brackets ───────────────────────────────────────────────────────

const SCAN_CORNERS = [
  "left-0 top-0 border-l-2 border-t-2 rounded-tl",
  "right-0 top-0 border-r-2 border-t-2 rounded-tr",
  "bottom-0 left-0 border-b-2 border-l-2 rounded-bl",
  "bottom-0 right-0 border-b-2 border-r-2 rounded-br",
];

// ── Props ─────────────────────────────────────────────────────────────────────

interface UploadZoneProps {
  onSubmit: (file: File | null, clinical: ClinicalContextRequest | null) => void;
  isLoading: boolean;
  progress: number | null;
  onCancel: () => void;
}

// ── Component ─────────────────────────────────────────────────────────────────

export function UploadZone({ onSubmit, isLoading, progress, onCancel }: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  // Prevents a second POST if the user clicks submit again during the React
  // re-render gap or the AnimatePresence 400 ms exit animation.
  const submittedRef = useRef(false);
  const dragCounterRef = useRef(0);
  const cameraInputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging]     = useState(false);
  const [selected, setSelected]         = useState<{ file: File; previewUrl: string } | null>(null);
  const [fileError, setFileError]       = useState<string | null>(null);

  // ── Clinical state ───────────────────────────────────────────────────────
  const [symptoms,            setSymptoms]            = useState<string[]>([]);
  const [age,                 setAge]                 = useState("");
  const [sex,                 setSex]                 = useState("");
  const [respiratorySeverity, setRespiratorySeverity] = useState("");
  const [oxygenationContext,  setOxygenationContext]  = useState("");
  const [feverSeverity,       setFeverSeverity]       = useState("");
  const [recentWorsening,     setRecentWorsening]     = useState("");
  const [rodentExposure,      setRodentExposure]      = useState("");
  const [durationTier,        setDurationTier]        = useState("");
  const [exposure,            setExposure]            = useState("");

  // ── File validation ──────────────────────────────────────────────────────

  const validate = (file: File): string | null => {
    if (!config.upload.allowedMimeTypes.includes(file.type as string))
      return "Desteklenmeyen dosya türü. JPEG, PNG veya WebP yükleyiniz.";
    const maxBytes = config.upload.maxMb * 1024 * 1024;
    if (file.size > maxBytes) return `Dosya çok büyük (${formatBytes(file.size)}). Maks. ${config.upload.maxMb} MB.`;
    if (file.size === 0) return "Dosya boş.";
    return null;
  };

  const handleFile = useCallback((file: File) => {
    const msg = validate(file);
    if (msg) { setFileError(msg); return; }
    setFileError(null);
    setSelected((prev) => {
      if (prev) URL.revokeObjectURL(prev.previewUrl);
      return { file, previewUrl: URL.createObjectURL(file) };
    });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const onDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    dragCounterRef.current = 0;
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  // ── Clinical helpers ─────────────────────────────────────────────────────

  const toggleSymptom = (val: string) =>
    setSymptoms((prev) => prev.includes(val) ? prev.filter((s) => s !== val) : [...prev, val]);

  const buildClinical = (): ClinicalContextRequest | null => {
    const ctx: ClinicalContextRequest = {};
    if (symptoms.length)     ctx.symptoms              = symptoms;
    if (exposure)            ctx.exposure_history      = exposure;
    if (age !== "") {
      const n = parseInt(age, 10);
      if (!isNaN(n))         ctx.age                   = n;
    }
    if (sex)                 ctx.sex                   = sex as "male" | "female";
    if (respiratorySeverity) ctx.respiratory_severity  = respiratorySeverity as "normal" | "mild" | "severe";
    if (oxygenationContext)  ctx.oxygenation_context   = oxygenationContext as "normal" | "mild_drop" | "severe_drop";
    // Only send these when a genuinely informative value is selected (exclude "none" sentinel).
    if (feverSeverity && feverSeverity !== "none")
                             ctx.fever_severity        = feverSeverity as "mild" | "moderate" | "high";
    if (recentWorsening && recentWorsening !== "none")
                             ctx.recent_worsening      = recentWorsening as "some" | "rapid_48h";
    if (rodentExposure && rodentExposure !== "none")
                             ctx.rodent_exposure_level = rodentExposure as "unsure" | "rural_env" | "possible_contact";
    if (durationTier)        ctx.symptom_duration_tier = durationTier as "1_2_days" | "3_7_days" | "over_1_week";
    return Object.keys(ctx).length ? ctx : null;
  };

  // ── Submit guard ─────────────────────────────────────────────────────────

  const hasImage    = selected !== null;
  const hasClinical = symptoms.length > 0 || !!exposure || age !== "" || !!sex ||
    !!respiratorySeverity || !!oxygenationContext || !!feverSeverity ||
    !!recentWorsening || !!rodentExposure || !!durationTier;
  const canSubmit   = hasImage || hasClinical;

  const handleSubmit = () => {
    if (!canSubmit || isLoading || submittedRef.current) return;
    submittedRef.current = true;
    const clinical = buildClinical();
    onSubmit(selected?.file ?? null, clinical);
  };

  const handleClear = () => {
    if (selected) URL.revokeObjectURL(selected.previewUrl);
    setSelected(null);
    setFileError(null);
  };

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <div className="space-y-6 animate-fade-up">

      {/* ── Section 1: Upload Zone + Patient Info ───────────────────────── */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1fr_310px]">

        {/* LEFT: Medical Imaging Zone */}
        <div className="relative flex flex-col overflow-hidden rounded-2xl border border-border-subtle glass-elevated">
          <div
            className="absolute inset-x-0 top-0 h-px"
            style={{ background: "linear-gradient(90deg, transparent, hsl(221 83% 53% / 0.5), transparent)" }}
          />

          <div className="flex items-center justify-between px-5 pb-3 pt-4">
            <div className="flex items-center gap-2.5">
              <div className="h-2 w-2 rounded-full bg-brand-400" />
              <p className="text-sm font-bold tracking-tight text-foreground">Akciğer Grafisi</p>
            </div>
            <span className="rounded-full border border-border-subtle bg-canvas/60 px-2.5 py-0.5 text-2xs font-medium text-foreground-muted">
              İsteğe bağlı
            </span>
          </div>

          <div className="flex-1 px-5 pb-5">
            {!selected ? (
              <div
                role="button"
                tabIndex={0}
                aria-disabled={isLoading}
                onClick={() => inputRef.current?.click()}
                onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") inputRef.current?.click(); }}
                onDragEnter={(e) => { e.preventDefault(); dragCounterRef.current += 1; if (!isLoading) setIsDragging(true); }}
                onDragOver={(e) => { e.preventDefault(); }}
                onDragLeave={() => { dragCounterRef.current -= 1; if (dragCounterRef.current <= 0) { dragCounterRef.current = 0; setIsDragging(false); } }}
                onDrop={onDrop}
                style={{ transform: isDragging ? "scale(1.015)" : "scale(1)", transition: "transform 0.2s ease" }}
                className={cn(
                  "group relative flex min-h-[256px] cursor-pointer flex-col items-center justify-center gap-5 rounded-2xl border-2 border-dashed transition-all duration-300",
                  isDragging
                    ? "border-brand-400 bg-brand-50/15 shadow-[0_0_90px_-4px_hsl(221_83%_53%/0.65)]"
                    : "border-border/60 bg-canvas/20 hover:border-brand-400/60 hover:bg-brand-50/8 hover:shadow-[0_0_65px_-8px_hsl(221_83%_53%/0.48)]",
                  isLoading && "cursor-not-allowed opacity-60",
                )}
              >
                {/* Scan-frame corner brackets */}
                <div className="pointer-events-none absolute inset-4 rounded-xl">
                  {SCAN_CORNERS.map((pos, i) => (
                    <div
                      key={i}
                      className={cn(
                        "absolute h-8 w-8 border-brand-400/40 transition-all duration-300",
                        "group-hover:border-brand-400/80",
                        isDragging && "border-brand-400/95 h-9 w-9",
                        pos,
                      )}
                    />
                  ))}
                </div>

                {/* Subtle scan-line hint */}
                <div
                  className={cn(
                    "pointer-events-none absolute inset-x-10 h-px",
                    "bg-gradient-to-r from-transparent via-brand-400 to-transparent",
                    "transition-opacity duration-500",
                    isDragging ? "opacity-25" : "opacity-0 group-hover:opacity-10",
                  )}
                  style={{ top: "44%", boxShadow: "0 0 10px 2px hsl(221 83% 53% / 0.25)" }}
                />

                {/* Icon with ambient glow */}
                <div className="relative flex items-center justify-center">
                  <div className={cn(
                    "absolute h-32 w-32 rounded-full bg-brand-500/12 blur-3xl transition-all duration-500",
                    "group-hover:bg-brand-500/28 group-hover:h-40 group-hover:w-40",
                    isDragging && "bg-brand-500/42 h-48 w-48",
                  )} />
                  <div className={cn(
                    "relative flex h-20 w-20 items-center justify-center rounded-2xl border border-brand-200/50 bg-brand-50/60 transition-all duration-300",
                    "group-hover:border-brand-300/70 group-hover:bg-brand-50/80 group-hover:shadow-[0_6px_36px_-4px_hsl(221_83%_53%/0.48)]",
                    isDragging && "scale-110 border-brand-300/85 bg-brand-50/90",
                  )}>
                    <ImagePlus className={cn(
                      "h-9 w-9 text-brand-400 transition-all duration-300 group-hover:text-brand-500",
                      isDragging && "text-brand-500",
                    )} />
                  </div>
                </div>

                <div className="space-y-1.5 px-8 text-center">
                  <p className="text-sm font-semibold text-foreground">
                    {isDragging ? "Bırakın, görüntü yüklensin" : "Akciğer grafisi yükleyin"}
                  </p>
                  <p className="text-xs leading-relaxed text-foreground-muted">
                    Sürükle &amp; bırak veya tıklayın · JPEG · PNG · WebP · Maks. {config.upload.maxMb} MB
                  </p>
                  {!isDragging && !isLoading && (
                    <button
                      type="button"
                      onClick={(e) => { e.stopPropagation(); cameraInputRef.current?.click(); }}
                      className="mt-2 inline-flex items-center gap-1.5 rounded-full border border-brand-300/50 bg-brand-50/40 px-3.5 py-1.5 text-xs font-semibold text-brand-500 transition-all hover:border-brand-400/70 hover:bg-brand-50/70"
                    >
                      <Camera className="h-3.5 w-3.5" />
                      Fotoğraf Çek
                    </button>
                  )}
                </div>
              </div>
            ) : (
              <div className="relative flex min-h-[256px] items-center justify-center overflow-hidden rounded-xl border border-border bg-canvas">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={selected.previewUrl}
                  alt="Seçilen görüntü"
                  className="max-h-64 w-full object-contain"
                />
                <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between">
                  <span className="max-w-[75%] truncate rounded-full bg-black/50 px-3 py-0.5 text-xs text-white backdrop-blur-sm">
                    {selected.file.name}
                  </span>
                  {!isLoading && (
                    <button
                      type="button"
                      onClick={handleClear}
                      className="flex h-7 w-7 items-center justify-center rounded-full bg-black/60 text-white hover:bg-black/80 transition-colors"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>
              </div>
            )}

            {fileError && (
              <p className="mt-3 rounded-xl border border-danger-200/60 bg-danger-50/40 px-3.5 py-2.5 text-xs text-danger-500">
                {fileError}
              </p>
            )}
          </div>
        </div>

        {/* RIGHT: Patient Information Panel */}
        <div className="relative flex flex-col gap-4 overflow-hidden rounded-2xl border border-border-subtle glass-card-light p-5">
          <div
            className="absolute inset-x-0 top-0 h-px"
            style={{ background: "linear-gradient(90deg, transparent, hsl(221 83% 53% / 0.6), transparent)" }}
          />

          <SectionLabel icon={User} label="Hasta Bilgileri" />

          {/* Age */}
          <div className="space-y-2">
            <label className={FL}>Yaş</label>
            <input
              type="number"
              min={0}
              max={120}
              value={age}
              onChange={(e) => setAge(e.target.value)}
              disabled={isLoading}
              placeholder="—"
              className="w-full rounded-xl border border-border-subtle bg-canvas/60 px-3.5 py-2.5 text-sm text-foreground placeholder:text-foreground-muted focus:border-brand-400 focus:outline-none disabled:opacity-60 transition-colors"
            />
          </div>

          {/* Sex */}
          <div className="space-y-2">
            <label className={FL}>Cinsiyet</label>
            <div className="grid grid-cols-2 gap-2">
              {SEX_OPTIONS.map((o) => (
                <button
                  key={o.value}
                  type="button"
                  disabled={isLoading}
                  onClick={() => setSex((v) => (v === o.value ? "" : o.value))}
                  className={cn(
                    CHIP,
                    sex === o.value ? "border-brand-400/80 bg-brand-50 text-brand-600 shadow-[0_2px_10px_-3px_hsl(221_83%_53%/0.28)]" : CHIP_OFF,
                    isLoading && "opacity-60 cursor-not-allowed",
                  )}
                >
                  {o.label}
                </button>
              ))}
            </div>
          </div>

          {/* Complaint duration */}
          <div className="space-y-2">
            <label className={FL}>Şikayet Süresi</label>
            <div className="flex flex-wrap gap-2">
              {DURATION_OPTIONS.map((o) => (
                <button
                  key={o.value}
                  type="button"
                  disabled={isLoading}
                  onClick={() => setDurationTier((v) => (v === o.value ? "" : o.value))}
                  className={cn(
                    CHIP,
                    durationTier === o.value ? "border-brand-400/80 bg-brand-50 text-brand-600 shadow-[0_2px_10px_-3px_hsl(221_83%_53%/0.28)]" : CHIP_OFF,
                    isLoading && "opacity-60 cursor-not-allowed",
                  )}
                >
                  {o.label}
                </button>
              ))}
            </div>
          </div>

          {/* System readiness — pushed to bottom */}
          <div className="mt-auto space-y-2 border-t border-border-subtle pt-4">
            <div className="flex items-center justify-between text-2xs">
              <span className="text-foreground-muted">AI Motoru</span>
              <div className="flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-success-500 animate-pulse-slow" />
                <span className="font-semibold text-success-400">Hazır</span>
              </div>
            </div>
            <div className="flex items-center justify-between text-2xs">
              <span className="text-foreground-muted">Model</span>
              <span className="font-medium text-foreground-secondary tabular-nums">EfficientNet-B0 v6</span>
            </div>
          </div>
        </div>
      </div>

      {/* Hidden file input */}
      <input
        ref={inputRef}
        type="file"
        accept={config.upload.allowedMimeTypes.join(",")}
        className="hidden"
        disabled={isLoading}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
          e.target.value = "";
        }}
      />
      {/* Hidden camera input */}
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="hidden"
        disabled={isLoading}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
          e.target.value = "";
        }}
      />

      {/* ── Section 2: Symptom Categories ───────────────────────────────── */}
      <div className="space-y-3">
        <SectionLabel
          icon={Activity}
          label="Semptomlar"
          right={
            symptoms.length > 0 ? (
              <span className="text-2xs font-semibold text-brand-400">{symptoms.length} seçili</span>
            ) : undefined
          }
        />

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          {SYMPTOM_CATEGORIES.map((cat) => {
            const Icon = cat.icon;
            const activeCount = cat.symptoms.filter((s) => symptoms.includes(s)).length;
            return (
              <div
                key={cat.key}
                className={cn(
                  "relative overflow-hidden rounded-2xl border glass-card-light p-4 space-y-3 transition-all duration-200",
                  activeCount > 0 ? cat.borderActiveClass : "border-border-subtle",
                )}
              >
                {/* Category header with trailing rule */}
                <div className="flex items-center gap-2.5">
                  <Icon className={cn("h-4 w-4 shrink-0", cat.colorClass)} />
                  <span className="whitespace-nowrap text-xs font-bold uppercase tracking-[0.1em] text-foreground">
                    {cat.label}
                  </span>
                  <div className="h-px flex-1 bg-border-subtle/70" />
                  {activeCount > 0 && (
                    <span className={cn(
                      "flex h-5 min-w-5 items-center justify-center rounded-full px-1.5 text-2xs font-bold",
                      cat.activeChipClass,
                    )}>
                      {activeCount}
                    </span>
                  )}
                </div>

                <div className="flex flex-wrap gap-1.5">
                  {cat.symptoms.map((sym) => (
                    <button
                      key={sym}
                      type="button"
                      disabled={isLoading}
                      onClick={() => toggleSymptom(sym)}
                      className={cn(
                        CHIP,
                        symptoms.includes(sym) ? cat.activeChipClass : CHIP_OFF,
                        isLoading && "opacity-60 cursor-not-allowed",
                      )}
                    >
                      {SYMPTOM_LABEL[sym] ?? sym}
                    </button>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Section 3: Secondary Clinical Context ───────────────────────── */}
      <div className="relative overflow-hidden rounded-2xl border border-border-subtle glass-card-light p-5 space-y-5">
        <div
          className="absolute inset-x-0 top-0 h-px"
          style={{ background: "linear-gradient(90deg, transparent, hsl(221 83% 53% / 0.35), transparent)" }}
        />
        <SectionLabel label="Klinik Bağlam" />

        <div className="grid grid-cols-1 gap-5 md:grid-cols-2">

          {/* Nefes Durumu */}
          <div className="space-y-2.5">
            <label className={FL}>Nefes Durumu</label>
            <div className="flex flex-col gap-1.5">
              {RESPIRATORY_OPTIONS.map((o) => (
                <button
                  key={o.value}
                  type="button"
                  disabled={isLoading}
                  onClick={() => setRespiratorySeverity((v) => (v === o.value ? "" : o.value))}
                  className={cn(
                    BLOCK,
                    respiratorySeverity === o.value ? BLOCK_ON : BLOCK_OFF,
                    isLoading && "opacity-60 cursor-not-allowed",
                  )}
                >
                  {o.label}
                </button>
              ))}
            </div>
          </div>

          {/* Oksijen Kapasitesi */}
          <div className="space-y-2.5">
            <label className={FL}>Oksijen Kapasitesi</label>
            <div className="flex flex-col gap-1.5">
              {OXYGENATION_OPTIONS.map((o) => (
                <button
                  key={o.value}
                  type="button"
                  disabled={isLoading}
                  onClick={() => setOxygenationContext((v) => (v === o.value ? "" : o.value))}
                  className={cn(
                    BLOCK,
                    oxygenationContext === o.value ? BLOCK_ON : BLOCK_OFF,
                    isLoading && "opacity-60 cursor-not-allowed",
                  )}
                >
                  {o.label}
                </button>
              ))}
            </div>
          </div>

          {/* Ateş Durumu */}
          <div className="space-y-2.5">
            <label className={FL}>Ateş Durumu</label>
            <div className="flex flex-wrap gap-2">
              {FEVER_OPTIONS.map((o) => (
                <button
                  key={o.value}
                  type="button"
                  disabled={isLoading}
                  onClick={() => setFeverSeverity((v) => (v === o.value ? "" : o.value))}
                  className={cn(
                    CHIP,
                    feverSeverity === o.value ? BLOCK_ON : CHIP_OFF,
                    isLoading && "opacity-60 cursor-not-allowed",
                  )}
                >
                  {o.label}
                </button>
              ))}
            </div>
          </div>

          {/* Son Günlerdeki Seyir */}
          <div className="space-y-2.5">
            <label className={FL}>Son Günlerdeki Seyir</label>
            <div className="flex flex-col gap-1.5">
              {WORSENING_OPTIONS.map((o) => (
                <button
                  key={o.value}
                  type="button"
                  disabled={isLoading}
                  onClick={() => setRecentWorsening((v) => (v === o.value ? "" : o.value))}
                  className={cn(
                    BLOCK,
                    recentWorsening === o.value ? BLOCK_ON : BLOCK_OFF,
                    isLoading && "opacity-60 cursor-not-allowed",
                  )}
                >
                  {o.label}
                </button>
              ))}
            </div>
          </div>

        </div>

        {/* Sub-section divider + Maruziyet */}
        <div className="space-y-4 pt-1">
          <div className="flex items-center gap-2.5">
            <span className="whitespace-nowrap text-2xs font-semibold uppercase tracking-[0.06em] text-foreground-muted">
              Maruziyet Öyküsü
            </span>
            <div className="h-px flex-1 bg-border-subtle/40" />
          </div>

          <div className="grid grid-cols-1 gap-5 md:grid-cols-2">

            <div className="space-y-2.5">
              <label className={FL}>Kemirgen Maruziyeti</label>
              <div className="flex flex-wrap gap-2">
                {RODENT_OPTIONS.map((o) => (
                  <button
                    key={o.value}
                    type="button"
                    disabled={isLoading}
                    onClick={() => setRodentExposure((v) => (v === o.value ? "" : o.value))}
                    className={cn(
                      CHIP,
                      rodentExposure === o.value ? BLOCK_ON : CHIP_OFF,
                      isLoading && "opacity-60 cursor-not-allowed",
                    )}
                  >
                    {o.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-2.5">
              <label className={FL}>Diğer Maruziyet Öyküsü</label>
              <select
                value={exposure}
                onChange={(e) => setExposure(e.target.value)}
                disabled={isLoading}
                className="w-full rounded-xl border border-border-subtle bg-canvas/60 px-3.5 py-2.5 text-sm text-foreground focus:border-brand-400 focus:outline-none disabled:opacity-60 transition-colors"
              >
                {EXPOSURE_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
            </div>

          </div>
        </div>

        {/* Clinical guidance note */}
        <div className="flex items-start gap-2.5 rounded-xl border border-brand-100/30 bg-brand-50/15 px-3.5 py-2.5">
          <div className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-brand-400/60" />
          <p className="text-xs leading-relaxed text-foreground-muted">
            Klinik bağlam verileri görüntü analizi ile birleştirilerek risk skoru hassasiyeti artırılır.
          </p>
        </div>
      </div>

      {/* ── Section 4: AI Launch Panel ───────────────────────────────────── */}
      <div
        className={cn(
          "relative overflow-hidden rounded-2xl border glass-elevated p-6 space-y-5 transition-all duration-500",
          canSubmit ? "border-brand-500/30" : "border-border-subtle",
        )}
        style={canSubmit ? { boxShadow: "0 0 90px -12px hsl(221 83% 53% / 0.38), inset 0 1px 0 hsl(221 83% 53% / 0.14)" } : undefined}
      >
        <div
          className="absolute inset-x-0 top-0 h-px transition-all duration-500"
          style={canSubmit ? {
            background: "linear-gradient(90deg, transparent, hsl(221 83% 53% / 0.85), transparent)",
          } : {
            background: "linear-gradient(90deg, transparent, hsl(221 83% 53% / 0.2), transparent)",
          }}
        />

        {/* Validation hint */}
        {!canSubmit && !isLoading && (
          <p className="text-center text-xs text-foreground-muted">
            Görüntü ekleyin veya en az bir semptom / klinik bilgi girin.
          </p>
        )}

        {/* Progress bar */}
        {isLoading && progress !== null && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-foreground-muted">Değerlendiriliyor…</span>
              <span className="tabular-nums font-semibold text-brand-400">{progress}%</span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-black/10">
              <div
                className="h-full rounded-full progress-shimmer transition-[width] duration-300 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {/* Actions */}
        {isLoading ? (
          <button
            type="button"
            onClick={onCancel}
            className="w-full rounded-xl border border-danger-200/60 bg-danger-50/20 py-3.5 text-sm font-semibold text-danger-500 hover:bg-danger-50/40 transition-colors"
          >
            İptal Et
          </button>
        ) : (
          <button
            type="button"
            onClick={handleSubmit}
            disabled={!canSubmit}
            className={cn(
              "group relative w-full overflow-hidden rounded-2xl transition-all duration-500",
              canSubmit
                ? [
                    "py-7 text-white",
                    "shadow-[0_20px_80px_-8px_hsl(221_83%_53%/0.70),0_4px_20px_-4px_hsl(221_83%_53%/0.45)]",
                    "hover:shadow-[0_24px_100px_-4px_hsl(221_83%_53%/0.85),0_8px_28px_-4px_hsl(221_83%_53%/0.6)]",
                    "hover:scale-[1.018] active:scale-[0.982]",
                  ]
                : "bg-brand-100/8 text-brand-400/35 cursor-not-allowed py-7 border border-border-subtle/50",
            )}
            style={canSubmit ? {
              background: "linear-gradient(165deg, hsl(221 83% 60%) 0%, hsl(221 83% 50%) 45%, hsl(221 83% 40%) 100%)",
            } : undefined}
          >
            {/* Deep ambient radial at bottom */}
            {canSubmit && (
              <div
                className="pointer-events-none absolute inset-0 animate-pulse-slow"
                style={{ background: "radial-gradient(ellipse at 50% 140%, hsl(221 83% 53% / 0.50), transparent 52%)" }}
              />
            )}
            {/* Top glass highlight edge */}
            {canSubmit && (
              <div
                className="pointer-events-none absolute inset-x-0 top-0 h-px"
                style={{ background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent)" }}
              />
            )}
            {/* Inner top tint */}
            {canSubmit && (
              <div
                className="pointer-events-none absolute inset-x-0 top-0 h-1/2 rounded-t-2xl"
                style={{ background: "linear-gradient(180deg, rgba(255,255,255,0.08), transparent)" }}
              />
            )}
            {/* Sweep shimmer on hover */}
            {canSubmit && (
              <div className="pointer-events-none absolute inset-0 -translate-x-full skew-x-[-15deg] bg-gradient-to-r from-transparent via-white/18 to-transparent transition-transform duration-700 ease-out group-hover:translate-x-full" />
            )}
            <div className="relative flex flex-col items-center justify-center gap-2">
              <div className="flex items-center gap-3">
                <Zap className={cn(
                  "h-6 w-6 transition-all duration-300",
                  canSubmit ? "drop-shadow-[0_0_10px_rgba(255,255,255,0.7)]" : "opacity-30",
                )} />
                <span className="text-lg font-bold tracking-tight">Tanısal Analizi Başlat</span>
              </div>
              {canSubmit && (
                <span className="text-xs font-semibold uppercase tracking-widest text-white/55">
                  EfficientNet-B0 v6 · Stage C · Pulmoner Risk Analizi
                </span>
              )}
            </div>
          </button>
        )}
      </div>

    </div>
  );
}
