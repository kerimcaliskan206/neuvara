"use client";

import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  Bot,
  Check,
  ChevronDown,
  ChevronUp,
  ImageIcon,
  Send,
  ShieldCheck,
  Stethoscope,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { assistantStream } from "@/lib/api/medical";
import type { MedicalAnalysisContext } from "@/lib/api/types";
import { cn } from "@/lib/utils";

// ── Types ─────────────────────────────────────────────────────────────────────

interface Message {
  role: "user" | "assistant";
  content: string;
  id: string;
}

interface MedicalAssistantPanelProps {
  analysisContext: MedicalAnalysisContext;
  sessionId: string;
}

// ── Tier-aware config ─────────────────────────────────────────────────────────

const TIER_CONFIG = {
  LOW:                     { label: "DÜŞÜK RİSK",  color: "hsl(152 65% 60%)", glow: "152 65% 48%", border: "hsl(152 65% 48% / 0.28)" },
  MODERATE:                { label: "ORTA RİSK",   color: "hsl(38 90% 64%)",  glow: "38 90% 55%",  border: "hsl(38 90% 55% / 0.30)" },
  HIGH_DIFFERENTIAL_RISK:  { label: "YÜKSEK RİSK", color: "hsl(0 78% 68%)",   glow: "0 78% 58%",   border: "hsl(0 78% 58% / 0.34)" },
  CRITICAL_PULMONARY_RISK: { label: "KRİTİK RİSK", color: "hsl(0 78% 68%)",   glow: "0 78% 52%",   border: "hsl(0 78% 52% / 0.42)" },
} as const;

// ── Suggested questions ───────────────────────────────────────────────────────

interface QuestionCategory {
  label: string;
  icon: React.ReactNode;
  questions: string[];
}

const QUESTION_CATEGORIES: QuestionCategory[] = [
  {
    label: "Analiz Sonucu",
    icon: <Activity className="h-3 w-3" />,
    questions: [
      "Bu risk skoru neden çıktı?",
      "AI hangi bölgeye odaklandı?",
      "Görüntü yeterince kaliteli miydi?",
    ],
  },
  {
    label: "Radyoloji",
    icon: <ImageIcon className="h-3 w-3" />,
    questions: [
      "Ground-glass opacity ne anlama gelir?",
      "Konsolidasyon ile infiltrat farkı nedir?",
      "Pnömoni X-ray'de nasıl görünür?",
    ],
  },
  {
    label: "Klinik",
    icon: <Stethoscope className="h-3 w-3" />,
    questions: [
      "Ne zaman doktora başvurmalıyım?",
      "Hanta virüsü akciğeri nasıl etkiler?",
      "Kemirgen teması neden risk faktörü?",
    ],
  },
];

// ── Predicted class labels ────────────────────────────────────────────────────

const PREDICTED_CLASS_LABELS: Record<string, string> = {
  pneumonia_xray: "Pulmoner Konsolidasyon",
  healthy_xray:   "Normal Akciğer Grafisi",
  hard_negative:  "Patolojik Bulgu Yok",
  fake_medical:   "Tıbbi Görüntü Değil",
};

// ── Waveform typing indicator ─────────────────────────────────────────────────

function TypingWave() {
  return (
    <div className="flex items-center gap-[3px] px-0.5 py-1">
      {[0, 1, 2, 3].map((i) => (
        <motion.div
          key={i}
          className="h-3.5 w-[2.5px] rounded-full bg-brand-400/70"
          animate={{ scaleY: [0.25, 1, 0.25] }}
          transition={{ duration: 0.72, delay: i * 0.13, repeat: Infinity, ease: "easeInOut" }}
        />
      ))}
    </div>
  );
}

// ── Message bubble ────────────────────────────────────────────────────────────

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: [0.2, 0, 0, 1] }}
      className={cn("flex gap-2.5", isUser ? "flex-row-reverse" : "flex-row")}
    >
      {!isUser && (
        <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-brand-400/30 bg-brand-500/12">
          <Bot className="h-3.5 w-3.5 text-brand-400" />
        </div>
      )}
      <div
        className={cn(
          "max-w-[84%] rounded-2xl px-4 py-3 text-sm leading-[1.7]",
          isUser
            ? "rounded-tr-sm bg-brand-600/90 text-white shadow-[0_2px_16px_-4px_hsl(221_83%_53%/0.50)]"
            : "rounded-tl-sm border border-white/[0.08] bg-white/[0.03] text-foreground",
        )}
      >
        {message.content}
      </div>
    </motion.div>
  );
}

// ── Left: suggestion sidebar ──────────────────────────────────────────────────

function SuggestionSidebar({
  onSelect,
  disabled,
}: {
  onSelect: (q: string) => void;
  disabled: boolean;
}) {
  const [openCat, setOpenCat] = useState<number | null>(0);

  return (
    <div className="flex flex-col gap-1.5 overflow-y-auto" style={{ maxHeight: "420px" }}>
      <p className="mb-2 text-[10px] font-bold uppercase tracking-[0.13em] text-foreground-muted/45">
        Önerilen Sorular
      </p>

      {QUESTION_CATEGORIES.map((cat, ci) => (
        <div
          key={cat.label}
          className="overflow-hidden rounded-xl border border-white/[0.07] transition-colors"
        >
          <button
            type="button"
            onClick={() => setOpenCat(openCat === ci ? null : ci)}
            className="flex w-full items-center gap-2 px-3 py-2.5 text-left transition-colors hover:bg-white/[0.03]"
          >
            <span className="text-brand-400/60">{cat.icon}</span>
            <span className="flex-1 text-xs font-semibold text-foreground-secondary">{cat.label}</span>
            {openCat === ci
              ? <ChevronUp className="h-3 w-3 text-foreground-muted/35" />
              : <ChevronDown className="h-3 w-3 text-foreground-muted/35" />}
          </button>

          <AnimatePresence initial={false}>
            {openCat === ci && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.22, ease: [0.2, 0, 0, 1] }}
                className="overflow-hidden border-t border-white/[0.05]"
              >
                <div className="flex flex-col gap-0.5 p-1.5">
                  {cat.questions.map((q) => (
                    <button
                      key={q}
                      type="button"
                      onClick={() => onSelect(q)}
                      disabled={disabled}
                      className={cn(
                        "rounded-lg px-2.5 py-2 text-left text-xs leading-snug transition-all duration-150",
                        "text-foreground-muted hover:bg-brand-500/10 hover:text-foreground",
                        "disabled:opacity-35 disabled:cursor-not-allowed",
                      )}
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      ))}
    </div>
  );
}

// ── Right: analysis context panel ────────────────────────────────────────────

function ContextPanel({ ctx }: { ctx: MedicalAnalysisContext }) {
  const tier  = TIER_CONFIG[ctx.risk_tier];
  const score = Math.round(ctx.final_score * 100);

  return (
    <div className="flex flex-col gap-3">
      <p className="text-[10px] font-bold uppercase tracking-[0.13em] text-foreground-muted/45">
        Aktif Analiz
      </p>

      {/* Risk badge */}
      <div
        className="overflow-hidden rounded-xl border p-3.5 space-y-3"
        style={{ borderColor: tier.border, background: `hsl(${tier.glow} / 0.08)` }}
      >
        <div className="flex items-start justify-between gap-2">
          <p className="text-[11px] font-bold leading-tight tracking-wider" style={{ color: tier.color }}>
            {tier.label}
          </p>
          {ctx.requires_immediate_action && (
            <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-danger-400 animate-pulse" />
          )}
        </div>

        <div className="space-y-1">
          <div className="flex items-baseline justify-between">
            <span className="text-[10px] text-foreground-muted/50">Risk Skoru</span>
            <span className="text-xs font-bold tabular-nums" style={{ color: tier.color }}>
              {score}/100
            </span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-white/[0.06]">
            <motion.div
              className="h-full rounded-full"
              style={{ background: tier.color, boxShadow: `0 0 8px 2px hsl(${tier.glow} / 0.4)` }}
              initial={{ width: "0%" }}
              animate={{ width: `${score}%` }}
              transition={{ duration: 1.0, delay: 0.3, ease: [0.4, 0, 0.2, 1] }}
            />
          </div>
        </div>

        {ctx.near_boundary && (
          <p className="text-[10px] text-warning-400/75">⚠ Sınır değerine yakın</p>
        )}
      </div>

      {/* Imaging status */}
      {ctx.has_image && (
        <div className="rounded-xl border border-white/[0.07] bg-white/[0.02] p-3 space-y-1.5">
          <div className="flex items-center gap-1.5">
            <ImageIcon className="h-3 w-3 text-brand-400/65" />
            <p className="text-[10px] font-bold uppercase tracking-[0.1em] text-foreground-muted/50">
              Görüntü Analizi
            </p>
          </div>
          {ctx.predicted_class && (
            <p className="text-xs font-medium leading-snug text-foreground-secondary">
              {PREDICTED_CLASS_LABELS[ctx.predicted_class] ?? ctx.predicted_class}
            </p>
          )}
          {ctx.imaging_score != null && (
            <p className="text-[10px] text-foreground-muted/45">
              Görüntü skoru: {Math.round(ctx.imaging_score * 100)}/100
            </p>
          )}
        </div>
      )}

      {/* Clinical data */}
      {ctx.has_clinical && ctx.symptoms_flagged && ctx.symptoms_flagged.length > 0 && (
        <div className="rounded-xl border border-white/[0.07] bg-white/[0.02] p-3 space-y-1.5">
          <div className="flex items-center gap-1.5">
            <Stethoscope className="h-3 w-3 text-brand-400/65" />
            <p className="text-[10px] font-bold uppercase tracking-[0.1em] text-foreground-muted/50">
              Klinik Veri
            </p>
          </div>
          <p className="text-xs text-foreground-secondary">
            {ctx.symptoms_flagged.length} semptom değerlendirildi
          </p>
        </div>
      )}

      {/* Context active badge */}
      <div className="flex items-center gap-1.5 rounded-lg border border-success-400/22 bg-success-500/7 px-2.5 py-2">
        <Check className="h-3 w-3 shrink-0 text-success-400" />
        <p className="text-[10px] font-medium text-success-400/80">Analiz bağlamı aktif</p>
      </div>
    </div>
  );
}

// ── Main panel ────────────────────────────────────────────────────────────────

export function MedicalAssistantPanel({
  analysisContext,
  sessionId,
}: MedicalAssistantPanelProps) {
  const [messages, setMessages]           = useState<Message[]>([]);
  const [input, setInput]                 = useState("");
  const [isLoading, setIsLoading]         = useState(false);
  const [streamingContent, setStreaming]  = useState("");
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const inputRef      = useRef<HTMLTextAreaElement>(null);
  const abortRef      = useRef<AbortController | null>(null);

  // Abort in-flight request when sessionId changes or component unmounts
  useEffect(() => {
    return () => { abortRef.current?.abort(); };
  }, [sessionId]);

  // Scroll only the chat container — never the page
  useEffect(() => {
    const el = chatScrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages, isLoading, streamingContent]);

  const send = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || isLoading) return;

    // Cancel any previous in-flight stream
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setMessages((prev) => [...prev, { role: "user", content: trimmed, id: `u-${Date.now()}` }]);
    setInput("");
    setIsLoading(true);
    setStreaming("");

    let accumulated = "";

    try {
      for await (const token of assistantStream(
        { message: trimmed, session_id: sessionId, analysis_context: analysisContext },
        controller.signal,
      )) {
        accumulated += token;
        setStreaming(accumulated);
      }

      // Commit the completed streaming message
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: accumulated || "Yanıt alınamadı.", id: `a-${Date.now()}` },
      ]);
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") return;
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Yanıt alınamadı — asistan şu an ulaşılamıyor. Lütfen tekrar deneyin.",
          id: `err-${Date.now()}`,
        },
      ]);
    } finally {
      setStreaming("");
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send(input);
    }
  };

  const hasMessages = messages.length > 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.2, 0, 0, 1] }}
      className="relative overflow-hidden rounded-2xl border border-white/[0.07]"
      style={{
        background: "hsl(222 45% 5%)",
        boxShadow: "0 0 70px -22px hsl(221 83% 53% / 0.18), inset 0 0 0 1px hsl(221 83% 53% / 0.04)",
      }}
    >
      {/* Ambient top glow */}
      <div
        className="pointer-events-none absolute inset-x-0 top-0 h-28"
        style={{ background: "radial-gradient(ellipse at 50% 0%, hsl(221 83% 53% / 0.10), transparent 60%)" }}
      />

      {/* ── Header (full width) ── */}
      <div
        className="relative z-10 flex items-center justify-between border-b border-white/[0.06] px-5 py-3.5"
        style={{ background: "hsl(221 83% 53% / 0.04)" }}
      >
        <div className="flex items-center gap-3">
          <motion.div
            className="relative flex h-8 w-8 items-center justify-center rounded-xl border border-brand-400/30 bg-brand-500/12"
            animate={{
              boxShadow: [
                "0 0 0 0px hsl(221 83% 53% / 0.30)",
                "0 0 0 7px hsl(221 83% 53% / 0)",
                "0 0 0 0px hsl(221 83% 53% / 0)",
              ],
            }}
            transition={{ duration: 3.2, repeat: Infinity, ease: "easeOut" }}
          >
            <Bot className="h-4 w-4 text-brand-400" />
          </motion.div>
          <div>
            <p className="text-sm font-semibold text-foreground">AI Klinik Asistanı</p>
            <p className="text-[11px] text-foreground-muted/55">
              Analiz bağlamında klinik sorularınızı yanıtlar
            </p>
          </div>
        </div>
        <div className="flex items-center gap-1.5 rounded-full border border-success-400/22 bg-success-500/7 px-3 py-1">
          <ShieldCheck className="h-3 w-3 text-success-400" />
          <span className="text-[11px] font-medium text-success-400">Eğitim amaçlı</span>
        </div>
      </div>

      {/* ── Three-column body ── */}
      <div
        className="relative z-10 flex divide-x divide-white/[0.05]"
        style={{ minHeight: "460px" }}
      >
        {/* LEFT: suggestion sidebar — desktop only */}
        <div className="hidden w-[204px] shrink-0 flex-col p-4 lg:flex">
          <SuggestionSidebar onSelect={send} disabled={isLoading} />
        </div>

        {/* CENTER: chat */}
        <div className="flex min-w-0 flex-1 flex-col">

          {/* Message list — scroll is isolated to this container */}
          <div
            ref={chatScrollRef}
            className="flex-1 space-y-3.5 overflow-y-auto px-5 py-5"
            style={{ maxHeight: "390px", minHeight: "200px" }}
          >
            {/* Empty state */}
            {!hasMessages && !isLoading && (
              <motion.div
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.2 }}
                className="flex flex-col items-center justify-center gap-4 py-14 text-center"
              >
                <motion.div
                  className="flex h-14 w-14 items-center justify-center rounded-2xl border border-brand-400/25 bg-brand-500/10"
                  animate={{
                    boxShadow: [
                      "0 0 0 0px hsl(221 83% 53% / 0.28)",
                      "0 0 0 12px hsl(221 83% 53% / 0)",
                    ],
                  }}
                  transition={{ duration: 2.6, repeat: Infinity, ease: "easeOut" }}
                >
                  <Bot className="h-7 w-7 text-brand-400" />
                </motion.div>
                <div className="space-y-1">
                  <p className="text-sm font-semibold text-foreground">
                    Analizi birlikte değerlendirelim
                  </p>
                  <p className="max-w-[240px] text-xs leading-relaxed text-foreground-muted/55">
                    Görüntü bulguları, risk skoru veya klinik yorum hakkında soru sorabilirsiniz.
                  </p>
                </div>

                {/* Mobile inline quick questions */}
                <div className="flex flex-wrap justify-center gap-1.5 lg:hidden">
                  {QUESTION_CATEGORIES.flatMap((c) => c.questions.slice(0, 1)).map((q) => (
                    <button
                      key={q}
                      type="button"
                      onClick={() => send(q)}
                      disabled={isLoading}
                      className="rounded-full border border-white/[0.10] bg-white/[0.02] px-3 py-1.5 text-xs text-foreground-muted transition-colors hover:border-brand-400/35 hover:text-foreground disabled:opacity-35"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </motion.div>
            )}

            {/* Messages */}
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}

            {/* Streaming / typing indicator */}
            {isLoading && (
              <motion.div
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.22 }}
                className="flex gap-2.5"
              >
                <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-brand-400/30 bg-brand-500/12">
                  <Bot className="h-3.5 w-3.5 text-brand-400" />
                </div>
                <div className="rounded-2xl rounded-tl-sm border border-white/[0.08] bg-white/[0.03] px-4 py-3 text-sm leading-[1.7] text-foreground max-w-[84%]">
                  {streamingContent ? (
                    <span>
                      {streamingContent}
                      <motion.span
                        className="ml-0.5 inline-block h-[1.1em] w-[2px] translate-y-[2px] rounded-full bg-brand-400/70 align-middle"
                        animate={{ opacity: [1, 0, 1] }}
                        transition={{ duration: 0.9, repeat: Infinity, ease: "linear" }}
                      />
                    </span>
                  ) : (
                    <TypingWave />
                  )}
                </div>
              </motion.div>
            )}

          </div>

          {/* Input area */}
          <div className="border-t border-white/[0.06] bg-white/[0.01] px-5 py-4">
            <div className="flex items-end gap-2.5">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Klinik soru sorun… (Enter ile gönder)"
                rows={1}
                disabled={isLoading}
                className={cn(
                  "flex-1 resize-none rounded-xl border border-white/[0.09] bg-white/[0.03] px-4 py-2.5",
                  "text-sm text-foreground placeholder:text-foreground-muted/35",
                  "focus:outline-none focus:ring-1 focus:ring-brand-400/45 focus:border-brand-400/45",
                  "disabled:opacity-45 transition-all max-h-28 min-h-[42px]",
                )}
                style={{ fieldSizing: "content" } as React.CSSProperties}
              />
              <button
                type="button"
                onClick={() => send(input)}
                disabled={!input.trim() || isLoading}
                className={cn(
                  "flex h-[42px] w-[42px] shrink-0 items-center justify-center rounded-xl transition-all duration-200",
                  "bg-brand-500 text-white hover:bg-brand-400",
                  "shadow-[0_2px_14px_-3px_hsl(221_83%_53%/0.50)]",
                  "disabled:opacity-30 disabled:cursor-not-allowed disabled:shadow-none",
                )}
                aria-label="Gönder"
              >
                <Send className="h-4 w-4" />
              </button>
            </div>
            <p className="mt-2.5 text-center text-[11px] text-foreground-muted/35">
              Bu yanıtlar tıbbi teşhis veya tedavi tavsiyesi değildir.
            </p>
          </div>
        </div>

        {/* RIGHT: analysis context — desktop only */}
        <div className="hidden w-[212px] shrink-0 p-4 lg:block">
          <ContextPanel ctx={analysisContext} />
        </div>
      </div>
    </motion.div>
  );
}
