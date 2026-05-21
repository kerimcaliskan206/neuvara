"use client";

import { motion } from "framer-motion";
import { AlertCircle, Eye, EyeOff } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { useAuthStore } from "@/stores/auth-store";
import { cn } from "@/lib/utils";

// ── Shared input style ────────────────────────────────────────────────────────

const INPUT_CLS = cn(
  "w-full h-12 rounded-xl px-4 text-sm text-white",
  "bg-white/[0.05] border border-white/[0.10]",
  "placeholder:text-white/22",
  "transition-all duration-150",
  "hover:bg-white/[0.07] hover:border-white/[0.18]",
  "focus:outline-none focus:bg-white/[0.07]",
  "focus:border-blue-400/50 focus:ring-1 focus:ring-blue-400/30",
  "disabled:opacity-40 disabled:cursor-not-allowed",
);

const LABEL_CLS = "block text-[11px] font-semibold uppercase tracking-[0.12em] text-white/42";

// ── Login form ────────────────────────────────────────────────────────────────

export function LoginForm() {
  const router     = useRouter();
  const login      = useAuthStore((s) => s.login);
  const error      = useAuthStore((s) => s.error);
  const clearError = useAuthStore((s) => s.clearError);

  const [email,        setEmail]        = useState("");
  const [password,     setPassword]     = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [submitting,   setSubmitting]   = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    clearError();
    setSubmitting(true);
    try {
      await login(email, password);
      router.replace("/dashboard");
    } catch {
      // Error surfaced via store
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form className="space-y-4" onSubmit={onSubmit}>
      {/* Email */}
      <div className="space-y-1.5">
        <label htmlFor="email" className={LABEL_CLS}>E-posta adresi</label>
        <input
          id="email"
          type="email"
          autoComplete="email"
          placeholder="ad@kurum.sağlık.tr"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          disabled={submitting}
          className={INPUT_CLS}
        />
      </div>

      {/* Password with "forgot" link */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <label htmlFor="password" className={LABEL_CLS}>Şifre</label>
          <Link
            href="/forgot-password"
            className="text-[11px] font-medium text-blue-300/70 hover:text-blue-300 transition-colors"
          >
            Şifremi unuttum?
          </Link>
        </div>
        <div className="relative">
          <input
            id="password"
            type={showPassword ? "text" : "password"}
            autoComplete="current-password"
            placeholder="••••••••"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            disabled={submitting}
            className={cn(INPUT_CLS, "pr-11")}
          />
          <button
            type="button"
            tabIndex={-1}
            onClick={() => setShowPassword((s) => !s)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-white/30 hover:text-white/55 transition-colors"
            aria-label={showPassword ? "Şifreyi gizle" : "Şifreyi göster"}
          >
            {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
          </button>
        </div>
      </div>

      {error && (
        <div className="flex items-start gap-2.5 rounded-xl border border-red-400/22 bg-red-500/10 px-4 py-3">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-400" />
          <p className="text-sm leading-relaxed text-red-300">{error}</p>
        </div>
      )}

      <motion.button
        type="submit"
        disabled={submitting}
        whileHover={{ scale: 1.008, filter: "brightness(1.10)" }}
        whileTap={{ scale: 0.975 }}
        transition={{ duration: 0.14 }}
        className="relative mt-2 flex h-12 w-full items-center justify-center gap-2.5 rounded-xl text-sm font-semibold text-white disabled:opacity-40 disabled:cursor-not-allowed"
        style={{
          background: "linear-gradient(135deg, hsl(221 83% 53%), hsl(258 84% 65%))",
          boxShadow: "0 4px 24px -4px hsl(221 83% 53% / 0.48)",
        }}
      >
        {submitting && (
          <span
            aria-hidden
            className="h-4 w-4 rounded-full border-2 border-white/30 border-t-white animate-spin"
          />
        )}
        Giriş Yap
      </motion.button>

      <p className="text-center text-sm text-white/32">
        Hesabınız yok mu?{" "}
        <Link href="/register" className="font-medium text-blue-300 hover:text-blue-200 transition-colors">
          Kayıt olun
        </Link>
      </p>
    </form>
  );
}
