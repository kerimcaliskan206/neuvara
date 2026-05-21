"use client";

import { motion } from "framer-motion";
import { AlertCircle, CheckCircle2, Eye, EyeOff } from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useState } from "react";

import { authApi } from "@/lib/api/auth";
import { ApiError } from "@/lib/api/client";
import { cn } from "@/lib/utils";

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

export function ResetPasswordForm() {
  const searchParams = useSearchParams();
  const token        = searchParams.get("token") ?? "";

  const [newPassword,     setNewPassword]     = useState("");
  const [showPassword,    setShowPassword]    = useState(false);
  const [submitting,      setSubmitting]      = useState(false);
  const [success,         setSuccess]         = useState(false);
  const [error,           setError]           = useState<string | null>(null);

  if (!token) {
    return (
      <div className="flex items-start gap-2.5 rounded-xl border border-red-400/22 bg-red-500/10 px-4 py-3">
        <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-400" />
        <p className="text-sm leading-relaxed text-red-300">
          Geçersiz sıfırlama bağlantısı. Lütfen yeniden talep edin.
        </p>
      </div>
    );
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await authApi.resetPassword(token, newPassword);
      setSuccess(true);
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.message
          : err instanceof Error
          ? err.message
          : "Şifre güncellenemedi.";
      setError(message);
    } finally {
      setSubmitting(false);
    }
  }

  if (success) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3 rounded-xl border border-emerald-400/25 bg-emerald-500/10 px-4 py-4">
          <CheckCircle2 className="h-5 w-5 shrink-0 text-emerald-400" />
          <div>
            <p className="text-sm font-semibold text-emerald-300">Şifre güncellendi</p>
            <p className="mt-0.5 text-xs text-emerald-300/70">
              Yeni şifrenizle giriş yapabilirsiniz.
            </p>
          </div>
        </div>
        <p className="text-center text-sm text-white/32">
          <Link href="/login" className="font-medium text-blue-300 hover:text-blue-200 transition-colors">
            Giriş yap
          </Link>
        </p>
      </div>
    );
  }

  return (
    <form className="space-y-4" onSubmit={onSubmit}>
      <div className="space-y-1.5">
        <label
          htmlFor="new-password"
          className="block text-[11px] font-semibold uppercase tracking-[0.12em] text-white/42"
        >
          Yeni şifre
        </label>
        <div className="relative">
          <input
            id="new-password"
            type={showPassword ? "text" : "password"}
            autoComplete="new-password"
            placeholder="••••••••"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            required
            minLength={8}
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
        <p className="text-[11px] text-white/25">
          En az 8 karakter; büyük harf, küçük harf, rakam ve özel karakter içermelidir.
        </p>
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
        Şifremi Güncelle
      </motion.button>
    </form>
  );
}
