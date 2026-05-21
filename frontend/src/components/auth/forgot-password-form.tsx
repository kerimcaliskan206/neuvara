"use client";

import { motion } from "framer-motion";
import { AlertCircle, CheckCircle2, Mail } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

import { authApi } from "@/lib/api/auth";
import { ApiError } from "@/lib/api/client";
import { config } from "@/lib/config";
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

export function ForgotPasswordForm() {
  const [email,      setEmail]      = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [success,    setSuccess]    = useState(false);
  const [error,      setError]      = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);

    const target = `${config.api.v1}/auth/forgot-password`;
    console.info(`[ForgotPassword] → POST ${target}`, { email: email.trim() });

    try {
      await authApi.forgotPassword(email.trim());
      console.info("[ForgotPassword] ← 200 OK — backend accepted request");
      console.info(
        "[ForgotPassword] If email is registered, check backend logs for RESET PASSWORD URL.\n" +
        "  docker compose logs api --tail=20 | grep -A3 'RESET PASSWORD'"
      );
      setSuccess(true);
    } catch (err) {
      const apiErr = err instanceof ApiError ? err : null;
      const message = apiErr
        ? apiErr.message
        : err instanceof Error
        ? err.message
        : "İstek gönderilemedi.";

      console.error("[ForgotPassword] ← request failed", {
        status: apiErr?.status,
        message,
        body: apiErr?.body,
      });
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
            <p className="text-sm font-semibold text-emerald-300">İstek alındı</p>
            <p className="mt-0.5 text-xs text-emerald-300/70">
              Eğer <span className="font-medium">{email}</span> adresiyle kayıtlı bir hesap varsa,
              şifre sıfırlama bağlantısı sunucu loglarında görünecektir.
            </p>
          </div>
        </div>
        <p className="text-center text-sm text-white/32">
          <Link href="/login" className="font-medium text-blue-300 hover:text-blue-200 transition-colors">
            Giriş sayfasına dön
          </Link>
        </p>
      </div>
    );
  }

  return (
    <form className="space-y-4" onSubmit={onSubmit}>
      <div className="space-y-1.5">
        <label
          htmlFor="email"
          className="block text-[11px] font-semibold uppercase tracking-[0.12em] text-white/42"
        >
          E-posta adresi
        </label>
        <div className="relative">
          <input
            id="email"
            type="email"
            autoComplete="email"
            placeholder="ad@kurum.sağlık.tr"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            className={cn(INPUT_CLS, "pl-10")}
          />
          <Mail className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-white/25" />
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
        Sıfırlama Bağlantısı Gönder
      </motion.button>

      <p className="text-center text-sm text-white/32">
        <Link href="/login" className="font-medium text-blue-300 hover:text-blue-200 transition-colors">
          Giriş sayfasına dön
        </Link>
      </p>
    </form>
  );
}
