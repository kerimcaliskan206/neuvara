"use client";

import { motion } from "framer-motion";
import { AlertCircle, CheckCircle2, Eye, EyeOff } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { useAuthStore } from "@/stores/auth-store";
import { cn } from "@/lib/utils";

// ── Dark glass input (auth-only) ──────────────────────────────────────────────

function AuthField({
  label, hint, id, type, autoComplete, placeholder,
  value, onChange, required, minLength, maxLength, disabled,
}: {
  label: string; hint?: string; id: string; type: string;
  autoComplete?: string; placeholder?: string;
  value: string; onChange: (v: string) => void;
  required?: boolean; minLength?: number; maxLength?: number; disabled?: boolean;
}) {
  const [show, setShow] = useState(false);
  const isPassword = type === "password";
  const inputType  = isPassword ? (show ? "text" : "password") : type;

  return (
    <div className="space-y-1.5">
      <label
        htmlFor={id}
        className="block text-[11px] font-semibold uppercase tracking-[0.12em] text-white/42"
      >
        {label}
      </label>
      <div className="relative">
        <input
          id={id}
          type={inputType}
          autoComplete={autoComplete}
          placeholder={placeholder}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          required={required}
          minLength={minLength}
          maxLength={maxLength}
          disabled={disabled}
          className={cn(
            "w-full h-12 rounded-xl px-4 text-sm text-white",
            "bg-white/[0.05] border border-white/[0.10]",
            "placeholder:text-white/22",
            "transition-all duration-150",
            "hover:bg-white/[0.07] hover:border-white/[0.18]",
            "focus:outline-none focus:bg-white/[0.07]",
            "focus:border-blue-400/50 focus:ring-1 focus:ring-blue-400/30",
            "disabled:opacity-40 disabled:cursor-not-allowed",
            isPassword && "pr-11",
          )}
        />
        {isPassword && (
          <button
            type="button"
            tabIndex={-1}
            onClick={() => setShow((s) => !s)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-white/30 hover:text-white/55 transition-colors"
            aria-label={show ? "Şifreyi gizle" : "Şifreyi göster"}
          >
            {show ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
          </button>
        )}
      </div>
      {hint && <p className="text-[11px] text-white/25">{hint}</p>}
    </div>
  );
}

// ── Register form ─────────────────────────────────────────────────────────────

const PASSWORD_RULE =
  "En az 8 karakter; büyük harf, küçük harf, rakam ve özel karakter içermelidir.";

export function RegisterForm() {
  const router     = useRouter();
  const register   = useAuthStore((s) => s.register);
  const error      = useAuthStore((s) => s.error);
  const clearError = useAuthStore((s) => s.clearError);

  const [username,   setUsername]   = useState("");
  const [email,      setEmail]      = useState("");
  const [password,   setPassword]   = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [success,    setSuccess]    = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    clearError();
    setSubmitting(true);
    try {
      await register(username.trim(), email.trim(), password);
      setSuccess(true);
      setTimeout(() => router.replace("/login"), 1200);
    } catch {
      // Error rendered via store
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form className="space-y-4" onSubmit={onSubmit}>
      <AuthField
        label="Kullanıcı adı"
        id="username"
        type="text"
        autoComplete="username"
        placeholder="drkaya"
        hint="3–50 karakter."
        value={username}
        onChange={setUsername}
        required minLength={3} maxLength={50}
        disabled={submitting}
      />
      <AuthField
        label="E-posta adresi"
        id="email"
        type="email"
        autoComplete="email"
        placeholder="ad@kurum.sağlık.tr"
        value={email}
        onChange={setEmail}
        required
        disabled={submitting}
      />
      <AuthField
        label="Şifre"
        id="password"
        type="password"
        autoComplete="new-password"
        placeholder="••••••••"
        hint={PASSWORD_RULE}
        value={password}
        onChange={setPassword}
        required minLength={8}
        disabled={submitting}
      />

      {error && (
        <div className="flex items-start gap-2.5 rounded-xl border border-red-400/22 bg-red-500/10 px-4 py-3">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-400" />
          <p className="text-sm leading-relaxed text-red-300">{error}</p>
        </div>
      )}
      {success && (
        <div className="flex items-center gap-2.5 rounded-xl border border-emerald-400/25 bg-emerald-500/10 px-4 py-3">
          <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-400" />
          <p className="text-sm text-emerald-300">Kayıt başarılı — yönlendiriliyorsunuz…</p>
        </div>
      )}

      {/* Premium gradient submit button */}
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
        {submitting ? (
          <span
            aria-hidden
            className="h-4 w-4 rounded-full border-2 border-white/30 border-t-white animate-spin"
          />
        ) : null}
        Hesap Oluştur
      </motion.button>

      <p className="text-center text-sm text-white/32">
        Zaten hesabınız var mı?{" "}
        <Link
          href="/login"
          className="font-medium text-blue-300 hover:text-blue-200 transition-colors"
        >
          Giriş yapın
        </Link>
      </p>
    </form>
  );
}
