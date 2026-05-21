import type { Metadata } from "next";

import { ForgotPasswordForm } from "@/components/auth/forgot-password-form";

export const metadata: Metadata = { title: "Şifremi Unuttum — NEURAVA" };

export default function ForgotPasswordPage() {
  return (
    <div className="space-y-6">
      <div>
        <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-[0.16em] text-white/32">
          Şifre Sıfırlama
        </p>
        <h2 className="text-2xl font-bold tracking-tight text-white">
          Şifremi unuttum
        </h2>
        <p className="mt-1 text-sm text-white/45">
          E-posta adresinize sıfırlama bağlantısı göndereceğiz.
        </p>
      </div>
      <ForgotPasswordForm />
    </div>
  );
}
