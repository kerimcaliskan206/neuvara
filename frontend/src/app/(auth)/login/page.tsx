import type { Metadata } from "next";

import { LoginForm } from "@/components/auth/login-form";

export const metadata: Metadata = { title: "Giriş Yap — NEURAVA" };

export default function LoginPage() {
  return (
    <div className="space-y-6">
      <div>
        <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-[0.16em] text-white/32">
          Kimlik Doğrulama
        </p>
        <h2 className="text-2xl font-bold tracking-tight text-white">
          Tekrar hoş geldiniz
        </h2>
        <p className="mt-1 text-sm text-white/45">
          Kurumsal hesabınızla erişim sağlayın.
        </p>
      </div>
      <LoginForm />
    </div>
  );
}
