import { Suspense } from "react";
import type { Metadata } from "next";

import { ResetPasswordForm } from "@/components/auth/reset-password-form";

export const metadata: Metadata = { title: "Şifre Güncelle — NEURAVA" };

export default function ResetPasswordPage() {
  return (
    <div className="space-y-6">
      <div>
        <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-[0.16em] text-white/32">
          Şifre Sıfırlama
        </p>
        <h2 className="text-2xl font-bold tracking-tight text-white">
          Yeni şifre belirle
        </h2>
        <p className="mt-1 text-sm text-white/45">
          Hesabınız için güçlü bir şifre oluşturun.
        </p>
      </div>
      <Suspense>
        <ResetPasswordForm />
      </Suspense>
    </div>
  );
}
