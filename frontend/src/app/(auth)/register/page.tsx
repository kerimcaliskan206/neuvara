import type { Metadata } from "next";

import { RegisterForm } from "@/components/auth/register-form";

export const metadata: Metadata = { title: "Kayıt Ol — NEURAVA" };

export default function RegisterPage() {
  return (
    <div className="space-y-6">
      <div>
        <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-[0.16em] text-white/32">
          Yeni Hesap
        </p>
        <h2 className="text-2xl font-bold tracking-tight text-white">
          Hesap oluşturun
        </h2>
        <p className="mt-1 text-sm text-white/45">
          Platforma erişmek için kurumsal e-postanızla kayıt olun.
        </p>
      </div>
      <RegisterForm />
    </div>
  );
}
