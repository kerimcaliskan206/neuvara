"use client";

import { Ban, Info } from "lucide-react";

import type { MedicalSemanticSignal } from "@/lib/api/types";

interface OodRejectionPanelProps {
  semantic: MedicalSemanticSignal | null;
}

export function OodRejectionPanel({ semantic }: OodRejectionPanelProps) {
  return (
    <div className="rounded-2xl glass-card border-warning-200/50 p-6 space-y-4 animate-fade-up">
      <div className="flex items-start gap-4">
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-warning-100/60 ring-1 ring-warning-200/40">
          <Ban className="h-6 w-6 text-warning-500" />
        </div>
        <div>
          <p className="text-base font-bold text-warning-700">
            Tıbbi Dışı Görüntü Tespit Edildi
          </p>
          <p className="mt-1 text-sm text-warning-600 leading-relaxed">
            Yüklenen görüntü tıbbi görüntüleme içeriği olarak tanımlanamadı.
            Analiz güvenli biçimde sonlandırıldı; klinik risk değerlendirmesi yapılmadı.
          </p>
        </div>
      </div>

      {semantic?.label && (
        <div className="rounded-xl bg-canvas/60 border border-warning-200/40 p-3.5">
          <p className="text-xs font-semibold text-warning-700 uppercase tracking-wider mb-1">
            Tespit Edilen İçerik Türü
          </p>
          <p className="text-sm font-medium text-foreground">{semantic.label}</p>
        </div>
      )}

      <div className="flex items-start gap-2 text-xs text-warning-700">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
        <p>
          Lütfen gerçek bir akciğer grafisi, BT taraması veya tıbbi görüntüleme
          dosyası yükleyiniz. Fotoğraf, çizim veya tıbbi olmayan görseller
          analize kabul edilmemektedir.
        </p>
      </div>
    </div>
  );
}
