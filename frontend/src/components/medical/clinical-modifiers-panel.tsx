"use client";

import { Stethoscope } from "lucide-react";

import { cn } from "@/lib/utils";
import type { MedicalClinicalModifier } from "@/lib/api/types";

const SYMPTOM_LABELS: Record<string, string> = {
  fever:                "Ateş",
  cough:                "Öksürük",
  dyspnea:              "Dispne",
  shortness_of_breath:  "Nefes darlığı",
  chest_pain:           "Göğüs ağrısı",
  hemoptysis:           "Hemoptizi",
  tachypnea:            "Takipne",
  hypoxia:              "Hipoksi",
  fatigue:              "Yorgunluk",
  myalgia:              "Miyalji",
  night_sweats:         "Gece terlemesi",
  weight_loss:          "Kilo kaybı",
  wheezing:             "Hışıltı",
  productive_cough:     "Balgamlı öksürük",
};

const EXPOSURE_LABELS: Record<string, string> = {
  rodent_contact:     "Kemirgen teması",
  hospital:           "Hastane maruziyeti",
  sick_contact:       "Hasta ile temas",
  travel:             "Seyahat öyküsü",
  healthcare_worker:  "Sağlık çalışanı",
  immunocompromised:  "İmmün yetmezlik",
};

interface ClinicalModifiersPanelProps {
  clinical: MedicalClinicalModifier;
}

export function ClinicalModifiersPanel({ clinical }: ClinicalModifiersPanelProps) {
  if (!clinical.provided) return null;

  const isUpward  = clinical.delta_direction === "upward";
  const isNeutral = clinical.delta_direction === "neutral";

  return (
    <div className="rounded-2xl glass-card-light p-5 space-y-4 animate-fade-up animate-delay-200">
      <div className="flex items-center gap-2">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-brand-50">
          <Stethoscope className="h-3.5 w-3.5 text-brand-600" />
        </div>
        <p className="text-sm font-semibold text-foreground">Klinik Değerlendirme</p>
      </div>

      {/* Clinical impact — plain language only */}
      {!isNeutral && (
        <p className={cn(
          "text-sm leading-relaxed",
          isUpward ? "text-warning-700" : "text-success-700",
        )}>
          {isUpward
            ? "Bildirilen klinik bulgular pulmoner risk değerlendirmesini artırıcı yönde etkiliyor."
            : "Bildirilen klinik bulgular mevcut risk düzeyini sınırlayıcı yönde etkiliyor."}
        </p>
      )}

      {/* Symptoms */}
      {clinical.symptoms_flagged.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-wider text-foreground-muted">
            Bildirilen Semptomlar
          </p>
          <div className="flex flex-wrap gap-1.5">
            {clinical.symptoms_flagged.map((s) => (
              <span
                key={s}
                className="rounded-full border border-brand-100 bg-brand-50 px-2.5 py-0.5 text-xs font-medium text-brand-700"
              >
                {SYMPTOM_LABELS[s] ?? s}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Exposure */}
      {clinical.exposure_flagged && (
        <div className="space-y-1.5">
          <p className="text-xs font-semibold uppercase tracking-wider text-foreground-muted">
            Maruziyet Öyküsü
          </p>
          <span className="inline-flex rounded-full border border-warning-100 bg-warning-50 px-3 py-0.5 text-xs font-semibold text-warning-700">
            {EXPOSURE_LABELS[clinical.exposure_flagged] ?? clinical.exposure_flagged}
          </span>
        </div>
      )}

      {/* Contradiction — simplified, no technical terms */}
      {clinical.contradiction_detected && clinical.contradiction_note && (
        <p className="rounded-lg border border-warning-100 bg-warning-50 px-3 py-2.5 text-xs text-warning-700 leading-relaxed">
          {clinical.contradiction_note}
        </p>
      )}
    </div>
  );
}
