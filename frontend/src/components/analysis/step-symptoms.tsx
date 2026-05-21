import { AlertTriangle, CheckCircle2, Thermometer } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { type SymptomData } from "@/stores/fusion-store";

interface ToggleCardProps {
  id: keyof SymptomData;
  label: string;
  sublabel: string;
  active: boolean;
  onToggle: (id: keyof SymptomData) => void;
  severity?: "high" | "normal";
}

function ToggleCard({ id, label, sublabel, active, onToggle, severity = "normal" }: ToggleCardProps) {
  return (
    <button
      type="button"
      onClick={() => onToggle(id)}
      className={cn(
        "flex w-full items-start gap-4 rounded-xl border p-4 text-left press-scale",
        "transition-[background-color,border-color,box-shadow] duration-200 ease-swift-out",
        active
          ? severity === "high"
            ? "border-warning-200 bg-warning-50"
            : "border-brand-200 bg-brand-50"
          : "border-border bg-surface hover:border-border-strong hover:shadow-sm",
      )}
    >
      <div
        className={cn(
          "mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border-2 transition-colors",
          active
            ? severity === "high"
              ? "border-warning-500 bg-warning-500"
              : "border-brand-600 bg-brand-600"
            : "border-border",
        )}
      >
        {active ? <CheckCircle2 className="h-3.5 w-3.5 text-white" /> : null}
      </div>
      <div>
        <p className={cn("text-sm font-semibold", active ? "text-foreground" : "text-foreground-secondary")}>
          {label}
        </p>
        <p className="mt-0.5 text-xs text-foreground-muted">{sublabel}</p>
      </div>
      {severity === "high" && (
        <AlertTriangle className={cn("ml-auto mt-0.5 h-4 w-4 shrink-0", active ? "text-warning-500" : "text-border")} />
      )}
    </button>
  );
}

const symptoms: {
  id: keyof SymptomData;
  label: string;
  sublabel: string;
  severity?: "high" | "normal";
}[] = [
  {
    id: "fever",
    label: "Ateş",
    sublabel: "38°C veya üzeri vücut ısısı",
  },
  {
    id: "myalgia",
    label: "Miyalji",
    sublabel: "Kas ağrısı veya hassasiyeti",
  },
  {
    id: "headache",
    label: "Baş Ağrısı",
    sublabel: "Şiddetli veya sürekli baş ağrısı",
  },
  {
    id: "thrombocytopenia",
    label: "Trombositopeni",
    sublabel: "Düşük trombosit sayısı — güçlü HPS göstergesi",
    severity: "high",
  },
];

interface StepSymptomsProps {
  data: SymptomData;
  onChange: (data: Partial<SymptomData>) => void;
  onNext: () => void;
}

export function StepSymptoms({ data, onChange, onNext }: StepSymptomsProps) {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-bold text-foreground">Mevcut Semptomlar</h2>
        <p className="mt-1 text-sm text-foreground-secondary">
          Hastanın gösterdiği belirtileri seçin. Trombositopeni güçlü bir HPS göstergesidir.
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        {symptoms.map((s) => (
          <ToggleCard
            key={s.id}
            id={s.id}
            label={s.label}
            sublabel={s.sublabel}
            active={data[s.id]}
            severity={s.severity}
            onToggle={(id) => onChange({ [id]: !data[id] })}
          />
        ))}
      </div>

      <div className="flex items-center gap-3 rounded-xl bg-brand-50 p-4">
        <Thermometer className="h-5 w-5 shrink-0 text-brand-500" />
        <p className="text-xs text-brand-800">
          Seçili semptom sayısı: <strong>{Object.values(data).filter(Boolean).length}</strong> / 4.
          Tüm alanlar isteğe bağlıdır; eksik veriler ML modeli tarafından
          eğitim istatistiklerine göre doldurulur.
        </p>
      </div>

      <div className="flex justify-end">
        <Button onClick={onNext} size="lg">
          Devam: Risk Faktörleri
        </Button>
      </div>
    </div>
  );
}
