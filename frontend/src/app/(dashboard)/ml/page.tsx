"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";

import { Alert } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import { mlApi } from "@/lib/api/ml";
import type { MLPredictionResponse, PatientInput } from "@/lib/api/types";

// ── Toggle pill ────────────────────────────────────────────────────────────────

function TogglePill({
  active,
  onToggle,
  children,
}: {
  active: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      className={cn(
        "rounded-lg border px-4 py-2 text-sm font-medium press-scale",
        "transition-[background-color,border-color,color] duration-200 ease-swift-out",
        active
          ? "border-brand-300 bg-brand-50 text-brand-700"
          : "border-border bg-surface text-foreground-secondary hover:border-border-strong",
      )}
    >
      {children}
    </button>
  );
}

// ── Symptom toggle card ────────────────────────────────────────────────────────

function SymptomToggle({
  label,
  sublabel,
  active,
  onToggle,
  highlight,
}: {
  label: string;
  sublabel: string;
  active: boolean;
  onToggle: () => void;
  highlight?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      className={cn(
        "flex w-full items-start gap-3 rounded-xl border p-3.5 text-left press-scale",
        "transition-[background-color,border-color,box-shadow] duration-200 ease-swift-out",
        active
          ? highlight
            ? "border-warning-200 bg-warning-50"
            : "border-brand-200 bg-brand-50"
          : "border-border bg-surface hover:border-border-strong hover:shadow-sm",
      )}
    >
      <div
        className={cn(
          "mt-0.5 h-4 w-4 shrink-0 rounded-full border-2 transition-colors",
          active
            ? highlight
              ? "border-warning-500 bg-warning-500"
              : "border-brand-600 bg-brand-600"
            : "border-border",
        )}
      />
      <div>
        <p className="text-sm font-semibold text-foreground">{label}</p>
        <p className="mt-0.5 text-xs text-foreground-muted">{sublabel}</p>
      </div>
    </button>
  );
}

// ── Result card ────────────────────────────────────────────────────────────────

function ResultCard({ result }: { result: MLPredictionResponse }) {
  const isPositive = result.prediction === 1;
  return (
    <div
      className={cn(
        "rounded-xl border p-5 space-y-4",
        isPositive
          ? "border-danger-100 bg-danger-50"
          : "border-success-100 bg-success-50",
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <p
            className={cn(
              "text-base font-bold",
              isPositive ? "text-danger-700" : "text-success-700",
            )}
          >
            {isPositive ? "HPS Pozitif" : "HPS Negatif"}
          </p>
          <p className="text-xs text-foreground-secondary mt-0.5">
            {result.model_name} · v{result.model_version}
          </p>
        </div>
        <span
          className={cn(
            "rounded-full px-3 py-1 text-sm font-bold",
            isPositive
              ? "bg-danger-100 text-danger-700"
              : "bg-success-100 text-success-700",
          )}
        >
          {result.label}
        </span>
      </div>

      {result.probability !== null && (
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs text-foreground-secondary">
            <span>Tahmin olasılığı</span>
            <span className="font-semibold tabular-nums">
              {(result.probability * 100).toFixed(1)}%
            </span>
          </div>
          <div className="h-2.5 w-full overflow-hidden rounded-full bg-white/70">
            <div
              className={cn(
                "h-full rounded-full",
                isPositive ? "bg-danger-500" : "bg-success-500",
              )}
              style={{
                width: `${result.probability * 100}%`,
                transition: "width 700ms cubic-bezier(0.2, 0, 0, 1)",
              }}
            />
          </div>
        </div>
      )}

      <div className="flex flex-wrap gap-3 text-xs text-foreground-secondary pt-1 border-t border-white/50">
        <span>Güven: <strong>{result.confidence}</strong></span>
        <span>·</span>
        <span>Çıkarım: <strong>{result.inference_duration_ms.toFixed(0)} ms</strong></span>
      </div>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

const REGIONS = ["north", "south", "east", "west", "central"] as const;
const REGION_TR: Record<string, string> = {
  north: "Kuzey", south: "Güney", east: "Doğu", west: "Batı", central: "Orta",
};
const SEASONS = ["spring", "summer", "fall", "winter"] as const;
const SEASON_TR: Record<string, string> = {
  spring: "İlkbahar", summer: "Yaz", fall: "Sonbahar", winter: "Kış",
};

interface FormState {
  age: number | "";
  gender: "M" | "F" | "";
  region: string;
  season: string;
  fever: boolean;
  myalgia: boolean;
  headache: boolean;
  thrombocytopenia: boolean;
  rodent_contact: boolean;
  outdoor_work: boolean;
  rodent_density: number | "";
  precipitation_mm: number | "";
  humidity_pct: number | "";
}

const DEFAULT_FORM: FormState = {
  age: "", gender: "", region: "", season: "",
  fever: false, myalgia: false, headache: false, thrombocytopenia: false,
  rodent_contact: false, outdoor_work: false,
  rodent_density: "", precipitation_mm: "", humidity_pct: "",
};

function formToPatient(f: FormState): PatientInput {
  return {
    age: f.age !== "" ? Number(f.age) : null,
    gender: f.gender || null,
    region: f.region || null,
    season: f.season || null,
    fever: f.fever ? 1 : 0,
    myalgia: f.myalgia ? 1 : 0,
    headache: f.headache ? 1 : 0,
    thrombocytopenia: f.thrombocytopenia ? 1 : 0,
    rodent_contact: f.rodent_contact ? 1 : 0,
    outdoor_work: f.outdoor_work ? 1 : 0,
    rodent_density: f.rodent_density !== "" ? Number(f.rodent_density) : null,
    precipitation_mm: f.precipitation_mm !== "" ? Number(f.precipitation_mm) : null,
    humidity_pct: f.humidity_pct !== "" ? Number(f.humidity_pct) : null,
  };
}

export default function MlPage() {
  const [form, setForm] = useState<FormState>({ ...DEFAULT_FORM });

  const prediction = useMutation<MLPredictionResponse, Error, PatientInput>({
    mutationFn: (patient) => mlApi.predict(patient),
  });

  function toggle<K extends keyof FormState>(key: K) {
    setForm((f) => ({ ...f, [key]: !f[key] }));
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    prediction.mutate(formToPatient(form));
  }

  function handleReset() {
    setForm({ ...DEFAULT_FORM });
    prediction.reset();
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6 pb-12">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-white">
          ML Risk Tahmini
        </h1>
        <p className="mt-1 text-sm text-white/65">
          Hasta verilerini girin. Eksik alanlar model tarafından eğitim istatistiklerine göre doldurulur.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Symptoms */}
        <div className="rounded-2xl glass-card-light p-5 space-y-4 animate-fade-up">
          <p className="text-xs font-semibold uppercase tracking-wider text-foreground-muted">
            Semptomlar
          </p>
          <div className="grid gap-3 sm:grid-cols-2">
            <SymptomToggle
              label="Ateş"
              sublabel="38°C veya üzeri"
              active={form.fever}
              onToggle={() => toggle("fever")}
            />
            <SymptomToggle
              label="Miyalji"
              sublabel="Kas ağrısı veya hassasiyeti"
              active={form.myalgia}
              onToggle={() => toggle("myalgia")}
            />
            <SymptomToggle
              label="Baş Ağrısı"
              sublabel="Şiddetli veya sürekli"
              active={form.headache}
              onToggle={() => toggle("headache")}
            />
            <SymptomToggle
              label="Trombositopeni"
              sublabel="Düşük trombosit sayısı — güçlü HPS göstergesi"
              active={form.thrombocytopenia}
              onToggle={() => toggle("thrombocytopenia")}
              highlight
            />
          </div>
        </div>

        {/* Demographic */}
        <div className="rounded-2xl glass-card-light p-5 space-y-4 animate-fade-up animate-delay-75">
          <p className="text-xs font-semibold uppercase tracking-wider text-foreground-muted">
            Demografik
          </p>
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="age">Yaş</Label>
              <Input
                id="age"
                type="number"
                min={0}
                max={120}
                placeholder="0–120"
                value={form.age}
                onChange={(e) =>
                  setForm((f) => ({
                    ...f,
                    age: e.target.value === "" ? "" : Number(e.target.value),
                  }))
                }
              />
            </div>

            <div className="space-y-2">
              <Label>Cinsiyet</Label>
              <div className="flex gap-2">
                {(["M", "F"] as const).map((g) => (
                  <TogglePill
                    key={g}
                    active={form.gender === g}
                    onToggle={() =>
                      setForm((f) => ({ ...f, gender: f.gender === g ? "" : g }))
                    }
                  >
                    {g === "M" ? "Erkek" : "Kadın"}
                  </TogglePill>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <Label>Bölge</Label>
              <div className="flex flex-wrap gap-2">
                {REGIONS.map((r) => (
                  <TogglePill
                    key={r}
                    active={form.region === r}
                    onToggle={() =>
                      setForm((f) => ({ ...f, region: f.region === r ? "" : r }))
                    }
                  >
                    {REGION_TR[r]}
                  </TogglePill>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <Label>Mevsim</Label>
              <div className="flex flex-wrap gap-2">
                {SEASONS.map((s) => (
                  <TogglePill
                    key={s}
                    active={form.season === s}
                    onToggle={() =>
                      setForm((f) => ({ ...f, season: f.season === s ? "" : s }))
                    }
                  >
                    {SEASON_TR[s]}
                  </TogglePill>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Exposure + Environmental */}
        <div className="rounded-2xl glass-card-light p-5 space-y-5 animate-fade-up animate-delay-150">
          <div className="space-y-3">
            <p className="text-xs font-semibold uppercase tracking-wider text-foreground-muted">
              Maruziyet
            </p>
            <div className="flex flex-wrap gap-3">
              <TogglePill
                active={form.rodent_contact}
                onToggle={() => toggle("rodent_contact")}
              >
                🐀 Kemirici Teması
              </TogglePill>
              <TogglePill
                active={form.outdoor_work}
                onToggle={() => toggle("outdoor_work")}
              >
                🌿 Dış Ortam Çalışması
              </TogglePill>
            </div>
          </div>

          <div className="space-y-3 border-t border-border pt-5">
            <p className="text-xs font-semibold uppercase tracking-wider text-foreground-muted">
              Çevresel
            </p>
            <div className="grid gap-4 sm:grid-cols-3">
              <div className="space-y-2">
                <Label htmlFor="rodent_density">Kemirici Yoğunluğu (0–10)</Label>
                <Input
                  id="rodent_density"
                  type="number"
                  min={0}
                  max={10}
                  step={0.5}
                  placeholder="0–10"
                  value={form.rodent_density}
                  onChange={(e) =>
                    setForm((f) => ({
                      ...f,
                      rodent_density: e.target.value === "" ? "" : Number(e.target.value),
                    }))
                  }
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="precipitation">Yağış (mm)</Label>
                <Input
                  id="precipitation"
                  type="number"
                  min={0}
                  placeholder="mm"
                  value={form.precipitation_mm}
                  onChange={(e) =>
                    setForm((f) => ({
                      ...f,
                      precipitation_mm: e.target.value === "" ? "" : Number(e.target.value),
                    }))
                  }
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="humidity">Nem (%)</Label>
                <Input
                  id="humidity"
                  type="number"
                  min={0}
                  max={100}
                  placeholder="0–100"
                  value={form.humidity_pct}
                  onChange={(e) =>
                    setForm((f) => ({
                      ...f,
                      humidity_pct: e.target.value === "" ? "" : Number(e.target.value),
                    }))
                  }
                />
              </div>
            </div>
          </div>
        </div>

        {/* Error */}
        {prediction.isError && (
          <Alert variant="danger" title="Tahmin başarısız">
            {prediction.error?.message ?? "Bilinmeyen hata."}
          </Alert>
        )}

        {/* Result */}
        {prediction.data && (
          <div className="animate-fade-up">
            <ResultCard result={prediction.data} />
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center justify-between gap-3">
          <Button type="button" variant="secondary" onClick={handleReset} size="sm">
            Temizle
          </Button>
          <Button type="submit" size="lg" isLoading={prediction.isPending} disabled={prediction.isPending}>
            Tahmin Et
          </Button>
        </div>
      </form>
    </div>
  );
}
