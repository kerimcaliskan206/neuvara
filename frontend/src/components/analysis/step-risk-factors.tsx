import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import { type RiskFactorData } from "@/stores/fusion-store";

const REGIONS = ["north", "south", "east", "west", "central"] as const;
const REGION_TR: Record<string, string> = {
  north: "Kuzey",
  south: "Güney",
  east: "Doğu",
  west: "Batı",
  central: "Orta",
};
const SEASONS = ["spring", "summer", "fall", "winter"] as const;
const SEASON_TR: Record<string, string> = {
  spring: "İlkbahar",
  summer: "Yaz",
  fall: "Sonbahar",
  winter: "Kış",
};

interface TogglePillProps {
  active: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}
function TogglePill({ active, onToggle, children }: TogglePillProps) {
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

interface StepRiskFactorsProps {
  data: RiskFactorData;
  onChange: (data: Partial<RiskFactorData>) => void;
  onNext: () => void;
  onBack: () => void;
}

export function StepRiskFactors({ data, onChange, onNext, onBack }: StepRiskFactorsProps) {
  return (
    <div className="space-y-7">
      <div>
        <h2 className="text-lg font-bold text-foreground">Risk Faktörleri</h2>
        <p className="mt-1 text-sm text-foreground-secondary">
          Demografik ve çevresel risk bilgilerini girin. Tüm alanlar isteğe bağlıdır.
        </p>
      </div>

      {/* Demographic */}
      <div className="space-y-4">
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
              value={data.age}
              onChange={(e) =>
                onChange({ age: e.target.value === "" ? "" : Number(e.target.value) })
              }
            />
          </div>

          <div className="space-y-2">
            <Label>Cinsiyet</Label>
            <div className="flex gap-2">
              {(["M", "F"] as const).map((g) => (
                <TogglePill
                  key={g}
                  active={data.gender === g}
                  onToggle={() => onChange({ gender: data.gender === g ? "" : g })}
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
                  active={data.region === r}
                  onToggle={() => onChange({ region: data.region === r ? "" : r })}
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
                  active={data.season === s}
                  onToggle={() => onChange({ season: data.season === s ? "" : s })}
                >
                  {SEASON_TR[s]}
                </TogglePill>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Exposure */}
      <div className="space-y-4">
        <p className="text-xs font-semibold uppercase tracking-wider text-foreground-muted">
          Maruziyet
        </p>
        <div className="flex flex-wrap gap-3">
          <TogglePill
            active={data.rodent_contact}
            onToggle={() => onChange({ rodent_contact: !data.rodent_contact })}
          >
            🐀 Kemirici Teması
          </TogglePill>
          <TogglePill
            active={data.outdoor_work}
            onToggle={() => onChange({ outdoor_work: !data.outdoor_work })}
          >
            🌿 Dış Ortam Çalışması
          </TogglePill>
        </div>
      </div>

      {/* Environmental */}
      <div className="space-y-4">
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
              value={data.rodent_density}
              onChange={(e) =>
                onChange({
                  rodent_density: e.target.value === "" ? "" : Number(e.target.value),
                })
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
              value={data.precipitation_mm}
              onChange={(e) =>
                onChange({
                  precipitation_mm: e.target.value === "" ? "" : Number(e.target.value),
                })
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
              value={data.humidity_pct}
              onChange={(e) =>
                onChange({
                  humidity_pct: e.target.value === "" ? "" : Number(e.target.value),
                })
              }
            />
          </div>
        </div>
      </div>

      <div className="flex justify-between gap-3">
        <Button variant="secondary" onClick={onBack}>
          Geri
        </Button>
        <Button onClick={onNext} size="lg">
          Devam: Görüntü
        </Button>
      </div>
    </div>
  );
}
