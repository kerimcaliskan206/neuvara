import type { ClinicalContextRequest, MedicalRiskAssessment, MedicalRiskTier } from "@/lib/api/types";

// ── Result type ───────────────────────────────────────────────────────────────

export interface ClinicalAnalysisResult {
  session_id: string;
  timestamp: string;
  source: "clinical_only";
  risk: MedicalRiskAssessment;
  requires_immediate_action: boolean;
  summary: string;
  reasoning_bullets: string[];
  clinical_summary: string;
  symptoms_provided: string[];
  exposure_provided: string | null;
  severity_provided: string | null;
  duration_days: number | null;
}

// ── Scoring tables ────────────────────────────────────────────────────────────

const SYMPTOM_WEIGHTS: Record<string, number> = {
  hypoxia:              0.25,
  hemoptysis:           0.22,
  dyspnea:              0.18,
  shortness_of_breath:  0.18,
  tachypnea:            0.15,
  chest_pain:           0.13,
  fever:                0.12,
  productive_cough:     0.10,
  night_sweats:         0.09,
  weight_loss:          0.08,
  wheezing:             0.08,
  cough:                0.07,
  fatigue:              0.06,
  myalgia:              0.05,
};

const EXPOSURE_WEIGHTS: Record<string, number> = {
  rodent_contact:     0.22,
  immunocompromised:  0.15,
  hospital:           0.10,
  sick_contact:       0.08,
  travel:             0.05,
  healthcare_worker:  0.05,
};

const SEVERITY_MULT: Record<string, number> = {
  severe:   1.30,
  moderate: 1.00,
  mild:     0.72,
};

// Phase 22 — structured clinical signal weights (null → 0.0 = neutral)
const RESPIRATORY_WEIGHTS: Record<string, number> = {
  normal: 0.0, mild: 0.15, severe: 0.30,
};

const OXYGENATION_WEIGHTS: Record<string, number> = {
  normal: 0.0, mild_drop: 0.12, severe_drop: 0.28,
};

const FEVER_SEVERITY_WEIGHTS: Record<string, number> = {
  none: 0.0, mild: 0.05, moderate: 0.10, high: 0.18,
};

const WORSENING_WEIGHTS: Record<string, number> = {
  none: 0.0, some: 0.10, rapid_48h: 0.25,  // rapid_48h raised: HIGH priority
};

const RODENT_LEVEL_WEIGHTS: Record<string, number> = {
  none: 0.0, unsure: 0.02, rural_env: 0.04, possible_contact: 0.08,  // reduced: LOW priority (differential-shaping only)
};

const DURATION_TIER_WEIGHTS: Record<string, number> = {
  "1_2_days": 0.02, "3_7_days": 0.05, over_1_week: 0.10,
};

const RESPIRATORY_SEVERITY_MULT: Record<string, number> = {
  normal: 0.35, mild: 0.70, severe: 1.0,
};

const TIER_THRESHOLDS = {
  LOW_upper:                    0.35,
  MODERATE_upper:               0.60,
  HIGH_DIFFERENTIAL_RISK_upper: 0.80,
} as const;

// ── Helpers ───────────────────────────────────────────────────────────────────

function resolveTier(score: number): MedicalRiskTier {
  if (score < TIER_THRESHOLDS.LOW_upper)                    return "LOW";
  if (score < TIER_THRESHOLDS.MODERATE_upper)               return "MODERATE";
  if (score < TIER_THRESHOLDS.HIGH_DIFFERENTIAL_RISK_upper) return "HIGH_DIFFERENTIAL_RISK";
  return "CRITICAL_PULMONARY_RISK";
}

const TIER_SUMMARIES: Record<MedicalRiskTier, string> = {
  LOW:
    "Mevcut klinik bulgular pulmoner patoloji açısından belirgin bir risk göstermemektedir. Semptomlar takip altında tutulmalıdır.",
  MODERATE:
    "Klinik bulgular pulmoner enfeksiyon açısından dikkat gerektirmektedir. Klinisyen değerlendirmesi önerilir.",
  HIGH_DIFFERENTIAL_RISK:
    "Klinik tablo ciddi pulmoner tutulumla uyumludur. İleri tetkik ve klinisyen değerlendirmesi gerekmektedir.",
  CRITICAL_PULMONARY_RISK:
    "Klinik bulgular kritik düzeyde pulmoner risk işaret etmektedir. Acil klinik değerlendirme gereklidir.",
};

const SYMPTOM_BULLETS: Record<string, string> = {
  hypoxia:             "Hipoksi, ciddi pulmoner kompromize işaret eden kritik bir bulgudur",
  hemoptysis:          "Hemoptizi, pulmoner patoloji açısından öncelikli değerlendirme gerektiren önemli bir bulgudur",
  dyspnea:             "Dispne, ciddi pulmoner tutulumun belirtisi olabilir",
  shortness_of_breath: "Nefes darlığı, solunum fonksiyon bozukluğunu düşündürmektedir",
  tachypnea:           "Takipne, solunum güçlüğünün nesnel bir göstergesidir",
  chest_pain:          "Göğüs ağrısı plevral veya pulmoner tutulumu düşündürmektedir",
  fever:               "Ateş varlığı, pulmoner enfeksiyon olasılığını artırmaktadır",
  productive_cough:    "Balgamlı öksürük, akut alt solunum yolu enfeksiyonu ile uyumludur",
  night_sweats:        "Gece terlemesi, kronik veya sistemik enfeksiyon sürecine işaret edebilir",
  weight_loss:         "Kilo kaybı, kronik pulmoner sürecin sistemik yansıması olabilir",
  wheezing:            "Hışıltı, obstrüktif veya enflamatuvar solunum yolu hastalığını düşündürmektedir",
  cough:               "Öksürük, alt solunum yolu tutulumunun sık görülen belirtisidir",
  fatigue:             "Yorgunluk, sistemik enfeksiyonun eşlik eden bulgusudur",
  myalgia:             "Miyalji, viral pulmoner enfeksiyon tablolarında sıkça görülür",
};

const EXPOSURE_BULLETS: Record<string, string> = {
  rodent_contact:    "Kemirgen teması, Hantavirüs Pulmoner Sendromu için önemli risk faktörüdür",
  immunocompromised: "İmmün yetmezlik durumu, atipik pulmoner enfeksiyon riskini belirgin şekilde artırmaktadır",
  hospital:          "Hastane maruziyeti, nozokomiyal pulmoner enfeksiyon riskini artırmaktadır",
  sick_contact:      "Hasta ile yakın temas, bulaşıcı pulmoner enfeksiyon riskini artırmaktadır",
  travel:            "Seyahat öyküsü, endemik bölge kaynaklı pulmoner patoloji olasılığını artırmaktadır",
  healthcare_worker: "Sağlık çalışanı olmak, mesleki pulmoner maruziyet riskini artırmaktadır",
};

const SYMPTOM_LABELS: Record<string, string> = {
  fever: "ateş", cough: "öksürük", dyspnea: "dispne",
  shortness_of_breath: "nefes darlığı", chest_pain: "göğüs ağrısı",
  hemoptysis: "hemoptizi", tachypnea: "takipne", hypoxia: "hipoksi",
  fatigue: "yorgunluk", myalgia: "miyalji", night_sweats: "gece terlemesi",
  weight_loss: "kilo kaybı", wheezing: "hışıltı", productive_cough: "balgamlı öksürük",
};

const EXPOSURE_LABELS: Record<string, string> = {
  rodent_contact: "kemirgen teması", hospital: "hastane maruziyeti",
  sick_contact: "hasta ile temas", travel: "seyahat öyküsü",
  healthcare_worker: "sağlık çalışanı", immunocompromised: "immün yetmezlik",
};

// eslint-disable-next-line @typescript-eslint/no-unused-vars
const SEVERITY_LABELS: Record<string, string> = {
  mild: "hafif", moderate: "orta", severe: "ağır",
};

function buildDifferentials(ctx: ClinicalContextRequest, tier: MedicalRiskTier, symptoms: string[]): string[] {
  const diffs: string[] = [];

  const hasPulmonaryKeySymptom =
    symptoms.includes("hemoptysis") ||
    symptoms.includes("hypoxia") ||
    (symptoms.includes("fever") &&
      (symptoms.includes("dyspnea") || symptoms.includes("shortness_of_breath")));

  // HPS differential — direct contact or new level-based rodent exposure
  const directRodent = ctx.exposure_history === "rodent_contact";
  const levelRodent  = ctx.rodent_exposure_level === "possible_contact" || ctx.rodent_exposure_level === "rural_env";
  if ((directRodent || levelRodent) && hasPulmonaryKeySymptom && tier !== "LOW") {
    diffs.push(
      directRodent
        ? "Hantavirüs Pulmoner Sendromu ⚑ kemirgen teması"
        : "Hantavirüs Pulmoner Sendromu (olası — kırsal ortam)"
    );
  }

  diffs.push("Toplum kökenli pnömoni");

  if (tier === "HIGH_DIFFERENTIAL_RISK" || tier === "CRITICAL_PULMONARY_RISK") {
    diffs.push("ARDS / Akut akciğer hasarı");
  }
  if (symptoms.includes("hemoptysis")) {
    diffs.push("Pulmoner tüberküloz");
  }
  if (ctx.exposure_history === "immunocompromised" || ctx.immunocompromised) {
    diffs.push("Fırsatçı pulmoner enfeksiyon");
  }
  if (tier === "LOW" || tier === "MODERATE") {
    diffs.push("Viral üst solunum yolu enfeksiyonu");
  }

  // Age-based differentials
  if (ctx.age_group === "older_adult" || ctx.age_group === "elderly") {
    if (tier !== "LOW") diffs.push("Aspirasyon pnömonisi");
  } else if ((ctx.age_group === "adolescent" || ctx.age_group === "young_adult") && tier !== "LOW") {
    if (!diffs.some(d => d.includes("Mycoplasma"))) {
      diffs.push("Mycoplasma pnömonisi (yürüyen pnömoni)");
    }
  }

  return diffs.slice(0, 5);
}

// ── Main export ───────────────────────────────────────────────────────────────

export function computeClinicalRisk(ctx: ClinicalContextRequest): ClinicalAnalysisResult {
  const symptoms = ctx.symptoms          ?? [];
  const exposure = ctx.exposure_history  ?? null;
  const severity = ctx.severity          ?? null;
  const duration = ctx.duration_days     ?? null;

  // ── Symptom score (capped at 0.65) ────────────────────────────────────────
  const rawSymptomScore = symptoms.reduce((sum, s) => sum + (SYMPTOM_WEIGHTS[s] ?? 0), 0);
  const symptomScore    = Math.min(rawSymptomScore, 0.65);

  // ── Exposure score ────────────────────────────────────────────────────────
  const exposureScore = exposure ? (EXPOSURE_WEIGHTS[exposure] ?? 0) : 0;

  // ── Severity multiplier: respiratory_severity (Phase 22) > legacy severity ─
  const severityMult = ctx.respiratory_severity != null
    ? (RESPIRATORY_SEVERITY_MULT[ctx.respiratory_severity] ?? 1.0)
    : (severity ? (SEVERITY_MULT[severity] ?? 1.0) : 1.0);

  // ── Phase 22 structured signals (null → 0.0 = neutral) ───────────────────
  const respScore    = RESPIRATORY_WEIGHTS[ctx.respiratory_severity ?? ""]   ?? 0;
  const oxyScore     = OXYGENATION_WEIGHTS[ctx.oxygenation_context ?? ""]    ?? 0;
  const feverScore   = FEVER_SEVERITY_WEIGHTS[ctx.fever_severity ?? ""]      ?? 0;
  const worsenScore  = WORSENING_WEIGHTS[ctx.recent_worsening ?? ""]         ?? 0;
  const rodentScore  = RODENT_LEVEL_WEIGHTS[ctx.rodent_exposure_level ?? ""] ?? 0;

  // Anti-stacking cap on structured signals
  const structuredSum = Math.min(0.50, respScore + oxyScore + feverScore + worsenScore);

  // ── Duration boost ────────────────────────────────────────────────────────
  let durationBoost = 0;
  if (ctx.symptom_duration_tier) {
    durationBoost = DURATION_TIER_WEIGHTS[ctx.symptom_duration_tier] ?? 0;
  } else if (duration != null) {
    if (duration > 14)      durationBoost = 0.06;
    else if (duration > 7)  durationBoost = 0.04;
    else if (duration > 3)  durationBoost = 0.02;
  }

  // ── Combine into score ────────────────────────────────────────────────────
  let score = Math.min(1.0,
    (symptomScore * severityMult)
    + (exposureScore * 0.50)
    + rodentScore
    + structuredSum
    + durationBoost
  );

  // HPS pattern floor: rodent exposure (direct or level) + pulmonary key symptom
  const hasPulmonaryKeySymptom =
    symptoms.includes("hemoptysis") ||
    symptoms.includes("hypoxia") ||
    (symptoms.includes("fever") &&
      (symptoms.includes("dyspnea") || symptoms.includes("shortness_of_breath")));
  const hasRodentRisk =
    exposure === "rodent_contact" ||
    ctx.rodent_exposure_level === "possible_contact" ||
    ctx.rodent_exposure_level === "rural_env";
  if (hasRodentRisk && hasPulmonaryKeySymptom) {
    score = Math.max(score, 0.60);
  }

  // Oxygenation floor: severe_drop alone → minimum MODERATE
  if (ctx.oxygenation_context === "severe_drop") {
    score = Math.max(score, 0.35);
  }

  score = Math.round(score * 10000) / 10000;

  // ── Derive age_group from numeric age ─────────────────────────────────────
  const ageGroup = ctx.age_group ?? (ctx.age != null ? _mapAgeToGroup(ctx.age) : null);
  const resolvedCtx: ClinicalContextRequest = { ...ctx, age_group: ageGroup ?? ctx.age_group };

  const tier = resolveTier(score);
  const thresholdValues = [TIER_THRESHOLDS.LOW_upper, TIER_THRESHOLDS.MODERATE_upper, TIER_THRESHOLDS.HIGH_DIFFERENTIAL_RISK_upper];
  const boundaryProximity = Math.min(...thresholdValues.map((t) => Math.abs(score - t)));
  const nearBoundary      = boundaryProximity < 0.05;

  // ── Reasoning bullets ─────────────────────────────────────────────────────
  const bullets: string[] = [];
  const sortedSymptoms = [...symptoms].sort((a, b) => (SYMPTOM_WEIGHTS[b] ?? 0) - (SYMPTOM_WEIGHTS[a] ?? 0));
  for (const s of sortedSymptoms.slice(0, 4)) {
    if (SYMPTOM_BULLETS[s]) bullets.push(SYMPTOM_BULLETS[s]);
  }
  if (exposure && EXPOSURE_BULLETS[exposure]) bullets.push(EXPOSURE_BULLETS[exposure]);

  if (ctx.respiratory_severity === "severe") {
    bullets.push("Ciddi nefes alma güçlüğü, belirgin pulmoner yük işaretçisidir");
  } else if (ctx.respiratory_severity === "mild") {
    bullets.push("Hafif nefes alma güçlüğü, dikkat gerektiren pulmoner bulgu");
  } else if (severity === "severe" && symptoms.length >= 2) {
    bullets.push("Semptomların ağır seyretmesi, klinik tablonun ciddi olduğuna işaret etmektedir");
  }

  if (ctx.oxygenation_context === "severe_drop") {
    bullets.push("Dinlenme halinde bile nefes darlığı, ciddi pulmoner kompromizi düşündürmektedir");
  } else if (ctx.oxygenation_context === "mild_drop") {
    bullets.push("Azalmış nefes kapasitesi, pulmoner işlev bozukluğunu göstermektedir");
  }

  if (ctx.recent_worsening === "rapid_48h") {
    bullets.push("Son 48 saatte hızlı kötüleşme, akut pulmoner süreçle uyumludur");
  } else if (ctx.recent_worsening === "some") {
    bullets.push("Semptomların kötüleşme seyri klinik dikkat gerektirmektedir");
  }

  if (ctx.fever_severity === "high") {
    bullets.push("Yüksek ateş varlığı, aktif enfeksiyöz süreci desteklemektedir");
  }

  if (hasRodentRisk && hasPulmonaryKeySymptom) {
    bullets.push("Kemirgen maruziyeti ve pulmoner semptom kombinasyonu HPS açısından dikkat gerektirir");
  }

  if (ctx.symptom_duration_tier === "over_1_week") {
    bullets.push("1 haftayı aşan semptom süresi, süreğen pulmoner tutulumu düşündürmektedir");
  } else if (duration != null && duration > 7) {
    bullets.push(`Semptomların ${duration} gündür sürmesi, süreğen pulmoner tutulumu düşündürmektedir`);
  }

  // ── Clinical summary string ───────────────────────────────────────────────
  const parts: string[] = [];
  if (symptoms.length > 0) {
    const labels = symptoms.map((s) => SYMPTOM_LABELS[s] ?? s).join(", ");
    parts.push(`Semptomlar: ${labels}`);
  }
  if (ctx.respiratory_severity && ctx.respiratory_severity !== "normal") {
    const respLabel: Record<string, string> = { mild: "hafif nefes güçlüğü", severe: "ciddi nefes güçlüğü" };
    parts.push(respLabel[ctx.respiratory_severity] ?? ctx.respiratory_severity);
  }
  if (ctx.oxygenation_context && ctx.oxygenation_context !== "normal") {
    const oxyLabel: Record<string, string> = { mild_drop: "azalmış kapasite", severe_drop: "ağır oksijen azalması" };
    parts.push(oxyLabel[ctx.oxygenation_context] ?? ctx.oxygenation_context);
  }
  if (ctx.fever_severity && ctx.fever_severity !== "none") {
    const feverLabel: Record<string, string> = { mild: "hafif ateş", moderate: "orta ateş", high: "yüksek ateş" };
    parts.push(feverLabel[ctx.fever_severity] ?? ctx.fever_severity);
  }
  if (ctx.recent_worsening && ctx.recent_worsening !== "none") {
    const worsenLabel: Record<string, string> = { some: "kötüleşme", rapid_48h: "hızlı kötüleşme (48s)" };
    parts.push(worsenLabel[ctx.recent_worsening] ?? ctx.recent_worsening);
  }
  if (ctx.symptom_duration_tier) {
    const durLabel: Record<string, string> = { "1_2_days": "1–2 gün", "3_7_days": "3–7 gün", over_1_week: ">1 hafta" };
    parts.push(`Süre: ${durLabel[ctx.symptom_duration_tier] ?? ctx.symptom_duration_tier}`);
  } else if (duration != null) {
    parts.push(`Süre: ${duration} gün`);
  }
  if (exposure) parts.push(`Maruziyet: ${EXPOSURE_LABELS[exposure] ?? exposure}`);
  if (ctx.rodent_exposure_level && ctx.rodent_exposure_level !== "none") {
    const rodLabel: Record<string, string> = { unsure: "kemirgen teması belirsiz", rural_env: "kırsal ortam", possible_contact: "kemirgen teması olası" };
    parts.push(rodLabel[ctx.rodent_exposure_level] ?? ctx.rodent_exposure_level);
  }
  if (ageGroup) {
    const ageLabel: Record<string, string> = { adolescent: "genç (<18)", young_adult: "genç yetişkin", adult: "yetişkin", older_adult: "orta yaş üstü", elderly: "yaşlı (65+)" };
    parts.push(ageLabel[ageGroup] ?? ageGroup);
  }

  const session_id = `clin_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;

  return {
    session_id,
    timestamp: new Date().toISOString(),
    source: "clinical_only",
    risk: {
      risk_tier: tier,
      final_score: score,
      imaging_score: score,
      clinical_modifier: 0,
      near_boundary: nearBoundary,
      boundary_proximity: Math.round(boundaryProximity * 10000) / 10000,
      requires_immediate_action: tier === "CRITICAL_PULMONARY_RISK",
      differential_classes: buildDifferentials(resolvedCtx, tier, symptoms),
      tier_thresholds: TIER_THRESHOLDS,
    },
    requires_immediate_action: tier === "CRITICAL_PULMONARY_RISK",
    summary: TIER_SUMMARIES[tier],
    reasoning_bullets: bullets,
    clinical_summary: parts.join(" · "),
    symptoms_provided: symptoms,
    exposure_provided: exposure,
    severity_provided: severity,
    duration_days: duration,
  };
}

function _mapAgeToGroup(age: number): "adolescent" | "young_adult" | "adult" | "older_adult" | "elderly" {
  if (age < 18)  return "adolescent";
  if (age < 30)  return "young_adult";
  if (age < 50)  return "adult";
  if (age < 65)  return "older_adult";
  return "elderly";
}
