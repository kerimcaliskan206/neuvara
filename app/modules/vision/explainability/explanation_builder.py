"""
Explanation builder — Phase 6.

Generates a single human-readable Turkish explanation_summary that captures
the most important signal combination for a given prediction.

Design contract
---------------
  - No access to raw pixel data; inputs are all post-inference metadata.
  - One terse sentence (≤ two clauses) per output.
  - Reasoning-type descriptions are bilingual keys; Turkish phrases live here.
  - No disease diagnosis language.
"""
from __future__ import annotations

# Reasoning type → short Turkish description used inside explanation sentences
_REASONING_TYPE_TR: dict[str, str] = {
    "radiology_candidate":  "radyoloji görüntüsü",
    "microscopy_candidate": "mikroskobi görüntüsü",
    "likely_medical":       "tıbbi görüntü",
    "ambiguous_medical":    "belirsiz tıbbi görüntü",
    "uncertain_semantic":   "semantiği belirsiz görüntü",
    "wildlife_scene":       "yaban hayatı sahnesi",
    "portrait_scene":       "insan portresi",
    "consumer_object":      "tüketici objesi",
    "natural_scene":        "doğal sahne",
    "clear_non_medical":    "tıbbi olmayan görüntü",
}


def _reasoning_type_tr(reasoning_type: str | None) -> str:
    return _REASONING_TYPE_TR.get(reasoning_type or "", "tıbbi bağlamlı görüntü")


def build_explanation_summary(
    *,
    trust_tier: str,
    semantic_alignment: str,
    reasoning_type: str | None,
    classifier_confidence: float,
    fusion_delta: float,
    medical_plausibility: float | None,
    fake_medical_score: float | None,
) -> str:
    """
    Build a single Turkish explanation sentence.

    Priority order (highest wins):
      1. suspicious   — fake content or strong misalignment
      2. misaligned   — semantic/classifier conflict
      3. very_high_trust  — strong aligned support
      4. high_trust       — consistent support
      5. moderate_trust   — partial support
      6. uncertain        — fallback
    """
    fake = fake_medical_score or 0.0
    plausibility = medical_plausibility or 0.0

    if trust_tier == "suspicious":
        return (
            f"Görüntü sahte veya yapay tıbbi içerik özellikleri taşıyor "
            f"(fake_score={fake:.2f}); sınıflandırma güvenilirliği düşük."
        )

    if semantic_alignment == "misaligned":
        return (
            "Sınıflandırıcı güveni yüksek ancak semantik katmanda çelişki gözlendi; "
            "görüntü tıbbi bağlamla tam uyuşmuyor."
        )

    rtype_tr = _reasoning_type_tr(reasoning_type)

    if trust_tier == "very_high_trust":
        return (
            f"Görüntü semantik olarak medikal bağlamla güçlü biçimde uyumlu bulundu "
            f"({rtype_tr}); tahmin yüksek güvenle destekleniyor."
        )

    if trust_tier == "high_trust":
        if fusion_delta >= 0.03:
            return (
                f"Semantik analiz ve sınıflandırıcı uyumlu sonuçlar üretiyor ({rtype_tr}); "
                "tahmin güvenilir olarak değerlendiriliyor."
            )
        return (
            "Semantik ve sınıflandırıcı sinyalleri tutarlı; "
            "tahmin güvenilir olarak destekleniyor."
        )

    if trust_tier == "moderate_trust":
        if plausibility > 0.55:
            return (
                "Tahmin orta düzeyde güvenle destekleniyor; "
                "tıbbi plausibility yeterli ancak semantik sinyaller tam uyumlu değil."
            )
        return (
            "Tahmin orta düzeyde güvenle destekleniyor; "
            "semantik ve sınıflandırıcı sinyaller kısmen uyumlu."
        )

    # uncertain
    if classifier_confidence < 0.55:
        return (
            "Karar düşük güven nedeniyle belirsiz olarak işaretlendi; "
            "ek klinik değerlendirme önerilir."
        )
    return (
        "Karar belirsiz semantik sinyaller nedeniyle düşük güvenle destekleniyor; "
        "dikkatli yorumlanmalıdır."
    )
