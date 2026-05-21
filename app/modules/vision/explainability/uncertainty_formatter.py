"""
Uncertainty reason formatter — Phase 6.

Converts numeric uncertainty signals into human-readable Turkish sentences
that explain *why* a prediction carries uncertainty.

Design
------
Returns None when uncertainty is negligible (score < NOTABLE_THRESHOLD),
so callers can omit the field from the response cleanly.
"""
from __future__ import annotations

_NOTABLE_THRESHOLD: float = 0.35   # below this → no uncertainty reason needed


def format_uncertainty_reason(
    *,
    uncertainty_score: float,
    semantic_alignment: str,
    semantic_uncertainty: float | None,
    ood_score: float,
    fake_medical_score: float | None,
    reasoning_decision: str | None,
) -> str | None:
    """
    Build a Turkish uncertainty reason string, or return None if negligible.

    Parameters
    ----------
    uncertainty_score     : combined [0,1] uncertainty from fusion layer
    semantic_alignment    : "aligned" | "misaligned" | "uncertain"
    semantic_uncertainty  : normalised Shannon entropy [0,1] from reasoning
    ood_score             : weighted OOD score from semantic gate
    fake_medical_score    : suspicion of fake/generated content [0,1]
    reasoning_decision    : "allow" | "reject" | "uncertain"
    """
    if uncertainty_score < _NOTABLE_THRESHOLD:
        return None

    sem_unc = semantic_uncertainty or 0.0
    fake    = fake_medical_score or 0.0
    parts: list[str] = []

    if semantic_alignment == "misaligned":
        parts.append("semantik ve sınıflandırıcı sinyalleri çelişiyor")

    if sem_unc > 0.65:
        parts.append(f"CLIP dağılımı düzdüzüne yayılmış (semantik belirsizlik={sem_unc:.2f})")

    if ood_score > 1.0:
        parts.append(f"görüntü dağılım dışında algılandı (OOD={ood_score:.2f})")

    if fake > 0.25:
        parts.append(f"sahte/yapay içerik şüphesi (fake_score={fake:.2f})")

    if reasoning_decision == "uncertain":
        parts.append("anlamsal akıl yürütme kararsız kaldı")

    if not parts:
        return (
            f"Genel belirsizlik skoru dikkate değer düzeyde "
            f"(uncertainty={uncertainty_score:.2f})."
        )

    return "Belirsizlik kaynakları: " + ", ".join(parts) + "."


def format_semantic_warning(
    *,
    semantic_alignment: str,
    reasoning_decision: str | None,
    fake_medical_score: float | None,
) -> str | None:
    """
    Return a Turkish semantic-conflict warning, or None when no conflict.

    Only surfaces cases where the semantic layer provides actionable caution
    that the caller should surface in the UI.
    """
    fake = fake_medical_score or 0.0

    if semantic_alignment == "misaligned":
        return (
            "Semantik analiz görüntünün tıbbi bağlamla uyuşmadığını gösteriyor; "
            "tahmin dikkatli yorumlanmalıdır."
        )

    if fake > 0.35:
        return (
            f"Görüntü tıbbi görüntü özellikleri taşımıyor ya da yapay içerik "
            f"olabilir (fake_score={fake:.2f})."
        )

    if reasoning_decision == "reject":
        return (
            "Semantik akıl yürütme görüntüyü reddetme eğiliminde; "
            "güven eşiği yeterince yüksek olmadığından geçişe izin verildi."
        )

    return None
