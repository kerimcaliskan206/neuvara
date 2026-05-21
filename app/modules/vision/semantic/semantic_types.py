"""
Semantic analysis data types.

All public types are plain dataclasses — no torch, no open_clip imports.
This keeps the type layer importable in any context without pulling in GPU deps.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SemanticCategory:
    """
    A named semantic concept with one or more natural-language prompts.

    Parameters
    ----------
    name : str
        Short machine-readable label (e.g. ``"wildlife"``).
    prompts : tuple[str, ...]
        Natural-language descriptions used to build CLIP text embeddings.
        Multiple prompts are averaged into one category embedding at load time.
    is_medical : bool
        True when the category represents medically relevant content.
        Used to compute ``medical_relevance_score``.
    ood_weight : float
        Weight in [0.0, 1.0] used to compute the OOD score contribution.
        0.0 = fully in-distribution for this pipeline (e.g., medical_xray).
        1.0 = maximally out-of-distribution (e.g., furniture, vehicle).
        Ignored for medical categories (``is_medical=True``).
    """

    name: str
    prompts: tuple[str, ...]
    is_medical: bool = False
    ood_weight: float = 1.0


@dataclass(frozen=True)
class SemanticMatch:
    """Single-category result in a ranked match list."""

    label: str
    score: float    # softmax probability over all categories
    rank: int       # 1-indexed rank (1 = best match)


@dataclass
class SemanticResult:
    """
    Structured output of one semantic analysis call.

    Fields
    ------
    top_semantic_label : str
        Name of the highest-probability category.
    medical_relevance_score : float
        Sum of softmax probabilities for all ``is_medical=True`` categories.
        Range [0, 1]. High = image looks medically relevant to the pipeline.
    ood_score : float
        Weighted sum of softmax probabilities for non-medical categories,
        where each weight is the category's ``ood_weight``.
        Range [0, 1]. High = image is far outside the medical domain.
    top_matches : list[SemanticMatch]
        Top-K categories by softmax probability, sorted descending.
    all_scores : dict[str, float]
        Full category-name → softmax-probability mapping (all categories).
    inference_ms : float
        Image encoding + scoring time in milliseconds.
    model_name : str
        CLIP model architecture (e.g., ``"ViT-B-32"``).
    model_pretrained : str
        CLIP pretrained weights tag (e.g., ``"openai"``).
    """

    top_semantic_label: str
    medical_relevance_score: float
    ood_score: float
    top_matches: list[SemanticMatch]
    all_scores: dict[str, float]
    inference_ms: float
    model_name: str
    model_pretrained: str

    def as_dict(self) -> dict:
        return {
            "top_semantic_label": self.top_semantic_label,
            "medical_relevance_score": round(self.medical_relevance_score, 4),
            "ood_score": round(self.ood_score, 4),
            "top_matches": [
                {
                    "label": m.label,
                    "score": round(m.score, 4),
                    "rank": m.rank,
                }
                for m in self.top_matches
            ],
            "all_scores": {k: round(v, 4) for k, v in self.all_scores.items()},
            "inference_ms": round(self.inference_ms, 2),
            "model": f"{self.model_name}/{self.model_pretrained}",
        }
