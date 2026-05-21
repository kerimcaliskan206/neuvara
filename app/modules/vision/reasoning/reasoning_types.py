"""
Data types for the semantic reasoning engine.

All types are plain Python dataclasses — no ML imports, no torch.
This layer is fully importable without GPU dependencies.

Type hierarchy
--------------
  SemanticDecision            ← allow | reject | uncertain
  LabelGroupScores            ← per-group softmax mass
  ReasoningEvidence           ← structured evidence from SemanticResult
  RuleScore                   ← score from a single reasoning rule
  ReasoningOutput             ← final decision + explanation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ── Core decision type ────────────────────────────────────────────────────────

SemanticDecision = Literal["allow", "reject", "uncertain"]

# ── Reasoning type vocabulary ─────────────────────────────────────────────────

# All valid reasoning type names (stable API — used as foreign keys in API output)
REASONING_TYPES: frozenset[str] = frozenset({
    # Medical — decision: allow
    "radiology_candidate",       # dominant medical_xray signal, high consistency
    "microscopy_candidate",      # dominant medical_microscopy signal, high consistency
    "likely_medical",            # medical group dominant, no specific candidate
    # Borderline — decision: uncertain
    "ambiguous_medical",         # medical and non-medical signals roughly equal
    "uncertain_semantic",        # flat probability distribution, cannot decide
    # Non-medical — decision: reject
    "wildlife_scene",            # wildlife label dominant
    "portrait_scene",            # human portrait dominant
    "consumer_object",           # food / vehicle / furniture / random_object
    "natural_scene",             # indoor or outdoor scene
    "clear_non_medical",         # catch-all: non-medical, high consistency
})

# Authoritative decision each reasoning type maps to
REASONING_TYPE_DECISIONS: dict[str, SemanticDecision] = {
    "radiology_candidate":  "allow",
    "microscopy_candidate": "allow",
    "likely_medical":       "allow",
    "ambiguous_medical":    "uncertain",
    "uncertain_semantic":   "uncertain",
    "wildlife_scene":       "reject",
    "portrait_scene":       "reject",
    "consumer_object":      "reject",
    "natural_scene":        "reject",
    "clear_non_medical":    "reject",
}


# ── Label group scores ────────────────────────────────────────────────────────


@dataclass
class LabelGroupScores:
    """
    Aggregated softmax probability for each semantic group.

    Each field is the sum of per-label probabilities for that group.
    All values are non-negative; their sum is approximately 1.0 (may exceed
    1.0 due to rounding in the aggregation).

    Groups mirror LABEL_GROUPS in reasoning_rules.py — keep in sync when
    new CLIP categories are added.
    """

    medical: float = 0.0    # medical_xray + medical_microscopy
    wildlife: float = 0.0   # wildlife
    rodent: float = 0.0     # rodent  (adjacent to domain)
    human: float = 0.0      # human
    consumer: float = 0.0   # food + furniture + vehicle + random_object
    scene: float = 0.0      # indoor_scene + outdoor_scene

    def as_dict(self) -> dict[str, float]:
        return {
            "medical":   round(self.medical,   4),
            "wildlife":  round(self.wildlife,  4),
            "rodent":    round(self.rodent,    4),
            "human":     round(self.human,     4),
            "consumer":  round(self.consumer,  4),
            "scene":     round(self.scene,     4),
        }

    @property
    def non_medical(self) -> float:
        """Total probability mass outside the medical group."""
        return max(0.0, 1.0 - self.medical)

    @property
    def dominant_group(self) -> str:
        """Name of the group with the highest probability mass."""
        as_items = [
            ("medical",   self.medical),
            ("wildlife",  self.wildlife),
            ("rodent",    self.rodent),
            ("human",     self.human),
            ("consumer",  self.consumer),
            ("scene",     self.scene),
        ]
        return max(as_items, key=lambda x: x[1])[0]


# ── Reasoning evidence ────────────────────────────────────────────────────────


@dataclass
class ReasoningEvidence:
    """
    Structured evidence computed from a raw SemanticResult.

    All downstream rule scorers work exclusively on this object —
    no SemanticResult fields are read in rule logic.  This isolation
    makes rule scoring fully unit-testable without running CLIP.
    """

    top_label: str
    top_label_score: float           # softmax probability of the top label
    top3_labels: list[str]
    top3_scores: list[float]
    medical_relevance: float         # pre-computed sum from SemanticResult
    ood_score: float                 # pre-computed weighted sum from SemanticResult
    semantic_uncertainty: float      # entropy-based  [0 = certain, 1 = flat]
    semantic_consistency: float      # group coherence [0 = spread, 1 = focused]
    group_scores: LabelGroupScores

    def as_dict(self) -> dict:
        return {
            "top_label": self.top_label,
            "top_label_score": round(self.top_label_score, 4),
            "top3": [
                {"label": lbl, "score": round(sc, 4)}
                for lbl, sc in zip(self.top3_labels, self.top3_scores)
            ],
            "medical_relevance": round(self.medical_relevance, 4),
            "ood_score": round(self.ood_score, 4),
            "semantic_uncertainty": round(self.semantic_uncertainty, 4),
            "semantic_consistency": round(self.semantic_consistency, 4),
            "group_scores": self.group_scores.as_dict(),
        }


# ── Rule score ────────────────────────────────────────────────────────────────


@dataclass
class RuleScore:
    """Score from evaluating a single reasoning rule."""

    reasoning_type: str
    confidence: float           # [0, 1] — how strongly this rule fired
    decision: SemanticDecision

    def as_dict(self) -> dict:
        return {
            "reasoning_type": self.reasoning_type,
            "confidence": round(self.confidence, 4),
            "decision": self.decision,
        }


# ── Reasoning output ──────────────────────────────────────────────────────────


@dataclass
class ReasoningOutput:
    """
    Final output of the semantic reasoning engine for one image.

    Attached to SemanticGateResult as ``reasoning`` and propagated to
    the API response via SemanticInfo fields.

    Fields
    ------
    semantic_decision      : "allow" / "reject" / "uncertain"
    reasoning_type         : dominant semantic scene interpretation
    reasoning_confidence   : effective confidence after margin penalty [0, 1]
    semantic_uncertainty   : entropy of the CLIP distribution [0 = certain]
    semantic_consistency   : group coherence of top-K matches [0 = spread]
    explanation            : human-readable Turkish explanation
    evidence               : all derived evidence metrics
    all_rule_scores        : confidence from every rule (for debug / audit)
    debug_info             : internal trace of the decision path
    """

    semantic_decision: SemanticDecision
    reasoning_type: str
    reasoning_confidence: float
    semantic_uncertainty: float
    semantic_consistency: float
    explanation: str
    evidence: ReasoningEvidence
    all_rule_scores: list[RuleScore] = field(default_factory=list)
    debug_info: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "semantic_decision": self.semantic_decision,
            "reasoning_type": self.reasoning_type,
            "reasoning_confidence": round(self.reasoning_confidence, 4),
            "semantic_uncertainty": round(self.semantic_uncertainty, 4),
            "semantic_consistency": round(self.semantic_consistency, 4),
            "explanation": self.explanation,
            "evidence": self.evidence.as_dict(),
            "top_rules": [s.as_dict() for s in self.all_rule_scores[:5]],
        }
