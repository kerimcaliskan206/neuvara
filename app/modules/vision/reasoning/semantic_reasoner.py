"""
Semantic reasoning engine for the HantaProject vision pipeline.

Architecture position
---------------------
    SemanticResult (from CLIP)
          │
          ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  SemanticReasoner                                            │
    │                                                              │
    │  1. build_evidence()                                         │
    │     • compute_semantic_uncertainty()  — entropy of dist.     │
    │     • compute_semantic_consistency()  — group coherence      │
    │     • compute_label_group_scores()    — aggregated probs     │
    │                                                              │
    │  2. _evaluate_rules()                                        │
    │     • 10 scorer functions, each → float [0, 1]               │
    │     • sorted by confidence descending                        │
    │                                                              │
    │  3. _select_winner()                                         │
    │     • top scorer wins                                        │
    │     • margin penalty: large gap → high confidence            │
    │       small gap (competing hypotheses) → reduced confidence  │
    │                                                              │
    │  4. _derive_decision()                                       │
    │     • confidence < DECISIVE_THRESHOLD → "uncertain"          │
    │     • else → winner rule's mapped decision                   │
    │                                                              │
    │  5. _select_explanation()                                    │
    │     • three tiers: high / medium / low confidence            │
    │     • language: Turkish (matches frontend)                   │
    └──────────────────┬───────────────────────────────────────────┘
                       │ ReasoningOutput
                       ▼
            semantic_gate.evaluate()
               │              │
           rejected         passed
               │              │
            reject         override?  (if reasoning rejects with conf ≥ 0.80)
                               │
                           EfficientNet classifier

Integration
-----------
SemanticGate.evaluate() calls SemanticReasoner.reason() internally.
The reasoning output is attached to SemanticGateResult and serialised
into the API response as SemanticInfo reasoning fields.

The reasoner does NOT replace the gate — it enriches its output with
interpretive context AND can override a gate PASS to REJECT when it is
very confident (reasoning_confidence ≥ OVERRIDE_CONFIDENCE_THRESHOLD)
about a non-medical scene type.

Robustness
----------
• Evidence-weighted rules (not binary if/else) → graceful degradation
  under borderline or adversarial inputs.
• Margin penalty on winner selection → competing hypotheses lower the
  effective confidence, preventing over-confident wrong decisions.
• Uncertainty fallback rule → flat distributions are correctly labelled
  "uncertain" and passed to the medical classifier rather than rejected.
• All rule scorer exceptions are caught and logged — a broken rule
  never crashes an inference request.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.modules.vision.reasoning.reasoning_rules import REASONING_RULES, ReasoningRule
from app.modules.vision.reasoning.reasoning_types import (
    ReasoningEvidence,
    ReasoningOutput,
    RuleScore,
    SemanticDecision,
)
from app.modules.vision.reasoning.reasoning_utils import build_evidence
from app.modules.vision.semantic.semantic_types import SemanticResult

logger = logging.getLogger(__name__)

# ── Decision thresholds ───────────────────────────────────────────────────────

# Below this winner confidence → override to "uncertain" decision.
_DECISIVE_CONFIDENCE_THRESHOLD: float = 0.45

# Minimum margin between #1 and #2 rule to call the confidence "full".
# A tighter race penalises confidence proportionally.
_MARGIN_FULL: float = 0.18

# Reasoning override: only override a gate PASS when the reasoning decision
# is "reject" AND the effective confidence reaches this bar.
# Conservative (0.80) — avoids false positives on genuine medical images.
OVERRIDE_CONFIDENCE_THRESHOLD: float = 0.80

# ── Human-readable explanations ───────────────────────────────────────────────
#
# Three confidence tiers per reasoning type (high / medium / low).
# Language: Turkish — matches the frontend locale for rejection_reason fields.

_EXPLANATIONS: dict[str, dict[str, str]] = {
    "radiology_candidate": {
        "high":   "Görüntü yüksek güvenle göğüs radyografisi ile eşleşiyor.",
        "medium": "Görüntü muhtemelen radyoloji görüntüsü; güven orta düzeyde.",
        "low":    "Görüntü radyoloji görüntüsüne benzeyebilir; belirsizlik yüksek.",
    },
    "microscopy_candidate": {
        "high":   "Görüntü yüksek güvenle tıbbi mikroskopi görüntüsü ile eşleşiyor.",
        "medium": "Görüntü muhtemelen mikroskopi görüntüsü; güven orta düzeyde.",
        "low":    "Görüntü mikroskopi görüntüsüne benzeyebilir; belirsizlik yüksek.",
    },
    "likely_medical": {
        "high":   "Görüntü tıbbi alana uygun semantik yapı sergiliyor.",
        "medium": "Görüntü tıbbi içerik ile kısmi uyum gösteriyor.",
        "low":    "Görüntüde zayıf tıbbi işaretler mevcut; kesin karar verilemiyor.",
    },
    "ambiguous_medical": {
        "high":   "Görüntü hem tıbbi hem tıbbi dışı sinyaller taşıyor; sınıflandırıcı kararı belirleyecek.",
        "medium": "Semantik analiz karışık sonuç verdi; görüntü doğrulanmalı.",
        "low":    "Semantik analiz belirsiz; görüntü hakkında kesin karar alınamıyor.",
    },
    "uncertain_semantic": {
        "high":   "Semantik dağılım düz; görüntü hiçbir kategoriye güçlü şekilde uymamaktadır.",
        "medium": "Görüntünün semantik içeriği belirsiz.",
        "low":    "Görüntünün semantiği tam olarak tanımlanamadı.",
    },
    "wildlife_scene": {
        "high":   "Görüntü yabani hayvan sahnesi olarak güçlü biçimde tanımlandı; tıbbi görüntü değil.",
        "medium": "Görüntü muhtemelen yabani hayvan fotoğrafı; tıbbi alaka düşük.",
        "low":    "Görüntü yabani hayvan içeriyor olabilir; ancak belirsizlik yüksek.",
    },
    "portrait_scene": {
        "high":   "Görüntü bir insan portresi olarak tanımlandı; tıbbi bağlamla ilgisiz.",
        "medium": "Görüntü muhtemelen bir insan fotoğrafı; tıbbi kullanım uygun değil.",
        "low":    "Görüntüde insan portresi işaretleri var; ancak güven düşük.",
    },
    "consumer_object": {
        "high":   "Görüntü tüketici nesnesi (yiyecek, araç veya mobilya) içeriyor; tıbbi değil.",
        "medium": "Görüntü muhtemelen günlük eşya içeriyor.",
        "low":    "Görüntüde tıbbi dışı nesne işaretleri mevcut.",
    },
    "natural_scene": {
        "high":   "Görüntü doğa veya iç mekân sahnesi olarak tanımlandı; tıbbi görüntü değil.",
        "medium": "Görüntü muhtemelen bir mekân fotoğrafı.",
        "low":    "Görüntüde sahne içeriği işaretleri var.",
    },
    "clear_non_medical": {
        "high":   "Görüntü tıbbi bağlamla ilgisiz olarak yüksek güvenle sınıflandırıldı.",
        "medium": "Görüntü büyük olasılıkla tıbbi alan dışındaki bir içerik.",
        "low":    "Görüntü tıbbi dışı görünüyor; ancak kesin karar yapılamadı.",
    },
}

_FALLBACK_EXPLANATION = "Semantik analiz sonucu: {type} ({tier} güven)."


def _confidence_tier(confidence: float) -> str:
    if confidence >= 0.72:
        return "high"
    if confidence >= 0.45:
        return "medium"
    return "low"


def _select_explanation(reasoning_type: str, confidence: float) -> str:
    tier = _confidence_tier(confidence)
    templates = _EXPLANATIONS.get(reasoning_type)
    if templates:
        return templates.get(tier, _FALLBACK_EXPLANATION.format(type=reasoning_type, tier=tier))
    return _FALLBACK_EXPLANATION.format(type=reasoning_type, tier=tier)


# ── Reasoner ──────────────────────────────────────────────────────────────────


class SemanticReasoner:
    """
    Evidence-aggregating semantic reasoner for medical image screening.

    Takes the raw CLIP SemanticResult and produces a ReasoningOutput that:
    - names the dominant semantic scene type (reasoning_type)
    - assigns effective confidence after margin penalty
    - quantifies distribution entropy (semantic_uncertainty)
    - quantifies label coherence (semantic_consistency)
    - gives a structured allow / reject / uncertain decision
    - generates a human-readable Turkish explanation

    The reasoner enriches the SemanticGate — it does NOT replace it.
    The gate's label and score checks remain the primary gate signals.
    The reasoning output is:
      1. Attached as metadata on every response (audit trail)
      2. Able to OVERRIDE a gate PASS to REJECT when reasoning_confidence
         ≥ OVERRIDE_CONFIDENCE_THRESHOLD (conservative: 0.80)

    Thread-safe: stateless after __init__, all state lives in local variables.
    """

    def __init__(
        self,
        rules: tuple[ReasoningRule, ...] = REASONING_RULES,
        decisive_threshold: float = _DECISIVE_CONFIDENCE_THRESHOLD,
        margin_full: float = _MARGIN_FULL,
    ) -> None:
        self._rules = rules
        self._decisive_threshold = decisive_threshold
        self._margin_full = margin_full

    # ── Primary API ───────────────────────────────────────────────────────────

    def reason(self, semantic_result: SemanticResult) -> ReasoningOutput:
        """
        Run all reasoning rules against a SemanticResult and return a
        fully structured ReasoningOutput.

        Parameters
        ----------
        semantic_result : output from ClipSemanticAnalyzer.analyze()

        Returns
        -------
        ReasoningOutput with decision, confidence, uncertainty,
        consistency, explanation, and full rule audit trail.
        """
        evidence = build_evidence(semantic_result)
        rule_scores = self._evaluate_rules(evidence)
        winner, effective_confidence = self._select_winner(rule_scores)
        decision = self._derive_decision(winner, effective_confidence)
        explanation = _select_explanation(winner.reasoning_type, effective_confidence)

        output = ReasoningOutput(
            semantic_decision=decision,
            reasoning_type=winner.reasoning_type,
            reasoning_confidence=round(effective_confidence, 4),
            semantic_uncertainty=round(evidence.semantic_uncertainty, 4),
            semantic_consistency=round(evidence.semantic_consistency, 4),
            explanation=explanation,
            evidence=evidence,
            all_rule_scores=rule_scores,
            debug_info=self._build_debug_info(
                evidence, rule_scores, winner, effective_confidence, decision
            ),
        )

        self._log_reasoning(output)
        return output

    # ── Helper predicates (public for unit testing) ───────────────────────────

    def is_clearly_non_medical(self, evidence: ReasoningEvidence) -> bool:
        """True when non-medical signals clearly dominate with high consistency."""
        return (
            evidence.group_scores.medical < 0.10
            and evidence.group_scores.non_medical > 0.70
            and evidence.semantic_consistency > 0.50
        )

    def is_likely_medical(self, evidence: ReasoningEvidence) -> bool:
        """True when medical signals are dominant."""
        return (
            evidence.group_scores.medical > 0.15
            or evidence.top_label in ("medical_xray", "medical_microscopy")
        )

    def is_semantically_ambiguous(self, evidence: ReasoningEvidence) -> bool:
        """True when the distribution is too flat or inconsistent to classify."""
        return (
            evidence.semantic_uncertainty > 0.85
            or evidence.semantic_consistency < 0.30
        )

    def compute_semantic_consistency(self, evidence: ReasoningEvidence) -> float:
        return evidence.semantic_consistency

    def compute_semantic_uncertainty(self, evidence: ReasoningEvidence) -> float:
        return evidence.semantic_uncertainty

    # ── Internal pipeline ─────────────────────────────────────────────────────

    def _evaluate_rules(self, evidence: ReasoningEvidence) -> list[RuleScore]:
        """
        Apply all registered rules.  Returns scores sorted descending.
        Broken scorers are skipped with a warning — never fatal.
        """
        scores: list[RuleScore] = []
        for rule in self._rules:
            try:
                raw = float(rule.scorer(evidence))
                confidence = max(0.0, min(1.0, raw))
                scores.append(RuleScore(
                    reasoning_type=rule.name,
                    confidence=confidence,
                    decision=rule.decision,
                ))
            except Exception:
                logger.warning(
                    "SemanticReasoner: rule '%s' scorer raised — skipping",
                    rule.name, exc_info=True,
                )
        scores.sort(key=lambda s: s.confidence, reverse=True)
        return scores

    def _select_winner(
        self,
        rule_scores: list[RuleScore],
    ) -> tuple[RuleScore, float]:
        """
        Select the highest-scoring rule and compute effective confidence.

        Effective confidence is penalised when the margin over the runner-up
        is small — competing hypotheses signal a weaker overall decision.

        Margin penalty formula:
          penalty_fraction = max(0, margin_full - margin) / margin_full
          effective = raw_confidence * (1 - penalty_fraction * 0.35)

        With margin_full = 0.18:
          margin=0.20 (clear winner)  → penalty=0.00 → effective = raw
          margin=0.09 (close race)    → penalty=0.50 → effective ≈ raw × 0.825
          margin=0.00 (tied)          → penalty=1.00 → effective ≈ raw × 0.65
        """
        if not rule_scores:
            return RuleScore(
                reasoning_type="uncertain_semantic",
                confidence=0.0,
                decision="uncertain",
            ), 0.0

        top = rule_scores[0]
        second = rule_scores[1].confidence if len(rule_scores) > 1 else 0.0
        margin = top.confidence - second

        penalty_fraction = max(0.0, self._margin_full - margin) / self._margin_full
        effective = top.confidence * (1.0 - penalty_fraction * 0.35)

        return top, round(effective, 4)

    def _derive_decision(
        self,
        winner: RuleScore,
        effective_confidence: float,
    ) -> SemanticDecision:
        """
        Map winner + effective confidence → allow / reject / uncertain.

        Low confidence always produces "uncertain" regardless of the rule's
        native decision — prevents a barely-firing wildlife rule from
        causing a hard rejection.
        """
        if effective_confidence < self._decisive_threshold:
            return "uncertain"
        return winner.decision

    def _build_debug_info(
        self,
        evidence: ReasoningEvidence,
        rule_scores: list[RuleScore],
        winner: RuleScore,
        effective_confidence: float,
        decision: SemanticDecision,
    ) -> dict:
        return {
            "winner_rule": winner.reasoning_type,
            "winner_raw_confidence": round(winner.confidence, 4),
            "effective_confidence": round(effective_confidence, 4),
            "final_decision": decision,
            "decisive_threshold": self._decisive_threshold,
            "override_threshold": OVERRIDE_CONFIDENCE_THRESHOLD,
            "would_override": (
                decision == "reject"
                and effective_confidence >= OVERRIDE_CONFIDENCE_THRESHOLD
            ),
            "top5_rules": [s.as_dict() for s in rule_scores[:5]],
            "group_scores": evidence.group_scores.as_dict(),
            "semantic_uncertainty": round(evidence.semantic_uncertainty, 4),
            "semantic_consistency": round(evidence.semantic_consistency, 4),
        }

    def _log_reasoning(self, output: ReasoningOutput) -> None:
        logger.debug(
            "SemanticReasoner: decision=%s type=%s conf=%.3f "
            "uncertainty=%.3f consistency=%.3f top_label=%s",
            output.semantic_decision,
            output.reasoning_type,
            output.reasoning_confidence,
            output.semantic_uncertainty,
            output.semantic_consistency,
            output.evidence.top_label,
        )
        logger.debug(
            "SemanticReasoner: rule_scores=%s",
            [(s.reasoning_type, round(s.confidence, 3)) for s in output.all_rule_scores[:5]],
        )
        logger.debug(
            "SemanticReasoner: group_scores=%s explanation=%s",
            output.evidence.group_scores.as_dict(),
            output.explanation,
        )


# ── Module-level singleton ────────────────────────────────────────────────────
#
# Stateless after construction — safe to share across threads.
# No model loading or I/O at construction time (unlike CLIP).

default_reasoner = SemanticReasoner()
