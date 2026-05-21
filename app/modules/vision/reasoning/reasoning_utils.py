"""
Utility functions for the semantic reasoning engine.

All functions are pure (no side effects, no state).  They take raw
SemanticResult or evidence components and return derived metrics used
by the rule scorers and the SemanticReasoner.
"""
from __future__ import annotations

import math
from collections import defaultdict

from app.modules.vision.reasoning.reasoning_rules import LABEL_TO_GROUP
from app.modules.vision.reasoning.reasoning_types import (
    LabelGroupScores,
    ReasoningEvidence,
)
from app.modules.vision.semantic.semantic_types import SemanticMatch, SemanticResult


# ── Uncertainty ───────────────────────────────────────────────────────────────


def compute_semantic_uncertainty(all_scores: dict[str, float]) -> float:
    """
    Shannon entropy of the full CLIP label distribution, normalised to [0, 1].

    Returns
    -------
    0.0  — perfectly peaked distribution (maximum certainty)
    1.0  — uniform distribution (maximum uncertainty, e.g. CLIP cannot decide)

    Notes
    -----
    ViT-B-32/openai on medical images:     entropy ≈ 0.88–0.92 (11 categories)
    ViT-B-32/openai on clear non-medical:  entropy ≈ 0.84–0.94 (dominant label)
    The distribution is inherently soft — perfect certainty is never observed.
    """
    probs = [p for p in all_scores.values() if p > 1e-9]
    n = len(probs)
    if n < 2:
        return 0.0
    entropy = -sum(p * math.log(p) for p in probs)
    max_entropy = math.log(n)
    return min(1.0, entropy / max_entropy)


# ── Consistency ───────────────────────────────────────────────────────────────


def compute_semantic_consistency(
    top_matches: list[SemanticMatch],
    top_k: int = 5,
) -> float:
    """
    Coherence of the top-K CLIP matches within a single semantic group.

    High consistency:  all top-K labels from the same group  → near 1.0
    Low consistency:   labels spread evenly across groups      → near 1/n_groups

    Algorithm
    ---------
    1. Sum softmax probability for each semantic group in the top-K window.
    2. Return (max group mass) / (total top-K mass).

    Parameters
    ----------
    top_matches : SemanticMatch objects from SemanticResult (any length)
    top_k       : how many leading matches to consider

    Returns
    -------
    float in [0, 1]

    Examples
    --------
    chest X-ray top-5: medical_xray(0.14) medical_microscopy(0.11) rodent(0.09) …
    → medical_group = 0.25, total = ~0.52, consistency ≈ 0.48

    wolf top-5: wildlife(0.19) human(0.14) outdoor_scene(0.09) …
    → wildlife_group = 0.19, total = ~0.51, consistency ≈ 0.37

    food image top-5: food(0.32) furniture(0.14) random_object(0.10) vehicle(0.09) …
    → consumer_group = 0.65, total = ~0.65, consistency ≈ 1.00
    """
    matches = top_matches[:top_k]
    if not matches:
        return 0.0

    group_mass: dict[str, float] = defaultdict(float)
    total = 0.0
    for m in matches:
        group = LABEL_TO_GROUP.get(m.label, "other")
        group_mass[group] += m.score
        total += m.score

    if total < 1e-9:
        return 0.0

    return max(group_mass.values()) / total


# ── Group aggregation ─────────────────────────────────────────────────────────


def compute_label_group_scores(all_scores: dict[str, float]) -> LabelGroupScores:
    """
    Aggregate per-label softmax probabilities into per-group totals.

    Labels not found in LABEL_TO_GROUP are silently ignored (future-proof
    for new CLIP categories that have not yet been assigned to a group).
    """
    gs = LabelGroupScores()
    for label, prob in all_scores.items():
        group = LABEL_TO_GROUP.get(label)
        if group == "medical":
            gs.medical += prob
        elif group == "wildlife":
            gs.wildlife += prob
        elif group == "rodent":
            gs.rodent += prob
        elif group == "human":
            gs.human += prob
        elif group == "consumer":
            gs.consumer += prob
        elif group == "scene":
            gs.scene += prob
    return gs


# ── Evidence builder ──────────────────────────────────────────────────────────


def build_evidence(semantic_result: SemanticResult) -> ReasoningEvidence:
    """
    Derive a fully structured ReasoningEvidence from a raw SemanticResult.

    This is the single authoritative place where SemanticResult fields are
    consumed.  All rule scorers and the reasoner work exclusively on the
    returned ReasoningEvidence object, making them fully testable without
    a live CLIP model.

    Parameters
    ----------
    semantic_result : output of ClipSemanticAnalyzer.analyze()

    Returns
    -------
    ReasoningEvidence with all metrics computed and cached.
    """
    all_scores = semantic_result.all_scores
    top_matches = semantic_result.top_matches

    uncertainty = compute_semantic_uncertainty(all_scores)
    consistency = compute_semantic_consistency(top_matches, top_k=5)
    group_scores = compute_label_group_scores(all_scores)

    top3 = top_matches[:3]
    top3_labels = [m.label for m in top3]
    top3_scores = [m.score for m in top3]

    top_label = semantic_result.top_semantic_label
    top_label_score = all_scores.get(top_label, 0.0)

    return ReasoningEvidence(
        top_label=top_label,
        top_label_score=top_label_score,
        top3_labels=top3_labels,
        top3_scores=top3_scores,
        medical_relevance=semantic_result.medical_relevance_score,
        ood_score=semantic_result.ood_score,
        semantic_uncertainty=uncertainty,
        semantic_consistency=consistency,
        group_scores=group_scores,
    )
