"""
Semantic reasoning engine — Phase 3 of the HantaProject vision pipeline.

Adds interpretive intelligence on top of the CLIP semantic gate by combining
multiple evidence signals (entropy, group coherence, confidence-weighted label
scoring) into a structured reasoning decision with explanation.

Public surface
--------------
    from app.modules.vision.reasoning import SemanticReasoner, ReasoningOutput

    reasoner = SemanticReasoner()        # stateless, no model loading
    output   = reasoner.reason(sem_result)
    print(output.reasoning_type, output.reasoning_confidence)
    print(output.explanation)

For direct access to the module-level singleton (used by SemanticGate):

    from app.modules.vision.reasoning.semantic_reasoner import default_reasoner
"""
from app.modules.vision.reasoning.semantic_reasoner import SemanticReasoner, default_reasoner
from app.modules.vision.reasoning.reasoning_types import (
    ReasoningEvidence,
    ReasoningOutput,
    RuleScore,
    LabelGroupScores,
)

__all__ = [
    "SemanticReasoner",
    "default_reasoner",
    "ReasoningOutput",
    "ReasoningEvidence",
    "RuleScore",
    "LabelGroupScores",
]
