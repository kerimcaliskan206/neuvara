"""
CLIP-based semantic analysis — phase-1 addition to the HantaProject
vision pipeline.

Public surface
--------------
    from app.modules.vision.semantic import get_semantic_analyzer, SemanticResult

    analyzer = get_semantic_analyzer()   # loads CLIP once (singleton)
    result   = analyzer.analyze(pil_image)
"""
from app.modules.vision.semantic.semantic_analyzer import (
    ClipSemanticAnalyzer,
    get_semantic_analyzer,
)
from app.modules.vision.semantic.semantic_types import (
    SemanticCategory,
    SemanticMatch,
    SemanticResult,
)

__all__ = [
    "ClipSemanticAnalyzer",
    "get_semantic_analyzer",
    "SemanticCategory",
    "SemanticMatch",
    "SemanticResult",
]
