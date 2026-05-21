"""
Medical semantic refinement layer — Phase 4.

Evaluates whether an image that passed the semantic gate genuinely resembles
real medical imagery, and detects suspicious fake/generated content.

Role in the pipeline
--------------------
    image
      ↓
    Semantic Gate  (CLIP — label/score/reasoning)     ← phase 2–3
      ↓ (passed)
    MedicalRefiner  (CLIP — medical sub-prompts)      ← THIS MODULE
      ↓ advisory scores
    EfficientNet classifier                           ← unchanged

Design contract
---------------
ADVISORY ONLY.  The refiner produces metadata that enriches the response;
it does NOT hard-reject images.  Downstream components (EfficientNet,
fusion engine) can use medical_plausibility and fake_medical_score to
adjust their confidence estimates.

Scores produced
---------------
  medical_plausibility  [0, 1] — how closely the image resembles real
                                  medical imagery (xray, microscopy).
  fake_medical_score    [0, 1] — how likely the image is fake/generated/
                                  generic grayscale, not genuine imaging.
  semantic_margin       [0, 1] — gap between top-1 and top-2 sub-group.
                                  High = clear category identification.
                                  Low  = ambiguous / mixed signals.
  semantic_medical_type str    — dominant sub-group name.
  refinement_reason     str    — Turkish advisory note.

Sub-group taxonomy
------------------
  Real medical (contribute to medical_plausibility):
    healthy_xray        — normal chest radiograph, clear lungs
    pneumonia_xray      — infiltrates, opacities, consolidation
    lung_opacity        — focal/diffuse opacity on radiograph
    radiology_scan      — generic radiological/CT/MRI scan
    medical_microscopy  — histopathology, blood smear, pathology slide

  Suspicious (contribute to fake_medical_score):
    fake_medical_texture  — artificial/generated scan patterns
    ai_generated_medical  — AI-synthesised radiograph or microscopy
    generic_grayscale     — blank or featureless grayscale image
    non_medical_grayscale — everyday black-and-white photograph

Implementation notes
--------------------
Text embeddings are pre-computed once at the first refine() call using the
CLIP singleton (get_semantic_analyzer()).  Image encoding also reuses the
singleton — no second CLIP model is loaded.

Thread safety: the lazy _load() is protected by a threading.Lock, same
pattern as get_semantic_analyzer().
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

# ── Sub-group prompt table ────────────────────────────────────────────────────

_SUBGROUPS: dict[str, list[str]] = {
    # ── Real medical ──────────────────────────────────────────────────────────
    "healthy_xray": [
        "a normal chest X-ray with clear lungs",
        "a healthy thoracic radiograph without abnormality",
        "a clear lung field chest radiograph",
    ],
    "pneumonia_xray": [
        "a chest X-ray showing pneumonia with lung opacity",
        "a radiograph with pulmonary infiltrate or consolidation",
        "chest X-ray evidence of pulmonary infection",
        "bilateral infiltrates on chest radiograph",
    ],
    "lung_opacity": [
        "focal lung opacity on chest X-ray",
        "diffuse pulmonary opacification on radiograph",
        "pleural effusion visible on chest X-ray",
    ],
    "radiology_scan": [
        "a clinical radiological scan",
        "a diagnostic medical imaging study",
        "a CT or MRI cross-sectional medical scan",
        "a thoracic radiology image",
    ],
    "medical_microscopy": [
        "a histopathology slide stained with hematoxylin and eosin",
        "blood cells on a microscopy slide",
        "tissue biopsy under a medical microscope",
        "a pathology specimen under magnification",
    ],
    # ── Suspicious / fake ────────────────────────────────────────────────────
    "fake_medical_texture": [
        "a synthetic or artificial grayscale scan pattern",
        "a procedurally generated medical-looking texture",
        "a fake scan with no diagnostic information",
    ],
    "ai_generated_medical": [
        "an AI generated or computer synthesised chest X-ray",
        "a machine learning generated radiograph",
        "a synthetic medical image created by neural network",
    ],
    "generic_grayscale": [
        "a blank or nearly blank grayscale image",
        "a uniformly gray featureless image",
        "a dark blurry image with no structure",
    ],
    "non_medical_grayscale": [
        "a black and white photograph of everyday objects",
        "a grayscale outdoor scene or landscape",
        "a monochrome photograph unrelated to medicine",
    ],
}

# Groups that indicate real medical content
_REAL_MEDICAL: frozenset[str] = frozenset({
    "healthy_xray", "pneumonia_xray", "lung_opacity",
    "radiology_scan", "medical_microscopy",
})

# Groups that indicate suspicious / fake content
_SUSPICIOUS: frozenset[str] = frozenset({
    "fake_medical_texture", "ai_generated_medical",
    "generic_grayscale", "non_medical_grayscale",
})


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass
class MedicalRefinementResult:
    """
    Advisory output of the medical semantic refinement layer.

    All scores are advisory — they do not override the classifier.
    """

    semantic_medical_type: str       # dominant sub-group (e.g. "healthy_xray")
    medical_plausibility: float      # [0, 1] — how medical-looking the image is
    fake_medical_score: float        # [0, 1] — suspicion of fake/generated content
    semantic_margin: float           # gap between top-1 and top-2 sub-group probs
    group_scores: dict[str, float]   # full sub-group probability map
    refinement_reason: str           # Turkish advisory note
    inference_ms: float

    def as_dict(self) -> dict:
        return {
            "semantic_medical_type": self.semantic_medical_type,
            "medical_plausibility": round(self.medical_plausibility, 4),
            "fake_medical_score": round(self.fake_medical_score, 4),
            "semantic_margin": round(self.semantic_margin, 4),
            "refinement_reason": self.refinement_reason,
            "inference_ms": round(self.inference_ms, 2),
        }


# ── Refinement reason templates ───────────────────────────────────────────────

def _build_reason(
    medical_plausibility: float,
    fake_medical_score: float,
    semantic_margin: float,
    top_type: str,
) -> str:
    if fake_medical_score > 0.35:
        return (
            "Yapay veya sahte tıbbi içerik işaretleri tespit edildi; "
            "görüntü gerçek klinik görüntü olmayabilir."
        )
    if medical_plausibility > 0.65 and semantic_margin > 0.08:
        return (
            f"Görüntü '{top_type}' semantiğiyle yüksek uyum gösteriyor; "
            "gerçek tıbbi görüntü olduğuna dair güçlü kanıt var."
        )
    if medical_plausibility > 0.40:
        return (
            "Görüntü orta düzeyde tıbbi semantik uyum gösteriyor; "
            "sınıflandırıcı kararı destekleyici nitelikte."
        )
    if semantic_margin < 0.04:
        return (
            "Tıbbi alt-kategori sinyalleri belirsiz; "
            "görüntü içeriği semantik açıdan net değil."
        )
    return (
        "Görüntü tıbbi semantik açıdan zayıf uyum gösteriyor; "
        "sınıflandırıcı bulgularla birlikte değerlendirilmeli."
    )


# ── Refiner class ─────────────────────────────────────────────────────────────


class MedicalRefiner:
    """
    Lightweight CLIP-based medical semantic refiner.

    Reuses the ClipSemanticAnalyzer singleton — no second model load.
    Text embeddings are pre-computed once on first refine() call and cached.
    Image encoding adds ~25 ms overhead (one extra CLIP forward pass).

    Thread-safe: lazy loading is protected by threading.Lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready: bool = False

        # Set at _load() time
        self._text_features: torch.Tensor | None = None  # (N_prompts, D)
        self._prompt_to_group: list[int] = []            # prompt idx → group idx
        self._group_names: list[str] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def refine(self, image: Image.Image) -> MedicalRefinementResult:
        """
        Evaluate whether the image resembles genuine medical imagery.

        Parameters
        ----------
        image : PIL.Image.Image — any mode, RGB conversion handled internally.

        Returns
        -------
        MedicalRefinementResult with advisory plausibility / fake scores.
        """
        self._ensure_loaded()

        from app.modules.vision.semantic.semantic_analyzer import get_semantic_analyzer
        from app.modules.vision.semantic.semantic_utils import pil_to_clip_tensor

        t0 = time.perf_counter()
        analyzer = get_semantic_analyzer()

        if image.mode != "RGB":
            image = image.convert("RGB")

        # ── Image encoding ────────────────────────────────────────────────────
        img_tensor = pil_to_clip_tensor(image, analyzer._preprocess, analyzer._device)
        with torch.no_grad():
            img_feats = analyzer._model.encode_image(img_tensor)
            img_feats = F.normalize(img_feats, dim=-1)   # (1, D)

        # ── Per-group cosine similarity → softmax ─────────────────────────────
        per_prompt_sim = (img_feats @ self._text_features.T).squeeze(0)   # (N_prompts,)

        n_groups = len(self._group_names)
        group_sims = torch.zeros(n_groups, device=img_feats.device)
        group_counts = torch.zeros(n_groups, device=img_feats.device)
        for p_idx, g_idx in enumerate(self._prompt_to_group):
            group_sims[g_idx] += per_prompt_sim[p_idx]
            group_counts[g_idx] += 1

        # Sharpen the distribution: raw CLIP cosine similarities are extremely
        # flat (the typical span across 9 medical sub-groups is ~0.04), so a
        # plain softmax produces near-uniform group probabilities and a
        # near-zero semantic_margin.  Apply CLIP's standard inverse-temperature
        # of 1/0.01 = 100 to recover discriminative behavior.
        _CLIP_LOGIT_SCALE: float = 100.0
        group_logits = (group_sims / group_counts.clamp(min=1)) * _CLIP_LOGIT_SCALE
        group_probs = F.softmax(group_logits, dim=0)

        probs = group_probs.cpu().tolist()
        group_scores = {name: round(p, 4) for name, p in zip(self._group_names, probs)}

        # ── Derived scores ────────────────────────────────────────────────────
        medical_plausibility = sum(
            group_scores.get(g, 0.0) for g in _REAL_MEDICAL
        )
        fake_medical_score = sum(
            group_scores.get(g, 0.0) for g in _SUSPICIOUS
        )

        sorted_probs = sorted(probs, reverse=True)
        semantic_margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 0.0

        top_type = self._group_names[int(group_probs.argmax())]
        inference_ms = (time.perf_counter() - t0) * 1000

        reason = _build_reason(medical_plausibility, fake_medical_score, semantic_margin, top_type)

        logger.debug(
            "MedicalRefiner: type=%s plausibility=%.3f fake=%.3f margin=%.3f ms=%.1f",
            top_type, medical_plausibility, fake_medical_score, semantic_margin, inference_ms,
        )

        return MedicalRefinementResult(
            semantic_medical_type=top_type,
            medical_plausibility=round(medical_plausibility, 4),
            fake_medical_score=round(fake_medical_score, 4),
            semantic_margin=round(semantic_margin, 4),
            group_scores=group_scores,
            refinement_reason=reason,
            inference_ms=round(inference_ms, 2),
        )

    # ── Loading ───────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            self._load()
            self._ready = True

    def _load(self) -> None:
        from app.modules.vision.semantic.semantic_analyzer import get_semantic_analyzer
        import open_clip

        analyzer = get_semantic_analyzer()
        tokenizer = open_clip.get_tokenizer(analyzer.model_name)

        all_prompts: list[str] = []
        prompt_to_group: list[int] = []
        group_names: list[str] = []

        for g_idx, (group_name, prompts) in enumerate(_SUBGROUPS.items()):
            group_names.append(group_name)
            for prompt in prompts:
                all_prompts.append(prompt)
                prompt_to_group.append(g_idx)

        tokens = tokenizer(all_prompts).to(analyzer._device)
        with torch.no_grad():
            text_feats = analyzer._model.encode_text(tokens)
            text_feats = F.normalize(text_feats, dim=-1)

        self._text_features = text_feats
        self._prompt_to_group = prompt_to_group
        self._group_names = group_names

        logger.info(
            "MedicalRefiner: loaded %d sub-groups, %d prompts",
            len(group_names), len(all_prompts),
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_refiner_lock = threading.Lock()
_refiner_instance: MedicalRefiner | None = None


def get_medical_refiner() -> MedicalRefiner:
    """Return the module-level MedicalRefiner singleton (lazy load)."""
    global _refiner_instance
    if _refiner_instance is not None:
        return _refiner_instance
    with _refiner_lock:
        if _refiner_instance is not None:
            return _refiner_instance
        _refiner_instance = MedicalRefiner()
    return _refiner_instance
