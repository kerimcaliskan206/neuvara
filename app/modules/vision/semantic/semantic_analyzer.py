"""
CLIP-based semantic analyzer for the HantaProject vision pipeline.

Adds a semantic understanding stage BEFORE the medical EfficientNet classifier.
This module is entirely isolated — it does NOT touch prediction routing, fusion,
or the medical classifier in any way.

Architecture position
---------------------
    image
      │
      ▼
  ┌─────────────────────────┐
  │   SemanticAnalyzer      │  ← This module (phase 1)
  │   CLIP ViT-B-32         │
  │   • scene semantics     │
  │   • medical relevance   │
  │   • OOD likelihood      │
  └────────────┬────────────┘
               │ SemanticResult
               ▼
  ┌─────────────────────────┐
  │   EfficientNet          │  (unchanged, existing pipeline)
  │   medical classifier    │
  └────────────┬────────────┘
               │
               ▼
          fusion decision

Singleton
---------
The CLIP model is large (~350 MB). It is loaded once on first call to
``get_semantic_analyzer()`` and reused for all subsequent requests.
The load is protected by a threading.Lock so concurrent startup calls
are safe in multi-threaded ASGI servers.

Usage
-----
    from app.modules.vision.semantic.semantic_analyzer import get_semantic_analyzer

    analyzer = get_semantic_analyzer()          # loads CLIP once
    result = analyzer.analyze(pil_image)
    print(result.as_dict())
"""
from __future__ import annotations

import logging
import threading
import time
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

from app.modules.vision.semantic.semantic_types import (
    SemanticCategory,
    SemanticMatch,
    SemanticResult,
)
from app.modules.vision.semantic.semantic_utils import (
    aggregate_prompt_similarities,
    compute_scores,
    pil_to_clip_tensor,
    resolve_clip_device,
)

logger = logging.getLogger(__name__)

# ── Model configuration ───────────────────────────────────────────────────────

_CLIP_MODEL_NAME = "ViT-B-32"
_CLIP_PRETRAINED = "openai"
_TOP_K_RESULTS = 5

# ── Semantic category table ───────────────────────────────────────────────────
#
# Prompts are averaged into a single per-category embedding at load time —
# multiple prompts improve coverage without runtime cost.
#
# ood_weight: how "out-of-distribution" this category is relative to the
#   medical hantavirus pipeline.
#   0.0 = fully in-distribution (medical images the classifier was built for)
#   1.0 = maximally OOD (furniture, vehicles, food — completely unrelated)
#   Intermediate values reflect partial relevance (rodent: 0.25 — relevant as
#   a disease vector but not a medical image the classifier processes).

_CATEGORIES: tuple[SemanticCategory, ...] = (
    SemanticCategory(
        name="medical_xray",
        prompts=(
            "a chest x-ray image",
            "a lung x-ray radiograph",
            "a medical x-ray scan",
            "a thoracic radiograph",
        ),
        is_medical=True,
        ood_weight=0.0,
    ),
    SemanticCategory(
        name="medical_microscopy",
        prompts=(
            "a medical microscope image",
            "a histopathology slide",
            "blood cells under a microscope",
            "microscopic tissue sample",
            "a pathology image",
        ),
        is_medical=True,
        ood_weight=0.0,
    ),
    SemanticCategory(
        name="rodent",
        prompts=(
            "a rodent",
            "a rat",
            "a mouse",
            "a hamster",
            "a small rodent mammal",
        ),
        is_medical=False,
        ood_weight=0.25,  # disease vector — adjacent to domain but not a classifier input
    ),
    SemanticCategory(
        name="wildlife",
        prompts=(
            "a wild animal",
            "a gorilla",
            "a monkey primate",
            "a bear",
            "a wolf",
            "a fox",
            "a wild mammal in nature",
        ),
        is_medical=False,
        ood_weight=0.90,
    ),
    SemanticCategory(
        name="human",
        prompts=(
            "a photograph of a person",
            "a human face",
            "a portrait photograph",
            "a person standing",
        ),
        is_medical=False,
        ood_weight=0.50,  # could be patient photo — partially OOD
    ),
    SemanticCategory(
        name="food",
        prompts=(
            "a food photograph",
            "a meal on a plate",
            "food ingredients",
            "a dish of food",
        ),
        is_medical=False,
        ood_weight=1.0,
    ),
    SemanticCategory(
        name="furniture",
        prompts=(
            "a piece of furniture",
            "a chair",
            "a table",
            "a sofa",
            "indoor furniture",
        ),
        is_medical=False,
        ood_weight=1.0,
    ),
    SemanticCategory(
        name="vehicle",
        prompts=(
            "a car",
            "a vehicle",
            "a truck",
            "a motorcycle",
            "an automobile",
        ),
        is_medical=False,
        ood_weight=1.0,
    ),
    SemanticCategory(
        name="indoor_scene",
        prompts=(
            "an indoor room scene",
            "inside a building",
            "an office interior",
            "a living room",
        ),
        is_medical=False,
        ood_weight=0.80,
    ),
    SemanticCategory(
        name="outdoor_scene",
        prompts=(
            "an outdoor landscape",
            "a nature scene",
            "a forest",
            "outdoor scenery",
            "a natural environment",
        ),
        is_medical=False,
        ood_weight=0.70,
    ),
    SemanticCategory(
        name="random_object",
        prompts=(
            "a random everyday object",
            "a household item",
            "a miscellaneous object",
            "an unrelated item",
        ),
        is_medical=False,
        ood_weight=1.0,
    ),
)


# ── Analyzer ──────────────────────────────────────────────────────────────────


class ClipSemanticAnalyzer:
    """
    CLIP-based semantic scene understanding for incoming images.

    Stateless after initialization — ``analyze()`` only reads model weights
    and cached text features. Safe to call from multiple threads concurrently.

    Parameters
    ----------
    model_name : str
        open_clip architecture name (default: ``"ViT-B-32"``).
    pretrained : str
        Pretrained weights tag (default: ``"openai"``).
    device : torch.device | None
        Inference device.  Auto-resolved when None.
    categories : tuple[SemanticCategory, ...]
        The semantic category table to use.  Defaults to the module-level
        ``_CATEGORIES`` constant.
    """

    def __init__(
        self,
        model_name: str = _CLIP_MODEL_NAME,
        pretrained: str = _CLIP_PRETRAINED,
        device: Optional[torch.device] = None,
        categories: tuple[SemanticCategory, ...] = _CATEGORIES,
    ) -> None:
        self._model_name = model_name
        self._pretrained = pretrained
        self._device = device or resolve_clip_device()
        self._categories = categories

        # Loaded by _load()
        self._model = None
        self._preprocess = None

        # Cached text features: (N_prompts_total, D) — computed once at load
        self._text_features: Optional[torch.Tensor] = None
        self._prompt_to_cat_idx: list[int] = []      # prompt index → category index
        self._category_names: list[str] = []
        self._is_medical_flags: list[bool] = []
        self._ood_weights: list[float] = []

        self._load()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load CLIP model and pre-encode all category text prompts."""
        import open_clip  # defer import — keeps this optional at module level

        t_start = time.perf_counter()
        logger.info(
            "ClipSemanticAnalyzer: loading %s/%s on %s …",
            self._model_name, self._pretrained, self._device,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model, _, preprocess = open_clip.create_model_and_transforms(
                self._model_name, pretrained=self._pretrained
            )

        model.to(self._device).eval()
        self._model = model
        self._preprocess = preprocess

        # ── Pre-encode all text prompts ──────────────────────────────────────
        tokenizer = open_clip.get_tokenizer(self._model_name)

        all_prompts: list[str] = []
        prompt_to_cat_idx: list[int] = []
        category_names: list[str] = []
        is_medical_flags: list[bool] = []
        ood_weights: list[float] = []

        for cat_idx, cat in enumerate(self._categories):
            category_names.append(cat.name)
            is_medical_flags.append(cat.is_medical)
            ood_weights.append(cat.ood_weight)
            for prompt in cat.prompts:
                all_prompts.append(prompt)
                prompt_to_cat_idx.append(cat_idx)

        tokens = tokenizer(all_prompts).to(self._device)
        with torch.no_grad():
            text_feats = self._model.encode_text(tokens)
            text_feats = F.normalize(text_feats, dim=-1)

        self._text_features = text_feats        # (N_prompts, D)
        self._prompt_to_cat_idx = prompt_to_cat_idx
        self._category_names = category_names
        self._is_medical_flags = is_medical_flags
        self._ood_weights = ood_weights

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "ClipSemanticAnalyzer: ready | model=%s/%s | device=%s | "
            "categories=%d | prompts=%d | load_ms=%.0f",
            self._model_name, self._pretrained, self._device,
            len(self._categories), len(all_prompts), elapsed_ms,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._text_features is not None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def pretrained(self) -> str:
        return self._pretrained

    def analyze(
        self,
        image: Image.Image,
        top_k: int = _TOP_K_RESULTS,
    ) -> SemanticResult:
        """
        Run semantic analysis on a single PIL image.

        Parameters
        ----------
        image : PIL.Image.Image
            Input image.  Any mode is accepted (converted to RGB internally).
        top_k : int
            Number of top-category matches to include in the result.

        Returns
        -------
        SemanticResult with:
            - ``top_semantic_label``: highest-probability category name
            - ``medical_relevance_score``: [0, 1] sum of medical category probs
            - ``ood_score``: [0, 1] weighted OOD probability
            - ``top_matches``: ranked list of top-K categories
            - ``all_scores``: full category probability map
            - ``inference_ms``: encoding time (ms), excludes text pre-computation
        """
        if not self.is_ready:
            raise RuntimeError(
                "ClipSemanticAnalyzer is not loaded. Call _load() first."
            )

        if image.mode != "RGB":
            image = image.convert("RGB")

        t_start = time.perf_counter()

        # ── Image encoding ────────────────────────────────────────────────────
        img_tensor = pil_to_clip_tensor(image, self._preprocess, self._device)
        with torch.no_grad():
            img_feats = self._model.encode_image(img_tensor)
            img_feats = F.normalize(img_feats, dim=-1)    # (1, D)

        # ── Per-category similarity (mean over prompts) ───────────────────────
        cat_sims = aggregate_prompt_similarities(
            image_features=img_feats,
            text_features_per_prompt=self._text_features,
            prompt_to_category=self._prompt_to_cat_idx,
            n_categories=len(self._categories),
        )                                                  # (N_categories,)

        # Softmax over categories → probability distribution
        cat_probs = F.softmax(cat_sims, dim=0)

        # ── Score computation ─────────────────────────────────────────────────
        top_label, medical_relevance, ood_score, top_matches_raw, all_scores = (
            compute_scores(
                category_probs=cat_probs,
                category_names=self._category_names,
                is_medical_flags=self._is_medical_flags,
                ood_weights=self._ood_weights,
                top_k=top_k,
            )
        )

        inference_ms = (time.perf_counter() - t_start) * 1000

        logger.debug(
            "SemanticAnalyzer: top=%s medical_rel=%.3f ood=%.3f "
            "top3=%s inference_ms=%.1f",
            top_label,
            medical_relevance,
            ood_score,
            [(m["label"], round(m["score"], 3)) for m in top_matches_raw[:3]],
            inference_ms,
        )

        top_matches = [
            SemanticMatch(
                label=m["label"],
                score=round(m["score"], 4),
                rank=m["rank"],
            )
            for m in top_matches_raw
        ]

        return SemanticResult(
            top_semantic_label=top_label,
            medical_relevance_score=round(medical_relevance, 4),
            ood_score=round(ood_score, 4),
            top_matches=top_matches,
            all_scores={k: round(v, 4) for k, v in all_scores.items()},
            inference_ms=round(inference_ms, 2),
            model_name=self._model_name,
            model_pretrained=self._pretrained,
        )

    def analyze_batch(
        self,
        images: list[Image.Image],
        top_k: int = _TOP_K_RESULTS,
    ) -> list[SemanticResult]:
        """
        Run semantic analysis on a batch of images.

        Stacks images into a single CLIP forward pass — more efficient than
        calling ``analyze()`` in a loop for large batches.
        """
        if not self.is_ready:
            raise RuntimeError("ClipSemanticAnalyzer is not loaded.")
        if not images:
            return []

        t_start = time.perf_counter()

        converted = [img.convert("RGB") if img.mode != "RGB" else img for img in images]
        tensors = torch.cat(
            [pil_to_clip_tensor(img, self._preprocess, self._device) for img in converted],
            dim=0,
        )   # (B, C, H, W)

        with torch.no_grad():
            img_feats = self._model.encode_image(tensors)
            img_feats = F.normalize(img_feats, dim=-1)    # (B, D)

        total_ms = (time.perf_counter() - t_start) * 1000
        per_image_ms = total_ms / len(images)

        results = []
        for i, single_feat in enumerate(img_feats):
            cat_sims = aggregate_prompt_similarities(
                image_features=single_feat.unsqueeze(0),
                text_features_per_prompt=self._text_features,
                prompt_to_category=self._prompt_to_cat_idx,
                n_categories=len(self._categories),
            )
            cat_probs = F.softmax(cat_sims, dim=0)
            top_label, medical_rel, ood_sc, top_matches_raw, all_scores = compute_scores(
                category_probs=cat_probs,
                category_names=self._category_names,
                is_medical_flags=self._is_medical_flags,
                ood_weights=self._ood_weights,
                top_k=top_k,
            )
            results.append(SemanticResult(
                top_semantic_label=top_label,
                medical_relevance_score=round(medical_rel, 4),
                ood_score=round(ood_sc, 4),
                top_matches=[
                    SemanticMatch(label=m["label"], score=round(m["score"], 4), rank=m["rank"])
                    for m in top_matches_raw
                ],
                all_scores={k: round(v, 4) for k, v in all_scores.items()},
                inference_ms=round(per_image_ms, 2),
                model_name=self._model_name,
                model_pretrained=self._pretrained,
            ))

        logger.debug(
            "SemanticAnalyzer batch: n=%d total_ms=%.1f per_image_ms=%.1f",
            len(images), total_ms, per_image_ms,
        )
        return results

    def model_info(self) -> dict:
        """Return metadata about the loaded CLIP model."""
        return {
            "model_name": self._model_name,
            "pretrained": self._pretrained,
            "device": str(self._device),
            "n_categories": len(self._categories),
            "category_names": self._category_names,
            "is_ready": self.is_ready,
        }


# ── Module-level singleton ────────────────────────────────────────────────────
#
# Loaded lazily on first call to get_semantic_analyzer().
# Protected by a lock so concurrent first-calls in threaded ASGI
# servers do not double-load the model.

_lock = threading.Lock()
_instance: Optional[ClipSemanticAnalyzer] = None


def get_semantic_analyzer() -> ClipSemanticAnalyzer:
    """
    Return the module-level singleton, loading CLIP on the first call.

    Thread-safe: concurrent first-calls block on the lock while the
    first caller loads the model; subsequent callers get the cached instance.

    Raises RuntimeError if the model fails to load.
    """
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        # Double-check inside the lock (classic DCLP)
        if _instance is not None:
            return _instance
        logger.info("get_semantic_analyzer: initializing CLIP singleton …")
        _instance = ClipSemanticAnalyzer()
        logger.info("get_semantic_analyzer: CLIP singleton ready")
    return _instance
