"""
Utility helpers for the semantic analysis module.

Keeps semantic_analyzer.py focused on the core algorithm by isolating
device resolution, image preparation, and tensor math here.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = logging.getLogger(__name__)


def resolve_clip_device() -> torch.device:
    """
    Pick the best available device for CLIP inference.

    Priority: CUDA > MPS > CPU.
    Falls back to CPU silently on import errors.
    """
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


def pil_to_clip_tensor(
    image: "PILImage.Image",
    preprocess,
    device: torch.device,
) -> torch.Tensor:
    """
    Apply CLIP's own preprocessing transform and move to device.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image (any mode; RGB conversion handled by preprocess).
    preprocess : callable
        The transform returned by ``open_clip.create_model_and_transforms``.
    device : torch.device
        Target device.

    Returns
    -------
    (1, C, H, W) float tensor on ``device``.
    """
    return preprocess(image).unsqueeze(0).to(device)


def aggregate_prompt_similarities(
    image_features: torch.Tensor,
    text_features_per_prompt: torch.Tensor,
    prompt_to_category: list[int],
    n_categories: int,
) -> torch.Tensor:
    """
    Compute per-category cosine similarities by averaging over prompts.

    Parameters
    ----------
    image_features : (1, D) normalized float tensor
    text_features_per_prompt : (N_prompts, D) normalized float tensor
    prompt_to_category : list mapping prompt index → category index
    n_categories : int

    Returns
    -------
    (N_categories,) float tensor of mean cosine similarities (not softmaxed).
    """
    # (1, D) × (D, N_prompts) → (1, N_prompts)
    per_prompt_sim = (image_features @ text_features_per_prompt.T).squeeze(0)

    category_sims = torch.zeros(n_categories, device=image_features.device)
    category_counts = torch.zeros(n_categories, device=image_features.device)

    for prompt_idx, cat_idx in enumerate(prompt_to_category):
        category_sims[cat_idx] += per_prompt_sim[prompt_idx]
        category_counts[cat_idx] += 1

    # Avoid division by zero for categories with no prompts
    counts_safe = category_counts.clamp(min=1)
    return category_sims / counts_safe


def compute_scores(
    category_probs: torch.Tensor,
    category_names: list[str],
    is_medical_flags: list[bool],
    ood_weights: list[float],
    top_k: int = 5,
) -> tuple[str, float, float, list[dict], dict]:
    """
    Derive semantic scores from the per-category softmax probability vector.

    Returns
    -------
    top_label : str
    medical_relevance_score : float   — sum of medical category probs
    ood_score : float                 — weighted sum of non-medical probs
    top_matches : list[dict]          — top-K matches with label/score/rank
    all_scores : dict[str, float]     — full category probabilities
    """
    probs = category_probs.cpu().tolist()

    medical_relevance: float = 0.0
    ood_raw: float = 0.0

    for prob, is_med, ood_w in zip(probs, is_medical_flags, ood_weights):
        if is_med:
            medical_relevance += prob
        else:
            ood_raw += prob * ood_w

    all_scores = {name: float(p) for name, p in zip(category_names, probs)}

    ranked = sorted(
        zip(category_names, probs),
        key=lambda x: x[1],
        reverse=True,
    )
    top_label = ranked[0][0] if ranked else "unknown"
    top_matches = [
        {"label": name, "score": score, "rank": i + 1}
        for i, (name, score) in enumerate(ranked[:top_k])
    ]

    return top_label, medical_relevance, ood_raw, top_matches, all_scores
