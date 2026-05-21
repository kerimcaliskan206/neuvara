"""
Class imbalance handling for vision training.

Three complementary strategies are provided:

  1. WeightedRandomSampler — over-samples the minority class at DataLoader
     level. The model sees a balanced batch distribution even from an
     imbalanced dataset. No images are discarded.

  2. CrossEntropyLoss class weights — the loss function penalizes errors on
     the minority class more heavily. Combines well with the sampler.

  3. Class ratio analysis — computes imbalance severity and recommends
     which strategy to use.

Strategy selection guide
------------------------
  imbalance_ratio < 2:1   → no action needed
  2:1 – 5:1               → weighted loss (light correction)
  5:1 – 20:1              → weighted sampler + weighted loss
  > 20:1                  → consider oversampling augmentation or
                             synthetic minority generation (SMOTE-like)
"""
from __future__ import annotations

import logging
import math
from collections import Counter

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


def compute_class_weights(
    class_counts: dict[str, int],
    strategy: str = "inverse_frequency",
) -> dict[str, float]:
    """
    Compute per-class weights for ``CrossEntropyLoss(weight=...)``.

    Parameters
    ----------
    class_counts : dict mapping class_name → count
    strategy : str
        "inverse_frequency" — weight = N_total / (N_classes * N_class)
        "effective_samples"  — weight = (1 - beta) / (1 - beta^n)
                               where beta=0.999 (recommended for N < 1000)
        "sqrt_inverse"       — softer version of inverse_frequency

    Returns
    -------
    dict mapping class_name → float weight
    """
    total = sum(class_counts.values())
    n_classes = len(class_counts)

    if n_classes == 0 or total == 0:
        return {}

    weights: dict[str, float] = {}

    if strategy == "inverse_frequency":
        for cls, count in class_counts.items():
            weights[cls] = total / (n_classes * max(count, 1))

    elif strategy == "sqrt_inverse":
        for cls, count in class_counts.items():
            weights[cls] = math.sqrt(total / (n_classes * max(count, 1)))

    elif strategy == "effective_samples":
        beta = 0.999
        for cls, count in class_counts.items():
            effective_n = (1.0 - beta ** count) / (1.0 - beta)
            weights[cls] = 1.0 / max(effective_n, 1e-8)
        # Normalize so the mean weight is 1.0
        mean_w = sum(weights.values()) / n_classes
        weights = {k: v / mean_w for k, v in weights.items()}

    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use: inverse_frequency, sqrt_inverse, effective_samples")

    logger.info(
        "Class weights (%s): %s",
        strategy,
        {k: round(v, 4) for k, v in weights.items()},
    )
    return weights


def compute_class_weights_tensor(
    class_counts: dict[str, int],
    class_names: list[str],
    strategy: str = "inverse_frequency",
) -> torch.Tensor:
    """
    Return class weights as a float Tensor in label-index order.

    Use as: ``CrossEntropyLoss(weight=tensor)``
    """
    weights_dict = compute_class_weights(class_counts, strategy)
    weights_list = [weights_dict.get(cls, 1.0) for cls in class_names]
    return torch.tensor(weights_list, dtype=torch.float32)


def build_weighted_sampler(
    dataset: Dataset,
    class_names: list[str],
    num_samples: int | None = None,
) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that equalizes class frequencies.

    Parameters
    ----------
    dataset : Dataset
        Must support indexing such that ``dataset[i][1]`` returns the class
        label (integer index). Compatible with ``ImageFolderDataset``.
    class_names : list[str]
        Class names in label-index order.
    num_samples : int | None
        Number of samples per epoch. Defaults to len(dataset) which
        produces exactly one synthetic balanced epoch.

    Returns
    -------
    WeightedRandomSampler — pass to ``DataLoader(sampler=...)``.
    Note: when using a sampler, set ``shuffle=False`` in DataLoader.
    """
    # Count per-class occurrences by scanning labels
    label_counts: Counter[int] = Counter()
    all_labels: list[int] = []
    for i in range(len(dataset)):  # type: ignore[arg-type]
        label = dataset[i][1]
        label_counts[label] += 1
        all_labels.append(label)

    class_counts = {class_names[idx]: cnt for idx, cnt in label_counts.items()}
    weights_dict = compute_class_weights(class_counts)

    # Map each sample to its per-class weight
    sample_weights = [
        weights_dict.get(class_names[label], 1.0)
        for label in all_labels
    ]

    n_samples = num_samples or len(dataset)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=n_samples,
        replacement=True,
    )
    logger.info(
        "WeightedRandomSampler: %d samples/epoch | class_counts=%s",
        n_samples, class_counts,
    )
    return sampler


def imbalance_report(class_counts: dict[str, int]) -> dict:
    """
    Analyze class imbalance and recommend a mitigation strategy.

    Returns
    -------
    dict with imbalance_ratio, recommendation, and per-class breakdown.
    """
    if not class_counts:
        return {"error": "empty class_counts"}

    total = sum(class_counts.values())
    majority = max(class_counts.values())
    minority = min(class_counts.values())
    ratio = majority / max(minority, 1)

    if ratio < 2:
        severity = "balanced"
        recommendation = "No action needed. Dataset is well-balanced."
    elif ratio < 5:
        severity = "mild"
        recommendation = (
            "Apply weighted CrossEntropyLoss. "
            "strategy='inverse_frequency' is sufficient."
        )
    elif ratio < 20:
        severity = "moderate"
        recommendation = (
            "Apply WeightedRandomSampler + weighted CrossEntropyLoss. "
            "Consider Focal Loss as an alternative."
        )
    else:
        severity = "severe"
        recommendation = (
            "Severe imbalance detected. Apply WeightedRandomSampler + Focal Loss. "
            "Consider augmenting the minority class more aggressively "
            "or collecting more minority samples."
        )

    logger.info(
        "Imbalance: ratio=%.1f:1 | severity=%s | %s",
        ratio, severity, class_counts,
    )

    return {
        "total_images": total,
        "class_counts": class_counts,
        "majority_class": max(class_counts, key=class_counts.get),  # type: ignore[arg-type]
        "minority_class": min(class_counts, key=class_counts.get),  # type: ignore[arg-type]
        "imbalance_ratio": round(ratio, 2),
        "severity": severity,
        "recommendation": recommendation,
        "suggested_weights": compute_class_weights(class_counts),
    }
