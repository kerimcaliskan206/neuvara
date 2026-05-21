"""
Spatial attention bias benchmark for chest X-ray classifiers.

Purpose
-------
Quantify whether a model is learning spatial shortcuts (border artifacts,
diaphragm bias, corner patterns) vs. genuine pulmonary features.  Run this
on both the old full-image model and the new ROI model to produce a
before/after comparison.

Metrics computed per image
--------------------------
edge_attention_pct   — fraction of GradCAM mass in the outer 12% border ring.
                       High → model relies on edge/corner artifacts.
center_attention_pct — fraction in the central 50% × 50% region.
                       Low on a good model → it is NOT narrowly fixated on
                       the exact center but spans the lung fields.
lung_attention_pct   — fraction in a conservative estimate of the lung zone
                       (upper-center 70% × 60% of the image).
confidence_stability — std of softmax confidence across 5 slight perturbations.
                       High → model is sensitive to non-diagnostic variation.

Benchmark categories
--------------------
  centered_lungs        — ideal reference case
  edge_heavy            — significant visible border artifacts
  diaphragm_heavy       — lower-half dominated
  asymmetric            — one lung field missing or occluded
  lateral_pathology     — disease visible near lateral edge
  cropped_borders       — image hard-cropped at borders

Usage
-----
    from app.modules.vision.evaluation.bias_benchmark import BiasBenchmark

    bench = BiasBenchmark(model, architecture="efficientnet_b0", device="cpu")
    result = bench.evaluate_single(pil_image, category="centered_lungs")
    report = bench.evaluate_set(images_with_categories, class_idx=1)
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


# ── Per-image result ──────────────────────────────────────────────────────────


@dataclass
class BiasMetrics:
    """Spatial attention metrics for one image + one model."""

    edge_attention_pct: float       # 0–100; lower is better
    center_attention_pct: float     # 0–100
    lung_attention_pct: float       # 0–100; higher is better
    confidence: float               # predicted class confidence
    confidence_stability: float     # std over micro-perturbations; lower is better
    category: str = ""
    image_path: str = ""

    def as_dict(self) -> dict:
        return {
            "edge_attention_pct": round(self.edge_attention_pct, 2),
            "center_attention_pct": round(self.center_attention_pct, 2),
            "lung_attention_pct": round(self.lung_attention_pct, 2),
            "confidence": round(self.confidence, 4),
            "confidence_stability": round(self.confidence_stability, 4),
            "category": self.category,
        }


@dataclass
class BenchmarkReport:
    """Aggregate benchmark report for a set of images."""

    n_images: int = 0
    model_tag: str = ""

    # Mean metrics over the whole set
    edge_attention_mean: float = 0.0
    center_attention_mean: float = 0.0
    lung_attention_mean: float = 0.0
    confidence_stability_mean: float = 0.0

    # Per-category breakdown
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)

    # Raw per-image results (optional)
    per_image: list[BiasMetrics] = field(default_factory=list)

    def as_dict(self, include_per_image: bool = False) -> dict:
        d: dict = {
            "model_tag": self.model_tag,
            "n_images": self.n_images,
            "means": {
                "edge_attention_pct": round(self.edge_attention_mean, 2),
                "center_attention_pct": round(self.center_attention_mean, 2),
                "lung_attention_pct": round(self.lung_attention_mean, 2),
                "confidence_stability": round(self.confidence_stability_mean, 4),
            },
            "by_category": self.by_category,
        }
        if include_per_image:
            d["per_image"] = [m.as_dict() for m in self.per_image]
        return d

    def log(self) -> None:
        logger.info(
            "BiasBenchmark [%s]: n=%d | edge=%.1f%% | center=%.1f%% | "
            "lung=%.1f%% | conf_stability=%.4f",
            self.model_tag, self.n_images,
            self.edge_attention_mean, self.center_attention_mean,
            self.lung_attention_mean, self.confidence_stability_mean,
        )
        for cat, metrics in self.by_category.items():
            logger.info(
                "  %-20s edge=%.1f%% lung=%.1f%% conf_stability=%.4f (n=%d)",
                cat,
                metrics.get("edge_attention_pct", 0),
                metrics.get("lung_attention_pct", 0),
                metrics.get("confidence_stability", 0),
                int(metrics.get("n", 0)),
            )


# ── Benchmark ─────────────────────────────────────────────────────────────────


class BiasBenchmark:
    """
    Evaluate a model's spatial attention bias using Grad-CAM heatmaps.

    Parameters
    ----------
    model : nn.Module
        Trained classifier in eval mode.
    architecture : str
        Architecture name (needed to locate the GradCAM target layer).
    device : str | torch.device
    image_size : int
        Target image size for preprocessing (default 224).
    target_class_idx : int
        Class index whose gradient to visualise.  Typically the positive /
        pathology class.  If -1, uses the argmax prediction per image.
    edge_frac : float
        Fraction of each image edge that counts as the border ring (default 0.12).
    center_frac : float
        Fraction of image that forms the central region (default 0.50).
    lung_h_frac : float
        Height fraction of the lung zone from the top (default 0.75 → upper ¾).
    lung_w_frac : float
        Width fraction of the lung zone centred horizontally (default 0.70).
    n_perturbations : int
        Number of micro-perturbations for confidence stability test.
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        architecture: str,
        device: "torch.device | str" = "cpu",
        image_size: int = 224,
        target_class_idx: int = -1,
        edge_frac: float = 0.12,
        center_frac: float = 0.50,
        lung_h_frac: float = 0.75,
        lung_w_frac: float = 0.70,
        n_perturbations: int = 5,
    ) -> None:
        self.model = model
        self.architecture = architecture
        self.device = torch.device(device) if isinstance(device, str) else device
        self.image_size = image_size
        self.target_class_idx = target_class_idx
        self.edge_frac = edge_frac
        self.center_frac = center_frac
        self.lung_h_frac = lung_h_frac
        self.lung_w_frac = lung_w_frac
        self.n_perturbations = n_perturbations

        self.model.to(self.device).eval()
        self._gradcam = None  # lazy-loaded

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate_single(
        self,
        image: Image.Image,
        category: str = "",
        image_path: str = "",
    ) -> BiasMetrics:
        """Compute bias metrics for one PIL image."""
        tensor = self._preprocess(image)
        heatmap = self._get_heatmap(tensor)
        confidence, class_idx = self._predict(tensor)
        conf_stability = self._confidence_stability(tensor, class_idx)

        return BiasMetrics(
            edge_attention_pct=self._edge_pct(heatmap),
            center_attention_pct=self._center_pct(heatmap),
            lung_attention_pct=self._lung_pct(heatmap),
            confidence=confidence,
            confidence_stability=conf_stability,
            category=category,
            image_path=image_path,
        )

    def evaluate_set(
        self,
        images: list[tuple[Image.Image, str]],
        model_tag: str = "model",
    ) -> BenchmarkReport:
        """
        Evaluate a list of (PIL Image, category_string) pairs.

        Parameters
        ----------
        images : list of (PIL Image, category)
        model_tag : str
            Label used in the report (e.g. "full_image_model" or "roi_model").

        Returns
        -------
        BenchmarkReport
        """
        per_image: list[BiasMetrics] = []
        for img, category in images:
            try:
                m = self.evaluate_single(img, category=category)
                per_image.append(m)
            except Exception:
                logger.warning(
                    "BiasBenchmark: failed to evaluate image (category=%s)",
                    category, exc_info=True,
                )

        if not per_image:
            return BenchmarkReport(n_images=0, model_tag=model_tag)

        return self._aggregate(per_image, model_tag=model_tag)

    # ── Heatmap computation ───────────────────────────────────────────────────

    def _get_heatmap(self, tensor: "torch.Tensor") -> np.ndarray:
        """
        Compute Grad-CAM heatmap for the given tensor.

        Returns H×W float array normalised to [0, 1].
        """
        from app.modules.vision.explainability.gradcam import build_gradcam
        with build_gradcam(self.model, architecture=self.architecture) as cam:
            heatmap = cam.generate(tensor.to(self.device))
        return np.array(heatmap)

    # ── Spatial metric helpers ────────────────────────────────────────────────

    def _edge_pct(self, heatmap: np.ndarray) -> float:
        """Fraction of total attention mass inside the border ring."""
        h, w = heatmap.shape
        eh = max(1, int(h * self.edge_frac))
        ew = max(1, int(w * self.edge_frac))
        total = float(heatmap.sum()) or 1.0

        mask = np.zeros_like(heatmap, dtype=bool)
        mask[:eh, :] = True
        mask[-eh:, :] = True
        mask[:, :ew] = True
        mask[:, -ew:] = True

        return 100.0 * float(heatmap[mask].sum()) / total

    def _center_pct(self, heatmap: np.ndarray) -> float:
        """Fraction of attention mass in the central square region."""
        h, w = heatmap.shape
        mh = int(h * (1 - self.center_frac) / 2)
        mw = int(w * (1 - self.center_frac) / 2)
        total = float(heatmap.sum()) or 1.0
        center = heatmap[mh: h - mh, mw: w - mw]
        return 100.0 * float(center.sum()) / total

    def _lung_pct(self, heatmap: np.ndarray) -> float:
        """
        Fraction of attention in the conservative lung zone.

        The zone spans the upper lung_h_frac of the image height,
        centred in the middle lung_w_frac of the width.
        """
        h, w = heatmap.shape
        h_end = int(h * self.lung_h_frac)
        w_margin = int(w * (1 - self.lung_w_frac) / 2)
        total = float(heatmap.sum()) or 1.0
        lung_zone = heatmap[:h_end, w_margin: w - w_margin]
        return 100.0 * float(lung_zone.sum()) / total

    # ── Inference helpers ─────────────────────────────────────────────────────

    def _preprocess(self, image: Image.Image) -> "torch.Tensor":
        """PIL Image → (1, 3, H, W) normalised tensor."""
        import torchvision.transforms as T
        tf = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return tf(image.convert("RGB")).unsqueeze(0)

    @torch.no_grad()
    def _predict(self, tensor: "torch.Tensor") -> tuple[float, int]:
        """Return (confidence, class_index) for a (1, C, H, W) tensor."""
        logits = self.model(tensor.to(self.device))
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()
        class_idx = int(probs.argmax().item())
        return float(probs[class_idx].item()), class_idx

    @torch.no_grad()
    def _confidence_stability(
        self, tensor: "torch.Tensor", class_idx: int
    ) -> float:
        """
        Measure confidence variance under micro-perturbations.

        A high-variance model relies on non-diagnostic details that are
        sensitive to small pixel changes.
        """
        confs: list[float] = []
        for _ in range(self.n_perturbations):
            noise = torch.randn_like(tensor) * 0.02
            perturbed = tensor + noise
            logits = self.model(perturbed.to(self.device))
            probs = F.softmax(logits, dim=1).squeeze(0).cpu()
            confs.append(float(probs[class_idx].item()))
        return statistics.stdev(confs) if len(confs) > 1 else 0.0

    # ── Aggregation ───────────────────────────────────────────────────────────

    @staticmethod
    def _aggregate(
        results: list[BiasMetrics], model_tag: str
    ) -> BenchmarkReport:
        def _mean(vals: list[float]) -> float:
            return statistics.mean(vals) if vals else 0.0

        # Global means
        edge_mean = _mean([m.edge_attention_pct for m in results])
        center_mean = _mean([m.center_attention_pct for m in results])
        lung_mean = _mean([m.lung_attention_pct for m in results])
        stab_mean = _mean([m.confidence_stability for m in results])

        # Per-category breakdown
        by_cat: dict[str, dict[str, float]] = {}
        cats = {m.category for m in results}
        for cat in sorted(cats):
            subset = [m for m in results if m.category == cat]
            by_cat[cat] = {
                "n": len(subset),
                "edge_attention_pct": round(_mean([m.edge_attention_pct for m in subset]), 2),
                "center_attention_pct": round(_mean([m.center_attention_pct for m in subset]), 2),
                "lung_attention_pct": round(_mean([m.lung_attention_pct for m in subset]), 2),
                "confidence_stability": round(_mean([m.confidence_stability for m in subset]), 4),
            }

        return BenchmarkReport(
            n_images=len(results),
            model_tag=model_tag,
            edge_attention_mean=edge_mean,
            center_attention_mean=center_mean,
            lung_attention_mean=lung_mean,
            confidence_stability_mean=stab_mean,
            by_category=by_cat,
            per_image=results,
        )


# ── Comparison helper ─────────────────────────────────────────────────────────


def compare_reports(
    old_report: BenchmarkReport,
    new_report: BenchmarkReport,
) -> dict:
    """
    Compute delta metrics between an old (full-image) and new (ROI) model report.

    Positive delta on lung_attention_pct and negative on edge_attention_pct
    indicate successful bias reduction.
    """
    def _delta(new_val: float, old_val: float) -> dict:
        abs_d = new_val - old_val
        rel_d = (abs_d / old_val * 100) if old_val != 0 else 0.0
        return {"old": round(old_val, 2), "new": round(new_val, 2),
                "delta": round(abs_d, 2), "delta_pct": round(rel_d, 1)}

    return {
        "models": {
            "old": old_report.model_tag,
            "new": new_report.model_tag,
        },
        "edge_attention_pct": _delta(
            new_report.edge_attention_mean, old_report.edge_attention_mean
        ),
        "center_attention_pct": _delta(
            new_report.center_attention_mean, old_report.center_attention_mean
        ),
        "lung_attention_pct": _delta(
            new_report.lung_attention_mean, old_report.lung_attention_mean
        ),
        "confidence_stability": _delta(
            new_report.confidence_stability_mean, old_report.confidence_stability_mean
        ),
        "by_category": {
            cat: {
                "edge_attention_pct": _delta(
                    new_report.by_category.get(cat, {}).get("edge_attention_pct", 0),
                    old_report.by_category.get(cat, {}).get("edge_attention_pct", 0),
                ),
                "lung_attention_pct": _delta(
                    new_report.by_category.get(cat, {}).get("lung_attention_pct", 0),
                    old_report.by_category.get(cat, {}).get("lung_attention_pct", 0),
                ),
            }
            for cat in set(old_report.by_category) | set(new_report.by_category)
        },
    }
