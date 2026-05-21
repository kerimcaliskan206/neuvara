"""
Systematic GradCAM audit for the segmented lung ROI model.

Evaluates whether GradCAM activations are anatomically plausible across a
dataset sample: lungs should dominate, edges/corners should be quiet.

Produces a `GradCAMAuditReport` with per-image records and aggregate stats.

Usage
-----
    from app.modules.vision.evaluation.gradcam_audit import GradCAMAudit

    audit = GradCAMAudit(model, architecture, device)
    report = audit.run(dataset, n_samples=100)
    print(report.summary())
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ── Result dataclasses ────────────────────────────────────────────────────────


@dataclass
class AuditRecord:
    """GradCAM quality assessment for one image."""

    image_path: str
    class_label: str
    confidence: float
    # Fraction of total GradCAM mass in each anatomical zone
    lung_attention_pct: float       # upper 75% height × central 70% width
    edge_attention_pct: float       # outer 12% border ring
    corner_attention_pct: float     # 4 corner squares (10% × 10%)
    diaphragm_attention_pct: float  # bottom 20% height
    left_right_balance: float       # abs(left_half - right_half); 0 = symmetric
    passed: bool                    # True when lung > edge and corner < 0.10


@dataclass
class GradCAMAuditReport:
    """Aggregate results from GradCAMAudit.run()."""

    n_images: int
    model_tag: str
    # Aggregate means
    mean_lung_attention: float
    mean_edge_attention: float
    mean_corner_attention: float
    mean_diaphragm_attention: float
    mean_left_right_balance: float
    pass_rate: float
    # Per-class breakdown {class_label: {metric: mean}}
    by_class: dict = field(default_factory=dict)
    # Full per-image records
    records: list[AuditRecord] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"GradCAM Audit — {self.model_tag}",
            f"  samples      : {self.n_images}",
            f"  pass rate    : {self.pass_rate:.1%}",
            f"  lung attn    : {self.mean_lung_attention:.1%}",
            f"  edge attn    : {self.mean_edge_attention:.1%}",
            f"  corner attn  : {self.mean_corner_attention:.1%}",
            f"  diaphragm    : {self.mean_diaphragm_attention:.1%}",
            f"  L/R balance  : {self.mean_left_right_balance:.3f}",
        ]
        if self.by_class:
            lines.append("  by class:")
            for cls, stats in self.by_class.items():
                lines.append(
                    f"    {cls}: lung={stats['lung']:.1%} "
                    f"edge={stats['edge']:.1%} pass={stats['pass']:.1%}"
                )
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "n_images": self.n_images,
            "model_tag": self.model_tag,
            "mean_lung_attention": round(self.mean_lung_attention, 4),
            "mean_edge_attention": round(self.mean_edge_attention, 4),
            "mean_corner_attention": round(self.mean_corner_attention, 4),
            "mean_diaphragm_attention": round(self.mean_diaphragm_attention, 4),
            "mean_left_right_balance": round(self.mean_left_right_balance, 4),
            "pass_rate": round(self.pass_rate, 4),
            "by_class": {
                cls: {k: round(v, 4) for k, v in stats.items()}
                for cls, stats in self.by_class.items()
            },
        }


# ── Audit engine ──────────────────────────────────────────────────────────────


class GradCAMAudit:
    """
    Runs GradCAM on a sample of images and scores anatomical plausibility.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded model in eval mode.
    architecture : str
        Architecture name (for GradCAM layer selection).
    device : torch.device | str
    image_size : int
        Square input size the model expects (default 224).
    lung_h_frac : float
        Fraction of image height considered the lung zone (from top).
    lung_w_frac : float
        Fraction of image width considered the lung zone (centered).
    edge_frac : float
        Outer border fraction considered the edge zone.
    corner_frac : float
        Corner square fraction (each corner = corner_frac × corner_frac).
    diaphragm_frac : float
        Bottom fraction considered the diaphragm zone.
    target_class_idx : int
        Class index for GradCAM. -1 = predicted class.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        architecture: str,
        device: torch.device | str,
        image_size: int = 224,
        lung_h_frac: float = 0.75,
        lung_w_frac: float = 0.70,
        edge_frac: float = 0.12,
        corner_frac: float = 0.10,
        diaphragm_frac: float = 0.20,
        target_class_idx: int = -1,
    ) -> None:
        self.model = model
        self.architecture = architecture
        self.device = torch.device(device)
        self.image_size = image_size
        self.lung_h_frac = lung_h_frac
        self.lung_w_frac = lung_w_frac
        self.edge_frac = edge_frac
        self.corner_frac = corner_frac
        self.diaphragm_frac = diaphragm_frac
        self.target_class_idx = target_class_idx

    def run(
        self,
        dataset,
        n_samples: int = 100,
        seed: int = 42,
        model_tag: str = "model",
        image_paths: Optional[list[str]] = None,
    ) -> GradCAMAuditReport:
        """
        Evaluate GradCAM quality on `n_samples` items from `dataset`.

        `dataset` must support `__len__`, `__getitem__` returning (tensor, label),
        and optionally expose `.samples` list of (path, class_idx) tuples
        (standard ImageFolder interface).

        Parameters
        ----------
        dataset
            Any torch Dataset returning (tensor, int).
        n_samples : int
            Maximum number of images to evaluate.
        seed : int
            RNG seed for reproducible sampling.
        model_tag : str
            Label embedded in the report.
        image_paths : list[str] | None
            Pre-resolved image paths aligned to dataset indices. If None,
            the code will attempt to read from dataset.samples.
        """
        from app.modules.vision.explainability.gradcam import build_gradcam
        from app.modules.vision.preprocessing.pipeline import ImagePreprocessingPipeline

        pipeline = ImagePreprocessingPipeline()

        indices = list(range(len(dataset)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        indices = indices[:n_samples]

        # Resolve paths
        paths: list[Optional[str]] = []
        if image_paths is not None:
            paths = [image_paths[i] for i in indices]
        elif hasattr(dataset, "samples"):
            paths = [str(dataset.samples[i][0]) for i in indices]
        elif hasattr(dataset, "_samples"):
            paths = [str(dataset._samples[i][0]) for i in indices]
        else:
            paths = [None] * len(indices)

        # Resolve class name mapping
        class_names: list[str] = []
        if hasattr(dataset, "class_names"):
            class_names = dataset.class_names
        elif hasattr(dataset, "classes"):
            class_names = dataset.classes

        records: list[AuditRecord] = []

        for i, (idx, img_path) in enumerate(zip(indices, paths)):
            try:
                tensor, label_idx = dataset[idx]
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
                tensor = tensor.to(self.device)

                with torch.no_grad():
                    import torch.nn.functional as F
                    logits = self.model(tensor)
                    probs = F.softmax(logits, dim=1).squeeze(0)

                pred_idx = int(probs.argmax().item())
                confidence = float(probs[pred_idx].item())

                class_idx = pred_idx if self.target_class_idx == -1 else self.target_class_idx

                with build_gradcam(self.model, self.architecture) as gradcam:
                    cam_tensor = gradcam.generate(tensor, class_idx=class_idx)
                heatmap = cam_tensor.numpy() if cam_tensor is not None else None
                if heatmap is None:
                    logger.debug("GradCAMAudit: no heatmap for idx=%d", idx)
                    continue

                record = self._assess(
                    heatmap=heatmap,
                    label_idx=int(label_idx),
                    pred_idx=pred_idx,
                    confidence=confidence,
                    img_path=str(img_path) if img_path else f"idx_{idx}",
                    class_names=class_names,
                )
                records.append(record)

                if (i + 1) % 20 == 0:
                    logger.info("GradCAMAudit: %d/%d done", i + 1, len(indices))

            except Exception:
                logger.exception("GradCAMAudit: error on idx=%d", idx)
                continue

        return self._build_report(records, model_tag)

    # ── Zone analysis ─────────────────────────────────────────────────────────

    def _assess(
        self,
        heatmap: np.ndarray,
        label_idx: int,
        pred_idx: int,
        confidence: float,
        img_path: str,
        class_names: list[str],
    ) -> AuditRecord:
        if heatmap.max() > 0:
            h = heatmap / heatmap.max()
        else:
            h = heatmap
        total = h.sum() or 1.0
        H, W = h.shape

        lung = self._lung_pct(h, H, W, total)
        edge = self._edge_pct(h, H, W, total)
        corner = self._corner_pct(h, H, W, total)
        diaphragm = self._diaphragm_pct(h, H, W, total)
        balance = self._lr_balance(h, H, W)

        class_label = (
            class_names[pred_idx] if class_names and pred_idx < len(class_names)
            else str(pred_idx)
        )
        passed = lung > edge and corner < 0.10

        return AuditRecord(
            image_path=img_path,
            class_label=class_label,
            confidence=confidence,
            lung_attention_pct=float(lung),
            edge_attention_pct=float(edge),
            corner_attention_pct=float(corner),
            diaphragm_attention_pct=float(diaphragm),
            left_right_balance=float(balance),
            passed=passed,
        )

    def _lung_pct(self, h, H, W, total) -> float:
        y2 = int(H * self.lung_h_frac)
        x_margin = int(W * (1 - self.lung_w_frac) / 2)
        x1, x2 = x_margin, W - x_margin
        return float(h[:y2, x1:x2].sum() / total)

    def _edge_pct(self, h, H, W, total) -> float:
        ey = int(H * self.edge_frac)
        ex = int(W * self.edge_frac)
        mask = np.zeros_like(h, dtype=bool)
        mask[:ey, :] = True
        mask[-ey:, :] = True
        mask[:, :ex] = True
        mask[:, -ex:] = True
        return float(h[mask].sum() / total)

    def _corner_pct(self, h, H, W, total) -> float:
        cy = int(H * self.corner_frac)
        cx = int(W * self.corner_frac)
        corners = (
            h[:cy, :cx].sum()
            + h[:cy, -cx:].sum()
            + h[-cy:, :cx].sum()
            + h[-cy:, -cx:].sum()
        )
        return float(corners / total)

    def _diaphragm_pct(self, h, H, W, total) -> float:
        y1 = int(H * (1 - self.diaphragm_frac))
        return float(h[y1:, :].sum() / total)

    def _lr_balance(self, h, H, W) -> float:
        mid = W // 2
        left = float(h[:, :mid].sum())
        right = float(h[:, mid:].sum())
        total = left + right or 1.0
        return abs(left / total - right / total)

    # ── Report construction ───────────────────────────────────────────────────

    def _build_report(
        self, records: list[AuditRecord], model_tag: str
    ) -> GradCAMAuditReport:
        n = len(records)
        if n == 0:
            logger.warning("GradCAMAudit: no valid records produced")
            return GradCAMAuditReport(
                n_images=0,
                model_tag=model_tag,
                mean_lung_attention=0.0,
                mean_edge_attention=0.0,
                mean_corner_attention=0.0,
                mean_diaphragm_attention=0.0,
                mean_left_right_balance=0.0,
                pass_rate=0.0,
            )

        mean_lung = sum(r.lung_attention_pct for r in records) / n
        mean_edge = sum(r.edge_attention_pct for r in records) / n
        mean_corner = sum(r.corner_attention_pct for r in records) / n
        mean_diaphragm = sum(r.diaphragm_attention_pct for r in records) / n
        mean_balance = sum(r.left_right_balance for r in records) / n
        pass_rate = sum(r.passed for r in records) / n

        # Per-class breakdown
        by_class: dict[str, dict] = {}
        for r in records:
            cls = r.class_label
            if cls not in by_class:
                by_class[cls] = {"lung": [], "edge": [], "corner": [], "pass": []}
            by_class[cls]["lung"].append(r.lung_attention_pct)
            by_class[cls]["edge"].append(r.edge_attention_pct)
            by_class[cls]["corner"].append(r.corner_attention_pct)
            by_class[cls]["pass"].append(float(r.passed))

        by_class_means = {
            cls: {k: sum(v) / len(v) for k, v in stats.items()}
            for cls, stats in by_class.items()
        }

        report = GradCAMAuditReport(
            n_images=n,
            model_tag=model_tag,
            mean_lung_attention=mean_lung,
            mean_edge_attention=mean_edge,
            mean_corner_attention=mean_corner,
            mean_diaphragm_attention=mean_diaphragm,
            mean_left_right_balance=mean_balance,
            pass_rate=pass_rate,
            by_class=by_class_means,
            records=records,
        )

        logger.info(
            "GradCAMAudit: %d images | pass=%.1f%% | lung=%.1f%% | edge=%.1f%%",
            n, pass_rate * 100, mean_lung * 100, mean_edge * 100,
        )
        return report
