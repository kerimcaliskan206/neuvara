"""
Bias stress suite for the segmented lung ROI classifier.

Applies synthetic transformations that previously exploited dataset shortcuts
(border artifacts, scanner patterns, text markers, etc.) and checks whether
the model's prediction is stable across all of them.

A robust segmented model should:
  - Produce consistent class predictions regardless of border content
  - Show low confidence variance across geometric transforms
  - Not flip its classification when distractors are added

Usage
-----
    from app.modules.vision.evaluation.stress_suite import BiasSuite

    suite = BiasSuite(model, architecture, device)
    report = suite.run(images, labels)
    print(report.summary())
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger(__name__)


# ── Stress transforms ─────────────────────────────────────────────────────────


def _add_black_border(img: Image.Image, border_pct: float = 0.12) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    H, W = arr.shape[:2]
    bh = int(H * border_pct)
    bw = int(W * border_pct)
    arr[:bh, :] = 0
    arr[-bh:, :] = 0
    arr[:, :bw] = 0
    arr[:, -bw:] = 0
    return Image.fromarray(arr)


def _add_bright_corners(img: Image.Image, corner_pct: float = 0.12) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    H, W = arr.shape[:2]
    ch = int(H * corner_pct)
    cw = int(W * corner_pct)
    arr[:ch, :cw] = 200
    arr[:ch, -cw:] = 200
    arr[-ch:, :cw] = 200
    arr[-ch:, -cw:] = 200
    return Image.fromarray(arr)


def _add_text_marker(img: Image.Image) -> Image.Image:
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    draw.text((5, 5), "TEST", fill=(255, 255, 255))
    draw.text((5, img.height - 20), "SAMPLE", fill=(255, 255, 255))
    return out


def _rotate(img: Image.Image, degrees: float) -> Image.Image:
    return img.rotate(degrees, expand=False, fillcolor=(128, 128, 128))


def _shift(img: Image.Image, shift_pct: float = 0.10) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    H, W = arr.shape[:2]
    dx = int(W * shift_pct)
    dy = int(H * shift_pct)
    shifted = np.full_like(arr, 128)
    shifted[dy:, dx:] = arr[: H - dy, : W - dx]
    return Image.fromarray(shifted)


def _add_noise(img: Image.Image, std: float = 30.0) -> Image.Image:
    arr = np.array(img.convert("RGB")).astype(np.float32)
    noise = np.random.normal(0, std, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _center_crop_80(img: Image.Image) -> Image.Image:
    W, H = img.size
    margin_x = int(W * 0.10)
    margin_y = int(H * 0.10)
    return img.crop((margin_x, margin_y, W - margin_x, H - margin_y)).resize(
        (W, H), Image.BILINEAR
    )


def _diaphragm_heavy(img: Image.Image, bottom_boost: float = 1.4) -> Image.Image:
    arr = np.array(img.convert("RGB")).astype(np.float32)
    H = arr.shape[0]
    y_start = int(H * 0.70)
    arr[y_start:] = np.clip(arr[y_start:] * bottom_boost, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def _unilateral_bright(img: Image.Image) -> Image.Image:
    """Brighten right lung only — tests asymmetric bias."""
    arr = np.array(img.convert("RGB")).astype(np.float32)
    W = arr.shape[1]
    mid = W // 2
    arr[:, mid:] = np.clip(arr[:, mid:] * 1.5, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def _asymmetric_crop(img: Image.Image) -> Image.Image:
    """Remove left 15% — simulates asymmetric positioning."""
    W, H = img.size
    x1 = int(W * 0.15)
    cropped = img.crop((x1, 0, W, H))
    return cropped.resize((W, H), Image.BILINEAR)


STRESS_TRANSFORMS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "rotate_+30": lambda img: _rotate(img, 30),
    "rotate_-30": lambda img: _rotate(img, -30),
    "shift": _shift,
    "black_border": _add_black_border,
    "bright_corners": _add_bright_corners,
    "text_marker": _add_text_marker,
    "noise": _add_noise,
    "center_crop_80": _center_crop_80,
    "diaphragm_heavy": _diaphragm_heavy,
    "unilateral_bright": _unilateral_bright,
    "asymmetric_crop": _asymmetric_crop,
}


# ── Result dataclasses ────────────────────────────────────────────────────────


@dataclass
class StressRecord:
    """Stress test result for one (image, transform) pair."""

    image_path: str
    true_label: str
    transform_name: str
    base_confidence: float
    stress_confidence: float
    base_class: str
    stress_class: str
    prediction_flipped: bool
    confidence_delta: float  # stress - base


@dataclass
class StressSuiteReport:
    """Aggregate stress suite results."""

    n_images: int
    n_transforms: int
    model_tag: str
    # Fraction of (image, transform) pairs where the class prediction flipped
    flip_rate: float
    # Mean absolute confidence shift across all pairs
    mean_confidence_delta: float
    # Per-transform breakdown
    by_transform: dict = field(default_factory=dict)
    # Full per-record list
    records: list[StressRecord] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Stress Suite — {self.model_tag}",
            f"  images       : {self.n_images}",
            f"  transforms   : {self.n_transforms}",
            f"  flip rate    : {self.flip_rate:.1%}  (lower is more robust)",
            f"  Δ confidence : {self.mean_confidence_delta:+.4f}  (mean signed shift)",
        ]
        if self.by_transform:
            lines.append("  by transform:")
            for t, stats in sorted(
                self.by_transform.items(), key=lambda x: -x[1]["flip_rate"]
            ):
                lines.append(
                    f"    {t:<22} flip={stats['flip_rate']:.1%}  "
                    f"Δconf={stats['mean_delta']:+.4f}"
                )
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "n_images": self.n_images,
            "n_transforms": self.n_transforms,
            "model_tag": self.model_tag,
            "flip_rate": round(self.flip_rate, 4),
            "mean_confidence_delta": round(self.mean_confidence_delta, 4),
            "by_transform": {
                t: {k: round(v, 4) for k, v in stats.items()}
                for t, stats in self.by_transform.items()
            },
        }


# ── Suite engine ──────────────────────────────────────────────────────────────


class BiasSuite:
    """
    Applies stress transforms to images and records whether the model's
    classification changes.

    A segmented model should be robust to all border-related transforms because
    the segmentation pipeline removes border content before inference. Flips on
    those transforms indicate incomplete bias removal.

    Parameters
    ----------
    model : torch.nn.Module
        Model in eval mode.
    architecture : str
        Architecture tag (for logging only).
    device : torch.device | str
    image_size : int
        Square input size (default 224).
    transforms : dict | None
        Custom transform dict. If None, uses `STRESS_TRANSFORMS`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        architecture: str,
        device: torch.device | str,
        image_size: int = 224,
        transforms: Optional[dict[str, Callable]] = None,
    ) -> None:
        self.model = model
        self.architecture = architecture
        self.device = torch.device(device)
        self.image_size = image_size
        self.transforms = transforms or STRESS_TRANSFORMS

    def run(
        self,
        images: list[tuple[Image.Image | str, str]],
        model_tag: str = "model",
        class_names: Optional[list[str]] = None,
        seed: int = 42,
    ) -> StressSuiteReport:
        """
        Run all stress transforms on each image.

        Parameters
        ----------
        images : list of (image, true_label) pairs
            `image` may be a PIL Image or a path string.
        model_tag : str
            Label for the report.
        class_names : list[str] | None
            Class index → name mapping for the model output.
        seed : int
            RNG seed (used if transforms involve randomness).
        """
        from app.modules.vision.preprocessing.pipeline import ImagePreprocessingPipeline
        from app.modules.vision.segmentation import LungSegmentationPipeline

        pipeline = ImagePreprocessingPipeline()
        seg_pipeline = LungSegmentationPipeline(padding_frac=0.07, save_debug=False)
        np.random.seed(seed)

        records: list[StressRecord] = []

        for img_item, true_label in images:
            img_path = str(img_item) if not isinstance(img_item, Image.Image) else "pil_image"
            try:
                if isinstance(img_item, Image.Image):
                    base_img = img_item.convert("RGB")
                else:
                    base_img = Image.open(img_item).convert("RGB")
            except Exception:
                logger.warning("BiasSuite: could not open %s, skipping", img_path)
                continue

            # Baseline: segmented inference
            try:
                roi_base, _ = seg_pipeline.process(base_img)
                base_conf, base_cls = self._infer(roi_base, pipeline, class_names)
            except Exception:
                logger.exception("BiasSuite: baseline inference failed for %s", img_path)
                continue

            for transform_name, transform_fn in self.transforms.items():
                try:
                    stressed_img = transform_fn(base_img)
                    roi_stress, _ = seg_pipeline.process(stressed_img)
                    stress_conf, stress_cls = self._infer(roi_stress, pipeline, class_names)

                    records.append(StressRecord(
                        image_path=img_path,
                        true_label=true_label,
                        transform_name=transform_name,
                        base_confidence=base_conf,
                        stress_confidence=stress_conf,
                        base_class=base_cls,
                        stress_class=stress_cls,
                        prediction_flipped=base_cls != stress_cls,
                        confidence_delta=stress_conf - base_conf,
                    ))
                except Exception:
                    logger.exception(
                        "BiasSuite: transform '%s' failed on %s", transform_name, img_path
                    )

        return self._build_report(records, model_tag)

    # ── Inference ─────────────────────────────────────────────────────────────

    def _infer(
        self,
        img: Image.Image,
        pipeline,
        class_names: Optional[list[str]],
    ) -> tuple[float, str]:
        tensor = pipeline.preprocess_for_inference(img).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)
        idx = int(probs.argmax().item())
        conf = float(probs[idx].item())
        label = (
            class_names[idx] if class_names and idx < len(class_names) else str(idx)
        )
        return conf, label

    # ── Report construction ───────────────────────────────────────────────────

    def _build_report(
        self, records: list[StressRecord], model_tag: str
    ) -> StressSuiteReport:
        n_images = len({r.image_path for r in records})
        n_transforms = len(self.transforms)
        n = len(records)

        if n == 0:
            logger.warning("BiasSuite: no valid records produced")
            return StressSuiteReport(
                n_images=0,
                n_transforms=n_transforms,
                model_tag=model_tag,
                flip_rate=0.0,
                mean_confidence_delta=0.0,
            )

        flip_rate = sum(r.prediction_flipped for r in records) / n
        mean_delta = sum(r.confidence_delta for r in records) / n

        by_transform: dict[str, dict] = {}
        for r in records:
            t = r.transform_name
            if t not in by_transform:
                by_transform[t] = {"flips": [], "deltas": []}
            by_transform[t]["flips"].append(float(r.prediction_flipped))
            by_transform[t]["deltas"].append(r.confidence_delta)

        by_transform_means = {
            t: {
                "flip_rate": sum(v["flips"]) / len(v["flips"]),
                "mean_delta": sum(v["deltas"]) / len(v["deltas"]),
                "n": len(v["flips"]),
            }
            for t, v in by_transform.items()
        }

        report = StressSuiteReport(
            n_images=n_images,
            n_transforms=n_transforms,
            model_tag=model_tag,
            flip_rate=flip_rate,
            mean_confidence_delta=mean_delta,
            by_transform=by_transform_means,
            records=records,
        )

        logger.info(
            "BiasSuite: %d images × %d transforms | flip=%.1f%% | Δconf=%+.4f",
            n_images, n_transforms, flip_rate * 100, mean_delta,
        )
        return report
