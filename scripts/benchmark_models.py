"""
Benchmark comparison: full-image model vs. segmented-lung ROI model.

Auto-discovers dataset split, class directories, and checkpoint files.
Runs BiasBenchmark + BiasSuite + GradCAMAudit on both models.

Usage (minimal)
---------------
    python scripts/benchmark_models.py \
        --old-checkpoint models/vision/v6_medical/stage_c_bilateral/best.pt \
        --new-checkpoint models/vision/v20260520_012338 \
        --dataset        data/segmented_dataset

Usage (full)
------------
    python scripts/benchmark_models.py \
        --old-checkpoint models/vision/v6_medical/stage_c_bilateral/best.pt \
        --new-checkpoint models/vision/v20260520_012338/weights.pt \
        --dataset        data/segmented_dataset \
        --split          val \
        --n-samples      40 \
        --output         results/benchmark.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.modules.vision.evaluation.bias_benchmark import (
    BiasBenchmark,
    BenchmarkReport,
    compare_reports,
)
from app.modules.vision.evaluation.gradcam_audit import GradCAMAudit
from app.modules.vision.evaluation.stress_suite import BiasSuite
from app.modules.vision.models.registry import VisionModelRegistry
from app.modules.vision.training.trainer import VisionTrainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
_SPLIT_PRIORITY = ["val", "test", "train"]


# ── Checkpoint loading ────────────────────────────────────────────────────────


def resolve_checkpoint_path(path: Path) -> Path:
    """
    Resolve a checkpoint path that may be:
      - A direct .pt file
      - A directory containing weights.pt
      - A directory containing best.pt
    """
    if path.is_file():
        return path
    if path.is_dir():
        for candidate in ("weights.pt", "best.pt", "last.pt"):
            p = path / candidate
            if p.exists():
                logger.info("Resolved checkpoint dir → %s", p)
                return p
        pts = list(path.glob("*.pt"))
        if pts:
            chosen = sorted(pts)[0]
            logger.info("Resolved checkpoint dir → %s (first .pt found)", chosen)
            return chosen
    raise FileNotFoundError(f"No checkpoint found at: {path}")


def load_checkpoint(
    path: Path,
    fallback_class_names: list[str],
    device: torch.device,
) -> tuple[torch.nn.Module, str, list[str]]:
    """
    Load a checkpoint into an eval-mode model.

    Supports three formats:
      1. Raw state dict (VisionModelStore weights.pt)
      2. Training-style dict with top-level model_state_dict
      3. v6 calibrated dict with model_state_dict + v6_meta

    Returns (model, architecture, class_names).
    """
    path = resolve_checkpoint_path(path)
    logger.info("Loading checkpoint: %s", path)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        # v6 format: metadata lives in v6_meta
        meta = ckpt.get("v6_meta", ckpt)
        architecture = meta.get("architecture", "efficientnet_b0")
        class_names = meta.get("classes") or meta.get("class_names") or fallback_class_names
        num_classes = meta.get("num_classes", len(class_names))
    else:
        # Raw OrderedDict — entire object is the state dict
        state_dict = ckpt
        architecture = "efficientnet_b0"
        class_names = fallback_class_names
        num_classes = len(class_names)

    # Try to load metadata sidecar (VisionModelStore layout)
    meta_path = path.parent / "metadata.json"
    if meta_path.exists():
        try:
            meta_json = json.loads(meta_path.read_text(encoding="utf-8"))
            if not class_names or class_names == fallback_class_names:
                class_names = meta_json.get("class_names", class_names)
            architecture = meta_json.get("architecture", architecture)
            num_classes = meta_json.get("num_classes", len(class_names))
        except Exception:
            pass

    model = VisionModelRegistry.build(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=False,
        freeze=False,
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()

    logger.info(
        "  arch=%s  classes=%s  device=%s", architecture, class_names, device
    )
    return model, architecture, list(class_names)


# ── Dataset discovery ─────────────────────────────────────────────────────────


def find_split_dir(dataset_root: Path, preferred_split: str | None) -> Path:
    """
    Return the path to the split directory inside dataset_root.

    If dataset_root itself contains class directories (no split sub-dirs),
    return it directly.
    """
    # If a preferred split was given, try it first
    if preferred_split:
        candidate = dataset_root / preferred_split
        if candidate.is_dir():
            return candidate

    # Check whether dataset_root already is a split (has class sub-dirs with images)
    subdirs = [d for d in dataset_root.iterdir() if d.is_dir()]
    has_splits = any(d.name in _SPLIT_PRIORITY for d in subdirs)

    if has_splits:
        for split in _SPLIT_PRIORITY:
            candidate = dataset_root / split
            if candidate.is_dir():
                logger.info("Auto-selected split: %s", candidate)
                return candidate

    # dataset_root is already a split directory
    return dataset_root


def discover_classes(split_dir: Path) -> list[str]:
    """Return sorted class names found as subdirectories of split_dir."""
    classes = sorted(
        d.name for d in split_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    return classes


def collect_images(
    split_dir: Path,
    class_names: list[str],
    n_samples: int,
) -> list[tuple[Image.Image, str]]:
    """
    Collect up to n_samples images spread evenly across class_names.
    Returns list of (PIL Image, class_name) pairs.
    """
    per_class = max(1, n_samples // max(len(class_names), 1))
    items: list[tuple[Image.Image, str]] = []

    for cls in class_names:
        cls_dir = split_dir / cls
        if not cls_dir.is_dir():
            logger.warning("Class dir missing — skipping: %s", cls_dir)
            continue
        paths = sorted(
            p for p in cls_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS
        )[:per_class]
        for p in paths:
            try:
                items.append((Image.open(p).convert("RGB"), cls))
            except Exception:
                logger.warning("Cannot open %s — skipping", p)

    logger.info(
        "Collected %d images from %s  [%s]",
        len(items), split_dir, ", ".join(f"{cls}:N" for cls in class_names),
    )
    return items


# ── Pretty-print table ────────────────────────────────────────────────────────


def print_table(rows: list[tuple[str, str, str]], title: str = "") -> None:
    if title:
        print(f"\n{'=' * 68}")
        print(f"  {title}")
        print(f"{'=' * 68}")
    print(f"  {'Metric':<34} {'Old (full img)':>14} {'New (ROI)':>14}")
    print(f"  {'-' * 64}")
    for metric, old_v, new_v in rows:
        print(f"  {metric:<34} {old_v:>14} {new_v:>14}")
    print(f"{'=' * 68}")


def _f(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return "—" if v is None else str(v)


# ── Evaluator wrappers ────────────────────────────────────────────────────────


def run_bias_benchmark(
    model: torch.nn.Module,
    arch: str,
    device: torch.device,
    images: list[tuple[Image.Image, str]],
    tag: str,
) -> BenchmarkReport | None:
    try:
        bench = BiasBenchmark(model, arch, device)
        report = bench.evaluate_set(images, model_tag=tag)
        report.log()
        return report
    except Exception:
        logger.exception("BiasBenchmark failed for %s", tag)
        return None


def run_stress_suite(
    model: torch.nn.Module,
    arch: str,
    device: torch.device,
    images: list[tuple[Image.Image, str]],
    class_names: list[str],
    tag: str,
    seed: int,
):
    try:
        suite = BiasSuite(model, arch, device)
        report = suite.run(images, model_tag=tag, class_names=class_names, seed=seed)
        return report
    except Exception:
        logger.exception("BiasSuite failed for %s", tag)
        return None


def run_gradcam_audit(
    model: torch.nn.Module,
    arch: str,
    device: torch.device,
    images: list[tuple[Image.Image, str]],
    class_names: list[str],
    tag: str,
    seed: int,
):
    from app.modules.vision.preprocessing.pipeline import ImagePreprocessingPipeline

    class _DS(torch.utils.data.Dataset):
        def __init__(self, items, cls_names):
            self._items = items
            self.class_names = cls_names
            self._pl = ImagePreprocessingPipeline()

        def __len__(self):
            return len(self._items)

        def __getitem__(self, idx):
            img, label = self._items[idx]
            tensor = self._pl.preprocess_for_inference(img).squeeze(0)
            label_idx = self.class_names.index(label) if label in self.class_names else 0
            return tensor, label_idx

    try:
        ds = _DS(images, class_names)
        audit = GradCAMAudit(model, arch, device)
        report = audit.run(ds, n_samples=len(images), model_tag=tag, seed=seed)
        return report
    except Exception:
        logger.exception("GradCAMAudit failed for %s", tag)
        return None


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark old vs. new vision model")
    parser.add_argument("--old-checkpoint", type=Path, required=True,
                        help=".pt file or directory containing best.pt / weights.pt")
    parser.add_argument("--new-checkpoint", type=Path, required=True,
                        help=".pt file or directory containing best.pt / weights.pt")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Dataset root (auto-discovers split dirs and class dirs)")
    parser.add_argument("--split", default=None,
                        help="Split to use: val | test | train (default: auto)")
    parser.add_argument("--n-samples", type=int, default=40,
                        help="Total images to sample across all classes (default 40)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = VisionTrainer._resolve_device(args.device)
    logger.info("Device: %s", device)

    # ── Discover dataset ──────────────────────────────────────────────────────
    split_dir = find_split_dir(args.dataset, args.split)
    all_classes = discover_classes(split_dir)
    logger.info("Discovered classes: %s", all_classes)
    print(f"\n  Dataset split : {split_dir}")
    print(f"  Classes found : {all_classes}")

    # ── Load models ───────────────────────────────────────────────────────────
    old_model, old_arch, old_classes = load_checkpoint(
        args.old_checkpoint, all_classes, device
    )
    new_model, new_arch, new_classes = load_checkpoint(
        args.new_checkpoint, all_classes, device
    )

    print(f"\n  Old model : {args.old_checkpoint}  arch={old_arch}  classes={old_classes}")
    print(f"  New model : {args.new_checkpoint}  arch={new_arch}  classes={new_classes}")

    # Use intersection of discovered classes with both models' class lists
    # (old model may have a different ordering — preserve it per model)
    eval_classes = [c for c in all_classes if c in old_classes and c in new_classes]
    if not eval_classes:
        eval_classes = all_classes
    logger.info("Eval classes: %s", eval_classes)

    # ── Collect images ────────────────────────────────────────────────────────
    images = collect_images(split_dir, eval_classes, args.n_samples)
    if not images:
        logger.error("No images found — cannot run benchmark.")
        sys.exit(1)
    logger.info("Running benchmark on %d images", len(images))

    results: dict = {
        "old_checkpoint": str(args.old_checkpoint),
        "new_checkpoint": str(args.new_checkpoint),
        "split_dir": str(split_dir),
        "n_images": len(images),
        "eval_classes": eval_classes,
        "old_class_order": old_classes,
        "new_class_order": new_classes,
    }

    # ── 1. Bias Benchmark ─────────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Running BiasBenchmark …")
    old_bias = run_bias_benchmark(old_model, old_arch, device, images, "old_full_image")
    new_bias = run_bias_benchmark(new_model, new_arch, device, images, "new_roi")

    if old_bias and new_bias:
        delta = compare_reports(old_bias, new_bias)
        results["bias_benchmark"] = {
            "old": old_bias.as_dict(),
            "new": new_bias.as_dict(),
            "delta": delta,
        }
        print_table([
            ("edge_attention_pct (%)",
             _f(old_bias.edge_attention_mean), _f(new_bias.edge_attention_mean)),
            ("lung_attention_pct (%)",
             _f(old_bias.lung_attention_mean), _f(new_bias.lung_attention_mean)),
            ("center_attention_pct (%)",
             _f(old_bias.center_attention_mean), _f(new_bias.center_attention_mean)),
            ("confidence_stability",
             _f(old_bias.confidence_stability_mean), _f(new_bias.confidence_stability_mean)),
        ], title="BiasBenchmark — Spatial Attention")
    else:
        results["bias_benchmark"] = {"error": "one or both runs failed"}

    # ── 2. Stress Suite ───────────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Running BiasSuite …")
    old_stress = run_stress_suite(old_model, old_arch, device, images, old_classes, "old_full_image", args.seed)
    new_stress = run_stress_suite(new_model, new_arch, device, images, new_classes, "new_roi", args.seed)

    if old_stress and new_stress:
        results["stress_suite"] = {
            "old": old_stress.as_dict(),
            "new": new_stress.as_dict(),
        }
        print_table([
            ("flip_rate (lower=more robust)",
             _f(old_stress.flip_rate), _f(new_stress.flip_rate)),
            ("mean_confidence_delta",
             _f(old_stress.mean_confidence_delta), _f(new_stress.mean_confidence_delta)),
        ], title="Stress Suite — Prediction Stability Under Transforms")

        # Per-transform breakdown
        all_transforms = sorted(
            set(old_stress.by_transform) | set(new_stress.by_transform)
        )
        rows = []
        for t in all_transforms:
            ov = old_stress.by_transform.get(t, {}).get("flip_rate")
            nv = new_stress.by_transform.get(t, {}).get("flip_rate")
            rows.append((f"  flip: {t}", _f(ov), _f(nv)))
        if rows:
            print_table(rows, title="Stress Suite — Per-Transform Flip Rate")
    else:
        results["stress_suite"] = {"error": "one or both runs failed"}

    # ── 3. GradCAM Audit ──────────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Running GradCAMAudit …")
    old_audit = run_gradcam_audit(old_model, old_arch, device, images, old_classes, "old_full_image", args.seed)
    new_audit = run_gradcam_audit(new_model, new_arch, device, images, new_classes, "new_roi", args.seed)

    if old_audit and new_audit:
        results["gradcam_audit"] = {
            "old": old_audit.as_dict(),
            "new": new_audit.as_dict(),
        }
        print_table([
            ("lung_attention_pct",
             _f(old_audit.mean_lung_attention), _f(new_audit.mean_lung_attention)),
            ("edge_attention_pct",
             _f(old_audit.mean_edge_attention), _f(new_audit.mean_edge_attention)),
            ("corner_attention_pct",
             _f(old_audit.mean_corner_attention), _f(new_audit.mean_corner_attention)),
            ("diaphragm_attention_pct",
             _f(old_audit.mean_diaphragm_attention), _f(new_audit.mean_diaphragm_attention)),
            ("pass_rate (lung>edge & corner<0.10)",
             _f(old_audit.pass_rate), _f(new_audit.pass_rate)),
        ], title="GradCAM Audit — Anatomical Plausibility")
    else:
        results["gradcam_audit"] = {"error": "one or both runs failed"}

    # ── Save output ───────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Benchmark saved → %s", args.output)
    print(f"\n  Results saved to: {args.output}\n")


if __name__ == "__main__":
    main()
