"""
HantaProject — Synthetic Vision Dataset Generator
==================================================

Generates a structurally diverse synthetic dataset for end-to-end pipeline
testing. No real patient data or proprietary images are used.

Dhash lesson applied here
--------------------------
dhash downsamples to 9×8. For uniform-noise images, all 9×8 pixels average
to approximately the same mean, producing near-zero Hamming distances across
ALL images. To generate genuinely distinct hashes, each image must have
large-scale brightness structure — i.e., different brightness levels in
different macro regions of the image — so that the coarse 9×8 thumbnail
is unique per image.

Design: each image consists of N randomly colored macro-regions layered
with fine-grain noise. The macro-regions survive downsampling and drive
distinct dhashes. The noise ensures high Laplacian variance (passes blur
quality check). A seeded RNG per image makes generation reproducible.

Defect images included:
  blurry*         → blur detection  (Laplacian var < 80)
  tiny*           → tiny-image detection (< 100px)
  dark*           → dark-image detection (mean < 20)
  bright*         → overexposure detection (mean > 235)
  low_contrast*   → contrast detection (std < 15)
  exact_dup*      → exact-duplicate detection (same SHA-256)
  near_dup*       → near-duplicate detection (Hamming ≤ 8)
  corrupt*        → corrupt-image detection (not decodable)

Usage
-----
  python scripts/generate_vision_samples.py

  python scripts/generate_vision_samples.py --good-per-class 50
  python scripts/generate_vision_samples.py --no-defects
  python scripts/generate_vision_samples.py --clear
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

RAW_DIR = _PROJECT_ROOT / "data" / "vision" / "raw"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_H, _W = 256, 256   # default good-image size


# ── Core: large-scale structured image generation ────────────────────────────


def _structured_image(
    seed: int,
    size: tuple[int, int],
    palette: list[tuple[int, int, int]],
    noise_std: int = 38,
    n_regions: int = 6,
) -> np.ndarray:
    """
    Build an image with N randomly sized/colored rectangular macro-regions,
    topped with fine-grain noise.

    The macro-regions ensure large-scale brightness variation → each image
    has a structurally unique 9×8 thumbnail → distinct dhash.
    The fine noise ensures high Laplacian variance → passes blur check.

    Parameters
    ----------
    seed : int
        Per-image seed for full reproducibility.
    size : (H, W)
        Output image spatial dimensions.
    palette : list of (R, G, B)
        Color bank to draw from.  Larger palette → more visual variety.
    noise_std : int
        Standard deviation of per-pixel Gaussian noise layer.
    n_regions : int
        Number of overlaid macro-regions.
    """
    rng  = random.Random(seed)
    nrng = np.random.default_rng(seed)

    h, w = size
    arr  = np.zeros((h, w, 3), dtype=np.float32)

    # Lay down macro-regions
    for _ in range(n_regions):
        color = palette[rng.randint(0, len(palette) - 1)]
        # Region covers a large fraction of the image so it survives downsampling
        x0 = rng.randint(0, w // 2)
        y0 = rng.randint(0, h // 2)
        x1 = rng.randint(w // 2, w)
        y1 = rng.randint(h // 2, h)
        for c in range(3):
            arr[y0:y1, x0:x1, c] = color[c]

    # Add fine-grain noise for high Laplacian variance
    noise = nrng.normal(0, noise_std, (h, w, 3))
    arr   = (arr + noise).clip(0, 255).astype(np.uint8)

    # Draw a few sharp geometric edges for additional Laplacian content
    img  = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(2, 5)):
        shape_color = tuple(palette[rng.randint(0, len(palette) - 1)])
        x0s = rng.randint(0, w - 1)
        y0s = rng.randint(0, h - 1)
        x1s = rng.randint(x0s, min(x0s + w // 2, w - 1))
        y1s = rng.randint(y0s, min(y0s + h // 2, h - 1))
        draw.rectangle([x0s, y0s, x1s, y1s], outline=shape_color, width=3)

    return np.array(img)


# ── Class-specific palettes ───────────────────────────────────────────────────

# related: earthy, warm browns/tans — rodent environments
_PALETTE_RELATED = [
    (30,  20,  10),  (80,  55,  25),  (130, 90,  40),
    (170, 125, 60),  (205, 160, 90),  (230, 195, 140),
    (100, 70,  30),  (50,  35,  15),  (150, 100, 50),
    (200, 155, 80),
]

# unrelated: vivid, diverse — generic everyday content
_PALETTE_UNRELATED = [
    (220, 50,  50),  (50,  180, 50),  (50,  50,  220),
    (220, 200, 50),  (180, 50,  180), (50,  180, 180),
    (255, 128, 0),   (128, 0,   255), (0,   200, 100),
    (200, 100, 0),   (100, 200, 200), (200, 0,   100),
]

# hard_negative: cool grays — medical imaging palette
_PALETTE_HARD_NEG = [
    (30,  35,  40),  (70,  80,  90),  (110, 120, 130),
    (150, 155, 160), (190, 195, 200), (220, 225, 230),
    (60,  65,  70),  (130, 135, 140), (170, 175, 180),
    (240, 242, 245),
]


# ── Per-class generators ──────────────────────────────────────────────────────


def make_related(index: int) -> np.ndarray:
    return _structured_image(
        seed=index * 31337 + 1,
        size=(_H, _W),
        palette=_PALETTE_RELATED,
        noise_std=38,
        n_regions=6,
    )


def make_unrelated(index: int) -> np.ndarray:
    return _structured_image(
        seed=index * 41761 + 3,
        size=(_H, _W),
        palette=_PALETTE_UNRELATED,
        noise_std=42,
        n_regions=7,
    )


def make_hard_negative(index: int) -> np.ndarray:
    arr = _structured_image(
        seed=index * 53189 + 5,
        size=(_H, _W),
        palette=_PALETTE_HARD_NEG,
        noise_std=30,
        n_regions=5,
    )
    # Overlay a synthetic "lesion" ellipse per hard-negative image
    img  = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    rng  = random.Random(index * 53189 + 5)
    cx   = rng.randint(_W // 4, 3 * _W // 4)
    cy   = rng.randint(_H // 4, 3 * _H // 4)
    r_px = rng.randint(20, 55)
    gray = rng.randint(160, 230)
    draw.ellipse(
        [cx - r_px, cy - r_px, cx + r_px, cy + r_px],
        fill=(gray, gray, gray),
        outline=(40, 40, 40),
        width=2,
    )
    return np.array(img)


# ── Defect generators ─────────────────────────────────────────────────────────


def make_blurry(base: np.ndarray, kernel: int = 61) -> np.ndarray:
    """Heavy Gaussian blur → Laplacian variance << 80."""
    return cv2.GaussianBlur(base, (kernel, kernel), 0)


def make_tiny() -> np.ndarray:
    """45×45 image → below 100px min-dimension threshold."""
    rng = np.random.default_rng(999)
    return rng.integers(50, 200, (45, 45, 3), dtype=np.uint8)


def make_dark() -> np.ndarray:
    """Mean brightness < 20."""
    rng = np.random.default_rng(1001)
    return rng.integers(0, 15, (224, 224, 3), dtype=np.uint8)


def make_overexposed() -> np.ndarray:
    """Mean brightness > 235."""
    rng = np.random.default_rng(1002)
    return rng.integers(240, 255, (224, 224, 3), dtype=np.uint8)


def make_low_contrast() -> np.ndarray:
    """Pixel std < 15 — nearly uniform surface."""
    base = 128
    rng  = np.random.default_rng(1003)
    noise = rng.integers(-4, 4, (224, 224, 3), dtype=np.int32)
    arr   = np.full((224, 224, 3), base, dtype=np.int32)
    return (arr + noise).clip(0, 255).astype(np.uint8)


def make_near_duplicate_of(base: np.ndarray) -> np.ndarray:
    """
    Create a near-duplicate by applying very light JPEG re-compression
    simulation (round-trip slight quantisation).  This keeps the dhash
    distance at 0–3 bits while making the content_hash different.
    """
    import io as _io
    buf = _io.BytesIO()
    Image.fromarray(base).save(buf, format="JPEG", quality=60)  # lossy
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def make_corrupt_bytes() -> bytes:
    """Bytes that cannot be decoded as any known image format."""
    return b"CORRUPT_NOT_AN_IMAGE\xff\xd8\x00" + b"\xab" * 50


# ── Saving helpers ────────────────────────────────────────────────────────────


def _save_jpg(arr: np.ndarray, path: Path, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format="JPEG", quality=quality)


def _save_corrupt(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(make_corrupt_bytes())


# ── Main generation ───────────────────────────────────────────────────────────


def generate(raw_dir: Path, good_per_class: int, include_defects: bool) -> dict[str, int]:
    counts: dict[str, int] = {}

    generators = {
        "related":       make_related,
        "unrelated":     make_unrelated,
        "hard_negative": make_hard_negative,
    }

    # ── Good images ───────────────────────────────────────────────────────────
    for cls_name, make_fn in generators.items():
        for i in range(good_per_class):
            arr = make_fn(index=i)
            _save_jpg(arr, raw_dir / cls_name / f"{cls_name}_{i:04d}.jpg")
        counts[cls_name] = good_per_class
        logger.info("  %-15s : %d good images", cls_name, good_per_class)

    if not include_defects:
        return counts

    # ── Defect images — all injected into related/ ────────────────────────────
    defect_dir = raw_dir / "related"

    # Blurry (3): very low Laplacian variance → blur detection
    for i in range(3):
        base = make_related(index=500 + i)
        _save_jpg(make_blurry(base), defect_dir / f"defect_blurry_{i}.jpg")

    # Tiny (3): below min-dimension threshold → tiny-image detection
    for i in range(3):
        _save_jpg(make_tiny(), defect_dir / f"defect_tiny_{i}.jpg")

    # Dark (2): mean < 20 → dark-image detection
    for i in range(2):
        _save_jpg(make_dark(), defect_dir / f"defect_dark_{i}.jpg")

    # Overexposed (2): mean > 235 → overexposure detection
    for i in range(2):
        _save_jpg(make_overexposed(), defect_dir / f"defect_bright_{i}.jpg")

    # Low contrast (2): pixel std < 15 → contrast detection
    for i in range(2):
        _save_jpg(make_low_contrast(), defect_dir / f"defect_lowcontrast_{i}.jpg")

    # Exact duplicates (1 base + 1 copy): same file bytes → exact-dup detection
    base_exact = make_related(index=700)
    _save_jpg(base_exact, defect_dir / "defect_exactdup_base.jpg")
    _save_jpg(base_exact, defect_dir / "defect_exactdup_copy.jpg")

    # Near-duplicates (1 base + 2 near-dups): JPEG re-compression → near-dup detection
    base_near = make_related(index=800)
    _save_jpg(base_near, defect_dir / "defect_neardup_base.jpg")
    _save_jpg(make_near_duplicate_of(base_near), defect_dir / "defect_neardup_1.jpg")
    _save_jpg(make_near_duplicate_of(base_near), defect_dir / "defect_neardup_2.jpg")

    # Corrupt (2): unreadable bytes → corrupt-image detection
    _save_corrupt(defect_dir / "defect_corrupt_1.jpg")
    _save_corrupt(defect_dir / "defect_corrupt_2.jpg")

    defect_total = 3 + 3 + 2 + 2 + 2 + 2 + 3 + 2
    counts["related"] += defect_total
    logger.info("  defects (related): %d injected", defect_total)

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HantaProject — Generate synthetic vision test dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--good-per-class", type=int, default=40,
                        help="Good (non-defect) images per class.")
    parser.add_argument("--no-defects", action="store_true",
                        help="Skip defect image injection.")
    parser.add_argument("--clear", action="store_true",
                        help="Delete existing images before generating.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.clear:
        for cls in ("related", "unrelated", "hard_negative"):
            d = args.raw_dir / cls
            if d.exists():
                for f in d.iterdir():
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                        f.unlink()
        logger.info("Cleared existing images from drop zone.")

    for cls in ("related", "unrelated", "hard_negative"):
        (args.raw_dir / cls).mkdir(parents=True, exist_ok=True)

    logger.info("Generating synthetic dataset → %s", args.raw_dir)
    logger.info("Good images per class : %d", args.good_per_class)
    logger.info("Defect images         : %s", not args.no_defects)

    generate(
        raw_dir=args.raw_dir,
        good_per_class=args.good_per_class,
        include_defects=not args.no_defects,
    )

    total = 0
    for cls in ("related", "unrelated", "hard_negative"):
        d = args.raw_dir / cls
        n = sum(1 for p in d.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"})
        logger.info("  %s/ → %d files", cls, n)
        total += n

    logger.info("Generation complete — %d total files", total)


if __name__ == "__main__":
    main()
