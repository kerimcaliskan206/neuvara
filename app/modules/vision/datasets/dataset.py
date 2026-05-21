"""
Image classification dataset supporting two usage modes:

  Binary gate (related vs unrelated+hard_negative)
  -------------------------------------------------
  CLASSES = ["unrelated", "related"]
  Use when training the content-relevance gate that filters off-domain images
  before the main classifier runs.

  Three-class (related / unrelated / hard_negative)
  -------------------------------------------------
  CLASSES_3 = ["unrelated", "related", "hard_negative"]
  Use when training a classifier that must also distinguish medically similar
  but hantavirus-unrelated imagery from true positive cases.

Expected directory layout (ImageFolder convention):

    data/vision/datasets/<version>/splits/<split>/
    ├── related/
    ├── unrelated/
    └── hard_negative/     ← omit this dir for binary gate training

Class names are configurable — pass ``classes`` to override the defaults.
Supported image extensions: .jpg, .jpeg, .png, .bmp, .tiff, .tif
"""
import logging
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ── Class name constants ───────────────────────────────────────────────────────

# Binary gate: related vs everything else
CLASSES: list[str] = ["unrelated", "related"]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS: dict[int, str] = {i: c for i, c in enumerate(CLASSES)}

# Three-class: for training the main classifier (after gate passes)
CLASSES_3: list[str] = ["unrelated", "related", "hard_negative"]
CLASS_TO_IDX_3: dict[str, int] = {c: i for i, c in enumerate(CLASSES_3)}
IDX_TO_CLASS_3: dict[int, str] = {i: c for i, c in enumerate(CLASSES_3)}


class ImageFolderDataset(Dataset):
    """
    Generic labeled image dataset for binary classification.

    Parameters
    ----------
    root_dir : Path
        Root containing split sub-directories (train/, val/, test/).
    split : str
        One of "train", "val", "test".
    transform : callable, optional
        torchvision transform pipeline applied to each image.
    classes : list[str], optional
        Class names in label-index order.  Defaults to ["unrelated", "related"].
    """

    def __init__(
        self,
        root_dir: Path | str,
        split: str = "train",
        transform=None,
        classes: list[str] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.classes: list[str] = classes if classes is not None else CLASSES
        self.class_to_idx: dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class: dict[int, str] = {i: c for i, c in enumerate(self.classes)}

        self.split_dir = self.root_dir / split
        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Dataset split directory not found: {self.split_dir}\n"
                f"Expected structure: {root_dir}/{split}/{{class_name}}/image.jpg"
            )

        self.samples: list[tuple[Path, int]] = self._discover_samples()

        if not self.samples:
            logger.warning(
                "No images found in '%s'. "
                "Populate %s/%s with class subdirectories containing images.",
                self.split_dir, split, self.split_dir,
            )
        else:
            logger.info(
                "ImageFolderDataset [%s]: %d images | %s",
                split, len(self.samples), self._distribution_str(),
            )

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    # ── Class distribution ────────────────────────────────────────────────────

    def class_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            counts[self.idx_to_class[label]] += 1
        return counts

    def is_balanced(self, tolerance: float = 0.2) -> bool:
        dist = self.class_distribution()
        total = sum(dist.values())
        if total == 0:
            return True
        minority = min(dist.values())
        majority = max(dist.values())
        return (minority / majority) >= (1 - tolerance)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _discover_samples(self) -> list[tuple[Path, int]]:
        samples = []
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                logger.warning("Class directory missing: %s", class_dir)
                continue
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() in _IMAGE_EXTENSIONS:
                    samples.append((path, class_idx))
        return samples

    def _distribution_str(self) -> str:
        dist = self.class_distribution()
        total = len(self.samples)
        return " | ".join(
            f"{cls}={n} ({100 * n / total:.0f}%)"
            for cls, n in dist.items()
        )
