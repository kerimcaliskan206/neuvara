"""
Training callbacks: early stopping and checkpointing.

ModelCheckpoint  — saves best-metric checkpoint with full metadata.
LatestCheckpoint — saves every epoch, keeps last K (rollback-safe).
EarlyStopping    — halts training when a metric stops improving.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ── Early Stopping ────────────────────────────────────────────────────────────


@dataclass
class EarlyStopping:
    """
    Stops training when a monitored metric stops improving.

    Parameters
    ----------
    patience : int
        Epochs to wait after last improvement before stopping.
    min_delta : float
        Minimum change that counts as an improvement.
    mode : str
        "max" for F1/accuracy, "min" for loss.
    """

    patience: int = 7
    min_delta: float = 0.001
    mode: str = "max"

    best_score: float = field(init=False, default=None)
    counter: int = field(init=False, default=0)
    should_stop: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        self.best_score = float("-inf") if self.mode == "max" else float("inf")

    def step(self, score: float) -> bool:
        """Update state with the latest score. Returns True if training should stop."""
        improved = (
            score > self.best_score + self.min_delta
            if self.mode == "max"
            else score < self.best_score - self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            logger.debug(
                "EarlyStopping: no improvement %d/%d (best=%.4f now=%.4f)",
                self.counter, self.patience, self.best_score, score,
            )
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "EarlyStopping: triggered after %d epochs without improvement.",
                    self.patience,
                )

        return self.should_stop

    def reset(self) -> None:
        self.counter = 0
        self.should_stop = False
        self.best_score = float("-inf") if self.mode == "max" else float("inf")


# ── Best-metric checkpoint ────────────────────────────────────────────────────


class ModelCheckpoint:
    """
    Saves the model whenever the monitored metric improves.

    The checkpoint dict contains enough metadata to reconstruct the model
    without the original training script (class_names, architecture, etc.).

    Parameters
    ----------
    checkpoint_dir : Path
        Directory for checkpoint files.
    monitor : str
        Name of the metric being watched (used for logging and filenames).
    mode : str
        "max" or "min".
    save_best_only : bool
        If True, delete the previous best when a new best is found.
    class_names : list[str] | None
        Class labels in index order — embedded in the checkpoint for
        inference without the training script.
    architecture : str | None
        Architecture name — embedded for registry lookup at load time.
    """

    def __init__(
        self,
        checkpoint_dir: Path | str,
        monitor: str = "val_f1",
        mode: str = "max",
        save_best_only: bool = True,
        filename: str = "best_{score:.4f}_epoch{epoch:03d}.pt",
        class_names: Optional[list[str]] = None,
        architecture: Optional[str] = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.filename = filename
        self.class_names = class_names or []
        self.architecture = architecture

        self._best_score: float = float("-inf") if mode == "max" else float("inf")
        self._best_path: Optional[Path] = None

    def step(
        self,
        model: torch.nn.Module,
        epoch: int,
        score: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> bool:
        """
        Save a checkpoint if the score improved.

        Parameters
        ----------
        optimizer : optional
            When provided, the optimizer state is embedded for full resumption.

        Returns True if a new checkpoint was saved.
        """
        improved = (
            score > self._best_score if self.mode == "max" else score < self._best_score
        )

        if not improved and self.save_best_only:
            return False

        self._best_score = score
        name = self.filename.format(epoch=epoch, score=score)
        save_path = self.checkpoint_dir / name

        ckpt: dict = {
            "epoch": epoch,
            "score": score,
            "monitor": self.monitor,
            "model_state_dict": model.state_dict(),
            "class_names": self.class_names,
            "architecture": self.architecture,
            "num_classes": len(self.class_names),
        }
        if optimizer is not None:
            ckpt["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(ckpt, save_path)

        if self.save_best_only and self._best_path is not None and self._best_path != save_path:
            try:
                self._best_path.unlink(missing_ok=True)
            except OSError:
                pass

        self._best_path = save_path
        logger.info(
            "ModelCheckpoint [best]: %s | %s=%.4f",
            save_path.name, self.monitor, score,
        )
        return True

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def best_checkpoint_path(self) -> Optional[Path]:
        return self._best_path


# ── Latest checkpoint (rollback-safe) ────────────────────────────────────────


class LatestCheckpoint:
    """
    Saves a checkpoint every epoch, keeping only the last K files.

    This provides rollback safety: if training diverges after the best
    checkpoint epoch, you can resume from the most recent saved state
    rather than only the historical best.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory for checkpoint files (separate from ModelCheckpoint's dir).
    keep : int
        How many latest checkpoints to retain (default: 2).
    class_names, architecture : embedded for self-contained checkpoints.
    """

    def __init__(
        self,
        checkpoint_dir: Path | str,
        keep: int = 2,
        filename: str = "latest_epoch{epoch:03d}.pt",
        class_names: Optional[list[str]] = None,
        architecture: Optional[str] = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep = max(keep, 1)
        self.filename = filename
        self.class_names = class_names or []
        self.architecture = architecture
        self._saved: list[Path] = []

    def step(
        self,
        model: torch.nn.Module,
        epoch: int,
        score: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Path:
        """
        Save a checkpoint for this epoch and prune old ones.

        Returns the path of the saved file.
        """
        name = self.filename.format(epoch=epoch)
        save_path = self.checkpoint_dir / name

        ckpt: dict = {
            "epoch": epoch,
            "score": score,
            "model_state_dict": model.state_dict(),
            "class_names": self.class_names,
            "architecture": self.architecture,
            "num_classes": len(self.class_names),
        }
        if optimizer is not None:
            ckpt["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(ckpt, save_path)
        self._saved.append(save_path)

        # Prune oldest files beyond keep limit
        while len(self._saved) > self.keep:
            old = self._saved.pop(0)
            try:
                old.unlink(missing_ok=True)
            except OSError:
                pass

        logger.debug("LatestCheckpoint: saved %s", save_path.name)
        return save_path

    @property
    def latest_path(self) -> Optional[Path]:
        return self._saved[-1] if self._saved else None
