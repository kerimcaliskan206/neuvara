"""
Vision model trainer — two-phase transfer learning.

Phase A (freeze_epochs):   backbone frozen, only head is trained.
Phase B (remaining epochs): backbone unfrozen, differential learning rates.
  backbone_lr = unfreeze_lr
  head_lr     = unfreeze_lr * differential_lr_factor

Improvements over original prototype
-------------------------------------
  - EpochMetrics computes true macro F1 (multiclass-safe via sklearn)
  - AMP uses torch.amp (CUDA-only; safely disabled on MPS/CPU)
  - Phase B uses differential LRs (backbone ≪ head)
  - Criterion supports class_weights and optional Focal Loss
  - ModelCheckpoint embeds class_names + architecture for self-contained checkpoints
  - LatestCheckpoint keeps last 2 checkpoints for rollback safety
  - Per-epoch statistics reporting
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)
from torch.utils.data import DataLoader

from app.modules.vision.models.base import BaseVisionModel
from app.modules.vision.training.callbacks import (
    EarlyStopping,
    LatestCheckpoint,
    ModelCheckpoint,
)
from app.modules.vision.training.config import VisionTrainingConfig

logger = logging.getLogger(__name__)


# ── Per-epoch metrics accumulator ─────────────────────────────────────────────


class EpochMetrics:
    """
    Accumulates per-batch predictions and labels; computes epoch-level metrics.

    Macro F1 is computed via sklearn at the end of each epoch, making it
    correct for any number of classes (binary, 3-class, N-class).
    """

    def __init__(self) -> None:
        self.total_loss = 0.0
        self.n_batches = 0
        self._preds: list[int] = []
        self._labels: list[int] = []

    def update(
        self,
        loss: float,
        preds: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        self.total_loss += loss
        self.n_batches += 1
        self._preds.extend(preds.cpu().tolist())
        self._labels.extend(labels.cpu().tolist())

    @property
    def loss(self) -> float:
        return self.total_loss / max(self.n_batches, 1)

    @property
    def accuracy(self) -> float:
        if not self._preds:
            return 0.0
        return sum(p == l for p, l in zip(self._preds, self._labels)) / len(self._preds)

    @property
    def f1(self) -> float:
        """Macro-averaged F1 across all classes (correct for 2- and N-class)."""
        if not self._preds:
            return 0.0
        from sklearn.metrics import f1_score
        return float(
            f1_score(self._labels, self._preds, average="macro", zero_division=0)
        )

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        p = f"{prefix}_" if prefix else ""
        return {
            f"{p}loss": round(self.loss, 5),
            f"{p}accuracy": round(self.accuracy, 4),
            f"{p}f1": round(self.f1, 4),
        }


# ── Trainer ───────────────────────────────────────────────────────────────────


class VisionTrainer:
    """
    Two-phase transfer learning trainer for vision classification.

    Parameters
    ----------
    model : BaseVisionModel
        CNN model with freeze_backbone() / unfreeze_backbone() interface.
    train_loader, val_loader : DataLoader
    config : VisionTrainingConfig
    output_dir : Path
        Checkpoint and log output directory.
    device : str | torch.device
        "auto" resolves: cuda → mps → cpu.
    class_names : list[str] | None
        Human-readable class labels in index order.
        Embedded in checkpoints; required for self-contained inference.
    class_weights : Tensor | None
        Per-class loss weights (shape: [num_classes]).
        Computed from class_distribution via balancer.compute_class_weights_tensor().
    """

    def __init__(
        self,
        model: BaseVisionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: VisionTrainingConfig,
        output_dir: Path | str,
        device: str | torch.device = "auto",
        class_names: Optional[list[str]] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = self._resolve_device(device)
        self.model = model.to(self.device)
        self.class_names = class_names or []

        # AMP: GradScaler is CUDA-only; autocast works on CUDA and MPS
        self._scale_enabled = config.mixed_precision and self.device.type == "cuda"
        self._amp_enabled = config.mixed_precision and self.device.type in ("cuda", "mps")
        self._scaler = GradScaler(enabled=self._scale_enabled)

        # Criterion
        weights_on_device = (
            class_weights.to(self.device) if class_weights is not None else None
        )
        if config.use_focal_loss:
            from app.modules.vision.training.focal_loss import FocalLoss
            self.criterion: nn.Module = FocalLoss(
                gamma=config.focal_gamma,
                weight=weights_on_device,
                label_smoothing=config.label_smoothing,
            )
            logger.info(
                "Criterion: FocalLoss(gamma=%.1f, label_smoothing=%.2f, class_weights=%s)",
                config.focal_gamma, config.label_smoothing,
                weights_on_device is not None,
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=weights_on_device,
                label_smoothing=config.label_smoothing,
            )
            logger.info(
                "Criterion: CrossEntropyLoss(label_smoothing=%.2f, class_weights=%s)",
                config.label_smoothing, weights_on_device is not None,
            )

        monitor_mode = (
            "max" if any(k in config.monitor_metric for k in ("f1", "acc")) else "min"
        )

        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            mode=monitor_mode,
        )

        ckpt_dir = self.output_dir / "checkpoints"
        arch = getattr(model, "architecture", None)

        self.checkpoint = ModelCheckpoint(
            checkpoint_dir=ckpt_dir / "best",
            monitor=config.monitor_metric,
            mode=monitor_mode,
            save_best_only=config.save_best_only,
            class_names=self.class_names,
            architecture=arch,
        )
        self.latest_checkpoint = LatestCheckpoint(
            checkpoint_dir=ckpt_dir / "latest",
            keep=2,
            class_names=self.class_names,
            architecture=arch,
        )

        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._scheduler = None
        self.history: list[dict] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self) -> list[dict]:
        """Run the full training loop. Returns per-epoch history list."""
        torch.manual_seed(self.config.random_state)
        logger.info(
            "VisionTrainer: device=%s | epochs=%d | freeze_epochs=%d | "
            "amp=%s | focal=%s | weighted_sampler=%s",
            self.device, self.config.epochs, self.config.freeze_epochs,
            self._amp_enabled, self.config.use_focal_loss, self.config.use_weighted_sampler,
        )

        for epoch in range(1, self.config.epochs + 1):
            phase_changed = self._maybe_transition_phase(epoch)
            if phase_changed or epoch == 1:
                self._build_optimizer_and_scheduler(epoch)

            t0 = time.perf_counter()
            train_m = self._train_epoch()
            val_m = self._val_epoch()
            elapsed = time.perf_counter() - t0

            monitor_val = val_m.as_dict("val").get(
                f"val_{self.config.monitor_metric.replace('val_', '')}",
                val_m.f1,
            )

            self.checkpoint.step(self.model, epoch, monitor_val, self._optimizer)
            self.latest_checkpoint.step(self.model, epoch, monitor_val, self._optimizer)

            row = {
                "epoch": epoch,
                **train_m.as_dict("train"),
                **val_m.as_dict("val"),
                "lr_backbone": self._current_lr(group=0),
                "lr_head": self._current_lr(group=-1),
                "elapsed_s": round(elapsed, 2),
            }
            self.history.append(row)
            self._log_epoch(row)

            if self._scheduler is not None:
                if self.config.scheduler == "plateau":
                    self._scheduler.step(monitor_val)
                else:
                    self._scheduler.step()

            if self.early_stopping.step(monitor_val):
                logger.info("VisionTrainer: early stopping at epoch %d", epoch)
                break

        self._log_training_summary()
        return self.history

    # ── Phase management ──────────────────────────────────────────────────────

    def _maybe_transition_phase(self, epoch: int) -> bool:
        if epoch == self.config.freeze_epochs + 1:
            logger.info(
                "VisionTrainer: Phase B — unfreezing backbone "
                "(backbone_lr=%.2e  head_lr=%.2e)",
                self.config.unfreeze_lr,
                self.config.unfreeze_lr * self.config.differential_lr_factor,
            )
            self.model.unfreeze_backbone()
            return True
        return False

    # ── Training / validation loops ───────────────────────────────────────────

    def _train_epoch(self) -> EpochMetrics:
        self.model.train()
        metrics = EpochMetrics()

        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self._optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self._amp_enabled,
            ):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self._scaler.scale(loss).backward()

            if self.config.grad_clip is not None:
                self._scaler.unscale_(self._optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self._scaler.step(self._optimizer)
            self._scaler.update()

            preds = logits.detach().argmax(dim=1)
            metrics.update(loss.item(), preds.cpu(), labels.cpu())

        return metrics

    @torch.no_grad()
    def _val_epoch(self) -> EpochMetrics:
        self.model.eval()
        metrics = EpochMetrics()

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self._amp_enabled,
            ):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            preds = logits.argmax(dim=1)
            metrics.update(loss.item(), preds.cpu(), labels.cpu())

        return metrics

    # ── Optimizer / scheduler factory ─────────────────────────────────────────

    def _build_optimizer_and_scheduler(self, current_epoch: int) -> None:
        in_phase_b = current_epoch > self.config.freeze_epochs

        if in_phase_b:
            # Differential LR: backbone at unfreeze_lr, head at unfreeze_lr * factor.
            # Protects pretrained ImageNet features while still letting the head adapt.
            backbone_lr = self.config.unfreeze_lr
            head_lr = self.config.unfreeze_lr * self.config.differential_lr_factor
            param_groups = [
                {
                    "params": list(self.model.get_backbone().parameters()),
                    "lr": backbone_lr,
                },
                {
                    "params": list(self.model.get_classifier().parameters()),
                    "lr": head_lr,
                },
            ]
            default_lr = backbone_lr
        else:
            default_lr = self.config.learning_rate
            param_groups = list(
                p for p in self.model.parameters() if p.requires_grad
            )

        opt_name = self.config.optimizer.lower()
        if opt_name == "adamw":
            self._optimizer = AdamW(
                param_groups, lr=default_lr, weight_decay=self.config.weight_decay
            )
        elif opt_name == "adam":
            self._optimizer = Adam(
                param_groups, lr=default_lr, weight_decay=self.config.weight_decay
            )
        elif opt_name == "sgd":
            self._optimizer = SGD(
                param_groups, lr=default_lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        remaining = self.config.epochs - current_epoch + 1
        self._scheduler = self._build_scheduler(remaining)

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            "Optimizer rebuilt (epoch %d) | phase=%s | trainable_params=%d",
            current_epoch, "B" if in_phase_b else "A", n_trainable,
        )

    def _build_scheduler(self, n_epochs: int):
        sched_name = self.config.scheduler.lower()
        warmup = self.config.warmup_epochs

        if sched_name == "none":
            return None
        if sched_name == "plateau":
            return ReduceLROnPlateau(
                self._optimizer, mode="max", factor=self.config.gamma,
                patience=3, min_lr=1e-7,
            )
        if sched_name == "cosine":
            main = CosineAnnealingLR(
                self._optimizer, T_max=max(n_epochs - warmup, 1)
            )
        elif sched_name == "step":
            main = StepLR(
                self._optimizer,
                step_size=self.config.step_size,
                gamma=self.config.gamma,
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_name}")

        if warmup > 0:
            warmup_sched = LinearLR(
                self._optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup
            )
            return SequentialLR(
                self._optimizer, [warmup_sched, main], milestones=[warmup]
            )
        return main

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _current_lr(self, group: int = 0) -> float:
        if self._optimizer is None or not self._optimizer.param_groups:
            return 0.0
        idx = group if group >= 0 else len(self._optimizer.param_groups) + group
        idx = max(0, min(idx, len(self._optimizer.param_groups) - 1))
        return self._optimizer.param_groups[idx]["lr"]

    def _log_epoch(self, row: dict) -> None:
        lr_b = row["lr_backbone"]
        lr_h = row["lr_head"]
        lr_str = f"{lr_b:.2e}" if lr_b == lr_h else f"backbone={lr_b:.2e} head={lr_h:.2e}"
        logger.info(
            "Epoch %03d | train_loss=%.4f train_f1=%.4f | "
            "val_loss=%.4f val_f1=%.4f | lr=%s | %.1fs",
            row["epoch"],
            row["train_loss"], row["train_f1"],
            row["val_loss"], row["val_f1"],
            lr_str, row["elapsed_s"],
        )

    def _log_training_summary(self) -> None:
        if not self.history:
            return
        best_row = max(self.history, key=lambda r: r.get("val_f1", 0))
        last_row = self.history[-1]
        logger.info(
            "Training summary | epochs=%d | best_val_f1=%.4f (epoch %d) | "
            "final_val_f1=%.4f | best_%s=%.4f",
            len(self.history),
            best_row["val_f1"], best_row["epoch"],
            last_row["val_f1"],
            self.config.monitor_metric, self.checkpoint.best_score,
        )

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        if isinstance(device, torch.device):
            return device
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)
