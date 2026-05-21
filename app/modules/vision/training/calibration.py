"""
Post-hoc confidence calibration via Temperature Scaling.

Temperature scaling is the simplest effective calibration method for neural
networks (Guo et al. 2017 — "On Calibration of Modern Neural Networks").

  calibrated_prob = softmax(logits / T)

A single scalar T is fit to minimize NLL on the validation set.
  T > 1 → softens probabilities (overconfident model → calibrated)
  T < 1 → sharpens probabilities (underconfident, rare)
  T = 1 → no change

ECE (Expected Calibration Error) measures confidence/accuracy alignment
across equal-width confidence bins. Perfect calibration → ECE = 0.

Usage
-----
    scaler = calibrate_model(model, val_loader, device)
    # temperature is stored as scaler.temperature (float)
    # apply at inference: probs = scaler.calibrate(raw_logits)
"""
from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ── Expected Calibration Error ────────────────────────────────────────────────


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """
    Compute ECE over n equal-width confidence bins.

    Parameters
    ----------
    probs  : (N, C) softmax probabilities (NOT logits)
    labels : (N,)   integer ground-truth class indices
    n_bins : int    number of bins (15 is standard)

    Returns float in [0, 1]. Lower is better.
    """
    confidences, predictions = probs.max(dim=1)
    correct = predictions.eq(labels).float()

    ece = 0.0
    boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
    n_total = float(len(labels))

    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        n_bin = mask.sum().item()
        if n_bin == 0:
            continue
        avg_acc  = correct[mask].mean().item()
        avg_conf = confidences[mask].mean().item()
        ece += abs(avg_conf - avg_acc) * (n_bin / n_total)

    return ece


# ── Logit collection ──────────────────────────────────────────────────────────


@torch.no_grad()
def collect_logits(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run one full pass over dataloader and collect raw logits + labels.

    Returns
    -------
    logits : (N, C) float tensor on CPU
    labels : (N,)   int tensor on CPU
    """
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        logits = model(images).cpu()
        all_logits.append(logits)
        all_labels.append(labels)

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


# ── Temperature Scaler ────────────────────────────────────────────────────────


class TemperatureScaler:
    """
    Single-parameter post-hoc calibration via temperature scaling.

    Fits a scalar T that minimizes NLL on a held-out validation set.
    No model weights are modified — calibration is always reversible.

    Usage
    -----
        scaler = TemperatureScaler()
        scaler.fit(logits, labels)
        cal_probs = scaler.calibrate(logits)
        print(f"T={scaler.temperature:.4f}  ECE={scaler.ece_after:.4f}")
    """

    def __init__(self) -> None:
        self.temperature: float = 1.0
        self.ece_before: float = float("nan")
        self.ece_after: float = float("nan")
        self.nll_before: float = float("nan")
        self.nll_after: float = float("nan")
        self._fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> "TemperatureScaler":
        """
        Optimize T on the validation set using L-BFGS.

        Parameters
        ----------
        logits : (N, C) — raw model outputs (pre-softmax)
        labels : (N,)   — integer ground-truth labels
        lr     : float  — L-BFGS step size
        max_iter : int  — max optimization iterations

        Returns self for chaining.
        """
        logits = logits.detach().float()
        labels = labels.detach().long()

        criterion = nn.CrossEntropyLoss()

        self.ece_before = expected_calibration_error(F.softmax(logits, dim=1), labels)
        self.nll_before = criterion(logits, labels).item()

        temperature = nn.Parameter(torch.ones(1) * 1.5)
        optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = criterion(logits / temperature.clamp(min=1e-3), labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        fitted_T = float(temperature.item())
        if not math.isfinite(fitted_T) or fitted_T <= 0:
            logger.warning(
                "TemperatureScaler: optimization produced invalid T=%.4f — defaulting to T=1.0",
                fitted_T,
            )
            fitted_T = 1.0

        self.temperature = fitted_T
        self._fitted = True

        cal_logits = logits / self.temperature
        self.ece_after = expected_calibration_error(F.softmax(cal_logits, dim=1), labels)
        self.nll_after = criterion(cal_logits, labels).item()

        logger.info(
            "TemperatureScaler: T=%.4f | ECE %.4f → %.4f | NLL %.4f → %.4f",
            self.temperature,
            self.ece_before, self.ece_after,
            self.nll_before, self.nll_after,
        )
        return self

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling → softmax probabilities. Shape: (..., C)."""
        return F.softmax(logits / self.temperature, dim=-1)

    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Return temperature-scaled logits (before softmax)."""
        return logits / self.temperature

    def as_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "ece_before": round(self.ece_before, 6) if math.isfinite(self.ece_before) else None,
            "ece_after": round(self.ece_after, 6) if math.isfinite(self.ece_after) else None,
            "nll_before": round(self.nll_before, 6) if math.isfinite(self.nll_before) else None,
            "nll_after": round(self.nll_after, 6) if math.isfinite(self.nll_after) else None,
        }


# ── Convenience wrapper ───────────────────────────────────────────────────────


def calibrate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> TemperatureScaler:
    """
    Collect validation logits and fit a TemperatureScaler in one call.

    Returns the fitted scaler. Access .temperature for the scalar value.
    """
    logger.info("Calibration: collecting validation logits…")
    logits, labels = collect_logits(model, val_loader, device)
    scaler = TemperatureScaler()
    scaler.fit(logits, labels)
    return scaler
