"""
Grad-CAM (Gradient-weighted Class Activation Mapping).

Produces a coarse spatial heatmap highlighting which regions of an input
image most influenced the model's predicted class.

Algorithm
---------
1. Open a *per-call* capture session: register a forward hook on the target
   conv layer that
     (a) saves the layer's output activations, and
     (b) attaches a tensor-level backward hook on that very output via
         `output.register_hook(...)` — far more reliable than
         `register_full_backward_hook` for `nn.Sequential` subclasses such
         as torchvision's `Conv2dNormActivation`.
2. Forward → logits.
3. Backprop target class score.
4. The tensor-level hook captures dY/dOutput.
5. GAP-pool the gradients → per-channel weights; weighted sum of activations
   → raw CAM; ReLU + normalize → [0, 1].

Reference: Selvaraju et al. 2017, https://arxiv.org/abs/1610.02391

Usage
-----
    with build_gradcam(model, architecture="efficientnet_b0") as cam:
        diag = cam.validate(input_tensor)        # diagnostic dict
        heatmap = cam.generate(input_tensor)     # (H, W) float in [0, 1]
        focused = cam.generate_pulmonary_focused(input_tensor)
"""
from __future__ import annotations

import logging
import threading
import weakref
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── Per-model serialization ───────────────────────────────────────────────────
# Two concurrent GradCAM calls on the SAME model would race on shared
# parameter .grad accumulators during backward(). We serialize per-model.
_model_locks: "weakref.WeakKeyDictionary[nn.Module, threading.Lock]" = weakref.WeakKeyDictionary()
_locks_guard = threading.Lock()


# ── Phase 27 — Light Gaussian smoothing for upsampled CAM ─────────────────────
# Suppresses speckle and the blocky artefacts from upsampling a tiny
# feature-map CAM (e.g. 7×7) to full image resolution. Keeps the macro
# attention pattern intact — sigma is deliberately small.

def _smooth_cam(cam: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    """Apply a small isotropic Gaussian blur to a 2D CAM tensor."""
    if sigma <= 0:
        return cam
    # 5×5 kernel covers ±2σ for σ=1.5; the cam is on CPU at this point.
    radius = 2
    coords = torch.arange(-radius, radius + 1, dtype=cam.dtype, device=cam.device)
    g1d = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g1d = g1d / g1d.sum()
    kernel = (g1d[:, None] * g1d[None, :]).unsqueeze(0).unsqueeze(0)
    blurred = F.conv2d(
        cam.unsqueeze(0).unsqueeze(0),
        kernel,
        padding=radius,
    ).squeeze(0).squeeze(0)
    return blurred


def _lock_for(model: nn.Module) -> threading.Lock:
    with _locks_guard:
        lk = _model_locks.get(model)
        if lk is None:
            lk = threading.Lock()
            _model_locks[model] = lk
    return lk


# ── Helpers ───────────────────────────────────────────────────────────────────

def _inference_mode_enabled() -> bool:
    """torch.is_inference_mode_enabled() doesn't exist on older PyTorch."""
    fn = getattr(torch, "is_inference_mode_enabled", None)
    try:
        return bool(fn()) if fn else False
    except Exception:
        return False


class GradCAM:
    """
    Grad-CAM explainability for any CNN model.

    Parameters
    ----------
    model : nn.Module
        The trained model. Must be in eval mode for inference.
    target_layer : nn.Module
        The convolutional layer to visualize (typically the last conv block).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer

        # Populated by generate_pulmonary_focused() / validate() / generate().
        self.last_telemetry: dict = {}
        self.last_diagnosis: dict = {}

        # Populated by generate_pulmonary_focused(): smooth pulmonary prior +
        # disjoint anatomical sub-masks for visualization and downstream
        # score-attenuation logic. All (H, W) float CPU tensors.
        self.last_prior: Optional[torch.Tensor] = None
        self.last_lung_mask: Optional[torch.Tensor] = None
        self.last_zone_masks: dict[str, torch.Tensor] = {}

        logger.debug(
            "GradCAM: initialized | model=%s target_layer=%s",
            type(self.model).__name__, type(self.target_layer).__name__,
        )
        # NB: We intentionally do NOT register persistent hooks here.
        # Persistent module hooks have two problems on a shared model:
        #   (1) `register_full_backward_hook` is known to fire inconsistently
        #       on `nn.Sequential` subclasses such as torchvision's
        #       Conv2dNormActivation, depending on PyTorch version.
        #   (2) Persistent hooks fire for *every* forward on the layer,
        #       including concurrent predict() calls from other request
        #       threads, polluting the capture.
        # Instead, each generate()/validate() call opens its own short-lived
        # capture session — see _open_capture_session().

    # ── Compat shims (legacy callers still call these) ────────────────────────

    def remove_hooks(self) -> None:
        """No-op kept for legacy callers. Hooks are now per-call."""
        return

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _clear_param_grads(self) -> None:
        """
        Safer per-parameter alternative to ``model.zero_grad(set_to_none=True)``.

        On MPS we have observed an intermittent
        ``RuntimeError: tensor does not have a device`` raised from inside
        ``zero_grad`` in certain autograd states. Manually setting
        ``.grad = None`` only on parameters that carry a grad avoids that path
        entirely while achieving the same semantic result.
        """
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad = None

    def _open_capture_session(self) -> tuple[dict, Callable[[], None]]:
        """
        Register a short-lived forward hook on ``self.target_layer`` that
        captures the output activations AND attaches a tensor-level backward
        hook on the very output tensor produced by that forward.

        Returns
        -------
        (capture, close_fn)
            capture is a dict updated by the hooks:
                - "acts": (1, C, h, w) detached activation tensor (or None)
                - "grads": (1, C, h, w) detached gradient tensor (or None)
                - "fwd_count": # of times the forward hook fired *on any thread*
                - "bwd_count": # of times the tensor-level grad hook fired
                - "output_requires_grad": whether the forward output was
                  connected to autograd (False ⇒ no backward chain)
            close_fn unregisters the forward hook.
        """
        # We only capture for THIS thread — predict() in another request
        # thread also fires this hook but we ignore those captures.
        my_tid = threading.get_ident()
        capture: dict = {
            "acts": None,
            "grads": None,
            "fwd_count": 0,
            "bwd_count": 0,
            "output_requires_grad": None,
            "owner_tid": my_tid,
        }

        def fwd_hook(_module, _input, output):
            capture["fwd_count"] += 1
            if threading.get_ident() != my_tid:
                # Another thread's forward (e.g. service.predict()). Ignore.
                return
            capture["acts"] = output.detach()
            capture["output_requires_grad"] = bool(output.requires_grad)
            if output.requires_grad:
                # Tensor-level grad hook — far more reliable than the
                # module-level full_backward_hook for our target layer.
                def grad_hook(grad: torch.Tensor) -> None:
                    capture["bwd_count"] += 1
                    capture["grads"] = grad.detach()
                output.register_hook(grad_hook)

        handle = self.target_layer.register_forward_hook(fwd_hook)

        def close() -> None:
            try:
                handle.remove()
            except Exception:
                logger.debug("GradCAM: forward-hook removal failed", exc_info=True)

        return capture, close

    def _materialise_leaf_input(
        self, input_tensor: torch.Tensor, model_device: torch.device
    ) -> torch.Tensor:
        """
        Build a fresh leaf tensor on the model device with grad enabled.

        .detach()    — sever any prior autograd history
        .to(device)  — co-locate with model parameters
        .clone()     — guarantee a new leaf
        .contiguous()— MPS prefers contiguous stride layouts
        """
        t = (
            input_tensor.detach()
            .to(device=model_device, non_blocking=False)
            .clone()
            .contiguous()
        )
        t.requires_grad_(True)
        return t

    # ── Diagnostic / validation ───────────────────────────────────────────────

    def validate(self, input_tensor: torch.Tensor) -> dict:
        """
        Run a forward+backward and return a rich diagnostic dict.

        Unlike before, this does NOT raise on hook silent failure — it returns
        ``ok=False`` so callers can inspect *why* (forward fired? backward fired?
        output had grad_fn? device mismatch?).

        Raises ValueError only on shape mismatch of the input.
        """
        if input_tensor.dim() != 4 or input_tensor.size(0) != 1:
            raise ValueError(
                f"validate() requires shape (1, C, H, W), got {tuple(input_tensor.shape)}"
            )

        with _lock_for(self.model):
            was_training = self.model.training
            self.model.eval()
            try:
                try:
                    model_device = next(self.model.parameters()).device
                except StopIteration:
                    model_device = input_tensor.device

                self._clear_param_grads()
                capture, close = self._open_capture_session()
                logits = None
                forward_ok = False
                backward_ok = False
                err: Optional[str] = None
                try:
                    t = self._materialise_leaf_input(input_tensor, model_device)
                    try:
                        with torch.enable_grad():
                            logits = self.model(t)
                            forward_ok = True
                            score = logits[0, int(logits.argmax(dim=1).item())]
                            score.backward()
                            backward_ok = True
                    except Exception as e:
                        err = repr(e)
                finally:
                    close()

                acts = capture["acts"]
                grads = capture["grads"]
                gnorm = float(grads.abs().mean()) if grads is not None else None
                amax = float(acts.abs().max()) if acts is not None else None

                result: dict = {
                    "ok":                       (acts is not None and grads is not None),
                    "target_layer":             type(self.target_layer).__name__,
                    "model_device":             str(model_device),
                    "input_shape":              tuple(input_tensor.shape),
                    "input_device":             str(input_tensor.device),
                    "input_dtype":              str(input_tensor.dtype),
                    "leaf_device":              None if logits is None else str(model_device),
                    "grad_enabled":             bool(torch.is_grad_enabled()),
                    "inference_mode_outer":     _inference_mode_enabled(),
                    "forward_ok":               forward_ok,
                    "backward_ok":              backward_ok,
                    "logits_requires_grad":     None if logits is None else bool(logits.requires_grad),
                    "logits_has_grad_fn":       None if logits is None else (logits.grad_fn is not None),
                    "forward_hook_fires":       capture["fwd_count"],
                    "backward_hook_fires":      capture["bwd_count"],
                    "activation_output_req_grad": capture["output_requires_grad"],
                    "activations_captured":     acts is not None,
                    "gradients_captured":       grads is not None,
                    "activations_shape":        None if acts  is None else tuple(acts.shape),
                    "activations_device":       None if acts  is None else str(acts.device),
                    "activations_max_abs":      amax,
                    "gradients_shape":          None if grads is None else tuple(grads.shape),
                    "gradients_device":         None if grads is None else str(grads.device),
                    "gradients_mean_abs":       gnorm,
                    "error":                    err,
                }
                self.last_diagnosis = result
                logger.info("GradCAM.validate: %s", result)
                return result
            finally:
                self._clear_param_grads()
                if was_training:
                    self.model.train()

    # ── Core algorithm ────────────────────────────────────────────────────────

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute the Grad-CAM heatmap for a single image.

        Parameters
        ----------
        input_tensor : (1, C, H, W) — any device. Will be moved to model device.
        class_idx : int | None
            Target class index. None → uses the predicted class.

        Returns
        -------
        (H, W) float tensor on CPU in [0, 1]. Upsampled to input spatial size.
        """
        if input_tensor.dim() != 4 or input_tensor.size(0) != 1:
            raise ValueError(
                "input_tensor must have shape (1, C, H, W). "
                f"Got: {tuple(input_tensor.shape)}"
            )

        with _lock_for(self.model):
            was_training = self.model.training
            self.model.eval()
            try:
                try:
                    model_device = next(self.model.parameters()).device
                except StopIteration:
                    model_device = input_tensor.device

                self._clear_param_grads()

                # Open a fresh capture session for THIS call. Hooks live only
                # for the duration of the forward+backward below.
                capture, close = self._open_capture_session()
                try:
                    t = self._materialise_leaf_input(input_tensor, model_device)

                    logger.debug(
                        "GradCAM.generate | device=%s t.shape=%s t.req_grad=%s "
                        "t.is_leaf=%s grad_enabled=%s inference_mode=%s",
                        model_device, tuple(t.shape), t.requires_grad, t.is_leaf,
                        torch.is_grad_enabled(), _inference_mode_enabled(),
                    )

                    # torch.enable_grad() overrides any outer torch.no_grad();
                    # it cannot override torch.inference_mode() — but that
                    # would have already broken our prior request, and the
                    # diagnostic in the raise-path below will surface it.
                    with torch.enable_grad():
                        logits = self.model(t)
                        if class_idx is None:
                            class_idx = int(logits.argmax(dim=1).item())
                        score = logits[0, class_idx]
                        score.backward()
                finally:
                    close()

                acts = capture["acts"]
                grads = capture["grads"]

                if acts is None or grads is None:
                    # Build the richest possible error to make this debuggable
                    # without re-running validate().
                    diag = {
                        "fwd_fired":             capture["fwd_count"],
                        "bwd_fired":             capture["bwd_count"],
                        "acts_captured":         acts  is not None,
                        "grads_captured":        grads is not None,
                        "output_required_grad":  capture["output_requires_grad"],
                        "target_layer":          type(self.target_layer).__name__,
                        "logits_requires_grad":  bool(logits.requires_grad),
                        "logits_has_grad_fn":    logits.grad_fn is not None,
                        "model_device":          str(model_device),
                        "grad_enabled":          bool(torch.is_grad_enabled()),
                        "inference_mode_outer":  _inference_mode_enabled(),
                    }
                    self.last_diagnosis = diag
                    raise RuntimeError(
                        "GradCAM: hooks did not capture activations or gradients. "
                        f"Diagnosis: {diag}"
                    )

                logger.debug(
                    "GradCAM | fwd_fired=%d bwd_fired=%d acts=%s@%s grads=%s@%s "
                    "grad_mean_abs=%.6g",
                    capture["fwd_count"], capture["bwd_count"],
                    tuple(acts.shape), acts.device,
                    tuple(grads.shape), grads.device,
                    float(grads.abs().mean()),
                )

                weights = grads.mean(dim=(2, 3), keepdim=True)
                cam = (weights * acts).sum(dim=1, keepdim=True)
                cam = F.relu(cam)

                h, w = input_tensor.shape[2], input_tensor.shape[3]
                cam = F.interpolate(
                    cam, size=(h, w), mode="bilinear", align_corners=False
                )
                cam = cam.squeeze().detach().cpu()

                # Phase 27: light Gaussian smoothing AFTER upsample stabilises
                # the heatmap on small inputs (upsampling a 4×4 feature map to
                # 128×128 produces blocky checkerboard artefacts; a small
                # blur removes them without changing the macro pattern).
                # Slightly larger sigma when the upsample ratio is large
                # (i.e. the input is small relative to the feature map).
                upsample_ratio = max(h, w) / max(1, max(cam.shape[-2:]))
                sigma = 1.5 if min(h, w) >= 256 else 2.0
                cam = _smooth_cam(cam, sigma=sigma)

                # Track the pre-normalization peak — used downstream by
                # generate_pulmonary_focused()'s adaptive trust gain so that
                # a genuinely weak CAM (healthy lung / fake image) stays calm
                # instead of being stretched to [0, 1] by min-max.
                raw_cam_max = float(cam.max().item())
                self.last_telemetry["raw_cam_max"] = round(raw_cam_max, 6)
                self.last_telemetry["upsample_ratio"] = round(float(upsample_ratio), 3)
                self.last_telemetry["smoothing_sigma"] = sigma

                cmin, cmax = cam.min(), cam.max()
                if (cmax - cmin) > 1e-8:
                    cam = (cam - cmin) / (cmax - cmin)
                else:
                    logger.warning(
                        "GradCAM.generate: uniform activation map for class %d — "
                        "heatmap will be all zeros.",
                        class_idx,
                    )
                    cam = torch.zeros_like(cam)

                return cam
            finally:
                self._clear_param_grads()
                if was_training:
                    self.model.train()

    def generate_for_classes(
        self,
        input_tensor: torch.Tensor,
        class_indices: Optional[list[int]] = None,
    ) -> dict[int, torch.Tensor]:
        """Generate Grad-CAM heatmaps for multiple output classes."""
        with torch.no_grad():
            try:
                model_device = next(self.model.parameters()).device
                logits = self.model(input_tensor.to(model_device))
            except StopIteration:
                logits = self.model(input_tensor)
        n_classes = logits.shape[1]
        indices = class_indices if class_indices is not None else list(range(n_classes))
        return {i: self.generate(input_tensor, class_idx=i) for i in indices}

    # ── Pulmonary spatial prior ───────────────────────────────────────────────

    @staticmethod
    def _build_pulmonary_prior(
        H: int,
        W: int,
        *,
        top_frac: float = 0.12,
        bottom_start_frac: float = 0.78,
        bottom_keep: float = 0.25,
        edge_frac: float = 0.08,
        corner_radius_frac: float = 0.18,
        corner_floor: float = 0.05,
        center_sigma_y: float = 0.32,
        center_sigma_x: float = 0.30,
        gauss_blend: float = 0.45,
    ) -> torch.Tensor:
        """Smooth spatial prior over the lung-field region (CPU, in [0, 1])."""
        ys = torch.linspace(0.0, 1.0, H)
        xs = torch.linspace(0.0, 1.0, W)
        yy = ys.unsqueeze(1).expand(H, W)
        xx = xs.unsqueeze(0).expand(H, W)

        top_taper = torch.clamp((yy - top_frac) / 0.10, 0.0, 1.0)

        bot_span = max(1e-3, 1.0 - bottom_start_frac)
        bot_ramp = torch.clamp((yy - bottom_start_frac) / bot_span, 0.0, 1.0)
        bot_taper = 1.0 - (1.0 - bottom_keep) * bot_ramp

        left_edge  = torch.clamp(xx / max(1e-3, edge_frac), 0.0, 1.0)
        right_edge = torch.clamp((1.0 - xx) / max(1e-3, edge_frac), 0.0, 1.0)

        dist_tl = torch.sqrt(yy ** 2 + xx ** 2)
        dist_tr = torch.sqrt(yy ** 2 + (1.0 - xx) ** 2)
        dist_bl = torch.sqrt((1.0 - yy) ** 2 + xx ** 2)
        dist_br = torch.sqrt((1.0 - yy) ** 2 + (1.0 - xx) ** 2)
        nearest_corner = torch.stack([dist_tl, dist_tr, dist_bl, dist_br]).min(dim=0).values
        corner_factor = torch.clamp(
            nearest_corner / max(1e-3, corner_radius_frac),
            corner_floor, 1.0,
        )

        cy, cx = 0.50, 0.50
        gauss = torch.exp(
            -(((yy - cy) ** 2) / (2.0 * center_sigma_y ** 2)
              + ((xx - cx) ** 2) / (2.0 * center_sigma_x ** 2))
        )

        base = top_taper * bot_taper * left_edge * right_edge * corner_factor
        prior = base * ((1.0 - gauss_blend) + gauss_blend * gauss)
        return torch.clamp(prior, 0.0, 1.0)

    # ── Stage-2: anatomical sub-mask construction ─────────────────────────────

    @staticmethod
    def _build_lung_zone_masks(
        H: int,
        W: int,
        prior: torch.Tensor,
        *,
        lung_threshold: float = 0.45,
        peripheral_x_frac: float = 0.28,
        diaphragm_y_start: float = 0.65,
        lateral_band_frac: float = 0.18,
    ) -> dict[str, torch.Tensor]:
        """
        Derive disjoint anatomical sub-masks from the smooth pulmonary prior.

        Zones (disjoint by construction, union = ``lung_mask``):
          central     — inner pulmonary core: inside the lung field, away
                        from both the lateral chest wall and the diaphragm.
                        The region we WANT attention to land in.
          peripheral  — lateral lung shell: inside the lung field but within
                        ``peripheral_x_frac`` of either lateral image edge,
                        excluding the diaphragm band. Shortcut-prone region
                        (chest-wall hardware, ECG leads, soft-tissue
                        gradients).
          diaphragm   — lower lung band: inside the lung field with
                        ``y > diaphragm_y_start``. Preserved at full weight
                        — real lower-lobe pneumonia lives here.

        Helpers:
          lung_mask        — union of the three.
          lateral_falloff  — 0 at lateral image edge, 1 once past
                             ``lateral_band_frac`` from either side. Multiply
                             with ``lung_mask * (1 - diaphragm)`` to obtain
                             the soft lateral-penalty target.
        """
        ys = torch.linspace(0.0, 1.0, H).unsqueeze(1).expand(H, W)
        xs = torch.linspace(0.0, 1.0, W).unsqueeze(0).expand(H, W)

        lung_mask = (prior > lung_threshold).to(torch.float32)

        diaphragm = lung_mask * (ys > diaphragm_y_start).to(torch.float32)

        near_lateral = (
            (xs < peripheral_x_frac).to(torch.float32)
            + (xs > (1.0 - peripheral_x_frac)).to(torch.float32)
        ).clamp(0.0, 1.0)

        peripheral = lung_mask * (1.0 - diaphragm) * near_lateral
        central    = lung_mask * (1.0 - diaphragm) * (1.0 - near_lateral)

        # Soft ramp from lateral image edge: 0 at x=0 or x=1, rising to 1.0 at
        # ``lateral_band_frac`` from either side. Used to construct a graded
        # penalty rather than a hard binary cutoff.
        lateral_dist = torch.minimum(xs, 1.0 - xs)
        lateral_falloff = torch.clamp(
            lateral_dist / max(1e-3, lateral_band_frac), 0.0, 1.0
        )

        return {
            "lung_mask":       lung_mask,
            "central":         central,
            "peripheral":      peripheral,
            "diaphragm":       diaphragm,
            "lateral_falloff": lateral_falloff,
        }

    def generate_pulmonary_focused(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        suppress_top_frac: float = 0.12,
        suppress_bottom_frac: float = 0.22,
        suppress_edge_frac: float = 0.08,
        lung_boost_zone: tuple = (0.28, 0.78),   # accepted for back-compat (no-op)
        lung_boost_factor: float = 1.35,         # accepted for back-compat (no-op)
        *,
        lateral_penalty: float = 0.10,
        peripheral_x_frac: float = 0.28,
        diaphragm_y_start: float = 0.65,
        lateral_band_frac: float = 0.25,
        apply_stage2_debias: bool = True,
        lung_mask_tensor: Optional[torch.Tensor] = None,
        seg_quality: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Lung-region focused Grad-CAM with a smooth spatial prior plus a
        second-stage anatomical debiasing pass.

        Stage 1 — multiply by ``prior`` (suppresses borders/corners/diaphragm
        shoulders). Stage 2 — apply a *soft* lateral-lung-edge penalty inside
        the lung field, scaled by ``lateral_falloff`` and explicitly skipping
        the diaphragm band so lower-lobe pneumonia sensitivity is preserved.

        Parameters
        ----------
        lateral_penalty : float
            Multiplier applied at the lateral lung edge (clamped to [0.05, 1.0]).
            Default 0.4 ⇒ lateral activations retain 40% of their value at
            x=0/x=1, ramping back to 1.0 over ``lateral_band_frac`` of width.
        apply_stage2_debias : bool
            Set False to fall back to the Stage-1-only behavior (debug only).
        lung_mask_tensor : (H, W) float tensor | None
            Real segmentation mask from LungSegmenter (0 = background, 1 = lung).
            When provided, Stage-3 zeroes all activations outside this boundary
            before normalization. May be in original-image coordinates — it is
            resized automatically to match the heatmap spatial size.
        seg_quality : str | None
            Segmentation quality tag ("good" | "fallback" | "single_lung").
            "fallback" triggers extra smoothing + strength reduction (Stage-3b).

        Telemetry (stored on ``self.last_telemetry``):
          lung_pct, edge_pct, corner_pct, center_pct            — Stage-1 keys
          central_lung_pct, peripheral_lung_pct, diaphragm_pct  — Stage-2 zone split
          edge_dominant                                          — peripheral > central + diaphragm
          lateral_penalty_applied                                — the value used
          stage2_attenuation_ratio                               — 1 - (total_v2 / total_v1)
          --- Stage-3 anatomical constraint keys ---
          masked_activation_pct, outside_lung_pct               — hard masking metrics
          attention_penalty, anatomical_violation_score          — edge penalty scalars
          edge_penalty_applied, border_activation_detected       — edge penalty flags
          fallback_penalty_applied                               — seg-quality flag
          lung_focus_score, border_focus_score                   — post-constraint focus
          anatomical_alignment_score                             — central-window fraction
          center_mass_x, center_mass_y                          — activation centroid
          lung_overlap_after_penalty, non_lung_energy_ratio      — lung containment
          bilateral_balance_score, central_bias_score            — spatial distribution
        """
        raw = self.generate(input_tensor, class_idx=class_idx)
        H, W = raw.shape

        prior = self._build_pulmonary_prior(
            H, W,
            top_frac=suppress_top_frac,
            bottom_start_frac=1.0 - suppress_bottom_frac,
            edge_frac=suppress_edge_frac,
        )
        focused_v1 = raw * prior

        zones = self._build_lung_zone_masks(
            H, W, prior,
            peripheral_x_frac=peripheral_x_frac,
            diaphragm_y_start=diaphragm_y_start,
            lateral_band_frac=lateral_band_frac,
        )

        lp = float(max(0.05, min(1.0, lateral_penalty)))
        if apply_stage2_debias:
            # Penalty active only inside lungs AND outside diaphragm band.
            # ``lateral_falloff`` is 0 at the lateral edge and 1 once past
            # ``lateral_band_frac``, so the penalty smoothly disappears toward
            # the center.
            penalty_target = (
                zones["lung_mask"] * (1.0 - zones["diaphragm"])
                * (1.0 - zones["lateral_falloff"])
            )
            penalty_map = 1.0 - (1.0 - lp) * penalty_target
            focused_raw = focused_v1 * penalty_map
        else:
            focused_raw = focused_v1

        # ── Telemetry ─────────────────────────────────────────────────────────
        total_v1 = float(focused_v1.sum().item()) + 1e-8
        total = float(focused_raw.sum().item()) + 1e-8
        raw_total = float(raw.sum().item()) + 1e-8

        ch = max(1, int(H * 0.12))
        cw = max(1, int(W * 0.12))
        corner_sum = float(
            focused_raw[:ch, :cw].sum()
            + focused_raw[:ch, W - cw:].sum()
            + focused_raw[H - ch:, :cw].sum()
            + focused_raw[H - ch:, W - cw:].sum()
        )

        edge_mask = torch.zeros_like(focused_raw)
        edge_mask[:int(H * 0.12), :] = 1.0
        edge_mask[int(H * 0.88):, :] = 1.0
        edge_mask[:, :int(W * 0.08)] = 1.0
        edge_mask[:, int(W * 0.92):] = 1.0
        edge_sum = float((focused_raw * edge_mask).sum())

        lung_mask = zones["lung_mask"]
        lung_sum = float((focused_raw * lung_mask).sum())

        central_sum    = float((focused_raw * zones["central"]).sum())
        peripheral_sum = float((focused_raw * zones["peripheral"]).sum())
        diaphragm_sum  = float((focused_raw * zones["diaphragm"]).sum())

        # Pre-Stage-2 zone shares — the distribution BEFORE the lateral
        # penalty was applied. Used for edge-dominant detection so the
        # score-attenuation policy can see the original shortcut behavior
        # even after Stage-2 has cleaned up the visible heatmap.
        central_sum_v1    = float((focused_v1 * zones["central"]).sum())
        peripheral_sum_v1 = float((focused_v1 * zones["peripheral"]).sum())
        diaphragm_sum_v1  = float((focused_v1 * zones["diaphragm"]).sum())
        central_pct_v1    = 100.0 * central_sum_v1    / total_v1
        peripheral_pct_v1 = 100.0 * peripheral_sum_v1 / total_v1
        diaphragm_pct_v1  = 100.0 * diaphragm_sum_v1  / total_v1

        cy0, cy1 = int(H * 0.30), int(H * 0.70)
        cx0, cx1 = int(W * 0.25), int(W * 0.75)
        center_sum = float(focused_raw[cy0:cy1, cx0:cx1].sum())

        max_idx = int(focused_raw.argmax().item())
        max_y, max_x = max_idx // W, max_idx % W

        suppression_ratio = max(0.0, 1.0 - (total / raw_total))
        suppression_fired = suppression_ratio > 0.05
        stage2_atten = max(0.0, 1.0 - (total / total_v1)) if apply_stage2_debias else 0.0

        central_pct    = 100.0 * central_sum    / total
        peripheral_pct = 100.0 * peripheral_sum / total
        diaphragm_pct  = 100.0 * diaphragm_sum  / total

        # Edge-dominant detection on the PRE-Stage-2 distribution. Zones are
        # roughly equal-area (~33% each); uniform activation produces
        # peripheral_pct ≈ central_pct ≈ 33%. A 20% lead + absolute floor at
        # 25% catches lateral shortcut focus without false-positive-ing on
        # uniform maps. Computed on _v1 so score-attenuation sees the
        # original behavior even after the heatmap has been debiased.
        edge_dominant = (
            peripheral_pct_v1 > 25.0
            and peripheral_pct_v1 > 1.2 * max(1e-6, central_pct_v1)
        )

        self.last_prior = prior.detach().cpu()
        self.last_lung_mask = lung_mask.detach().cpu()
        self.last_zone_masks = {
            k: v.detach().cpu() for k, v in zones.items()
        }

        self.last_telemetry = {
            "class_idx":                 int(class_idx) if class_idx is not None else None,
            "shape":                     (H, W),
            "max_y_frac":                round(max_y / max(1, H - 1), 3),
            "max_x_frac":                round(max_x / max(1, W - 1), 3),
            "lung_pct":                  round(100.0 * lung_sum / total, 2),
            "edge_pct":                  round(100.0 * edge_sum / total, 2),
            "corner_pct":                round(100.0 * corner_sum / total, 2),
            "center_pct":                round(100.0 * center_sum / total, 2),
            # Post-Stage-2 zone distribution (what the visible heatmap shows)
            "central_lung_pct":          round(central_pct, 2),
            "peripheral_lung_pct":       round(peripheral_pct, 2),
            "diaphragm_pct":             round(diaphragm_pct, 2),
            # Pre-Stage-2 zone distribution (used for shortcut detection)
            "central_lung_pct_pre":      round(central_pct_v1, 2),
            "peripheral_lung_pct_pre":   round(peripheral_pct_v1, 2),
            "diaphragm_pct_pre":         round(diaphragm_pct_v1, 2),
            "edge_dominant":             bool(edge_dominant),
            "lateral_penalty_applied":   lp if apply_stage2_debias else 1.0,
            "stage2_attenuation_ratio":  round(stage2_atten, 3),
            "suppression_fired":         bool(suppression_fired),
            "suppression_ratio":         round(suppression_ratio, 3),
        }
        logger.info("GradCAM telemetry: %s", self.last_telemetry)

        # ── Stage-3: Anatomical constraints ───────────────────────────────────
        # Applied AFTER Stage-1/2 telemetry so existing metrics still reflect
        # the model's raw attention. Constraints are applied before final
        # normalization so the returned heatmap is anatomically constrained.
        from app.modules.vision.explainability.anatomical_constraint import (
            apply_edge_corner_penalty,
            apply_fallback_penalty,
            apply_lung_mask,
            build_gaussian_center_prior,
            compute_bilateral_balance_score,
            compute_central_bias_score,
        )

        constrained = focused_raw
        stage3_tel: dict = {}

        # 3a. Fallback quality penalty — reduces sharpness when seg is poor
        if seg_quality is not None:
            constrained, fallback_applied = apply_fallback_penalty(
                constrained, quality=seg_quality
            )
            stage3_tel["fallback_penalty_applied"] = bool(fallback_applied)
        else:
            stage3_tel["fallback_penalty_applied"] = False

        # 3b. Hard lung ROI masking — zero activations outside segmented lungs
        if lung_mask_tensor is not None:
            constrained, mask_tel = apply_lung_mask(constrained, lung_mask_tensor)
        else:
            # Use synthetic prior as fallback mask for metric computation only
            synth_mask = (prior > 0.35).float()
            _, mask_tel = apply_lung_mask(constrained, synth_mask)
        stage3_tel.update(mask_tel)

        # 3c. Edge / corner penalty — uses Stage-1/2 violation flags
        constrained, edge_tel = apply_edge_corner_penalty(
            constrained,
            max_x_frac=self.last_telemetry["max_x_frac"],
            max_y_frac=self.last_telemetry["max_y_frac"],
            edge_pct=self.last_telemetry["edge_pct"],
            edge_dominant=bool(edge_dominant),
        )
        stage3_tel.update(edge_tel)

        # 3d. Gaussian center prior — soft amplification of central lung field
        gauss_prior = build_gaussian_center_prior(H, W)
        gauss_blend = 0.28  # 28% center boost; 72% original distribution kept
        constrained = constrained * (1.0 - gauss_blend + gauss_blend * gauss_prior)

        # ── Stage-3 telemetry ─────────────────────────────────────────────────
        total_c = float(constrained.sum().item()) + 1e-8

        ys_c = torch.linspace(0.0, 1.0, H).unsqueeze(1).expand(H, W)
        xs_c = torch.linspace(0.0, 1.0, W).unsqueeze(0).expand(H, W)
        center_mass_y = float((constrained * ys_c).sum().item()) / total_c
        center_mass_x = float((constrained * xs_c).sum().item()) / total_c

        eff_mask = lung_mask_tensor if lung_mask_tensor is not None else (prior > 0.35).float()
        if eff_mask.shape != constrained.shape:
            eff_mask = F.interpolate(
                eff_mask.unsqueeze(0).unsqueeze(0).float(),
                size=constrained.shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0).clamp(0.0, 1.0)

        lung_overlap_after = float((constrained * eff_mask).sum().item()) / total_c
        non_lung_energy    = 1.0 - lung_overlap_after

        bh = max(1, int(H * 0.10))
        bw = max(1, int(W * 0.10))
        border_mask_c = torch.zeros_like(constrained)
        border_mask_c[:bh, :]    = 1.0
        border_mask_c[H - bh:, :] = 1.0
        border_mask_c[:, :bw]    = 1.0
        border_mask_c[:, W - bw:] = 1.0
        border_focus_score = float((constrained * border_mask_c).sum().item()) / total_c

        cy0c, cy1c = int(H * 0.25), int(H * 0.75)
        cx0c, cx1c = int(W * 0.20), int(W * 0.80)
        anatomical_alignment_score = float(
            constrained[cy0c:cy1c, cx0c:cx1c].sum().item()
        ) / total_c

        stage3_tel.update({
            "center_mass_x":             round(center_mass_x, 3),
            "center_mass_y":             round(center_mass_y, 3),
            "lung_focus_score":          round(lung_overlap_after, 3),
            "border_focus_score":        round(border_focus_score, 3),
            "anatomical_alignment_score": round(anatomical_alignment_score, 3),
            "lung_overlap_after_penalty": round(lung_overlap_after, 3),
            "non_lung_energy_ratio":     round(non_lung_energy, 3),
            "bilateral_balance_score":   compute_bilateral_balance_score(
                constrained,
                eff_mask if lung_mask_tensor is not None else None,
            ),
            "central_bias_score":        compute_central_bias_score(constrained),
        })

        self.last_telemetry.update(stage3_tel)
        logger.debug("GradCAM Stage-3 telemetry: %s", stage3_tel)

        # ── Normalize after Stage-3 ───────────────────────────────────────────
        f_min, f_max = constrained.min(), constrained.max()
        if (f_max - f_min) > 1e-8:
            stage3_out = (constrained - f_min) / (f_max - f_min)
        else:
            stage3_out = torch.zeros_like(constrained)

        # ── Stage-4: Opacity-sensitive pulmonary attention shaping ────────────
        from app.modules.vision.explainability.opacity_shaping import apply_opacity_shaping

        shaped, shape_tel = apply_opacity_shaping(
            stage3_out,
            input_tensor,
            eff_mask,
        )
        self.last_telemetry.update(shape_tel)
        logger.debug("GradCAM Stage-4 telemetry: %s", shape_tel)

        # ── Final normalization ───────────────────────────────────────────────
        s_min, s_max = shaped.min(), shaped.max()
        if (s_max - s_min) > 1e-8:
            focused = (shaped - s_min) / (s_max - s_min)
        else:
            focused = torch.zeros_like(shaped)

        # ── Phase 27 — Anatomical sanity metrics + adaptive trust gain ───────
        # Compute the four debug metrics on the FINAL heatmap and derive a
        # multiplicative trust gain in [0, 1] that dampens unreliable CAMs.
        # The goal is calm/healthy → low intensity, anatomically-localised
        # pneumonia → full intensity, fake / border-heavy → attenuated.
        from app.modules.vision.explainability.anatomical_constraint import (
            compute_lung_overlap_score,
            compute_border_activation_ratio,
            compute_cam_entropy,
        )

        focused_cpu = focused.detach().cpu()
        lung_overlap_score = compute_lung_overlap_score(
            focused_cpu, eff_mask.detach().cpu() if lung_mask_tensor is not None else None,
        )
        border_activation_ratio = compute_border_activation_ratio(focused_cpu)
        cam_entropy = compute_cam_entropy(focused_cpu)
        # central_bias_score is already computed in Stage-3 telemetry above —
        # surface it under the requested canonical name too.
        center_bias_score = float(self.last_telemetry.get("central_bias_score", 0.0))

        raw_cam_max = float(self.last_telemetry.get("raw_cam_max", 1.0))

        # Trust gain components, each in [0, 1]:
        #   strength_gain — was the raw CAM weak to begin with? (healthy lungs
        #     produce a near-zero raw CAM; min-max would lie about that).
        #     A typical pneumonic raw_cam_max is >0.5; healthy is often <0.15.
        strength_gain = float(min(1.0, max(0.0, (raw_cam_max - 0.05) / 0.45)))
        #   localisation_gain — is most of the energy inside the lungs?
        #     Below ~0.4 the CAM is leaning on extra-pulmonary shortcuts.
        localisation_gain = float(min(1.0, max(0.0, (lung_overlap_score - 0.30) / 0.50)))
        #   border_penalty — borders should hold very little energy.
        #     Above ~0.25 the CAM is dominated by frame/annotation artefacts.
        border_penalty = float(max(0.0, 1.0 - max(0.0, (border_activation_ratio - 0.10) / 0.25)))

        cam_trust_gain = round(strength_gain * localisation_gain * border_penalty, 4)
        # Floor at 0.15 so a heatmap is never entirely zeroed — even a weak
        # signal can be useful as a sanity check for the demo.
        cam_trust_gain_effective = max(0.15, cam_trust_gain)

        focused_cpu = focused_cpu * cam_trust_gain_effective

        self.last_telemetry.update({
            "lung_overlap_score":      lung_overlap_score,
            "border_activation_ratio": border_activation_ratio,
            "center_bias_score":       round(center_bias_score, 4),
            "cam_entropy":             cam_entropy,
            "cam_trust_gain":          cam_trust_gain,
            "cam_trust_gain_effective": round(cam_trust_gain_effective, 4),
            "strength_gain":           round(strength_gain, 4),
            "localisation_gain":       round(localisation_gain, 4),
            "border_penalty":          round(border_penalty, 4),
        })
        logger.debug(
            "GradCAM Phase-27 sanity: lung_overlap=%.2f border=%.2f entropy=%.2f "
            "raw_cam_max=%.3f → strength=%.2f loc=%.2f border_pen=%.2f → "
            "trust_gain=%.2f",
            lung_overlap_score, border_activation_ratio, cam_entropy,
            raw_cam_max, strength_gain, localisation_gain, border_penalty,
            cam_trust_gain_effective,
        )

        return focused_cpu

    # ── Stage-2: score-attenuation policy ─────────────────────────────────────

    def compute_score_attenuation(
        self,
        classifier_confidence: float,
        *,
        high_confidence_threshold: float = 0.70,
        min_attenuation: float = 0.82,
        peripheral_dominance_threshold: float = 40.0,
    ) -> dict:
        """
        Derive a final-score attenuation multiplier from the most recent
        ``generate_pulmonary_focused`` telemetry.

        Policy
        ------
        - No telemetry yet → 1.0 (no-op).
        - Peripheral attention is not dominant AND not near-dominant → 1.0.
        - Peripheral-dominant + classifier confidence above the high-confidence
          threshold → attenuate. The higher the confidence and the larger the
          peripheral share, the harder the attenuation. Clamped to
          ``min_attenuation`` so genuine lateral pneumonia is never zeroed.

        Returns
        -------
        dict with ``attenuation`` (float in [min_attenuation, 1.0]),
        ``reason`` (str), and the diagnostic inputs used.
        """
        tm = self.last_telemetry
        if not tm:
            return {
                "attenuation": 1.0,
                "reason":      "no telemetry available — call generate_pulmonary_focused() first",
            }

        # Use the PRE-Stage-2 distribution: that's what reveals the model's
        # original attention behavior. The post-Stage-2 distribution is the
        # visible heatmap, already cleaned up by the lateral penalty.
        peripheral = float(tm.get("peripheral_lung_pct_pre",
                                  tm.get("peripheral_lung_pct", 0.0)))
        central    = float(tm.get("central_lung_pct_pre",
                                  tm.get("central_lung_pct", 0.0)))
        diaphragm  = float(tm.get("diaphragm_pct_pre",
                                  tm.get("diaphragm_pct", 0.0)))
        edge_dom   = bool(tm.get("edge_dominant", False))

        in_lung_legitimate = central + diaphragm
        is_peripheral_dominant = (
            edge_dom or peripheral >= peripheral_dominance_threshold
        )

        if not is_peripheral_dominant:
            return {
                "attenuation":      1.0,
                "reason":           "peripheral attention not dominant",
                "peripheral_pct":   peripheral,
                "central_pct":      central,
                "diaphragm_pct":    diaphragm,
                "confidence":       classifier_confidence,
            }

        if classifier_confidence < high_confidence_threshold:
            # Peripheral-dominant but low confidence — the score is already
            # weak; further attenuation risks suppressing weak true positives.
            return {
                "attenuation":      1.0,
                "reason":           "peripheral-dominant but classifier confidence below threshold",
                "peripheral_pct":   peripheral,
                "central_pct":      central,
                "diaphragm_pct":    diaphragm,
                "confidence":       classifier_confidence,
            }

        # Peripheral share of in-lung attention (ignore image-edge garbage).
        denom = max(1.0, peripheral + in_lung_legitimate)
        peripheral_share = peripheral / denom

        # Confidence margin above threshold, normalized to [0, 1].
        conf_margin = (classifier_confidence - high_confidence_threshold) / (
            1.0 - high_confidence_threshold + 1e-8
        )
        conf_margin = max(0.0, min(1.0, conf_margin))

        # Stronger penalty when both peripheral_share and conf_margin are high.
        atten = 1.0 - 0.6 * peripheral_share * (0.5 + 0.5 * conf_margin)
        atten = max(min_attenuation, min(1.0, atten))

        return {
            "attenuation":      round(atten, 4),
            "reason":           "peripheral-dominant attention with high classifier confidence",
            "peripheral_pct":   peripheral,
            "central_pct":      central,
            "diaphragm_pct":    diaphragm,
            "peripheral_share": round(peripheral_share, 3),
            "confidence":       classifier_confidence,
            "conf_margin":      round(conf_margin, 3),
        }

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_) -> None:
        # No persistent hooks to remove — kept for API compatibility.
        return None


# ── Target-layer selection ───────────────────────────────────────────────────

def _find_efficientnet_target(model: nn.Module) -> nn.Module:
    """
    Pick the final spatial feature block on a torchvision EfficientNet.

    Preference order (most-deep-then-most-conv-heavy first):
      1. model._backbone.features[-1]   — typical Conv2dNormActivation head
      2. model._backbone.features[-2]   — fallback if [-1] is degenerate
    """
    backbone = getattr(model, "_backbone", model)
    features = getattr(backbone, "features", None)
    if features is None:
        raise ValueError(
            "EfficientNet wrapper has no `_backbone.features`. "
            "Pass target_layer explicitly to GradCAM()."
        )
    return features[-1]


def build_gradcam(model: nn.Module, architecture: str) -> GradCAM:
    """
    Factory: pick the correct target layer for known architectures.

    ResNets       → model._backbone.layer4 (last residual block)
    EfficientNets → model._backbone.features[-1] (last Conv2dNormActivation)
    """
    arch = architecture.lower()

    if arch.startswith("resnet"):
        target = model._backbone.layer4
    elif arch.startswith("efficientnet"):
        target = _find_efficientnet_target(model)
    else:
        raise ValueError(
            f"No default target layer for architecture '{architecture}'. "
            "Pass target_layer explicitly to GradCAM()."
        )

    logger.debug(
        "GradCAM: target_layer=%s for %s",
        type(target).__name__, architecture,
    )
    return GradCAM(model=model, target_layer=target)


# ── Folder-level validation utility ──────────────────────────────────────────

def validate_pulmonary_focus(
    model: nn.Module,
    architecture: str,
    image_folder: "str | Path",
    preprocess_fn: Callable,
    device: "torch.device | str",
    *,
    max_images: int = 50,
    class_idx: Optional[int] = None,
) -> dict:
    """
    Aggregate pulmonary-focus statistics across a folder of X-rays.

    Useful for confirming on a batch of *healthy* images that the heatmap
    is no longer parking on borders / diaphragm / corners.
    """
    from PIL import Image  # local import — heavy and not always available

    folder = Path(image_folder)
    if not folder.is_dir():
        return {"n_images": 0, "error": f"Not a directory: {folder}"}

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )[:max_images]
    if not paths:
        return {"n_images": 0, "error": f"No images found in {folder}"}

    keys = [
        "corner_pct", "edge_pct", "center_pct", "lung_pct",
        "central_lung_pct", "peripheral_lung_pct", "diaphragm_pct",
        "stage2_attenuation_ratio",
        "max_y_frac", "max_x_frac", "suppression_ratio",
    ]
    sums = {k: 0.0 for k in keys}
    suppression_count = 0
    edge_dominant_count = 0
    n_ok = 0
    per_image: list[dict] = []
    failures: list[dict] = []

    for path in paths:
        try:
            img = Image.open(path).convert("RGB")
            t = preprocess_fn(img)
            if t.dim() == 3:
                t = t.unsqueeze(0)
            t = t.to(device)
            with build_gradcam(model, architecture) as cam:
                cam.generate_pulmonary_focused(t, class_idx=class_idx)
                tm = dict(cam.last_telemetry)
            for k in keys:
                sums[k] += float(tm.get(k, 0.0))
            if tm.get("suppression_fired"):
                suppression_count += 1
            if tm.get("edge_dominant"):
                edge_dominant_count += 1
            per_image.append({"path": str(path), **tm})
            n_ok += 1
        except Exception as e:
            logger.warning(
                "validate_pulmonary_focus: %s failed: %s", path, e, exc_info=True
            )
            failures.append({"path": str(path), "error": str(e)})

    if n_ok == 0:
        return {"n_images": 0, "error": "All images failed", "failures": failures}

    return {
        "n_images":                  n_ok,
        "n_failed":                  len(failures),
        "avg_corner_pct":            round(sums["corner_pct"] / n_ok, 2),
        "avg_edge_pct":              round(sums["edge_pct"] / n_ok, 2),
        "avg_center_pct":            round(sums["center_pct"] / n_ok, 2),
        "avg_lung_overlap_pct":      round(sums["lung_pct"] / n_ok, 2),
        "avg_central_lung_pct":      round(sums["central_lung_pct"] / n_ok, 2),
        "avg_peripheral_lung_pct":   round(sums["peripheral_lung_pct"] / n_ok, 2),
        "avg_diaphragm_pct":         round(sums["diaphragm_pct"] / n_ok, 2),
        "avg_stage2_attenuation":    round(sums["stage2_attenuation_ratio"] / n_ok, 3),
        "edge_dominant_frac":        round(edge_dominant_count / n_ok, 3),
        "avg_max_y_frac":            round(sums["max_y_frac"] / n_ok, 3),
        "avg_max_x_frac":            round(sums["max_x_frac"] / n_ok, 3),
        "avg_suppression_ratio":     round(sums["suppression_ratio"] / n_ok, 3),
        "suppression_fired_frac":    round(suppression_count / n_ok, 3),
        "per_image":                 per_image,
        "failures":                  failures,
    }
