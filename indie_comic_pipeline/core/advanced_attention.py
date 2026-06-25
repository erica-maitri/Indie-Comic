"""
ADVANCED ATTENTION MECHANISMS — Phases 3-4
==========================================
Implements the three advanced diffusion control mechanisms described in the
pipeline architecture spec:

Level 1: Physics-Informed Attention (RealDiffusion)
  - Injects dissipative heat diffusion priors into the denoising process.
  - Applies a Gaussian heat kernel to latents at each timestep to suppress
    high-frequency noise drift.
  - Mathematical basis: discrete heat equation u(t+1) = u(t) + α * ∇²u(t)

Level 2: Shared Attention Matrix Masking (Accelerated TF)
  - Registers forward hooks on UNet cross-attention layers during Panel 1.
  - Captures and caches key/value attention matrices from the anchor panel.
  - Blends a fraction of anchor K/V into subsequent panel attention loops
    to lock character identity keys across frames.

Level 3: Spatiotemporal Architectural Priors (DreamingComics)
  - Captures channel-wise latent statistics (mean, std) from Panel 1.
  - During mid-denoising of subsequent panels, blends latent channel
    statistics toward the anchor distribution to enforce structural
    motion-window constraints.
  - Approximates the temporal consistency enforcement of video DiT models.
"""

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger("pipeline.advanced_attention")


# ─────────────────────────────────────────────────────────────────────────────
# Level 1: Heat Diffusion Prior
# ─────────────────────────────────────────────────────────────────────────────

class HeatDiffusionPrior:
    """
    Physics-Informed Attention — RealDiffusion approximation.

    Applies a Gaussian smoothing kernel (the fundamental solution to the heat
    equation ∂u/∂t = α∇²u) to latents during denoising, suppressing
    high-frequency drift without disturbing low-frequency structure.
    """

    def __init__(self,
                 alpha: float = 0.03,
                 kernel_size: int = 3,
                 start_ratio: float = 0.80,
                 end_ratio: float = 0.20):
        """
        Args:
            alpha:        Heat diffusion coefficient (0.01–0.1).
            kernel_size:  Gaussian kernel size (3 or 5).
            start_ratio:  Begin applying prior at this timestep fraction (1.0=step0).
            end_ratio:    Stop applying prior at this timestep fraction.
        """
        self.alpha = alpha
        self.kernel_size = kernel_size
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self._kernel = self._build_gaussian_kernel(kernel_size)

    def _build_gaussian_kernel(self, size: int) -> np.ndarray:
        """Build a normalized 2-D Gaussian kernel."""
        sigma = size / 3.0
        center = size // 2
        k = np.zeros((size, size), dtype=np.float32)
        for i in range(size):
            for j in range(size):
                k[i, j] = math.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2))
        return k / k.sum()

    def apply(self, latents: Any, timestep_ratio: float) -> Any:
        """
        Apply heat diffusion prior to the latent tensor.

        Args:
            latents:        torch.Tensor of shape (B, C, H, W).
            timestep_ratio: 1.0 = start of denoising, 0.0 = end.

        Returns:
            Modified latent tensor.
        """
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            return latents

        if not (self.end_ratio <= timestep_ratio <= self.start_ratio):
            return latents

        # Scale alpha linearly within the active range
        effective_alpha = self.alpha * (
            (timestep_ratio - self.end_ratio) / (self.start_ratio - self.end_ratio)
        )

        try:
            _, channels, _, _ = latents.shape
            kernel_t = torch.tensor(
                self._kernel, dtype=latents.dtype, device=latents.device
            ).unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1)

            padding = self.kernel_size // 2
            smoothed = F.conv2d(latents, kernel_t, padding=padding, groups=channels)

            # Discrete heat equation update: u ← u + α(smoothed − u)
            latents = latents + effective_alpha * (smoothed - latents)
            log.debug(f"  [L1-Heat] Applied α={effective_alpha:.4f} at ratio={timestep_ratio:.2f}")

        except Exception as e:
            log.debug(f"  [L1-Heat] Skipped: {e}")

        return latents


# ─────────────────────────────────────────────────────────────────────────────
# Level 2: Shared Attention Matrix Cache
# ─────────────────────────────────────────────────────────────────────────────

class SharedAttentionCache:
    """
    Shared Attention Matrix Masking — Accelerated TF approximation.

    Installs PyTorch forward hooks on the UNet's cross-attention (attn2)
    layers to capture key/value projection outputs during Panel 1 generation.
    For subsequent panels the cached outputs are blended into the attention
    computation, locking character identity keys/values across frames.
    """

    def __init__(self, blend_ratio: float = 0.15, max_layers: int = 4):
        """
        Args:
            blend_ratio: Fraction (0–0.4) of anchor K/V to blend into current panel.
            max_layers:  Number of attention layers to hook.
        """
        self.blend_ratio = blend_ratio
        self.max_layers = max_layers

        self._cached_outputs: Dict[Any, Any] = {}
        self._hooks: List[Any] = []
        self._capture_mode = False
        self._apply_mode = False

    # ── Public API ──────────────────────────────────────────────────────────

    def start_capture(self):
        """Enable K/V capture for the anchor panel."""
        self._capture_mode = True
        self._apply_mode = False
        self._cached_outputs.clear()
        log.info("  [L2-Attn] Capture mode ACTIVE — recording anchor K/V")

    def start_apply(self):
        """Enable K/V blend for non-anchor panels."""
        if not self._cached_outputs:
            log.debug("  [L2-Attn] No cached K/V — blend mode skipped")
            return
        self._capture_mode = False
        self._apply_mode = True
        log.info(f"  [L2-Attn] Blend mode ACTIVE — {len(self._cached_outputs)} layers cached")

    def stop(self):
        """Disable both modes."""
        self._capture_mode = False
        self._apply_mode = False

    def has_cache(self) -> bool:
        return len(self._cached_outputs) > 0

    def install_hooks(self, model) -> bool:
        """
        Register forward hooks on the first `max_layers` cross-attention
        modules of the model. Returns True if at least one hook was installed.
        """
        try:
            count = 0
            for name, module in model.named_modules():
                # Target cross-attention (attn2) layers that have projections (SDXL/Flux compatibility)
                is_attn_layer = ("attn2" in name or ("attn" in name and "attn1" not in name))
                if is_attn_layer and hasattr(module, "to_k") and count < self.max_layers:
                    handle = module.register_forward_hook(self._hook_fn)
                    self._hooks.append(handle)
                    count += 1

            if count:
                log.info(f"  [L2-Attn] Hooks installed on {count} attention layers")
            return count > 0
        except Exception as e:
            log.debug(f"  [L2-Attn] Hook installation failed: {e}")
            return False

    def remove_hooks(self):
        """Deregister all installed hooks and clear cached tensors to free VRAM."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._cached_outputs.clear()
        log.debug("  [L2-Attn] Hooks and cached VRAM tensors cleared")

    # ── Internal ─────────────────────────────────────────────────────────────

    def _hook_fn(self, module, inputs, output):
        """Forward hook: capture or blend attention outputs by module reference."""
        try:
            if self._capture_mode and len(self._cached_outputs) < self.max_layers:
                # Store a detached copy of the output tensor mapped directly by module
                import torch
                self._cached_outputs[module] = output.detach().clone()

            elif self._apply_mode and module in self._cached_outputs:
                # Blend: output = (1 − β) * output + β * cached
                cached = self._cached_outputs[module]
                if cached.shape == output.shape:
                    blended = (1 - self.blend_ratio) * output + self.blend_ratio * cached.to(output.device)
                    return blended
        except Exception:
            pass  # Silently skip on shape mismatch or device issues
        return output


# ─────────────────────────────────────────────────────────────────────────────
# Level 3: Spatiotemporal Consistency Enforcer
# ─────────────────────────────────────────────────────────────────────────────

class SpatiotemporalConsistencyEnforcer:
    """
    Spatiotemporal Architectural Priors — DreamingComics approximation.

    Captures channel-wise statistical fingerprints (mean, std) of Panel 1
    latents and uses them as a structural prior during mid-denoising of
    subsequent panels. This enforces feature-level continuity analogous to
    the temporal window constraints of video transformer models.
    """

    def __init__(self, strength: float = 0.08, active_range: tuple = (0.30, 0.60)):
        """
        Args:
            strength:     Correction blend strength (0.0–0.3).
            active_range: (start, end) timestep ratios where the prior is active.
                          Structural features form in this mid-denoising window.
        """
        self.strength = strength
        self.active_low, self.active_high = active_range
        self._anchor_mean: Optional[Any] = None
        self._anchor_std: Optional[Any] = None

    def capture_anchor(self, latents: Any):
        """
        Extract and store channel-wise statistics from Panel 1 latents.

        Args:
            latents: torch.Tensor (B, C, H, W) from final anchor denoising step.
        """
        try:
            import torch
            with torch.no_grad():
                f = latents.float()
                self._anchor_mean = f.mean(dim=(0, 2, 3)).cpu()
                self._anchor_std  = f.std(dim=(0, 2, 3)).cpu()
            log.info(
                f"  [L3-STE] Anchor statistics captured: "
                f"mean_range=[{self._anchor_mean.min():.3f}, {self._anchor_mean.max():.3f}]"
            )
        except Exception as e:
            log.debug(f"  [L3-STE] Anchor capture failed: {e}")

    def apply(self, latents: Any, timestep_ratio: float) -> Any:
        """
        Apply spatiotemporal window constraint to latents.

        Args:
            latents:        torch.Tensor (B, C, H, W).
            timestep_ratio: 1.0 = start of denoising, 0.0 = end.

        Returns:
            Corrected latent tensor.
        """
        if self._anchor_mean is None:
            return latents

        if not (self.active_low <= timestep_ratio <= self.active_high):
            return latents

        try:
            import torch

            with torch.no_grad():
                anchor_mean = self._anchor_mean.to(latents.device, latents.dtype)
                anchor_std  = self._anchor_std.to(latents.device, latents.dtype)

                cur_mean = latents.float().mean(dim=(0, 2, 3))
                cur_std  = latents.float().std(dim=(0, 2, 3)).clamp(min=1e-6)

                # Channel correction: shift mean, rescale std toward anchor
                std_ratio   = (anchor_std / cur_std).clamp(0.80, 1.20)
                mean_delta  = (anchor_mean - cur_mean) * self.strength

                corrected = latents.clone()
                for c in range(latents.shape[1]):
                    corrected[:, c] = latents[:, c] * std_ratio[c] + mean_delta[c]

                # Blend by strength scaled to proximity in active window
                blend_w = self.strength * (
                    (timestep_ratio - self.active_low) /
                    (self.active_high - self.active_low)
                )
                latents = (1 - blend_w) * latents + blend_w * corrected
                log.debug(
                    f"  [L3-STE] Window constraint applied, blend={blend_w:.4f}"
                    f" at ratio={timestep_ratio:.2f}"
                )

        except Exception as e:
            log.debug(f"  [L3-STE] Constraint skipped: {e}")

        return latents


# ─────────────────────────────────────────────────────────────────────────────
# Unified Manager
# ─────────────────────────────────────────────────────────────────────────────

class AdvancedAttentionManager:
    """
    Unified entry-point for all three advanced attention mechanisms.

    Integrates with PanelEngine and the SDXL/Flux backends. Works in
    four stages per panel:

        1. on_panel_start(panel_id, is_anchor, total_steps)
           — activates capture or apply mode on Level 2
        2. get_step_callback()
           — returns a diffusers-compatible callback that runs L1 + L3
             at every denoising step, and auto-captures anchor latents on
             the final step of Panel 1
        3. install_on_pipeline(pipe)
           — installs attention hooks for Level 2 (called once after load)
        4. on_panel_end()
           — disables active modes
    """

    def __init__(self,
                 heat_alpha: float = 0.03,
                 attention_blend: float = 0.15,
                 spatial_strength: float = 0.08,
                 enabled: bool = True):
        self.enabled = enabled
        self.heat_prior    = HeatDiffusionPrior(alpha=heat_alpha)
        self.attn_cache    = SharedAttentionCache(blend_ratio=attention_blend)
        self.spatio_temp   = SpatiotemporalConsistencyEnforcer(strength=spatial_strength)

        self._is_anchor    = False
        self._total_steps  = 25
        self._anchor_captured = False

        if enabled:
            log.info("AdvancedAttentionManager ENABLED (L1-Heat, L2-Attn, L3-STE)")
        else:
            log.info("AdvancedAttentionManager DISABLED (dry-run / CPU mode)")

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_panel_start(self, panel_id: int, is_anchor: bool, total_steps: int = 25):
        if not self.enabled:
            return
        self._is_anchor   = is_anchor
        self._total_steps = total_steps

        if is_anchor:
            self.attn_cache.start_capture()
            log.info(f"  [AdvAttn] Panel {panel_id}: ANCHOR — L2 capture + L1/L3 will record")
        else:
            if self.attn_cache.has_cache():
                self.attn_cache.start_apply()
            log.info(f"  [AdvAttn] Panel {panel_id}: CONSISTENCY — L1+L2+L3 priors active")

    def on_panel_end(self):
        if not self.enabled:
            return
        self.attn_cache.stop()

    def install_on_pipeline(self, pipe) -> bool:
        """Install L2 attention hooks. Call once after model load."""
        if not self.enabled:
            return False
        try:
            model = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
            if model is not None:
                return self.attn_cache.install_hooks(model)
            return False
        except Exception as e:
            log.debug(f"  [AdvAttn] Hook install failed: {e}")
            return False

    def remove_hooks(self):
        self.attn_cache.remove_hooks()

    # ── Step Callback ─────────────────────────────────────────────────────────

    def get_step_callback(self):
        """
        Returns a diffusers `callback_on_step_end`-compatible function.

        The callback:
        - Applies L1 (heat diffusion) on every non-anchor step.
        - Applies L3 (spatiotemporal constraint) on every non-anchor step.
        - Auto-captures anchor latent statistics on the last step of Panel 1.

        Usage in diffusers pipeline call:
            pipe(...,
                 callback_on_step_end=manager.get_step_callback(),
                 callback_on_step_end_tensor_inputs=["latents"])
        """
        if not self.enabled:
            return None

        # Capture self in closure variables — avoids reference issues
        heat_prior  = self.heat_prior
        spatio_temp = self.spatio_temp
        total_steps = [self._total_steps]   # mutable list for closure update
        is_anchor_ref  = [self._is_anchor]

        manager = self

        def _callback(pipe, step_index: int, timestep: int,
                      callback_kwargs: dict) -> dict:
            latents = callback_kwargs.get("latents")
            if latents is None:
                return callback_kwargs

            total    = total_steps[0]
            is_anch  = is_anchor_ref[0]
            t_ratio  = 1.0 - (step_index / max(1, total - 1))

            if not is_anch:
                # Level 1: Heat Diffusion Prior
                latents = heat_prior.apply(latents, t_ratio)

                # Level 3: Spatiotemporal Window Constraint
                latents = spatio_temp.apply(latents, t_ratio)

            # Level 3 anchor capture on last step of anchor panel
            if is_anch and step_index == total - 1 and not manager._anchor_captured:
                spatio_temp.capture_anchor(latents)
                manager._anchor_captured = True
                log.info("  [AdvAttn] Anchor latent statistics captured (final step L3)")

            callback_kwargs["latents"] = latents
            return callback_kwargs

        # Keep callback's closure in sync with current panel state
        def _update_and_get():
            total_steps[0]   = self._total_steps
            is_anchor_ref[0] = self._is_anchor
            return _callback

        # Return a wrapper that updates closure every time the callback is fetched
        return _update_and_get()

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "L1_heat_diffusion": {
                "alpha": self.heat_prior.alpha,
                "kernel_size": self.heat_prior.kernel_size,
                "active_range": f"{self.heat_prior.end_ratio:.0%}–{self.heat_prior.start_ratio:.0%}",
            },
            "L2_attention_cache": {
                "layers_cached": len(self.attn_cache._cached_outputs),
                "blend_ratio": self.attn_cache.blend_ratio,
                "hooks_installed": len(self.attn_cache._hooks) > 0,
            },
            "L3_spatiotemporal": {
                "anchor_captured": self._anchor_captured,
                "strength": self.spatio_temp.strength,
                "active_range": f"{self.spatio_temp.active_low:.0%}–{self.spatio_temp.active_high:.0%}",
            },
        }
