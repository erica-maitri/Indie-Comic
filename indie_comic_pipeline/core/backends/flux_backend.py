"""
FLUX BACKEND — High-Fidelity Character Scene Backend (Stub)
=============================================================
Interface for the Flux model backend for high-fidelity character scenes.
Currently a stub that falls back to SDXL with enhanced settings.
Will be replaced with real Flux integration when available.
"""

import logging
from typing import Dict, Any, Optional
from PIL import Image

from core.backends.base_backend import BaseBackend

log = logging.getLogger("pipeline.backends.flux")


class FluxBackend(BaseBackend):
    """
    Flux Model Backend (Stub).

    Placeholder for the Flux diffusion model backend. Currently delegates
    to the SDXL backend with enhanced quality settings. When real Flux
    support is added, this class will wrap the Flux pipeline directly.
    """

    def __init__(self):
        self._sdxl_fallback: Optional[BaseBackend] = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "Flux"

    @property
    def supports_lora(self) -> bool:
        return False  # Flux LoRA support TBD

    def load(self, config: Dict[str, Any]):
        """Load Flux backend (currently falls back to SDXL with enhanced settings)."""
        log.info("Flux backend requested — using SDXL enhanced fallback")

        from core.backends.sdxl_backend import SDXLBackend
        self._sdxl_fallback = SDXLBackend()

        # Override config for higher quality
        enhanced_config = config.copy()
        enhanced_config.setdefault("model_name", "stabilityai/stable-diffusion-xl-base-1.0")

        self._sdxl_fallback.load(enhanced_config)
        self._loaded = True

    def generate(self, prompt: str, negative_prompt: str,
                 config: Dict[str, Any]) -> Image.Image:
        """Generate using Flux (SDXL enhanced fallback)."""
        if not self.is_loaded():
            log.info("Flux backend not fully loaded — auto-loading fallback")
            self.load(config)

        if not self.is_loaded() or self._sdxl_fallback is None:
            raise RuntimeError("Flux backend failed to load. Call load() first.")

        # Enhance config for higher fidelity output
        enhanced_config = config.copy()
        enhanced_config["num_steps"] = max(config.get("num_steps", 25), 30)
        enhanced_config["guidance_scale"] = max(config.get("guidance_scale", 7.5), 8.0)

        # Prepend quality markers
        enhanced_prompt = f"masterpiece, best quality, {prompt}"

        return self._sdxl_fallback.generate(enhanced_prompt, negative_prompt,
                                            enhanced_config)

    def unload(self):
        """Unload Flux backend."""
        if self._sdxl_fallback:
            self._sdxl_fallback.unload()
            self._sdxl_fallback = None
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded and self._sdxl_fallback is not None and self._sdxl_fallback.is_loaded()

    def get_vram_estimate_mb(self) -> int:
        return 7000  # Slightly more than SDXL due to enhanced settings
