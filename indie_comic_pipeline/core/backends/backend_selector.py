"""
BACKEND SELECTOR — Model Backend Selection Engine
====================================================
Chooses the optimal diffusion model backend (SDXL, Flux, Video DiT)
based on scene metadata from the Layout Agent. Implements a fallback
chain to ensure generation always succeeds.
"""

import logging
from typing import Dict, Any, Optional

from core.backends.base_backend import BaseBackend
from core.backends.sdxl_backend import SDXLBackend
from core.backends.flux_backend import FluxBackend

log = logging.getLogger("pipeline.backends.selector")


# Selection rules: (size_class, camera_angle) → preferred backend
BACKEND_RULES = {
    # Full-page spreads and bird's eye → try Flux for quality
    ("full_page", "bird_eye"):      "flux",
    ("full_page", "wide_shot"):     "flux",
    ("full_page", "medium_shot"):   "flux",
    # Large panels with close-ups → SDXL with LoRA
    ("large", "close_up"):          "sdxl",
    ("large", "medium_shot"):       "sdxl",
    # Everything else → SDXL (reliable workhorse)
    ("medium", "close_up"):         "sdxl",
    ("medium", "medium_shot"):      "sdxl",
    ("medium", "wide_shot"):        "sdxl",
    ("small", "close_up"):          "sdxl",
    ("small", "medium_shot"):       "sdxl",
    ("small", "wide_shot"):         "sdxl",
}


class BackendSelector:
    """
    Model Backend Selector.

    Chooses the optimal backend based on:
    - Panel size class from Layout Agent
    - Camera angle requirements
    - Available VRAM
    - User configuration overrides

    Implements a fallback chain: preferred → SDXL → error
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._backends: Dict[str, BaseBackend] = {}
        self._active_backend: Optional[str] = None
        self._model_config: Dict[str, Any] = {}

        # User override: force a specific backend for all panels
        self._force_backend = self._config.get("force_backend", None)

    def register_backend(self, name: str, backend: BaseBackend):
        """Register an available backend."""
        self._backends[name] = backend
        log.info(f"Registered backend: {name} ({backend.name})")

    def _ensure_backend_loaded(self, name: str, backend: BaseBackend):
        """Loads the weights for the specified backend lazily if not already loaded."""
        if not backend.is_loaded():
            log.info(f"Lazily loading model weights for backend: {name} ({backend.name})...")
            model_config = self._model_config
            sdxl_config = {
                "model_name": model_config.get("sdxl", {}).get("name",
                             "stabilityai/stable-diffusion-xl-base-1.0"),
                "lora_name": model_config.get("lora", {}).get("name"),
                "lora_scale": model_config.get("lora", {}).get("adapter_scale", 0.8),
                "device": model_config.get("sdxl", {}).get("device", "cuda"),
                "enable_cpu_offload": model_config.get("sdxl", {}).get("cpu_offload", True),
                "enable_attention_slicing": True,
                "enable_vae_slicing": True,
                "safety_checker": False,
            }
            backend.load(sdxl_config)

    def select(self, context: Dict[str, Any]) -> BaseBackend:
        """
        Select the best backend for a given panel context, loading it dynamically if needed.

        Args:
            context: Generation context from AgentCoordinator

        Returns:
            The selected BaseBackend instance
        """
        # User override
        if self._force_backend:
            backend = self._backends.get(self._force_backend)
            if backend:
                self._ensure_backend_loaded(self._force_backend, backend)
                return backend
            log.warning(f"Forced backend '{self._force_backend}' not available, using fallback")

        # Get layout info from context
        layout = context.get("layout", {})
        size_class = layout.get("size_class", "medium") if layout else "medium"
        camera_angle = layout.get("camera_angle", "medium_shot") if layout else "medium_shot"

        # Look up preferred backend
        rule_key = (size_class, camera_angle)
        preferred = BACKEND_RULES.get(rule_key, "sdxl")

        # Try preferred backend
        backend = self._backends.get(preferred)
        if backend:
            self._ensure_backend_loaded(preferred, backend)
            log.debug(f"Selected backend: {preferred} for ({size_class}, {camera_angle})")
            return backend

        # Fallback to SDXL
        sdxl = self._backends.get("sdxl")
        if sdxl:
            self._ensure_backend_loaded("sdxl", sdxl)
            log.debug(f"Fallback to SDXL (preferred '{preferred}' not available)")
            return sdxl

        # Emergency: try any registered backend
        for name, b in self._backends.items():
            self._ensure_backend_loaded(name, b)
            log.warning(f"Emergency fallback to '{name}'")
            return b

        raise RuntimeError("No loaded backends available for generation")

    def initialize_backends(self, model_config: Dict[str, Any]):
        """
        Initialize available backends based on configuration without loading weights.

        Args:
            model_config: Model configuration from settings.yaml
        """
        self._model_config = model_config
        
        # Instantiate SDXL Backend
        sdxl = SDXLBackend()
        self.register_backend("sdxl", sdxl)

        # Instantiate Flux Backend if enabled
        enable_flux = model_config.get("flux", {}).get("enabled", False)
        if enable_flux:
            flux = FluxBackend()
            self.register_backend("flux", flux)

    def unload_all(self):
        """Unload all backends to free memory."""
        for name, backend in self._backends.items():
            if backend.is_loaded():
                backend.unload()
                log.info(f"Unloaded backend: {name}")
