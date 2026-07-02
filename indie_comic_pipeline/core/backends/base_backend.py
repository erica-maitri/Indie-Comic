"""
BASE BACKEND — Abstract Model Backend Interface
=================================================
Defines the contract all diffusion model backends must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from PIL import Image


class BaseBackend(ABC):
    """
    Abstract interface for diffusion model backends.
    All backends (SDXL, Flux, Video DiT) implement this protocol.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
        ...

    @property
    @abstractmethod
    def supports_lora(self) -> bool:
        """Whether this backend supports LoRA adapters."""
        ...

    @abstractmethod
    def load(self, config: Dict[str, Any]):
        """Load the model with the given configuration."""
        ...

    @abstractmethod
    def generate(self, prompt: str, negative_prompt: str,
                 config: Dict[str, Any]) -> Image.Image:
        """
        Generate an image from prompts and configuration.

        Args:
            prompt: Positive prompt text
            negative_prompt: Negative prompt text
            config: Generation config dict with:
                - width, height: image dimensions
                - num_steps: inference steps
                - guidance_scale: classifier-free guidance
                - lora_scale: LoRA adapter weight
                - seed: random seed

        Returns:
            Generated PIL Image
        """
        ...

    @abstractmethod
    def unload(self):
        """Unload model from memory."""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        ...

    def get_cross_attention_modules(self) -> list:
        """
        Expose cross-attention modules for consistency hook installation.
        Overridden by backends that support hooks natively (Backend Adapter Pattern).
        """
        return []

    def get_vram_estimate_mb(self) -> int:
        """Estimated VRAM usage in MB. Override in subclasses."""
        return 0
