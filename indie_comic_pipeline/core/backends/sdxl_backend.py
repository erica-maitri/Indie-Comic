"""
SDXL BACKEND — Stable Diffusion XL Model Backend
===================================================
Wraps the existing SDXL + LoRA pipeline into the composable backend
interface. Includes T4 GPU optimizations, scheduler configuration,
and LoRA adapter management.
"""

import os
import gc
import logging
from typing import Dict, Any, Optional
from PIL import Image

from core.backends.base_backend import BaseBackend

log = logging.getLogger("pipeline.backends.sdxl")


class SDXLBackend(BaseBackend):
    """
    SDXL Model Backend with LoRA support and T4 GPU optimizations.

    Wraps stabilityai/stable-diffusion-xl-base-1.0 with:
    - DPMSolver++ scheduler (Karras sigmas)
    - LoRA adapter for comic/lineart style
    - Attention slicing + VAE slicing for T4
    - Optional CPU offload for low-VRAM environments
    """

    def __init__(self):
        self._pipe = None
        self._config: Dict[str, Any] = {}
        self._lora_loaded = False
        self.device = "cuda"

    @property
    def name(self) -> str:
        return "SDXL"

    @property
    def supports_lora(self) -> bool:
        return True

    def load(self, config: Dict[str, Any]):
        """
        Load the SDXL pipeline with optional LoRA.

        Config keys:
        - model_name: HuggingFace model ID
        - lora_name: LoRA adapter name (optional)
        - lora_scale: LoRA weight (default 0.8)
        - device: "cuda" or "cpu"
        - enable_cpu_offload: bool
        - enable_attention_slicing: bool
        - enable_vae_slicing: bool
        """
        import torch
        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

        self._config = config
        model_name = config.get("model_name", "stabilityai/stable-diffusion-xl-base-1.0")
        device = config.get("device", "cuda")

        # Resolve device based on availability
        if device == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA is requested but not available. Falling back to CPU.")
            device = "cpu"

        self.device = device

        log.info(f"Loading SDXL pipeline: {model_name} on device: {device}")

        # Choose dtype and variant based on target device
        # CPU does not support float16 for many torch operations; use float32.
        dtype = torch.float32 if device == "cpu" else torch.float16
        variant = None if device == "cpu" else "fp16"

        try:
            self._pipe = StableDiffusionXLPipeline.from_pretrained(
                model_name,
                torch_dtype=dtype,
                use_safetensors=True,
                variant=variant,
                low_cpu_mem_usage=True,
            )
        except Exception:
            log.warning(f"{variant or 'FP16'} variant not available, trying without variant")
            self._pipe = StableDiffusionXLPipeline.from_pretrained(
                model_name,
                torch_dtype=dtype,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            )

        # Configure scheduler
        scheduler_config = dict(self._pipe.scheduler.config)
        scheduler_config.pop("_class_name", None)
        scheduler_config.pop("algorithm_type", None)
        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            scheduler_config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++",
            solver_order=2
        )

        # Optimizations and device placement
        if device == "cpu":
            self._pipe = self._pipe.to("cpu")
            log.info("  Moved pipeline to CPU")
        else:
            if config.get("enable_cpu_offload", True):
                try:
                    self._pipe.enable_model_cpu_offload()
                    log.info("  CPU offload enabled")
                except (ImportError, RuntimeError) as e:
                    log.warning(f"  CPU offload failed (likely missing 'accelerate' package): {e}. Falling back to .to({device})")
                    self._pipe = self._pipe.to(device)
            else:
                self._pipe = self._pipe.to(device)

        if config.get("enable_attention_slicing", True):
            self._pipe.enable_attention_slicing()

        if config.get("enable_vae_slicing", True):
            self._pipe.enable_vae_slicing()

        # Safety checker removal
        if not config.get("safety_checker", False):
            self._pipe.safety_checker = None

        # Load LoRA adapter if specified
        lora_name = config.get("lora_name")
        if lora_name:
            try:
                self._pipe.load_lora_weights(lora_name)
                self._lora_loaded = True
                log.info(f"  LoRA adapter loaded: {lora_name}")
            except Exception as e:
                log.warning(f"  LoRA loading failed: {e}")
                self._lora_loaded = False

        log.info("SDXL pipeline loaded successfully")

    def get_raw_pipeline(self):
        """
        Return the underlying diffusers pipeline object.
        Used by AdvancedAttentionManager to install UNet attention hooks (L2).
        """
        return self._pipe

    def generate(self, prompt: str, negative_prompt: str,
                 config: Dict[str, Any]) -> Image.Image:
        """
        Generate an image using SDXL.

        Extended config keys (beyond base spec):
        - step_callback: callable — diffusers callback_on_step_end function
                         (passed by AdvancedAttentionManager for L1 + L3 priors)
        - callback_tensor_inputs: list[str] — tensor names exposed to callback
                                  (default: ["latents"])
        """
        import torch

        if self._pipe is None:
            raise RuntimeError("SDXL backend not loaded. Call load() first.")

        width = config.get("width", 768)
        height = config.get("height", 768)
        steps = config.get("num_steps", 25)
        guidance = config.get("guidance_scale", 7.5)
        seed = config.get("seed", 42)
        lora_scale = config.get("lora_scale", 0.8)

        generator = torch.Generator(device=self._pipe.device).manual_seed(seed)

        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": generator,
        }

        # Apply LoRA scale if LoRA is loaded
        if self._lora_loaded:
            gen_kwargs["cross_attention_kwargs"] = {"scale": lora_scale}

        # --- COMPEL INTEGRATION FOR LONG PROMPTS (>77 tokens) ---
        try:
            from compel import Compel, ReturnedEmbeddingsType
            if getattr(self, "_compel", None) is None:
                self._compel = Compel(
                    tokenizer=[self._pipe.tokenizer, self._pipe.tokenizer_2],
                    text_encoder=[self._pipe.text_encoder, self._pipe.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True],
                    device=self.device
                )
            
            # Encode prompt
            prompt_embeds, pooled_prompt_embeds = self._compel(prompt)
            gen_kwargs.pop("prompt")
            gen_kwargs["prompt_embeds"] = prompt_embeds
            gen_kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds
            
            # Encode negative prompt
            if negative_prompt:
                neg_prompt_embeds, neg_pooled_prompt_embeds = self._compel(negative_prompt)
                gen_kwargs.pop("negative_prompt")
                gen_kwargs["negative_prompt_embeds"] = neg_prompt_embeds
                gen_kwargs["negative_pooled_prompt_embeds"] = neg_pooled_prompt_embeds
                
            log.info("  [SDXL] Using Compel for prompt embeddings (bypassing 77 token limit)")
        except ImportError:
            log.warning("  [SDXL] 'compel' library not installed. Token truncation > 77 may occur.")
        # --------------------------------------------------------

        # ── Advanced Attention step callback (L1 Heat + L3 Spatiotemporal) ──
        step_callback = config.get("step_callback")
        if step_callback is not None:
            gen_kwargs["callback_on_step_end"] = step_callback
            gen_kwargs["callback_on_step_end_tensor_inputs"] = (
                config.get("callback_tensor_inputs", ["latents"])
            )
            log.info(f"  [SDXL] Advanced attention callback ACTIVE ({steps} steps)")

        with torch.inference_mode():
            if self._pipe is None:
                raise RuntimeError("SDXL pipeline is not loaded. Call load() first.")
            result = self._pipe(**gen_kwargs)

        # Clean up VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result.images[0]

    def unload(self):
        """Unload the pipeline from GPU memory."""
        import torch

        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            self._lora_loaded = False
            gc.collect()
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.info("SDXL pipeline unloaded")

    def is_loaded(self) -> bool:
        return self._pipe is not None

    def get_cross_attention_modules(self) -> list:
        """Expose cross-attention modules for SDXL's UNet (Backend Adapter Pattern)."""
        if self._pipe is None:
            return []
        model = getattr(self._pipe, "unet", None)
        if model is None:
            return []
        
        modules = []
        count = 0
        for name, module in model.named_modules():
            # Target cross-attention (attn2) layers with projections
            is_attn = ("attn2" in name or ("attn" in name and "attn1" not in name))
            if is_attn and hasattr(module, "to_k") and count < 4:  # max_layers limit (default 4)
                modules.append(module)
                count += 1
        return modules

    def get_vram_estimate_mb(self) -> int:
        return 6500  # ~6.5GB for SDXL FP16 with CPU offload
