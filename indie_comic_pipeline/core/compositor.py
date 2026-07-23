"""
CHARCOM INFERENCE COMPOSITOR — Phase 3
========================================
Calculates dynamic model weight blending at runtime:
W_total = W_base + Σ(α_i * W_i)

Manages LoRA weight scaling per panel based on scene requirements,
character emotion intensity, and action level.
"""

import logging
from typing import Dict, Any, Optional, Tuple

log = logging.getLogger("pipeline.compositor")


class CharComCompositor:
    """
    CharCom Inference Compositor.

    Dynamically adjusts model parameters at runtime based on:
    - Scene action intensity (from Layout Agent)
    - Character emotional state (from Character Agent)
    - Panel position in the story arc (from Storyboard Agent)
    - Consistency requirements (from anchor tokens)

    Outputs generation parameters for the backend to use.
    """

    def __init__(self, base_lora_scale: float = 0.8,
                 base_guidance: float = 7.5,
                 base_steps: int = 25):
        self.base_lora_scale = base_lora_scale
        self.base_guidance = base_guidance
        self.base_steps = base_steps

    def compute_weights(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate dynamic generation parameters for a panel.

        Args:
            context: Generation context from AgentCoordinator.get_generation_context()

        Returns:
            Dict with:
            - lora_scale: adjusted LoRA adapter weight
            - guidance_scale: adjusted classifier-free guidance
            - num_steps: adjusted inference steps
            - seed_offset: deterministic seed variation
        """
        # Extract context values
        panel_id = context.get("panel_id", 1)
        layout = context.get("layout", {})
        beat = context.get("panel_emotion_beat", "neutral")
        has_anchor = context.get("has_anchor", False)

        # Base weights
        lora_scale = self.base_lora_scale
        guidance = self.base_guidance
        
        # Adaptive steps based on scene complexity
        size_class = layout.get("size_class", "medium") if layout else "medium"
        camera_angle = layout.get("camera_angle", "medium_shot") if layout else "medium_shot"
        action_intensity = context.get("action_intensity", "medium")
        
        # High complexity (Full-page, large, action-heavy, or wide shots)
        if size_class in ["full_page", "large"] or action_intensity == "high" or camera_angle == "wide_shot":
            steps = 25
        # Low complexity (Close-up, extreme close-up, small panels, or low action)
        elif camera_angle in ["close_up", "extreme_close_up"] or size_class == "small" or action_intensity == "low":
            steps = 15
        # Medium complexity
        else:
            steps = 20

        # ── Apply critic adjustment overrides (from reject-regenerate loop) ──
        if "guidance_scale_override" in context:
            guidance = context["guidance_scale_override"]
        if "steps_override" in context:
            steps = context["steps_override"]

        # ── Adjustment 1: Action Intensity (Guidance only, steps are already adaptive) ──
        # Higher intensity → slight guidance increase for more dramatic output
        if size_class == "full_page":
            guidance += 0.5
        elif size_class == "large":
            guidance += 0.25

        # ── Adjustment 2: Emotion Intensity ──
        # Strong emotions → slightly more guidance for clearer expression
        high_emotion_beats = {
            "contained_fire", "fracture", "peak_noise", "overflow",
            "breakthrough", "triumph", "ache", "spiral",
        }
        low_emotion_beats = {
            "stillness", "drift", "quiet_rest", "fade",
            "softness", "surrender",
        }

        if beat in high_emotion_beats:
            guidance += 0.5
            lora_scale = min(1.0, lora_scale + 0.05)
        elif beat in low_emotion_beats:
            guidance -= 0.25
            lora_scale = max(0.5, lora_scale - 0.05)

        # ── Adjustment 3: Consistency Enforcement ──
        # After anchor is set, slightly increase guidance for consistency
        if has_anchor and panel_id > 1:
            guidance += 0.25

        # ── Adjustment 4: Panel Position ──
        # First and last panels get more steps for quality
        total_panels = context.get("total_panels", 1)
        if panel_id == 1 or panel_id == total_panels:
            steps += 3

        # ── Clamp values ──
        guidance = max(5.0, min(12.0, guidance))
        lora_scale = max(0.3, min(1.0, lora_scale))
        steps = max(15, min(50, steps))

        # Deterministic seed with panel variation (process-stable hash)
        beat_hash = sum(ord(c) for c in beat)
        seed_offset = panel_id * 7 + beat_hash % 100

        result = {
            "lora_scale": round(lora_scale, 3),
            "guidance_scale": round(guidance, 2),
            "num_steps": steps,
            "seed_offset": seed_offset,
        }

        log.debug(
            f"Compositor weights for panel {panel_id}: "
            f"lora={result['lora_scale']}, guidance={result['guidance_scale']}, "
            f"steps={result['num_steps']}"
        )

        return result

    def get_negative_prompt_augments(self, context: Dict[str, Any]) -> str:
        """
        Generate additional negative prompt terms based on context.
        """
        augments = []
        beat = context.get("panel_emotion_beat", "neutral")

        # Camera angle specific negatives
        layout = context.get("layout", {})
        angle = layout.get("camera_angle", "medium_shot") if layout else "medium_shot"

        if angle == "close_up":
            augments.append("full body shot, wide angle distortion")
        elif angle == "wide_shot":
            augments.append("extreme close up, blurry background")
        elif angle == "bird_eye":
            augments.append("ground level, low angle")

        # Consistency negatives
        if context.get("has_anchor", False):
            augments.append("inconsistent character design, changing art style")

        return ", ".join(augments) if augments else ""
