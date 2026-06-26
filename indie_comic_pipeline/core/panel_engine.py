"""
UNIFIED PANEL GENERATION ENGINE — Phases 2-4
===============================================
Orchestrates the full panel generation flow:
1. Pull context from Story Section Memory
2. Run CharCom Compositor for weight blending
3. Select backend via Backend Selector
4. Apply consistency priors (prompt augmentation from anchor tokens)
5. Generate image via selected backend
6. Extract features from generated panel
7. Update Memory with new panel data
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from PIL import Image
from pathlib import Path

if TYPE_CHECKING:
    from core.agents.agent_coordinator import AgentCoordinator

from core.memory import StorySectionMemory, PanelRecord
from core.anchoring import ReferenceFreeAnchor
from core.compositor import CharComCompositor
from core.backends.backend_selector import BackendSelector
from core.advanced_attention import AdvancedAttentionManager

log = logging.getLogger("pipeline.panel_engine")


class PanelEngine:
    """
    Unified Panel Generation Engine.

    Replaces the monolithic ModelEnsemble / PanelGenerator with a modular,
    context-aware pipeline that integrates all Phase 2-4 components.
    """

    def __init__(self, memory: StorySectionMemory,
                 backend_selector: BackendSelector,
                 compositor: Optional[CharComCompositor] = None,
                 anchor_system: Optional[ReferenceFreeAnchor] = None,
                 advanced_attention: Optional[AdvancedAttentionManager] = None,
                 output_dir: str = "outputs/panels"):
        self.memory = memory
        self.backend_selector = backend_selector
        self.compositor = compositor or CharComCompositor()
        self.anchor_system = anchor_system or ReferenceFreeAnchor()
        self.advanced_attention = advanced_attention  # None = disabled
        self.output_dir = output_dir
        self._prompt_optimizer = None
        self._hooks_installed = False

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def generate_panel(self, panel_id: int,
                       context: Dict[str, Any],
                       style_prompt: str = "",
                       negative_base: str = "") -> Dict[str, Any]:
        """
        Generate a single panel using the full Phase 2-4 pipeline.

        Args:
            panel_id: Panel number (1-indexed)
            context: Generation context from AgentCoordinator.get_generation_context()
            style_prompt: Base style prompt string
            negative_base: Base negative prompt string

        Returns:
            Panel result dict with: panel_id, image, image_path, prompt,
            quality_score, consistency_score, generation_time
        """
        start_time = time.time()
        log.info(f"\n{'─'*50}")
        log.info(f"Panel Engine: Generating Panel {panel_id}/{self.memory.total_panels}")
        log.info(f"{'─'*50}")

        # ── Step 1: Build the generation prompt ──
        prompt = self._build_prompt(context, style_prompt)
        log.info(f"  [1] Prompt built ({len(prompt)} chars)")

        # ── Step 2: Run CharCom Compositor ──
        weights = self.compositor.compute_weights(context)
        log.info(f"  [2] Compositor: guidance={weights['guidance_scale']}, "
                 f"lora={weights['lora_scale']}, steps={weights['num_steps']}")

        # ── Step 3: Select backend ──
        backend = self.backend_selector.select(context)
        log.info(f"  [3] Backend selected: {backend.name}")

        # ── Step 3b: Install advanced attention hooks (once per pipeline) ──
        if (self.advanced_attention and not self._hooks_installed
                and hasattr(backend, "get_raw_pipeline")):
            pipe = backend.get_raw_pipeline()
            if pipe is not None:
                self.advanced_attention.install_on_pipeline(pipe)
                self._hooks_installed = True
                log.info("  [3b] Advanced attention hooks installed on UNet")

        # ── Step 4: Apply consistency priors ──
        negative_prompt = self._build_negative(context, negative_base)
        consistency_guidance = self.anchor_system.get_consistency_guidance(self.memory)
        prompt += consistency_guidance.get("prompt_suffix", "")
        neg_augment = consistency_guidance.get("negative_augment", "")
        if neg_augment:
            negative_prompt += f", {neg_augment}"

        # Add compositor negative augments
        comp_neg = self.compositor.get_negative_prompt_augments(context)
        if comp_neg:
            negative_prompt += f", {comp_neg}"

        log.info(f"  [4] Consistency priors applied")

        # ── Step 4b: Advanced Attention — activate per-panel modes ──
        if self.advanced_attention:
            is_anchor = (panel_id == 1)
            steps = weights["num_steps"]
            self.advanced_attention.on_panel_start(
                panel_id=panel_id, is_anchor=is_anchor, total_steps=steps
            )
            log.info(f"  [4b] Advanced attention: "
                     f"{'ANCHOR capture' if is_anchor else 'consistency priors'} active")

        # ── Step 5: Generate image ──
        seed = 42 + weights["seed_offset"]
        gen_config = {
            "width": self._get_resolution(context)[0],
            "height": self._get_resolution(context)[1],
            "num_steps": weights["num_steps"],
            "guidance_scale": weights["guidance_scale"],
            "lora_scale": weights["lora_scale"],
            "seed": seed,
        }

        # Inject advanced attention step callback (L1 + L3) if active
        if self.advanced_attention and self.advanced_attention.enabled:
            cb = self.advanced_attention.get_step_callback()
            if cb is not None:
                gen_config["step_callback"] = cb
                gen_config["callback_tensor_inputs"] = ["latents"]

        image = backend.generate(prompt, negative_prompt, gen_config)
        log.info(f"  [5] Image generated ({image.size[0]}x{image.size[1]})")

        # Save the generated image
        page_num = (panel_id - 1) // 4 + 1
        filename = f"panel_{panel_id:03d}_page_{page_num}.png"
        image_path = os.path.join(self.output_dir, filename)
        image.save(image_path)
        log.info(f"  Saved: {image_path}")

        # ── Step 6: Phase 2 Anchoring (if first panel) ──
        if panel_id == 1:
            char_name = list(self.memory.characters.keys())[0] if self.memory.characters else "Wanderer"
            self.anchor_system.establish_anchor(image, panel_id, char_name, self.memory)
            log.info(f"  [6] Anchor established from Panel 1")

        # ── Step 6b: Finalise advanced attention for this panel ──
        if self.advanced_attention:
            self.advanced_attention.on_panel_end()
            status = self.advanced_attention.get_status()
            log.info(
                f"  [6b] AdvAttn status — "
                f"L1-Heat active={status['L1_heat_diffusion']['alpha']}, "
                f"L2-Attn cached={status['L2_attention_cache']['layers_cached']} layers, "
                f"L3-STE anchor={status['L3_spatiotemporal']['anchor_captured']}"
            )

        # ── Step 7: Update memory ──
        elapsed = time.time() - start_time
        panel_record = PanelRecord(
            panel_id=panel_id,
            page_num=page_num,
            prompt_used=prompt[:500],
            emotion=context.get("panel_emotion_beat", "neutral"),
            dialogue=context.get("panel_dialogue", "..."),
            action_intensity=self._get_action_intensity(context),
            image_path=image_path,
        )
        self.memory.add_panel(panel_record)
        log.info(f"  [7] Memory updated (panel history: {self.memory.get_panel_count()})")
        log.info(f"  Generation time: {elapsed:.2f}s")

        return {
            "panel_id": panel_id,
            "page_num": page_num,
            "image": image,
            "image_path": image_path,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "generation_time": elapsed,
            "backend": backend.name,
            "weights": weights,
            "action_intensity": panel_record.action_intensity,
        }

    def generate_all_panels(self, coordinator: "AgentCoordinator",
                            style_prompt: str = "",
                            negative_base: str = "",
                            progress_callback=None) -> List[Dict[str, Any]]:
        """
        Generate all panels in sequence using the coordinator for context.

        Args:
            coordinator: The AgentCoordinator that provides generation contexts
            style_prompt: Base style prompt
            negative_base: Base negative prompt
            progress_callback: Optional callback(panel_id, total, result)

        Returns:
            List of panel result dicts
        """
        results = []
        total = self.memory.total_panels

        log.info(f"\n{'═'*60}")
        log.info(f"PANEL ENGINE: Generating {total} panels across {self.memory.total_pages} pages")
        log.info(f"{'═'*60}")

        for panel_id in range(1, total + 1):
            context = coordinator.get_generation_context(panel_id)
            result = self.generate_panel(panel_id, context, style_prompt, negative_base)
            results.append(result)

            # Notify agents of generation
            coordinator.notify_panel_generated(result)

            # Progress callback
            if progress_callback:
                progress_callback(panel_id, total, result)

        log.info(f"\n{'═'*60}")
        log.info(f"PANEL ENGINE COMPLETE: {len(results)} panels generated")
        total_time = sum(r.get("generation_time", 0) for r in results)
        log.info(f"Total generation time: {total_time:.1f}s")
        log.info(f"{'═'*60}")

        return results

    # ─────────────────────────────────────────────────────────────────────
    # Prompt Construction
    # ─────────────────────────────────────────────────────────────────────

    def _build_prompt(self, context: Dict[str, Any],
                      style_prompt: str = "") -> str:
        """
        Build the full generation prompt from context.

        Combines: style + character visual + scene atmosphere + action/emotion + motif
        """
        parts = []

        # Base styling
        if style_prompt:
            parts.append(style_prompt)
        else:
            parts.append("clean minimalist line art, flat color palette, "
                         "crisp continuous outlines, cel-shaded with no gradients")

        # Parse Scene Graph
        scene_graph = context.get("scene_graph", {})
        
        # 1. Environment and Camera
        if "camera" in scene_graph:
            cam = scene_graph["camera"]
            if isinstance(cam, dict):
                cam = ", ".join(f"{k}: {v}" for k, v in cam.items())
            elif isinstance(cam, list):
                cam = ", ".join(str(x) for x in cam)
            parts.append(str(cam))
            
        if "environment" in scene_graph:
            env = scene_graph["environment"]
            if isinstance(env, dict):
                env = ", ".join(f"{k}: {v}" for k, v in env.items())
            elif isinstance(env, list):
                env = ", ".join(str(x) for x in env)
            parts.append(str(env))
            
        # 2. Characters (Pose and Expression)
        for char in scene_graph.get("characters", []):
            char_id = char.get("id", "character")
            pose = char.get("pose", {})
            expr = char.get("expression", {})
            
            # Construct body pose string
            pose_str = f"{char_id} is {pose.get('body', 'standing')}, arms {pose.get('arms', 'relaxed')}, legs {pose.get('legs', 'normal')}, head {pose.get('head', 'forward')}."
            parts.append(pose_str)
            
            # Construct expression string
            expr_str = f"Facial expression: {expr.get('emotion', 'neutral')} (eyes: {expr.get('eyes', 'neutral')}, mouth: {expr.get('mouth', 'neutral')})."
            parts.append(expr_str)
            
        # 3. Actions
        for action in scene_graph.get("actions", []):
            act_str = f"{action.get('actor', '')} {action.get('verb', '')} {action.get('target', '')}."
            parts.append(act_str.strip())
            
        # 4. Recurring Motif
        motif = self.memory.recurring_motif
        if motif:
            parts.append(f"Recurring motif: {motif}")

        return ", ".join(p for p in parts if p)

    def _build_negative(self, context: Dict[str, Any],
                        negative_base: str = "") -> str:
        """Build the negative prompt."""
        negatives = [negative_base] if negative_base else [
            "photorealistic", "3D render", "shading", "gradients",
            "blurry", "messy lines", "extra fingers", "deformed face",
            "bad anatomy", "watermark", "text in image",
        ]
        return ", ".join(negatives)

    def _get_resolution(self, context: Dict[str, Any]) -> tuple:
        """Get resolution based on layout directive."""
        layout = context.get("layout", {})
        size_class = layout.get("size_class", "medium") if layout else "medium"

        if size_class == "full_page":
            return (1024, 1024)
        elif size_class == "large":
            return (768, 768)
        else:
            return (768, 768)

    def _get_action_intensity(self, context: Dict[str, Any]) -> float:
        """Extract action intensity from context."""
        layout = context.get("layout", {})
        if not layout:
            return 0.5

        size_map = {"small": 0.3, "medium": 0.5, "large": 0.7, "full_page": 0.9}
        return size_map.get(layout.get("size_class", "medium"), 0.5)

    def cleanup(self):
        """Cleanup hooks and cached VRAM tensors to prevent leaks."""
        if self.advanced_attention:
            self.advanced_attention.remove_hooks()
            self._hooks_installed = False
            log.info("PanelEngine cleaned up: hooks removed and cached VRAM cleared.")
