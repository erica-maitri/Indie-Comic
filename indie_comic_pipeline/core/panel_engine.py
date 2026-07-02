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
import threading
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


# ─────────────────────────────────────────────────────────────────────────────
# Visual Language Lookup Tables
# ─────────────────────────────────────────────────────────────────────────────

# Maps emotion beats → concrete visual descriptors for image generation
EMOTION_VISUAL_MAP: Dict[str, Dict[str, str]] = {
    # Sad / Grief arc
    "heaviness":        {"lighting": "overcast diffused grey light, no shadows", "palette": "desaturated blue-grey tones", "atmosphere": "oppressive stillness, heavy air"},
    "stillness":        {"lighting": "flat ambient light, soft grey haze", "palette": "muted monochrome with faint blue", "atmosphere": "quiet suspended moment, held breath"},
    "faint_warmth":     {"lighting": "single candle glow from below, warm amber", "palette": "warm amber against cold blue", "atmosphere": "fragile emerging warmth"},
    "tentative_light":  {"lighting": "early dawn rim light from left", "palette": "pale gold and lavender", "atmosphere": "cautious hopeful opening"},
    "soft_openness":    {"lighting": "gentle diffused morning sunlight", "palette": "soft warm cream and peach", "atmosphere": "spacious emotional release"},
    "quiet_hope":       {"lighting": "golden hour backlight, warm halo", "palette": "warm gold, soft rose", "atmosphere": "serene forward motion"},
    # Angry arc
    "contained_fire":   {"lighting": "harsh high-contrast side-light, deep red tint", "palette": "crimson and near-black shadows", "atmosphere": "tightly coiled explosive tension"},
    "fracture":         {"lighting": "sharp overhead spotlight with cracks of red", "palette": "blood red and charcoal", "atmosphere": "controlled violence breaking through"},
    "exhale":           {"lighting": "dimming light, cooler tones emerging", "palette": "cooling orange transitioning to grey", "atmosphere": "post-explosion exhaustion"},
    "cooling":          {"lighting": "blue-grey diffused light", "palette": "slate blue and pale ash", "atmosphere": "tension draining, muscles releasing"},
    "ground":           {"lighting": "steady neutral warm ground-level light", "palette": "earthy brown and terracotta", "atmosphere": "feet on solid earth, centered"},
    # Tired / Exhausted arc
    "drag":             {"lighting": "flat fluorescent overhead, washing out color", "palette": "washed-out beige and grey", "atmosphere": "leaden heaviness in every motion"},
    "surrender":        {"lighting": "dim backlight silhouette", "palette": "dark grey and muted navy", "atmosphere": "body yielding to gravity"},
    "softness":         {"lighting": "low warm lamp-light from side", "palette": "warm muted ochre", "atmosphere": "gentle release, body softening"},
    "drift":            {"lighting": "soft unfocused haze, dreamlike blur", "palette": "lavender and pale grey", "atmosphere": "half-conscious floating"},
    "quiet_rest":       {"lighting": "deep blue moonlight through window", "palette": "deep indigo and silver", "atmosphere": "safe stillness, breath slowing"},
    "renewal":          {"lighting": "first light of dawn, golden streak", "palette": "gold and soft white", "atmosphere": "waking energy returning"},
    # Happy / Elation arc
    "spark":            {"lighting": "sharp bright point of light, electric", "palette": "vivid yellow and white", "atmosphere": "sudden electric joy"},
    "expansion":        {"lighting": "wide bright open sunlight", "palette": "bright cyan and warm yellow", "atmosphere": "opening chest, arms wide"},
    "overflow":         {"lighting": "saturated bright mid-day sun", "palette": "vivid saturated warm spectrum", "atmosphere": "overflowing emotion, barely contained"},
    "radiance":         {"lighting": "golden backlight halo, soft lens flare", "palette": "gold and luminous white", "atmosphere": "glowing inner light visible outward"},
    "luminous_still":   {"lighting": "soft internal glow, no hard shadows", "palette": "warm ivory and pale gold", "atmosphere": "deeply peaceful fullness"},
    "transcendence":    {"lighting": "white fill light, minimal shadow", "palette": "white, pale gold, ethereal", "atmosphere": "beyond physical, pure light"},
    # Anxious arc
    "spiral":           {"lighting": "harsh flickering fluorescent light", "palette": "sickly green-white and black", "atmosphere": "chaotic, closing walls, losing control"},
    "peak_noise":       {"lighting": "strobing high-contrast light", "palette": "high-contrast black and white", "atmosphere": "sensory overload, maximum tension"},
    "pause":            {"lighting": "single dim spotlight, total darkness around", "palette": "black with small white circle", "atmosphere": "the eye of the storm, held breath"},
    "breath":           {"lighting": "soft blue-white ambient", "palette": "cool pale blue", "atmosphere": "deliberate breath, recalibrating"},
    "root":             {"lighting": "warm steady ground-level light", "palette": "earthy terracotta and warm brown", "atmosphere": "feet solid, weight dropping"},
    "present":          {"lighting": "clear natural daylight", "palette": "natural warm neutral tones", "atmosphere": "clear-eyed awareness, grounded"},
    # Grief arc
    "absence":          {"lighting": "empty cold daylight, nothing warm", "palette": "cold white and pale grey", "atmosphere": "space where someone was, hollow"},
    "ache":             {"lighting": "dim lamp in dark room, pooled light", "palette": "amber pool against darkness", "atmosphere": "physical weight of missing someone"},
    "memory":           {"lighting": "warm faded sepia wash", "palette": "warm sepia and muted gold", "atmosphere": "past surfacing, bittersweet recall"},
    "held":             {"lighting": "gentle warm ambient, softened edges", "palette": "warm rose and soft amber", "atmosphere": "being held through pain"},
    "continuance":      {"lighting": "early morning, gently lit", "palette": "muted dawn tones", "atmosphere": "choosing to continue walking"},
    "carried_forward":  {"lighting": "soft warm walking light", "palette": "warm earth tones in motion", "atmosphere": "moving forward, weight transformed"},
    # Determined arc
    "doubt":            {"lighting": "shadow-heavy, uneven patches of light", "palette": "dark with uncertain patches of light", "atmosphere": "second-guessing, hesitation visible"},
    "challenge":        {"lighting": "dramatic side-key light, long shadows", "palette": "deep shadow and sharp highlight", "atmosphere": "obstacle looming large"},
    "resistance":       {"lighting": "hard directional light, gritted tones", "palette": "iron grey and deep blue", "atmosphere": "pushing against opposing force"},
    "breakthrough":     {"lighting": "light bursting through, explosive rim", "palette": "white-gold explosion of light", "atmosphere": "wall shattering, breakthrough moment"},
    "momentum":         {"lighting": "dynamic motion, directional speed light", "palette": "vivid warm red-orange and gold", "atmosphere": "unstoppable forward drive"},
    "triumph":          {"lighting": "full bright light, no shadow anywhere", "palette": "vibrant warm spectrum, pure colour", "atmosphere": "victory, full presence, arms raised"},
    # Love arc
    "recognition":      {"lighting": "soft warm backlight, gentle bokeh", "palette": "soft peach and warm gold", "atmosphere": "moment of truly seeing someone"},
    "vulnerability":    {"lighting": "low gentle lamplight, exposed warmth", "palette": "warm amber and soft pink", "atmosphere": "open and unguarded, tender"},
    "trust":            {"lighting": "shared warm light, equal bilateral", "palette": "matching warm tones, harmonious", "atmosphere": "mutual presence, safety established"},
    "embrace":          {"lighting": "enveloping warm light cocoon", "palette": "deep warm amber, enclosing", "atmosphere": "held and fully protected"},
    "unity":            {"lighting": "single unified light source on both figures", "palette": "harmonious warm spectrum", "atmosphere": "two becoming one shared presence"},
    # Generic fallbacks
    "neutral":          {"lighting": "balanced natural three-point light", "palette": "neutral natural tones", "atmosphere": "clean, clear, present moment"},
    "resolution":       {"lighting": "soft closing warm golden light", "palette": "muted warm finale tones", "atmosphere": "settling, completing, at peace"},
    "acknowledgment":   {"lighting": "steady natural front light", "palette": "grounded neutral tones", "atmosphere": "honest, direct, witnessing"},
    "presence":         {"lighting": "clear even warm light", "palette": "clean balanced tones", "atmosphere": "fully here, attentive"},
    "shift":            {"lighting": "transitional changing light between two tones", "palette": "blending between two palettes", "atmosphere": "pivoting moment, change in motion"},
    "openness":         {"lighting": "expanding warm open light", "palette": "open airy warm tones", "atmosphere": "receptive, available, welcoming"},
}

# Named art style presets for image generation
STYLE_PRESETS: Dict[str, str] = {
    "indie_comic":
        "indie graphic novel illustration, clean confident ink outlines, "
        "flat cel-shaded color, expressive characters, professional panel composition, "
        "graphic novel art quality",
    "manga":
        "detailed manga illustration, precise technical linework, "
        "screentone shading, dynamic speed lines, expressive character design, "
        "professional Japanese manga art",
    "noir_comic":
        "film noir graphic novel style, heavy black ink, hard chiaroscuro shadows, "
        "stark high-contrast lighting, Edward Hopper color palette, Raymond Chandler atmosphere, "
        "dark expressionist comic art",
    "watercolor_indie":
        "hand-painted watercolor comic illustration, loose expressive brushwork, "
        "bleed edges, organic paper texture, muted earth palette, editorial illustration quality",
    "ghibli":
        "Studio Ghibli animation art style, lush detailed painterly backgrounds, "
        "soft warm color palette, highly expressive character design, "
        "whimsical atmospheric illustration",
    "moebius":
        "Moebius Jean Giraud bande dessinee style, ultra-fine hatched linework, "
        "pastel desert color palette, vast architectural sci-fi landscapes, "
        "European comics tradition",
    "lofi_zine":
        "DIY photocopied zine aesthetic, raw expressive linework, "
        "limited 2-color risograph palette, imperfect textures, underground comics energy",
    "superhero":
        "classic superhero comic book art, dynamic action poses, bold primary colors, "
        "heavy ink outlines, Jim Lee composition, dramatic foreshortening, "
        "professional Marvel DC art quality",
    "painterly":
        "fully painted comic illustration, oil paint texture, rich chiaroscuro, "
        "cinematic composition, realistic proportions, masterwork painting quality",
    "retro_60s":
        "1960s vintage comic book style, Ben-Day dot printing texture, bold flat colors, "
        "Pop Art palette, retro halftone printing, classic American comics",
    "default":
        "professional indie comic illustration, confident ink outlines, "
        "flat clean color palette, expressive character art, "
        "cinematic panel composition, award-winning graphic novel quality",
}

# Panel position grammar — what narrative role does this panel play?
PANEL_POSITION_MODIFIERS: Dict[str, str] = {
    "opening":
        "establishing wide shot, atmospheric scene-setter, "
        "world-building composition, introduces tone and location, "
        "cinematic first-impression panel",
    "early":
        "character introduction panel, intimate establishing composition, "
        "world and character grounding, reader orientation",
    "middle_early":
        "rising tension panel, situation developing, "
        "stakes becoming clear, engagement building",
    "midpoint":
        "emotional pivot panel, the turning point, "
        "revelation or confrontation building, maximum mid-story tension",
    "middle_late":
        "confrontation panel, direct conflict or emotional peak, "
        "consequences manifesting, high-drama composition",
    "climax":
        "climax panel, maximum dramatic intensity, "
        "largest most dynamic composition, most saturated emotional moment, "
        "explosive visual language",
    "resolution":
        "resolution panel, emotional landing, "
        "quieting composition, the breath after the storm, "
        "soft concluding image",
    "coda":
        "final coda panel, epilogue closing image, "
        "forward-looking gentle composition, lasting final impression",
}

# Camera angle visual descriptors
CAMERA_VISUAL_MAP: Dict[str, str] = {
    "close_up":         "extreme close-up, face filling frame, intense emotional detail, shallow depth of field",
    "medium_shot":      "medium shot, waist-up framing, conversational distance",
    "wide_shot":        "wide establishing shot, full character in environment, cinematic landscape",
    "bird_eye":         "bird's eye overhead view, character small in vast space, environmental dominance",
    "low_angle":        "low angle upshot, character dominant and powerful, dramatic foreshortening",
    "dutch_tilt":       "Dutch tilt off-axis framing, psychological unease, unstable world",
    "over_shoulder":    "over-the-shoulder shot, two-character dynamic, POV intimacy",
}

# Quality booster suffix — appended to every prompt
QUALITY_BOOSTERS = (
    "masterwork comic illustration, award-winning graphic novel art, "
    "perfect panel composition, professional coloring, high detail, "
    "8k resolution comic art"
)


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
                 output_dir: str = "outputs/panels",
                 style_preset: str = "default"):
        self.memory = memory
        self.backend_selector = backend_selector
        self.compositor = compositor or CharComCompositor()
        self.anchor_system = anchor_system or ReferenceFreeAnchor()
        self.advanced_attention = advanced_attention  # None = disabled
        self.output_dir = output_dir
        self.style_preset = style_preset
        self._prompt_optimizer = None
        self._hooks_installed = False
        self._lock = threading.Lock()

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def generate_panel(self, panel_id: int,
                       context: Dict[str, Any],
                       style_prompt: str = "",
                       negative_base: str = "") -> Dict[str, Any]:
        """Generate a single panel using the full Phase 2-4 pipeline with thread safety."""
        with self._lock:
            return self._generate_panel_locked(panel_id, context, style_prompt, negative_base)

    def _generate_panel_locked(self, panel_id: int,
                               context: Dict[str, Any],
                               style_prompt: str = "",
                               negative_base: str = "") -> Dict[str, Any]:
        """
        Generate a single panel using the full Phase 2-4 pipeline.

        Args:
            panel_id: Panel number (1-indexed)
            context: Generation context from AgentCoordinator.get_generation_context()
            style_prompt: Base style prompt string (overrides preset if given)
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
        if self.advanced_attention and not self._hooks_installed:
            # Check if backend provides cross-attention modules list directly (Backend Adapter Pattern)
            modules = getattr(backend, "get_cross_attention_modules", lambda: [])()
            if modules:
                self.advanced_attention.install_on_modules(modules)
                self._hooks_installed = True
                log.info(f"  [3b] Advanced attention hooks installed on {len(modules)} modules (Backend Adapter)")
            elif hasattr(backend, "get_raw_pipeline"):
                pipe = backend.get_raw_pipeline()
                if pipe is not None:
                    self.advanced_attention.install_on_pipeline(pipe)
                    self._hooks_installed = True
                    log.info("  [3b] Advanced attention hooks installed on UNet (Fallback)")

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
        page_num = self.memory.get_page_num(panel_id)
        filename = f"panel_{panel_id:03d}_page_{page_num}.png"
        image_path = os.path.join(self.output_dir, filename)
        image.save(image_path)
        log.info(f"  Saved: {image_path}")

        # ── Step 6: Phase 2 Anchoring (if first panel) ──
        if panel_id == 1:
            char_name = self.memory.main_character or (list(self.memory.characters.keys())[0] if self.memory.characters else "Wanderer")
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

        log.info(f"\n{'='*60}")
        log.info(f"PANEL ENGINE: Generating {total} panels across {self.memory.total_pages} pages")
        log.info(f"{'='*60}")

        for panel_id in range(1, total + 1):
            context = coordinator.get_generation_context(panel_id)
            result = self.generate_panel(panel_id, context, style_prompt, negative_base)
            results.append(result)

            # Notify agents of generation
            coordinator.notify_panel_generated(result)

            # Progress callback
            if progress_callback:
                progress_callback(panel_id, total, result)

        log.info(f"\n{'='*60}")
        log.info(f"PANEL ENGINE COMPLETE: {len(results)} panels generated")
        total_time = sum(r.get("generation_time", 0) for r in results)
        log.info(f"Total generation time: {total_time:.1f}s")
        log.info(f"{'='*60}")

        return results

    # ─────────────────────────────────────────────────────────────────────
    # Prompt Construction
    # ─────────────────────────────────────────────────────────────────────

    def _build_prompt(self, context: Dict[str, Any],
                      style_prompt: str = "") -> str:
        """
        Build a rich, cinematic generation prompt from context.

        Combines: style preset + panel position grammar + character visual +
        scene atmosphere + emotion lighting/palette + action + motif + quality boosters.
        """
        parts = []

        # ── 1. Style Foundation ──
        if style_prompt:
            parts.append(style_prompt)
        else:
            style_ref = context.get("style_reference", "")
            if style_ref:
                # User provided an explicit style reference (e.g. "Moebius", "Ghibli")
                parts.append(f"{style_ref} art style, {STYLE_PRESETS['default']}")
            else:
                parts.append(STYLE_PRESETS.get(self.style_preset, STYLE_PRESETS["default"]))

        # ── 2. Panel Narrative Position ──
        panel_id = context.get("panel_id", 1)
        total_panels = context.get("total_panels", 1)
        position_key = self._get_panel_position_key(panel_id, total_panels)
        position_modifier = PANEL_POSITION_MODIFIERS.get(position_key, "")
        if position_modifier:
            parts.append(position_modifier)

        # ── 3. Emotion → Visual Mood Translation ──
        beat = context.get("panel_emotion_beat", "neutral")
        visual = EMOTION_VISUAL_MAP.get(beat, EMOTION_VISUAL_MAP["neutral"])
        parts.append(visual["lighting"])
        parts.append(visual["palette"])
        parts.append(visual["atmosphere"])

        # ── 4. Scene Graph — Camera ──
        scene_graph = context.get("scene_graph", {})

        if "camera" in scene_graph:
            cam = scene_graph["camera"]
            if isinstance(cam, dict):
                cam = ", ".join(f"{k}: {v}" for k, v in cam.items())
            elif isinstance(cam, list):
                cam = ", ".join(str(x) for x in cam)
            cam_str = str(cam)
            # Enrich with camera visual language if we recognise the angle keyword
            for angle_key, angle_desc in CAMERA_VISUAL_MAP.items():
                if angle_key.replace("_", " ") in cam_str.lower():
                    cam_str = angle_desc
                    break
            parts.append(cam_str)

        # ── 5. Scene Graph — Environment ──
        if "environment" in scene_graph:
            env = scene_graph["environment"]
            if isinstance(env, dict):
                env = ", ".join(f"{k}: {v}" for k, v in env.items())
            elif isinstance(env, list):
                env = ", ".join(str(x) for x in env)
            parts.append(str(env))

        # ── 6. Characters (Pose, Expression, Costume) ──
        for char in scene_graph.get("characters", []):
            char_id = char.get("id", "character")
            pose = char.get("pose", {})
            expr = char.get("expression", {})

            # Pull costume from memory for visual consistency
            char_state = (self.memory.get_character(char_id)
                          or self.memory.get_character(char_id.capitalize()))
            costume = ""
            if char_state and char_state.costume_desc:
                costume = f" wearing {char_state.costume_desc},"

            pose_str = (
                f"{char_id}{costume} "
                f"{pose.get('body', 'standing')}, "
                f"arms {pose.get('arms', 'relaxed')}, "
                f"head {pose.get('head', 'forward')}"
            )
            parts.append(pose_str)

            expr_str = (
                f"facial expression: {expr.get('emotion', beat)}, "
                f"eyes {expr.get('eyes', 'forward-looking')}, "
                f"mouth {expr.get('mouth', 'neutral')}"
            )
            parts.append(expr_str)

        # ── 7. Actions ──
        for action in scene_graph.get("actions", []):
            act_str = (
                f"{action.get('actor', '')} "
                f"{action.get('verb', '')} "
                f"{action.get('target', '')}."
            )
            parts.append(act_str.strip())

        # ── 8. Recurring Motif ──
        motif = self.memory.recurring_motif
        if motif:
            parts.append(f"visible recurring motif in scene: {motif}")

        # ── 9. Story/Style Reference as Visual Direction ──
        story_ref = context.get("story_reference", "")
        if story_ref:
            parts.append(f"visual direction inspired by {story_ref}")

        # ── 10. Quality Boosters ──
        parts.append(QUALITY_BOOSTERS)

        return ", ".join(p.strip() for p in parts if p.strip())

    def _build_negative(self, context: Dict[str, Any],
                        negative_base: str = "") -> str:
        """Build a rich, style-aware and emotion-aware negative prompt."""
        if negative_base:
            return negative_base

        # Base quality negatives — universal
        negatives = [
            "photorealistic", "3D render", "CGI", "photograph", "realistic skin texture",
            "blurry", "out of focus", "motion blur",
            "extra fingers", "deformed hands", "bad anatomy", "missing limbs",
            "deformed face", "crossed eyes", "ugly", "disfigured", "mutated",
            "watermark", "signature", "text overlay", "caption",
            "low quality", "jpeg artifacts", "noise", "grain",
            "amateur drawing", "poorly drawn", "sketch quality",
            "multiple panels in one image", "panel borders in image",
        ]

        # Style-specific negatives
        preset = self.style_preset
        if preset in ("manga", "indie_comic"):
            negatives += ["gradients", "airbrushed", "photoshop glow", "painterly"]
        elif preset == "noir_comic":
            negatives += ["bright cheerful colors", "saturated palette", "flat lighting", "comedy"]
        elif preset == "watercolor_indie":
            negatives += ["hard ink lines", "cel shading", "flat vector art", "digital clean"]
        elif preset == "ghibli":
            negatives += ["dark horror", "violence", "gritty", "realistic", "ugly"]
        elif preset == "painterly":
            negatives += ["flat color", "cel shading", "line art", "cartoon outline"]
        elif preset == "superhero":
            negatives += ["realistic", "photographic", "dark muted colors", "static pose"]

        # Emotion-specific negatives
        beat = context.get("panel_emotion_beat", "neutral")
        if beat in ("triumph", "breakthrough", "overflow", "radiance", "luminous_still"):
            negatives += ["dark", "gloomy", "sad", "depressing", "muted colors"]
        elif beat in ("contained_fire", "fracture", "spiral", "peak_noise"):
            negatives += ["calm", "peaceful", "soft", "pastel colors", "cheerful"]
        elif beat in ("stillness", "quiet_rest", "drift", "absence"):
            negatives += ["busy background", "chaotic", "high contrast", "intense action"]
        elif beat in ("heaviness", "drag", "surrender"):
            negatives += ["bright", "colorful", "vivid", "cheerful", "uplifting"]

        return ", ".join(negatives)

    def _get_panel_position_key(self, panel_id: int, total_panels: int) -> str:
        """Map panel_id to narrative position key."""
        if total_panels <= 1:
            return "opening"
        ratio = (panel_id - 1) / (total_panels - 1)
        if ratio == 0.0:
            return "opening"
        elif ratio <= 0.2:
            return "early"
        elif ratio <= 0.4:
            return "middle_early"
        elif ratio <= 0.55:
            return "midpoint"
        elif ratio <= 0.7:
            return "middle_late"
        elif ratio <= 0.85:
            return "climax"
        elif ratio < 1.0:
            return "resolution"
        else:
            return "coda"

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
