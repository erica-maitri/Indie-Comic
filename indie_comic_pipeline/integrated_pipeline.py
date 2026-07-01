"""
INTEGRATED PIPELINE — 8-Phase Master Orchestrator
================================================
Entry point that replaces the older sequential pipeline with the full
integrated methodology. Executes:
- Phase 0: Story Intake Engine (Ollama/template fallback)
- Phase 1: Multi-Agent Planning (blackboard pattern)
- Phase 2: Reference-Free Anchoring
- Phase 3-4: Unified Generation Loop (compositor, backend selection)
- Phase 5: Text-Image Integration (DiffSensei layout planner via Ollama & local JSON)
- Phase 6: Quality Critic Loop (reject-and-regenerate validation)
- Phase 7: MangaFlow Layout Engine (dynamic pacing layout)
- Phase 8: Multi-format Export & RLHF Telemetry Loop
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, Any, List, Optional
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("pipeline.orchestrator")

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# Load environment variables from .env file if it exists at the repo root
try:
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(PROJECT_ROOT), ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
except Exception:
    pass

# Suppress Hugging Face/Transformers tokenization warnings
try:
    import logging as py_logging
    py_logging.getLogger("transformers.tokenization_utils_base").setLevel(py_logging.ERROR)
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    # Set Hugging Face verbosity
    from transformers.utils import logging as tf_logging
    tf_logging.set_verbosity_error()
except Exception:
    pass

from utils.config_helper import load_settings, get_output_path
from core.memory import StorySectionMemory, PanelRecord
from core.story_intake import StoryIntakeEngine
from core.agents.agent_coordinator import AgentCoordinator
from core.backends.backend_selector import BackendSelector
from core.backends.base_backend import BaseBackend
from core.panel_engine import PanelEngine
from core.quality_critic import QualityCritic
from core.text_image_integrator import TextImageIntegrator
from core.layout_engine import MangaFlowLayoutEngine
from core.feedback import RLHFFeedbackLoop
from core.optimizer import SystemOptimizer
from core.advanced_attention import AdvancedAttentionManager
from comic_exporter import ComicExporter


class IntegratedComicPipeline:
    """The master pipeline orchestrator for the Ultimate AI Indie Comic Generator."""
    
    def __init__(self, model_override: Optional[str] = None):
        import torch
        if not torch.cuda.is_available():
            log.error("❌ CRITICAL ERROR: CUDA GPU is not available! The pipeline has been configured to run ONLY on GPU (dry-run and mock modes are disabled). Please enable GPU acceleration in your Kaggle/Colab notebook settings.")
            raise RuntimeError("❌ CRITICAL ERROR: CUDA GPU is not available!")

        from utils.config_helper import load_env_with_defaults
        from typing import Optional
        env_defaults = load_env_with_defaults()
        
        self.settings = load_settings()
        if not self.settings:
            self.settings = {}
            
        # Initialize memory
        self.memory = StorySectionMemory()
        
        # Initialize layout directories
        self.output_dir = self.settings.get("outputs", {}).get("comics_dir", "outputs/comics")
        self.panels_dir = "outputs/panels"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.panels_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize sub-components
        if "langchain" not in self.settings:
            self.settings["langchain"] = {}
        langchain_conf = self.settings["langchain"]
        
        ollama_model = model_override or langchain_conf.get("model") or os.environ.get("OLLAMA_MODEL") or env_defaults.get("llm_provider", "llama3.2")
        self.story_intake = StoryIntakeEngine(
            ollama_model=ollama_model,
            ollama_url=langchain_conf.get("ollama_url", env_defaults.get("ollama_url", "http://localhost:11434"))
        )
        
        self.agent_coordinator = AgentCoordinator(self.memory)
        
        # Choose backend configuration
        self.backend_selector = BackendSelector()
        log.info("Initializing GPU Model Backends...")
        self.backend_selector.initialize_backends(self.settings.get("models", {}))
            
        # ── Advanced Attention Manager (L1 + L2 + L3 mechanisms) ──
        # Enabled for real generation.
        adv_attn_enabled = True
        self.advanced_attention = AdvancedAttentionManager(
            heat_alpha=0.03,           # L1: heat diffusion strength
            attention_blend=0.15,      # L2: anchor K/V blend ratio
            spatial_strength=0.08,     # L3: spatiotemporal correction strength
            enabled=adv_attn_enabled
        )
        if adv_attn_enabled:
            log.info("Advanced Attention Mechanisms ENABLED (L1-Heat, L2-Attn, L3-STE)")

        # Panel Generation Engine
        self.panel_engine = PanelEngine(
            memory=self.memory,
            backend_selector=self.backend_selector,
            advanced_attention=self.advanced_attention,
            output_dir=self.panels_dir
        )
        
        # Quality control critic
        critic_conf = self.settings.get("consistency", {})
        self.quality_critic = QualityCritic(
            threshold=critic_conf.get("threshold", 0.55),
            strict_threshold=critic_conf.get("strict_threshold", 0.70),
            max_retries=2
        )
        
        # Text-Image Integrator (DiffSensei approximation)
        self.text_integrator = TextImageIntegrator(
            output_dir=self.panels_dir,
            ollama_model=langchain_conf.get("model", env_defaults.get("llm_provider", "llama3.2")),
            ollama_url=langchain_conf.get("ollama_url", env_defaults.get("ollama_url", "http://localhost:11434"))
        )
        
        # MangaFlow Layout Engine
        self.layout_engine = MangaFlowLayoutEngine(
            page_width=1000,
            page_height=1500,
            gutter_width=12,
            margin=40
        )
        
        # Feedback & Optimizer
        self.feedback_loop = RLHFFeedbackLoop(
            feedback_path=os.path.join(self.output_dir, "rlhf_feedback.json")
        )
        self.optimizer = SystemOptimizer(
            feedback_loop=self.feedback_loop,
            settings_path=os.path.join(PROJECT_ROOT, "config", "settings.yaml")
        )
        
        self.exporter = ComicExporter(output_dir=self.output_dir)

    def run(self, prompt: str, character_name: str = "Wanderer",
            story_world: str = "The Abstract", panel_count: int = 4,
            style_reference: str = "", character_characteristics: str = "",
            story_reference: str = "", mood_shifts: Optional[List[str]] = None,
            _prebuilt_story: Optional[Dict[str, Any]] = None,
            weave_mood: bool = False,
            **_kwargs) -> Dict[str, Any]:
        """Runs the entire 8-phase comic generation pipeline.

        Parameters
        ----------
        _prebuilt_story : dict, optional
            A pre-built story config (from PipelineLauncher / StoryWeaverBridge)
            that bypasses Phase 0 (StoryIntakeEngine). Must conform to the
            StoryIntakeEngine output format (has ``"panels"``, ``"recurring_motif"``).
        **_kwargs
            Extra keyword arguments are silently ignored for forward compatibility.
        """
        # Reset memory blackboard for a fresh run
        self.memory = StorySectionMemory()
        self.agent_coordinator.memory = self.memory
        self.panel_engine.memory = self.memory

        log.info("=" * 80)
        log.info("Starting Ultimate Indie Comic Generator Pipeline")
        log.info("=" * 80)

        # ── Phase 0: Story Intake (skipped when a pre-built story is supplied) ──
        if _prebuilt_story is not None:
            log.info("\n--- Phase 0: Story Intake [BYPASSED — using pre-built story] ---")
            story_config = _prebuilt_story
            # Patch panel_count from pre-built script if available
            if "panels" in story_config:
                panel_count = len(story_config["panels"])
                log.info(f"[Phase 0] Using {panel_count} panels from pre-built story")
        else:
            log.info("\n--- Phase 0: Story Intake ---")
            story_config = self.story_intake.process_prompt(
                user_prompt=prompt,
                panel_count=panel_count,
                character_name=character_name,
                story_world=story_world,
                style_reference=style_reference,
                character_characteristics=character_characteristics,
                story_reference=story_reference,
                mood_shifts=mood_shifts,
                weave_mood=weave_mood
            )

        # Validate that story configuration has correct layout/format
        if not story_config or "panels" not in story_config:
            raise ValueError("Story intake failed to return a valid story configuration with panels.")
        
        # ── Phase 1: Multi-Agent Planning ──
        log.info("\n--- Phase 1: Multi-Agent Planning ---")
        self.agent_coordinator.run_planning(story_config)
        log.info(f"Loaded emotional pacing arc beats: {self.memory.arc_beats}")
        
        # Save story plan overview
        plan_path = os.path.join(self.output_dir, "storyboard_plan.json")
        self.memory.save_checkpoint(plan_path)
        log.info(f"Storyboard plan saved to: {plan_path}")
        
        # Iterate over all panels and generate with quality checks
        panels_completed = []
        total_panels = self.memory.total_panels
        
        log.info(f"\n--- Phases 2-6: Generation & Quality Control Loops ({total_panels} Panels) ---")
        for panel_id in range(1, total_panels + 1):
            context = self.agent_coordinator.get_generation_context(panel_id)
            scene_graph = context.get("scene_graph", {})
            
            dialogue = context.get("panel_dialogue", "...")
            emotion = context.get("panel_emotion_beat", "neutral")
            scene_desc = scene_graph.get("environment", "")
            # Reject & Regenerate Quality loop
            retry = 0
            max_retries = self.quality_critic.max_retries
            panel_result = None
            
            while retry <= max_retries:
                # Compile parameters and adjust based on critic deltas
                style_prompt = ""
                negative_base = ""
                
                # Generate image
                panel_result = self.panel_engine.generate_panel(
                    panel_id=panel_id,
                    context=context,
                    style_prompt=style_prompt,
                    negative_base=negative_base
                )
                
                # Evaluate panel quality
                evaluation = self.quality_critic.evaluate(panel_result, self.memory)
                
                if not self.quality_critic.should_regenerate(evaluation) or self.dry_run:
                    log.info(f"  Panel {panel_id} PASSED quality critic on try {retry + 1}")
                    break
                else:
                    retry += 1
                    log.warning(f"  Panel {panel_id} FAILED quality critic. Verdict: {evaluation['verdict']}. Retrying ({retry}/{max_retries})...")
                    # Adjust parameters for the next try based on critic recommendations
                    adjusts = evaluation.get("adjustments", {})
                    if "guidance_scale_delta" in adjusts:
                        context["guidance_scale_override"] = 7.5 + adjusts["guidance_scale_delta"]
                    if "steps_delta" in adjusts:
                        context["steps_override"] = 25 + adjusts["steps_delta"]
                        
            if not panel_result:
                raise RuntimeError(f"Failed to generate panel {panel_id} after {max_retries} retries.")
                        
            # Once accepted, run Phase 5: Text-Image bubble overlay
            log.info(f"  Phase 5: Overlaying text on Panel {panel_id}")
            speaker_pos = "center"
            if context.get("scene_graph", {}).get("characters"):
                speaker_pos = context["scene_graph"]["characters"][0].get("position", "center")
                
            final_img = self.text_integrator.integrate(
                image=panel_result["image"],
                dialogue=dialogue,
                emotion_beat=emotion,
                speaker_position=speaker_pos,
                panel_id=panel_id,
                scene_desc=scene_desc
            )
            
            # Save visual output to disk and replace image in result
            annotated_filename = f"panel_{panel_id:03d}_final.png"
            annotated_path = os.path.join(self.panels_dir, annotated_filename)
            final_img.save(annotated_path)
            
            panel_result["image"] = final_img
            panel_result["image_path"] = annotated_path
            panel_result["dialogue"] = dialogue
            panel_result["emotion_beat"] = emotion
            panel_result["page_num"] = self.memory.get_page_num(panel_id)
            
            panels_completed.append(panel_result)
            self.agent_coordinator.notify_panel_generated(panel_result)
            
            # Save checkpoint to prevent memory loss
            checkpoint_name = "storyboard_checkpoint_latest.json"
            checkpoint_path = os.path.join(self.output_dir, checkpoint_name)
            self.memory.save_checkpoint(checkpoint_path)
            log.info(f"Saved mid-generation checkpoint for panel {panel_id} to: {checkpoint_path}")
            
        # Clean up hooks and cached VRAM tensors
        self.panel_engine.cleanup()
            
        # ── Phase 7: MangaFlow Page Assembly ──
        log.info("\n--- Phase 7: MangaFlow Layout Page Assembly ---")
        pages = []
        
        # Group panels by page
        panels_by_page = {}
        for p in panels_completed:
            page_num = p["page_num"]
            panels_by_page.setdefault(page_num, []).append(p)
            
        for page_num, page_panels in sorted(panels_by_page.items()):
            page_image = self.layout_engine.layout_page(page_panels, page_num)
            
            # Save page layouts conforming to the naming pattern expected by compile_comic_pdf.py
            page_path = os.path.join(self.output_dir, f"page_{page_num:03d}_layout_integrated.png")
            page_image.save(page_path)
            log.info(f"Saved assembled Page {page_num} to: {page_path}")
            
            pages.append({
                "page_num": page_num,
                "page_image": page_image,
                "panels": page_panels
            })
            
        # ── Phase 8: Multi-Format Export ──
        log.info("\n--- Phase 8: Exporting Formats ---")
        cbz_path = self.exporter.export_cbz(pages, title=prompt[:30])
        html_path = self.exporter.export_web_comic(pages, os.path.join(self.output_dir, "web_comic.html"))
        pdf_path = self.exporter.export_pdf(pages, title=prompt[:30])
            
        # Unload model selector backends to save GPU memory
        self.backend_selector.unload_all()
        
        return {
            "pages": pages,
            "cbz_path": cbz_path,
            "html_path": html_path,
            "pdf_path": pdf_path,
            "panels": panels_completed
        }

    def run_batch(self, start_panel: int, end_panel: int, prompt: str = "", character_name: str = "Wanderer",
                  story_world: str = "The Abstract", panel_count: int = 4,
                  style_reference: str = "", character_characteristics: str = "",
                  story_reference: str = "", mood_shifts: Optional[List[str]] = None,
                  load_checkpoint: str = "", save_checkpoint: str = "") -> Dict[str, Any]:
        """Runs the pipeline in chunks, allowing for pause/resume via JSON checkpointing."""
        log.info("=" * 80)
        log.info(f"Starting Batch Pipeline (Panels {start_panel} to {end_panel})")
        log.info("=" * 80)

        if load_checkpoint and os.path.exists(load_checkpoint):
            log.info(f"Loading memory checkpoint from {load_checkpoint}")
            self.memory = StorySectionMemory.load_checkpoint(load_checkpoint)
            self.agent_coordinator.memory = self.memory
            self.panel_engine.memory = self.memory
        else:
            # ── Phase 0 & 1 ──
            log.info("\n--- Phase 0: Story Intake ---")
            story_config = self.story_intake.process_prompt(
                user_prompt=prompt, panel_count=panel_count, character_name=character_name,
                story_world=story_world, style_reference=style_reference,
                character_characteristics=character_characteristics,
                story_reference=story_reference, mood_shifts=mood_shifts
            )
            log.info("\n--- Phase 1: Multi-Agent Planning ---")
            self.agent_coordinator.run_planning(story_config)

        panels_completed = []
        actual_end = min(end_panel, self.memory.total_panels)

        log.info(f"\n--- Generating Panels {start_panel} to {actual_end} ---")
        for panel_id in range(start_panel, actual_end + 1):
            context = self.agent_coordinator.get_generation_context(panel_id)
            dialogue = context.get("panel_dialogue", "...")
            emotion = context.get("panel_emotion_beat", "neutral")
            
            retry = 0
            max_retries = self.quality_critic.max_retries
            panel_result = None
            
            while retry <= max_retries:
                panel_result = self.panel_engine.generate_panel(
                    panel_id=panel_id, context=context, style_prompt="", negative_base=""
                )
                evaluation = self.quality_critic.evaluate(panel_result, self.memory)
                if not self.quality_critic.should_regenerate(evaluation) or self.dry_run:
                    break
                else:
                    retry += 1
                    adjusts = evaluation.get("adjustments", {})
                    if "guidance_scale_delta" in adjusts:
                        context["guidance_scale_override"] = 7.5 + adjusts["guidance_scale_delta"]
                    if "steps_delta" in adjusts:
                        context["steps_override"] = 25 + adjusts["steps_delta"]
            
            if not panel_result:
                raise RuntimeError(f"Failed to generate panel {panel_id}")
                        
            final_img = self.text_integrator.integrate(
                image=panel_result["image"], dialogue=dialogue, emotion_beat=emotion,
                panel_id=panel_id, scene_desc=context.get("panel_visual")
            )
            
            annotated_path = os.path.join(self.panels_dir, f"panel_{panel_id:03d}_final.png")
            final_img.save(annotated_path)
            panel_result["image"] = final_img
            panel_result["image_path"] = annotated_path
            panel_result["dialogue"] = dialogue
            panel_result["emotion_beat"] = emotion
            
            panels_completed.append(panel_result)
            self.agent_coordinator.notify_panel_generated(panel_result)
            
        # Assemble Layouts for the generated panels
        pages = []
        panels_by_page = {}
        for p in panels_completed:
            panels_by_page.setdefault(p["page_num"], []).append(p)
            
        for page_num, page_panels in sorted(panels_by_page.items()):
            page_image = self.layout_engine.layout_page(page_panels, page_num)
            page_path = os.path.join(self.output_dir, f"page_{page_num:03d}_batch_integrated.png")
            page_image.save(page_path)
            pages.append({"page_num": page_num, "page_image": page_image, "panels": page_panels})

        if save_checkpoint:
            self.memory.save_checkpoint(save_checkpoint)
            log.info(f"Memory state saved to {save_checkpoint}")

        self.backend_selector.unload_all()
        return {"pages": pages, "panels": panels_completed, "last_panel_generated": actual_end}

    def collect_interactive_feedback(self, run_results: Dict[str, Any], no_feedback: bool = False):
        """Collect rating and comments from the user for RLHF tracking."""
        if self.dry_run or no_feedback:
            log.info("RLHF skipped in dry-run mode or non-interactive environment")
            return

        print("\n" + "=" * 70)
        print("INTERACTIVE RLHF TELEMETRY COLLECTOR")
        print("=" * 70)
        print("Provide your ratings (1-5 stars) to guide the system optimization.")
        
        try:
            # Rate panels
            for p in run_results["panels"]:
                p_id = p["panel_id"]
                prompt = p["prompt"][:60]
                backend = p["backend"]
                
                print(f"\n[Panel {p_id}] Prompt: '{prompt}...' (Backend: {backend})")
                rating_str = input("Rate this panel (1-5 stars, enter to skip): ").strip()
                if rating_str.isdigit():
                    rating = int(rating_str)
                    comment = input("Any comments on this panel? ").strip()
                    self.feedback_loop.add_panel_feedback(
                        panel_id=p_id,
                        rating=rating,
                        comment=comment,
                        prompt_used=p["prompt"],
                        generation_backend=backend
                    )

            # Auto-train user preference model if sufficient feedback exists
            try:
                from core.user_preference_critic import UserPreferenceCritic
                pref_critic = UserPreferenceCritic(
                    model_path=self.quality_critic.user_pref_model_path
                )
                pref_critic.train_from_feedback_file(
                    feedback_file=self.feedback_loop.feedback_path,
                    panels_dir=self.panels_dir,
                    min_records=3
                )
            except Exception as e:
                log.warning(f"Failed to auto-train user preference critic: {e}")

            # Suggest updates
            log.info("RLHF entries logged. Suggesting pipeline optimizations...")
            adjusts = self.optimizer.optimize_system_parameters()
            if adjusts.get("quality_critic_threshold_delta", 0.0) != 0.0:
                log.info(f"Recommendation: Adjust quality threshold by {adjusts['quality_critic_threshold_delta']}")
            if adjusts.get("lora_scale_adjustment", 0.0) != 0.0:
                log.info(f"Recommendation: Adjust LoRA scale by {adjusts['lora_scale_adjustment']}")
                
            # Apply weight adjustments and mutate prompt templates
            if self.optimizer.apply_optimizations(adjusts):
                log.info("[OK] Optimization adjustments successfully applied to configuration!")
            else:
                log.info("Optimization adjustments evaluated, no config updates needed.")
                
        except (KeyboardInterrupt, EOFError):
            print("\nFeedback collection skipped.")


def main():
    parser = argparse.ArgumentParser(description="Integrated AI Comic Generator — 8-Phase Pipeline")
    parser.add_argument("--prompt", type=str, default="A lone wanderer discovers hope",
                        help="Raw narrative or emotional story prompt")
    parser.add_argument("--character", type=str, default="Wanderer",
                        help="Main character name")
    parser.add_argument("--world", type=str, default="The Abstract",
                        help="Story world / setting")
    parser.add_argument("--panels", type=int, default=4,
                        help="Number of panels to generate (typically 4 or 8)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use mock images instead of diffusion generation (saves GPU VRAM)")
    parser.add_argument("--no-feedback", action="store_true",
                        help="Skip interactive RLHF collection at the end")
                        
    parser.add_argument("--weave-mood", action="store_true",
                        help="Enable Mood Weaver mode: auto-detect emotion, map character archetype, and select a random franchise setting style")
    parser.add_argument("--model", type=str, default=None,
                        help="Override default Ollama model (e.g. qwen2.5, mistral, llama3.2)")
                        
    args = parser.parse_args()
    
    pipeline = IntegratedComicPipeline(model_override=args.model)
    
    results = pipeline.run(
        prompt=args.prompt,
        character_name=args.character,
        story_world=args.world,
        panel_count=args.panels,
        weave_mood=args.weave_mood
    )
    
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE!")
    print("=" * 70)
    print(f"CBZ Export: {results['cbz_path']}")
    print(f"HTML scrollbook: {results['html_path']}")
    print(f"PDF document: {results['pdf_path']}")
    print("=" * 70)
    
    pipeline.collect_interactive_feedback(results, no_feedback=args.no_feedback)


if __name__ == "__main__":
    main()
