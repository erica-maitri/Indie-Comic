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
from core.feedback_tuner import HeuristicFeedbackTuner
from core.advanced_attention import AdvancedAttentionManager
from comic_exporter import ComicExporter


class MockBackend(BaseBackend):
    """
    Mock Backend used in dry-run mode.
    Renders deterministic placeholder PIL images with prompt text overlay.
    """
    def __init__(self):
        self._loaded = False

    @property
    def name(self) -> str:
        return "Mock"

    @property
    def supports_lora(self) -> bool:
        return False

    def load(self, config: Dict[str, Any]):
        self._loaded = True

    def generate(self, prompt: str, negative_prompt: str,
                 config: Dict[str, Any]) -> Image.Image:
        width = config.get("width", 768)
        height = config.get("height", 768)
        seed = config.get("seed", 42)
        
        # Deterministic color background based on seed
        import random
        random.seed(seed)
        r = random.randint(50, 180)
        g = random.randint(50, 180)
        b = random.randint(50, 180)
        
        image = Image.new("RGB", (width, height), color=(r, g, b))
        
        # Draw placeholder text on the image
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        text = f"MOCK PANEL\nSeed: {seed}\nPrompt: {prompt[:60]}..."
        draw.text((20, 20), text, fill=(255, 255, 255))
        return image

    def unload(self):
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded


class IntegratedComicPipeline:
    """The master pipeline orchestrator for the Ultimate AI Indie Comic Generator."""
    
    def __init__(self, model_override: Optional[str] = None, skip_backends: bool = False, dry_run: bool = False):
        self.dry_run = dry_run
        import torch
        
        # Enable PyTorch TF32 & Matmul Precision GPU Acceleration (additive optimization)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision('high')
                log.info("[Pipeline] Enabled TF32 matmul and cudnn acceleration with 'high' precision.")
            except Exception as e:
                log.warning(f"[Pipeline] Failed to set matmul precision: {e}")

        if not skip_backends and not self.dry_run and not torch.cuda.is_available():
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
        if not skip_backends and not self.dry_run:
            log.info("Initializing GPU Model Backends...")
            self.backend_selector.initialize_backends(self.settings.get("models", {}))
        else:
            log.info("Skipping GPU Model Backends initialization (rebuild/editing or dry-run mode)...")
            if self.dry_run:
                log.info("Registering MockBackend for dry-run...")
                mock = MockBackend()
                self.backend_selector.register_backend("sdxl", mock)
                self.backend_selector.register_backend("flux", mock)
            
        # ── Advanced Attention Manager (L1 + L2 + L3 mechanisms) ──
        # Enabled for real generation.
        adv_attn_enabled = True
        mdcp_conf = self.settings.get("mdcp", {})
        self.advanced_attention = AdvancedAttentionManager(
            heat_alpha=0.03,           # L1: heat diffusion strength
            attention_blend=mdcp_conf.get("beta", 0.15),      # L2: anchor K/V blend ratio
            spatial_strength=0.08,     # L3: spatiotemporal correction strength
            enabled=adv_attn_enabled,
            # MDCP Core Hyperparameters
            lam1=mdcp_conf.get("lambda_1", 1.0),
            lam2=mdcp_conf.get("lambda_2", 1.0),
            lam3=mdcp_conf.get("lambda_3", 1.0),
            omega=mdcp_conf.get("omega", 0.50)
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
        self.feedback_tuner = HeuristicFeedbackTuner(
            feedback_loop=self.feedback_loop,
            settings_path=os.path.join(PROJECT_ROOT, "config", "settings.yaml")
        )
        
        self.exporter = ComicExporter(output_dir=self.output_dir)
        self._preheat_thread = None
        self._export_thread = None

    def wait_for_preheat(self):
        """Wait for the background model preheating thread to complete if it is running."""
        if getattr(self, "_preheat_thread", None) is not None:
            if self._preheat_thread.is_alive():
                log.info("[Pipeline] Waiting for background model preheating to complete...")
                start_time = time.time()
                self._preheat_thread.join()
                log.info(f"[Pipeline] Background preheating completed. Waited {time.time() - start_time:.2f}s.")
            self._preheat_thread = None

    def wait_for_export(self):
        """Wait for the background export thread to complete if it is running."""
        if getattr(self, "_export_thread", None) is not None:
            if self._export_thread.is_alive():
                log.info("[Pipeline] Waiting for background export to complete...")
                start_time = time.time()
                self._export_thread.join()
                log.info(f"[Pipeline] Background export completed. Waited {time.time() - start_time:.2f}s.")
            self._export_thread = None

    def _run_phase8_export(self, pages: list, prompt: str):
        """Runs Phase 8 export in a background thread."""
        try:
            log.info("\n--- Phase 8: Exporting Formats in Background ---")
            self.exporter.export_cbz(pages, title=prompt[:30])
            self.exporter.export_web_comic(pages, os.path.join(self.output_dir, "web_comic.html"))
            self.exporter.export_pdf(pages, title=prompt[:30])
            log.info("--- Phase 8: Background Exporting Complete ---")
        except Exception as e:
            log.error(f"Error in background Phase 8 export: {e}")

    def _generate_panels_in_parallel(self, pids: List[int], checkpoint_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate a list of panels in parallel with a Memory Budgeter and OOM recovery."""
        import concurrent.futures
        import gc
        import torch
        
        if not pids:
            return []
            
        panels_completed = []
        total_panels = len(pids)
        max_workers = min(4, total_panels)
        
        # Pre-compute memory cost and adjust initial max_workers
        try:
            if torch.cuda.is_available():
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**2) # in MB
                # Cost per panel: ~6.5GB base
                safe_limit = total_mem * 0.90
                cost_per_panel = 6500.0
                if getattr(self.panel_engine.advanced_attention, "enabled", False):
                    cost_per_panel += 1500.0 # T1 backward pass overhead
                
                calculated_max = int(safe_limit // cost_per_panel)
                calculated_max = max(1, min(calculated_max, 4))
                if calculated_max < max_workers:
                    log.info(f"Memory Budgeter: Reduced initial max_workers from {max_workers} to {calculated_max} based on VRAM ({total_mem:.0f}MB, limit {safe_limit:.0f}MB)")
                    max_workers = calculated_max
        except Exception as e:
            log.debug(f"Memory Budgeter pre-compute failed: {e}")

        pids_to_generate = list(pids)
        while pids_to_generate:
            log.info(f"Starting parallel execution chunk with max_workers={max_workers} for panels {pids_to_generate}")
            oom_occurred = False
            
            def _gen_task(pid):
                return self._generate_single_panel_with_retry(pid)
                
            completed_in_this_run = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit only the first batch of size max_workers to prevent launching too many tasks
                batch_pids = pids_to_generate[:max_workers]
                future_to_pid = {executor.submit(_gen_task, pid): pid for pid in batch_pids}
                
                for future in concurrent.futures.as_completed(future_to_pid):
                    pid = future_to_pid[future]
                    try:
                        res = future.result()
                        panels_completed.append(res)
                        self.agent_coordinator.notify_panel_generated(res)
                        completed_in_this_run.append(pid)
                        
                        # Save checkpoint if requested
                        if checkpoint_path:
                            self.memory.save_checkpoint(checkpoint_path)
                            log.info(f"Saved mid-generation checkpoint for panel {pid} to: {checkpoint_path}")
                    except Exception as exc:
                        exc_str = str(exc).lower()
                        if "out of memory" in exc_str or "oom" in exc_str:
                            log.warning(f"CUDA Out of Memory caught during panel {pid} generation: {exc}")
                            oom_occurred = True
                        else:
                            log.error(f"Panel {pid} generation generated an exception: {exc}")
                            raise exc
                            
            # Update list of panels to generate
            for pid in completed_in_this_run:
                if pid in pids_to_generate:
                    pids_to_generate.remove(pid)
                    
            if oom_occurred:
                # Clear GPU memory cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                if max_workers > 1:
                    new_workers = max(1, max_workers // 2)
                    log.info(f"OOM Occurred. Halving max_workers from {max_workers} to {new_workers} and retrying remaining panels {pids_to_generate}")
                    max_workers = new_workers
                else:
                    log.error("OOM occurred even with max_workers=1. Cannot reduce further.")
                    raise RuntimeError("CUDA Out of Memory occurred during sequential generation. Exiting.")
                    
        return panels_completed

    def _generate_single_panel_with_retry(self, panel_id: int) -> Dict[str, Any]:
        """Generate a single panel, execute the reject-regenerate loop, and apply typesetting."""
        context = self.agent_coordinator.get_generation_context(panel_id)
        scene_graph = context.get("scene_graph", {})
    
        dialogue = context.get("panel_dialogue", "...")
        emotion = context.get("panel_emotion_beat", "neutral")
        scene_desc = scene_graph.get("environment", "")
        
        retry = 0
        max_retries = self.quality_critic.max_retries
        panel_result = None
    
        while retry <= max_retries:
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
                adjusts = evaluation.get("adjustments", {})
                if "guidance_scale_delta" in adjusts:
                    context["guidance_scale_override"] = 7.5 + adjusts["guidance_scale_delta"]
                if "steps_delta" in adjusts:
                    context["steps_override"] = 25 + adjusts["steps_delta"]
                
        if not panel_result:
            raise RuntimeError(f"Failed to generate panel {panel_id} after {max_retries} retries.")
                
        log.info(f"  Phase 5: Overlaying text on Panel {panel_id}")
        speaker_pos = "center"
        if context.get("scene_graph", {}).get("characters"):
            speaker_pos = context["scene_graph"]["characters"][0].get("position", "center")
        
        # Keep a copy of the raw generated image for dynamic layout typesetting
        raw_img = panel_result.get("raw_image") or panel_result.get("image")
        if raw_img is None:
            raise ValueError(f"No image was generated for panel {panel_id}")
        panel_result["raw_image"] = raw_img
        
        final_img = self.text_integrator.integrate(
            image=raw_img,
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
        
        return panel_result

    def run(self, prompt: str, character_name: str = "Wanderer",
            story_world: str = "The Abstract", panel_count: int = 4,
            style_reference: str = "", character_characteristics: str = "",
            story_reference: str = "", mood_shifts: Optional[List[str]] = None,
            _prebuilt_story: Optional[Dict[str, Any]] = None,
            weave_mood: bool = False,
            story_mode: str = "literal",
            **_kwargs) -> Dict[str, Any]:
        """Runs the entire 8-phase comic generation pipeline.

        Parameters
        ----------
        _prebuilt_story : dict, optional
            A pre-built story config (from PipelineLauncher / StoryWeaverBridge)
            that bypasses Phase 0 (StoryIntakeEngine). Must conform to the
            StoryIntakeEngine output format (has ``"panels"``, ``"recurring_motif"``).
        story_mode : str, default "literal"
            "literal" makes `prompt` the primary structural driver — panels are
            the story split into panel_count sequential moments, and detected
            emotion only shades tone/lighting per panel. "mood_arc" restores the
            legacy behaviour where a fixed generic emotional-arc template
            dictates each panel's beat and `prompt` is passed as background
            context only.
        **_kwargs
            Extra keyword arguments are silently ignored for forward compatibility.
        """
        try:
            # Reset memory blackboard for a fresh run
            self.memory = StorySectionMemory()
            self.agent_coordinator.memory = self.memory
            self.panel_engine.memory = self.memory

            log.info("=" * 80)
            log.info("Starting Ultimate Indie Comic Generator Pipeline")
            log.info("=" * 80)

            # Start background preheating thread if backends are registered
            self._preheat_thread = None
            if getattr(self.backend_selector, "_backends", {}):
                import threading
                log.info("[Pipeline] Starting background model preheating thread...")
                self._preheat_thread = threading.Thread(
                    target=self.backend_selector.select,
                    args=({"layout": {"size_class": "medium", "camera_angle": "medium_shot"}},),
                    daemon=True
                )
                self._preheat_thread.start()

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
                    weave_mood=weave_mood,
                    story_mode=story_mode
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
            
            panels_completed = []
            total_panels = self.memory.total_panels
        
            # Panel 1 (Anchor) must run first to establish consistency priors
            log.info("\n--- Phase 2: Anchor Panel Generation (Sequential) ---")
            self.wait_for_preheat()
            panel_1_result = self._generate_single_panel_with_retry(1)
            panels_completed.append(panel_1_result)
            self.agent_coordinator.notify_panel_generated(panel_1_result)
            
            # Save mid-generation checkpoint for panel 1
            checkpoint_name = "storyboard_checkpoint_latest.json"
            checkpoint_path = os.path.join(self.output_dir, checkpoint_name)
            self.memory.save_checkpoint(checkpoint_path)
            log.info(f"Saved mid-generation checkpoint for panel 1 to: {checkpoint_path}")
            
            # Generate remaining panels (2 to N) in parallel
            if total_panels > 1:
                log.info(f"\n--- Generating Remaining {total_panels - 1} Panels in Parallel ---")
                remaining_results = self._generate_panels_in_parallel(list(range(2, total_panels + 1)), checkpoint_path=checkpoint_path)
                panels_completed.extend(remaining_results)
            
            # Sort panels by ID to restore sequential order
            panels_completed.sort(key=lambda x: x["panel_id"])
            
            # Clean up hooks and cached VRAM tensors
            self.panel_engine.cleanup()
            
            # Explicit VRAM Garbage Collection & Defragmentation (additive optimization)
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                log.info("[Pipeline] Cleared PyTorch CUDA cache and defragmented VRAM.")
            
            # ── Phase 7: MangaFlow Page Assembly ──
            log.info("\n--- Phase 7: MangaFlow Layout Page Assembly ---")
            pages = []
        
            # Group panels by page
            panels_by_page = {}
            for p in panels_completed:
                page_num = p["page_num"]
                panels_by_page.setdefault(page_num, []).append(p)
            
            for page_num, page_panels in sorted(panels_by_page.items()):
                page_image = self.layout_engine.layout_page(
                    page_panels, page_num, text_integrator=self.text_integrator
                )
            
                # Save page layouts conforming to the naming pattern expected by compile_comic_pdf.py
                page_path = os.path.join(self.output_dir, f"page_{page_num:03d}_layout_integrated.png")
                page_image.save(page_path)
                log.info(f"Saved assembled Page {page_num} to: {page_path}")
            
                pages.append({
                    "page_num": page_num,
                    "page_image": page_image,
                    "panels": page_panels
                })
            
            # ── Phase 8: Multi-Format Export (Asynchronous) ──
            # Precompute paths to return immediately
            title = prompt[:30]
            safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
            safe_title = safe_title.replace(" ", "_")
            cbz_path = os.path.join(self.output_dir, f"{safe_title}.cbz")
            pdf_path = os.path.join(self.output_dir, f"{safe_title}.pdf")
            html_path = os.path.join(self.output_dir, "web_comic.html")
            
            # Start background thread to run Phase 8
            import threading
            self._export_thread = threading.Thread(
                target=self._run_phase8_export,
                args=(pages, prompt),
                daemon=True
            )
            self._export_thread.start()
        
            return {
                "pages": pages,
                "cbz_path": cbz_path,
                "html_path": html_path,
                "pdf_path": pdf_path,
                "panels": panels_completed
            }

        finally:
            self.wait_for_preheat()
            log.info("Ensuring GPU resources are cleaned up (unload all backends)...")
            self.backend_selector.unload_all()

    def run_batch(self, start_panel: int, end_panel: int, prompt: str = "", character_name: str = "Wanderer",
                  story_world: str = "The Abstract", panel_count: int = 4,
                  style_reference: str = "", character_characteristics: str = "",
                  story_reference: str = "", mood_shifts: Optional[List[str]] = None,
                  load_checkpoint: str = "", save_checkpoint: str = "",
                  story_mode: str = "literal") -> Dict[str, Any]:
        """Runs the pipeline in chunks, allowing for pause/resume via JSON checkpointing."""
        log.info("=" * 80)
        log.info(f"Starting Batch Pipeline (Panels {start_panel} to {end_panel})")
        log.info("=" * 80)

        # Start background preheating thread if backends are registered
        self._preheat_thread = None
        if getattr(self.backend_selector, "_backends", {}):
            import threading
            log.info("[Pipeline] Starting background model preheating thread (batch)...")
            self._preheat_thread = threading.Thread(
                target=self.backend_selector.select,
                args=({"layout": {"size_class": "medium", "camera_angle": "medium_shot"}},),
                daemon=True
            )
            self._preheat_thread.start()

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
                story_reference=story_reference, mood_shifts=mood_shifts,
                story_mode=story_mode
            )
            log.info("\n--- Phase 1: Multi-Agent Planning ---")
            self.agent_coordinator.run_planning(story_config)

        panels_completed = []
        actual_end = min(end_panel, self.memory.total_panels)

        log.info(f"\n--- Generating Panels {start_panel} to {actual_end} ---")
        start_idx = start_panel
        # Panel 1 (Anchor) must run first if within range
        if start_panel == 1:
            log.info("\n--- Phase 2: Anchor Panel Generation (Sequential) ---")
            self.wait_for_preheat()
            panel_1_result = self._generate_single_panel_with_retry(1)
            panels_completed.append(panel_1_result)
            self.agent_coordinator.notify_panel_generated(panel_1_result)
            start_idx = 2
            
        # Generate remaining panels in parallel
        if actual_end >= start_idx:
            log.info(f"\n--- Generating Remaining Panels {start_idx} to {actual_end} in Parallel ---")
            self.wait_for_preheat()
            remaining_results = self._generate_panels_in_parallel(list(range(start_idx, actual_end + 1)))
            panels_completed.extend(remaining_results)
            
        panels_completed.sort(key=lambda x: x["panel_id"])
            
        # Assemble Layouts for the generated panels
        pages = []
        panels_by_page = {}
        for p in panels_completed:
            panels_by_page.setdefault(p["page_num"], []).append(p)
            
        for page_num, page_panels in sorted(panels_by_page.items()):
            page_image = self.layout_engine.layout_page(
                page_panels, page_num, text_integrator=self.text_integrator
            )
            page_path = os.path.join(self.output_dir, f"page_{page_num:03d}_batch_integrated.png")
            page_image.save(page_path)
            pages.append({"page_num": page_num, "page_image": page_image, "panels": page_panels})

        if save_checkpoint:
            self.memory.save_checkpoint(save_checkpoint)
            log.info(f"Memory state saved to {save_checkpoint}")

        self.wait_for_preheat()
        self.backend_selector.unload_all()
        return {"pages": pages, "panels": panels_completed, "last_panel_generated": actual_end}

    def collect_interactive_feedback(self, run_results: Dict[str, Any], no_feedback: bool = False):
        """Collect rating and comments from the user for RLHF tracking."""
        if no_feedback:
            log.info("RLHF feedback collection skipped.")
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
            adjusts = self.feedback_tuner.tune_from_feedback()
            if adjusts.get("quality_critic_threshold_delta", 0.0) != 0.0:
                log.info(f"Recommendation: Adjust quality threshold by {adjusts['quality_critic_threshold_delta']}")
            if adjusts.get("lora_scale_adjustment", 0.0) != 0.0:
                log.info(f"Recommendation: Adjust LoRA scale by {adjusts['lora_scale_adjustment']}")
                
            # Apply weight adjustments and mutate prompt templates
            if self.feedback_tuner.apply_optimizations(adjusts):
                log.info("[OK] Optimization adjustments successfully applied to configuration!")
            else:
                log.info("Optimization adjustments evaluated, no config updates needed.")
                
        except (KeyboardInterrupt, EOFError):
            print("\nFeedback collection skipped.")

    def rebuild_comic(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Rebuilds the comic page layouts and multi-format exports from the latest panel images
        and their layout JSON configurations. This allows updating bubble coordinates
        without re-generating the images.
        """
        import glob
        
        if not checkpoint_path:
            checkpoint_path = os.path.join(self.output_dir, "storyboard_checkpoint_latest.json")
            if not os.path.exists(checkpoint_path):
                checkpoint_path = os.path.join(self.output_dir, "storyboard_plan.json")
            
        if os.path.exists(checkpoint_path):
            log.info(f"Rebuilding comic: Loading memory checkpoint from {checkpoint_path}")
            try:
                self.memory = StorySectionMemory.load_checkpoint(checkpoint_path)
                self.agent_coordinator.memory = self.memory
                self.panel_engine.memory = self.memory
            except Exception as e:
                log.warning(f"Failed to load checkpoint: {e}")
            
        panels_completed = []
        total_panels = self.memory.total_panels
        
        # If total_panels is 0 or empty, try to deduce from raw files
        if total_panels == 0:
            raw_files = glob.glob(os.path.join(self.panels_dir, "panel_*_page_*.png"))
            total_panels = len(raw_files)
            self.memory.total_panels = total_panels
            
        log.info(f"Re-integrating text and bubbles for {total_panels} panels...")
        for panel_id in range(1, total_panels + 1):
            raw_pattern = os.path.join(self.panels_dir, f"panel_{panel_id:03d}_page_*.png")
            matches = glob.glob(raw_pattern)
            if not matches:
                log.warning(f"Raw image for panel {panel_id} not found. Skipping re-render.")
                continue
            raw_path = matches[0]
            
            try:
                base = os.path.basename(raw_path)
                parts = base.replace(".png", "").split("_page_")
                page_num = int(parts[1]) if len(parts) > 1 else 1
            except Exception:
                page_num = self.memory.get_page_num(panel_id) or 1
                
            raw_img = Image.open(raw_path)
            
            context = {}
            if self.agent_coordinator is not None:
                try:
                    context = self.agent_coordinator.get_generation_context(panel_id)
                except Exception:
                    pass
            
            dialogue = context.get("panel_dialogue", "...")
            emotion = context.get("panel_emotion_beat", "neutral")
            scene_desc = context.get("scene_graph", {}).get("environment", "")
            speaker_pos = "center"
            if context.get("scene_graph", {}).get("characters"):
                speaker_pos = context["scene_graph"]["characters"][0].get("position", "center")
                
            json_filename = f"panel_{panel_id:03d}_bubble_layout.json"
            json_path = os.path.join(self.panels_dir, json_filename)
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        plan = json.load(f)
                    dialogue_clean = plan.get("dialogue_clean", "")
                    speaker = plan.get("speaker", "")
                    if dialogue_clean:
                        if speaker:
                            dialogue = f"{speaker}: {dialogue_clean}"
                        else:
                            dialogue = dialogue_clean
                except Exception as e:
                    log.warning(f"Error reading layout json: {e}")

            final_img = self.text_integrator.integrate(
                image=raw_img,
                dialogue=dialogue,
                emotion_beat=emotion,
                speaker_position=speaker_pos,
                panel_id=panel_id,
                scene_desc=scene_desc
            )
            
            annotated_filename = f"panel_{panel_id:03d}_final.png"
            annotated_path = os.path.join(self.panels_dir, annotated_filename)
            final_img.save(annotated_path)
            
            panel_result = {
                "panel_id": panel_id,
                "page_num": page_num,
                "image": final_img,
                "image_path": annotated_path,
                "dialogue": dialogue,
                "emotion_beat": emotion
            }
            panels_completed.append(panel_result)
            
        # Re-assemble pages
        pages = []
        panels_by_page = {}
        for p in panels_completed:
            page_num = p["page_num"]
            panels_by_page.setdefault(page_num, []).append(p)
            
        for page_num, page_panels in sorted(panels_by_page.items()):
            page_image = self.layout_engine.layout_page(page_panels, page_num)
            page_path = os.path.join(self.output_dir, f"page_{page_num:03d}_layout_integrated.png")
            page_image.save(page_path)
            log.info(f"Saved assembled Page {page_num} to: {page_path}")
            
            pages.append({
                "page_num": page_num,
                "page_image": page_image,
                "panels": page_panels
            })
            
        # Multi-Format Export
        cbz_path = self.exporter.export_cbz(pages, title="Rebuilt Comic")
        html_path = self.exporter.export_web_comic(pages, os.path.join(self.output_dir, "web_comic.html"))
        pdf_path = self.exporter.export_pdf(pages, title="Rebuilt Comic")
        
        return {
            "pages": pages,
            "cbz_path": cbz_path,
            "html_path": html_path,
            "pdf_path": pdf_path,
            "panels": panels_completed
        }


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
    parser.add_argument("--story-mode", type=str, default="literal", choices=["literal", "mood_arc"],
                        help="'literal' (default) adapts your --prompt panel by panel, preserving its "
                             "characters/events. 'mood_arc' uses the legacy generic emotional-arc template.")
                        
    args = parser.parse_args()
    
    pipeline = IntegratedComicPipeline(model_override=args.model, dry_run=args.dry_run)
    
    results = pipeline.run(
        prompt=args.prompt,
        character_name=args.character,
        story_world=args.world,
        panel_count=args.panels,
        weave_mood=args.weave_mood,
        story_mode=args.story_mode
    )
    
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE!")
    print("=" * 70)
    print(f"CBZ Export: {results['cbz_path']}")
    print(f"HTML scrollbook: {results['html_path']}")
    print(f"PDF document: {results['pdf_path']}")
    print("=" * 70)
    
    pipeline.collect_interactive_feedback(results, no_feedback=args.no_feedback)
    pipeline.wait_for_export()


if __name__ == "__main__":
    main()