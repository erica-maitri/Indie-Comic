"""
PipelineLauncher — Phase 9 End-to-End Launcher
================================================
Unifies the three subsystems into a single call:

    Mood Weaver (classify)
        → EmotionRouter (arc routing)
            → Story Weaver (narrative script)  ← optional deep path
                → Indie Comic Pipeline (visual generation)

Typical usage::

    from integration.pipeline_launcher import PipelineLauncher

    launcher = PipelineLauncher(dry_run=True)
    result   = launcher.launch("I feel lost and tired today", panel_count=6)
    print(result["output_files"])   # list of generated files

Advanced — pass a pre-built Story Weaver script::

    from integration.pipeline_launcher import PipelineLauncher
    story_dict = {...}   # from stage2_story_generation.StoryGenerator
    result = launcher.launch_from_story_weaver(story_dict, panel_count=6)
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("integration.pipeline_launcher")

# Ensure parent directory is on path so indie_comic_pipeline can be imported
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_PIPELINE_ROOT = _REPO_ROOT / "indie_comic_pipeline"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from integration.emotion_router import EmotionRouter


# ---------------------------------------------------------------------------
# Story Weaver Bridge
# ---------------------------------------------------------------------------

class StoryWeaverBridge:
    """
    Wraps Story-Weaver/stage2_story_generation.StoryGenerator so that
    PipelineLauncher can call it with an EmotionOutput-style dict.

    The bridge converts between the two APIs:

        Mood Weaver  →  EmotionOutput dataclass  →  StoryGenerator.generate()
    """

    def __init__(self, model_key: str = "finetuned"):
        self.model_key = model_key
        self._generator = None

    def _get_generator(self):
        if self._generator is not None:
            return self._generator
        try:
            sw_path = _REPO_ROOT / "Story-Weaver"
            if str(sw_path) not in sys.path:
                sys.path.insert(0, str(sw_path))
            from stage2_story_generation import StoryGenerator  # type: ignore
            self._generator = StoryGenerator(model_key=self.model_key)
            log.info(f"[StoryWeaverBridge] Loaded StoryGenerator (model={self.model_key})")
        except Exception as e:
            log.warning(f"[StoryWeaverBridge] Could not load StoryGenerator: {e}")
        return self._generator

    def generate(self, story_context: Dict[str, Any],
                 panel_count: int = 6) -> Optional[Dict[str, Any]]:
        """
        Generate a story script from a story_context dict (as returned by
        EmotionRouter.full_pipeline()["story_context"]).

        Returns a StoryScript.to_dict() compatible dict, or None on failure.
        """
        gen = self._get_generator()
        if gen is None:
            log.warning("[StoryWeaverBridge] StoryGenerator unavailable — skipping")
            return None

        try:
            from stage2_story_generation import EmotionOutput  # type: ignore

            emotion_obj = EmotionOutput(
                primary_emotion    = story_context["primary_emotion"],
                confidence         = story_context["primary_confidence"],
                secondary_emotions = story_context.get("secondary_emotions", []),
                somatic_markers    = bool(story_context.get("somatic_markers")),
                user_text          = story_context["user_text"],
            )
            script = gen.generate(emotion_obj, panel_count=panel_count)
            return script.to_dict()
        except Exception as e:
            log.warning(f"[StoryWeaverBridge] Story generation failed: {e}")
            return None


# ---------------------------------------------------------------------------
# Story Script → StoryIntakeEngine Adapter
# ---------------------------------------------------------------------------

def _adapt_story_weaver_script(script: Dict[str, Any],
                                story_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a Stage-2 StoryScript dict into the rich format that
    StoryIntakeEngine / PanelEngine expects (panels with camera, environment, etc.).

    Stage-2 format::
        {
            "recurring_motif": "...",
            "mood_journey":    "...",
            "panels": [
                {"panel": 1, "visual": "...", "dialogue": "...",
                 "emotion_beat": "...", "motion": "..."},
                ...
            ],
            "_meta": { ... }
        }

    StoryIntakeEngine format (simplified, used as direct input to pipeline)::
        {
            "recurring_motif": "...",
            "mood_journey":    "...",
            "panels": [
                {
                    "panel": 1,
                    "emotion_beat": "...",
                    "characters": [{"id": "wanderer", "pose": {...}, "expression": {...},
                                    "dialogue": {"text": "..."}}],
                    "camera":      "...",
                    "environment": "...",
                }
            ],
            "_metadata": { ... }
        }
    """
    char_name = story_context.get("character_name", "Wanderer").lower()
    panels_out = []

    for raw in script.get("panels", []):
        # Build a simplified character entry from the Stage-2 fields
        dialogue_text = raw.get("dialogue", "...") or "..."
        panels_out.append({
            "panel":        raw["panel"],
            "emotion_beat": raw.get("emotion_beat", ""),
            "characters": [
                {
                    "id": char_name,
                    "pose": {
                        "body": raw.get("motion", "standing still"),
                        "head": "neutral",
                        "arms": "at sides",
                        "legs": "planted",
                    },
                    "expression": {
                        "emotion": raw.get("emotion_beat", "neutral"),
                        "eyes": "focused",
                        "mouth": "closed",
                    },
                    "dialogue": {
                        "text":   dialogue_text,
                        "tone":   "neutral",
                        "bubble": "speech" if dialogue_text != "..." else "silent",
                    },
                }
            ],
            "actions": [],
            "camera":      "Medium shot, static",
            "environment": raw.get("visual", story_context.get("visual_prompt", "")),
        })

    return {
        "recurring_motif": script.get("recurring_motif", ""),
        "mood_journey":    script.get("mood_journey", ""),
        "panels":          panels_out,
        "_metadata": {
            "emotion":   story_context["primary_emotion"],
            "character": story_context.get("character_name", "Wanderer"),
            "world":     story_context.get("character_world", "The Abstract"),
            "source":    "story_weaver_stage2",
            **script.get("_meta", {}),
        },
    }


# ---------------------------------------------------------------------------
# PipelineLauncher
# ---------------------------------------------------------------------------

class PipelineLauncher:
    """
    End-to-end launcher for the three-system pipeline.

    Parameters
    ----------
    dry_run : bool
        When True, uses MockBackend (no GPU needed). Useful for testing.
    story_weaver_model : str
        Model key passed to StoryWeaverBridge. Ignored when ``use_story_weaver``
        is False or the model cannot be loaded.
    emotion_router : EmotionRouter | None
        Pre-built router instance. Constructed automatically when None.
    mood_weaver_model_path : str
        Path to the Mood Weaver model directory.
    """

    def __init__(
        self,
        dry_run: bool = False,
        story_weaver_model: str = "finetuned",
        emotion_router: Optional[EmotionRouter] = None,
        mood_weaver_model_path: str = "./mood_weaver_model",
    ):
        self.dry_run = dry_run
        self.router  = emotion_router or EmotionRouter(model_path=mood_weaver_model_path)
        self.sw_bridge = StoryWeaverBridge(model_key=story_weaver_model)
        self._pipeline = None   # lazy-loaded IntegratedComicPipeline

    # ── Pipeline loading ───────────────────────────────────────────────────

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from integrated_pipeline import IntegratedComicPipeline  # type: ignore
            self._pipeline = IntegratedComicPipeline(dry_run=self.dry_run)
            log.info("[PipelineLauncher] IntegratedComicPipeline loaded")
        except Exception as e:
            log.error(f"[PipelineLauncher] Could not load IntegratedComicPipeline: {e}")
            raise
        return self._pipeline

    # ── Public launch methods ──────────────────────────────────────────────

    def launch(
        self,
        user_text: str,
        panel_count: int = 6,
        use_story_weaver: bool = False,
        style_reference: str = "",
        mood_shifts: Optional[List[str]] = None,
        override_character: Optional[str] = None,
        override_world: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline: text → emotion → arc routing → comic generation.

        Parameters
        ----------
        user_text : str
            Raw user input describing their emotional state or story idea.
        panel_count : int
            Number of comic panels (4–10).
        use_story_weaver : bool
            When True, runs Stage-2 StoryGenerator (LLM) before handing off
            to the Indie Comic Pipeline. When False, the Comic Pipeline's own
            StoryIntakeEngine handles narrative generation (Ollama / template).
        style_reference : str
            Optional comic style reference title (e.g. "Saga", "Akira").
        mood_shifts : list[str] | None
            Optional ordered list of emotion_beat values for custom arc override.
        override_character : str | None
            Override the auto-selected character name.
        override_world : str | None
            Override the auto-selected story world.

        Returns
        -------
        dict with keys:
            - ``"story_context"``  : EmotionRouter.full_pipeline() result
            - ``"story_script"``   : Stage-2 script dict (or None)
            - ``"pipeline_result"``: IntegratedComicPipeline.run() result
            - ``"output_files"``   : list of file paths created
        """
        log.info(f"[PipelineLauncher] Starting pipeline for: '{user_text[:80]}'")

        # Step 1: Classify emotion and route to arc
        log.info("[PipelineLauncher] Step 1 — Emotion classification + arc routing")
        routing = self.router.full_pipeline(user_text)
        ctx     = routing["story_context"]

        # Allow caller to override character/world
        char_name  = override_character or ctx["character_name"]
        story_world = override_world   or ctx["character_world"]

        story_script: Optional[Dict[str, Any]] = None

        # Step 2 (optional): Story Weaver Stage-2 LLM
        if use_story_weaver:
            log.info("[PipelineLauncher] Step 2 — Story Weaver Stage-2 generation")
            raw_script = self.sw_bridge.generate(ctx, panel_count=panel_count)
            if raw_script:
                story_script = _adapt_story_weaver_script(raw_script, ctx)
                log.info("[PipelineLauncher] Story Weaver script adapted for pipeline")
            else:
                log.warning("[PipelineLauncher] Story Weaver unavailable — falling back to pipeline intake")

        # Step 3: Run Indie Comic Pipeline
        log.info("[PipelineLauncher] Step 3 — Indie Comic Pipeline execution")
        pipeline = self._get_pipeline()

        if story_script:
            # Inject the pre-built script directly into the pipeline's story config
            result = pipeline.run(
                prompt              = user_text,
                character_name      = char_name,
                story_world         = story_world,
                panel_count         = panel_count,
                style_reference     = style_reference,
                character_characteristics = ctx.get("character_description", ""),
                story_reference     = ctx.get("arc_journey", ""),
                mood_shifts         = mood_shifts,
                _prebuilt_story     = story_script,   # passed through if pipeline supports it
            )
        else:
            result = pipeline.run(
                prompt              = user_text,
                character_name      = char_name,
                story_world         = story_world,
                panel_count         = panel_count,
                style_reference     = style_reference,
                character_characteristics = ctx.get("character_description", ""),
                story_reference     = ctx.get("arc_journey", ""),
                mood_shifts         = mood_shifts,
            )

        # Collect output files
        output_files: List[str] = []
        if isinstance(result, dict):
            for key in ("cbz_path", "cbr_path", "pdf_path", "output_path"):
                val = result.get(key)
                if val and Path(val).exists():
                    output_files.append(val)
            # Also look in exports list
            for f in result.get("exports", []):
                if isinstance(f, str) and Path(f).exists():
                    output_files.append(f)

        log.info(f"[PipelineLauncher] Pipeline complete. Files: {output_files}")
        return {
            "story_context":   routing,
            "story_script":    story_script,
            "pipeline_result": result,
            "output_files":    output_files,
        }

    def launch_from_story_weaver(
        self,
        story_dict: Dict[str, Any],
        panel_count: int = 6,
        character_name: Optional[str] = None,
        story_world: Optional[str] = None,
        style_reference: str = "",
    ) -> Dict[str, Any]:
        """
        Accept a pre-generated StoryScript dict (from stage2_story_generation)
        and feed it directly into the Indie Comic Pipeline.

        Useful when the caller already has a Story Weaver result and just wants
        to drive the visual generation step.

        Parameters
        ----------
        story_dict : dict
            A ``StoryScript.to_dict()`` result from Story Weaver Stage-2.
        panel_count : int
            Number of panels (validated against len(story_dict["panels"])).
        character_name : str | None
            Override character name. Defaults to "Wanderer".
        story_world : str | None
            Override story world. Defaults to "The Abstract".
        style_reference : str
            Optional comic style reference title.

        Returns
        -------
        Same structure as ``launch()``.
        """
        char_name   = character_name or "Wanderer"
        world_name  = story_world    or "The Abstract"
        actual_panels = len(story_dict.get("panels", []))
        if actual_panels:
            panel_count = actual_panels

        log.info(f"[PipelineLauncher] launch_from_story_weaver: {actual_panels} panels")

        # Build a minimal story_context from the script metadata
        meta = story_dict.get("_meta", {})
        ctx = {
            "user_text":             story_dict.get("mood_journey", ""),
            "primary_emotion":       meta.get("emotion", "sadness"),
            "primary_confidence":    1.0,
            "secondary_emotions":    [],
            "character_name":        char_name,
            "character_world":       world_name,
            "character_description": "",
            "arc_journey":           story_dict.get("mood_journey", ""),
        }

        adapted = _adapt_story_weaver_script(story_dict, ctx)
        pipeline = self._get_pipeline()

        result = pipeline.run(
            prompt                    = ctx["user_text"],
            character_name            = char_name,
            story_world               = world_name,
            panel_count               = panel_count,
            style_reference           = style_reference,
            character_characteristics = "",
            story_reference           = "",
        )

        output_files: List[str] = []
        if isinstance(result, dict):
            for key in ("cbz_path", "cbr_path", "pdf_path", "output_path"):
                val = result.get(key)
                if val and Path(val).exists():
                    output_files.append(val)
            for f in result.get("exports", []):
                if isinstance(f, str) and Path(f).exists():
                    output_files.append(f)

        return {
            "story_context":   {"story_context": ctx, "route": {}, "character": {}},
            "story_script":    adapted,
            "pipeline_result": result,
            "output_files":    output_files,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(
        description="PipelineLauncher — Mood Weaver → Story Weaver → Indie Comic"
    )
    parser.add_argument("prompt", nargs="?", default="I feel lost and exhausted",
                        help="Emotional text prompt")
    parser.add_argument("--panels",     type=int, default=6,      help="Panel count (4–10)")
    parser.add_argument("--dry-run",    action="store_true",       help="Use MockBackend (no GPU)")
    parser.add_argument("--use-sw",     action="store_true",       help="Use Story Weaver Stage-2")
    parser.add_argument("--sw-model",   default="tiny",            help="Story Weaver model key")
    parser.add_argument("--style",      default="",                help="Comic style reference")
    parser.add_argument("--route-only", action="store_true",       help="Only classify+route, no pipeline")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    router = EmotionRouter()

    if args.route_only:
        result = router.full_pipeline(args.prompt)
        print(json.dumps({
            "emotion":   result["emotion"],
            "arc_key":   result["route"]["arc_key"],
            "journey":   result["route"].get("journey", ""),
            "character": result["character"]["name"],
            "world":     result["character"]["world"],
            "uncertain": result["classification"]["uncertain"],
        }, indent=2))
        sys.exit(0)

    launcher = PipelineLauncher(
        dry_run             = args.dry_run,
        story_weaver_model  = args.sw_model,
        emotion_router      = router,
    )

    result = launcher.launch(
        user_text         = args.prompt,
        panel_count       = args.panels,
        use_story_weaver  = args.use_sw,
        style_reference   = args.style,
    )

    print("\n=== Pipeline Complete ===")
    ctx = result["story_context"]
    print(f"Emotion  : {ctx.get('emotion', 'N/A')}")
    route = ctx.get("route", {})
    print(f"Arc      : {route.get('arc_key', 'N/A')}  ({route.get('journey', '')})")
    char = ctx.get("character", {})
    print(f"Character: {char.get('name', 'N/A')} in {char.get('world', 'N/A')}")
    print(f"Files    : {result['output_files']}")
