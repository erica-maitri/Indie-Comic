"""
STORY INTAKE ENGINE — Phase 0
================================
Accepts raw user emotion/story prompt, processes it through the
Story-Weaver LLM to parse thematic elements, emotional pacing,
and structural flow. Outputs a structured story configuration.

Uses local Ollama for LLM inference.
Arc definitions are loaded from config/arcs_config.json (shared
source of truth with Story Weaver); inline dict is the fallback.
"""

import json
import os
import sys
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

log = logging.getLogger("pipeline.story_intake")


def _load_arcs_config() -> dict:
    """Load the shared arcs_config.json. Returns {} on any failure."""
    candidates = [
        Path(__file__).parent.parent / "config" / "arcs_config.json",
        Path(__file__).parent / "config" / "arcs_config.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8-sig") as f:
                    cfg = json.load(f)
                log.info(f"[Phase 0] Loaded arc config from {p}")
                return cfg
            except Exception as e:
                log.warning(f"[Phase 0] Could not parse arcs_config.json at {p}: {e}")
    log.debug("[Phase 0] arcs_config.json not found — using inline MOOD_ARCS fallback")
    return {}


def _build_mood_arcs(cfg: dict) -> dict:
    """Convert arcs_config.json into MOOD_ARCS format expected by StoryIntakeEngine."""
    result = {}
    mood_to_arc = cfg.get("mood_to_arc", {})
    for _label, entry in mood_to_arc.items():
        arc_key = entry.get("arc_key")
        if not arc_key:
            continue  # surprise uses fallback arc; no key to register
        result[arc_key] = {
            "journey":     entry.get("journey", ""),
            "description": entry.get("description", ""),
            "arc_beats":   entry.get("arc_beats", []),
        }
    # tired is a top-level key (not inside mood_to_arc)
    if "tired" in cfg:
        t = cfg["tired"]
        result["tired"] = {
            "journey":     t.get("journey", ""),
            "description": t.get("description", ""),
            "arc_beats":   t.get("arc_beats", []),
        }
    return result


_ARCS_CONFIG = _load_arcs_config()
_config_arcs = _build_mood_arcs(_ARCS_CONFIG) if _ARCS_CONFIG else {}


MOOD_ARCS = {
    "sad": {
        "journey": "uplifting",
        "description": "From heaviness toward genuine small warmth",
        "arc_beats": ["heaviness", "stillness", "faint_warmth",
                      "tentative_light", "soft_openness", "quiet_hope"],
    },
    "angry": {
        "journey": "calming",
        "description": "From contained fire toward stillness",
        "arc_beats": ["contained_fire", "fracture", "exhale",
                      "cooling", "ground", "stillness"],
    },
    "tired": {
        "journey": "relaxing",
        "description": "From bone-deep drag toward rest",
        "arc_beats": ["drag", "surrender", "softness",
                      "drift", "quiet_rest", "renewal"],
    },
    "happy": {
        "journey": "elation",
        "description": "From spark of joy toward luminous transcendence",
        "arc_beats": ["spark", "expansion", "overflow",
                      "radiance", "luminous_still", "transcendence"],
    },
    "anxious": {
        "journey": "grounding",
        "description": "From spiral toward root",
        "arc_beats": ["spiral", "peak_noise", "pause",
                      "breath", "root", "present"],
    },
    "grief": {
        "journey": "tender continuance",
        "description": "From the shape of absence toward carrying",
        "arc_beats": ["absence", "ache", "memory",
                      "held", "continuance", "carried_forward"],
    },
    "determined": {
        "journey": "heroic rise",
        "description": "From doubt toward resolute action",
        "arc_beats": ["doubt", "challenge", "resistance",
                      "breakthrough", "momentum", "triumph"],
    },
    "love": {
        "journey": "deepening",
        "description": "From spark toward enduring warmth",
        "arc_beats": ["spark", "recognition", "vulnerability",
                      "trust", "embrace", "unity"],
    },
}

if _config_arcs:
    MOOD_ARCS.update(_config_arcs)

DEFAULT_ARC = {
    "journey": "reflective",
    "description": "From feeling toward witnessing",
    "arc_beats": ["acknowledgment", "presence", "shift", "openness"],
}


class TemplateStoryGeneratorResponse:
    def __init__(self, content: str):
        self.content = content


class TemplateStoryGenerator:
    """Mock/Fallback LLM that returns a structured template story as JSON."""
    
    def invoke(self, messages: list) -> TemplateStoryGeneratorResponse:
        content = ""
        for msg in messages:
            if hasattr(msg, "content") and "Write exactly" in msg.content:
                content = msg.content
                break
                
        import re
        panel_count_match = re.search(r"Write exactly (\d+) panels", content)
        panel_count = int(panel_count_match.group(1)) if panel_count_match else 4
        
        char_match = re.search(r"Character:\s*(.*)", content)
        character_name = char_match.group(1).strip() if char_match else "Wanderer"
        
        world_match = re.search(r"Story world:\s*(.*)", content)
        story_world = world_match.group(1).strip() if world_match else "The Abstract"
        
        emotion_match = re.search(r"Emotion:\s*(.*)", content)
        emotion = emotion_match.group(1).strip() if emotion_match else "determined"
        
        engine = StoryIntakeEngine()
        story_dict = engine._generate_fallback(
            user_prompt="Fallback story",
            emotion=emotion,
            panel_count=panel_count,
            character_name=character_name,
            story_world=story_world
        )
        
        import json
        return TemplateStoryGeneratorResponse(json.dumps(story_dict))


class StoryIntakeEngine:
    """
    Phase 0: Story Intake

    Takes a raw user prompt (emotion + story idea) and produces a
    structured story configuration that feeds into the Multi-Agent
    Planning Layer (Phase 1).
    """

    def __init__(self, ollama_model: str = "llama3.2",
                 ollama_url: str = "http://localhost:11434"):
        self.ollama_model = os.environ.get("OLLAMA_MODEL") or ollama_model
        self.ollama_url = os.environ.get("OLLAMA_URL") or ollama_url
        self._llm = None

    def _get_llm(self):
        """Lazy-load the appropriate LLM connection based on provider configuration."""
        if self._llm is None:
            provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
            log.info(f"[StoryIntakeEngine] Initializing LLM provider: {provider}")
            
            if provider == "openai":
                try:
                    from langchain_openai import ChatOpenAI  # type: ignore
                    self._llm = ChatOpenAI(
                        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                        temperature=0.3
                    )
                    log.info(f"Connected to OpenAI: {self._llm.model_name}")
                except Exception as e:
                    log.warning(f"OpenAI init failed: {e}. Using templates.")
                    return TemplateStoryGenerator()
            elif provider == "gemini":
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
                    self._llm = ChatGoogleGenerativeAI(
                        model=os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"),
                        temperature=0.3
                    )
                    log.info(f"Connected to Gemini: {self._llm.model}")
                except Exception as e:
                    log.warning(f"Gemini init failed: {e}. Using templates.")
                    return TemplateStoryGenerator()
            elif provider == "anthropic":
                try:
                    from langchain_anthropic import ChatAnthropic  # type: ignore
                    self._llm = ChatAnthropic(
                        model=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
                        temperature=0.3
                    )
                    log.info(f"Connected to Anthropic: {self._llm.model}")
                except Exception as e:
                    log.warning(f"Anthropic init failed: {e}. Using templates.")
                    return TemplateStoryGenerator()
            else:
                # Default: Ollama
                try:
                    from langchain_ollama import ChatOllama
                    self._llm = ChatOllama(
                        model=self.ollama_model,
                        temperature=0.3,
                        base_url=self.ollama_url,
                    )
                    log.info(f"Connected to Ollama: {self.ollama_model}")
                except Exception as e:
                    log.warning(f"Ollama init failed: {e}. Using templates.")
                    return TemplateStoryGenerator()
        return self._llm

    def process_prompt(self, user_prompt: str,
                       panel_count: int = 6,
                       character_name: str = "Wanderer",
                       story_world: str = "The Abstract",
                       style_reference: str = "",
                       character_characteristics: str = "",
                       story_reference: str = "",
                       mood_shifts: Optional[List[str]] = None,
                       weave_mood: bool = False,
                       story_mode: str = "literal") -> Dict[str, Any]:
        """
        Process a raw user prompt into a structured story configuration.

        Args:
            user_prompt: Raw emotional / story prompt from user
            panel_count: Number of panels to generate (4-10)
            character_name: Main character name
            story_world: Story world/setting name
            weave_mood: Enable Mood Weaver mode (auto-detects emotion, maps character, selects franchise style)
            story_mode: "literal" (default) makes user_prompt the primary structural
                driver — panels are the story split into N sequential moments, and
                detected emotion only shades tone/lighting. "mood_arc" restores the
                legacy behaviour where a fixed generic emotional-arc template
                dictates each panel's beat and user_prompt is passed as one line
                of background context.

        Returns:
            Structured story config dict with:
            - recurring_motif, mood_journey, panels[], metadata
        """
        panel_count = max(1, min(10, panel_count))
        log.info(f"[Phase 0] Processing prompt: '{user_prompt[:80]}...'")

        # Dynamically load mood-weaver analyzer if available
        try:
            import importlib.util
            # Load mood_analyzer dynamically from the mood-weaver directory
            repo_root = Path(__file__).parent.parent.parent
            path = repo_root / "mood-weaver" / "scripts" / "mood_analyzer.py"
            if path.exists():
                spec = importlib.util.spec_from_file_location("mood_analyzer", path)
                if spec is not None and spec.loader is not None:
                    mood_analyzer = importlib.util.module_from_spec(spec)
                    sys.modules["mood_analyzer"] = mood_analyzer
                    spec.loader.exec_module(mood_analyzer)
                    log.info("[StoryIntakeEngine] Successfully loaded custom mood analyzer from mood-weaver")
        except Exception as e:
            log.warning(f"[StoryIntakeEngine] Failed to load custom mood analyzer: {e}")

        # Detect primary emotion from the prompt
        emotion = self._detect_emotion(user_prompt)
        log.info(f"  Detected emotion: {emotion}")

        if weave_mood:
            # Compile and parse unified references from all three CSV files
            csv_path = "outputs/unified_references.csv"
            franchise = select_franchise_from_unified_csv(emotion, user_prompt, csv_path)
            
            style_reference = f"A dramatic crossover matching {franchise['source']} title '{franchise['title']}' in style '{franchise['genre']}'"
            story_world = f"crossover universe of {franchise['title']}"
            story_reference = franchise.get("description", "")
            
            # Map character name based on franchise or fallback
            character_name = get_character_for_franchise(franchise["title"], emotion)
            character_characteristics = f"the main protagonist from {franchise['title']} experiencing intense emotion: {emotion}"
            
            log.info(f"  [Mood Weaver] Mapped Character: '{character_name}'")
            log.info(f"  [Mood Weaver] Selected Crossover Setting: '{franchise['title']}' ({franchise['source']}) in '{story_world}'")

        log.info(f"  Character: {character_name}, World: {story_world}, Panels: {panel_count}")

        # Try LLM-based story generation first
        llm = self._get_llm()
        if llm is not None:
            try:
                story_config = self._generate_with_llm(
                    user_prompt, emotion, panel_count, character_name, story_world, style_reference,
                    character_characteristics, story_reference, mood_shifts, story_mode
                )
                if story_config and self._validate_story(story_config, panel_count):
                    # Inject metadata so downstream agents can find character/world
                    story_config["_metadata"] = {
                        "emotion": emotion,
                        "character": character_name,
                        "world": story_world,
                        "source": f"ollama_{self.ollama_model}",
                    }
                    log.info(f"[Phase 0] LLM story generation successful — {len(story_config.get('panels', []))} panels")
                    return story_config
            except Exception as e:
                log.warning(f"LLM story generation failed: {e}")

        # Fallback: template-based generation
        log.info("[Phase 0] Using template-based story generation")
        return self._generate_fallback(user_prompt, emotion, panel_count,
                                       character_name, story_world, character_characteristics, story_reference, mood_shifts)

    def load_existing_story(self, path: str) -> Dict[str, Any]:
        """Load an existing story_dynamic.json from Story-Weaver."""
        story_path = Path(path)
        if not story_path.exists():
            raise FileNotFoundError(f"Story file not found: {story_path}")

        with open(story_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        log.info(f"[Phase 0] Loaded existing story from {story_path}")
        log.info(f"  Motif: {data.get('recurring_motif', 'N/A')}")
        log.info(f"  Panels: {len(data.get('panels', []))}")
        return data

    # ─────────────────────────────────────────────────────────────────────
    # LLM-Based Generation
    # ─────────────────────────────────────────────────────────────────────

    def _generate_with_llm(self, user_prompt: str, emotion: str,
                           panel_count: int, character_name: str,
                           story_world: str, style_reference: str = "",
                           character_characteristics: str = "",
                           story_reference: str = "",
                           mood_shifts: Optional[List[str]] = None,
                           story_mode: str = "literal") -> Optional[Dict[str, Any]]:
        """Generate story by delegating to Story-Weaver's dynamic generator."""
        import sys
        import importlib.util
        from pathlib import Path
        import os
        import re

        repo_root = Path(__file__).parent.parent.parent
        story_weaver_path = repo_root / "Story-Weaver" / "story_gen.py"
        if story_weaver_path.exists():
            log.info("[StoryIntakeEngine] Delegating generation to Story-Weaver story_gen module...")
            try:
                # Build references strings
                style_str = (
                    f"VISUAL STYLE ENFORCEMENT: Every panel's camera and environment fields MUST "
                    f"reflect the exact visual style of '{style_reference}' — its color palette, "
                    f"linework, atmosphere, and artistic treatment.\n"
                ) if style_reference else ""

                char_str = (
                    f"CHARACTER CONSISTENCY: The main character is '{character_name}'. "
                    f"Physical description: '{character_characteristics}'. "
                    f"Every panel MUST include their clothing, hair, and physical traits "
                    f"in the pose.body field so the visual model keeps them consistent.\n"
                ) if character_characteristics else (
                    f"CHARACTER: The main character is '{character_name}'. "
                    f"Choose a consistent visual description for them and include it in every panel's pose.body field.\n"
                )

                story_ref_str = (
                    f"NARRATIVE REFERENCE: Model the story's pacing, dialogue tone, "
                    f"and dramatic structure after '{story_reference}'. "
                    f"Use its tropes, scene rhythms, and character dynamics as inspiration.\n"
                ) if story_reference else ""

                # Push constraint strings to env for story_gen prepending
                os.environ["STYLE_STR"] = style_str
                os.environ["CHAR_STR"] = char_str
                os.environ["STORY_REF_STR"] = story_ref_str
                os.environ["COMIC_SCHEMA_OVERRIDE"] = "true"
                os.environ["OLLAMA_URL"] = self.ollama_url

                # Load module
                spec = importlib.util.spec_from_file_location("story_gen_module", story_weaver_path)
                if spec is not None and spec.loader is not None:
                    story_gen_module = importlib.util.module_from_spec(spec)
                    sys.modules["story_gen_module"] = story_gen_module
                    spec.loader.exec_module(story_gen_module)

                    # Cast to Any to prevent dynamic property type check errors
                    from typing import Any
                    story_gen_any: Any = story_gen_module

                    # Set global options in the imported module
                    story_gen_any.EMOTION = emotion
                    story_gen_any.PANEL_COUNT = panel_count
                    story_gen_any.USER_TEXT = user_prompt
                    story_gen_any.MODEL_PATH = self.ollama_model
                    story_gen_any.STORY_MODE = story_mode

                    # If custom mood shift beats are passed, override the arc beats in the module
                    if mood_shifts and len(mood_shifts) > 0:
                        story_gen_any.MOOD_ARCS[emotion] = {
                            "journey": "Custom User Arc",
                            "description": "Custom sequence of emotions",
                            "arc_beats": mood_shifts,
                            "motif_hint": "a recurring symbol of the user's journey",
                            "end_note": "End with the final shift panel."
                        }

                    # Instantiate and run
                    generator = story_gen_any.DynamicStoryGenerator()
                    story_config = generator.generate()
                else:
                    raise ImportError("Failed to load Story-Weaver story_gen spec")
                if story_config:
                    return story_config
            except Exception as e:
                log.warning(f"Failed to run Story-Weaver generator: {e}. Falling back to default generation.")

        # Legacy generation fallback in case Story-Weaver is missing or fails
        if mood_shifts and len(mood_shifts) > 0:
            beats = self._distribute_beats(panel_count, mood_shifts)
            arc = {"journey": "Custom User Arc", "description": "Custom sequence of emotions provided by user"}
        else:
            arc = MOOD_ARCS.get(emotion, DEFAULT_ARC)
            beats = self._distribute_beats(panel_count, arc["arc_beats"])

        # ── Reference injections ──
        style_str = (
            f"VISUAL STYLE ENFORCEMENT: Every panel's camera and environment fields MUST "
            f"reflect the exact visual style of '{style_reference}' — its color palette, "
            f"linework, atmosphere, and artistic treatment.\n"
        ) if style_reference else ""

        char_str = (
            f"CHARACTER CONSISTENCY: The main character is '{character_name}'. "
            f"Physical description: '{character_characteristics}'. "
            f"Every panel MUST include their clothing, hair, and physical traits "
            f"in the pose.body field so the visual model keeps them consistent.\n"
        ) if character_characteristics else (
            f"CHARACTER: The main character is '{character_name}'. "
            f"Choose a consistent visual description for them and include it in every panel's pose.body field.\n"
        )

        story_ref_str = (
            f"NARRATIVE REFERENCE: Model the story's pacing, dialogue tone, "
            f"and dramatic structure after '{story_reference}'. "
            f"Use its tropes, scene rhythms, and character dynamics as inspiration.\n"
        ) if story_reference else ""

        fidelity_str = (
            f"STORY FIDELITY (highest priority): The user has supplied a specific "
            f"story below. Adapt THAT story panel by panel, in order — keep its "
            f"named characters, setting, and plot events intact rather than "
            f"substituting a generic scene. Emotion is a tone/lighting choice "
            f"layered on top of the real events, never a replacement for them.\n"
        ) if story_mode == "literal" else ""

        system_prompt = f"""You are a master graphic novelist, cinematic director, and stunt choreographer.
Respond ONLY with a single valid JSON object. No markdown fences, no explanation, no comments.

{fidelity_str}{style_str}{char_str}{story_ref_str}

CORE RULE — THE SINGLE MOST IMPORTANT INSTRUCTION:
This is a COMIC, not prose. Every single panel MUST depict ONE specific, visually distinct,
dramatically exaggerated ACTION EVENT. Think Michael Bay storyboard meets Frank Miller.
Diffusion models regress to static portraits when given weak verbs. You must FIGHT that
by writing with maximum cinematic specificity.

PER-PANEL MANDATORY CHECKLIST (every panel must pass ALL 5 points):
1. ONE DOMINANT ACTION — a single physical event happening RIGHT NOW in this panel
2. EXAGGERATED MECHANICS — describe exact body-part positions under maximum tension
   - BAD: "Kage runs" / GOOD: "Kage hurls himself forward, spine horizontal, knuckles
     grazing ground, one foot driving off a wall"
3. IMPACT OR CONSEQUENCE — what physically happens as a result of the action
   - BAD: "hits" / GOOD: "boot connects with the panel, spiderweb cracks radiating outward"
4. ENVIRONMENTAL REACTION — how does the world around them respond to this action
   - dust clouds, neon signs flickering, windows shattering, embers rising, sparks
5. CINEMATIC FREEZE-FRAME CUE — tell the model WHICH millisecond of the action
   - "anticipation frame", "maximum-force impact freeze", "follow-through recovery",
     "peak of arc silhouetted against sky", "landing impact hold"

EVENT SEQUENCING RULE:
Each panel MUST depict a DIFFERENT physical state from every other panel.
Panel 1 = setup/anticipation. Panel 2 = rising conflict. Panel 3 = escalation.
Middle panels = confrontation + consequence. Final panels = climax + aftermath.
No two panels can show the same action or the same body position.
Consecutive panels must feel like frames from a single kinetic sequence.

LITERARY CONSTRAINTS (from Story-Weaver blueprint):
- NEVER name emotions directly. Show through action, objects, sensation.
- ONE recurring visual motif must appear in every single panel.
- Every panel MUST include a physical body sensation.
- No moral lessons.
- Dialogue reveals what characters choose NOT to say as much as what they say.

VISUAL SPECIFICITY RULES — you MUST follow these for every panel:
1. "camera" field: MUST specify angle + movement. Examples:
   - "Extreme close-up, static, face filling frame"
   - "Low-angle medium shot, slow upward tilt"
   - "Wide establishing shot, handheld drift left"
   - "Dutch tilt over-shoulder, locked off"
   - "Bird's eye overhead, slow zoom out"

2. "environment" field: MUST include ALL four of:
   (a) LOCATION — where exactly (rooftop, cramped subway, open field, neon alley)
   (b) TIME — time of day and sky condition (3am overcast, golden hour, harsh noon)
   (c) DOMINANT COLOR PALETTE — the 2-3 colours that define this scene
   (d) LIGHT SOURCE — where the light comes from and its quality
   Example: "Narrow rain-soaked alleyway, 2am, dominant palette deep indigo and neon pink,
   single overhead streetlamp casting hard shadows downward"

3. actions field — THE MOST IMPORTANT FIELD. Must contain:
   - "verb": a POWERFUL, SPECIFIC action verb (NOT walk/stand/look)
     Use: sprint, hurl, slam, explode, pivot, wrench, crash, erupt, lunge, shatter
   - "target": what or whom the verb acts on, with specific physical detail
   - "mechanics": exact body position and tension (NEW MANDATORY FIELD)
   - "impact": what physically happens at the point of contact or consequence
   - "reaction": how the environment responds to this action
   - "timing": the freeze-frame moment label

4. pose.body field: MUST include character's clothing/costume AND EXTREME body position.
   Good: "lunging forward in torn grey hoodie, spine parallel to ground,
   right arm cocked back past the ear, left hand braced on a wall"
   Bad: "crouching in grey hoodie"

5. dialogue.text: Write REAL dialogue. Punchy. Max 12 words. No placeholders.
   Characters speak like real people under extreme emotional pressure.

Generate a {panel_count}-panel comic story. Output this exact JSON:
{{
  "story_bible": {{
    "plot_summary": "2-3 sentence story summary",
    "side_characters": [ {{"name": "...", "role": "...", "description": "..."}} ]
  }},
  "recurring_motif": "A single specific visual object that recurs (e.g. a cracked compass)",
  "mood_journey": "One sentence arc description",
  "panels": [
    {{
      "panel": 1,
      "emotion_beat": "beat_name",
      "characters": [
        {{
          "id": "character_name_lowercase",
          "pose": {{"body": "clothing + EXTREME physical stance", "head": "head direction", "arms": "arm position", "legs": "leg position"}},
          "expression": {{"emotion": "specific emotion", "eyes": "eye description", "mouth": "mouth position"}},
          "dialogue": {{"text": "Actual spoken words.", "tone": "tone descriptor", "bubble": "speech|thought|shout|whisper"}}
        }}
      ],
      "actions": [ {{
        "actor": "character_id",
        "verb": "powerful specific action verb",
        "target": "what/whom with physical detail",
        "mechanics": "exact body-part positions under tension",
        "impact": "what physically happens at contact/consequence",
        "reaction": "how the environment responds",
        "timing": "freeze-frame moment label"
      }} ],
      "camera": "angle + movement descriptor",
      "environment": "location, time, dominant palette, light source"
    }}
  ]
}}"""

        TIMING_PHASES = {
            4: ["setup", "confrontation", "climax", "resolution"],
            6: ["status_quo", "inciting_incident", "rising_action", "crisis", "climax", "resolution"],
            8: ["exposition", "inciting_incident", "rising_action", "complication", "crisis", "climax", "resolution", "aftermath"]
        }
        phases = TIMING_PHASES.get(panel_count, ["—"] * panel_count)

        if story_mode == "literal":
            # The story itself drives panel content; arc_beats become an
            # optional tone vocabulary instead of a mandatory per-panel list.
            beat_vocab = ", ".join(arc.get("arc_beats", []))
            user_msg = (
                f'STORY TO ADAPT (primary source — follow it exactly, in order; '
                f'do not invent an unrelated plot):\n"""\n{user_prompt}\n"""\n\n'
                f"Character: {character_name}\n"
                f"Story world: {story_world}\n"
                f"Detected emotional tone: {emotion}\n"
                f"TONE VOCABULARY (optional, use only where it genuinely fits): {beat_vocab}\n\n"
                f"Write exactly {panel_count} panels that divide the story above into "
                f"{panel_count} sequential moments, in story order. Carry the story's "
                f"specific character names, setting details, and events into every "
                f"panel's dialogue and visuals. Each panel's emotion_beat should name "
                f"what that specific moment feels like.\n\n"
                f"Remember: real dialogue, cinematic cameras, specific environments. Output JSON only."
            )
        else:
            beat_guide = "\n".join(
                f"  Panel {i+1} [{phases[i] if i < len(phases) else '—'}]: emotion_beat = \"{beats[i]}\""
                for i in range(panel_count)
            )

            user_msg = (
                f"Emotion: {emotion}\n"
                f"Story world: {story_world}\n"
                f"Character: {character_name}\n"
                f"User prompt: \"{user_prompt}\"\n\n"
                f"MOOD JOURNEY: {arc['journey']} — {arc['description']}\n\n"
                f"Write exactly {panel_count} panels. Assign these emotion_beat values:\n"
                f"{beat_guide}\n\n"
                f"Remember: real dialogue, cinematic cameras, specific environments. Output JSON only."
            )

        llm = self._get_llm()
        if llm is None:
            log.warning("LLM connection not available — using TemplateStoryGenerator fallback")
            llm = TemplateStoryGenerator()

        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_msg),
        ]
        try:
            response = llm.invoke(messages).content
        except Exception as e:
            log.warning(f"LLM invoke failed: {e}. Falling back to TemplateStoryGenerator.")
            fallback_llm = TemplateStoryGenerator()
            response = fallback_llm.invoke(messages).content
            
        return self._extract_and_repair_json(response)

    def _extract_and_repair_json(self, raw: str) -> Optional[Dict[str, Any]]:
        """Multi-pass JSON extraction and repair for LLM output."""
        import re

        # Pass 1: strip markdown fences
        text = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.MULTILINE)
        text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE).strip()

        # Pass 2: find the outermost JSON object
        start = text.find("{")
        if start == -1:
            log.warning("No JSON object found in LLM response")
            return None

        depth, end = 0, -1
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if end == -1:
            log.warning("JSON object not properly closed in LLM response")
            return None

        json_str = text[start:end + 1]

        # Pass 3: try direct parse
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            log.warning(f"Direct JSON parse failed ({e}), attempting repair...")

        # Pass 4: common LLM output fixes
        repaired = json_str
        # Remove trailing commas before } or ]
        repaired = re.sub(r',\s*(\}|\])', r'\1', repaired)
        # Remove single-line // comments
        repaired = re.sub(r'//[^\n]*', '', repaired)
        # Remove block /* */ comments
        repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)
        # Replace smart/curly quotes with straight quotes
        repaired = repaired.replace('\u201c', '"').replace('\u201d', '"')
        repaired = repaired.replace('\u2018', "'").replace('\u2019', "'")
        # Collapse newlines to spaces
        repaired = repaired.replace('\n', ' ').replace('\r', ' ')
        # Fix unquoted keys (e.g. {key: "val"} → {"key": "val"})
        repaired = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', repaired)

        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e2:
            log.warning(f"JSON repair failed: {e2}")
            raise

    # ─────────────────────────────────────────────────────────────────────
    # Fallback Template Generation
    # ─────────────────────────────────────────────────────────────────────

    def _generate_fallback(self, user_prompt: str, emotion: str,
                           panel_count: int, character_name: str,
                           story_world: str,
                           character_characteristics: str = "",
                           story_reference: str = "",
                           mood_shifts: List[str] | None = None) -> Dict[str, Any]:
        """Template-based fallback when LLM is unavailable."""
        if mood_shifts and len(mood_shifts) > 0:
            beats = self._distribute_beats(panel_count, mood_shifts)
            arc = {"description": "Custom sequence of emotions provided by user"}
        else:
            arc = MOOD_ARCS.get(emotion, DEFAULT_ARC)
            beats = self._distribute_beats(panel_count, arc["arc_beats"])

        motif_hints = {
            "sad": "A solitary paper boat floating in a dark puddle",
            "angry": "Cracks spreading across a concrete wall",
            "tired": "A folded blanket at the foot of a bed",
            "happy": "Late afternoon light through a dusty window",
            "anxious": "A single houseplant on the windowsill",
            "grief": "An empty chair at a kitchen table",
            "determined": "A flickering torch in a dark corridor",
            "love": "Two shadows overlapping on warm pavement",
        }

        # Per-beat camera and environment templates for visual variety
        camera_by_position = [
            "Wide establishing shot, slow drift right",          # opening
            "Medium shot, static, character centre-frame",       # early 1
            "Over-the-shoulder shot, locked off",               # early 2
            "Close-up, static, face filling frame",             # midpoint
            "Low-angle medium shot, slow upward tilt",          # rising
            "Dutch tilt medium shot, handheld",                 # tension
            "Extreme close-up, eyes only, static",              # climax
            "Wide shot, slow zoom out, character small",        # resolution
            "Medium shot, static, soft focus background",       # wind-down
            "Wide overhead bird's eye, slow descend",           # coda
        ]

        beat_cameras = {
            "contained_fire": "Low-angle medium shot, slow upward tilt",
            "fracture": "Dutch tilt close-up, handheld shake",
            "spiral": "Dutch tilt medium shot, slow clockwise rotation",
            "peak_noise": "Extreme close-up eyes, handheld shake",
            "breakthrough": "Low-angle wide shot, fast upward tilt",
            "triumph": "Wide shot, slow pull-back reveal",
            "stillness": "Wide static shot, character small in frame",
            "drift": "Medium shot, very slow drift, soft focus",
            "quiet_rest": "Wide overhead, slow zoom out",
            "absence": "Wide empty room shot, static",
            "radiance": "Low-angle medium shot, glowing backlight",
            "transcendence": "Overhead bird's eye, slow pull back to sky",
        }

        beat_environments = {
            "contained_fire": f"cramped rooftop of {story_world}, deep night, dominant palette crimson and charcoal, single harsh sodium streetlamp from below",
            "fracture": f"cracked concrete wall in {story_world}, predawn darkness, dominant palette blood red and near-black, harsh single spotlight",
            "spiral": f"narrow corridor in {story_world}, flickering fluorescent overhead, dominant palette sickly green-white, harsh top-down light",
            "stillness": f"empty room in {story_world}, early morning grey, dominant palette muted monochrome blue-grey, flat diffused overcast light through window",
            "drift": f"soft interior space in {story_world}, late evening, dominant palette lavender and pale grey, dim warm lamp in corner",
            "breakthrough": f"open sky above {story_world}, sunrise, dominant palette white-gold explosion, backlit rim light",
            "triumph": f"elevated vista above {story_world}, full golden hour, dominant palette vibrant warm spectrum, full open sunlight",
            "quiet_rest": f"small bedroom in {story_world}, 3am, dominant palette deep indigo and silver, moonlight through curtain",
            "absence": f"empty kitchen in {story_world}, cold daylight, dominant palette cold white and pale grey, flat cold window light",
            "radiance": f"open hillside in {story_world}, golden hour, dominant palette gold and luminous white, warm backlit halo",
        }

        pose_by_beat = {
            "contained_fire": {"body": "standing rigid, fists clenched at sides", "head": "jaw tight, chin slightly lowered", "arms": "locked straight down", "legs": "planted wide"},
            "fracture": {"body": "lurching forward, off-balance", "head": "snapping upward", "arms": "one extended forward", "legs": "staggered"},
            "spiral": {"body": "hunched, curling inward", "head": "tilted down", "arms": "wrapped around torso", "legs": "knees bent"},
            "stillness": {"body": "standing very still, weight centred", "head": "level, forward", "arms": "hanging loose", "legs": "feet together"},
            "drift": {"body": "lying or sitting, body slack", "head": "tilted back softly", "arms": "open, palms up", "legs": "extended relaxed"},
            "breakthrough": {"body": "lunging forward, full extension", "head": "raised, eyes wide open", "arms": "one arm thrusting forward", "legs": "back leg extended in stride"},
            "triumph": {"body": "standing tall, chest open, weight back", "head": "raised, face to sky", "arms": "raised wide above head", "legs": "planted strong and wide"},
            "quiet_rest": {"body": "lying down, fully relaxed", "head": "resting sideways on pillow", "arms": "beside body or folded", "legs": "extended, uncrossed"},
        }
        default_pose = {"body": "standing, natural weight distribution", "head": "forward, level", "arms": "relaxed at sides", "legs": "shoulder-width apart"}

        dialogue_by_beat = {
            "contained_fire": {"text": "Not yet.", "tone": "low and controlled", "bubble": "speech"},
            "fracture": {"text": "That's enough.", "tone": "sharp", "bubble": "shout"},
            "spiral": {"text": "I can't — I can't stop it.", "tone": "panicked whisper", "bubble": "thought"},
            "breakthrough": {"text": "Move.", "tone": "commanding", "bubble": "speech"},
            "triumph": {"text": "We did it.", "tone": "breathless", "bubble": "speech"},
            "stillness": {"text": "...", "tone": "silent", "bubble": "thought"},
            "drift": {"text": "Just for a moment.", "tone": "murmured", "bubble": "thought"},
            "quiet_rest": {"text": "...", "tone": "silent", "bubble": "thought"},
            "absence": {"text": "Where did you go?", "tone": "hollow", "bubble": "thought"},
            "radiance": {"text": "I see it now.", "tone": "soft wonder", "bubble": "speech"},
        }
        default_dialogue = {"text": "...", "tone": "neutral", "bubble": "speech"}

        # ── Beat → cinematic action mapping for the fallback generator ──
        # Each entry is a raw action dict that ActionDirector will ALSO exaggerate,
        # so even the fallback path produces 5-layer cinematic prompts.
        # The verb keys match ACTION_EXAGGERATION_MAP in director_swarm.py.
        _BEAT_ACTIONS: Dict[str, Dict[str, str]] = {
            # Angry arc
            "contained_fire": {"verb": "holds",    "target": "the edge, knuckles white on the railing"},
            "fracture":       {"verb": "punch",    "target": "the cracked wall, plaster exploding"},
            "exhale":         {"verb": "stands",   "target": "in the aftermath, chest heaving"},
            "cooling":        {"verb": "sits",     "target": "on the ground, back against the wall"},
            "ground":         {"verb": "watches",  "target": "the horizon, breath finally slowing"},
            # Determined arc
            "doubt":          {"verb": "watches",  "target": "the obstacle, jaw tight, calculating"},
            "challenge":      {"verb": "stands",   "target": "at the base of the impossible climb"},
            "resistance":     {"verb": "holds",    "target": "position against the crushing force"},
            "breakthrough":   {"verb": "charge",   "target": "through the barrier, arms breaking through"},
            "momentum":       {"verb": "run",      "target": "forward, nothing left to slow them down"},
            "triumph":        {"verb": "raises",   "target": "arms to the open sky"},
            # Sad arc
            "heaviness":      {"verb": "crawl",    "target": "out of bed, the weight of everything"},
            "stillness":      {"verb": "stands",   "target": "in the empty room, hands at sides"},
            "faint_warmth":   {"verb": "reaches",  "target": "toward the single point of warm light"},
            "tentative_light":{"verb": "reaches",  "target": "for the door handle, not yet turning it"},
            "soft_openness":  {"verb": "watches",  "target": "the morning come through the curtain"},
            "quiet_hope":     {"verb": "stands",   "target": "at the window, face tilted into the light"},
            # Happy arc
            "spark":          {"verb": "leap",     "target": "from the starting line, arms wide"},
            "expansion":      {"verb": "runs",     "target": "across the open field, arms spread"},
            "overflow":       {"verb": "raises",   "target": "both arms laughing into the sky"},
            "radiance":       {"verb": "stands",   "target": "glowing in the full warm light"},
            "luminous_still": {"verb": "watches",  "target": "the world with complete peace"},
            "transcendence":  {"verb": "floats",   "target": "in the light, gravity forgotten"},
            # Anxious arc
            "spiral":         {"verb": "clutches", "target": "head, the noise becoming unbearable"},
            "peak_noise":     {"verb": "block",    "target": "ears with both palms, eyes screwed shut"},
            "pause":          {"verb": "stands",   "target": "at the eye of the storm, utterly still"},
            "breath":         {"verb": "sits",     "target": "on the floor, hands on knees, breathing"},
            "root":           {"verb": "stands",   "target": "with both feet planted, weight dropping"},
            "present":        {"verb": "watches",  "target": "everything clearly for the first time"},
            # Grief arc
            "absence":        {"verb": "stands",   "target": "in the door of the empty room"},
            "ache":           {"verb": "sits",     "target": "holding the last thing left of them"},
            "memory":         {"verb": "reaches",  "target": "toward the photograph, not touching it"},
            "held":           {"verb": "watches",  "target": "someone hold them without flinching"},
            "continuance":    {"verb": "stands",   "target": "at the door, coat on, ready to go back"},
            "carried_forward":{"verb": "walks",    "target": "forward, carrying everything that matters"},
            # Tired arc
            "drag":           {"verb": "crawl",    "target": "toward the sink, one step at a time"},
            "surrender":      {"verb": "fall",     "target": "onto the bed without taking off shoes"},
            "softness":       {"verb": "sits",     "target": "letting the blanket pull around them"},
            "drift":          {"verb": "floats",   "target": "between waking and sleep"},
            "quiet_rest":     {"verb": "lies",     "target": "completely still, finally resting"},
            "renewal":        {"verb": "stands",   "target": "at the window, seeing morning"},
            # Love arc
            "recognition":    {"verb": "watches",  "target": "them across the room, seeing clearly"},
            "vulnerability":  {"verb": "reaches",  "target": "out with open hand, not yet taken"},
            "trust":          {"verb": "stands",   "target": "beside them, close enough to touch"},
            "embrace":        {"verb": "holds",    "target": "them with every bit of strength left"},
            "unity":          {"verb": "watches",  "target": "the same thing at the same moment"},
            # Generic
            "neutral":        {"verb": "stands",   "target": "taking stock of the moment"},
        }

        panels = []
        for i in range(panel_count):
            beat = beats[i]
            progress = i / max(1, panel_count - 1)
            intensity = 0.3 + 0.4 * abs(progress - 0.5) * 2

            cam_idx = min(i, len(camera_by_position) - 1)
            camera = beat_cameras.get(beat, camera_by_position[cam_idx])
            environment = beat_environments.get(
                beat,
                f"{story_world}, {'dawn' if i == 0 else 'dusk' if i == panel_count - 1 else 'midday'}, "
                f"dominant palette neutral warm tones, natural ambient light"
            )
            pose = pose_by_beat.get(beat, default_pose)
            dialogue = dialogue_by_beat.get(beat, default_dialogue)

            # Resolve action: use the beat-specific dramatic action, fallback to generic
            action_entry = _BEAT_ACTIONS.get(beat, {"verb": "stands", "target": "in the scene"})

            panels.append({
                "panel": i + 1,
                "emotion_beat": beat,
                "characters": [
                    {
                        "id": character_name.lower(),
                        "pose": pose,
                        "expression": {"emotion": beat, "eyes": "focused", "mouth": "neutral"},
                        "dialogue": dialogue,
                    }
                ],
                "actions": [
                    {
                        "actor": character_name.lower(),
                        "verb": action_entry["verb"],
                        "target": action_entry["target"],
                        # mechanics/impact/reaction/timing are left empty here
                        # so ActionDirector.plan() will populate them from
                        # ACTION_EXAGGERATION_MAP during Phase 1 planning
                    }
                ],
                "camera": camera,
                "environment": environment,
                "_action_intensity": intensity,
            })

        return {
            "recurring_motif": motif_hints.get(emotion, "A symbolic recurring object"),
            "mood_journey": f"{arc['description']} — from {beats[0]} to {beats[-1]}",
            "panels": panels,
            "_metadata": {
                "emotion": emotion,
                "character": character_name,
                "world": story_world,
                "source": "template_fallback",
            }
        }

    # ─────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────

    def _detect_emotion(self, text: str) -> str:
        """Detect primary emotion from user text using mood-weaver custom analyzer or keyword matching."""
        try:
            if "mood_analyzer" in sys.modules:
                mood_analyzer = sys.modules["mood_analyzer"]
                res = mood_analyzer.analyze_mood(text)
                emotion = res.get("primary_emotion", "").lower()
                # Map standard labels from model (e.g., joy -> happy, sadness -> sad)
                label_map = {
                    "joy": "happy",
                    "sadness": "sad"
                }
                emotion = label_map.get(emotion, emotion)
                if emotion in ["sad", "angry", "tired", "happy", "anxious", "grief", "determined", "love", "surprise"]:
                    return emotion
        except Exception as e:
            log.warning(f"Error calling custom mood_analyzer: {e}. Falling back to keywords.")

        # Fallback keyword matching
        text_lower = text.lower()

        emotion_keywords = {
            "sad": ["sad", "heavy", "down", "depressed", "blue", "cry", "tears", "loss"],
            "angry": ["angry", "mad", "furious", "rage", "hate", "frustrated", "annoyed"],
            "tired": ["tired", "exhausted", "empty", "drained", "numb", "can't get up"],
            "happy": ["happy", "joy", "wonderful", "beautiful", "excited", "love"],
            "anxious": ["anxious", "worried", "nervous", "racing", "panic", "stress"],
            "grief": ["grief", "passed away", "died", "mourning", "miss", "gone"],
            "determined": ["determined", "fight", "hero", "battle", "quest", "adventure"],
            "love": ["love", "romance", "heart", "together", "soulmate", "passion"],
        }

        scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[emotion] = score

        if scores:
            return max(scores, key=lambda k: scores[k])
        return "determined"  # Default for action/adventure prompts

    def _distribute_beats(self, n: int, beats: list) -> list:
        """Distribute arc beats across N panels."""
        if not beats:
            return ["neutral"] * n
        if n <= 1:
            return [beats[len(beats) // 2]] * n
        if n <= len(beats):
            # Evenly sample beats without skipping middle sequences
            indices = [int(i * (len(beats) - 1) / (n - 1)) for i in range(n)]
            return [beats[idx] for idx in indices]
        # If more panels than beats, stretch them out evenly
        result = []
        for i in range(n):
            idx = int(i * (len(beats) - 1) / (n - 1))
            result.append(beats[idx])
        return result

    def _validate_story(self, data: dict, expected_panels: int) -> bool:
        """Validate the story config has required fields."""
        panels = data.get("panels", [])
        if len(panels) != expected_panels:
            return False
        for p in panels:
            for key in ("panel", "characters", "camera", "environment"):
                if key not in p:
                    return False
        return True


def compile_unified_references(csv_dir: Optional[str] = None, output_path: str = "outputs/unified_references.csv"):
    import csv
    import os
    
    if csv_dir is None:
        csv_dir = str(Path(__file__).parent.parent.parent)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["title", "genre", "description", "source", "popularity_score"]
    
    unified_rows = []
    
    # 1. Parse myanilist.csv
    myanilist_path = os.path.join(csv_dir, "myanilist.csv")
    if os.path.exists(myanilist_path):
        try:
            with open(myanilist_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    title = row.get("Title_English") or row.get("Title_Romaji")
                    genre = row.get("Genres", "")
                    source = f"Anime/Manga ({row.get('Source', 'MANGA')})"
                    desc = f"A popular series from {row.get('Studios', 'various studios')}."
                    
                    # Popularity score
                    pop_val = row.get("Mean_Score") or row.get("Average_Score")
                    try:
                        popularity_score = float(pop_val) if pop_val else 50.0
                    except ValueError:
                        popularity_score = 50.0
                        
                    if title:
                        unified_rows.append({
                            "title": title,
                            "genre": genre,
                            "description": desc,
                            "source": source,
                            "popularity_score": str(popularity_score)
                        })
        except Exception as e:
            log.warning(f"Error compiling myanilist: {e}")
            
    # 2. Parse Trending_Movies.csv
    movies_path = os.path.join(csv_dir, "Trending_Movies.csv")
    if os.path.exists(movies_path):
        try:
            with open(movies_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    title = row.get("title")
                    desc = row.get("overview", "")
                    desc_lower = desc.lower() if desc else ""
                    genres = []
                    if "kill" in desc_lower or "detective" in desc_lower or "agent" in desc_lower:
                        genres.append("Thriller")
                    if "action" in desc_lower or "fight" in desc_lower or "war" in desc_lower:
                        genres.append("Action")
                    if "love" in desc_lower or "romance" in desc_lower:
                        genres.append("Romance")
                    if "space" in desc_lower or "ship" in desc_lower or "alien" in desc_lower:
                        genres.append("Sci-Fi")
                    if not genres:
                        genres.append("Drama")
                    genre = ", ".join(genres)
                    
                    # Popularity score
                    pop_val = row.get("vote_average")
                    try:
                        popularity_score = float(pop_val) * 10 if pop_val else 50.0
                    except ValueError:
                        popularity_score = 50.0
                        
                    if title:
                        unified_rows.append({
                            "title": title,
                            "genre": genre,
                            "description": desc,
                            "source": "Movie",
                            "popularity_score": str(popularity_score)
                        })
        except Exception as e:
            log.warning(f"Error compiling Trending_Movies: {e}")
            
    # 3. Parse tv_movie_animation.csv
    tv_path = os.path.join(csv_dir, "tv_movie_animation.csv")
    if os.path.exists(tv_path):
        try:
            with open(tv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    title = row.get("title")
                    genre = row.get("genre", "")
                    desc = row.get("desc", "")
                    
                    # Popularity score
                    pop_val = row.get("rating")
                    try:
                        popularity_score = float(pop_val) * 10 if pop_val else 50.0
                    except ValueError:
                        popularity_score = 50.0
                        
                    if title:
                        unified_rows.append({
                            "title": title,
                            "genre": genre,
                            "description": desc,
                            "source": "Animation/TV/Movie",
                            "popularity_score": str(popularity_score)
                        })
        except Exception as e:
            log.warning(f"Error compiling tv_movie_animation: {e}")
            
    # Write to unified CSV
    try:
        with open(output_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(unified_rows)
        log.info(f"[Mood Weaver] Successfully compiled {len(unified_rows)} unified references to {output_path}")
    except Exception as e:
        log.error(f"Error writing unified references: {e}")


def select_franchise_from_unified_csv(emotion: str, user_prompt: str, csv_path: str = "outputs/unified_references.csv") -> dict:
    import csv
    import os
    import re
    
    # Force recompile if missing or doesn't have popularity_score in headers
    recompile_needed = False
    if not os.path.exists(csv_path):
        recompile_needed = True
    else:
        try:
            with open(csv_path, mode="r", encoding="utf-8") as f:
                header = f.readline()
                if "popularity_score" not in header:
                    recompile_needed = True
        except Exception:
            recompile_needed = True
            
    if recompile_needed:
        compile_unified_references(output_path=csv_path)
        
    genre_mapping = {
        "sadness": ["Drama", "Psychological", "Romance", "Slice of Life"],
        "sad": ["Drama", "Psychological", "Romance", "Slice of Life"],
        "grief": ["Drama", "Psychological", "Slice of Life"],
        "joy": ["Comedy", "Romance", "Slice of Life"],
        "happy": ["Comedy", "Romance", "Slice of Life"],
        "love": ["Romance", "Drama", "Slice of Life"],
        "anger": ["Action", "Adventure", "Fantasy"],
        "angry": ["Action", "Adventure", "Fantasy"],
        "determined": ["Action", "Adventure", "Fantasy", "Drama"],
        "fear": ["Thriller", "Mystery", "Psychological", "Supernatural"],
        "anxious": ["Thriller", "Mystery", "Psychological", "Supernatural"],
        "surprise": ["Mystery", "Supernatural", "Thriller"]
    }
    
    target_genres = [g.lower() for g in genre_mapping.get(emotion.lower(), ["Action", "Adventure", "Drama"])]
    
    def tokenize(text: str) -> set:
        return set(re.findall(r'\w+', text.lower()))
        
    prompt_words = tokenize(user_prompt)
    if not prompt_words:
        prompt_words = tokenize(emotion)
        
    ranked_candidates = []
    
    if os.path.exists(csv_path):
        try:
            with open(csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    genres_str = row.get("genre", "").lower()
                    row_genres = [g.strip() for g in genres_str.split(",")]
                    # Check if matching categories
                    if any(tg in row_genres for tg in target_genres):
                        title = row.get("title", "")
                        desc = row.get("description", "")
                        combined_text = f"{title} {desc}".lower()
                        row_words = tokenize(combined_text)
                        
                        # Match overlap score
                        overlap_score = len(prompt_words.intersection(row_words))
                        title_words = tokenize(title)
                        title_score = len(prompt_words.intersection(title_words)) * 5
                        total_match = overlap_score + title_score
                        
                        pop_score = float(row.get("popularity_score") or 50.0)
                        
                        # Final rank score prioritizing overlap matches first, then popularity
                        rank_score = (total_match * 1000) + pop_score
                        ranked_candidates.append((rank_score, row))
        except Exception as e:
            log.warning(f"Error ranking references: {e}")
            
    if ranked_candidates:
        # Sort by rank_score descending
        ranked_candidates.sort(key=lambda x: x[0], reverse=True)
        chosen = ranked_candidates[0][1]
        log.info(f"[Mood Weaver] Deterministic Ranker selected title '{chosen.get('title')}' with score {ranked_candidates[0][0]:.1f}")
        return {
            "title": chosen.get("title"),
            "genre": chosen.get("genre"),
            "description": chosen.get("description"),
            "source": chosen.get("source")
        }
        
    # Standard fallback
    return {
        "title": "Naruto",
        "genre": "Action, Adventure, Fantasy",
        "description": "Ninja shinobi adventures.",
        "source": "Anime/Manga"
    }


def get_character_for_franchise(title: str, emotion: str) -> str:
    title_lower = title.lower()
    if "naruto" in title_lower: return "Naruto"
    if "death note" in title_lower: return "Light Yagami"
    if "attack on titan" in title_lower or "shingeki no kyojin" in title_lower: return "Eren Yeager"
    if "demon slayer" in title_lower or "kimetsu no yaiba" in title_lower: return "Tanjiro"
    if "fullmetal alchemist" in title_lower: return "Edward Elric"
    if "my hero academia" in title_lower or "boku no hero" in title_lower: return "Midoriya"
    if "assassination classroom" in title_lower: return "Nagisa"
    if "your name" in title_lower or "kimi no na wa" in title_lower: return "Taki"
    if "silent voice" in title_lower or "koe no katachi" in title_lower: return "Shoya"
    if "your lie in april" in title_lower: return "Kousei"
    if "re:zero" in title_lower: return "Subaru"
    if "jujutsu kaisen" in title_lower: return "Yuji Itadori"
    if "evangelion" in title_lower: return "Shinji Ikari"
    if "hunter x hunter" in title_lower: return "Gon Freecss"
    if "one piece" in title_lower: return "Luffy"
    
    # Fallback to emotion characters
    emotion_chars = {
        "grief": "Aria", "heaviness": "Kora", "sadness": "Aria",
        "determined": "Captain Vance",
        "love": "Lyra", "trust": "Kael",
        "anxious": "Zephyr", "fear": "Kiri",
        "angry": "Valen", "rage": "Valen"
    }
    return emotion_chars.get(emotion.lower(), "Wanderer")