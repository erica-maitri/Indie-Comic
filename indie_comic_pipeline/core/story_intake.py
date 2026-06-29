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
        self.ollama_model = ollama_model
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
                       mood_shifts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process a raw user prompt into a structured story configuration.

        Args:
            user_prompt: Raw emotional / story prompt from user
            panel_count: Number of panels to generate (4-10)
            character_name: Main character name
            story_world: Story world/setting name

        Returns:
            Structured story config dict with:
            - recurring_motif, mood_journey, panels[], metadata
        """
        panel_count = max(1, min(10, panel_count))
        log.info(f"[Phase 0] Processing prompt: '{user_prompt[:80]}...'")
        log.info(f"  Character: {character_name}, World: {story_world}, Panels: {panel_count}")

        # Detect primary emotion from the prompt
        emotion = self._detect_emotion(user_prompt)
        log.info(f"  Detected emotion: {emotion}")

        # Try LLM-based story generation first
        llm = self._get_llm()
        if llm is not None:
            try:
                story_config = self._generate_with_llm(
                    user_prompt, emotion, panel_count, character_name, story_world, style_reference,
                    character_characteristics, story_reference, mood_shifts
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
                           mood_shifts: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Generate story using local Ollama LLM."""
        import re

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

        system_prompt = f"""You are a master graphic novelist and cinematic director.
Respond ONLY with a single valid JSON object. No markdown fences, no explanation, no comments.

{style_str}{char_str}{story_ref_str}
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

3. pose.body field: MUST include character's clothing/costume AND body position.
   Example: "crouching forward in torn grey hoodie and black cargo pants, weight on left foot"

4. dialogue.text: Write REAL dialogue. Punchy. Max 12 words. No placeholders.
   Characters speak like real people under emotional pressure.

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
          "pose": {{"body": "clothing + physical stance", "head": "head direction", "arms": "arm position", "legs": "leg position"}},
          "expression": {{"emotion": "specific emotion", "eyes": "eye description", "mouth": "mouth position"}},
          "dialogue": {{"text": "Actual spoken words.", "tone": "tone descriptor", "bubble": "speech|thought|shout|whisper"}}
        }}
      ],
      "actions": [ {{"actor": "character_id", "verb": "action verb", "target": "what/whom"}} ],
      "camera": "angle + movement descriptor",
      "environment": "location, time, dominant palette, light source"
    }}
  ]
}}"""

        beat_guide = "\n".join(
            f"  Panel {i+1}: emotion_beat = \"{beats[i]}\""
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
                    {"actor": character_name.lower(), "verb": "moves through", "target": beat}
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
        """Detect primary emotion from user text using keyword matching."""
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
