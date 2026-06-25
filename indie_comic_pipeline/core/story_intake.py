"""
STORY INTAKE ENGINE — Phase 0
================================
Accepts raw user emotion/story prompt, processes it through the
Story-Weaver LLM to parse thematic elements, emotional pacing,
and structural flow. Outputs a structured story configuration.

Uses local Ollama for LLM inference.
"""

import json
import os
import sys
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

log = logging.getLogger("pipeline.story_intake")


# Default mood arc definitions (mirrored from Story-Weaver)
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

DEFAULT_ARC = {
    "journey": "reflective",
    "description": "From feeling toward witnessing",
    "arc_beats": ["acknowledgment", "presence", "shift", "openness"],
}


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
        """Lazy-load the Ollama LLM connection."""
        if self._llm is None:
            try:
                from langchain_ollama import ChatOllama
                self._llm = ChatOllama(
                    model=self.ollama_model,
                    temperature=0.3,
                    base_url=self.ollama_url,
                )
                log.info(f"Connected to Ollama: {self.ollama_model}")
            except ImportError:
                log.warning("langchain_ollama not installed — using fallback story generation")
                self._llm = None
        return self._llm

    def process_prompt(self, user_prompt: str,
                       panel_count: int = 6,
                       character_name: str = "Wanderer",
                       story_world: str = "The Abstract",
                       style_reference: str = "",
                       character_characteristics: str = "",
                       story_reference: str = "",
                       mood_shifts: List[str] = None) -> Dict[str, Any]:
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
        panel_count = max(4, min(10, panel_count))
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
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Story file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        log.info(f"[Phase 0] Loaded existing story from {path}")
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
                           mood_shifts: List[str] = None) -> Optional[Dict[str, Any]]:
        """Generate story using local Ollama LLM."""
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        import re

        if mood_shifts and len(mood_shifts) > 0:
            beats = self._distribute_beats(panel_count, mood_shifts)
            arc = {"journey": "Custom User Arc", "description": "Custom sequence of emotions provided by user"}
        else:
            arc = MOOD_ARCS.get(emotion, DEFAULT_ARC)
            beats = self._distribute_beats(panel_count, arc["arc_beats"])

        style_str = f"STYLE ENFORCEMENT: The user has requested the exact visual style of: '{style_reference}'. Every single visual prompt you generate MUST include this style instruction and maintain its specific color palette and theme.\n" if style_reference else ""
        char_str = f"CHARACTER CHARACTERISTICS: The main character is '{character_name}'. Here are their physical and personality traits: '{character_characteristics}'. Maintain these consistently in visual descriptions.\n" if character_characteristics else ""
        story_ref_str = f"STORY REFERENCE & THEME: The user requested the story to follow the thematic tone and style of: '{story_reference}'. You MUST adapt the narrative pacing, tropes, and dialogue style to match this reference.\n" if story_reference else ""

        system_prompt = f"""You are a master graphic novelist and cinematic director, renowned for writing gripping, emotionally resonant, and visually stunning comic book scripts.
Respond ONLY with valid JSON. No markdown fences. No explanation.

Generate a {panel_count}-panel comic story based on the user's emotional prompt.
Your writing must be professional: use "show, don't tell" for visuals, write cinematic and highly descriptive visual prompts (lighting, camera angles, atmosphere), and craft punchy, natural, and compelling dialogue.

{style_str}
{char_str}
{story_ref_str}
INTELLIGENT GENERATION CHAIN (WRITER'S ROOM):
You must output a JSON object containing two main sections:
1. "story_bible": Deconstruct the story reference to generate a plot_summary and 1-2 thematic side_characters.
2. "panels": Write the panel-by-panel script as a HIERARCHICAL SCENE GRAPH. Do not use flat text fields.

Every panel in the "panels" list MUST follow this exact schema:
- "characters": Array of entities present in the panel. Each character must have:
    - "id": Lowercase name (e.g. "kael")
    - "pose": Object with "body", "head", "arms", "legs" describing physical stance.
    - "expression": Object with "emotion", "eyes", "mouth" describing facial state.
    - "dialogue": Object with "text" (or "..."), "tone", and "bubble" (e.g. "speech", "thought", "shout").
- "actions": Array of events happening (e.g., {{"actor": "kael", "verb": "looking", "target": "artifact"}}).
- "camera": Cinematic framing (e.g. "Low-angle medium shot").
- "environment": The background setting and lighting.

Output this exact JSON structure:
{{
  "story_bible": {{
    "plot_summary": "...",
    "side_characters": [ {{"name": "...", "role": "...", "description": "..."}} ]
  }},
  "recurring_motif": "...",
  "mood_journey": "...",
  "panels": [
    {{
      "panel": 1,
      "characters": [
        {{
          "id": "kael",
          "pose": {{"body": "...", "head": "...", "arms": "...", "legs": "..."}},
          "expression": {{"emotion": "...", "eyes": "...", "mouth": "..."}},
          "dialogue": {{"text": "...", "tone": "...", "bubble": "..."}}
        }}
      ],
      "actions": [ {{"actor": "...", "verb": "...", "target": "..."}} ],
      "camera": "...",
      "environment": "..."
    }}
  ]
}}"""

        beat_guide = "\n".join(
            f"  Panel {i+1} → emotion_beat should evoke: {beats[i]}"
            for i in range(panel_count)
        )

        user_msg = (
            f"Emotion: {emotion}\n"
            f"Story world: {story_world}\n"
            f"Character: {character_name}\n"
            f"User prompt: \"{user_prompt}\"\n\n"
            f"MOOD JOURNEY: {arc['journey']} — {arc['description']}\n\n"
            f"Write exactly {panel_count} panels following this arc:\n"
            f"{beat_guide}\n\n"
            f"Write the JSON now."
        )

        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_msg),
        ]
        response = self._llm.invoke(messages).content

        # Extract JSON from response
        clean = re.sub(r"^```(?:json)?\s*", "", response.strip())
        clean = re.sub(r"\s*```$", "", clean).strip()
        start = clean.find("{")
        if start == -1:
            return None

        depth, end = 0, -1
        for i, ch in enumerate(clean[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if end == -1:
            return None

        json_str = clean[start:end + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            log.warning(f"Initial JSON parse failed: {e}. Attempting cleanup...")
            # Clean trailing commas and newlines
            import re
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*\]', ']', json_str)
            json_str = json_str.replace('\n', ' ').replace('\r', ' ')
            return json.loads(json_str)

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

        panels = []
        for i in range(panel_count):
            beat = beats[i]
            progress = i / max(1, panel_count - 1)
            intensity = 0.3 + 0.4 * abs(progress - 0.5) * 2

            panels.append({
                "panel": i + 1,
                "characters": [
                    {
                        "id": character_name.lower(),
                        "pose": {"body": "neutral", "head": "neutral", "arms": "relaxed", "legs": "standing"},
                        "expression": {"emotion": beat, "eyes": "neutral", "mouth": "neutral"},
                        "dialogue": {"text": "..." if i % 2 == 0 else f"{character_name}: ...", "tone": "neutral", "bubble": "speech"}
                    }
                ],
                "actions": [
                    {"actor": character_name.lower(), "verb": "expresses", "target": beat}
                ],
                "camera": "Medium shot",
                "environment": f"Scene {i+1} in {story_world}",
                "_action_intensity": intensity
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
            return max(scores, key=scores.get)
        return "determined"  # Default for action/adventure prompts

    def _distribute_beats(self, n: int, beats: list) -> list:
        """Distribute arc beats across N panels."""
        if n <= len(beats):
            step = len(beats) / n
            return [beats[int(i * step)] for i in range(n)]
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
