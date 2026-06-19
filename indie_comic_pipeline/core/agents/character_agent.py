"""
CHARACTER AGENT — Phase 1
===========================
Builds character visual profiles and tracks emotional arc drift.
Maintains a character visual descriptor that evolves with the narrative.
Registers characters into the Story Section Memory blackboard.
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
from core.agents.base_agent import BaseAgent

if TYPE_CHECKING:
    from core.memory import StorySectionMemory


# Emotion-to-expression mapping for visual prompt enrichment
EMOTION_EXPRESSIONS = {
    "heaviness":       "weary eyes, slumped shoulders, slow breathing",
    "stillness":       "blank gaze, motionless posture, tight jaw",
    "faint_warmth":    "softening eyes, loosening shoulders, deeper breath",
    "tentative_light": "slight upturn of mouth, relaxed brow, open posture",
    "soft_openness":   "gentle gaze, open hands, relaxed chest",
    "quiet_hope":      "calm eyes, steady breath, hint of a smile",
    "contained_fire":  "clenched jaw, rigid posture, intense stare",
    "fracture":        "cracking composure, trembling hands, tight throat",
    "exhale":          "releasing breath, loosening fists, dropping shoulders",
    "cooling":         "slower movements, calming expression, deeper breathing",
    "ground":          "planted feet, steady stance, measured breathing",
    "drag":            "heavy limbs, unfocused eyes, minimal movement",
    "surrender":       "letting go, closing eyes, body settling",
    "softness":        "relaxed muscles, gentle expression, slow movement",
    "drift":           "floating movement, distant gaze, peaceful face",
    "quiet_rest":      "sleeping expression, peaceful face, still body",
    "renewal":         "opening eyes, taking deep breath, sitting up",
    "spark":           "brightening eyes, slight lean forward, alert posture",
    "expansion":       "open posture, lifted chin, broadening smile",
    "overflow":        "laughing, tears of joy, expressive gestures",
    "radiance":        "glowing expression, uplifted face, open arms",
    "luminous_still":  "serene smile, closed eyes, perfect stillness",
    "transcendence":   "ethereal calm, distant gaze, weightless posture",
    "spiral":          "darting eyes, gripping hands, rapid breathing",
    "peak_noise":      "squeezed eyes, rocking body, overwhelmed expression",
    "pause":           "attention caught, breath held, sudden focus",
    "breath":          "deliberate exhale, conscious breathing, slowing down",
    "root":            "grounded stance, palms on floor, anchored body",
    "present":         "aware gaze, relaxed face, conscious presence",
    "absence":         "hollow eyes, empty gaze, body curled inward",
    "ache":            "held tears, tight throat, reaching hand",
    "memory":          "distant tender gaze, touching an object, half-smile",
    "held":            "accepting tears, softened posture, open chest",
    "continuance":     "moving through space, carrying something, walking on",
    "carried_forward": "steady steps, objects in hand, looking ahead",
    "doubt":           "furrowed brow, hesitant posture, looking down",
    "challenge":       "bracing stance, determined glare, clenched fists",
    "resistance":      "pushing back, gritted teeth, wide stance",
    "breakthrough":    "explosive movement, fierce eyes, forward surge",
    "momentum":        "running, charging, dynamic action pose",
    "triumph":         "raised arms, victorious expression, powerful stance",
    "fade":            "fading presence, dissolving edges, quiet withdrawal",
}


class CharacterAgent(BaseAgent):
    """
    Builds and maintains character visual profiles across the story.

    Responsibilities:
    - Parse character info from story config
    - Build visual descriptors for SDXL prompt construction
    - Track emotional state drift panel-by-panel
    - Register characters into the memory blackboard
    """

    def __init__(self):
        super().__init__("character")

    def plan(self, story_config: Dict[str, Any],
             memory: "StorySectionMemory") -> Dict[str, Any]:
        """
        Build character profiles from story config and register them in memory.

        Extracts character name, builds initial visual descriptor,
        and maps out the emotional arc for each character.
        """
        metadata = story_config.get("_metadata", {})
        character_name = metadata.get("character",
                                      memory.story_config.get("_metadata", {}).get("character", "Wanderer"))
        story_world = metadata.get("world",
                                   memory.story_config.get("_metadata", {}).get("world", "The Abstract"))

        # Build emotional arc from panel beats
        panels = story_config.get("panels", [])
        emotional_arc = [p.get("emotion_beat", "neutral") for p in panels]

        # Check if character has a custom costume in the config list
        characters_list = story_config.get("characters", [])
        custom_costume = ""
        for c in characters_list:
            if isinstance(c, dict) and c.get("name") == character_name:
                custom_costume = c.get("costume", c.get("costume_desc", ""))
                break

        # Build visual descriptor base
        if custom_costume:
            costume_desc = custom_costume
        else:
            costume_desc = self._build_costume_descriptor(character_name, story_world)

        # Register main character
        memory.register_character(
            name=character_name,
            emotion=emotional_arc[0] if emotional_arc else "neutral",
            costume_desc=costume_desc,
            arc_phase="introduction",
        )

        # Build per-panel character directives
        panel_directives = []
        for i, panel in enumerate(panels):
            beat = panel.get("emotion_beat", "neutral")
            expression = EMOTION_EXPRESSIONS.get(beat, "neutral expression, composed posture")

            # Determine arc phase
            progress = i / max(1, len(panels) - 1)
            if progress < 0.2:
                arc_phase = "introduction"
            elif progress < 0.5:
                arc_phase = "development"
            elif progress < 0.8:
                arc_phase = "climax"
            else:
                arc_phase = "resolution"

            directive = {
                "panel": i + 1,
                "character_name": character_name,
                "emotion_beat": beat,
                "expression_prompt": expression,
                "arc_phase": arc_phase,
                "costume_desc": costume_desc,
                "visual_note": f"{character_name}, {expression}, {costume_desc}",
            }
            panel_directives.append(directive)

        self._state = {
            "character_name": character_name,
            "emotional_arc": emotional_arc,
            "costume_desc": costume_desc,
            "panel_directives": panel_directives,
        }

        self.log.info(
            f"Character '{character_name}' profiled: "
            f"{len(emotional_arc)} emotion beats mapped"
        )

        return {"panel_directives": panel_directives}

    def update(self, panel_result: Dict[str, Any],
               memory: "StorySectionMemory"):
        """Update character state after panel generation."""
        panel_id = panel_result.get("panel_id", 0)
        char_name = self._state.get("character_name", "Wanderer")

        # Find the directive for this panel
        directives = self._state.get("panel_directives", [])
        directive = None
        for d in directives:
            if d["panel"] == panel_id:
                directive = d
                break

        if directive:
            memory.update_character(
                char_name,
                emotion=directive["emotion_beat"],
                arc_phase=directive["arc_phase"],
                last_action=directive.get("expression_prompt", ""),
            )

            # Track panel appearances
            char = memory.get_character(char_name)
            if char and panel_id not in char.panel_appearances:
                char.panel_appearances.append(panel_id)

    def get_expression_for_panel(self, panel_id: int) -> str:
        """Get the expression prompt for a specific panel."""
        directives = self._state.get("panel_directives", [])
        for d in directives:
            if d["panel"] == panel_id:
                return d.get("expression_prompt", "neutral expression")
        return "neutral expression, composed posture"

    def get_visual_note_for_panel(self, panel_id: int) -> str:
        """Get the full visual note for a specific panel."""
        directives = self._state.get("panel_directives", [])
        for d in directives:
            if d["panel"] == panel_id:
                return d.get("visual_note", "")
        return ""

    def _build_costume_descriptor(self, character_name: str,
                                  story_world: str) -> str:
        """Build a visual costume descriptor for the character in the story world."""
        # Context-aware costume generation
        world_lower = story_world.lower()

        if "cyberpunk" in world_lower:
            return (f"{character_name} wearing a high-collared dark jacket with neon-trim "
                    f"accents, tactical gear, cybernetic visor, urban tech aesthetic")
        elif "noir" in world_lower or "gothic" in world_lower:
            return (f"{character_name} in a long dark coat, fedora hat, "
                    f"high contrast noir aesthetic, sharp shadows")
        elif "fantasy" in world_lower or "medieval" in world_lower:
            return (f"{character_name} in weathered leather armor, flowing cape, "
                    f"runic accessories, fantasy adventurer aesthetic")
        elif "space" in world_lower or "sci-fi" in world_lower:
            return (f"{character_name} in sleek spacesuit with glowing panels, "
                    f"helmet visor, zero-gravity aesthetic")
        else:
            return (f"{character_name} in distinctive clothing appropriate for "
                    f"{story_world}, consistent visual design, recognizable silhouette")
