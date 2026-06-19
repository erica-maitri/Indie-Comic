"""
SCENE AGENT — Phase 1
=======================
Tracks environmental settings, lighting conditions, and location continuity.
Ensures spatial consistency across panels by maintaining scene state
in the Story Section Memory blackboard.
"""

from typing import Dict, Any, List, TYPE_CHECKING
from core.agents.base_agent import BaseAgent

if TYPE_CHECKING:
    from core.memory import StorySectionMemory


# Emotion beat to lighting/atmosphere mapping
BEAT_ATMOSPHERES = {
    "heaviness":       {"lighting": "dim overcast", "weather": "heavy rain", "mood_color": "dark slate blue"},
    "stillness":       {"lighting": "flat muted", "weather": "drizzle", "mood_color": "grey"},
    "faint_warmth":    {"lighting": "warm lamp glow", "weather": "clearing", "mood_color": "soft amber"},
    "tentative_light": {"lighting": "early dawn light", "weather": "mist breaking", "mood_color": "pale gold"},
    "soft_openness":   {"lighting": "gentle diffused", "weather": "calm", "mood_color": "warm white"},
    "quiet_hope":      {"lighting": "golden hour", "weather": "clear", "mood_color": "warm gold"},
    "contained_fire":  {"lighting": "harsh overhead", "weather": "still", "mood_color": "deep crimson"},
    "fracture":        {"lighting": "flickering", "weather": "wind picking up", "mood_color": "angry orange"},
    "exhale":          {"lighting": "softening", "weather": "wind dying", "mood_color": "cool blue"},
    "cooling":         {"lighting": "twilight", "weather": "evening calm", "mood_color": "indigo"},
    "ground":          {"lighting": "steady", "weather": "clear night", "mood_color": "earth brown"},
    "drag":            {"lighting": "flat morning", "weather": "overcast", "mood_color": "washed grey"},
    "surrender":       {"lighting": "dim ambient", "weather": "quiet", "mood_color": "soft grey"},
    "softness":        {"lighting": "warm filtered", "weather": "afternoon sun", "mood_color": "cream"},
    "drift":           {"lighting": "hazy", "weather": "warm breeze", "mood_color": "lavender"},
    "quiet_rest":      {"lighting": "moonlight", "weather": "silent night", "mood_color": "deep blue"},
    "renewal":         {"lighting": "fresh morning", "weather": "clear dawn", "mood_color": "sunrise pink"},
    "spark":           {"lighting": "sudden bright", "weather": "clear", "mood_color": "bright yellow"},
    "expansion":       {"lighting": "growing radiance", "weather": "sunny", "mood_color": "golden"},
    "overflow":        {"lighting": "brilliant", "weather": "clear sky", "mood_color": "vivid warm"},
    "radiance":        {"lighting": "peak golden hour", "weather": "perfect", "mood_color": "pure gold"},
    "luminous_still":  {"lighting": "fading golden", "weather": "twilight", "mood_color": "amber"},
    "transcendence":   {"lighting": "ethereal glow", "weather": "surreal calm", "mood_color": "white gold"},
    "spiral":          {"lighting": "harsh fluorescent", "weather": "still dark", "mood_color": "sickly green"},
    "peak_noise":      {"lighting": "strobing", "weather": "oppressive", "mood_color": "electric white"},
    "pause":           {"lighting": "single point light", "weather": "still", "mood_color": "cool grey"},
    "breath":          {"lighting": "steady dim", "weather": "gentle", "mood_color": "soft blue"},
    "root":            {"lighting": "low warm", "weather": "quiet", "mood_color": "earth tone"},
    "present":         {"lighting": "early grey light", "weather": "dawn", "mood_color": "silver"},
    "absence":         {"lighting": "cold empty", "weather": "still", "mood_color": "hollow grey"},
    "ache":            {"lighting": "dim corner", "weather": "threatening", "mood_color": "bruise purple"},
    "memory":          {"lighting": "warm nostalgic", "weather": "afternoon", "mood_color": "sepia"},
    "held":            {"lighting": "soft warm", "weather": "gentle", "mood_color": "warm rose"},
    "continuance":     {"lighting": "practical daylight", "weather": "normal", "mood_color": "neutral warm"},
    "carried_forward": {"lighting": "evening lamp", "weather": "calm evening", "mood_color": "soft orange"},
    "doubt":           {"lighting": "shadowed", "weather": "cloudy", "mood_color": "dark grey"},
    "challenge":       {"lighting": "dramatic side light", "weather": "windy", "mood_color": "steel blue"},
    "resistance":      {"lighting": "harsh contrast", "weather": "storm approaching", "mood_color": "dark red"},
    "breakthrough":    {"lighting": "explosive bright", "weather": "storm breaking", "mood_color": "electric blue"},
    "momentum":        {"lighting": "dynamic", "weather": "rushing wind", "mood_color": "bright cyan"},
    "triumph":         {"lighting": "heroic backlight", "weather": "clearing after storm", "mood_color": "brilliant gold"},
}

DEFAULT_ATMOSPHERE = {"lighting": "natural", "weather": "clear", "mood_color": "neutral"}


class SceneAgent(BaseAgent):
    """
    Tracks environmental state and ensures spatial continuity.

    Responsibilities:
    - Map emotion beats to atmospheric conditions (lighting, weather, color)
    - Track location changes across panels
    - Maintain props and recurring motif presence
    - Write scene state updates to the memory blackboard
    """

    def __init__(self):
        super().__init__("scene")

    def plan(self, story_config: Dict[str, Any],
             memory: "StorySectionMemory") -> Dict[str, Any]:
        """
        Build scene directives for each panel based on story config.
        Sets initial scene state in the memory blackboard.
        """
        panels = story_config.get("panels", [])
        recurring_motif = story_config.get("recurring_motif", "")
        metadata = story_config.get("_metadata", {})
        story_world = metadata.get("world", "The Abstract")

        # Build per-panel scene directives
        scene_directives = []
        for i, panel in enumerate(panels):
            beat = panel.get("emotion_beat", "neutral")
            atmo = BEAT_ATMOSPHERES.get(beat, DEFAULT_ATMOSPHERE)

            # Extract location from panel visual if available
            visual = panel.get("visual", "")
            location = self._extract_location(visual, story_world)

            # Check if motif is mentioned in visual
            motif_present = recurring_motif.lower() in visual.lower() if recurring_motif else False

            directive = {
                "panel": i + 1,
                "location": location,
                "lighting": atmo["lighting"],
                "weather": atmo["weather"],
                "mood_color": atmo["mood_color"],
                "recurring_motif": recurring_motif,
                "motif_present": motif_present,
                "atmosphere_prompt": (
                    f"Scene atmosphere: {atmo['lighting']} lighting, "
                    f"{atmo['weather']} weather, dominant mood color {atmo['mood_color']}. "
                    f"{'The ' + recurring_motif + ' is visible.' if motif_present else ''}"
                ),
            }
            scene_directives.append(directive)

        # Set initial scene state in memory
        if scene_directives:
            first = scene_directives[0]
            memory.update_scene(
                location=first["location"],
                lighting=first["lighting"],
                weather=first["weather"],
                mood_color=first["mood_color"],
                recurring_motif=recurring_motif,
            )

        self._state = {
            "story_world": story_world,
            "recurring_motif": recurring_motif,
            "scene_directives": scene_directives,
        }

        self.log.info(
            f"Scene planned: {len(scene_directives)} atmosphere mappings, "
            f"motif='{recurring_motif}'"
        )

        return {"scene_directives": scene_directives}

    def update(self, panel_result: Dict[str, Any],
               memory: "StorySectionMemory"):
        """Update scene state after panel generation."""
        panel_id = panel_result.get("panel_id", 0)
        directives = self._state.get("scene_directives", [])

        for d in directives:
            if d["panel"] == panel_id:
                memory.update_scene(
                    location=d["location"],
                    lighting=d["lighting"],
                    weather=d["weather"],
                    mood_color=d["mood_color"],
                )
                break

    def get_atmosphere_for_panel(self, panel_id: int) -> str:
        """Get the atmosphere prompt for a specific panel."""
        directives = self._state.get("scene_directives", [])
        for d in directives:
            if d["panel"] == panel_id:
                return d.get("atmosphere_prompt", "")
        return ""

    def _extract_location(self, visual_text: str, default_world: str) -> str:
        """Extract location from panel visual description."""
        location_keywords = [
            "kitchen", "bedroom", "street", "alley", "rooftop",
            "forest", "city", "room", "corridor", "bridge",
            "market", "temple", "cave", "garden", "shore",
            "window", "doorway", "balcony", "staircase",
        ]

        visual_lower = visual_text.lower()
        for kw in location_keywords:
            if kw in visual_lower:
                return kw.capitalize()

        return default_world
