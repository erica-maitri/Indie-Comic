"""
LAYOUT AGENT — Phase 1
========================
Defines camera framing parameters, angles, and panel geometry limits.
Determines dynamic panel sizing based on scene action intensity.
Writes LayoutDirectives into the Story Section Memory blackboard.
"""

from typing import Dict, Any, List, Tuple, TYPE_CHECKING
from core.agents.base_agent import BaseAgent
from core.memory import LayoutDirective

if TYPE_CHECKING:
    from core.memory import StorySectionMemory


# Camera angle selection rules based on pacing phase and intensity
CAMERA_RULES = {
    # (pacing_phase, intensity_range) → camera settings
    ("setup", "low"):         {"angle": "wide_shot", "framing": "center", "size": "medium"},
    ("setup", "medium"):      {"angle": "medium_shot", "framing": "left_third", "size": "medium"},
    ("setup", "high"):        {"angle": "medium_shot", "framing": "center", "size": "medium"},
    ("rising_action", "low"): {"angle": "medium_shot", "framing": "right_third", "size": "medium"},
    ("rising_action", "medium"): {"angle": "close_up", "framing": "center", "size": "medium"},
    ("rising_action", "high"):   {"angle": "close_up", "framing": "center", "size": "large"},
    ("climax", "low"):        {"angle": "medium_shot", "framing": "center", "size": "medium"},
    ("climax", "medium"):     {"angle": "close_up", "framing": "center", "size": "large"},
    ("climax", "high"):       {"angle": "bird_eye", "framing": "center", "size": "full_page"},
    ("resolution", "low"):    {"angle": "wide_shot", "framing": "center", "size": "medium"},
    ("resolution", "medium"): {"angle": "wide_shot", "framing": "center", "size": "medium"},
    ("resolution", "high"):   {"angle": "medium_shot", "framing": "center", "size": "large"},
}

# Emotion beat overrides for camera angle
BEAT_CAMERA_OVERRIDES = {
    "close_up_beats": {
        "fracture", "ache", "memory", "held",
        "pause", "breath", "spark", "overflow",
    },
    "wide_shot_beats": {
        "transcendence", "triumph", "renewal", "carried_forward",
        "quiet_hope", "ground", "present",
    },
    "full_page_beats": {
        "breakthrough", "climax", "explosion",
    },
}


class LayoutAgent(BaseAgent):
    """
    Determines camera framing, panel sizing, and geometry for each panel.

    Responsibilities:
    - Map pacing phases + intensity → camera angles
    - Apply emotion beat overrides for dramatic moments
    - Determine panel sizes (small/medium/large/full_page)
    - Calculate aspect ratios based on content type
    - Write LayoutDirectives to the memory blackboard
    """

    def __init__(self):
        super().__init__("layout")

    def plan(self, story_config: Dict[str, Any],
             memory: "StorySectionMemory") -> Dict[str, Any]:
        """
        Create layout directives for all panels based on storyboard data.
        """
        page_plans = memory.page_plans
        all_directives = []
        panel_counter = 0

        for page_plan in page_plans:
            pacing = page_plan.get("pacing_phase", "setup")
            panels = page_plan.get("panels", [])
            avg_intensity = page_plan.get("avg_intensity", 0.5)

            for i, panel in enumerate(panels):
                panel_counter += 1
                panel_id = panel_counter
                beat = panel.get("emotion_beat", "neutral")
                panel_intensity = panel.get("_action_intensity",
                                            self._get_beat_intensity(beat, avg_intensity))

                # Get camera settings from rules
                intensity_level = self._classify_intensity(panel_intensity)
                rule_key = (pacing, intensity_level)
                camera = CAMERA_RULES.get(rule_key, CAMERA_RULES[("setup", "medium")])

                # Apply beat overrides
                angle = camera["angle"]
                size = camera["size"]

                if beat in BEAT_CAMERA_OVERRIDES.get("close_up_beats", set()):
                    angle = "close_up"
                elif beat in BEAT_CAMERA_OVERRIDES.get("wide_shot_beats", set()):
                    angle = "wide_shot"
                elif beat in BEAT_CAMERA_OVERRIDES.get("full_page_beats", set()):
                    angle = "bird_eye"
                    size = "full_page"

                # Determine aspect ratio
                aspect = self._get_aspect_ratio(angle, size)

                # Gutter emphasis based on transition
                gutter = "normal"
                if i == 0 and pacing in ("climax", "resolution"):
                    gutter = "wide"  # Breathing room at scene transitions
                elif panel_intensity > 0.8:
                    gutter = "tight"  # Quick cuts for action

                directive = LayoutDirective(
                    panel_id=panel_id,
                    size_class=size,
                    camera_angle=angle,
                    camera_framing=camera["framing"],
                    aspect_ratio=aspect,
                    gutter_emphasis=gutter,
                )

                # Write to memory
                memory.set_layout_directive(panel_id, directive)
                all_directives.append(directive)

        self._state = {
            "num_directives": len(all_directives),
            "size_distribution": self._compute_distribution(all_directives),
        }

        self.log.info(
            f"Layout planned: {len(all_directives)} panel directives, "
            f"distribution={self._state['size_distribution']}"
        )

        return {"layout_directives": [d.to_dict() for d in all_directives]}

    def update(self, panel_result: Dict[str, Any],
               memory: "StorySectionMemory"):
        """Layout agent doesn't need post-generation updates."""
        pass

    def _classify_intensity(self, intensity: float) -> str:
        """Classify intensity value into low/medium/high."""
        if intensity < 0.35:
            return "low"
        elif intensity < 0.65:
            return "medium"
        else:
            return "high"

    def _get_beat_intensity(self, beat: str, default: float) -> float:
        """Estimate intensity from a beat if no explicit value available."""
        high_beats = {"breakthrough", "contained_fire", "fracture", "peak_noise", "momentum", "triumph"}
        low_beats = {"stillness", "quiet_hope", "drift", "quiet_rest", "softness", "fade"}

        if beat in high_beats:
            return 0.85
        elif beat in low_beats:
            return 0.2
        return default

    def _get_aspect_ratio(self, angle: str, size: str) -> Tuple[int, int]:
        """Determine aspect ratio based on camera angle and panel size."""
        if size == "full_page":
            return (2, 3)  # Portrait full page
        elif angle == "wide_shot":
            return (16, 9)  # Cinematic widescreen
        elif angle == "close_up":
            return (1, 1)  # Square for faces
        elif angle == "bird_eye":
            return (3, 2)  # Landscape
        else:
            return (1, 1)  # Default square

    def _compute_distribution(self, directives: List[LayoutDirective]) -> Dict[str, int]:
        """Count panel size distribution."""
        dist = {"small": 0, "medium": 0, "large": 0, "full_page": 0}
        for d in directives:
            if d.size_class in dist:
                dist[d.size_class] += 1
        return dist
