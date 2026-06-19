"""
STORYBOARD AGENT — Phase 1
============================
Establishes sequence beats, narrative pacing, and scene breaks.
Determines panel count per page, page breaks, and emotional arc phases.
Writes page_plans into the Story Section Memory blackboard.
"""

from typing import Dict, Any, List, TYPE_CHECKING
from core.agents.base_agent import BaseAgent

if TYPE_CHECKING:
    from core.memory import StorySectionMemory


class StoryboardAgent(BaseAgent):
    """
    Breaks down the structured story config into a paginated storyboard.
    Determines which panels go on which pages, assigns emotional beats,
    and establishes pacing rhythm.
    """

    def __init__(self):
        super().__init__("storyboard")

    def plan(self, story_config: Dict[str, Any],
             memory: "StorySectionMemory") -> Dict[str, Any]:
        """
        Create a paginated storyboard plan from the story config.

        Writes to memory:
        - page_plans: list of page plan dicts
        - total_panels, total_pages
        - arc_beats, mood_journey, recurring_motif
        """
        panels = list(story_config.get("panels", []))  # Copy to avoid mutating original
        num_panels = len(panels)

        # Pad panels to multiple of 4 for page layout
        while len(panels) % 4 != 0:
            last = panels[-1] if panels else {
                "visual": "Fade to quiet.",
                "dialogue": "...",
                "emotion_beat": "stillness",
                "motion": "fading",
            }
            panels.append({
                "panel": len(panels) + 1,
                "visual": f"{last['visual']} (transition)",
                "dialogue": "...",
                "emotion_beat": "fade",
                "motion": "slow transition",
            })

        num_panels = len(panels)
        num_pages = num_panels // 4

        # Build page plans
        page_plans = []
        for page_idx in range(num_pages):
            page_num = page_idx + 1
            page_start = page_idx * 4
            page_panels = panels[page_start:page_start + 4]

            # Determine page pacing phase
            progress = page_idx / max(1, num_pages - 1)
            if progress < 0.25:
                pacing_phase = "setup"
            elif progress < 0.5:
                pacing_phase = "rising_action"
            elif progress < 0.75:
                pacing_phase = "climax"
            else:
                pacing_phase = "resolution"

            # Compute page action intensity (average of panel intensities)
            intensities = []
            for p in page_panels:
                intensity = p.get("_action_intensity",
                                  self._estimate_intensity(p.get("emotion_beat", "neutral")))
                intensities.append(intensity)

            page_plan = {
                "page_number": page_num,
                "pacing_phase": pacing_phase,
                "panels": page_panels,
                "avg_intensity": sum(intensities) / len(intensities),
                "beats": [p.get("emotion_beat", "neutral") for p in page_panels],
            }
            page_plans.append(page_plan)

        # Write to memory blackboard
        memory.mood_journey = story_config.get("mood_journey", "")
        memory.recurring_motif = story_config.get("recurring_motif", "")
        memory.arc_beats = [p.get("emotion_beat", "neutral") for p in panels]
        memory.page_plans = page_plans
        memory.total_panels = num_panels
        memory.total_pages = num_pages
        memory.story_config = story_config

        self._state = {
            "num_pages": num_pages,
            "num_panels": num_panels,
            "pacing_phases": [pp["pacing_phase"] for pp in page_plans],
        }

        self.log.info(
            f"Storyboard planned: {num_panels} panels across {num_pages} pages"
        )

        return {"page_plans": page_plans}

    def update(self, panel_result: Dict[str, Any],
               memory: "StorySectionMemory"):
        """Advance the beat index after each panel generation."""
        if memory.current_beat_index < len(memory.arc_beats) - 1:
            memory.current_beat_index += 1

    def _estimate_intensity(self, emotion_beat: str) -> float:
        """Estimate action intensity from an emotion beat word."""
        high_intensity = {
            "contained_fire", "fracture", "peak_noise", "rage",
            "breakthrough", "climax", "conflict", "action",
            "explosion", "fight", "charge", "attack",
        }
        low_intensity = {
            "stillness", "quiet_hope", "fade", "softness",
            "rest", "pause", "breath", "present",
            "quiet_rest", "drift", "surrender",
        }

        beat_lower = emotion_beat.lower()
        if beat_lower in high_intensity:
            return 0.85
        elif beat_lower in low_intensity:
            return 0.25
        else:
            return 0.5
