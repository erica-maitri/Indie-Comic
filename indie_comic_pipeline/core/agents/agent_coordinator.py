"""
AGENT COORDINATOR — Phase 1 Multi-Agent Orchestrator
=====================================================
Orchestrates the 4 planning agents (Storyboard, Character, Scene, Layout)
using the blackboard pattern. Runs them sequentially, merges their outputs
into the Story Section Memory, and provides a unified planning interface.
"""

import time
import logging
import copy
from typing import Dict, Any, List, Optional

from core.memory import StorySectionMemory
from core.agents.base_agent import BaseAgent
from core.agents.storyboard_agent import StoryboardAgent
from core.agents.character_agent import CharacterAgent
from core.agents.scene_agent import SceneAgent
from core.agents.layout_agent import LayoutAgent

log = logging.getLogger("pipeline.coordinator")


class AgentCoordinator:
    """
    Multi-Agent Coordinator for Phase 1: Narrative Planning Layer.

    Orchestration flow:
    1. Storyboard Agent → establishes page structure, pacing, beats
    2. Character Agent  → builds character profiles, maps expression arcs
    3. Scene Agent      → maps atmospheres, tracks spatial continuity
    4. Layout Agent     → determines camera framing, panel geometry

    All agents read from and write to the shared StorySectionMemory blackboard.
    """

    def __init__(self, memory: Optional[StorySectionMemory] = None):
        self.memory = memory or StorySectionMemory()

        # Initialize agents in execution order
        self.agents: List[BaseAgent] = [
            StoryboardAgent(),
            CharacterAgent(),
            SceneAgent(),
            LayoutAgent(),
        ]

        self._planning_results: Dict[str, Any] = {}
        self._planning_time: float = 0.0

    def run_planning(self, story_config: Dict[str, Any]) -> StorySectionMemory:
        """
        Execute the full multi-agent planning pipeline.

        Args:
            story_config: Structured story configuration from Phase 0 (Story Intake)

        Returns:
            The populated StorySectionMemory blackboard, ready for Phase 2
        """
        log.info("=" * 60)
        log.info("PHASE 1: NARRATIVE PLANNING LAYER — Multi-Agent System")
        log.info("=" * 60)

        story_config = copy.deepcopy(story_config)
        start_time = time.time()

        for agent in self.agents:
            agent_start = time.time()
            log.info(f"\n  Running {agent.name} agent...")

            try:
                result = agent.plan(story_config, self.memory)
                self._planning_results[agent.name] = result

                agent_elapsed = time.time() - agent_start
                log.info(f"  ✓ {agent.name} agent complete ({agent_elapsed:.2f}s)")

            except Exception as e:
                log.error(f"  ✗ {agent.name} agent failed: {e}")
                self._planning_results[agent.name] = {"error": str(e)}

        self._planning_time = time.time() - start_time

        log.info(f"\n  Planning complete in {self._planning_time:.2f}s")
        log.info(f"  Memory state: {self.memory}")
        log.info("=" * 60)

        return self.memory

    def notify_panel_generated(self, panel_result: Dict[str, Any]):
        """
        Notify all agents that a panel has been generated.
        Each agent updates its internal state based on the result.

        Args:
            panel_result: Dict with panel generation results
                         (panel_id, image, prompt, quality_score, etc.)
        """
        for agent in self.agents:
            try:
                agent.update(panel_result, self.memory)
            except Exception as e:
                log.warning(f"Agent {agent.name} update failed: {e}")

    def get_generation_context(self, panel_id: int) -> Dict[str, Any]:
        """
        Build a comprehensive generation context for a specific panel.
        Combines memory blackboard context with agent-specific directives.

        Args:
            panel_id: The panel number to build context for

        Returns:
            Context dict ready for the Panel Engine
        """
        # Base context from memory
        context = self.memory.build_generation_context(panel_id)

        # Add character-specific directives
        char_agent = self._get_agent("character")
        if char_agent and isinstance(char_agent, CharacterAgent):
            context["character_expression"] = char_agent.get_expression_for_panel(panel_id)
            context["character_visual_note"] = char_agent.get_visual_note_for_panel(panel_id)

        # Add scene-specific atmosphere
        scene_agent = self._get_agent("scene")
        if scene_agent and isinstance(scene_agent, SceneAgent):
            context["scene_atmosphere"] = scene_agent.get_atmosphere_for_panel(panel_id)

        # Add panel-specific story data from page plans
        panel_data = self._get_panel_data(panel_id)
        if panel_data:
            context["panel_visual"] = panel_data.get("visual", "")
            context["panel_dialogue"] = panel_data.get("dialogue", "...")
            context["panel_motion"] = panel_data.get("motion", "")
            context["panel_emotion_beat"] = panel_data.get("emotion_beat", "neutral")

        return context

    def get_planning_summary(self) -> Dict[str, Any]:
        """Get a summary of the planning results for logging/display."""
        return {
            "planning_time_s": round(self._planning_time, 2),
            "total_panels": self.memory.total_panels,
            "total_pages": self.memory.total_pages,
            "characters": list(self.memory.characters.keys()),
            "mood_journey": self.memory.mood_journey,
            "recurring_motif": self.memory.recurring_motif,
            "agent_results": {
                name: "success" if "error" not in result else result["error"]
                for name, result in self._planning_results.items()
            },
        }

    def _get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get a specific agent by name."""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def _get_panel_data(self, panel_id: int) -> Optional[Dict[str, Any]]:
        """Get the raw panel data from page plans for a given panel_id."""
        for page_plan in self.memory.page_plans:
            for panel in page_plan.get("panels", []):
                if panel.get("panel") == panel_id:
                    return panel
        return None
