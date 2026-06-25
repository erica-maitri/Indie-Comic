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
from core.agents.director_swarm import (
    StoryDirector, ActionDirector, DialogueWriter,
    PoseDirector, EmotionDirector, CameraDirector
)

log = logging.getLogger("pipeline.coordinator")


class AgentCoordinator:
    """
    Multi-Agent Coordinator for Phase 1: Narrative Planning Layer.

    Orchestration flow:
    1. Story Director    → establishes core panel event and characters
    2. Action Director   → defines relational verbs/actions
    3. Dialogue Writer   → structures speech schema based on action tone
    4. Pose Director     → translates actions into explicit body states
    5. Emotion Director  → translates dialogue/action into facial features
    6. Camera Director   → determines cinematic framing and angle

    All agents read from and write to the shared StorySectionMemory blackboard.
    """

    def __init__(self, memory: Optional[StorySectionMemory] = None):
        self.memory = memory or StorySectionMemory()

        # Initialize agents in execution order
        self.agents: List[BaseAgent] = [
            StoryDirector(),
            ActionDirector(),
            DialogueWriter(),
            PoseDirector(),
            EmotionDirector(),
            CameraDirector()
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

        # Add panel-specific story data from the new Scene Graph
        panel_data = self._get_panel_data(panel_id)
        if panel_data:
            context["scene_graph"] = panel_data
            
            emotion = panel_data.get("emotion_beat")
            dialogue = panel_data.get("dialogue")
            
            chars = panel_data.get("characters", [])
            if not emotion and chars:
                emotion = chars[0].get("expression", {}).get("emotion")
            if not dialogue and chars:
                dialogue = chars[0].get("dialogue", {}).get("text")
                
            context["panel_emotion_beat"] = emotion or "neutral"
            context["panel_dialogue"] = dialogue or "..."
            context["panel_id"] = panel_data.get("panel", panel_id)
            
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
        """Get the raw panel data for a given panel_id."""
        if hasattr(self.memory, 'raw_panels'):
            for panel in self.memory.raw_panels:
                if panel.get("panel") == panel_id:
                    return panel
        return None
