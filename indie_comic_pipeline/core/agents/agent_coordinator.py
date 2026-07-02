"""
AGENT COORDINATOR — Phase 1 Multi-Agent Orchestrator
=====================================================
Orchestrates the 4 planning agents (Storyboard, Character, Scene, Layout)
using the blackboard pattern. Runs them sequentially, merges their outputs
into the Story Section Memory, and provides a unified planning interface.
"""

import os
import json
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

    def __init__(self, memory: Optional[StorySectionMemory] = None, agent_config_path: Optional[str] = None):
        self.memory = memory or StorySectionMemory()
        self.agents: List[BaseAgent] = []
        
        if agent_config_path is None:
            agent_config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                "config", "agents.json"
            )
            
        if os.path.exists(agent_config_path):
            self.load_agents_from_config(agent_config_path)
        else:
            raise FileNotFoundError(f"Agent configuration file not found at {agent_config_path}")
            
        self._planning_results: Dict[str, Any] = {}
        self._planning_time: float = 0.0

    def register_agent(self, agent: BaseAgent):
        """Register a custom director agent into the swarm planning pipeline."""
        self.agents.append(agent)
        log.info(f"Dynamically registered custom planning agent: {agent.name}")

    def load_agents_from_config(self, config_path: str):
        """
        Dynamically instantiate and load agents from a JSON registry file.
        Matches class names to objects in the director swarm.
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            
            agent_names = cfg.get("swarm_agents", [])
            if not agent_names:
                raise ValueError(f"No 'swarm_agents' found in agent config: {config_path}")
                
            # Dynamic lookup table mapping name to class in core.agents.director_swarm
            import core.agents.director_swarm as swarm_module
            for name in agent_names:
                cls = getattr(swarm_module, name, None)
                if cls is not None and issubclass(cls, BaseAgent):
                    self.register_agent(cls())  # type: ignore
                else:
                    log.warning(f"Could not find agent class {name} in director_swarm.")
            if not self.agents:
                raise ValueError("No valid agents were loaded from config.")
        except Exception as e:
            log.error(f"Failed to load dynamic agent registry: {e}")
            raise

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

        # Run StoryDirector and ActionDirector sequentially as they form the core dependency chain
        story_agent = next((a for a in self.agents if a.__class__.__name__ == "StoryDirector"), None)
        action_agent = next((a for a in self.agents if a.__class__.__name__ == "ActionDirector"), None)
        
        def _safe_run_agent(agent) -> Dict[str, Any]:
            agent_start = time.time()
            log.info(f"\n  Running {agent.name} agent...")
            try:
                result = agent.plan(story_config, self.memory)
                agent_elapsed = time.time() - agent_start
                log.info(f"  ✓ {agent.name} agent complete ({agent_elapsed:.2f}s)")
                return result
            except Exception as e:
                log.warning(f"  ✗ {agent.name} agent failed: {e}. Applying heuristic fallback.")
                
                # CameraDirector fallback: guarantee layout directives exist
                if agent.__class__.__name__ == "CameraDirector" and hasattr(self.memory, "raw_panels") and self.memory.raw_panels:
                    from core.memory import LayoutDirective
                    for idx, panel in enumerate(self.memory.raw_panels):
                        panel_id = panel.get("panel", idx + 1)
                        if panel_id not in self.memory.layout_directives:
                            self.memory.set_layout_directive(panel_id, LayoutDirective(
                                panel_id=panel_id,
                                size_class="medium",
                                camera_angle="medium_shot"
                            ))
                return {"error": str(e)}

        if story_agent:
            self._planning_results[story_agent.name] = _safe_run_agent(story_agent)
        if action_agent:
            self._planning_results[action_agent.name] = _safe_run_agent(action_agent)

        # Run remaining independent planning agents concurrently
        independent_agents = [
            a for a in self.agents 
            if a != story_agent and a != action_agent
        ]
        
        if independent_agents:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=len(independent_agents)) as executor:
                future_to_agent = {
                    executor.submit(_safe_run_agent, agent): agent
                    for agent in independent_agents
                }
                for future in as_completed(future_to_agent):
                    agent = future_to_agent[future]
                    try:
                        self._planning_results[agent.name] = future.result()
                    except Exception as e:
                        log.error(f"Agent {agent.name} crashed completely: {e}")
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
