"""
BASE AGENT — Abstract Agent Protocol
======================================
Defines the contract all Phase 1 planning agents must follow.
Each agent reads from the Story Section Memory blackboard, performs
its specialized planning task, and writes results back.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

log = logging.getLogger("pipeline.agents")


class BaseAgent(ABC):
    """
    Abstract base class for all planning agents.

    Each agent operates on the shared StorySectionMemory blackboard:
    1. plan() — reads story config + memory, produces planning output
    2. update() — called after panel generation to update agent's state
    3. get_state() — returns the agent's current internal state
    """

    def __init__(self, name: str):
        self.name = name
        self._state: Dict[str, Any] = {}
        self.log = logging.getLogger(f"pipeline.agents.{name}")

    @abstractmethod
    def plan(self, story_config: Dict[str, Any],
             memory: "StorySectionMemory") -> Dict[str, Any]:
        """
        Execute planning based on story config and current memory state.

        Args:
            story_config: Parsed story configuration from Phase 0
            memory: The shared Story Section Memory blackboard

        Returns:
            Planning output dict (agent-specific structure)
        """
        ...

    @abstractmethod
    def update(self, panel_result: Dict[str, Any],
               memory: "StorySectionMemory"):
        """
        Update internal state after a panel has been generated.

        Args:
            panel_result: Result dict from panel generation
            memory: The shared memory blackboard
        """
        ...

    def get_state(self) -> Dict[str, Any]:
        """Return the agent's current internal state."""
        return self._state.copy()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} '{self.name}'>"
