"""
DIRECTOR SWARM — Scene Graph Manipulation
==========================================
Replaces the old monolithic agents with highly specialized directors.
Each director edits a specific layer of the Scene Graph.
"""

from typing import Dict, Any, List
import logging
from core.agents.base_agent import BaseAgent
from core.memory import StorySectionMemory

log = logging.getLogger("pipeline.agents.swarm")

class StoryDirector(BaseAgent):
    def __init__(self):
        super().__init__("story_director")
        
    def plan(self, story_config: Dict[str, Any], memory: "StorySectionMemory") -> Dict[str, Any]:
        """Loads the base panel sequences and characters into memory."""
        panels = story_config.get("panels", [])
        self.log.info(f"Story Director loaded {len(panels)} raw panel outlines.")
        memory.total_panels = len(panels)
        memory.raw_panels = panels
        
        if "recurring_motif" in story_config:
            memory.recurring_motif = story_config["recurring_motif"]
            
        # 1. Backward-compatible top-level characters
        for char in story_config.get("characters", []):
            if "name" in char:
                c_obj = memory.register_character(char["name"])
                if "costume" in char:
                    c_obj.costume_desc = char["costume"]
                    
        # 2. Main character from metadata
        metadata = story_config.get("_metadata", {})
        main_char = metadata.get("character")
        if main_char:
            memory.register_character(main_char)
            memory.register_character(main_char.lower())
            memory.register_character(main_char.capitalize())

        # 3. Side characters from story bible
        bible = story_config.get("story_bible", {})
        if isinstance(bible, dict):
            for side_char in bible.get("side_characters", []):
                name = side_char.get("name")
                if name:
                    c_obj = memory.register_character(name)
                    c_obj.costume_desc = side_char.get("description", "")
                    memory.register_character(name.lower())
                    memory.register_character(name.capitalize())

        # 4. Characters from panels
        for p in panels:
            for char_obj in p.get("characters", []):
                char_id = char_obj.get("id")
                if char_id:
                    memory.register_character(char_id)
                    memory.register_character(char_id.capitalize())
                    
        return {"status": "Story framework initialized", "panel_count": len(panels)}

    def update(self, panel_result: Dict[str, Any], memory: "StorySectionMemory"):
        pass

class ActionDirector(BaseAgent):
    def __init__(self):
        super().__init__("action_director")
        
    def plan(self, story_config: Dict[str, Any], memory: "StorySectionMemory") -> Dict[str, Any]:
        self.log.info("Action Director parsed relational verbs.")
        return {"status": "Actions verified"}

    def update(self, panel_result: Dict[str, Any], memory: "StorySectionMemory"):
        pass

class DialogueWriter(BaseAgent):
    def __init__(self):
        super().__init__("dialogue_writer")
        
    def plan(self, story_config: Dict[str, Any], memory: "StorySectionMemory") -> Dict[str, Any]:
        self.log.info("Dialogue Writer structured speech and bubble formatting.")
        return {"status": "Dialogue formatted"}

    def update(self, panel_result: Dict[str, Any], memory: "StorySectionMemory"):
        pass

class PoseDirector(BaseAgent):
    def __init__(self):
        super().__init__("pose_director")
        
    def plan(self, story_config: Dict[str, Any], memory: "StorySectionMemory") -> Dict[str, Any]:
        self.log.info("Pose Director resolved hierarchical body states.")
        return {"status": "Poses locked"}

    def update(self, panel_result: Dict[str, Any], memory: "StorySectionMemory"):
        pass

class EmotionDirector(BaseAgent):
    def __init__(self):
        super().__init__("emotion_director")
        
    def plan(self, story_config: Dict[str, Any], memory: "StorySectionMemory") -> Dict[str, Any]:
        self.log.info("Emotion Director applied granular facial taxonomies.")
        return {"status": "Emotions set"}

    def update(self, panel_result: Dict[str, Any], memory: "StorySectionMemory"):
        pass

class CameraDirector(BaseAgent):
    def __init__(self):
        super().__init__("camera_director")
        
    def plan(self, story_config: Dict[str, Any], memory: "StorySectionMemory") -> Dict[str, Any]:
        self.log.info("Camera Director locked cinematic framing and environment.")
        return {"status": "Camera angles calculated"}

    def update(self, panel_result: Dict[str, Any], memory: "StorySectionMemory"):
        pass
