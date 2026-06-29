"""
STORY SECTION MEMORY — Explicit RAM Blackboard
================================================
Phase 1 Core Component: The central shared memory that all agents read/write.

Replaces the basic NarrativeMemory dict with a structured blackboard that
tracks cross-panel character states, spatial data, identity tokens, and
structural values. Supports serialization for checkpoint/resume.
"""

import json
import time
import copy
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path


@dataclass
class CharacterState:
    """Tracks a single character's state across panels."""
    name: str
    emotion: str = "neutral"
    position: str = "center"           # Spatial position hint
    facing: str = "forward"            # Direction character faces
    costume_desc: str = ""             # Visual descriptor for consistency
    last_action: str = ""              # What they were doing in prior panel
    arc_phase: str = "introduction"    # Where they are in their arc
    panel_appearances: List[int] = field(default_factory=list)

    # Identity embedding tokens (injected by Phase 2 anchoring)
    identity_tokens: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class SceneState:
    """Tracks environment / scene state for spatial continuity."""
    location: str = ""
    time_of_day: str = "day"
    weather: str = "clear"
    lighting: str = "natural"
    mood_color: str = ""                # Dominant mood color
    props: List[str] = field(default_factory=list)
    recurring_motif: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PanelRecord:
    """Immutable record of a generated panel."""
    panel_id: int
    page_num: int
    prompt_used: str = ""
    emotion: str = "neutral"
    dialogue: str = ""
    action_intensity: float = 0.5      # 0=calm, 1=intense action
    consistency_score: float = 0.0
    quality_score: float = 0.0
    image_path: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    # Features extracted from the generated image (for consistency tracking)
    extracted_features: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Don't serialize raw numpy arrays / tensors
        if d.get("extracted_features"):
            d["extracted_features"] = {
                k: v for k, v in d["extracted_features"].items()
                if isinstance(v, (int, float, str, list, bool, type(None)))
            }
        return d


@dataclass
class LayoutDirective:
    """Layout hints for the assembly engine, produced by the Layout Agent."""
    panel_id: int
    size_class: str = "medium"         # "small", "medium", "large", "full_page"
    camera_angle: str = "medium_shot"  # "close_up", "medium_shot", "wide_shot", "bird_eye"
    camera_framing: str = "center"     # "center", "left_third", "right_third"
    aspect_ratio: Tuple[int, int] = (1, 1)
    gutter_emphasis: str = "normal"    # "normal", "tight", "wide"

    def to_dict(self) -> dict:
        return asdict(self)


class StorySectionMemory:
    """
    Explicit RAM Blackboard — the central shared memory for the multi-agent system.

    All agents read from and write to this blackboard. The Panel Engine pulls
    context from here to construct generation prompts, and writes back features
    extracted from generated images.

    Supports:
    - Character state tracking with identity tokens
    - Scene/environment continuity
    - Panel history with configurable retention window
    - Layout directives per panel
    - Story arc tracking (mood journey, beats)
    - Serialization to/from JSON for checkpointing
    """

    def __init__(self, retention_window: int = 20):
        # ── Core State ──
        self.characters: Dict[str, CharacterState] = {}
        self.scene: SceneState = SceneState()
        self.panel_history: List[PanelRecord] = []
        self.layout_directives: Dict[int, LayoutDirective] = {}

        # ── Story Arc ──
        self.story_config: Dict[str, Any] = {}
        self.mood_journey: str = ""
        self.recurring_motif: str = ""
        self.arc_beats: List[str] = []
        self.current_beat_index: int = 0
        self.main_character: Optional[str] = None

        # ── Planning Data ──
        self.page_plans: List[Dict[str, Any]] = []   # Storyboard Agent output
        self.raw_panels: List[Dict[str, Any]] = []   # Un-enriched/base panels from Intake
        self.total_panels: int = 0
        self.total_pages: int = 0

        # ── Identity Anchor ──
        self.anchor_panel_id: Optional[int] = None
        self.anchor_features: Optional[Dict[str, Any]] = None

        # ── Configuration ──
        self.retention_window = retention_window

        # ── Metadata ──
        self.created_at: float = time.time()
        self.last_updated: float = time.time()

    # ─────────────────────────────────────────────────────────────────────
    # Character Management
    # ─────────────────────────────────────────────────────────────────────

    def register_character(self, name: str, **kwargs) -> CharacterState:
        """Register or update a character in the blackboard."""
        if name not in self.characters:
            self.characters[name] = CharacterState(name=name, **kwargs)
        else:
            for k, v in kwargs.items():
                if hasattr(self.characters[name], k):
                    setattr(self.characters[name], k, v)
        self._touch()
        return self.characters[name]

    def get_character(self, name: str) -> Optional[CharacterState]:
        """Get a character's current state."""
        return self.characters.get(name)

    def update_character(self, name: str, **kwargs):
        """Update specific fields of a character's state."""
        if name in self.characters:
            for k, v in kwargs.items():
                if hasattr(self.characters[name], k):
                    setattr(self.characters[name], k, v)
            self._touch()

    def inject_identity_tokens(self, name: str, tokens: Dict[str, Any]):
        """Phase 2: Inject identity embedding tokens for a character."""
        if name in self.characters:
            self.characters[name].identity_tokens = tokens
            self._touch()

    def get_identity_tokens(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve identity tokens for consistency enforcement."""
        char = self.characters.get(name)
        if char:
            return char.identity_tokens
        return None

    # ─────────────────────────────────────────────────────────────────────
    # Scene Management
    # ─────────────────────────────────────────────────────────────────────

    def update_scene(self, **kwargs):
        """Update the current scene state."""
        for k, v in kwargs.items():
            if hasattr(self.scene, k):
                setattr(self.scene, k, v)
        self._touch()

    def get_scene(self) -> SceneState:
        """Get the current scene state."""
        return self.scene

    # ─────────────────────────────────────────────────────────────────────
    # Panel History
    # ─────────────────────────────────────────────────────────────────────

    def add_panel(self, record: PanelRecord):
        """Add a generated panel to history, replacing any existing record for the same panel_id."""
        for i, p in enumerate(self.panel_history):
            if p.panel_id == record.panel_id:
                self.panel_history[i] = record
                self._touch()
                return

        self.panel_history.append(record)
        # Trim to retention window
        if len(self.panel_history) > self.retention_window:
            self.panel_history = self.panel_history[-self.retention_window:]
        self._touch()

    def get_recent_panels(self, n: int = 3) -> List[PanelRecord]:
        """Get the N most recent panels."""
        return self.panel_history[-n:]

    def get_panel(self, panel_id: int) -> Optional[PanelRecord]:
        """Get a specific panel by ID."""
        for p in self.panel_history:
            if p.panel_id == panel_id:
                return p
        return None

    def get_panel_count(self) -> int:
        """Get total number of panels generated so far."""
        return len(self.panel_history)

    # ─────────────────────────────────────────────────────────────────────
    # Layout Directives
    # ─────────────────────────────────────────────────────────────────────

    def set_layout_directive(self, panel_id: int, directive: LayoutDirective):
        """Set layout directive for a specific panel."""
        self.layout_directives[panel_id] = directive
        self._touch()

    def get_layout_directive(self, panel_id: int) -> Optional[LayoutDirective]:
        """Get layout directive for a specific panel."""
        return self.layout_directives.get(panel_id)

    # ─────────────────────────────────────────────────────────────────────
    # Anchor Management (Phase 2)
    # ─────────────────────────────────────────────────────────────────────

    def set_anchor(self, panel_id: int, features: Dict[str, Any]):
        """Set the primary visual anchor panel and its features."""
        self.anchor_panel_id = panel_id
        self.anchor_features = features
        self._touch()

    def get_anchor_features(self) -> Optional[Dict[str, Any]]:
        """Retrieve anchor features for consistency enforcement."""
        return self.anchor_features

    # ─────────────────────────────────────────────────────────────────────
    # Context Generation (for prompt building)
    # ─────────────────────────────────────────────────────────────────────

    def get_page_num(self, panel_id: int) -> int:
        """Dynamically compute page number for a panel based on layout directives."""
        current_page = 1
        current_page_slots = 0.0
        for pid in range(1, panel_id + 1):
            layout = self.get_layout_directive(pid)
            size_class = layout.size_class if layout else "medium"
            
            if size_class == "full_page":
                weight = 4.0
            elif size_class == "large":
                weight = 2.0
            else:
                weight = 1.0
                
            if current_page_slots > 0 and current_page_slots + weight > 4.0:
                current_page += 1
                current_page_slots = 0.0
                
            current_page_slots += weight
            if size_class == "full_page":
                current_page_slots = 4.0
        return current_page

    def build_generation_context(self, panel_id: int) -> Dict[str, Any]:
        """
        Build a comprehensive context dict for Panel Engine prompt construction.
        Pulls from all blackboard sections to give the generation engine
        full awareness of story state.
        """
        recent = self.get_recent_panels(3)
        recent_dicts = [
            {"panel_id": p.panel_id, "emotion": p.emotion,
             "dialogue": p.dialogue, "action_intensity": p.action_intensity}
            for p in recent
        ]

        char_states = {
            name: {"emotion": cs.emotion, "position": cs.position,
                   "facing": cs.facing, "last_action": cs.last_action,
                   "costume_desc": cs.costume_desc,
                   "has_identity_tokens": cs.identity_tokens is not None}
            for name, cs in self.characters.items()
        }

        layout = self.get_layout_directive(panel_id)

        context = {
            "panel_id": panel_id,
            "total_panels": self.total_panels,
            "total_pages": self.total_pages,
            "current_page": self.get_page_num(panel_id) if self.total_panels > 0 else 1,
            "mood_journey": self.mood_journey,
            "recurring_motif": self.recurring_motif,
            "current_beat": self.arc_beats[self.current_beat_index]
                if self.current_beat_index < len(self.arc_beats) else "resolution",
            "scene": self.scene.to_dict(),
            "characters": char_states,
            "recent_panels": recent_dicts,
            "layout": layout.to_dict() if layout else None,
            "has_anchor": self.anchor_panel_id is not None,
        }
        return context

    # ─────────────────────────────────────────────────────────────────────
    # Serialization (Checkpoint / Resume)
    # ─────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return the full blackboard state as a serializable dict (mirrors save_checkpoint)."""
        return {
            "characters": {n: cs.to_dict() for n, cs in self.characters.items()},
            "scene": self.scene.to_dict(),
            "panel_history": [p.to_dict() for p in self.panel_history],
            "layout_directives": {
                str(k): ld.to_dict() for k, ld in self.layout_directives.items()
            },
            "story_config": self.story_config,
            "mood_journey": self.mood_journey,
            "recurring_motif": self.recurring_motif,
            "arc_beats": self.arc_beats,
            "current_beat_index": self.current_beat_index,
            "main_character": self.main_character,
            "page_plans": self.page_plans,
            "raw_panels": self.raw_panels,
            "total_panels": self.total_panels,
            "total_pages": self.total_pages,
            "anchor_panel_id": self.anchor_panel_id,
            "anchor_features": self.anchor_features,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }

    def save_checkpoint(self, path: str):
        """Serialize the entire blackboard to JSON for checkpoint/resume."""
        data = {
            "characters": {n: cs.to_dict() for n, cs in self.characters.items()},
            "scene": self.scene.to_dict(),
            "panel_history": [p.to_dict() for p in self.panel_history],
            "layout_directives": {
                str(k): ld.to_dict() for k, ld in self.layout_directives.items()
            },
            "story_config": self.story_config,
            "mood_journey": self.mood_journey,
            "recurring_motif": self.recurring_motif,
            "arc_beats": self.arc_beats,
            "current_beat_index": self.current_beat_index,
            "main_character": self.main_character,
            "page_plans": self.page_plans,
            "raw_panels": self.raw_panels,
            "total_panels": self.total_panels,
            "total_pages": self.total_pages,
            "anchor_panel_id": self.anchor_panel_id,
            "anchor_features": self.anchor_features,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load_checkpoint(cls, path: str) -> "StorySectionMemory":
        """Restore blackboard from a JSON checkpoint."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        mem = cls()
        mem.mood_journey = data.get("mood_journey", "")
        mem.recurring_motif = data.get("recurring_motif", "")
        mem.arc_beats = data.get("arc_beats", [])
        mem.current_beat_index = data.get("current_beat_index", 0)
        mem.main_character = data.get("main_character")
        mem.story_config = data.get("story_config", {})
        mem.page_plans = data.get("page_plans", [])
        mem.raw_panels = data.get("raw_panels", [])
        mem.total_panels = data.get("total_panels", 0)
        mem.total_pages = data.get("total_pages", 0)
        mem.anchor_panel_id = data.get("anchor_panel_id")
        mem.anchor_features = data.get("anchor_features")
        mem.created_at = data.get("created_at", time.time())
        mem.last_updated = data.get("last_updated", time.time())

        # Restore characters
        for name, cs_data in data.get("characters", {}).items():
            cs = CharacterState(name=name)
            for k, v in cs_data.items():
                if hasattr(cs, k) and k != "name":
                    setattr(cs, k, v)
            mem.characters[name] = cs

        # Restore scene
        for k, v in data.get("scene", {}).items():
            if hasattr(mem.scene, k):
                setattr(mem.scene, k, v)

        # Restore panel history
        for p_data in data.get("panel_history", []):
            pr = PanelRecord(
                panel_id=p_data["panel_id"],
                page_num=p_data["page_num"],
            )
            for k, v in p_data.items():
                if hasattr(pr, k) and k not in ("panel_id", "page_num"):
                    setattr(pr, k, v)
            mem.panel_history.append(pr)

        # Restore layout directives
        for p_id_str, ld_data in data.get("layout_directives", {}).items():
            try:
                p_id = int(p_id_str)
            except ValueError:
                p_id = p_id_str
            ld = LayoutDirective(panel_id=p_id)
            for k, v in ld_data.items():
                if hasattr(ld, k) and k != "panel_id":
                    if k == "aspect_ratio" and isinstance(v, list):
                        v = tuple(v)
                    setattr(ld, k, v)
            mem.layout_directives[p_id] = ld

        return mem

    def reset(self):
        """Reset the blackboard for a new story."""
        self.__init__(retention_window=self.retention_window)

    # ─────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────

    def _touch(self):
        """Update the last-modified timestamp."""
        self.last_updated = time.time()

    def __repr__(self) -> str:
        return (
            f"<StorySectionMemory chars={len(self.characters)} "
            f"panels={len(self.panel_history)} "
            f"pages={self.total_pages} "
            f"anchored={self.anchor_panel_id is not None}>"
        )
