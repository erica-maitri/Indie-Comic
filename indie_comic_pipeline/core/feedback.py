"""
HEURISTIC FEEDBACK TELEMETRY LOOP — Phase 8
==========================================
Collects user ratings and feedback per panel/page, storing them in a 
local JSON file. Provides analytical aggregations to help optimize 
pipeline settings over time. 

NOTE: This is a heuristic parameter adjustment loop from logged user ratings,
not formal Reinforcement Learning from Human Feedback (RLHF), as there is no 
trained reward model or policy-gradient update. The class name RLHFFeedbackLoop
is kept for compatibility.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

log = logging.getLogger("pipeline.feedback")


class RLHFFeedbackLoop:
    """
    Phase 8: Heuristic Feedback Telemetry Loop.
    
    Manages collection and local JSON serialization of user star ratings (1-5),
    qualitative feedback, and engagement metrics for pages and panels. Functionally
    serves as a telemetry recorder for heuristic hyperparameter tuning, rather than
    formal RLHF with trained reward models.
    """

    def __init__(self, feedback_path: str = "outputs/rlhf_feedback.json"):
        self.feedback_path = feedback_path
        self.data = {"panels": [], "pages": [], "global_metrics": {}}
        
        # Ensure parent directory exists
        Path(self.feedback_path).parent.mkdir(parents=True, exist_ok=True)
        self._load_feedback()

    def _load_feedback(self):
        """Load existing feedback data from local JSON file."""
        if os.path.exists(self.feedback_path):
            try:
                with open(self.feedback_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                log.info(f"Loaded {len(self.data.get('panels', []))} panel feedback records from {self.feedback_path}")
            except Exception as e:
                log.warning(f"Error loading feedback file {self.feedback_path}: {e}. Initializing clean state.")

    def _save_feedback(self):
        """Save feedback data to local JSON file."""
        try:
            with open(self.feedback_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
            log.info(f"RLHF feedback successfully saved to {self.feedback_path}")
        except Exception as e:
            log.error(f"Failed to save RLHF feedback to {self.feedback_path}: {e}")

    def add_panel_feedback(self, panel_id: int, 
                           rating: int, 
                           comment: str = "", 
                           engagement_time: float = 0.0,
                           prompt_used: str = "",
                           generation_backend: str = ""):
        """
        Record feedback for a specific panel.
        
        Args:
            panel_id: ID of the panel (1-indexed)
            rating: User score (1 to 5 stars)
            comment: Optional qualitative feedback text
            engagement_time: Optional time user spent viewing the panel (in seconds)
            prompt_used: Prompt that generated the panel (for template tuning)
            generation_backend: Backend name used (e.g. sdxl)
        """
        rating = max(1, min(5, rating))
        record = {
            "panel_id": panel_id,
            "rating": rating,
            "comment": comment,
            "engagement_time": engagement_time,
            "prompt_used": prompt_used,
            "backend": generation_backend
        }
        self.data["panels"].append(record)
        self._save_feedback()

    def add_page_feedback(self, page_num: int, 
                           rating: int, 
                           comment: str = ""):
        """
        Record feedback for a compiled page.
        
        Args:
            page_num: Page number (1-indexed)
            rating: User score (1 to 5 stars)
            comment: Optional comments
        """
        rating = max(1, min(5, rating))
        record = {
            "page_num": page_num,
            "rating": rating,
            "comment": comment
        }
        self.data["pages"].append(record)
        self._save_feedback()

    def get_average_rating(self) -> float:
        """Get the average rating across all logged panels."""
        panels = self.data.get("panels", [])
        if not panels:
            return 0.0
        return sum(r["rating"] for r in panels) / len(panels)

    def get_feedback_summary(self) -> Dict[str, Any]:
        """Compile statistical summary of collected feedback."""
        panels = self.data.get("panels", [])
        pages = self.data.get("pages", [])
        
        avg_panel_rating = sum(r["rating"] for r in panels) / len(panels) if panels else 0.0
        avg_page_rating = sum(r["rating"] for r in pages) / len(pages) if pages else 0.0
        
        # Analyze which backends perform best
        backend_ratings = {}
        for r in panels:
            b = r.get("backend", "unknown")
            backend_ratings.setdefault(b, []).append(r["rating"])
            
        backend_stats = {
            b: round(sum(l) / len(l), 2) for b, l in backend_ratings.items()
        }
        
        return {
            "total_panels_rated": len(panels),
            "total_pages_rated": len(pages),
            "average_panel_rating": round(avg_panel_rating, 2),
            "average_page_rating": round(avg_page_rating, 2),
            "backend_performances": backend_stats
        }
