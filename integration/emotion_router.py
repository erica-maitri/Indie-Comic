"""
EmotionRouter — Phase 9 Integration Layer
==========================================
Bridges Mood Weaver (emotion classification) to the Indie Comic Pipeline
and Story Weaver (narrative arc selection) using arcs_config.json as the
single source of truth.

Flow:
    user_text
        → EmotionRouter.classify()   → raw Mood Weaver dict
        → EmotionRouter.route()      → arc config from arcs_config.json
        → EmotionRouter.full_pipeline() → unified context dict

Usage::

    from integration.emotion_router import EmotionRouter

    router = EmotionRouter()
    result = router.full_pipeline("I feel so anxious about everything")
    print(result["emotion"])           # "fear"
    print(result["route"]["arc_key"])  # "anxious"
    print(result["route"]["journey"])  # "grounding"
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("integration.emotion_router")

# ---------------------------------------------------------------------------
# Config loading — single source of truth
# ---------------------------------------------------------------------------

def _load_arcs_config() -> dict:
    """Load arcs_config.json from the indie_comic_pipeline/config directory."""
    candidates = [
        Path(__file__).parent.parent / "indie_comic_pipeline" / "config" / "arcs_config.json",
        Path(__file__).parent / "arcs_config.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8-sig") as f:
                    cfg = json.load(f)
                log.info(f"[EmotionRouter] Loaded arcs_config from {p}")
                return cfg
            except Exception as e:
                log.warning(f"[EmotionRouter] Could not load arcs_config.json at {p}: {e}")
    log.warning("[EmotionRouter] arcs_config.json not found — emotion routing will use defaults")
    return {}


_ARCS_CONFIG: dict = _load_arcs_config()

# ---------------------------------------------------------------------------
# Mood Weaver label → arc_key mapping (from config, with hardcoded fallback)
# ---------------------------------------------------------------------------

_LABEL_MAP: Dict[str, str] = _ARCS_CONFIG.get("mood_weaver_label_map", {
    "sadness":    "sadness",
    "joy":        "joy",
    "anger":      "anger",
    "fear":       "fear",
    "love":       "love",
    "surprise":   "surprise",
    "grief":      "grief",
    "determined": "determined",
    "tired":      "tired",
})

# Build arc_key → full arc entry lookup once at module load
def _build_arc_lookup(cfg: dict) -> Dict[str, Dict[str, Any]]:
    """Build a flat {arc_key: entry} dict from arcs_config.json."""
    result: Dict[str, Dict[str, Any]] = {}

    mood_to_arc = cfg.get("mood_to_arc", {})
    for _label, entry in mood_to_arc.items():
        arc_key = entry.get("arc_key")
        if arc_key:
            result[arc_key] = entry

    # tired is a top-level key, not inside mood_to_arc
    if "tired" in cfg:
        result["tired"] = cfg["tired"]

    return result


_ARC_LOOKUP: Dict[str, Dict[str, Any]] = _build_arc_lookup(_ARCS_CONFIG)

# Default arc when nothing matches
_DEFAULT_ARC: Dict[str, Any] = {
    "arc_key":    "reflective",
    "journey":    "reflective",
    "description": "From feeling toward witnessing",
    "arc_beats":  ["acknowledgment", "presence", "shift", "openness"],
    "mood_stages": ["acknowledgment", "presence", "shift", "openness"],
    "motif_hint": "something ordinary that carries unexpected weight",
    "end_note":   "End with openness. No resolution required.",
}

# ---------------------------------------------------------------------------
# EmotionRouter
# ---------------------------------------------------------------------------

class EmotionRouter:
    """
    Routes Mood Weaver emotion labels to Story Weaver arc configurations.

    Attributes
    ----------
    model_path : str
        Path to the trained Mood Weaver model directory.
    _pipe : pipeline or None
        Lazy-loaded HuggingFace text-classification pipeline.
    """

    def __init__(self, model_path: str = "./mood_weaver_model"):
        self.model_path = model_path
        self._pipe = None

    # ── Model loading ──────────────────────────────────────────────────────

    def _get_pipeline(self):
        """Lazy-load the Mood Weaver text-classification pipeline."""
        if self._pipe is not None:
            return self._pipe
        try:
            from transformers import pipeline as hf_pipeline
            path = Path(self.model_path)
            if not path.exists():
                # Try repo-relative paths
                candidates = [
                    Path(__file__).parent.parent / "mood-weaver" / "mood_weaver_model",
                    Path(__file__).parent.parent / "mood-weaver" / self.model_path,
                    Path(self.model_path),
                ]
                for c in candidates:
                    if c.exists():
                        path = c
                        break
                else:
                    log.warning(
                        f"[EmotionRouter] Mood Weaver model not found at '{self.model_path}'. "
                        "classify() will return a placeholder result."
                    )
                    return None
            self._pipe = hf_pipeline(
                "text-classification",
                model=str(path),
                top_k=None,
            )
            log.info(f"[EmotionRouter] Loaded Mood Weaver model from {path}")
        except ImportError:
            log.warning("[EmotionRouter] transformers not installed — classify() unavailable")
        except Exception as e:
            log.warning(f"[EmotionRouter] Failed to load Mood Weaver model: {e}")
        return self._pipe

    # ── Public API ─────────────────────────────────────────────────────────

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Run Mood Weaver classification on ``text``.

        Returns a dict compatible with mood_analyzer.analyze_mood():
            {
                "primary_emotion":    str,   # e.g. "sadness"
                "primary_confidence": float,
                "uncertain":          bool,
                "secondary_emotions": [{"emotion": str, "weight": float}, ...],
                "ambivalence_score":  float,
                "somatic_markers":    list,
                "intensity":          float,
                "raw_probabilities":  {label: score, ...},
            }

        Falls back to a mock result when the model is unavailable.
        """
        pipe = self._get_pipeline()

        if pipe is None:
            # Model not available — return a mock "uncertain" result
            log.info("[EmotionRouter] Using mock classification (model unavailable)")
            return {
                "primary_emotion":    "sadness",
                "primary_confidence": 0.35,
                "uncertain":          True,
                "secondary_emotions": [
                    {"emotion": "fear", "weight": 0.25},
                    {"emotion": "anger", "weight": 0.15},
                ],
                "ambivalence_score": 0.70,
                "somatic_markers":   [],
                "intensity":         0.35,
                "raw_probabilities": {"sadness": 0.35, "fear": 0.25, "anger": 0.15},
                "_mock": True,
            }

        try:
            from mood_weaver_analyzer_compat import analyze_mood as _analyze
            return _analyze(text)
        except ImportError:
            pass  # fall through to inline logic

        # Inline classification (mirrors mood_analyzer.analyze_mood logic)
        results = pipe(text)[0]
        results.sort(key=lambda x: x["score"], reverse=True)

        primary   = results[0]
        secondary = results[1:3]

        ambivalence = round(
            max(0.0, 1.0 - (primary["score"] - results[1]["score"]) * 2), 2
        ) if len(results) > 1 else 0.0

        uncertain = primary["score"] < 0.40

        return {
            "primary_emotion":    primary["label"],
            "primary_confidence": round(primary["score"], 2),
            "uncertain":          uncertain,
            "secondary_emotions": [
                {"emotion": s["label"], "weight": round(s["score"], 2)}
                for s in secondary
            ],
            "ambivalence_score":  ambivalence,
            "somatic_markers":    [],   # somatic detection requires mood_analyzer import
            "intensity":          round(min(1.0, primary["score"]), 2),
            "raw_probabilities":  {r["label"]: round(r["score"], 3) for r in results},
        }

    def route(self, emotion_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map a Mood Weaver output dict to the matching arc config entry.

        ``emotion_output`` must have at minimum a ``"primary_emotion"`` key.
        Returns a full arc config dict from arcs_config.json, with an
        extra ``"arc_key"`` field guaranteed to be present.

        Handles the ``surprise`` fallback (routes to ``happy``/``joy`` arc).
        """
        label = emotion_output.get("primary_emotion", "").lower()

        # Normalise via label map (e.g. "sadness" → "sadness", "joy" → "joy")
        normalised = _LABEL_MAP.get(label, label)

        # Find the arc entry by checking mood_to_arc first
        mood_to_arc = _ARCS_CONFIG.get("mood_to_arc", {})
        arc_entry: Optional[Dict[str, Any]] = None

        # 1. Direct lookup in mood_to_arc by label
        if normalised in mood_to_arc:
            entry = mood_to_arc[normalised]
            arc_key = entry.get("arc_key")
            if arc_key:
                arc_entry = _ARC_LOOKUP.get(arc_key, entry)
            else:
                # surprise: use fallback
                fallback_key = entry.get("fallback", "happy")
                arc_entry = _ARC_LOOKUP.get(fallback_key, _DEFAULT_ARC)
                arc_entry = {**arc_entry, "arc_key": fallback_key, "_routed_via_fallback": True}

        # 2. Direct arc_key lookup (e.g. if Mood Weaver already returns "sad")
        if arc_entry is None and normalised in _ARC_LOOKUP:
            arc_entry = _ARC_LOOKUP[normalised]

        # 3. Default fallback
        if arc_entry is None:
            log.warning(f"[EmotionRouter] No arc found for label '{label}' — using default")
            arc_entry = dict(_DEFAULT_ARC)

        # Guarantee arc_key is present
        if "arc_key" not in arc_entry:
            arc_entry = {**arc_entry, "arc_key": normalised or "reflective"}

        return arc_entry

    def get_character_for_arc(self, arc_key: str) -> Dict[str, Any]:
        """
        Return the character profile recommended for a given arc.

        Uses the ``default_character`` field from the arc entry, then looks up
        the full profile in ``character_profiles``.
        """
        arc_entry = _ARC_LOOKUP.get(arc_key, {})
        char_name = arc_entry.get("default_character", "Wanderer")
        char_world = arc_entry.get("default_world", "The Abstract")
        profiles = _ARCS_CONFIG.get("character_profiles", {})
        profile = profiles.get(char_name, {
            "description": "a figure with a quiet expression and worn travel clothes",
            "traits": ["introspective"],
            "visual_style": "muted tones, soft edges",
        })
        return {
            "name":  char_name,
            "world": char_world,
            **profile,
        }

    def full_pipeline(self, text: str) -> Dict[str, Any]:
        """
        Run the full Mood Weaver → Arc Router pipeline on a single text input.

        Returns
        -------
        dict with keys:
            - ``"emotion"``        : raw label from Mood Weaver (e.g. "sadness")
            - ``"classification"`` : full classify() output
            - ``"route"``          : full arc config dict
            - ``"character"``      : recommended character profile
            - ``"story_context"``  : pre-built context ready for PipelineLauncher
        """
        classification = self.classify(text)
        route          = self.route(classification)
        arc_key        = route.get("arc_key", "reflective")
        character      = self.get_character_for_arc(arc_key)

        story_context = {
            "user_text":              text,
            "primary_emotion":        classification["primary_emotion"],
            "primary_confidence":     classification["primary_confidence"],
            "uncertain":              classification["uncertain"],
            "secondary_emotions":     classification["secondary_emotions"],
            "arc_key":                arc_key,
            "arc_journey":            route.get("journey", ""),
            "arc_description":        route.get("description", ""),
            "arc_beats":              route.get("arc_beats", []),
            "mood_stages":            route.get("mood_stages", []),
            "motif_hint":             route.get("motif_hint", ""),
            "end_note":               route.get("end_note", ""),
            "character_name":         character["name"],
            "character_world":        character["world"],
            "character_description":  character.get("description", ""),
            "character_visual_style": character.get("visual_style", ""),
            "visual_prompt":          route.get("visual_prompt", ""),
        }

        return {
            "emotion":        classification["primary_emotion"],
            "classification": classification,
            "route":          route,
            "character":      character,
            "story_context":  story_context,
        }


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    text = " ".join(sys.argv[1:]) or "I feel completely hollow and exhausted"
    router = EmotionRouter()
    result = router.full_pipeline(text)

    print(f"\nInput     : {text}")
    print(f"Emotion   : {result['emotion']}  (conf={result['classification']['primary_confidence']})")
    if result["classification"].get("uncertain"):
        print("           ⚠️  Uncertain prediction")
    print(f"Arc key   : {result['route']['arc_key']}")
    print(f"Journey   : {result['route'].get('journey', '')}")
    print(f"Character : {result['character']['name']} in {result['character']['world']}")
    print(f"Motif     : {result['route'].get('motif_hint', '')}")
