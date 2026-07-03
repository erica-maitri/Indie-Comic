"""
DIRECTOR SWARM — Scene Graph Manipulation
==========================================
Replaces the old monolithic agents with highly specialized directors.
Each director edits a specific layer of the Scene Graph stored in memory.
"""

from typing import Dict, Any, List, Optional
import logging
from core.agents.base_agent import BaseAgent
from core.memory import StorySectionMemory, LayoutDirective

log = logging.getLogger("pipeline.agents.swarm")


# ─────────────────────────────────────────────────────────────────────────────
# 5-LAYER CINEMATIC EXAGGERATION MAP  (The Cinematic Thesaurus)
# ─────────────────────────────────────────────────────────────────────────────
# Every entry produces a distinct, uniquely-identifiable visual in SDXL.
# Diffusion models regress to the mean on generic verbs — these coordinates
# push the generation to the extreme edge of cinematic language.
#
# Structure per entry:
#   verb        — the aggressive, active-voice action verb phrase
#   mechanics   — exact body-part positions under maximum tension
#   impact      — the precise moment of contact or consequence
#   reaction    — environmental / secondary object response to the action
#   timing      — freeze-frame cue (anticipation / impact / follow-through)
# ─────────────────────────────────────────────────────────────────────────────
ACTION_EXAGGERATION_MAP: Dict[str, Dict[str, str]] = {
    # ── Combat ──
    "punch": {
        "verb":      "delivers a devastating haymaker with full body rotation",
        "mechanics": "entire torso twisted, arm cocked far back, knuckles white, veins raised on forearm",
        "impact":    "fist craters into the target's face, skin distorting under impact wave",
        "reaction":  "spit and sweat explode sideways, head snapping backward violently, hair whipping",
        "timing":    "maximum-force impact freeze-frame, kinetic energy at absolute peak",
    },
    "kick": {
        "verb":      "drives a brutal flying side-kick with explosive force",
        "mechanics": "body fully horizontal, kicking leg fully extended, support leg tucked",
        "impact":    "boot heel connects with center of mass, concussive shockwave radiating outward",
        "reaction":  "target's torso buckles, shattered debris bursting outward, dust plume rising",
        "timing":    "mid-air freeze-frame at maximum extension, peak velocity moment",
    },
    "dodge": {
        "verb":      "contorts into a desperate last-millisecond limbo dodge",
        "mechanics": "spine arched backward impossibly far, knees buckling, hair brushing the ground",
        "impact":    "weapon grazes chest fabric, tearing threads, millimetres from skin",
        "reaction":  "shockwave ripples past skin, neon sparks trailing from near-miss, dirt exploding upward",
        "timing":    "anticipation freeze-frame, maximum tension before the action resolves",
    },
    "block": {
        "verb":      "throws up a bone-rattling forearm block under brutal force",
        "mechanics": "forearm raised vertical, elbow locked, knees bent absorbing impact, feet skidding",
        "impact":    "strike slams forearm with bone-shaking force, shockwave up the arm",
        "reaction":  "clothing rippling from force wave, gravel skidding under feet, dust kicked up",
        "timing":    "impact hold-frame, force peak visible in the arm bend",
    },
    "slash": {
        "verb":      "unleashes a wide diagonal slash with reckless power",
        "mechanics": "sword arm extended fully, body rotating at the hip, opposite arm counterbalancing",
        "impact":    "blade arc cutting clean through the air, energy trail visible",
        "reaction":  "pressure wave parting hair and cloth, sparks if striking metal, air displacement visible",
        "timing":    "mid-swing freeze-frame, blade at maximum velocity",
    },
    "tackle": {
        "verb":      "launches into a full-body rugby tackle, shoulder lowered",
        "mechanics": "shoulder dropped, legs driving, arms wrapping around torso",
        "impact":    "shoulder drives into midsection, both bodies leaving the ground",
        "reaction":  "ground cracks on landing, debris cloud, tumbling momentum",
        "timing":    "collision impact freeze-frame, bodies airborne",
    },
    "throw": {
        "verb":      "executes a judo hip-throw with explosive commitment",
        "mechanics": "hips pivoting, attacker's centre of gravity dropping below the target's",
        "impact":    "target arcing overhead in an inverted parabola, feet leaving ground",
        "reaction":  "ground impact crater, dust explosion, shockwave through the floor",
        "timing":    "apex of the throw, target fully inverted in mid-air",
    },
    # ── Movement ──
    "run": {
        "verb":      "hurtles forward in a desperate full-sprint, arms pumping",
        "mechanics": "extreme forward lean, feet barely touching ground, hair streaming backward",
        "impact":    "each footstrike leaving a small crater in the surface",
        "reaction":  "dust wake trailing, air pressure wave bending nearby objects",
        "timing":    "peak-stride freeze-frame, one foot off the ground entirely",
    },
    "land": {
        "verb":      "crashes down from height into a predator's crouching impact stance",
        "mechanics": "knees deeply bent absorbing impact, one fist touching ground, head raised",
        "impact":    "boots hit concrete, spiderweb cracks radiating from impact point",
        "reaction":  "dust and debris erupting in a circle around the landing point, loose gravel bouncing",
        "timing":    "landing impact hold-frame, crater visible, dust cloud mid-expansion",
    },
    "leap": {
        "verb":      "launches into a soaring aerial leap from a running start",
        "mechanics": "legs fully extended, arms stretched wide, body forming a clean arc",
        "impact":    "peak of arc, silhouetted against sky or background",
        "reaction":  "air displaced beneath, cape or clothing billowing upward from velocity",
        "timing":    "apex freeze-frame, maximum height, body fully extended",
    },
    "charge": {
        "verb":      "erupts into a reckless full-body charge with total commitment",
        "mechanics": "head lowered, arms back, legs driving with explosive force",
        "impact":    "nothing in the path has time to react",
        "reaction":  "ground churning underfoot, obstacles scattering aside, dust wake",
        "timing":    "mid-charge freeze, maximum velocity, unstoppable force visible",
    },
    "fall": {
        "verb":      "crumples and collapses in a dead-weight free-fall",
        "mechanics": "knees buckling first, then torso folding, limbs loose and uncontrolled",
        "impact":    "body hitting ground in a heap, unable to break the fall",
        "reaction":  "dust rising, loose items scattering from impact, eerie stillness after",
        "timing":    "moment of total surrender to gravity, body mid-collapse",
    },
    "crawl": {
        "verb":      "drags forward on hands and knees through sheer will alone",
        "mechanics": "arms trembling under weight, chin barely above ground, one knee forward",
        "impact":    "each handprint pressing into the ground surface",
        "reaction":  "trail of effort visible, surrounding environment emphasising the struggle",
        "timing":    "mid-crawl hold, the weight of every centimetre visible",
    },
    # ── Dramatic ──
    "stands": {
        "verb":      "stands like a monument, rooted and immovable under pressure",
        "mechanics": "feet planted wide, fists clenched, every muscle in visible tension",
        "impact":    "occupies the centre of the frame with gravitational authority",
        "reaction":  "wind moves around him rather than past him, environment yielding",
        "timing":    "held still-frame, maximum presence, atmosphere charged with intent",
    },
    "raises": {
        "verb":      "raises arms skyward in a full-body victory declaration",
        "mechanics": "both arms fully extended overhead, chest open, chin raised to sky",
        "impact":    "silhouetted against open sky or explosion, the embodiment of triumph",
        "reaction":  "light catches the figure, surrounding space opens wide",
        "timing":    "peak triumphant hold, the moment of absolute victory",
    },
    "holds": {
        "verb":      "holds absolute ground under immense external pressure",
        "mechanics": "heels dug in, arms braced, body angled against opposing force",
        "impact":    "the line held, ground not given",
        "reaction":  "wind and force bending everything around the fixed point",
        "timing":    "maximum-resistance hold-frame, the tension of not breaking",
    },
    "sits": {
        "verb":      "collapses into a seated position with the full weight of exhaustion",
        "mechanics": "knees drawn up, arms resting on knees, head dropped forward",
        "impact":    "the weight of everything visible in the curve of the spine",
        "reaction":  "stillness around the figure, empty space amplifying the loneliness",
        "timing":    "held contemplative frame, the breath between events",
    },
    "reaches": {
        "verb":      "stretches every centimetre of reach toward something just out of grasp",
        "mechanics": "full arm extension, body leaning forward at maximum tilt, fingers spread wide",
        "impact":    "fingertips millimetres from the target, almost there",
        "reaction":  "the gap between fingers and target visually charged",
        "timing":    "anticipation freeze-frame, everything hanging on this moment",
    },
    "watches": {
        "verb":      "stares with burning intensity at something that changes everything",
        "mechanics": "body completely still, eyes locked, jaw tight, not breathing",
        "impact":    "the thing being watched visible or implied in frame",
        "reaction":  "the stillness itself is charged with unspoken reaction",
        "timing":    "held observation frame, the weight of what is being witnessed",
    },
    "clutches": {
        "verb":      "clutches head in both hands as sensation becomes overwhelming",
        "mechanics": "fingers digging into scalp, elbows pulled inward, body curling around itself",
        "impact":    "the internal experience externalised in every tense muscle",
        "reaction":  "world blurring or warping around the figure, suggesting inner chaos",
        "timing":    "peak internal crisis hold-frame, maximum overwhelm visible",
    },
    "floats": {
        "verb":      "drifts with total surrender to weightlessness",
        "mechanics": "limbs extended and relaxed, body horizontal, face upward",
        "impact":    "separation from gravity complete, floating in the moment",
        "reaction":  "environment becomes soft and secondary, depth-of-field blurring",
        "timing":    "suspended drift frame, time itself slowing to a stop",
    },
    # ── Generic fallback ──
    "observes": {
        "verb":      "witnesses the moment with total, unguarded presence",
        "mechanics": "body open and still, hands at sides, face forward",
        "impact":    "the thing being witnessed reflected in the character's expression",
        "reaction":  "surrounding environment resonating with the emotional weight",
        "timing":    "contemplative witness frame, the now fully inhabited",
    },
}


# Camera angle assignments based on emotion beat
_BEAT_CAMERA_MAP: Dict[str, str] = {
    "contained_fire": "low_angle",
    "fracture":       "dutch_tilt",
    "spiral":         "dutch_tilt",
    "peak_noise":     "close_up",
    "breakthrough":   "low_angle",
    "triumph":        "wide_shot",
    "stillness":      "wide_shot",
    "drift":          "medium_shot",
    "quiet_rest":     "bird_eye",
    "absence":        "wide_shot",
    "radiance":       "low_angle",
    "transcendence":  "bird_eye",
    "heaviness":      "medium_shot",
    "ache":           "close_up",
    "memory":         "medium_shot",
    "expansion":      "wide_shot",
    "momentum":       "low_angle",
    "spark":          "close_up",
    "surrender":      "medium_shot",
    "doubt":          "medium_shot",
    "challenge":      "low_angle",
    "resistance":     "medium_shot",
    "vulnerability":  "close_up",
    "recognition":    "medium_shot",
    "embrace":        "medium_shot",
    "neutral":        "medium_shot",
}

# Size class per emotion beat for LayoutDirective
_BEAT_SIZE_MAP: Dict[str, str] = {
    "contained_fire": "large",
    "fracture":       "large",
    "breakthrough":   "full_page",
    "triumph":        "full_page",
    "peak_noise":     "large",
    "spiral":         "medium",
    "stillness":      "medium",
    "quiet_rest":     "small",
    "drift":          "small",
    "ache":           "small",
    "absence":        "large",
    "momentum":       "large",
    "radiance":       "full_page",
    "neutral":        "medium",
}

# Body pose templates per emotion beat
_BEAT_POSE_MAP: Dict[str, Dict[str, str]] = {
    "contained_fire": {"body": "standing rigid, fists clenched", "head": "chin down, jaw set", "arms": "locked at sides", "legs": "planted wide"},
    "fracture":       {"body": "lurching forward, off-balance", "head": "snapping upward", "arms": "one arm extended", "legs": "staggered"},
    "spiral":         {"body": "hunched and curling inward", "head": "bowed down", "arms": "wrapped around torso", "legs": "knees bent"},
    "stillness":      {"body": "standing perfectly still", "head": "level, forward", "arms": "hanging loose at sides", "legs": "feet together"},
    "drift":          {"body": "slouched or lying back, slack", "head": "tilted back softly", "arms": "open palms up", "legs": "extended and relaxed"},
    "breakthrough":   {"body": "lunging forward, full extension", "head": "raised, eyes wide", "arms": "one arm thrusting forward", "legs": "back leg extended in stride"},
    "triumph":        {"body": "standing tall, chest open", "head": "raised to sky", "arms": "raised wide above head", "legs": "planted strong and wide"},
    "quiet_rest":     {"body": "lying fully relaxed", "head": "resting sideways", "arms": "beside body", "legs": "extended uncrossed"},
    "peak_noise":     {"body": "rigid, hands pressed to ears", "head": "wincing", "arms": "bent up blocking", "legs": "braced"},
    "ache":           {"body": "curled slightly, shoulders dropped", "head": "looking downward", "arms": "folded across chest", "legs": "knees softly bent"},
    "vulnerability":  {"body": "open stance, shoulders back but exposed", "head": "level, eyes meeting", "arms": "slightly open outward", "legs": "close together"},
    "momentum":       {"body": "mid-stride, leaning forward into motion", "head": "forward-set, determined", "arms": "pumping in stride", "legs": "one forward one back, dynamic"},
    "neutral":        {"body": "standing naturally, weight balanced", "head": "forward and level", "arms": "relaxed at sides", "legs": "shoulder-width apart"},
}

# Expression templates per beat
_BEAT_EXPRESSION_MAP: Dict[str, Dict[str, str]] = {
    "contained_fire": {"emotion": "controlled fury", "eyes": "narrowed and burning", "mouth": "pressed tight"},
    "fracture":       {"emotion": "cracking anger", "eyes": "wide and sharp", "mouth": "open in a yell"},
    "spiral":         {"emotion": "panicked desperation", "eyes": "darting, unfocused", "mouth": "slightly open"},
    "stillness":      {"emotion": "quiet emptiness", "eyes": "soft and distant", "mouth": "neutral, closed"},
    "drift":          {"emotion": "peaceful numbness", "eyes": "half-closed", "mouth": "slightly open, relaxed"},
    "breakthrough":   {"emotion": "fierce determination", "eyes": "blazing wide open", "mouth": "set hard"},
    "triumph":        {"emotion": "exhausted relief and joy", "eyes": "bright and wide", "mouth": "open smile"},
    "quiet_rest":     {"emotion": "deep peaceful calm", "eyes": "closed", "mouth": "softly closed"},
    "ache":           {"emotion": "quiet grief", "eyes": "glassy, downcast", "mouth": "trembling slightly"},
    "absence":        {"emotion": "hollow emptiness", "eyes": "unfocused, staring", "mouth": "slightly parted"},
    "radiance":       {"emotion": "transcendent wonder", "eyes": "wide, lit from within", "mouth": "parted softly"},
    "momentum":       {"emotion": "determined urgency", "eyes": "focused ahead", "mouth": "set, lips pressed"},
    "neutral":        {"emotion": "neutral present", "eyes": "forward and clear", "mouth": "closed, relaxed"},
}


class StoryDirector(BaseAgent):
    def __init__(self):
        super().__init__("story_director")

    def plan(self, story_config: Dict[str, Any], memory: "StorySectionMemory") -> Dict[str, Any]:
        """Loads the base panel sequences and characters into memory."""
        panels = story_config.get("panels", [])
        self.log.info(f"Story Director loaded {len(panels)} raw panel outlines.")
        memory.total_panels = len(panels)
        memory.raw_panels = panels

        # Story arc data
        if "recurring_motif" in story_config:
            memory.recurring_motif = story_config["recurring_motif"]
        if "mood_journey" in story_config:
            memory.mood_journey = story_config["mood_journey"]

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
            memory.main_character = main_char
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

        # 4. Characters from panels (scene graph format)
        for p in panels:
            for char_obj in p.get("characters", []):
                char_id = char_obj.get("id")
                if char_id:
                    memory.register_character(char_id)
                    memory.register_character(char_id.capitalize())
                    # Extract costume from pose.body if available
                    pose = char_obj.get("pose", {})
                    body_desc = pose.get("body", "")
                    if body_desc and char_id in memory.characters:
                        if not memory.characters[char_id].costume_desc:
                            memory.characters[char_id].costume_desc = body_desc

        return {"status": "Story framework initialized", "panel_count": len(panels)}

    def update(self, panel_result: Dict[str, Any], memory: "StorySectionMemory"):
        pass


class ActionDirector(BaseAgent):
    def __init__(self):
        super().__init__("action_director")

    def plan(self, story_config: Dict[str, Any], memory: "StorySectionMemory") -> Dict[str, Any]:
        """
        Validates and cinematically exaggerates every action in every panel.

        For each action dict on the blackboard, resolves the verb against
        ACTION_EXAGGERATION_MAP and writes 5 additional fields:
            exaggerated_verb, body_mechanics, impact_fx,
            environmental_reaction, cinematic_timing

        These fields are consumed by panel_engine._build_prompt() Step 7
        to construct a 40-60 word action clause instead of a 3-word fragment.
        If the raw verb does not match any entry, falls back to the 'observes'
        entry so the prompt always has dimensional language.
        """
        panels = memory.raw_panels or story_config.get("panels", [])
        enriched = 0
        exaggerated = 0

        # Beat → canonical verb key for panels that arrive with NO actions at all
        _BEAT_DEFAULT_VERB: Dict[str, str] = {
            "contained_fire": "holds",
            "breakthrough":   "charge",
            "triumph":        "raises",
            "stillness":      "stands",
            "drift":          "floats",
            "spiral":         "clutches",
            "ache":           "sits",
            "momentum":       "run",
            "fracture":       "punch",
            "peak_noise":     "clutches",
            "absence":        "watches",
            "radiance":       "raises",
            "transcendence":  "floats",
            "vulnerability":  "reaches",
            "recognition":    "watches",
            "heaviness":      "crawl",
            "surrender":      "fall",
            "drag":           "crawl",
        }

        for panel in panels:
            actions = panel.get("actions", [])
            beat = panel.get("emotion_beat", "neutral")

            # ── Synthesise a default action if the panel has none ──
            if not actions:
                char_list = panel.get("characters", [])
                actor = char_list[0].get("id", "character") if char_list else "character"
                default_verb = _BEAT_DEFAULT_VERB.get(beat, "observes")
                panel["actions"] = [{"actor": actor, "verb": default_verb, "target": ""}]
                enriched += 1

            # ── Exaggerate every action with the 5-layer cinematic map ──
            for action in panel["actions"]:
                # Only enrich if mechanics is missing or empty
                if not action.get("mechanics") or action["mechanics"].strip() == "":
                    raw_verb = action.get("verb", "observes").lower().strip()

                    # Fuzzy match: 'runs' hits 'run', 'punching' hits 'punch'
                    matched_key = None
                    for key in ACTION_EXAGGERATION_MAP:
                        if raw_verb.startswith(key) or key.startswith(raw_verb):
                            matched_key = key
                            break
                    if not matched_key:
                        matched_key = "observes"

                    cinematic = ACTION_EXAGGERATION_MAP[matched_key]

                    # Enrich: use cinematic verb if original verb is empty or generic
                    if action.get("verb", "").strip() in ("", "observes") or action.get("verb", "").lower().strip() == matched_key:
                        action["verb"] = cinematic["verb"]

                    action["mechanics"] = action.get("mechanics") or cinematic["mechanics"]
                    action["impact"]    = action.get("impact") or cinematic["impact"]
                    action["reaction"]  = action.get("reaction") or cinematic["reaction"]
                    action["timing"]    = action.get("timing") or cinematic["timing"]
                    exaggerated += 1

        self.log.info(
            f"Action Director: {enriched} panels given default actions, "
            f"{exaggerated} action(s) exaggerated with 5-layer cinematic descriptors."
        )
        return {"status": "Actions verified and cinematically exaggerated",
                "enriched": enriched, "exaggerated": exaggerated}

    def update(self, panel_result: Dict[str, Any], memory: "StorySectionMemory"):
        pass


class DialogueWriter(BaseAgent):
    def __init__(self, ollama_model: Optional[str] = None, ollama_url: Optional[str] = None):
        super().__init__("dialogue_writer")
        import os
        from utils.config_helper import load_env_with_defaults
        env_defaults = load_env_with_defaults()
        self.ollama_model = ollama_model or os.environ.get("OLLAMA_MODEL") or env_defaults.get("llm_provider", "llama3.2")
        self.ollama_url = ollama_url or os.environ.get("OLLAMA_URL") or env_defaults.get("ollama_url", "http://localhost:11434")

    def _call_llm(self, prompt: str, system_prompt: str) -> Optional[str]:
        import urllib.request
        import urllib.error
        import json
        
        url = f"{self.ollama_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {"temperature": 0.5}
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                resp_data = json.loads(response.read().decode("utf-8"))
                return resp_data.get("response", "").strip()
        except Exception as e:
            self.log.warning(f"Ollama direct call failed in DialogueWriter: {e}")
            return None

    def plan(self, story_config: Dict[str, Any], memory: "StorySectionMemory") -> Dict[str, Any]:
        """Validates dialogue — fills missing/empty text dynamically using LLM or fallbacks."""
        panels = memory.raw_panels or story_config.get("panels", [])
        filled = 0
        beat_lines = {
            "contained_fire": "Not yet.",
            "fracture":       "That's enough.",
            "spiral":         "I can't stop it.",
            "breakthrough":   "Move.",
            "triumph":        "We did it.",
            "stillness":      "...",
            "drift":          "Just for a moment.",
            "quiet_rest":     "...",
            "absence":        "Where did you go?",
            "radiance":       "I see it now.",
            "ache":           "I miss you.",
            "momentum":       "Keep going.",
            "doubt":          "Am I ready for this?",
            "challenge":      "It's bigger than I thought.",
            "renewal":        "Starting again.",
        }
        
        main_char = memory.main_character or "Wanderer"
        mood_journey = memory.mood_journey or "a journey of emotion"
        
        for panel in panels:
            beat = panel.get("emotion_beat", "neutral")
            actions = panel.get("actions", [])
            action_desc = ""
            if actions:
                action_desc = ", ".join(f"{a.get('actor', '')} {a.get('verb', '')} {a.get('target', '')}" for a in actions)
            
            for char in panel.get("characters", []):
                char_name = char.get("id", main_char)
                dlg = char.get("dialogue", {})
                
                if not isinstance(dlg, dict):
                    dlg = {"text": str(dlg)}
                    
                text = dlg.get("text", "").strip()
                if not text or text in ("...", ""):
                    sys_prompt = (
                        "You are a professional comic book writer. Write a single line of speech or thought "
                        f"for the character '{char_name}'. Keep it short (max 15 words), punchy, and natural. "
                        "Return ONLY the dialogue text. No quotation marks, no parentheticals, no formatting."
                    )
                    prompt = (
                        f"Character: {char_name}\n"
                        f"Scene Action: {action_desc or 'standing in the scene'}\n"
                        f"Current Emotion Beat: {beat}\n"
                        f"Emotional Journey context: {mood_journey}\n"
                        "Write a single expressive line of dialogue or inner thought that fits this moment:"
                    )
                    
                    generated_text = self._call_llm(prompt, sys_prompt)
                    if generated_text:
                        generated_text = generated_text.replace('"', '').replace("'", "")
                        if generated_text.lower().startswith(f"{char_name.lower()}:"):
                            generated_text = generated_text[len(char_name)+1:].strip()
                        
                        dlg["text"] = generated_text
                        char["dialogue"] = dlg
                        filled += 1
                        self.log.info(f"Dynamically generated dialogue for '{char_name}': {generated_text}")
                    else:
                        dlg["text"] = beat_lines.get(beat, "...")
                        char["dialogue"] = dlg
                        filled += 1
                        
        self.log.info(f"Dialogue Writer validated panels, filled {filled} blank lines.")
        return {"status": "Dialogue validated", "filled": filled}

    def update(self, panel_result: Dict[str, Any], memory: "StorySectionMemory"):
        pass


class PoseDirector(BaseAgent):
    def __init__(self):
        super().__init__("pose_director")

    def plan(self, story_config: Dict[str, Any], memory: "StorySectionMemory") -> Dict[str, Any]:
        """Assigns default body poses based on emotion beat where missing."""
        panels = memory.raw_panels or story_config.get("panels", [])
        filled = 0
        for panel in panels:
            beat = panel.get("emotion_beat", "neutral")
            default_pose = _BEAT_POSE_MAP.get(beat, _BEAT_POSE_MAP["neutral"])
            default_expr = _BEAT_EXPRESSION_MAP.get(beat, _BEAT_EXPRESSION_MAP["neutral"])
            for char in panel.get("characters", []):
                pose = char.get("pose", {})
                expr = char.get("expression", {})
                # Fill missing pose fields
                for field, val in default_pose.items():
                    if not pose.get(field):
                        pose[field] = val
                        filled += 1
                char["pose"] = pose
                # Fill missing expression fields
                for field, val in default_expr.items():
                    if not expr.get(field):
                        expr[field] = val
                char["expression"] = expr

                # Update memory character state
                char_id = char.get("id", "")
                if char_id and char_id in memory.characters:
                    memory.update_character(char_id,
                                           emotion=default_expr.get("emotion", "neutral"),
                                           last_action=panel.get("actions", [{}])[0].get("verb", ""))

        self.log.info(f"Pose Director filled {filled} missing pose fields across panels.")
        return {"status": "Poses locked", "filled": filled}

    def update(self, panel_result: Dict[str, Any], memory: "StorySectionMemory"):
        pass


class EmotionDirector(BaseAgent):
    def __init__(self):
        super().__init__("emotion_director")

    def plan(self, story_config: Dict[str, Any], memory: "StorySectionMemory") -> Dict[str, Any]:
        """Updates character emotion states in memory per panel and validates arc beats."""
        panels = memory.raw_panels or story_config.get("panels", [])
        arc_beats = []
        for panel in panels:
            beat = panel.get("emotion_beat", "neutral")
            if not beat:
                # Try to infer from expression
                chars = panel.get("characters", [])
                if chars:
                    beat = chars[0].get("expression", {}).get("emotion", "neutral")
                panel["emotion_beat"] = beat or "neutral"
            arc_beats.append(beat)

            # Update character emotion in memory
            for char in panel.get("characters", []):
                char_id = char.get("id", "")
                expr = char.get("expression", {})
                if char_id and char_id in memory.characters:
                    memory.update_character(char_id, emotion=expr.get("emotion", beat))

        # Store arc beats in memory
        if arc_beats:
            memory.arc_beats = arc_beats

        self.log.info(f"Emotion Director set arc beats: {arc_beats}")
        return {"status": "Emotions set", "arc": arc_beats}

    def update(self, panel_result: Dict[str, Any], memory: "StorySectionMemory"):
        # Advance the beat index after each panel is generated
        if memory.current_beat_index < len(memory.arc_beats) - 1:
            memory.current_beat_index += 1


class CameraDirector(BaseAgent):
    def __init__(self):
        super().__init__("camera_director")

    def plan(self, story_config: Dict[str, Any], memory: "StorySectionMemory") -> Dict[str, Any]:
        """
        Assigns camera angles and layout directives to each panel based on
        emotion beat and panel narrative position.
        """
        panels = memory.raw_panels or story_config.get("panels", [])
        total = len(panels)
        assigned = 0

        for i, panel in enumerate(panels):
            panel_id = panel.get("panel", i + 1)
            beat = panel.get("emotion_beat", "neutral")

            # Assign camera angle to panel if missing or generic
            current_camera = panel.get("camera", "")
            if not current_camera or current_camera.lower() in ("medium shot", ""):
                camera_angle = _BEAT_CAMERA_MAP.get(beat, "medium_shot")
                panel["camera"] = camera_angle

            # Create LayoutDirective for this panel
            size_class = _BEAT_SIZE_MAP.get(beat, "medium")
            camera_angle_key = _BEAT_CAMERA_MAP.get(beat, "medium_shot")

            # Panel position adjustments
            ratio = i / max(1, total - 1)
            if ratio == 0.0 or ratio == 1.0:
                # First and last panels always get larger treatment
                size_class = max(size_class, "large",
                                 key=lambda s: ["small", "medium", "large", "full_page"].index(s))

            directive = LayoutDirective(
                panel_id=panel_id,
                size_class=size_class,
                camera_angle=camera_angle_key,
                camera_framing="center",
                aspect_ratio=(1, 1),
                gutter_emphasis="normal",
            )
            memory.set_layout_directive(panel_id, directive)
            assigned += 1

        # Update scene location from first panel's environment
        if panels:
            env = panels[0].get("environment", "")
            if env and isinstance(env, str):
                memory.update_scene(location=env[:80])

        self.log.info(f"Camera Director assigned layout directives to {assigned} panels.")
        return {"status": "Camera angles and layout directives set", "assigned": assigned}

    def update(self, panel_result: Dict[str, Any], memory: "StorySectionMemory"):
        pass
