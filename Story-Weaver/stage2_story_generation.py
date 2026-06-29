"""
MoodWeaver — Stage 2: Story Generation Engine (Unified)
========================================================
Combines:
  - Dynamic 4–10 panel generation with mood arcs  (from story_gen.py)
  - Training pipeline with rich examples           (from story_gen_old.py)
  - Fixed prompt that enforces non-empty fields
  - .env config support

Modes:
  generate  → run inference with merged/finetuned model
  train     → fine-tune base model with Unsloth
  dataset   → just build & save the JSONL training file

Usage:
  python stage2_story_generation.py                          # generate (reads .env)
  python stage2_story_generation.py --mode train             # fine-tune
  python stage2_story_generation.py --mode dataset           # save JSONL only
  python stage2_story_generation.py --mode generate --emotion angry --panels 8
  python stage2_story_generation.py --mode train --epochs 5 --model llama

pip install transformers accelerate bitsandbytes unsloth datasets trl torch python-dotenv
"""

import json, re, time, argparse, logging, os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("moodweaver.stage2")

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.split("#")[0].strip())

def _env(key, default=""):
    return os.environ.get(key, default).strip()

# ---------------------------------------------------------------------------
# 1. Schemas
# ---------------------------------------------------------------------------

@dataclass
class EmotionOutput:
    primary_emotion:    str
    confidence:         float
    secondary_emotions: list       # [{"emotion": str, "score": float}]
    somatic_markers:    bool
    user_text:          str


@dataclass
class Panel:
    panel:        int
    visual:       str
    dialogue:     str
    emotion_beat: str
    motion:       str


@dataclass
class StoryScript:
    recurring_motif:  str
    mood_journey:     str
    panels:           list
    generation_time_s: float = 0.0
    model_used:       str   = ""

    def to_dict(self) -> dict:
        return {
            "recurring_motif": self.recurring_motif,
            "mood_journey":    self.mood_journey,
            "panels": [
                {"panel": p.panel, "visual": p.visual, "dialogue": p.dialogue,
                 "emotion_beat": p.emotion_beat, "motion": p.motion}
                for p in self.panels
            ],
            "_meta": {
                "generation_time_s": round(self.generation_time_s, 2),
                "model_used":        self.model_used,
            },
        }


# ---------------------------------------------------------------------------
# 2. Model Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "llama": {
        "repo":          "unsloth/Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit",
        "display":       "Qwen2.5 1.5B (Llama slot)",
        "max_new_tokens": 800,
        "no_quantize":   True,   # already 4-bit from unsloth hub
    },
    "mistral": {
        "repo":          "mistralai/Mistral-7B-Instruct-v0.2",
        "display":       "Mistral 7B",
        "max_new_tokens": 900,
        "no_quantize":   False,
    },
    "tiny": {
        "repo":          "Qwen/Qwen2.5-0.5B-Instruct",
        "display":       "Qwen 0.5B (CPU test)",
        "max_new_tokens": 600,
        "no_quantize":   True,
    },
    "finetuned": {
        "repo":          _env("MODEL_PATH", "moodweaver_stage2_merged"),
        "display":       "MoodWeaver Fine-tuned (merged)",
        "max_new_tokens": 1400,   # raised: model was truncating mid-panel at 900
        "no_quantize":   True,
    },
}

# ---------------------------------------------------------------------------
# 3. Mood Arc Definitions
# ---------------------------------------------------------------------------

def _load_arcs_config() -> dict:
    """Load arc definitions from shared arcs_config.json, with fallback to inline."""
    candidates = [
        Path(__file__).parent.parent / "indie_comic_pipeline" / "config" / "arcs_config.json",
        Path(__file__).parent / "arcs_config.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8-sig") as f:
                    cfg = json.load(f)
                logging.getLogger("moodweaver.stage2").info(
                    f"Loaded arc config from {p}"
                )
                return cfg
            except Exception as e:
                logging.getLogger("moodweaver.stage2").warning(
                    f"Could not load arcs_config.json from {p}: {e}"
                )
    return {}


def _build_mood_arcs_from_config(cfg: dict) -> dict:
    """Convert arcs_config.json mood_to_arc entries into MOOD_ARCS format."""
    mood_to_arc = cfg.get("mood_to_arc", {})
    result = {}
    for _label, entry in mood_to_arc.items():
        arc_key = entry.get("arc_key")
        if not arc_key:  # surprise has no arc_key (uses fallback)
            continue
        result[arc_key] = {
            "journey":     entry.get("journey", ""),
            "description": entry.get("description", ""),
            "arc_beats":   entry.get("arc_beats", []),
            "motif_hint":  entry.get("motif_hint", ""),
            "end_note":    entry.get("end_note", ""),
        }
    # Also pull the tired arc (top-level entry)
    if "tired" in cfg:
        t = cfg["tired"]
        result["tired"] = {
            "journey":     t.get("journey", ""),
            "description": t.get("description", ""),
            "arc_beats":   t.get("arc_beats", []),
            "motif_hint":  t.get("motif_hint", ""),
            "end_note":    t.get("end_note", ""),
        }
    return result


def _build_secondary_defaults_from_config(cfg: dict) -> dict:
    """Convert secondary_emotion_defaults from config keys to short mood names."""
    raw_sec = cfg.get("secondary_emotion_defaults", {})
    if not raw_sec:
        return {}
    key_map = {
        "sadness": "sad",
        "joy": "happy",
        "anger": "angry",
        "fear": "anxious",
        "grief": "grief",
        "love": "love",
        "determined": "determined",
        "surprise": "surprise"
    }
    result = {}
    for raw_key, val in raw_sec.items():
        mapped_key = key_map.get(raw_key, raw_key)
        result[mapped_key] = val
    return result


_ARCS_CONFIG = _load_arcs_config()
_config_arcs = _build_mood_arcs_from_config(_ARCS_CONFIG) if _ARCS_CONFIG else {}

MOOD_ARCS = _config_arcs if _config_arcs else {
    "sad": {
        "journey":     "uplifting",
        "description": "From heaviness toward genuine small warmth — not forced positivity",
        "arc_beats":   ["heaviness", "stillness", "faint_warmth", "tentative_light", "soft_openness", "quiet_hope"],
        "motif_hint":  "something small that holds warmth — a ceramic cup, a candle flame, a patch of late sunlight on a floor",
        "end_note":    "Panel ends with something small but warm. Not solved. Lighter.",
    },
    "angry": {
        "journey":     "calming",
        "description": "From contained fire toward stillness — anger is valid, the body finds ground",
        "arc_beats":   ["contained_fire", "fracture", "exhale", "cooling", "ground", "stillness"],
        "motif_hint":  "something that absorbs or dissipates heat — running tap water, an open window, a cold countertop",
        "end_note":    "Panel ends with body grounded. Situation unresolved but breath returned.",
    },
    "tired": {
        "journey":     "relaxing",
        "description": "From bone-deep drag toward rest — permission to stop, body softening",
        "arc_beats":   ["drag", "surrender", "softness", "drift", "quiet_rest", "renewal"],
        "motif_hint":  "something soft and horizontal — a folded blanket, late evening light pooling on floorboards, a pillow",
        "end_note":    "Panel ends with genuine rest. Tomorrow does not exist in this panel.",
    },
    "happy": {
        "journey":     "elation",
        "description": "From spark of joy toward luminous transcendence — expanding, overflowing",
        "arc_beats":   ["spark", "expansion", "overflow", "radiance", "luminous_still", "transcendence"],
        "motif_hint":  "something that multiplies or radiates light — water reflections, open hands, laughter lines around eyes",
        "end_note":    "Panel ends with pure presence. Joy has become larger than its original cause.",
    },
    "anxious": {
        "journey":     "grounding",
        "description": "From spiral toward root — the mind slows, the body remembers the floor",
        "arc_beats":   ["spiral", "peak_noise", "pause", "breath", "root", "present"],
        "motif_hint":  "something tactile and grounding — a rough-textured surface, bare feet on cool tile, one object in sharp focus",
        "end_note":    "Panel ends with presence. Not solved. But here, now, inside this body.",
    },
    "grief": {
        "journey":     "tender continuance",
        "description": "From the shape of absence toward carrying — grief continues, so does the person",
        "arc_beats":   ["absence", "ache", "memory", "held", "continuance", "carried_forward"],
        "motif_hint":  "something that was shared — a particular chair, a mug with a chip, a quality of afternoon light",
        "end_note":    "Panel ends with both things true at once: the loss is real, and life moves alongside it.",
    },
    "determined": {
        "journey":     "heroic rise",
        "description": "From doubt toward resolute action — the cost of the climb is visible",
        "arc_beats":   ["doubt", "challenge", "resistance", "breakthrough", "momentum", "triumph"],
        "motif_hint":  "something that holds the cost of the climb (scarred hands, broken weapon, worn path)",
        "end_note":    "End with victory earned, not given. The character is changed by the climb.",
    },
    "love": {
        "journey":     "deepening",
        "description": "From spark toward enduring warmth — love as transformation, not destination",
        "arc_beats":   ["spark", "recognition", "vulnerability", "trust", "embrace", "unity"],
        "motif_hint":  "something shared between two people (a held hand, a shared window, intertwined roots)",
        "end_note":    "End with both people changed. Love as transformation, not destination.",
    },
}

DEFAULT_ARC = {
    "journey":     "reflective",
    "description": "From feeling toward witnessing",
    "arc_beats":   ["acknowledgment", "presence", "shift", "openness"],
    "motif_hint":  "something ordinary that carries unexpected weight",
    "end_note":    "End with openness. No resolution required.",
}

TIMING_PHASES = {
    4:  ["validation","validation","complication","openness"],
    5:  ["validation","validation","complication","shift","openness"],
    6:  ["validation","validation","complication","shift","shift","openness"],
    7:  ["validation","validation","validation","complication","shift","shift","openness"],
    8:  ["validation","validation","validation","complication","complication","shift","shift","openness"],
    9:  ["validation","validation","validation","complication","complication","shift","shift","shift","openness"],
    10: ["validation","validation","validation","complication","complication","shift","shift","shift","shift","openness"],
}

_cfg_secondary = _build_secondary_defaults_from_config(_ARCS_CONFIG) if _ARCS_CONFIG else {}

SECONDARY_DEFAULTS = _cfg_secondary if _cfg_secondary else {
    "sad":     [{"emotion":"exhaustion","score":0.15}, {"emotion":"longing","score":0.08}],
    "angry":   [{"emotion":"shame","score":0.12},      {"emotion":"hurt","score":0.09}],
    "tired":   [{"emotion":"numbness","score":0.13},   {"emotion":"longing","score":0.07}],
    "happy":   [{"emotion":"gratitude","score":0.14},  {"emotion":"wonder","score":0.08}],
    "anxious": [{"emotion":"dread","score":0.11},      {"emotion":"restless","score":0.09}],
    "grief":   [{"emotion":"loneliness","score":0.10}, {"emotion":"numbness","score":0.07}],
    "determined": [{"emotion":"fear","score":0.10},    {"emotion":"resolve","score":0.12}],
    "love":       [{"emotion":"tenderness","score":0.13},{"emotion":"wonder","score":0.09}],
}

# ---------------------------------------------------------------------------
# 4. Prompt Builder
# ---------------------------------------------------------------------------

def _get_beats(n: int, arc: dict) -> list:
    beats = arc["arc_beats"]
    if n <= len(beats):
        step = len(beats) / n
        return [beats[int(i * step)] for i in range(n)]
    result = []
    for i in range(n):
        idx = int(i * (len(beats) - 1) / (n - 1))
        result.append(beats[idx])
    return result


def build_system_prompt(panel_count: int) -> str:
    """
    Explicit field-level instructions so the model never leaves dialogue/motion empty.
    This is the key fix vs the old version.
    """
    return f"""\
You are a literary graphic novelist writing for adults.
Respond ONLY with valid JSON. No markdown fences. No explanation. No preamble.

ABSOLUTE RULES — violating any of these makes the output unusable:
1. Output exactly {panel_count} panels. No more, no fewer.
2. Every panel MUST have ALL four fields filled with real content:
   - "visual"       : a scene description (2–4 sentences). MUST contain the recurring motif word.
   - "dialogue"     : spoken text OR exactly "..." if silence. NEVER empty string "".
   - "emotion_beat" : exactly ONE word. Not a phrase. Not empty.
   - "motion"       : what the character physically does (1–3 sentences). NEVER empty string "".
3. NEVER name emotions directly (no "sad", "angry", "tired", "anxious", "grief", "happy").
   Show emotions only through objects, physical actions, and body sensations.
4. ONE recurring visual motif must appear explicitly in every single panel's "visual" field.
5. Every panel's "visual" or "motion" must contain a body sensation word
   (chest, breath, throat, hands, stomach, jaw, shoulders, skin, pulse, etc.).
6. No moral lessons. Nothing beginning with "The lesson is" or "Remember that".
7. Dialogue shows what is NOT said as much as what is.

Output ONLY this exact JSON structure:
{{
  "recurring_motif": "precise 4–8 word description of the single visual motif",
  "mood_journey": "one sentence describing the emotional arc of the story",
  "panels": [
    {{"panel": 1, "visual": "...", "dialogue": "...", "emotion_beat": "one_word", "motion": "..."}},
    {{"panel": 2, "visual": "...", "dialogue": "...", "emotion_beat": "one_word", "motion": "..."}},
    {{"panel": 3, "visual": "...", "dialogue": "...", "emotion_beat": "one_word", "motion": "..."}},
    {{"panel": 4, "visual": "...", "dialogue": "...", "emotion_beat": "one_word", "motion": "..."}}
  ]
}}"""


def build_user_prompt(emotion: EmotionOutput, panel_count: int) -> str:
    arc    = MOOD_ARCS.get(emotion.primary_emotion.lower(), DEFAULT_ARC)
    beats  = _get_beats(panel_count, arc)
    phases = TIMING_PHASES.get(panel_count, TIMING_PHASES[6])
    sec    = ", ".join(f"{e['emotion']} {e['score']}" for e in emotion.secondary_emotions)

    beat_guide = "\n".join(
        f"  Panel {i+1} [{phases[i]}]  →  emotion_beat should evoke: {beats[i]}"
        for i in range(panel_count)
    )

    return (
        f"Primary emotion: {emotion.primary_emotion} (confidence {emotion.confidence})\n"
        f"Secondary emotions: {sec}\n"
        f"Somatic markers present: {emotion.somatic_markers}\n"
        f'User context: "{emotion.user_text}"\n\n'
        f"MOOD JOURNEY: {arc['journey']} — {arc['description']}\n"
        f"MOTIF HINT: {arc['motif_hint']}\n\n"
        f"Write exactly {panel_count} panels. Follow this emotional arc:\n"
        f"{beat_guide}\n\n"
        f"Arc direction: {beats[0]} → {beats[-1]}\n"
        f"{arc['end_note']}\n\n"
        f"REMINDER: Every panel needs non-empty visual, dialogue, emotion_beat, AND motion.\n"
        f"Write the JSON now."
    )


# ---------------------------------------------------------------------------
# 5. Story Generator (Inference)
# ---------------------------------------------------------------------------

class StoryGenerator:
    tokenizer: Any
    model: Any

    def __init__(self, model_key: str = "finetuned", device: str = "auto"):
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{model_key}'. Choose: {list(MODEL_REGISTRY)}")
        self.cfg       = MODEL_REGISTRY[model_key]
        self.model_key = model_key
        self._load(device)

    def _load(self, device: str):
        repo = self.cfg["repo"]
        log.info(f"Loading {self.cfg['display']} from {repo} ...")

        bnb_cfg = None
        if not self.cfg.get("no_quantize"):
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
        assert tok is not None
        if getattr(tok, "pad_token", None) is None:
            tok.pad_token = tok.eos_token
        self.tokenizer = tok

        dtype = None if bnb_cfg else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            repo,
            quantization_config=bnb_cfg,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        self.model.eval()
        log.info(f"{self.cfg['display']} ready.")

    def generate(self, emotion: EmotionOutput, panel_count: int = 6,
                 temperature: Optional[float] = None, top_p: Optional[float] = None,
                 rep_penalty: Optional[float] = None, max_retries: int = 3) -> StoryScript:

        tok = self.tokenizer
        model = self.model
        assert tok is not None and model is not None

        # Dynamically set optimal parameter overrides per emotion when defaults/None are passed
        emo_key = emotion.primary_emotion.lower()
        if emo_key == "sadness": emo_key = "sad"
        elif emo_key == "joy": emo_key = "happy"
        elif emo_key == "anger": emo_key = "angry"
        elif emo_key == "fear": emo_key = "anxious"

        opt_params = {
            "sad":        {"t": 0.5,  "p": 0.95, "r": 1.1,  "m": 250},
            "happy":      {"t": 0.5,  "p": 0.95, "r": 1.1,  "m": 160},
            "angry":      {"t": 0.6,  "p": 0.90, "r": 1.0,  "m": 160},
            "anxious":    {"t": 1.0,  "p": 0.70, "r": 1.1,  "m": 100},
            "grief":      {"t": 0.95, "p": 0.70, "r": 1.05, "m": 180},
            "tired":      {"t": 0.55, "p": 0.88, "r": 1.2,  "m": 220},
            "determined": {"t": 0.80, "p": 0.85, "r": 1.1,  "m": 180},
            "love":       {"t": 0.70, "p": 0.90, "r": 1.15, "m": 200},
        }.get(emo_key, {"t": 0.72, "p": 0.92, "r": 1.15, "m": 150})

        # Allow fallback from legacy default values (0.72, 0.92, 1.15) to optimal ones
        if temperature is None or temperature == 0.72:
            temperature = opt_params["t"]
        if top_p is None or top_p == 0.92:
            top_p = opt_params["p"]
        if rep_penalty is None or rep_penalty == 1.15:
            rep_penalty = opt_params["r"]

        # Ensure max_new_tokens is large enough for the dynamic per-panel token budget
        max_tokens_budget = max(self.cfg.get("max_new_tokens", 900), opt_params["m"] * panel_count)

        messages = [
            {"role": "system", "content": build_system_prompt(panel_count)},
            {"role": "user",   "content": build_user_prompt(emotion, panel_count)},
        ]

        # Apply chat template
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        for attempt in range(1, max_retries + 1):
            log.info(f"Generation attempt {attempt}/{max_retries} ...")
            t0 = time.time()

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens_budget,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=rep_penalty,
                    pad_token_id=tok.eos_token_id,
                )

            raw     = tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
            elapsed = round(time.time() - t0, 2)
            log.info(f"Raw output ({elapsed}s):\n{raw[:400]}...")

            try:
                data   = self._extract_json(raw)
                script = self._validate(data, panel_count, emotion)
                script.generation_time_s = elapsed
                script.model_used        = str(self.cfg["display"])
                return script
            except Exception as e:
                log.warning(f"Attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Generation failed after {max_retries} attempts.\n"
                        f"Last error: {e}\n\nRaw output:\n{raw}"
                    )
        raise RuntimeError("Unexpected end of generation loop")

    def _extract_json(self, raw: str) -> dict:
        # 1. Strip markdown fences
        clean = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        clean = re.sub(r"\s*```$", "", clean).strip()

        # 2. Find outermost { } block
        start = clean.find("{")
        if start == -1:
            raise ValueError("No JSON object found in output")
        depth, end = 0, -1
        for i, ch in enumerate(clean[start:], start):
            if ch == "{":   depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0: end = i; break
        if end == -1:
            # truncated output — grab everything from start and let repair fix it
            clean = clean[start:]
        else:
            clean = clean[start:end+1]

        # 3. Direct parse
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            pass

        # 4. Repair then parse
        repaired = self._repair_json(clean)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON repair failed: {e}\nRepaired snippet:\n{repaired[:400]}")

    def _repair_json(self, text: str) -> str:
        """
        Fix the three failure modes from the logs:

        Mode 1 — Missing comma between adjacent key-value pairs:
            "dialogue": "...even when I hold onto my hopes... "
            "emotion_beat": "faint_warmth"
          → needs a comma after the dialogue value

        Mode 2 — Trailing comma before } or ]:
            "motion": "...",
            }
          → remove the trailing comma

        Mode 3 — Truncated / unclosed string (model hit max_new_tokens):
            "motion": "She slowly reaches for
          → close the string, then close open objects and arrays
        """

        # -- Fix 1: missing comma between two adjacent "key": value lines --
        # Matches: end-of-value  whitespace+newline  start-of-next-key
        text = re.sub(
            r'("(?:[^"\\]|\\.)*"|true|false|null|-?\d+(?:\.\d+)?)'
            r'(\s*\n\s*)'
            r'(")'  ,
            r'\1,\2\3',
            text
        )

        # -- Fix 2: trailing comma before } or ] ---------------------------
        text = re.sub(r',([\s\n]*[}\]])', r'\1', text)

        # -- Fix 3: detect and close truncated strings / open delimiters ---
        in_string = False
        escape    = False
        depth_obj = 0
        depth_arr = 0

        for ch in text:
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string:
                if   ch == "{": depth_obj += 1
                elif ch == "}": depth_obj = max(0, depth_obj - 1)
                elif ch == "[": depth_arr += 1
                elif ch == "]": depth_arr = max(0, depth_arr - 1)

        tail = ""
        if in_string:
            tail += '"' + "..."   # close the open string with ellipsis placeholder
        tail += "]" * depth_arr
        tail += "}" * depth_obj
        text = text + tail

        # Re-apply trailing comma fix after tail additions
        text = re.sub(r',([\s\n]*[}\]])', r'\1', text)

        return text

    def _validate(self, data: dict, expected_panels: int, emotion: EmotionOutput) -> StoryScript:
        panels_raw = data.get("panels", [])

        # -- Soft panel-count fix: if model gave n-1 or n+1, accept and pad/trim --
        if len(panels_raw) == 0:
            raise ValueError(f"Got 0 panels — output did not follow schema at all")

        if len(panels_raw) != expected_panels:
            log.warning(
                f"Panel count mismatch: expected {expected_panels}, "
                f"got {len(panels_raw)}. Auto-adjusting."
            )
            if len(panels_raw) > expected_panels:
                panels_raw = panels_raw[:expected_panels]   # trim extras
            else:
                # Duplicate last panel to fill gap (better than crashing)
                while len(panels_raw) < expected_panels:
                    dup = dict(panels_raw[-1])
                    dup["panel"] = len(panels_raw) + 1
                    panels_raw.append(dup)

        FALLBACKS = {
            "visual":       "A quiet room. The recurring motif present in the corner.",
            "dialogue":     "...",
            "emotion_beat": "stillness",
            "motion":       "Stands still. Breathes slowly.",
        }

        panels = []
        for idx, p in enumerate(panels_raw):
            # Fill missing or empty fields with fallbacks instead of crashing
            for k, fallback in FALLBACKS.items():
                if k not in p or not str(p.get(k, "")).strip():
                    log.warning(f"Panel {p.get('panel', idx+1)} missing/empty '{k}' — using fallback")
                    p[k] = fallback

            panels.append(Panel(
                panel=int(p.get("panel", idx + 1)),
                visual=p["visual"],
                dialogue=p["dialogue"],
                emotion_beat=p["emotion_beat"],
                motion=p["motion"],
            ))

        return StoryScript(
            recurring_motif=data.get("recurring_motif", "an object in the room"),
            mood_journey=data.get("mood_journey", ""),
            panels=panels,
        )


# ---------------------------------------------------------------------------
# 6. Training Dataset  (rich examples — all 6 emotions)
# ---------------------------------------------------------------------------

TRAINING_EXAMPLES = [

    # ── SADNESS → UPLIFT ────────────────────────────────────────────────────
    {
        "emotion": EmotionOutput(
            primary_emotion="sadness", confidence=0.72,
            secondary_emotions=[{"emotion":"exhaustion","score":0.15},{"emotion":"longing","score":0.08}],
            somatic_markers=True,
            user_text="i dont know why but everything just feels really heavy lately. even small things.",
        ),
        "panel_count": 6,
        "target": {
            "recurring_motif": "a ceramic mug with a small chip on the rim",
            "mood_journey": "From the weight of an ordinary evening toward a small, unforced warmth.",
            "panels": [
                {
                    "panel": 1,
                    "visual": "Kitchen table at dusk. The chipped ceramic mug sits half-full, tea gone cold. A hand rests flat beside it, not reaching. Chest heavy, breath shallow.",
                    "dialogue": "...",
                    "emotion_beat": "heaviness",
                    "motion": "The hand does not move. Fingers spread slightly against the wood grain. Stays.",
                },
                {
                    "panel": 2,
                    "visual": "Same table. Window now dark. The chipped mug still there, now a small lamp on. The hand has moved — index finger tracing the chip on the rim, slowly.",
                    "dialogue": "I should eat something.",
                    "emotion_beat": "stillness",
                    "motion": "Finger circles the chip. Does not get up. The thought floats and dissolves.",
                },
                {
                    "panel": 3,
                    "visual": "Close on the chipped mug. Steam — someone has refilled it. Hands wrapped around the warm ceramic now, knuckles pale then relaxing. Chest loosens slightly.",
                    "dialogue": "...",
                    "emotion_beat": "faint_warmth",
                    "motion": "Both hands cup the mug. Shoulders drop one centimetre. Eyes close briefly.",
                },
                {
                    "panel": 4,
                    "visual": "The person at the window, mug in hand. Outside, a streetlamp has come on. The chip in the rim catches the light. Throat still tight, but breath has deepened.",
                    "dialogue": "It's still there.",
                    "emotion_beat": "tentative_light",
                    "motion": "Looks out. Does not explain what they mean. Takes a slow sip.",
                },
                {
                    "panel": 5,
                    "visual": "Back at the table. The mug sits empty. Person seated again, but posture different — one elbow on the table, head slightly tilted. The chip visible in the quiet light.",
                    "dialogue": "...",
                    "emotion_beat": "soft_openness",
                    "motion": "Runs thumb over the chip one more time. Sets the mug down gently. Doesn't rush.",
                },
                {
                    "panel": 6,
                    "visual": "Wide shot. Kitchen quiet. The chipped mug on the counter now, lamp still on. The person has moved to the doorway. Shoulders not as high. Chest softer.",
                    "dialogue": "Okay.",
                    "emotion_beat": "quiet_hope",
                    "motion": "One hand on the doorframe. Not going anywhere yet. Just standing there, a little lighter.",
                },
            ],
        },
    },

    # ── ANGER → CALMING ─────────────────────────────────────────────────────
    {
        "emotion": EmotionOutput(
            primary_emotion="angry", confidence=0.79,
            secondary_emotions=[{"emotion":"shame","score":0.12},{"emotion":"hurt","score":0.09}],
            somatic_markers=True,
            user_text="i said something i regret and i can't take it back. i keep replaying it.",
        ),
        "panel_count": 6,
        "target": {
            "recurring_motif": "cold running water from a kitchen tap",
            "mood_journey": "From fire coiled in the chest toward the body finding the floor again.",
            "panels": [
                {
                    "panel": 1,
                    "visual": "Kitchen. The tap running full. Person standing with both hands gripping the counter edge, not under the water. Jaw locked. Chest tight as a fist.",
                    "dialogue": "...",
                    "emotion_beat": "contained_fire",
                    "motion": "Knuckles white on the counter edge. The running tap water ignored. Body rigid.",
                },
                {
                    "panel": 2,
                    "visual": "Same kitchen. The water still running. One hand has released the counter — hovering in the air. The sound of the water louder now, closer. Throat constricted.",
                    "dialogue": "Why did I say that.",
                    "emotion_beat": "fracture",
                    "motion": "The hovering hand makes a fist, then opens. Fist, open. Doesn't know what to do with it.",
                },
                {
                    "panel": 3,
                    "visual": "Hands under the cold running tap now. Water flowing over them. Close on the hands — red at the knuckles, slowly returning to normal colour. Jaw starting to unclench.",
                    "dialogue": "...",
                    "emotion_beat": "exhale",
                    "motion": "Turns wrists slowly under the water. Watches the water, not the room. Lets out a long breath.",
                },
                {
                    "panel": 4,
                    "visual": "Wide shot. The person leaning over the sink, water still running, forehead nearly touching the cupboard above. The running water visible, constant. Shoulders beginning to drop.",
                    "dialogue": "I can't undo it.",
                    "emotion_beat": "cooling",
                    "motion": "Forehead touches the cupboard. Holds that position. Lets the water keep running.",
                },
                {
                    "panel": 5,
                    "visual": "The tap turned off now. Silence except for the drip. Hands on the edge of the sink, wet, not drying them yet. Chest quieter. Stomach still uneasy but looser.",
                    "dialogue": "...",
                    "emotion_beat": "ground",
                    "motion": "Stands upright. Feet flat on the floor — noticing the floor. Breathes through the nose.",
                },
                {
                    "panel": 6,
                    "visual": "Person at the kitchen window, hands finally dried on a towel. The tap behind them, still. Outside, ordinary street. Chest open. Jaw loose. The water is gone, the heat is too.",
                    "dialogue": "Okay.",
                    "emotion_beat": "stillness",
                    "motion": "Folds the towel. Sets it down. Stands in the quiet kitchen. Not fixed. But still.",
                },
            ],
        },
    },

    # ── TIRED → RELAXING ────────────────────────────────────────────────────
    {
        "emotion": EmotionOutput(
            primary_emotion="tired", confidence=0.83,
            secondary_emotions=[{"emotion":"numbness","score":0.13},{"emotion":"longing","score":0.07}],
            somatic_markers=True,
            user_text="can't get out of bed today. not sad exactly. just... empty.",
        ),
        "panel_count": 6,
        "target": {
            "recurring_motif": "a folded blanket at the foot of the bed",
            "mood_journey": "From the drag of not-starting toward the body's genuine permission to rest.",
            "panels": [
                {
                    "panel": 1,
                    "visual": "Bedroom, mid-morning light. The folded blanket at the foot of the bed untouched. Person lying sideways, not asleep, staring at nothing. Limbs like something poured.",
                    "dialogue": "...",
                    "emotion_beat": "drag",
                    "motion": "Does not move. Blinks slowly. One hand near face, not doing anything. Stays.",
                },
                {
                    "panel": 2,
                    "visual": "Same position. The folded blanket in the background. Phone on the bed, screen dark. The person's eyes open but not looking at it. Chest slow, shallow.",
                    "dialogue": "I should get up.",
                    "emotion_beat": "drag",
                    "motion": "Does not get up. The thought passes through. Body stays where it is.",
                },
                {
                    "panel": 3,
                    "visual": "Person has pulled the folded blanket over themselves now. It's not neat. Light has shifted — slightly later. Eyes closed. Shoulders, for the first time, not braced.",
                    "dialogue": "...",
                    "emotion_beat": "surrender",
                    "motion": "Tucks the blanket loosely. Does not set an alarm. Stops trying to begin.",
                },
                {
                    "panel": 4,
                    "visual": "Under the blanket now, only the person's face visible. The fold of fabric at the chin. Room quieter. Afternoon light. Jaw unclenched. Skin warm.",
                    "dialogue": "...",
                    "emotion_beat": "softness",
                    "motion": "Shifts once to a more comfortable position. Settles. Breath longer.",
                },
                {
                    "panel": 5,
                    "visual": "Wide shot. The whole bed, blanket now fully around the person. Late afternoon light on the wall. Everything still. Even the folded blanket — its original shape forgotten, now used.",
                    "dialogue": "...",
                    "emotion_beat": "quiet_rest",
                    "motion": "Asleep. Or nearly. Chest rising and falling without effort.",
                },
                {
                    "panel": 6,
                    "visual": "Evening. The folded blanket half off the bed — dishevelled, lived in. Person sitting on the edge of the mattress, feet on the floor. Not energised. But present. Neck looser.",
                    "dialogue": "Okay.",
                    "emotion_beat": "renewal",
                    "motion": "Places both feet flat on the floor. Feels the floor. Sits for a moment before anything else.",
                },
            ],
        },
    },

    # ── HAPPY → ELATION ─────────────────────────────────────────────────────
    {
        "emotion": EmotionOutput(
            primary_emotion="happy", confidence=0.88,
            secondary_emotions=[{"emotion":"gratitude","score":0.14},{"emotion":"wonder","score":0.08}],
            somatic_markers=True,
            user_text="today was unexpectedly beautiful. i don't know how else to say it.",
        ),
        "panel_count": 6,
        "target": {
            "recurring_motif": "late afternoon light through a dusty window",
            "mood_journey": "From an ordinary moment catching fire toward something wordless and full.",
            "panels": [
                {
                    "panel": 1,
                    "visual": "A small room. The late afternoon light through the dusty window landing in a long stripe across the floor. Person seated, noticing. Chest with a lightness they haven't placed yet.",
                    "dialogue": "Oh.",
                    "emotion_beat": "spark",
                    "motion": "Leans forward slightly. Watches the light. Does not look away.",
                },
                {
                    "panel": 2,
                    "visual": "Same room. The late afternoon light through the dusty window now catching dust motes — visible, slow. The person has moved to sit in the light. Skin warm. Shoulders back.",
                    "dialogue": "...",
                    "emotion_beat": "expansion",
                    "motion": "Tilts face toward the light. Eyes half-closed. Hands open in the lap, palms up.",
                },
                {
                    "panel": 3,
                    "visual": "Close on the person's face in the late afternoon light. Eyes bright. The corners of the mouth — not smiling exactly, something more interior. Throat open. Breath easy.",
                    "dialogue": "I didn't expect this.",
                    "emotion_beat": "overflow",
                    "motion": "Reaches one hand out into the light stripe. Watches the dust move around the fingers.",
                },
                {
                    "panel": 4,
                    "visual": "The room now in golden light — late afternoon light through the dusty window filling everything. The person standing in the middle of it. Not doing anything. Just inside it.",
                    "dialogue": "...",
                    "emotion_beat": "radiance",
                    "motion": "Turns slowly, once, in the light. Arms slightly out. A full breath.",
                },
                {
                    "panel": 5,
                    "visual": "The light has shifted — almost gone now. The dusty window still visible. The person seated again, quiet. The room ordinary again. But something in the posture is different. Chest open.",
                    "dialogue": "...",
                    "emotion_beat": "luminous_still",
                    "motion": "Completely still. Hands folded. Looking at where the light was. In no hurry.",
                },
                {
                    "panel": 6,
                    "visual": "Wide. The room, the window, the fading late afternoon light through the dusty glass. The person a small figure in it. Full. The room the same. Something has passed through it.",
                    "dialogue": "...",
                    "emotion_beat": "transcendence",
                    "motion": "Does not move. Remains. The light keeps going. The person remains in the room it has left.",
                },
            ],
        },
    },

    # ── ANXIOUS → GROUNDING ──────────────────────────────────────────────────
    {
        "emotion": EmotionOutput(
            primary_emotion="anxious", confidence=0.81,
            secondary_emotions=[{"emotion":"dread","score":0.11},{"emotion":"restless","score":0.09}],
            somatic_markers=True,
            user_text="my mind just won't stop racing at night. i can't sleep. everything feels like a threat.",
        ),
        "panel_count": 6,
        "target": {
            "recurring_motif": "a single houseplant on the windowsill",
            "mood_journey": "From the spiral of a sleepless mind toward the body remembering the floor.",
            "panels": [
                {
                    "panel": 1,
                    "visual": "Bedroom, 2am. A single houseplant on the windowsill, silhouetted. Phone screen glowing. The person lying rigid, eyes open. Chest tight, breath fast and shallow.",
                    "dialogue": "...",
                    "emotion_beat": "spiral",
                    "motion": "Lies completely still — not restful still, locked still. Jaw clenched. Mind elsewhere.",
                },
                {
                    "panel": 2,
                    "visual": "Same room. The single houseplant on the windowsill unchanged. The person now sitting up, back against the headboard. Hands gripping knees. Throat tight.",
                    "dialogue": "Stop. Stop. Stop.",
                    "emotion_beat": "peak_noise",
                    "motion": "Grips knees harder. Rocks slightly. Eyes squeezed shut.",
                },
                {
                    "panel": 3,
                    "visual": "Close on the single houseplant on the windowsill. One leaf. Caught in a small draft. The person's face turned toward it now. Chest still fast, but attention has snagged on the leaf.",
                    "dialogue": "...",
                    "emotion_beat": "pause",
                    "motion": "Watches the leaf move. Stops gripping. Hands loosen in the lap.",
                },
                {
                    "panel": 4,
                    "visual": "The person now beside the windowsill. Hand near the single houseplant — not touching, hovering. Streetlamp outside. Breath audible. Slower than before. Shoulders beginning to drop.",
                    "dialogue": "It's just a plant.",
                    "emotion_beat": "breath",
                    "motion": "Breathes out slowly. Then in. Watching the plant. Nothing else.",
                },
                {
                    "panel": 5,
                    "visual": "Wide shot. Person seated on the floor below the windowsill. The single houseplant above and behind them. Back against the wall. Feet flat on the floor. Chest quieter.",
                    "dialogue": "...",
                    "emotion_beat": "root",
                    "motion": "Feels the floor with both palms. Presses down gently. Stays on the floor.",
                },
                {
                    "panel": 6,
                    "visual": "Still on the floor. The single houseplant on the windowsill above, the first grey of morning at the glass now. Person leaning against the wall, not asleep. Present. Jaw loose. Stomach softer.",
                    "dialogue": "...",
                    "emotion_beat": "present",
                    "motion": "Stays. Breathes. The floor is still there. The plant is still there. So is the person.",
                },
            ],
        },
    },

    # ── GRIEF → TENDER CONTINUANCE ───────────────────────────────────────────
    {
        "emotion": EmotionOutput(
            primary_emotion="grief", confidence=0.88,
            secondary_emotions=[{"emotion":"loneliness","score":0.10},{"emotion":"numbness","score":0.07}],
            somatic_markers=True,
            user_text="my dog passed away yesterday. the house is too quiet.",
        ),
        "panel_count": 6,
        "target": {
            "recurring_motif": "an orange dog bowl on the kitchen floor",
            "mood_journey": "From the shape of absence toward carrying what was loved into the continuing day.",
            "panels": [
                {
                    "panel": 1,
                    "visual": "Kitchen floor, morning. The orange dog bowl sits where it has always sat, half-full of water. No dog. The absence a presence. Chest hollow.",
                    "dialogue": "...",
                    "emotion_beat": "absence",
                    "motion": "Person walks past. Stops. Does not move the orange dog bowl. Cannot yet.",
                },
                {
                    "panel": 2,
                    "visual": "Living room. The leash still on the hook by the door. The orange dog bowl visible through the doorway behind. Throat tight. Eyes not wet yet, but close.",
                    "dialogue": "I should put that away.",
                    "emotion_beat": "ache",
                    "motion": "Reaches toward the leash. Hand stops. Does not take it. Returns to side.",
                },
                {
                    "panel": 3,
                    "visual": "Kitchen. The person seated on the floor beside the orange dog bowl. Not doing anything. Just beside it. Stomach tight, the particular weight of remembering.",
                    "dialogue": "You liked the cold water.",
                    "emotion_beat": "memory",
                    "motion": "Sits cross-legged beside the bowl. One hand resting near it, not touching. Stays.",
                },
                {
                    "panel": 4,
                    "visual": "Still on the floor. The orange dog bowl close. A friend's text glowing on the phone nearby: 'Thinking of you.' Chest cracked open but not collapsing. Held.",
                    "dialogue": "...",
                    "emotion_beat": "held",
                    "motion": "Does not reply yet. Holds the phone. Lets the message exist. Breathes.",
                },
                {
                    "panel": 5,
                    "visual": "Later. Person standing at the counter making tea. The orange dog bowl still on the floor behind them. Not moved. Life moving around it. Shoulders softer than this morning.",
                    "dialogue": "...",
                    "emotion_beat": "continuance",
                    "motion": "Makes tea. Moves carefully around the bowl. Does not pretend it isn't there.",
                },
                {
                    "panel": 6,
                    "visual": "Evening. Kitchen lamp on. The orange dog bowl on the floor, unchanged. Person at the table, mug in hand. Both things true in the same room. Breath steadier. Eyes still soft.",
                    "dialogue": "Still here.",
                    "emotion_beat": "carried_forward",
                    "motion": "Looks at the bowl. Looks away. Takes a sip. Stays in the room with both things.",
                },
            ],
        },
    },
]


# ---------------------------------------------------------------------------
# 7. Dataset Builder
# ---------------------------------------------------------------------------

def build_training_jsonl(
    examples: Optional[List[Dict[str, Any]]] = None,
    output_path: str = "moodweaver_stage2_train.jsonl",
) -> Path:
    """
    Build JSONL fine-tuning dataset in ChatML format.
    Compatible with: Unsloth, TRL SFTTrainer, Axolotl, LLaMA-Factory.
    """
    if examples is None:
        examples = TRAINING_EXAMPLES  # type: ignore

    records = []
    for ex in examples:
        assert isinstance(ex, dict)
        target = ex.get("target")
        assert isinstance(target, dict)
        panels = target.get("panels")
        assert isinstance(panels, list)
        n = int(ex.get("panel_count", len(panels)))
        emotion = ex.get("emotion")
        assert isinstance(emotion, EmotionOutput)
        records.append({
            "messages": [
                {"role": "system",    "content": build_system_prompt(n)},
                {"role": "user",      "content": build_user_prompt(emotion, n)},
                {"role": "assistant", "content": json.dumps(target, indent=2)},
            ]
        })

    path = Path(output_path)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info(f"Saved {len(records)} training examples → {path}")
    return path


# ---------------------------------------------------------------------------
# 8. Trainer (Unsloth / TRL)
# ---------------------------------------------------------------------------

def train_with_unsloth(
    model_key:    str   = "llama",
    dataset_path: str   = "moodweaver_stage2_train.jsonl",
    output_dir:   str   = "moodweaver_stage2_finetuned",
    epochs:       int   = 5,
    batch_size:   int   = 2,
    lr:           float = 2e-4,
    lora_r:       int   = 16,
    lora_alpha:   int   = 32,
):
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, SFTConfig
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install unsloth datasets trl")

    repo = MODEL_REGISTRY[model_key]["repo"]
    log.info(f"Loading base model for fine-tuning: {repo}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=repo, max_seq_length=2048, dtype=None, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.map(lambda ex: {
        "text": tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
    })

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        args=SFTConfig(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=20,
            learning_rate=lr,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            save_strategy="epoch",
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
        ),
    )
    log.info("Starting fine-tuning ...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info(f"Fine-tuned model saved → {output_dir}")


# ---------------------------------------------------------------------------
# 9. Pretty Print
# ---------------------------------------------------------------------------

PHASE_ICONS = {"validation":"🔵","complication":"🟠","shift":"🟡","openness":"🟢"}

def print_story(script: StoryScript, panel_count: int, emotion: str):
    arc    = MOOD_ARCS.get(emotion.lower(), DEFAULT_ARC)
    phases = TIMING_PHASES.get(panel_count, ["—"] * panel_count)
    print(f"\n{'═'*65}")
    print(f"  🎨 MOODWEAVER — {panel_count}-PANEL STORY")
    print(f"  Emotion  : {str(emotion).upper()}  →  {str(arc['journey']).upper()}")
    print(f"  Journey  : {script.mood_journey}")
    print(f"  Motif    : {script.recurring_motif}")
    print(f"  Time     : {script.generation_time_s}s  |  Model: {script.model_used}")
    print(f"{'═'*65}\n")
    for p in script.panels:
        i     = p.panel - 1
        phase = phases[i] if i < len(phases) else "—"
        icon  = PHASE_ICONS.get(phase, "⚪")
        print(f"┌─ Panel {p.panel:02d}  {icon} {phase:<14}  ⏱ {i*5}–{i*5+5}s  beat: {p.emotion_beat}")
        print(f"│  📷 VISUAL  : {p.visual}")
        print(f"│  💬 DIALOGUE: {p.dialogue}")
        print(f"│  🏃 MOTION  : {p.motion}")
        print(f"└{'─'*63}\n")


# ---------------------------------------------------------------------------
# 10. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MoodWeaver Stage 2 — Story Generation")
    parser.add_argument("--mode",    default="generate",
                        choices=["generate","train","dataset"],
                        help="generate | train | dataset")
    parser.add_argument("--model",   default=_env("MODEL_KEY","finetuned"),
                        choices=list(MODEL_REGISTRY))
    parser.add_argument("--device",  default="auto")
    parser.add_argument("--emotion", default=_env("EMOTION","sad"),
                        choices=list(MOOD_ARCS))
    parser.add_argument("--panels",  type=int,
                        default=int(_env("PANEL_COUNT","6")),
                        choices=range(4,11), metavar="[4-10]")
    parser.add_argument("--text",    default=_env("USER_TEXT","everything feels heavy lately"))
    parser.add_argument("--conf",    type=float, default=float(_env("EMOTION_CONFIDENCE","0.72")))
    parser.add_argument("--temp",    type=float, default=float(_env("TEMPERATURE","0.72")))
    parser.add_argument("--output",  default=_env("OUTPUT_FILE","story_output.json"))
    parser.add_argument("--epochs",  type=int,   default=5)
    args = parser.parse_args()

    # ── Dataset only ────────────────────────────────────────────────────────
    if args.mode == "dataset":
        build_training_jsonl()
        return

    # ── Train ───────────────────────────────────────────────────────────────
    if args.mode == "train":
        jsonl = build_training_jsonl()
        train_with_unsloth(
            model_key=args.model,
            dataset_path=str(jsonl),
            epochs=args.epochs,
        )
        return

    # ── Generate ────────────────────────────────────────────────────────────
    sec = SECONDARY_DEFAULTS.get(args.emotion, [{"emotion":"unnamed","score":0.10}])
    emotion_input = EmotionOutput(
        primary_emotion=args.emotion,
        confidence=args.conf,
        secondary_emotions=sec,
        somatic_markers=True,
        user_text=args.text,
    )

    gen    = StoryGenerator(model_key=args.model, device=args.device)
    script = gen.generate(
        emotion=emotion_input,
        panel_count=args.panels,
        temperature=args.temp,
        top_p=float(_env("TOP_P","0.92")),
        rep_penalty=float(_env("REPETITION_PENALTY","1.15")),
        max_retries=int(_env("RETRY_ATTEMPTS","5")),
    )

    print_story(script, args.panels, args.emotion)

    out = Path(args.output)
    payload = script.to_dict()
    payload["_meta"]["emotion"]     = args.emotion
    payload["_meta"]["panel_count"] = args.panels
    payload["_meta"]["arc"]         = MOOD_ARCS.get(args.emotion, DEFAULT_ARC)["journey"]

    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log.info(f"Saved → {out}")


if __name__ == "__main__":
    main()