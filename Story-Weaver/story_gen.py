"""
MoodWeaver — Stage 2 Dynamic: 4–10 Panel Story with Mood Arc
=============================================================
Config lives in .env — edit that, not this file.
Arc definitions are loaded from indie_comic_pipeline/config/arcs_config.json
when available, with a built-in inline dict as fallback.

Usage:
    python story_gen.py
"""

import json, re, time, logging, os
from pathlib import Path
from dataclasses import dataclass


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
                logging.getLogger("moodweaver.dynamic").info(
                    f"Loaded arc config from {p}"
                )
                return cfg
            except Exception as e:
                logging.getLogger("moodweaver.dynamic").warning(
                    f"Could not load arcs_config.json from {p}: {e}"
                )
    return {}

# --- load .env before anything else ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # manual fallback — no pip install needed
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            v = v.split("#")[0].strip()   # strip inline comments
            import os; os.environ.setdefault(k.strip(), v)

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("moodweaver.dynamic")

# ---------------------------------------------------------------------------
# Read config from env
# ---------------------------------------------------------------------------

def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()

MODEL_PATH          = env("MODEL_PATH",           "moodweaver_stage2_merged")
EMOTION             = env("EMOTION",              "sad").lower()
PANEL_COUNT         = max(4, min(10, int(env("PANEL_COUNT", "6"))))
USER_TEXT           = env("USER_TEXT",            "everything feels heavy lately")
CONFIDENCE          = float(env("EMOTION_CONFIDENCE", "0.72"))

# Optimal parameter maps from parameter search GP optimization
_OPT_MAP = {
    "sad":        {"t": 0.5,  "p": 0.95, "r": 1.1,  "m": 250},
    "happy":      {"t": 0.5,  "p": 0.95, "r": 1.1,  "m": 160},
    "angry":      {"t": 0.6,  "p": 0.90, "r": 1.0,  "m": 160},
    "anxious":    {"t": 1.0,  "p": 0.70, "r": 1.1,  "m": 100},
    "grief":      {"t": 0.95, "p": 0.70, "r": 1.05, "m": 180},
    "tired":      {"t": 0.55, "p": 0.88, "r": 1.2,  "m": 220},
    "determined": {"t": 0.80, "p": 0.85, "r": 1.1,  "m": 180},
    "love":       {"t": 0.70, "p": 0.90, "r": 1.15, "m": 200},
}
_opt = _OPT_MAP.get(EMOTION, {"t": 0.72, "p": 0.92, "r": 1.15, "m": 150})

TEMPERATURE         = float(env("TEMPERATURE",    str(_opt["t"])))
TOP_P               = float(env("TOP_P",          str(_opt["p"])))
REP_PENALTY         = float(env("REPETITION_PENALTY", str(_opt["r"])))
MAX_TPP             = int(env("MAX_TOKENS_PER_PANEL",  str(_opt["m"])))
RETRIES             = int(env("RETRY_ATTEMPTS",   "3"))
OUTPUT_FILE         = env("OUTPUT_FILE",          "story_dynamic.json")

log.info(f"Config loaded — emotion={EMOTION}, panels={PANEL_COUNT}, model={MODEL_PATH}")

# ---------------------------------------------------------------------------
# Mood Arc Definitions
# ---------------------------------------------------------------------------

# ── Arc definitions (8 arcs, mirroring indie_comic_pipeline) ──────────────
# Loaded from shared arcs_config.json when available; inline dict is fallback.

_ARCS_CONFIG = _load_arcs_config()

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


# Attempt config-driven build, fall through to inline definitions
_config_arcs = _build_mood_arcs_from_config(_ARCS_CONFIG) if _ARCS_CONFIG else {}

MOOD_ARCS = _config_arcs if _config_arcs else {
    "sad": {
        "journey":     "uplifting",
        "description": "From heaviness toward light — not forced positivity, but genuine small warmth",
        "arc_beats":   ["heaviness","stillness","faint_warmth","tentative_light","soft_openness","quiet_hope"],
        "motif_hint":  "something small that holds warmth (a cup, a candle, a patch of sunlight)",
        "end_note":    "End with something small but genuinely warm. Not fixed. But lighter.",
    },
    "angry": {
        "journey":     "calming",
        "description": "From fire toward stillness — the anger is valid, the body finds ground",
        "arc_beats":   ["contained_fire","fracture","exhale","cooling","ground","stillness"],
        "motif_hint":  "something that absorbs heat (running water, open window, cold surface)",
        "end_note":    "End with the body calm. Situation unresolved but person grounded.",
    },
    "tired": {
        "journey":     "relaxing",
        "description": "From exhaustion toward rest — permission to stop, body softening",
        "arc_beats":   ["drag","surrender","softness","drift","quiet_rest","renewal"],
        "motif_hint":  "something soft and horizontal (a pillow, blanket fold, evening light)",
        "end_note":    "End with genuine rest. Tomorrow is not in this panel.",
    },
    "happy": {
        "journey":     "elation",
        "description": "From joy toward transcendence — expanding, overflowing, luminous",
        "arc_beats":   ["spark","expansion","overflow","radiance","luminous_still","transcendence"],
        "motif_hint":  "something that multiplies light (reflections, laughter lines, open hands)",
        "end_note":    "End with pure presence. Joy has become larger than its cause.",
    },
    "anxious": {
        "journey":     "grounding",
        "description": "From spiral toward root — the mind slows, the body finds earth",
        "arc_beats":   ["spiral","peak_noise","pause","breath","root","present"],
        "motif_hint":  "something tactile and grounding (textured surface, bare feet, single object)",
        "end_note":    "End with presence. Not solved. But here, now, in this body.",
    },
    "grief": {
        "journey":     "tender continuance",
        "description": "From loss toward carrying — grief doesn't end, but the person continues",
        "arc_beats":   ["absence","ache","memory","held","continuance","carried_forward"],
        "motif_hint":  "something that was shared (a chair, a mug, a particular quality of light)",
        "end_note":    "End with both things true: the loss is real, and life continues.",
    },
    # ── Two arcs added to match indie_comic_pipeline ──────────────────────
    "determined": {
        "journey":     "heroic rise",
        "description": "From doubt toward resolute action — the cost of the climb is visible",
        "arc_beats":   ["doubt","challenge","resistance","breakthrough","momentum","triumph"],
        "motif_hint":  "something that holds the cost of the climb (scarred hands, broken weapon, worn path)",
        "end_note":    "End with victory earned, not given. The character is changed by the climb.",
    },
    "love": {
        "journey":     "deepening",
        "description": "From spark toward enduring warmth — love as transformation, not destination",
        "arc_beats":   ["spark","recognition","vulnerability","trust","embrace","unity"],
        "motif_hint":  "something shared between two people (a held hand, a shared window, intertwined roots)",
        "end_note":    "End with both people changed. Love as transformation, not destination.",
    },
}

DEFAULT_ARC = {
    "journey":     "reflective",
    "description": "From feeling toward understanding",
    "arc_beats":   ["acknowledgment","presence","shift","openness"],
    "motif_hint":  "something ordinary that carries weight",
    "end_note":    "End with openness.",
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


# Load secondary defaults from config when available, otherwise use inline dict.
_cfg_secondary = _build_secondary_defaults_from_config(_ARCS_CONFIG) if _ARCS_CONFIG else {}

SECONDARY_DEFAULTS: dict = _cfg_secondary if _cfg_secondary else {
    "sad":        [{"emotion":"exhaustion", "score":0.15},{"emotion":"longing",   "score":0.08}],
    "angry":      [{"emotion":"shame",      "score":0.12},{"emotion":"hurt",       "score":0.09}],
    "tired":      [{"emotion":"numbness",   "score":0.13},{"emotion":"longing",   "score":0.07}],
    "happy":      [{"emotion":"gratitude",  "score":0.14},{"emotion":"wonder",    "score":0.08}],
    "anxious":    [{"emotion":"dread",      "score":0.11},{"emotion":"restless",  "score":0.09}],
    "grief":      [{"emotion":"loneliness", "score":0.10},{"emotion":"numbness",  "score":0.07}],
    # New arcs
    "determined": [{"emotion":"fear",       "score":0.10},{"emotion":"resolve",   "score":0.12}],
    "love":       [{"emotion":"tenderness",  "score":0.13},{"emotion":"wonder",   "score":0.09}],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_beats(n: int, arc: dict) -> list:
    beats = arc["arc_beats"]
    if n <= len(beats):
        step = len(beats) / n
        return [beats[int(i * step)] for i in range(n)]
    result = []
    for i in range(n):
        idx = int(i * (len(beats) - 1) / (n - 1))
        result.append(beats[idx])
    return result


SYSTEM = """\
You are a literary graphic novelist writing for adults.
Respond ONLY with valid JSON. No markdown, no explanation, no preamble.

LITERARY CONSTRAINTS:
- NEVER name emotions directly. Show through action, objects, sensation.
- ONE recurring visual motif must appear in every single panel.
- Every panel MUST include a physical body sensation.
- No moral lessons.
- Dialogue must be emotionally expressive, realistic, context-aware, and natural. Avoid empty dialogue or '...' unless absolutely necessary for dramatic silence.

Output ONLY this JSON:
{
  "recurring_motif": "one precise visual motif present in every panel",
  "mood_journey": "one sentence describing the emotional arc",
  "panels": [
    {"panel": 1, "visual": "...", "dialogue": "...", "emotion_beat": "one_word", "motion": "..."}
  ]
}"""


def build_prompt(n: int) -> str:
    arc    = MOOD_ARCS.get(EMOTION, DEFAULT_ARC)
    beats  = get_beats(n, arc)
    phases = TIMING_PHASES.get(n, TIMING_PHASES[6])
    sec    = SECONDARY_DEFAULTS.get(EMOTION, [{"emotion":"unnamed","score":0.10}])
    sec_str = ", ".join(f"{e['emotion']} {e['score']}" for e in sec)

    beat_guide = "\n".join(
        f"  Panel {i+1} [{phases[i]}] → beat: {beats[i]}"
        for i in range(n)
    )
    return (
        f"Primary emotion: {EMOTION} (confidence {CONFIDENCE})\n"
        f"Secondary emotions: {sec_str}\n"
        f'User context: "{USER_TEXT}"\n\n'
        f"MOOD JOURNEY: {arc['journey']} — {arc['description']}\n"
        f"MOTIF HINT: {arc['motif_hint']}\n\n"
        f"Write exactly {n} panels. Emotional arc:\n{beat_guide}\n\n"
        f"Arc direction: {beats[0]} → {beats[-1]}\n"
        f"{arc['end_note']}\n\n"
        "Write the JSON now."
    )

# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class DynamicStoryGenerator:
    def __init__(self):
        log.info(f"Loading {MODEL_PATH} ...")
        # Auto-detect fallback to Ollama if the local trained model path does not exist on disk
        self.is_ollama = (
            MODEL_PATH.startswith("ollama") or 
            MODEL_PATH in ["llama3.2", "qwen2.5", "mistral", "llama3.1"] or
            not os.path.exists(MODEL_PATH)
        )
        if self.is_ollama:
            self.tok = None
            self.model = None
            log.info(f"Ollama backend enabled using model '{MODEL_PATH}' (fallback={not os.path.exists(MODEL_PATH)})")
        else:
            tok = AutoTokenizer.from_pretrained(MODEL_PATH)
            assert tok is not None
            if getattr(tok, "pad_token", None) is None:
                tok.pad_token = tok.eos_token
            self.tok = tok
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
            )
            self.model.eval()
            log.info("Model ready.")

    def generate(self) -> dict:
        system_prompt = SYSTEM
        # If schema override is requested, swap the target output spec in the system prompt
        if os.environ.get("COMIC_SCHEMA_OVERRIDE") == "true":
            schema_spec = """Output ONLY this JSON:
{
  "story_bible": {
    "plot_summary": "2-3 sentence story summary",
    "side_characters": [ {"name": "...", "role": "...", "description": "..."} ]
  },
  "recurring_motif": "one precise visual motif present in every panel",
  "mood_journey": "one sentence describing the emotional arc",
  "panels": [
    {
      "panel": 1,
      "emotion_beat": "beat_name",
      "characters": [
        {
          "id": "character_name_lowercase",
          "pose": {"body": "clothing + physical stance", "head": "head direction", "arms": "arm position", "legs": "leg position"},
          "expression": {"emotion": "specific emotion", "eyes": "eye description", "mouth": "mouth position"},
          "dialogue": {"text": "Actual spoken words.", "tone": "tone descriptor", "bubble": "speech|thought|shout|whisper"}
        }
      ],
      "actions": [ {"actor": "character_id", "verb": "action verb", "target": "what/whom"} ],
      "camera": "angle + movement descriptor",
      "environment": "location, time, dominant palette, light source"
    }
  ]
}"""
            system_prompt = SYSTEM.replace(
                'Output ONLY this JSON:\n{\n  "recurring_motif": "one precise visual motif present in every panel",\n  "mood_journey": "one sentence describing the emotional arc",\n  "panels": [\n    {"panel": 1, "visual": "...", "dialogue": "...", "emotion_beat": "one_word", "motion": "..."}\n  ]\n}',
                schema_spec
            )
            
        # Prepend character/style constraints if loaded from environment
        style_str = os.environ.get("STYLE_STR", "")
        char_str = os.environ.get("CHAR_STR", "")
        story_ref_str = os.environ.get("STORY_REF_STR", "")
        if style_str or char_str or story_ref_str:
            system_prompt = f"{style_str}{char_str}{story_ref_str}\n\n{system_prompt}"

        if self.is_ollama:
            import httpx
            url = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/chat"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": build_prompt(PANEL_COUNT)},
            ]
            for attempt in range(1, RETRIES + 1):
                log.info(f"Attempt {attempt}/{RETRIES} via Ollama ({MODEL_PATH})")
                t0 = time.time()
                try:
                    payload = {
                        "model": MODEL_PATH,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": TEMPERATURE,
                            "top_p": TOP_P,
                            "num_predict": MAX_TPP * PANEL_COUNT
                        }
                    }
                    r = httpx.post(url, json=payload, timeout=60.0)
                    r.raise_for_status()
                    resp = r.json()
                    raw = resp["message"]["content"].strip()
                    elapsed = round(time.time() - t0, 2)
                    
                    data = self._parse(raw)
                    self._validate(data)
                    data["_meta"] = {
                        "emotion": EMOTION, "panel_count": PANEL_COUNT,
                        "arc": MOOD_ARCS.get(EMOTION, DEFAULT_ARC)["journey"],
                        "generation_time_s": elapsed,
                        "config": {
                            "temperature": TEMPERATURE, "top_p": TOP_P,
                            "rep_penalty": REP_PENALTY, "max_tpp": MAX_TPP,
                        }
                    }
                    log.info(f"Done in {elapsed}s")
                    return data
                except Exception as e:
                    log.warning(f"Ollama attempt {attempt} failed: {e}")
                    if attempt == RETRIES:
                        raise RuntimeError(f"Ollama failed after {RETRIES} attempts: {e}")
            raise RuntimeError("Unexpected end of Ollama generation loop")

        tok = self.tok
        model = self.model
        assert tok is not None and model is not None
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": build_prompt(PANEL_COUNT)},
        ]
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        ids = tok(
            prompt,
            return_tensors="pt",
        ).to(model.device)

        for attempt in range(1, RETRIES + 1):
            log.info(f"Attempt {attempt}/{RETRIES}")
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **ids,
                    max_new_tokens=MAX_TPP * PANEL_COUNT,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    repetition_penalty=REP_PENALTY,
                    pad_token_id=tok.eos_token_id,
                )
            prompt_len = ids["input_ids"].shape[1]

            raw = tok.decode(
                out[0][prompt_len:],
                skip_special_tokens=True
            ).strip()
            
            elapsed = round(time.time() - t0, 2)
            try:
                data = self._parse(raw)
                self._validate(data)
                data["_meta"] = {
                    "emotion": EMOTION, "panel_count": PANEL_COUNT,
                    "arc": MOOD_ARCS.get(EMOTION, DEFAULT_ARC)["journey"],
                    "generation_time_s": elapsed,
                    "config": {
                        "temperature": TEMPERATURE, "top_p": TOP_P,
                        "rep_penalty": REP_PENALTY, "max_tpp": MAX_TPP,
                    }
                }
                log.info(f"Done in {elapsed}s")
                return data
            except Exception as e:
                log.warning(f"Attempt {attempt} failed: {e}")
                if attempt == RETRIES:
                    raise RuntimeError(f"Failed after {RETRIES} attempts: {e}\n\nRaw:\n{raw}")
        raise RuntimeError("Unexpected end of generation loop")

    def _parse(self, raw: str) -> dict:
        clean = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        clean = re.sub(r"\s*```$", "", clean).strip()
        s = clean.find("{")
        if s == -1: raise ValueError("No JSON found")
        depth, end = 0, -1
        for i, ch in enumerate(clean[s:], s):
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0: end = i; break
        if end == -1: raise ValueError("Unbalanced braces")
        
        # Simple JSON parser error repair (e.g. trailing commas before end of arrays/objects)
        try:
            return json.loads(clean[s:end+1])
        except Exception:
            # Attempt to repair common trailing commas
            repaired = re.sub(r',\s*([\]}])', r'\1', clean[s:end+1])
            return json.loads(repaired)

    def _validate(self, data: dict):
        if len(data.get("panels", [])) != PANEL_COUNT:
            raise ValueError(f"Expected {PANEL_COUNT} panels, got {len(data.get('panels',[]))}")
        # Bypass layout validation validation keys if we are using the dynamic schema override
        if os.environ.get("COMIC_SCHEMA_OVERRIDE") == "true":
            for p in data["panels"]:
                for k in ("panel", "characters", "camera", "environment"):
                    if k not in p: raise ValueError(f"Panel {p.get('panel')} missing '{k}'")
            return
            
        for p in data["panels"]:
            for k in ("visual", "dialogue", "emotion_beat", "motion"):
                if k not in p: raise ValueError(f"Panel {p.get('panel')} missing '{k}'")

# ---------------------------------------------------------------------------
# Pretty Print
# ---------------------------------------------------------------------------

ICONS = {"validation":"🔵","complication":"🟠","shift":"🟡","openness":"🟢"}

def print_story(data: dict):
    phases = TIMING_PHASES.get(PANEL_COUNT, ["—"] * PANEL_COUNT)
    print(f"\n{'═'*62}")
    print(f"  🎨 MOODWEAVER — {PANEL_COUNT}-PANEL STORY")
    print(f"  Emotion : {EMOTION.upper()}  →  {data['_meta']['arc'].upper()}")
    print(f"  Journey : {data.get('mood_journey','—')}")
    print(f"  Motif   : {data.get('recurring_motif','—')}")
    print(f"  Time    : {data['_meta']['generation_time_s']}s")
    print(f"{'═'*62}\n")
    
    # Bypass details printing for customized layout schema override
    if os.environ.get("COMIC_SCHEMA_OVERRIDE") == "true":
        print("Detailed Comic layout generated successfully.")
        return
        
    for p in data["panels"]:
        i = p["panel"] - 1
        phase = phases[i] if i < len(phases) else "—"
        print(f"┌─ Panel {p['panel']:02d}  {ICONS.get(phase,'⚪')} {phase:<14}  ⏱ {i*5}–{i*5+5}s  beat: {p['emotion_beat']}")
        print(f"│  📷 {p['visual']}")
        print(f"│  💬 {p['dialogue']}")
        print(f"│  🏃 {p['motion']}")
        print(f"└{'─'*60}\n")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info(f"Running with: EMOTION={EMOTION}, PANELS={PANEL_COUNT}, TEMP={TEMPERATURE}")
    gen  = DynamicStoryGenerator()
    data = gen.generate()
    print_story(data)
    out  = Path(OUTPUT_FILE)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info(f"Saved → {out}")