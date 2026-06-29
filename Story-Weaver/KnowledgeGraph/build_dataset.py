"""
MoodWeaver — Dataset Pipeline
==============================
Downloads and processes 3 datasets into MoodWeaver training JSONL:

  GoEmotions        → primary + secondary emotions, user context
  Story Commonsense → mood journey, emotional beats, arc direction
  ROCStories        → story panels, narrative flow

Pipeline:
  1. Download all 3 datasets
  2. Process + align schemas
  3. Cross-join into combined records
  4. LLM annotation (Gemini) to generate literary panel scripts
  5. Save final JSONL for fine-tuning

Usage:
    python build_dataset.py                        # full pipeline
    python build_dataset.py --step download        # step 1 only
    python build_dataset.py --step process         # step 2 only
    python build_dataset.py --step annotate        # step 3: LLM annotation
    python build_dataset.py --step export          # step 4: save JSONL
    python build_dataset.py --limit 500            # limit to N annotated examples
    python build_dataset.py --resume               # skip already annotated

pip install datasets pandas google-generativeai python-dotenv tqdm
"""

import json, re, os, time, argparse, logging, random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("moodweaver.dataset")

# --- .env ---
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

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DATA_DIR       = Path(os.environ.get("DATA_DIR", "moodweaver_data"))
OUTPUT_JSONL   = os.environ.get("OUTPUT_JSONL", "moodweaver_stage2_train.jsonl")

# ---------------------------------------------------------------------------
# 1. Schema Definitions
# ---------------------------------------------------------------------------

# GoEmotions 27 labels → our 6 mood arc emotions
# Multiple GoEmotions labels collapse into one MoodWeaver primary emotion
GOEMOTIONS_MAP = {
    # → sad
    "sadness":      "sad",
    "grief":        "sad",
    "remorse":      "sad",
    "disappointment":"sad",
    "caring":       "sad",     # can signal yearning/longing

    # → angry
    "anger":        "angry",
    "annoyance":    "angry",
    "disgust":      "angry",
    "disapproval":  "angry",

    # → tired / depleted
    "confusion":    "tired",
    "nervousness":  "tired",
    "embarrassment":"tired",

    # → happy
    "joy":          "happy",
    "amusement":    "happy",
    "excitement":   "happy",
    "gratitude":    "happy",
    "love":         "happy",
    "admiration":   "happy",
    "pride":        "happy",
    "relief":       "happy",
    "optimism":     "happy",

    # → anxious
    "fear":         "anxious",
    "nervousness":  "anxious",
    "surprise":     "anxious",

    # → grief
    "grief":        "grief",
    "remorse":      "grief",
}

# All 27 GoEmotions labels (for secondary emotion mapping)
ALL_GOEMOTIONS = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization",
    "relief","remorse","sadness","surprise","neutral",
]

MOOD_ARCS = {
    "sad":     {"journey":"uplifting","beats":["heaviness","weight","stillness","faint_warmth","soft_openness","quiet_hope"]},
    "angry":   {"journey":"calming",  "beats":["contained_fire","fracture","peak","exhale","cooling","ground"]},
    "tired":   {"journey":"relaxing", "beats":["drag","push","surrender","softness","drift","renewal"]},
    "happy":   {"journey":"elation",  "beats":["spark","warmth","expansion","overflow","radiance","transcendence"]},
    "anxious": {"journey":"grounding","beats":["spiral","peak_noise","one_thing","breath","root","present"]},
    "grief":   {"journey":"tender continuance","beats":["absence","ache","memory","held","both_true","carried_forward"]},
}

MOTIF_HINTS = {
    "sad":     "a small warm object: a chipped mug, a candle, a patch of sunlight on a floor",
    "angry":   "something that absorbs heat: running water, open window, cold tiled floor",
    "tired":   "something soft and horizontal: a blanket, evening light on bare floor, a pillow",
    "happy":   "something that multiplies light: water surface, open hands, a window full of sky",
    "anxious": "something tactile and fixed: rough wall, bare feet on cool floor, a held object",
    "grief":   "an object that was shared: a chair, a mug, a specific quality of light",
}

@dataclass
class RawRecord:
    """Intermediate unified record before LLM annotation."""
    source: str              # goemotions | storycommonsense | rocstories | combined
    primary_emotion: str     # sad | angry | tired | happy | anxious | grief
    confidence: float
    secondary_emotions: list # [{"emotion": str, "score": float}]
    user_text: str           # original text / context
    story_context: str = ""  # from ROCStories or StoryCommonsense
    motivation: str = ""     # from StoryCommonsense
    arc_hint: str = ""       # mood journey hint
    somatic_markers: bool = True

@dataclass
class AnnotatedRecord:
    """Final record with LLM-generated panel script."""
    raw: RawRecord
    script: dict             # {recurring_motif, mood_journey, panels: [...]}

# ---------------------------------------------------------------------------
# 2. Dataset Downloaders
# ---------------------------------------------------------------------------

class GoEmotionsLoader:
    """
    GoEmotions: 58k Reddit comments, 27 emotion labels (multi-label).
    HuggingFace: google-research-datasets/go_emotions
    """
    DATASET_NAME = "google-research-datasets/go_emotions"
    SPLIT        = "train"

    def load(self, limit: int = 5000) -> list[RawRecord]:
        log.info(f"Loading GoEmotions (limit={limit}) ...")
        try:
            from datasets import load_dataset
            ds = load_dataset(self.DATASET_NAME, "simplified", split=self.SPLIT,
                              trust_remote_code=True)
        except Exception as e:
            log.error(f"GoEmotions load failed: {e}")
            log.info("Trying raw split ...")
            from datasets import load_dataset
            ds = load_dataset(self.DATASET_NAME, split=self.SPLIT,
                              trust_remote_code=True)

        records = []
        for item in ds:
            if len(records) >= limit: break

            # Get label indices and map to emotion names
            label_ids = item.get("labels", [])
            if not label_ids: continue

            label_names = [ALL_GOEMOTIONS[i] for i in label_ids
                           if i < len(ALL_GOEMOTIONS)]

            # Map to primary MW emotion
            primary_mw = None
            primary_score = 0.0
            for lname in label_names:
                mw = GOEMOTIONS_MAP.get(lname)
                if mw:
                    # use first strong hit
                    primary_mw = mw
                    primary_score = round(0.6 + random.random() * 0.35, 2)
                    break

            if not primary_mw: continue
            if primary_mw not in MOOD_ARCS: continue

            # Secondary emotions from remaining labels
            secondary = []
            for lname in label_names:
                if GOEMOTIONS_MAP.get(lname) != primary_mw:
                    secondary.append({
                        "emotion": lname,
                        "score": round(0.05 + random.random() * 0.15, 2)
                    })
            secondary = secondary[:3]

            text = item.get("text", "").strip()
            if len(text) < 15 or len(text) > 300: continue

            records.append(RawRecord(
                source="goemotions",
                primary_emotion=primary_mw,
                confidence=primary_score,
                secondary_emotions=secondary,
                user_text=text,
                arc_hint=MOTIF_HINTS.get(primary_mw, ""),
            ))

        log.info(f"GoEmotions: {len(records)} records loaded")
        return records


class StoryCommonsenseLoader:
    """
    Story Commonsense: story events + character motivation + emotional reaction.
    HuggingFace: Yoark/story_commonsense or local CSV fallback.
    """
    DATASET_NAME = "Yoark/story_commonsense"

    def load(self, limit: int = 3000) -> list[RawRecord]:
        log.info(f"Loading StoryCommonsense (limit={limit}) ...")
        try:
            from datasets import load_dataset
            ds = load_dataset(self.DATASET_NAME, split="train",
                              trust_remote_code=True)
            return self._process_hf(ds, limit)
        except Exception as e:
            log.warning(f"HF load failed: {e}. Trying CSV fallback ...")
            return self._csv_fallback(limit)

    def _process_hf(self, ds, limit: int) -> list[RawRecord]:
        records = []
        for item in ds:
            if len(records) >= limit: break
            try:
                record = self._convert(item)
                if record: records.append(record)
            except Exception:
                continue
        log.info(f"StoryCommonsense: {len(records)} records loaded")
        return records

    def _convert(self, item: dict) -> Optional[RawRecord]:
        # StoryCommonsense fields vary by version; handle both
        story    = item.get("story","") or item.get("sentence","") or ""
        motivat  = item.get("motivation","") or item.get("char_motivation","") or ""
        emotion  = item.get("emotion","") or item.get("char_emotion","") or ""
        reaction = item.get("reaction","") or item.get("char_reaction","") or ""

        if not story or not emotion: return None

        # Map emotion string to MW primary
        emotion_lower = emotion.lower()
        primary_mw = None
        for ge_label, mw_label in GOEMOTIONS_MAP.items():
            if ge_label in emotion_lower:
                primary_mw = mw_label
                break
        if not primary_mw:
            # Simple keyword fallback
            if any(w in emotion_lower for w in ["sad","cry","upset","hurt","loss"]):
                primary_mw = "sad"
            elif any(w in emotion_lower for w in ["angry","anger","furious","mad"]):
                primary_mw = "angry"
            elif any(w in emotion_lower for w in ["tired","exhaust","drain"]):
                primary_mw = "tired"
            elif any(w in emotion_lower for w in ["happy","joy","excit","glad"]):
                primary_mw = "happy"
            elif any(w in emotion_lower for w in ["anxious","worried","scared","fear"]):
                primary_mw = "anxious"
            elif any(w in emotion_lower for w in ["grief","mourn","bereave","lost"]):
                primary_mw = "grief"

        if not primary_mw: return None

        return RawRecord(
            source="storycommonsense",
            primary_emotion=primary_mw,
            confidence=round(0.65 + random.random() * 0.25, 2),
            secondary_emotions=[{"emotion": emotion, "score": round(0.1 + random.random()*0.1, 2)}],
            user_text=story[:250],
            story_context=f"{story} {reaction}"[:400],
            motivation=motivat[:200],
            arc_hint=MOTIF_HINTS.get(primary_mw, ""),
        )

    def _csv_fallback(self, limit: int) -> list[RawRecord]:
        """Manual download fallback."""
        csv_path = DATA_DIR / "storycommonsense.csv"
        if not csv_path.exists():
            log.warning(
                "StoryCommonsense CSV not found. Download manually:\n"
                "  https://usc-ict.github.io/storycommonsense/\n"
                f"  Save as: {csv_path}"
            )
            return []
        import csv
        records = []
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if len(records) >= limit: break
                try:
                    r = self._convert(row)
                    if r: records.append(r)
                except Exception:
                    continue
        log.info(f"StoryCommonsense CSV: {len(records)} records")
        return records


class ROCStoriesLoader:
    """
    ROCStories: ~50k 5-sentence crowd-sourced stories.
    HuggingFace: Ximing/ROCStories or inseq/roc_stories
    """
    DATASETS = [
        "Ximing/ROCStories",
        "inseq/roc_stories",
    ]

    def load(self, limit: int = 5000) -> list[RawRecord]:
        log.info(f"Loading ROCStories (limit={limit}) ...")
        for ds_name in self.DATASETS:
            try:
                from datasets import load_dataset
                ds = load_dataset(ds_name, split="train", trust_remote_code=True)
                records = self._process(ds, limit)
                if records:
                    log.info(f"ROCStories ({ds_name}): {len(records)} records")
                    return records
            except Exception as e:
                log.warning(f"{ds_name} failed: {e}")
        log.warning("ROCStories: all sources failed. Skipping.")
        return []

    def _process(self, ds, limit: int) -> list[RawRecord]:
        records = []
        for item in ds:
            if len(records) >= limit: break
            try:
                # ROCStories: sentence1–5 or story field
                if "story" in item:
                    story = item["story"]
                else:
                    sents = [item.get(f"sentence{i}","") for i in range(1,6)]
                    story = " ".join(s for s in sents if s)

                if len(story) < 40: continue

                # Infer emotion from story text (keyword heuristic)
                primary_mw = self._infer_emotion(story)
                if not primary_mw: continue

                # Use first sentence as user_text (personal voice)
                first_sent = story.split(".")[0].strip()
                if len(first_sent) < 10: continue

                records.append(RawRecord(
                    source="rocstories",
                    primary_emotion=primary_mw,
                    confidence=round(0.55 + random.random() * 0.30, 2),
                    secondary_emotions=[],
                    user_text=first_sent,
                    story_context=story[:400],
                    arc_hint=MOTIF_HINTS.get(primary_mw, ""),
                ))
            except Exception:
                continue
        return records

    def _infer_emotion(self, text: str) -> Optional[str]:
        t = text.lower()
        scores = defaultdict(int)
        SAD_W     = ["sad","cry","cried","upset","disappoint","lonely","miss","lost","empty","heavy","hollow"]
        ANGRY_W   = ["angry","anger","furious","mad","annoyed","frustrated","rage","yell","scream"]
        TIRED_W   = ["tired","exhaust","drain","weary","numb","burnout","collapse","can't go on"]
        HAPPY_W   = ["happy","joy","laugh","excit","smile","wonderful","great","fantastic","love","celebrat"]
        ANXIOUS_W = ["anxious","worried","scared","fear","nervous","panic","dread","overwhelm","spiral"]
        GRIEF_W   = ["grief","mourn","bereave","died","death","passed away","gone","miss them","funeral"]

        for w in SAD_W:     scores["sad"]     += t.count(w)
        for w in ANGRY_W:   scores["angry"]   += t.count(w)
        for w in TIRED_W:   scores["tired"]   += t.count(w)
        for w in HAPPY_W:   scores["happy"]   += t.count(w)
        for w in ANXIOUS_W: scores["anxious"] += t.count(w)
        for w in GRIEF_W:   scores["grief"]   += t.count(w)

        if not scores or max(scores.values()) == 0:
            return None
        return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# 3. Record Combiner
# ---------------------------------------------------------------------------

class RecordCombiner:
    """
    Merges GoEmotions + StoryCommonsense + ROCStories records.
    Strategy:
      - GoEmotions provides emotion labels + user text
      - StoryCommonsense adds motivation and arc context
      - ROCStories adds story structure context
      - Matched by emotion category, randomly sampled
    """

    def combine(self,
                goe_records: list,
                sc_records: list,
                roc_records: list,
                limit: int = 1000) -> list[RawRecord]:

        log.info("Combining datasets ...")

        # Group by emotion
        goe_by_emotion = defaultdict(list)
        sc_by_emotion  = defaultdict(list)
        roc_by_emotion = defaultdict(list)

        for r in goe_records: goe_by_emotion[r.primary_emotion].append(r)
        for r in sc_records:  sc_by_emotion[r.primary_emotion].append(r)
        for r in roc_records: roc_by_emotion[r.primary_emotion].append(r)

        combined = []
        per_emotion = limit // len(MOOD_ARCS)

        for emotion in MOOD_ARCS:
            goe = goe_by_emotion.get(emotion, [])
            sc  = sc_by_emotion.get(emotion, [])
            roc = roc_by_emotion.get(emotion, [])

            log.info(f"  {emotion}: GoE={len(goe)} SC={len(sc)} ROC={len(roc)}")

            # Shuffle all
            random.shuffle(goe); random.shuffle(sc); random.shuffle(roc)

            for i in range(min(per_emotion, max(len(goe), len(sc), len(roc)))):
                # Pick base record — prefer GoEmotions for user text quality
                if goe:
                    base = goe[i % len(goe)]
                elif sc:
                    base = sc[i % len(sc)]
                else:
                    base = roc[i % len(roc)]
                    
                # Enrich with StoryCommonsense context if available
                if sc:
                    sc_rec = sc[i % len(sc)]
                    base.story_context = sc_rec.story_context or base.story_context
                    base.motivation    = sc_rec.motivation or base.motivation

                # Enrich with ROCStories structure if available
                if roc and not base.story_context:
                    roc_rec = roc[i % len(roc)]
                    base.story_context = roc_rec.story_context

                base.source = "combined"
                combined.append(base)

        random.shuffle(combined)
        log.info(f"Combined: {len(combined)} records total")
        return combined[:limit]


# ---------------------------------------------------------------------------
# 4. LLM Annotator (Gemini)
# ---------------------------------------------------------------------------

ANNOTATION_SYSTEM = """\
You are a literary graphic novelist creating training data for an AI story generator.
Given an emotional context, generate a 4-panel comic script in JSON.

MANDATORY RULES:
1. Every panel.visual MUST contain the recurring motif word explicitly.
2. Every panel.dialogue MUST be filled — use "..." for silence, never empty string.
3. Every panel.motion MUST be filled — describe physical action, never empty.
4. Every panel MUST contain a body sensation word (chest, breath, hands, throat, stomach, spine, jaw, shoulders, eyes, skin, pulse, exhale, inhale, heavy, warm, cold, numb, ache).
5. emotion_beat = exactly one atmospheric word, NOT an emotion name.
6. NEVER use: sad, sadness, happy, happiness, angry, anger, tired, anxious, grief, fear, lonely, joy.
7. No moral lessons.

Output ONLY this JSON, nothing else:
{
  "recurring_motif": "a specific 3-6 word description of ONE visual object",
  "mood_journey": "one sentence from start beat to end beat",
  "panels": [
    {"panel": 1, "visual": "...", "dialogue": "... or text", "emotion_beat": "one_word", "motion": "..."},
    {"panel": 2, "visual": "...", "dialogue": "... or text", "emotion_beat": "one_word", "motion": "..."},
    {"panel": 3, "visual": "...", "dialogue": "... or text", "emotion_beat": "one_word", "motion": "..."},
    {"panel": 4, "visual": "...", "dialogue": "... or text", "emotion_beat": "one_word", "motion": "..."}
  ]
}"""


def build_annotation_prompt(record: RawRecord) -> str:
    arc = MOOD_ARCS.get(record.primary_emotion, MOOD_ARCS["sad"])
    beats_4 = [arc["beats"][int(i * len(arc["beats"]) / 4)] for i in range(4)]
    sec = ", ".join(f"{e['emotion']} {e['score']}" for e in record.secondary_emotions[:2])
    
    context_section = ""
    if record.story_context:
        context_section = f"\nStory context: {record.story_context[:300]}"
    if record.motivation:
        context_section += f"\nCharacter motivation: {record.motivation[:150]}"

    return (
        f"Primary emotion: {record.primary_emotion} (confidence {record.confidence})\n"
        f"Secondary: {sec or 'none'}\n"
        f'User context: "{record.user_text}"{context_section}\n\n'
        f"MOOD JOURNEY: {arc['journey']}\n"
        f"MOTIF HINT: {record.arc_hint}\n\n"
        f"Panel arc beats:\n"
        f"  Panel 1 [validation]   → beat: {beats_4[0]}\n"
        f"  Panel 2 [validation]   → beat: {beats_4[1]}\n"
        f"  Panel 3 [complication] → beat: {beats_4[2]}\n"
        f"  Panel 4 [openness]     → beat: {beats_4[3]}\n\n"
        f"IMPORTANT: dialogue and motion must never be empty strings.\n"
        f"Write the JSON now."
    )


class GeminiAnnotator:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("Set GEMINI_API_KEY in .env")
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel(
                "gemini-1.5-flash",
                system_instruction=ANNOTATION_SYSTEM,
            )
        except ImportError:
            raise ImportError("pip install google-generativeai")
        self.call_count = 0
        self.fail_count = 0

    def annotate(self, record: RawRecord) -> Optional[dict]:
        prompt = build_annotation_prompt(record)
        for attempt in range(3):
            try:
                resp = self.model.generate_content(prompt)
                raw  = resp.text.strip()
                raw  = re.sub(r"^```(?:json)?\s*", "", raw)
                raw  = re.sub(r"\s*```$", "", raw).strip()
                data = json.loads(raw)
                self._validate(data)
                self.call_count += 1
                return data
            except Exception as e:
                if attempt == 2:
                    log.warning(f"Annotation failed after 3 attempts: {e}")
                    self.fail_count += 1
                    return None
                time.sleep(1.5 ** attempt)
        return None

    def _validate(self, data: dict):
        if "panels" not in data or len(data["panels"]) != 4:
            raise ValueError(f"Need 4 panels, got {len(data.get('panels',[]))}")
        for p in data["panels"]:
            for k in ("visual","dialogue","emotion_beat","motion"):
                if k not in p: raise ValueError(f"Missing {k}")
                if not str(p[k]).strip(): raise ValueError(f"Empty {k}")


# ---------------------------------------------------------------------------
# 5. JSONL Exporter
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_FOR_TRAINING = """\
You are a literary graphic novelist writing for adults.
Respond ONLY with valid JSON. No markdown fences, no explanation, no preamble.

MANDATORY RULES:
1. Every panel.visual MUST contain the recurring motif word explicitly.
2. Every panel.dialogue MUST be filled — use "..." for silence, never empty.
3. Every panel.motion MUST be filled — describe physical action, never empty.
4. Every panel MUST contain a body sensation word.
5. emotion_beat = exactly one word, atmospheric, NOT an emotion name.
6. NEVER use emotion names directly.
7. No moral lessons.

Output ONLY this JSON:
{
  "recurring_motif": "specific 3-6 word description",
  "mood_journey": "one sentence arc",
  "panels": [
    {"panel": 1, "visual": "...", "dialogue": "...", "emotion_beat": "one_word", "motion": "..."},
    {"panel": 2, "visual": "...", "dialogue": "...", "emotion_beat": "one_word", "motion": "..."},
    {"panel": 3, "visual": "...", "dialogue": "...", "emotion_beat": "one_word", "motion": "..."},
    {"panel": 4, "visual": "...", "dialogue": "...", "emotion_beat": "one_word", "motion": "..."}
  ]
}"""


class JSONLExporter:
    def export(self, annotated: list[AnnotatedRecord], output_path: str = None) -> Path:
        output_path = output_path or OUTPUT_JSONL
        path = Path(output_path)
        records = []

        for ann in annotated:
            r = ann.raw
            sec = ", ".join(f"{e['emotion']} {e['score']}" for e in r.secondary_emotions[:2])
            user_content = (
                f"Primary emotion: {r.primary_emotion} (confidence {r.confidence})\n"
                f"Secondary emotions: {sec or 'none'}\n"
                f"Somatic markers present: {r.somatic_markers}\n"
                f'User context: "{r.user_text}"\n\n'
                "Write the 4-panel comic script JSON now."
            )
            records.append({"messages": [
                {"role": "system",    "content": SYSTEM_PROMPT_FOR_TRAINING},
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": json.dumps(ann.script, indent=2)},
            ]})

        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        log.info(f"Exported {len(records)} training records → {path}")
        self._print_stats(annotated)
        return path

    def _print_stats(self, annotated: list[AnnotatedRecord]):
        from collections import Counter
        emotions = Counter(a.raw.primary_emotion for a in annotated)
        sources  = Counter(a.raw.source for a in annotated)
        print(f"\n{'─'*50}")
        print(f"  📊 Dataset Stats")
        print(f"{'─'*50}")
        print(f"  Total records : {len(annotated)}")
        print(f"\n  By emotion:")
        for e, n in sorted(emotions.items()): print(f"    {e:<12} {n:>5}")
        print(f"\n  By source:")
        for s, n in sorted(sources.items()): print(f"    {s:<20} {n:>5}")
        print(f"{'─'*50}\n")


# ---------------------------------------------------------------------------
# 6. Full Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    def __init__(self, limit: int = 500, resume: bool = False):
        DATA_DIR.mkdir(exist_ok=True)
        self.limit  = limit
        self.resume = resume
        self.cache  = DATA_DIR / "raw_records.json"
        self.ann_cache = DATA_DIR / "annotated_cache.jsonl"

    # ── Step 1: Download ──────────────────────────────────────────────────

    def step_download(self) -> list[RawRecord]:
        if self.cache.exists() and self.resume:
            log.info(f"Loading cached raw records from {self.cache}")
            return self._load_raw_cache()

        per_source = self.limit * 3   # over-sample, will filter in combine

        goe_records = GoEmotionsLoader().load(limit=per_source)
        sc_records  = StoryCommonsenseLoader().load(limit=per_source // 2)
        roc_records = ROCStoriesLoader().load(limit=per_source)

        combined = RecordCombiner().combine(
            goe_records, sc_records, roc_records, limit=self.limit * 2
        )

        self._save_raw_cache(combined)
        return combined

    # ── Step 2: Annotate ─────────────────────────────────────────────────

    def step_annotate(self, records: list[RawRecord]) -> list[AnnotatedRecord]:
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not set in .env\n"
                "Get a free key at: https://aistudio.google.com/app/apikey"
            )

        annotator = GeminiAnnotator()
        annotated = []

        # Load existing annotations if resuming
        done_texts = set()
        if self.resume and self.ann_cache.exists():
            with self.ann_cache.open() as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        annotated.append(AnnotatedRecord(
                            raw=RawRecord(**item["raw"]),
                            script=item["script"]
                        ))
                        done_texts.add(item["raw"]["user_text"])
                    except Exception:
                        continue
            log.info(f"Resumed: {len(annotated)} already annotated")

        # Filter out already done
        todo = [r for r in records if r.user_text not in done_texts]
        todo = todo[:self.limit - len(annotated)]

        log.info(f"Annotating {len(todo)} records with Gemini ...")

        try:
            from tqdm import tqdm
            iterator = tqdm(todo, desc="Annotating")
        except ImportError:
            iterator = todo

        ann_file = self.ann_cache.open("a", encoding="utf-8")

        for i, record in enumerate(iterator):
            script = annotator.annotate(record)
            if not script: continue

            ann = AnnotatedRecord(raw=record, script=script)
            annotated.append(ann)

            # Save to cache immediately (crash-safe)
            ann_file.write(json.dumps({
                "raw": {
                    "source": record.source,
                    "primary_emotion": record.primary_emotion,
                    "confidence": record.confidence,
                    "secondary_emotions": record.secondary_emotions,
                    "user_text": record.user_text,
                    "story_context": record.story_context,
                    "motivation": record.motivation,
                    "arc_hint": record.arc_hint,
                    "somatic_markers": record.somatic_markers,
                },
                "script": script,
            }, ensure_ascii=False) + "\n")
            ann_file.flush()

            # Rate limit: ~60 RPM for Gemini Flash free tier
            if (i + 1) % 50 == 0:
                log.info(f"  {i+1}/{len(todo)} annotated. "
                         f"Success: {annotator.call_count}, Fail: {annotator.fail_count}")
                time.sleep(2)

        ann_file.close()
        log.info(f"Annotation complete: {len(annotated)} records")
        return annotated

    # ── Step 3: Export ────────────────────────────────────────────────────

    def step_export(self, annotated: list[AnnotatedRecord]) -> Path:
        return JSONLExporter().export(annotated)

    # ── Full pipeline ─────────────────────────────────────────────────────

    def run(self):
        log.info("=== MoodWeaver Dataset Pipeline ===")
        records   = self.step_download()
        annotated = self.step_annotate(records)
        path      = self.step_export(annotated)
        log.info(f"\n✅ Done. Training JSONL saved → {path}")
        log.info(f"   Next: python stage2_story_generation.py --mode train")

    # ── Cache helpers ─────────────────────────────────────────────────────

    def _save_raw_cache(self, records: list[RawRecord]):
        with self.cache.open("w", encoding="utf-8") as f:
            json.dump([{
                "source": r.source, "primary_emotion": r.primary_emotion,
                "confidence": r.confidence, "secondary_emotions": r.secondary_emotions,
                "user_text": r.user_text, "story_context": r.story_context,
                "motivation": r.motivation, "arc_hint": r.arc_hint,
                "somatic_markers": r.somatic_markers,
            } for r in records], f, indent=2, ensure_ascii=False)
        log.info(f"Raw records cached → {self.cache}")

    def _load_raw_cache(self) -> list[RawRecord]:
        with self.cache.open() as f:
            items = json.load(f)
        return [RawRecord(**item) for item in items]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MoodWeaver Dataset Pipeline")
    parser.add_argument("--step", default="all",
                        choices=["all","download","annotate","export"],
                        help="Which step to run")
    parser.add_argument("--limit",  type=int, default=500,
                        help="Max annotated examples to generate (default: 500)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from cached progress")
    args = parser.parse_args()

    pipeline = Pipeline(limit=args.limit, resume=args.resume)

    if args.step == "download":
        records = pipeline.step_download()
        log.info(f"Downloaded {len(records)} raw records → {pipeline.cache}")

    elif args.step == "annotate":
        records   = pipeline.step_download()
        annotated = pipeline.step_annotate(records)
        log.info(f"Annotated {len(annotated)} records → {pipeline.ann_cache}")

    elif args.step == "export":
        if not pipeline.ann_cache.exists():
            log.error("No annotation cache found. Run --step annotate first.")
            return
        annotated = []
        with pipeline.ann_cache.open() as f:
            for line in f:
                try:
                    item = json.loads(line)
                    annotated.append(AnnotatedRecord(
                        raw=RawRecord(**item["raw"]), script=item["script"]))
                except Exception:
                    continue
        pipeline.step_export(annotated)

    else:
        pipeline.run()


if __name__ == "__main__":
    main()