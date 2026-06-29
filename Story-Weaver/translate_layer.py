"""
MoodWeaver — Hindi Translation Layer (Stage 2.5)
==================================================
Wraps your existing English story generator and translates output
into Hindi using IndicTrans2 — WITHOUT retraining or touching your
fine-tuned merged model.

Pipeline:
  English Generation (your existing merged model)
        ↓
  IndicTrans2 Translation (en → hi)
        ↓
  Hindi JSON  (same schema, same keys, translated values)

Only these fields are translated, keys stay identical:
  recurring_motif, mood_journey, visual, dialogue, emotion_beat, motion

Controlled via .env:
  LANGUAGE=english   → no translation, pass-through (default)
  LANGUAGE=hindi     → generate in English, then translate to Hindi

Usage:
    python translate_layer.py                  # uses LANGUAGE from .env
    python translate_layer.py --lang hindi      # override
    python translate_layer.py --lang english    # pass-through, no translation
    python translate_layer.py --file story.json # translate an existing JSON file

pip install transformers torch sentencepiece sacremoses python-dotenv
pip install IndicTransToolkit  (for proper IndicTrans2 pre/post-processing)
"""

import json, re, os, time, argparse, logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("moodweaver.translate")

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

def env(key, default=""): return os.environ.get(key, default).strip()

LANGUAGE        = env("LANGUAGE", "english").lower()          # english | hindi
TRANSLATE_MODEL = env("TRANSLATE_MODEL", "ai4bharat/indictrans2-en-indic-1B")
OUTPUT_FILE     = env("OUTPUT_FILE", "story_dynamic.json")
TRANSLATED_FILE = env("TRANSLATED_OUTPUT_FILE", "story_dynamic_hi.json")

TRANSLATABLE_FIELDS = ["recurring_motif", "mood_journey", "visual", "dialogue", "emotion_beat", "motion"]

TRANSLATE_EMOTION_BEAT_INLINE = False

# ---------------------------------------------------------------------------
# IndicTrans2 Translator
# ---------------------------------------------------------------------------

class IndicTrans2Translator:
    """
    Thin wrapper around AI4Bharat's IndicTrans2 (en → indic, 1B variant).
    Falls back to a smaller distilled model if the 1B model can't load
    (e.g. low VRAM), and finally to googletrans as a last resort.
    """

    SRC_LANG = "eng_Latn"
    TGT_LANG = "hin_Deva"

    def __init__(self, model_name: str = None):
        self.model_name = model_name or TRANSLATE_MODEL
        self.backend = None
        self._load()

    def _load(self):
        # 1) Try IndicTrans2 via IndicTransToolkit (preferred, best quality)
        try:
            self._load_indictrans2()
            self.backend = "indictrans2"
            log.info(f"Loaded IndicTrans2: {self.model_name}")
            return
        except Exception as e:
            log.warning(f"IndicTrans2 ({self.model_name}) failed to load: {e}")

        # 2) Try the smaller distilled 200M variant
        try:
            self.model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
            self._load_indictrans2()
            self.backend = "indictrans2"
            log.info(f"Loaded IndicTrans2 (distilled 200M fallback)")
            return
        except Exception as e:
            log.warning(f"IndicTrans2 distilled fallback failed: {e}")

        # 3) Last resort: googletrans (no GPU needed, lower quality)
        try:
            from googletrans import Translator as GoogleTranslator
            self.google = GoogleTranslator()
            self.backend = "googletrans"
            log.warning("Using googletrans fallback (lower quality, needs internet)")
            return
        except Exception as e:
            raise RuntimeError(
                "No translation backend available.\n"
                "Install one of:\n"
                "  pip install IndicTransToolkit transformers torch sentencepiece\n"
                "  pip install googletrans==4.0.0-rc1"
            )

    def _load_indictrans2(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit.processor import IndicProcessor

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()
        self.ip = IndicProcessor(inference=True)
        self.torch = torch

    def translate_batch(self, texts: list) -> list:
        """Translate a list of English strings to Hindi."""
        if not texts:
            return []

        if self.backend == "indictrans2":
            return self._translate_indictrans2(texts)
        elif self.backend == "googletrans":
            return self._translate_googletrans(texts)
        return texts

    def _translate_indictrans2(self, texts: list) -> list:
        batch = self.ip.preprocess_batch(texts, src_lang=self.SRC_LANG, tgt_lang=self.TGT_LANG)
        inputs = self.tokenizer(
            batch, truncation=True, padding="longest",
            return_tensors="pt", max_length=256,
        ).to(self.device)

        with self.torch.no_grad():
            generated = self.model.generate(
                **inputs, use_cache=True, min_length=0,
                max_length=256, num_beams=5, num_return_sequences=1,
            )

        decoded = self.tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        translations = self.ip.postprocess_batch(decoded, lang=self.TGT_LANG)
        return translations

    def _translate_googletrans(self, texts: list) -> list:
        results = []
        for t in texts:
            try:
                r = self.google.translate(t, src="en", dest="hi")
                results.append(r.text)
                time.sleep(0.3)  # avoid rate limiting
            except Exception as e:
                log.warning(f"Translation failed for '{t[:30]}...': {e}")
                results.append(t)  # fallback: keep English
        return results


# ---------------------------------------------------------------------------
# Story Translator — operates on MoodWeaver JSON schema
# ---------------------------------------------------------------------------

class StoryTranslator:
    """
    Translates a MoodWeaver story JSON in-place (keys unchanged),
    keeping _meta, panel numbers, and structure exactly as-is.

    emotion_beat is kept in English by default (used by evaluate.py's
    arc-direction logic) but a parallel "emotion_beat_hi" is added
    for display purposes.
    """

    def __init__(self, translator: Optional[IndicTrans2Translator] = None):
        self.translator = translator or IndicTrans2Translator()

    def translate_story(self, data: dict) -> dict:
        log.info("Collecting translatable strings ...")
        texts, paths = self._collect(data)

        if not texts:
            log.warning("No translatable text found.")
            return data

        log.info(f"Translating {len(texts)} strings (en → hi) ...")
        t0 = time.time()
        translations = self.translator.translate_batch(texts)
        elapsed = round(time.time() - t0, 2)
        log.info(f"Translation done in {elapsed}s "
                 f"(backend={self.translator.backend})")

        result = self._apply(data, paths, translations)
        result["_meta"] = result.get("_meta", {})
        result["_meta"]["language"] = "hindi"
        result["_meta"]["translation_backend"] = self.translator.backend
        result["_meta"]["translation_time_s"] = elapsed
        return result

    def _collect(self, data: dict):
        """
        Walk the story JSON and gather (text, json_path) pairs
        for every translatable field.
        """
        texts, paths = [], []

        for field in ("recurring_motif", "mood_journey"):
            val = data.get(field, "")
            if val and val.strip() and val.strip() != "...":
                texts.append(val)
                paths.append(("top", field, None))

        for i, panel in enumerate(data.get("panels", [])):
            for field in ("visual", "dialogue", "motion"):
                val = panel.get(field, "")
                if val and val.strip() and val.strip() != "...":
                    texts.append(val)
                    paths.append(("panel", field, i))
                # "..." (silence) stays as "..." — no translation needed

            if TRANSLATE_EMOTION_BEAT_INLINE:
                val = panel.get("emotion_beat", "")
                if val and val.strip():
                    texts.append(val)
                    paths.append(("panel", "emotion_beat", i))
            else:
                # Add a parallel Hindi label for display, English kept for logic
                val = panel.get("emotion_beat", "")
                if val and val.strip():
                    texts.append(val.replace("_", " "))
                    paths.append(("panel", "emotion_beat_hi", i))

        return texts, paths

    def _apply(self, data: dict, paths: list, translations: list) -> dict:
        import copy
        result = copy.deepcopy(data)

        for (scope, field, idx), translated in zip(paths, translations):
            if scope == "top":
                result[field] = translated
            elif scope == "panel":
                result["panels"][idx][field] = translated

        # Preserve "..." for silence (in case translator mangled it)
        for i, panel in enumerate(data.get("panels", [])):
            for field in ("dialogue",):
                if panel.get(field, "").strip() == "...":
                    result["panels"][i][field] = "..."

        return result


# ---------------------------------------------------------------------------
# Pretty Print (Hindi-aware)
# ---------------------------------------------------------------------------

ICONS = {"validation":"🔵","complication":"🟠","shift":"🟡","openness":"🟢"}

def print_story_bilingual(en_data: dict, hi_data: dict):
    panels_en = en_data.get("panels", [])
    panels_hi = hi_data.get("panels", [])
    n = len(panels_en)

    print(f"\n{'═'*70}")
    print(f"  🎨 MOODWEAVER — {n}-PANEL STORY (EN + HI)")
    print(f"  Motif (EN) : {en_data.get('recurring_motif','—')}")
    print(f"  Motif (HI) : {hi_data.get('recurring_motif','—')}")
    print(f"{'═'*70}\n")

    for i in range(n):
        pe, ph = panels_en[i], panels_hi[i]
        print(f"┌─ Panel {pe['panel']:02d}  beat: {pe['emotion_beat']}"
              f"  ({ph.get('emotion_beat_hi', pe['emotion_beat'])})")
        print(f"│  📷 EN: {pe['visual']}")
        print(f"│  📷 HI: {ph['visual']}")
        print(f"│  💬 EN: {pe['dialogue']}")
        print(f"│  💬 HI: {ph['dialogue']}")
        print(f"│  🏃 EN: {pe['motion']}")
        print(f"│  🏃 HI: {ph['motion']}")
        print(f"└{'─'*66}\n")


def print_story_hindi_only(data: dict):
    panels = data.get("panels", [])
    print(f"\n{'═'*70}")
    print(f"  🎨 मूडवीवर — {len(panels)}-पैनल कहानी")
    print(f"  विषयवस्तु : {data.get('recurring_motif','—')}")
    print(f"  यात्रा    : {data.get('mood_journey','—')}")
    print(f"{'═'*70}\n")

    for p in panels:
        beat_hi = p.get("emotion_beat_hi", p.get("emotion_beat",""))
        print(f"┌─ पैनल {p['panel']:02d}  भाव: {beat_hi}")
        print(f"│  📷 दृश्य  : {p['visual']}")
        print(f"│  💬 संवाद : {p['dialogue']}")
        print(f"│  🏃 क्रिया : {p['motion']}")
        print(f"└{'─'*66}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MoodWeaver Hindi Translation Layer")
    parser.add_argument("--lang", default=LANGUAGE, choices=["english","hindi"],
                        help="Target language (default from .env LANGUAGE)")
    parser.add_argument("--file", default=OUTPUT_FILE,
                        help="Input English story JSON (default: from .env OUTPUT_FILE)")
    parser.add_argument("--output", default=TRANSLATED_FILE,
                        help="Output Hindi JSON path")
    parser.add_argument("--bilingual", action="store_true",
                        help="Print both English and Hindi side by side")
    args = parser.parse_args()

    in_path = Path(args.file)
    if not in_path.exists():
        log.error(f"Input file not found: {in_path}\n"
                  f"Run your story generator first:\n"
                  f"  python stage2_story_generation.py")
        return

    en_data = json.loads(in_path.read_text(encoding="utf-8"))

    if args.lang == "english":
        log.info("LANGUAGE=english → pass-through, no translation applied.")
        log.info(f"Story already at: {in_path}")
        return

    # Hindi requested
    log.info("LANGUAGE=hindi → translating via IndicTrans2 ...")
    translator     = IndicTrans2Translator()
    story_translator = StoryTranslator(translator)
    hi_data = translator_result = story_translator.translate_story(en_data)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(hi_data, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Saved Hindi story → {out_path}")

    if args.bilingual:
        print_story_bilingual(en_data, hi_data)
    else:
        print_story_hindi_only(hi_data)


if __name__ == "__main__":
    main()