"""
MoodWeaver — Full Evaluation Engine
=====================================
Metrics:
  Rule-based    : structural + literary constraint checks
  Hallucination : input-output faithfulness checks
  Perplexity    : model fluency score (lower = better)
  ROUGE         : n-gram overlap vs reference stories
  BERTScore     : semantic similarity vs reference stories
  LLM-as-Judge  : literary quality scoring (Gemini or Anthropic)

Usage:
    python evaluate.py                                  # rule-based + hallucination only
    python evaluate.py --perplexity                     # + perplexity (loads model)
    python evaluate.py --nlp                            # + ROUGE + BERTScore
    python evaluate.py --llm                            # + LLM judge
    python evaluate.py --all                            # everything
    python evaluate.py --compare a.json b.json --all   # full model comparison

pip install google-generativeai anthropic transformers torch rouge-score bert-score nltk python-dotenv
"""

import json, re, os, argparse, logging, math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("moodweaver.eval")

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

# ---------------------------------------------------------------------------
# Config from .env
# ---------------------------------------------------------------------------

GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
LLM_PROVIDER      = os.environ.get("LLM_PROVIDER", "gemini").lower()   # gemini | anthropic
MODEL_PATH        = os.environ.get("MODEL_PATH", "moodweaver_stage2_merged")

# ---------------------------------------------------------------------------
# Word banks
# ---------------------------------------------------------------------------

EMOTION_WORDS = {
    "sad","sadness","happy","happiness","angry","anger","tired","exhausted",
    "anxious","anxiety","grief","grieving","depressed","depression","scared",
    "fear","lonely","loneliness","hopeless","hopeful","excited","joy","joyful",
    "pain","painful","upset","miserable","elated","furious","terrified",
}

SOMATIC_WORDS = {
    "chest","breath","breathing","throat","stomach","hands","hand","fingers",
    "lungs","jaw","shoulders","back","legs","feet","skin","pulse","heart",
    "eyes","face","neck","spine","gut","belly","exhale","inhale","breathe",
    "tightens","loosens","heavy","warm","cold","numb","ache","aching",
    "trembling","still","weight","tight","loose","knot","flutter","sink",
}

MORAL_PHRASES = [
    "the lesson is","remember that","always remember","never forget",
    "the moral","what matters is","in the end we","life teaches",
    "you should","we should","one must","the truth is",
]

EARLY_BEATS = {
    "heaviness","drag","spiral","contained_fire","absence","fracture",
    "peak_noise","ache","weight","numbness","tension","contained_rage",
    "sadness","sad","spark","doubt","jolt","resolve","tenderness","loss",
    "fire","tight",
}
LATE_BEATS = {
    "quiet_hope","renewal","present","transcendence","carried_forward",
    "soft_openness","tentative_light","luminous_still","radiance","ground",
    "root","stillness","open_silence","quiet_rest","faint_warmth",
    "triumph","victory","unity","forever","openness","rest","calm","light",
    "grounded","share","glow",
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RuleScore:
    name: str; passed: bool; detail: str = ""; weight: float = 1.0

@dataclass
class HallucinationScore:
    check: str; hallucinated: bool; detail: str = ""; severity: str = "low"

@dataclass
class LLMScore:
    dimension: str; score: Optional[float]; reasoning: str

@dataclass
class NLPScore:
    metric: str; value: float; detail: str = ""

@dataclass
class EvalReport:
    file: str
    rule_scores:          list = field(default_factory=list)
    hallucination_scores: list = field(default_factory=list)
    llm_scores:           list = field(default_factory=list)
    nlp_scores:           list = field(default_factory=list)
    perplexity:           Optional[float] = None

    @property
    def rule_pass_rate(self) -> float:
        if not self.rule_scores: return 0.0
        return round(
            sum(r.weight for r in self.rule_scores if r.passed)
            / sum(r.weight for r in self.rule_scores) * 100, 1)

    @property
    def hallucination_rate(self) -> float:
        if not self.hallucination_scores: return 0.0
        high   = sum(1 for h in self.hallucination_scores if h.hallucinated and h.severity == "high")
        medium = sum(1 for h in self.hallucination_scores if h.hallucinated and h.severity == "medium")
        low    = sum(1 for h in self.hallucination_scores if h.hallucinated and h.severity == "low")
        weighted = high * 1.0 + medium * 0.5 + low * 0.25
        return round(weighted / len(self.hallucination_scores) * 100, 1)

    @property
    def llm_avg(self) -> float:
        if not self.llm_scores: return 0.0
        valid = [s for s in self.llm_scores if s.score > 0]
        return round(sum(s.score for s in valid) / len(valid), 2) if valid else 0.0

    @property
    def composite(self) -> float:
        """
        Composite score formula (0–100):

          Component          Weight   Notes
          ─────────────────────────────────────────────────────
          Rule pass rate      30%     structural + literary rules
          Hallucination       20%     (100 - halluc_rate) / 100
          Perplexity          15%     normalized: <20=1.0, >200=0.0
          NLP (ROUGE/BERT)    15%     avg of available NLP scores
          LLM judge           20%     normalized from 1–5 → 0–1

        Components only count if they were actually run.
        Weights are redistributed proportionally if a component is missing.
        ─────────────────────────────────────────────────────────────────
        Example with all 5 components:
          rule=80%   → 0.80 × 30 = 24.0
          halluc=10% → (1-0.10) × 20 = 18.0
          perp=35    → ((200-35)/180) × 15 ≈ 13.75   [linear interp 20→200]
          nlp=0.42   → 0.42 × 15 = 6.3
          llm=3.8/5  → ((3.8-1)/4) × 20 = 14.0
          total_weight = 100
          composite = (24+18+13.75+6.3+14) / 100 × 100 = 76.05 → 76.1
        """
        components = {}   # name → (raw_score_0_to_1, weight)

        # Rule-based (always present)
        components["rule"] = (self.rule_pass_rate / 100, 30)

        # Hallucination (always present if evaluated)
        if self.hallucination_scores:
            components["halluc"] = ((100 - self.hallucination_rate) / 100, 20)

        # Perplexity: linear interpolation between 20 (best=1.0) and 200 (worst=0.0)
        if self.perplexity is not None:
            perp_norm = max(0.0, min(1.0, (200 - self.perplexity) / 180))
            components["perp"] = (perp_norm, 15)

        # NLP metrics: average of all available (ROUGE, BERTScore, BLEU)
        if self.nlp_scores:
            nlp_avg = sum(s.value for s in self.nlp_scores) / len(self.nlp_scores)
            components["nlp"] = (nlp_avg, 15)

        # LLM judge: normalize 1–5 → 0–1
        if self.llm_scores and self.llm_avg > 0:
            llm_norm = (self.llm_avg - 1) / 4
            components["llm"] = (llm_norm, 20)

        if not components: return 0.0

        # Redistribute weights proportionally so they always sum to 100
        total_weight  = sum(w for _, w in components.values())
        weighted_sum  = sum(score * weight for score, weight in components.values())
        return round((weighted_sum / total_weight) * 100, 1)


# ---------------------------------------------------------------------------
# 1. Rule-Based Evaluator
# ---------------------------------------------------------------------------

class RuleEvaluator:
    def evaluate(self, data: dict) -> list:
        panels  = data.get("panels", [])
        emotion = data.get("_meta", {}).get("emotion", "sad")
        motif   = data.get("recurring_motif", "").lower()
        s = []
        s.append(self._panel_count(panels, data))
        s.append(self._json_structure(panels))
        s.append(self._motif_in_all_panels(panels, motif))
        s.append(self._no_direct_emotion(panels))
        s.append(self._beat_single_word(panels))
        s.append(self._somatic_every_panel(panels))
        s.append(self._dialogue_brevity(panels))
        s.append(self._no_moral_lesson(panels))
        s.append(self._arc_direction(panels, emotion))
        s.append(self._panel_length_balance(panels))
        s.append(self._motif_specificity(motif))
        s.append(self._no_empty_fields(panels))
        return s

    def _panel_count(self, panels, data):
        expected = data.get("_meta", {}).get("panel_count", len(panels))
        ok = len(panels) == expected
        return RuleScore("panel_count", ok, f"got {len(panels)}, expected {expected}", 2.0)

    def _json_structure(self, panels):
        req = {"panel","visual","dialogue","emotion_beat","motion"}
        bad = [f"p{p.get('panel')}: {req-set(p)}" for p in panels if req-set(p)]
        return RuleScore("json_structure", not bad,
                         "all fields present" if not bad else "; ".join(bad), 2.0)

    def _motif_in_all_panels(self, panels, motif):
        if not motif:
            return RuleScore("motif_in_all_panels", False, "no motif set", 1.5)
        words = [w.lower() for w in motif.split() if len(w) > 3]
        if not words:
            words = [motif.split()[0].lower()] if motif.split() else [""]
        
        miss = []
        for p in panels:
            visual_text = p.get("visual", "").lower()
            if not any(w in visual_text for w in words if w):
                miss.append(p["panel"])
        
        return RuleScore("motif_in_all_panels", not miss,
                         f"motif words {words} missing in panels {miss}" if miss
                         else f"motif present in all panels", 1.5)

    def _no_direct_emotion(self, panels):
        bad = []
        for p in panels:
            text = f"{p.get('visual','')} {p.get('dialogue','')} {p.get('motion','')}".lower()
            hits = [w for w in EMOTION_WORDS if re.search(r'\b'+w+r'\b', text)]
            if hits: bad.append(f"p{p['panel']}:{hits}")
        return RuleScore("no_direct_emotion", not bad,
                         "clean" if not bad else "; ".join(bad), 1.5)

    def _beat_single_word(self, panels):
        bad = [f"p{p['panel']}:'{p.get('emotion_beat','')}'" for p in panels
               if len(p.get("emotion_beat","").split()) != 1]
        return RuleScore("beat_single_word", not bad,
                         "all single-word" if not bad else "; ".join(bad))

    def _somatic_every_panel(self, panels):
        miss = [p["panel"] for p in panels
                if not any(w in f"{p.get('visual','')} {p.get('motion','')}".lower()
                           for w in SOMATIC_WORDS)]
        return RuleScore("somatic_every_panel", not miss,
                         "body sensation in all" if not miss else f"missing p{miss}", 1.5)

    def _dialogue_brevity(self, panels):
        bad = [f"p{p['panel']}({len(p.get('dialogue','').split())}w)" for p in panels
               if p.get("dialogue","") not in ("...","") and
               len(p.get("dialogue","").split()) > 14]
        return RuleScore("dialogue_brevity", not bad,
                         "all concise" if not bad else f"too long: {bad}")

    def _no_moral_lesson(self, panels):
        bad = []
        for p in panels:
            text = f"{p.get('visual','')} {p.get('dialogue','')}".lower()
            hits = [ph for ph in MORAL_PHRASES if ph in text]
            if hits: bad.append(f"p{p['panel']}:{hits}")
        return RuleScore("no_moral_lesson", not bad,
                         "none found" if not bad else "; ".join(bad), 1.5)

    def _arc_direction(self, panels, emotion):
        beats  = [p.get("emotion_beat","").lower() for p in panels]
        f_ok   = beats[0] in EARLY_BEATS if beats else False
        l_ok   = beats[-1] in LATE_BEATS  if beats else False
        return RuleScore("arc_direction", f_ok and l_ok,
                         f"start='{beats[0]}'({'✓' if f_ok else '✗'}) "
                         f"end='{beats[-1]}'({'✓' if l_ok else '✗'})", 2.0)

    def _panel_length_balance(self, panels):
        lens  = [len(p.get("visual","") + p.get("motion","")) for p in panels]
        ratio = max(lens) / max(min(lens), 1) if lens else 0
        return RuleScore("panel_length_balance", ratio < 3.5, f"ratio={ratio:.1f} (ok<3.5)")

    def _motif_specificity(self, motif):
        ok = len(motif.split()) >= 3
        return RuleScore("motif_specificity", ok,
                         f"'{motif}' — {'specific' if ok else 'too vague (needs ≥3 words)'}")

    def _no_empty_fields(self, panels):
        bad = [f"p{p.get('panel')}.{k}" for p in panels
               for k in ("visual","dialogue","emotion_beat","motion")
               if not p.get(k,"").strip()]
        return RuleScore("no_empty_fields", not bad,
                         "all filled" if not bad else f"empty: {bad}", 2.0)


# ---------------------------------------------------------------------------
# 2. Hallucination Evaluator
# ---------------------------------------------------------------------------

class HallucinationEvaluator:
    def evaluate(self, data: dict) -> list:
        panels  = data.get("panels", [])
        emotion = data.get("_meta", {}).get("emotion", "sad")
        motif   = data.get("recurring_motif", "").lower()
        return [
            self._emotion_contradiction(panels, emotion),
            self._arc_reversal(panels, emotion),
            self._motif_abandonment(panels, motif),
            self._instruction_violation(panels),
            self._phantom_characters(panels),
            self._temporal_inconsistency(panels),
            self._panel_numbering_drift(panels),
        ]

    def _emotion_contradiction(self, panels, emotion):
        POSITIVE_BEATS = {"joy","elation","happiness","delight","ecstasy",
                          "transcendence","spark","radiance","overflow"}
        NEGATIVE_INPUT = {"sad","angry","tired","anxious","grief"}
        beats = [p.get("emotion_beat","").lower() for p in panels]
        pos   = sum(1 for b in beats if b in POSITIVE_BEATS)
        if emotion in NEGATIVE_INPUT and pos > len(panels) * 0.6:
            return HallucinationScore("emotion_contradiction", True,
                f"Input={emotion} but {pos}/{len(panels)} panels have positive beats",
                severity="high")
        return HallucinationScore("emotion_contradiction", False, "arc matches input emotion")

    def _arc_reversal(self, panels, emotion):
        beats = [p.get("emotion_beat","").lower() for p in panels]
        if not beats:
            return HallucinationScore("arc_reversal", False, "no panels")
        uplift = {"sad","tired","anxious","grief"}
        if emotion in uplift:
            if beats[-1] in EARLY_BEATS and beats[0] not in EARLY_BEATS:
                return HallucinationScore("arc_reversal", True,
                    f"Gets darker: '{beats[0]}'→'{beats[-1]}'", severity="high")
        return HallucinationScore("arc_reversal", False,
                                  f"ok: '{beats[0]}'→'{beats[-1]}'")

    def _motif_abandonment(self, panels, motif):
        if not motif:
            return HallucinationScore("motif_abandonment", True,
                                      "no recurring_motif declared", severity="medium")
        words = [w.lower() for w in motif.split() if len(w) > 3]
        if not words:
            words = [motif.split()[0].lower()] if motif.split() else [""]
            
        present = sum(1 for p in panels if any(w in p.get("visual", "").lower() for w in words if w))
        if present == 0:
            return HallucinationScore("motif_abandonment", True,
                f"'{motif}' never appears in any visual", severity="high")
        if present < len(panels) * 0.5:
            return HallucinationScore("motif_abandonment", True,
                f"motif words {words} only in {present}/{len(panels)} panels", severity="medium")
        return HallucinationScore("motif_abandonment", False,
                                  f"motif present in {present}/{len(panels)} panels")

    def _instruction_violation(self, panels):
        bad = []
        for p in panels:
            text = f"{p.get('visual','')} {p.get('dialogue','')}".lower()
            hits = [ph for ph in MORAL_PHRASES if ph in text]
            if hits: bad.append(f"p{p['panel']}:{hits}")
        return HallucinationScore("instruction_violation", bool(bad),
            "; ".join(bad) if bad else "none found",
            severity="medium" if bad else "low")

    def _phantom_characters(self, panels):
        # Proper nouns that are NOT common sentence starters
        IGNORE = {"The","She","He","They","Her","His","Their","Panel","Visual",
                  "With","Sometimes","Outside","Softly","Then","Now","After",
                  "Before","Inside","Above","Below","Beside","Against","Through",
                  "Slowly","Quietly","Suddenly","Finally","Later","Still","Just",
                  "Only","Even","Also","Both","Each","Every","Some","Many","Most"}
        VALID_CHARACTERS = {"Wanderer", "Ember", "Kael", "Aria", "Elias", "Vesper", "Riven"}
        all_text = " ".join(p.get("visual","") + " " + p.get("dialogue","")
                            for p in panels)
        
        # Only extract capitalized words that are not starting a sentence,
        # unless they are known characters.
        raw_names = set()
        for m in re.finditer(r'\b[A-Z][a-z]{2,}\b', all_text):
            word = m.group()
            if word in VALID_CHARACTERS:
                raw_names.add(word)
                continue
            
            # Check if this occurrence is at the start of a sentence
            preceding = all_text[:m.start()].rstrip(' "\'')
            if len(preceding) > 0 and preceding[-1] not in ('.', '!', '?'):
                raw_names.add(word)

        names = raw_names - IGNORE
        hallu = len(names) > 2
        return HallucinationScore("phantom_characters", hallu,
            f"found names: {names}" if hallu else "character count normal",
            severity="low")

    def _temporal_inconsistency(self, panels):
        TIME = {"morning":1,"dawn":1,"sunrise":1,"afternoon":2,"noon":2,
                "evening":3,"dusk":3,"sunset":3,"night":4,"midnight":4,
                "3am":4,"2am":4,"1am":4,"4am":4}
        times = []
        for p in panels:
            text = p.get("visual","").lower()
            for w, v in TIME.items():
                if w in text: times.append((p["panel"], v, w)); break
        if len(times) < 2:
            return HallucinationScore("temporal_inconsistency", False, "insufficient time refs")
        jumps = [(a,b) for a,b in zip(times, times[1:]) if abs(a[1]-b[1]) >= 3]
        return HallucinationScore("temporal_inconsistency", bool(jumps),
            "; ".join(f"p{a[0]}:{a[2]}→p{b[0]}:{b[2]}" for a,b in jumps)
            if jumps else "temporal flow consistent", severity="low")

    def _panel_numbering_drift(self, panels):
        nums = [p.get("panel", 0) for p in panels]
        expected = list(range(1, len(panels)+1))
        hallu = nums != expected
        return HallucinationScore("panel_numbering_drift", hallu,
            f"got {nums}" if hallu else "numbering correct",
            severity="medium" if hallu else "low")


# ---------------------------------------------------------------------------
# 3. Perplexity Evaluator
# ---------------------------------------------------------------------------

class PerplexityEvaluator:
    """
    Perplexity = how surprised your model is by its own output.
    Lower = more fluent and confident.

    Bands:
      < 20  : excellent
      20-50 : normal for instruction-tuned models
      50-100: uncertain phrasing
      > 100 : incoherent / out-of-distribution
    """
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        log.info(f"Loading model for perplexity: {MODEL_PATH}")
        self.tok   = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
        self.model.eval()
        self.torch = torch

    def compute(self, data: dict) -> float:
        parts = [
            data.get("recurring_motif",""), data.get("mood_journey",""),
            *[f"{p.get('visual','')} {p.get('dialogue','')} {p.get('motion','')}"
              for p in data.get("panels",[])]
        ]
        text = " ".join([str(p) for p in parts if p])
        tok = self.tok
        assert tok is not None
        enc = tok(text, return_tensors="pt").to(self.model.device)
        ids = enc["input_ids"][:, :512]
        with self.torch.no_grad():
            loss = self.model(ids, labels=ids).loss
        ppl = round(math.exp(loss.item()), 2)
        log.info(f"Perplexity: {ppl}")
        return ppl


# ---------------------------------------------------------------------------
# 4. NLP Metrics (ROUGE + BERTScore + BLEU)
# ---------------------------------------------------------------------------

class NLPEvaluator:
    def evaluate(self, data: dict, reference_files: list) -> list:
        if not reference_files:
            log.warning("No --refs provided — skipping ROUGE/BERTScore/BLEU")
            return []

        hyp  = self._to_text(data)
        refs = []
        for f in reference_files:
            try: refs.append(self._to_text(json.loads(Path(f).read_text())))
            except Exception as e: log.warning(f"Ref load failed {f}: {e}")
        if not refs: return []

        scores = []

        # ROUGE
        try:
            from rouge_score import rouge_scorer
            sc = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
            r1s,r2s,rls = [],[],[]
            for r in refs:
                s = sc.score(r, hyp)
                r1s.append(s["rouge1"].fmeasure)
                r2s.append(s["rouge2"].fmeasure)
                rls.append(s["rougeL"].fmeasure)
            scores += [
                NLPScore("ROUGE-1", round(sum(r1s)/len(r1s),4),
                         "unigram overlap — how similar vocabulary is to references"),
                NLPScore("ROUGE-2", round(sum(r2s)/len(r2s),4),
                         "bigram overlap — captures more phrase-level similarity"),
                NLPScore("ROUGE-L", round(sum(rls)/len(rls),4),
                         "longest common subsequence — structure similarity"),
            ]
        except ImportError:
            log.warning("pip install rouge-score")

        # BERTScore
        try:
            from bert_score import score as bscore
            P, R, F = bscore([hyp]*len(refs), refs, lang="en", verbose=False)
            scores.append(NLPScore("BERTScore-F1", round(F.mean().item(),4),
                                   "semantic similarity (meaning-level, 0–1)"))
        except ImportError:
            log.warning("pip install bert-score")

        # BLEU
        try:
            import nltk
            nltk.download("punkt", quiet=True)
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            hyp_t = hyp.lower().split()
            ref_t = [r.lower().split() for r in refs]
            bleu  = sentence_bleu(ref_t, hyp_t,
                                  smoothing_function=SmoothingFunction().method1)
            scores.append(NLPScore("BLEU", round(bleu,4),
                                   "n-gram precision (less meaningful for creative text)"))
        except ImportError:
            log.warning("pip install nltk")

        return scores

    def _to_text(self, data: dict) -> str:
        parts = [data.get("recurring_motif",""), data.get("mood_journey","")]
        for p in data.get("panels",[]):
            parts += [p.get("visual",""), p.get("dialogue",""), p.get("motion","")]
        return " ".join([str(p) for p in parts if p])


# ---------------------------------------------------------------------------
# 5. LLM-as-Judge  (Gemini or Anthropic)
# ---------------------------------------------------------------------------

LLM_DIMENSIONS = [
    ("literary_quality",
     "Does this read like actual literary graphic novel writing for adults? "
     "Score 1 (generic AI text) to 5 (genuinely literary, precise, restrained)."),
    ("show_dont_tell",
     "Are emotions shown ONLY through objects, actions, physical sensation? "
     "Score 1 (emotions named directly) to 5 (never named, always shown)."),
    ("motif_coherence",
     "Does the recurring visual motif feel intentional, not mechanically inserted? Score 1-5."),
    ("arc_authenticity",
     "Does the mood journey feel emotionally earned and gradual, not rushed? Score 1-5."),
    ("dialogue_subtext",
     "Does the dialogue or silence carry emotional weight and subtext? "
     "Score 1 (flat) to 5 (deeply restrained and resonant)."),
    ("panel_flow",
     "Do panels transition naturally, building a coherent experience? Score 1-5."),
]


class LLMJudge:
    def __init__(self):
        if LLM_PROVIDER == "gemini":
            self._init_gemini()
        else:
            self._init_anthropic()

    def _init_gemini(self):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("pip install google-generativeai")

        if not GEMINI_API_KEY:
            raise ValueError("Set GEMINI_API_KEY in .env")

        genai.configure(api_key=GEMINI_API_KEY)

        self.gemini_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash"
        )

        self.provider = "gemini"
        log.info("LLM judge: using Gemini")

    def _init_anthropic(self):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        if not ANTHROPIC_API_KEY:
            raise ValueError("Set ANTHROPIC_API_KEY in .env")

        self.anthropic_client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY
        )

        self.provider = "anthropic"
        log.info("LLM judge: using Anthropic")

    def _call(self, prompt: str) -> str:
        import time

        for attempt in range(3):
            try:
                if self.provider == "gemini":
                    resp = self.gemini_model.generate_content(prompt)
                    return resp.text.strip()

                resp = self.anthropic_client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=800,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                return resp.content[0].text.strip()

            except Exception as e:
                err = str(e)

                if "429" in err and attempt < 2:
                    wait = 35
                    log.warning(
                        f"Rate limited. Waiting {wait}s before retry..."
                    )
                    time.sleep(wait)
                    continue

                raise

        raise RuntimeError("LLM request failed after retries")

    def evaluate(self, data: dict) -> list:
        story = "\n".join(
            f"Panel {p['panel']} [{p.get('emotion_beat', '')}]\n"
            f"Visual: {p.get('visual', '')}\n"
            f"Dialogue: {p.get('dialogue', '')}\n"
            f"Motion: {p.get('motion', '')}"
            for p in data.get("panels", [])
        )

        header = (
            f"Motif: {data.get('recurring_motif', '')}\n"
            f"Journey: {data.get('mood_journey', '')}\n\n"
            f"{story}"
        )

        prompt = f"""
You are an expert literary critic evaluating a graphic novel script.

STORY:
{header}

Evaluate the story on the following dimensions:

1. literary_quality
2. show_dont_tell
3. motif_coherence
4. arc_authenticity
5. dialogue_subtext
6. panel_flow

Score each category from 1.0 to 5.0.

Return ONLY valid JSON.

{{
  "literary_quality": {{
    "score": 4.0,
    "reasoning": "..."
  }},
  "show_dont_tell": {{
    "score": 4.0,
    "reasoning": "..."
  }},
  "motif_coherence": {{
    "score": 4.0,
    "reasoning": "..."
  }},
  "arc_authenticity": {{
    "score": 4.0,
    "reasoning": "..."
  }},
  "dialogue_subtext": {{
    "score": 4.0,
    "reasoning": "..."
  }},
  "panel_flow": {{
    "score": 4.0,
    "reasoning": "..."
  }}
}}
"""

        try:
            raw = self._call(prompt)

            raw = re.sub(
                r"^```(?:json)?\s*",
                "",
                raw.strip()
            )
            raw = re.sub(
                r"\s*```$",
                "",
                raw
            ).strip()

            parsed = json.loads(raw)

            dimensions = [
                "literary_quality",
                "show_dont_tell",
                "motif_coherence",
                "arc_authenticity",
                "dialogue_subtext",
                "panel_flow",
            ]

            scores = []

            for dim in dimensions:
                score = float(parsed[dim]["score"])
                reasoning = str(parsed[dim]["reasoning"])

                scores.append(
                    LLMScore(
                        dim,
                        score,
                        reasoning,
                    )
                )

                log.info(f"  {dim}: {score}/5")

            return scores

        except Exception as e:
            log.warning(f"LLM judge failed: {e}")

            return [
                LLMScore(
                    dim,
                    None,
                    f"failed: {e}"
                )
                for dim, _ in LLM_DIMENSIONS
            ]

# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

SEV = {"high":"🔴","medium":"🟡","low":"⚪"}

def print_report(report: EvalReport):
    print(f"\n{'═'*65}")
    print(f"  📊 EVALUATION — {Path(report.file).name}")
    print(f"{'═'*65}")

    print(f"\n── RULE-BASED  ({report.rule_pass_rate}% weighted) ──\n")
    for r in report.rule_scores:
        w = f"[w={r.weight}]" if r.weight != 1.0 else "      "
        print(f"  {'✅' if r.passed else '❌'} {r.name:<28} {w}  {r.detail}")

    if report.hallucination_scores:
        print(f"\n── HALLUCINATION  (weighted rate: {report.hallucination_rate}%) ──\n")
        for h in report.hallucination_scores:
            icon = "🚨" if h.hallucinated else "✅"
            sev  = SEV.get(h.severity,"") if h.hallucinated else ""
            print(f"  {icon} {sev} {h.check:<28}  {h.detail}")

    if report.perplexity is not None:
        band = ("excellent <20" if report.perplexity < 20 else
                "good 20–50"    if report.perplexity < 50 else
                "fair 50–100"   if report.perplexity < 100 else "poor >100")
        print(f"\n── PERPLEXITY ──\n  📉 {report.perplexity}  ({band})")

    if report.nlp_scores:
        print(f"\n── NLP METRICS ──\n")
        for s in report.nlp_scores:
            bar = "█" * int(s.value*10) + "░" * (10 - int(s.value*10))
            print(f"  {s.metric:<14} {s.value:.4f}  {bar}  {s.detail}")

    if report.llm_scores:
        print(f"\n── LLM-AS-JUDGE ({LLM_PROVIDER})  avg: {report.llm_avg}/5.0 ──\n")
        for s in report.llm_scores:
            bar = "█" * int(s.score) + "░" * (5-int(s.score))
            print(f"  {s.score:.1f}/5  {bar}  {s.dimension:<22}  {s.reasoning}")

    grade = "A" if report.composite>=85 else "B" if report.composite>=70 else \
            "C" if report.composite>=55 else "D"
    print(f"\n{'─'*65}")
    print(f"  COMPOSITE : {report.composite}/100   GRADE: {grade}")
    print(f"\n  Formula breakdown:")
    print(f"    Rule pass rate  (30% weight) = {report.rule_pass_rate}%")
    print(f"    Halluc penalty  (20% weight) = {100-report.hallucination_rate}% clean")
    if report.perplexity: print(f"    Perplexity      (15% weight) = {report.perplexity} (lower=better)")
    if report.nlp_scores: print(f"    NLP metrics     (15% weight) = avg {round(sum(s.value for s in report.nlp_scores)/len(report.nlp_scores),3)}")
    if report.llm_scores: print(f"    LLM judge       (20% weight) = {report.llm_avg}/5.0")
    print(f"    → Composite     = {report.composite}/100  [{grade}]")
    print(f"{'═'*65}\n")


def print_comparison(r1: EvalReport, r2: EvalReport):
    n1, n2 = Path(r1.file).name, Path(r2.file).name
    print(f"\n{'═'*65}")
    print(f"  ⚖️  COMPARISON  |  {n1}  vs  {n2}")
    print(f"{'═'*65}")
    rows = [
        ("Rule pass %",          r1.rule_pass_rate,      r2.rule_pass_rate,      False),
        ("Hallucination %",      r1.hallucination_rate,  r2.hallucination_rate,  True),
        ("Perplexity",           r1.perplexity or 999,   r2.perplexity or 999,   True),
        ("LLM avg /5",           r1.llm_avg,             r2.llm_avg,             False),
        ("COMPOSITE /100",       r1.composite,           r2.composite,           False),
    ]
    for label, v1, v2, lower_better in rows:
        win = "◀" if (v1<v2 if lower_better else v1>v2) else ("▶" if (v2<v1 if lower_better else v2>v1) else "=")
        print(f"  {label:<24} {str(v1):>10}  {str(v2):>10}   {win}  {'(lower=better)' if lower_better else ''}")
    best = n1 if r1.composite >= r2.composite else n2
    print(f"\n  🏆 Better overall: {best}")
    print(f"{'═'*65}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eval(file: str, args) -> EvalReport:
    data   = json.loads(Path(file).read_text(encoding="utf-8"))
    report = EvalReport(file=file)
    report.rule_scores          = RuleEvaluator().evaluate(data)
    report.hallucination_scores = HallucinationEvaluator().evaluate(data)

    if getattr(args,"perplexity",False) or getattr(args,"all",False):
        try: report.perplexity = PerplexityEvaluator().compute(data)
        except Exception as e: log.warning(f"Perplexity: {e}")

    if getattr(args,"nlp",False) or getattr(args,"all",False):
        report.nlp_scores = NLPEvaluator().evaluate(data, getattr(args,"refs",None) or [])

    if getattr(args,"llm",False) or getattr(args,"all",False):
        try: report.llm_scores = LLMJudge().evaluate(data)
        except Exception as e: log.warning(f"LLM judge: {e}")

    out = Path(file).with_suffix(".eval.json")
    out.write_text(json.dumps({
        "file":file, "rule_pass_rate":report.rule_pass_rate,
        "hallucination_rate":report.hallucination_rate, "perplexity":report.perplexity,
        "nlp":[{"metric":s.metric,"value":s.value} for s in report.nlp_scores],
        "llm":[{"dim":s.dimension,"score":s.score,"reason":s.reasoning} for s in report.llm_scores],
        "composite":report.composite,
    }, indent=2), encoding="utf-8")
    return report

def _env(key, default=""):
    return os.environ.get(key, default).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",default=_env("OUTPUT_FILE", "story_dynamic.json"))
    parser.add_argument("--compare",    nargs=2, metavar=("A","B"))
    parser.add_argument("--refs",       nargs="*", metavar="REF")
    parser.add_argument("--perplexity", action="store_true")
    parser.add_argument("--nlp",        action="store_true")
    parser.add_argument("--llm",        action="store_true")
    parser.add_argument("--all",        action="store_true")
    args = parser.parse_args()

    if args.compare:
        r1 = run_eval(args.compare[0], args)
        r2 = run_eval(args.compare[1], args)
        print_report(r1); print_report(r2); print_comparison(r1, r2)
    else:
        print_report(run_eval(args.file, args))

if __name__ == "__main__":
    main()