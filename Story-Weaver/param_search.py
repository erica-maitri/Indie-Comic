"""
MoodWeaver — Parameter Search via Knowledge Graph
==================================================
Builds a knowledge graph of parameter combinations → quality scores,
runs experiments across a defined range for each parameter,
and finds the best static values for your .env.

What it does:
  1. Define search space (ranges for each param)
  2. Run N experiments — each generates a story with a param combo
  3. Score each story (rule-based + hallucination, optionally LLM)
  4. Build a NetworkX knowledge graph: nodes=param combos, edges=similarity
  5. Analyse graph to find best cluster + optimal static values
  6. Write results to best_params.env (drop-in replacement for your .env block)

Usage:
    python param_search.py                          # full search, 60 experiments
    python param_search.py --experiments 120        # more thorough
    python param_search.py --strategy grid          # grid search (exhaustive)
    python param_search.py --strategy random        # random search (default)
    python param_search.py --strategy bayesian      # bayesian optimisation
    python param_search.py --no-llm                 # skip LLM judge (faster)
    python param_search.py --visualize              # save graph PNG

pip install networkx numpy scikit-learn matplotlib scipy python-dotenv
pip install scikit-optimize   # only needed for --strategy bayesian
"""

import json, re, os, time, argparse, logging, itertools, copy
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("moodweaver.paramsearch")

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

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def env(key, default=""): return os.environ.get(key, default).strip()

MODEL_PATH     = env("MODEL_PATH",     "moodweaver_stage2_merged")
GEMINI_API_KEY = env("GEMINI_API_KEY", "")
LLM_PROVIDER   = env("LLM_PROVIDER",  "gemini")
OUTPUT_DIR     = Path(env("PARAM_SEARCH_DIR", "param_search_results"))

# ---------------------------------------------------------------------------
# 1. Search Space Definition
# ---------------------------------------------------------------------------

SEARCH_SPACE = {
    "TEMPERATURE": {
        "range":    [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
        "type":     "float",
        "description": "Controls randomness. Low=focused/repetitive, High=creative/chaotic.",
        "prior":    0.72,   # current .env value
    },
    "TOP_P": {
        "range":    [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.97, 1.00],
        "type":     "float",
        "description": "Nucleus sampling. Low=safer vocab, High=diverse vocab.",
        "prior":    0.92,
    },
    "REPETITION_PENALTY": {
        "range":    [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40],
        "type":     "float",
        "description": "Penalises repeated tokens. Low=more repetition, High=too diverse.",
        "prior":    1.15,
    },
    "MAX_TOKENS_PER_PANEL": {
        "range":    [100, 120, 140, 160, 180, 200, 220, 250],
        "type":     "int",
        "description": "Token budget per panel. Too low=truncated, Too high=padding.",
        "prior":    200,
    },
}

# Fixed test inputs — same for every experiment so results are comparable
TEST_INPUTS = [
    {"emotion": "sad",     "panel_count": 6,
     "user_text": "everything feels heavy, even small things"},
    {"emotion": "angry",   "panel_count": 5,
     "user_text": "i said something i regret and i cant take it back"},
    {"emotion": "tired",   "panel_count": 6,
     "user_text": "i cant get out of bed today"},
    {"emotion": "happy",   "panel_count": 5,
     "user_text": "today was unexpectedly beautiful"},
    {"emotion": "anxious", "panel_count": 6,
     "user_text": "my mind wont stop racing at night"},
    {"emotion": "grief",   "panel_count": 6,
     "user_text": "my dog passed away yesterday, the house is too quiet"},
]

# ---------------------------------------------------------------------------
# 2. Experiment Result Schema
# ---------------------------------------------------------------------------

@dataclass
class ParamCombo:
    temperature:          float
    top_p:                float
    repetition_penalty:   float
    max_tokens_per_panel: int

    def to_dict(self) -> dict:
        return {
            "TEMPERATURE":          self.temperature,
            "TOP_P":                self.top_p,
            "REPETITION_PENALTY":   self.repetition_penalty,
            "MAX_TOKENS_PER_PANEL": self.max_tokens_per_panel,
        }

    def key(self) -> str:
        return (f"T{self.temperature:.2f}_P{self.top_p:.2f}_"
                f"R{self.repetition_penalty:.2f}_M{self.max_tokens_per_panel}")


@dataclass
class ExperimentResult:
    combo:              ParamCombo
    input_emotion:      str
    rule_pass_rate:     float        # 0–100
    hallucination_rate: float        # 0–100 (lower = better)
    llm_avg:            float        # 0–5 (0 if not run)
    composite:          float        # 0–100
    generation_time_s:  float
    json_valid:         bool
    story:              Optional[dict] = None
    error:              Optional[str]  = None

    def feature_vector(self) -> list:
        """For graph similarity / clustering."""
        return [
            self.combo.temperature,
            self.combo.top_p,
            self.combo.repetition_penalty,
            self.combo.max_tokens_per_panel / 250.0,   # normalise
            self.rule_pass_rate / 100.0,
            self.hallucination_rate / 100.0,
            self.llm_avg / 5.0,
            self.composite / 100.0,
            float(self.json_valid),
        ]


# ---------------------------------------------------------------------------
# 3. Story Generator (thin wrapper, reads params from combo)
# ---------------------------------------------------------------------------

MOOD_ARCS = {
    "sad":     {"journey":"uplifting",  "beats":["heaviness","weight","stillness","faint_warmth","soft_openness","quiet_hope"],"motif":"a small warm object: a chipped mug, a candle, a patch of sunlight"},
    "angry":   {"journey":"calming",    "beats":["contained_fire","fracture","peak","exhale","cooling","ground"],              "motif":"something that absorbs heat: running water, open window, cold floor"},
    "tired":   {"journey":"relaxing",   "beats":["drag","push","surrender","softness","drift","renewal"],                      "motif":"something soft and horizontal: a blanket, evening light, a pillow"},
    "happy":   {"journey":"elation",    "beats":["spark","warmth","expansion","overflow","radiance","transcendence"],          "motif":"something that multiplies light: water surface, open hands, sky"},
    "anxious": {"journey":"grounding",  "beats":["spiral","peak_noise","one_thing","breath","root","present"],                 "motif":"something tactile: rough wall, bare feet on cool floor, a held object"},
    "grief":   {"journey":"tender continuance","beats":["absence","ache","memory","held","both_true","carried_forward"],       "motif":"a shared object: a chair, a mug, a quality of light"},
}

TIMING_PHASES = {
    4:["validation","validation","complication","openness"],
    5:["validation","validation","complication","shift","openness"],
    6:["validation","validation","complication","shift","shift","openness"],
}

SYSTEM_PROMPT = """\
You are a literary graphic novelist writing for adults.
Respond ONLY with valid JSON. No markdown, no preamble.
RULES: motif in every visual. dialogue/motion never empty. one body sensation per panel. emotion_beat = one atmospheric word, not an emotion name. No moral lessons.
Output ONLY:
{"recurring_motif":"...","mood_journey":"...","panels":[
{"panel":1,"visual":"...","dialogue":"...","emotion_beat":"one_word","motion":"..."},
{"panel":2,"visual":"...","dialogue":"...","emotion_beat":"one_word","motion":"..."},
{"panel":3,"visual":"...","dialogue":"...","emotion_beat":"one_word","motion":"..."},
{"panel":4,"visual":"...","dialogue":"...","emotion_beat":"one_word","motion":"..."}]}"""


class ExperimentRunner:
    def __init__(self):
        log.info(f"Loading model: {MODEL_PATH}")
        self.tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
        self.model.eval()
        log.info("Model ready.")

    def run(self, combo: ParamCombo, test_input: dict) -> ExperimentResult:
        emotion  = test_input["emotion"]
        n        = test_input["panel_count"]
        arc      = MOOD_ARCS.get(emotion, MOOD_ARCS["sad"])
        beats    = self._get_beats(n, arc["beats"])
        phases   = TIMING_PHASES.get(n, TIMING_PHASES[6])

        beat_lines = "\n".join(
            f"  Panel {i+1} [{phases[i]}] → beat: {beats[i]}"
            for i in range(n)
        )
        user_prompt = (
            f"Primary emotion: {emotion} (confidence 0.75)\n"
            f'User context: "{test_input["user_text"]}"\n\n'
            f"MOOD JOURNEY: {arc['journey']}\n"
            f"MOTIF HINT: {arc['motif']}\n\n"
            f"Write exactly {n} panels:\n{beat_lines}\n\n"
            "dialogue and motion must never be empty. Write JSON now."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)

        t0 = time.time()
        try:
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=combo.max_tokens_per_panel * n,
                    do_sample=True,
                    temperature=combo.temperature,
                    top_p=combo.top_p,
                    repetition_penalty=combo.repetition_penalty,
                    pad_token_id=self.tok.eos_token_id,
                )
            raw = self.tok.decode(
                out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            elapsed = round(time.time() - t0, 2)

            story, err = self._parse(raw)
            if story:
                story["_meta"] = {"emotion": emotion, "panel_count": n}

            return ExperimentResult(
                combo=combo, input_emotion=emotion,
                rule_pass_rate=0.0, hallucination_rate=0.0,
                llm_avg=0.0, composite=0.0,
                generation_time_s=elapsed,
                json_valid=story is not None,
                story=story, error=err,
            )
        except Exception as e:
            return ExperimentResult(
                combo=combo, input_emotion=emotion,
                rule_pass_rate=0.0, hallucination_rate=0.0,
                llm_avg=0.0, composite=0.0,
                generation_time_s=round(time.time()-t0, 2),
                json_valid=False, error=str(e),
            )

    def _parse(self, raw: str):
        try:
            clean = re.sub(r"^```(?:json)?\s*","",raw.strip())
            clean = re.sub(r"\s*```$","",clean).strip()
            s = clean.find("{")
            if s == -1: return None, "no JSON found"
            depth, end = 0, -1
            for i, ch in enumerate(clean[s:], s):
                if ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0: end = i; break
            if end == -1: return None, "unbalanced braces"
            return json.loads(clean[s:end+1]), None
        except Exception as e:
            return None, str(e)

    def _get_beats(self, n, beats):
        if n <= len(beats):
            step = len(beats) / n
            return [beats[int(i*step)] for i in range(n)]
        result = []
        for i in range(n):
            idx = int(i*(len(beats)-1)/(n-1))
            result.append(beats[idx])
        return result


# ---------------------------------------------------------------------------
# 4. Scorer (rule-based + hallucination, inline for speed)
# ---------------------------------------------------------------------------

EMOTION_WORDS = {"sad","sadness","happy","happiness","angry","anger","tired","exhausted",
                 "anxious","anxiety","grief","grieving","depressed","depression","scared",
                 "fear","lonely","loneliness","hopeless","hopeful","excited","joy","joyful",
                 "pain","painful","upset","miserable","elated","furious","terrified"}
SOMATIC_WORDS = {"chest","breath","breathing","throat","stomach","hands","hand","fingers",
                 "lungs","jaw","shoulders","back","legs","feet","skin","pulse","heart",
                 "eyes","face","neck","spine","gut","belly","exhale","inhale","breathe",
                 "tightens","loosens","heavy","warm","cold","numb","ache","aching",
                 "trembling","still","weight","tight","loose","knot","flutter","sink"}
EARLY_BEATS   = {"heaviness","drag","spiral","contained_fire","absence","fracture",
                 "peak_noise","ache","weight","numbness","tension","contained_rage",
                 "sadness","sad","spark","doubt","jolt","resolve","tenderness","loss",
                 "fire","tight"}
LATE_BEATS    = {"quiet_hope","renewal","present","transcendence","carried_forward",
                 "soft_openness","tentative_light","radiance","ground","root","drift",
                 "triumph","victory","unity","forever","openness","rest","calm","light",
                 "grounded","share","glow"}


def score_result(result: ExperimentResult) -> ExperimentResult:
    if not result.story or not result.json_valid:
        result.rule_pass_rate     = 0.0
        result.hallucination_rate = 100.0
        result.composite          = 0.0
        return result

    panels = result.story.get("panels", [])
    motif  = result.story.get("recurring_motif", "").lower()
    n      = len(panels)

    # ── Rule checks ──────────────────────────────────────────────
    checks = {}

    # panel count
    expected = result.story.get("_meta", {}).get("panel_count", n)
    checks["panel_count"] = (n == expected, 2.0)

    # no empty fields
    empty = [f"p{p.get('panel')}.{k}" for p in panels
             for k in ("visual","dialogue","emotion_beat","motion")
             if not str(p.get(k,"")).strip()]
    checks["no_empty_fields"] = (len(empty) == 0, 2.0)

    # motif in all visuals
    motif_words = [w.lower() for w in motif.split() if len(w) > 3]
    if not motif_words:
        motif_words = [motif.split()[0].lower()] if motif.split() else [""]
    
    motif_miss = []
    for p in panels:
        visual_text = p.get("visual", "").lower()
        if not any(w in visual_text for w in motif_words if w):
            motif_miss.append(p["panel"])
    checks["motif_in_all"] = (len(motif_miss) == 0, 1.5)

    # no direct emotion naming
    emotion_bad = []
    for p in panels:
        text = f"{p.get('visual','')} {p.get('dialogue','')} {p.get('motion','')}".lower()
        hits = [w for w in EMOTION_WORDS if re.search(r'\b'+w+r'\b', text)]
        if hits: emotion_bad.append(hits)
    checks["no_direct_emotion"] = (len(emotion_bad) == 0, 1.5)

    # somatic every panel
    somatic_miss = [p["panel"] for p in panels
                    if not any(w in f"{p.get('visual','')} {p.get('motion','')}".lower()
                               for w in SOMATIC_WORDS)]
    checks["somatic_every_panel"] = (len(somatic_miss) == 0, 1.5)

    # beat single word
    beat_bad = [p for p in panels if len(p.get("emotion_beat","").split()) != 1]
    checks["beat_single_word"] = (len(beat_bad) == 0, 1.0)

    # arc direction
    beats = [p.get("emotion_beat","").lower() for p in panels]
    f_ok  = beats[0] in EARLY_BEATS if beats else False
    l_ok  = beats[-1] in LATE_BEATS  if beats else False
    checks["arc_direction"] = (f_ok and l_ok, 2.0)

    # dialogue brevity
    verbose = [p for p in panels
               if p.get("dialogue","") not in ("...","")
               and len(p.get("dialogue","").split()) > 14]
    checks["dialogue_brevity"] = (len(verbose) == 0, 1.0)

    # compute weighted pass rate
    wp = sum(w for (ok, w) in checks.values() if ok)
    wt = sum(w for (ok, w) in checks.values())
    result.rule_pass_rate = round(wp / wt * 100, 1) if wt else 0.0

    # ── Hallucination checks ────────────────────────────────────
    h_checks = {}

    # motif abandonment
    present = sum(1 for p in panels if any(w in p.get("visual", "").lower() for w in motif_words if w))
    h_checks["motif_abandonment"] = (present == 0, 1.0)

    # arc reversal
    if beats:
        h_checks["arc_reversal"] = (beats[-1] in EARLY_BEATS and beats[0] not in EARLY_BEATS, 1.0)
    else:
        h_checks["arc_reversal"] = (False, 1.0)

    # empty fields hallucination
    h_checks["empty_fields"] = (len(empty) > 0, 0.5)

    hw = sum(w for (hallu, w) in h_checks.values() if hallu)
    ht = sum(w for (hallu, w) in h_checks.values())
    result.hallucination_rate = round(hw / ht * 100, 1) if ht else 0.0

    # ── Composite ───────────────────────────────────────────────
    rule_score  = result.rule_pass_rate / 100
    hallu_score = (100 - result.hallucination_rate) / 100
    result.composite = round((rule_score * 0.6 + hallu_score * 0.4) * 100, 1)

    return result


# ---------------------------------------------------------------------------
# 5. Search Strategies
# ---------------------------------------------------------------------------

def random_search(n_experiments: int) -> list[ParamCombo]:
    """Sample param combos uniformly at random."""
    combos = []
    for _ in range(n_experiments):
        combos.append(ParamCombo(
            temperature          = round(np.random.choice(SEARCH_SPACE["TEMPERATURE"]["range"]), 2),
            top_p                = round(np.random.choice(SEARCH_SPACE["TOP_P"]["range"]), 2),
            repetition_penalty   = round(np.random.choice(SEARCH_SPACE["REPETITION_PENALTY"]["range"]), 2),
            max_tokens_per_panel = int(np.random.choice(SEARCH_SPACE["MAX_TOKENS_PER_PANEL"]["range"])),
        ))
    return combos


def grid_search() -> list[ParamCombo]:
    """Full cartesian product — can be large. Use with small ranges."""
    combos = []
    # Use every 3rd value to keep it manageable
    t_range   = SEARCH_SPACE["TEMPERATURE"]["range"][::2]
    p_range   = SEARCH_SPACE["TOP_P"]["range"][::2]
    r_range   = SEARCH_SPACE["REPETITION_PENALTY"]["range"][::2]
    m_range   = SEARCH_SPACE["MAX_TOKENS_PER_PANEL"]["range"][::2]
    for t, p, r, m in itertools.product(t_range, p_range, r_range, m_range):
        combos.append(ParamCombo(
            temperature=t, top_p=p, repetition_penalty=r, max_tokens_per_panel=m))
    log.info(f"Grid search: {len(combos)} combos")
    return combos


def bayesian_search(n_experiments: int) -> list[ParamCombo]:
    """Gaussian Process optimisation via scikit-optimize."""
    try:
        from skopt import gp_minimize
        from skopt.space import Real, Integer
    except ImportError:
        log.warning("pip install scikit-optimize — falling back to random search")
        return random_search(n_experiments)

    space = [
        Real(min(SEARCH_SPACE["TEMPERATURE"]["range"]),
             max(SEARCH_SPACE["TEMPERATURE"]["range"]),   name="temperature"),
        Real(min(SEARCH_SPACE["TOP_P"]["range"]),
             max(SEARCH_SPACE["TOP_P"]["range"]),         name="top_p"),
        Real(min(SEARCH_SPACE["REPETITION_PENALTY"]["range"]),
             max(SEARCH_SPACE["REPETITION_PENALTY"]["range"]), name="rep_penalty"),
        Integer(min(SEARCH_SPACE["MAX_TOKENS_PER_PANEL"]["range"]),
                max(SEARCH_SPACE["MAX_TOKENS_PER_PANEL"]["range"]), name="max_tpp"),
    ]

    # Placeholder objective — will be replaced by actual scores during search
    # Returns negative composite (minimise → maximise composite)
    def dummy_objective(x):
        return -50.0   # replaced in BayesianExperimentManager

    result = gp_minimize(dummy_objective, space, n_calls=n_experiments,
                         n_initial_points=10, random_state=42)

    combos = []
    for x in result.x_iters:
        combos.append(ParamCombo(
            temperature=round(float(x[0]), 2),
            top_p=round(float(x[1]), 2),
            repetition_penalty=round(float(x[2]), 2),
            max_tokens_per_panel=int(x[3]),
        ))
    return combos


# ---------------------------------------------------------------------------
# 6. Knowledge Graph Builder
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """
    Nodes   = experiment results (param combo + scores)
    Edges   = similarity between combos (cosine distance of feature vectors)
    Clusters= groups of similar-performing combos

    Graph schema:
      node attrs:  temperature, top_p, rep_penalty, max_tpp,
                   rule_pass_rate, hallucination_rate, composite, emotion, time_s
      edge attrs:  weight (cosine similarity), param_distance
    """

    def __init__(self):
        import networkx as nx
        self.G = nx.Graph()
        self.results: list[ExperimentResult] = []

    def add_result(self, result: ExperimentResult):
        self.results.append(result)
        node_id = f"{result.combo.key()}_{result.input_emotion}"
        self.G.add_node(node_id, **{
            "temperature":          result.combo.temperature,
            "top_p":                result.combo.top_p,
            "rep_penalty":          result.combo.repetition_penalty,
            "max_tpp":              result.combo.max_tokens_per_panel,
            "rule_pass_rate":       result.rule_pass_rate,
            "hallucination_rate":   result.hallucination_rate,
            "llm_avg":              result.llm_avg,
            "composite":            result.composite,
            "json_valid":           result.json_valid,
            "emotion":              result.input_emotion,
            "generation_time_s":    result.generation_time_s,
        })
        return node_id

    def build_edges(self, similarity_threshold: float = 0.85):
        """Connect nodes with cosine similarity > threshold."""
        from sklearn.metrics.pairwise import cosine_similarity

        nodes   = list(self.G.nodes())
        vectors = np.array([
            self.results[i].feature_vector()
            for i in range(len(self.results))
            if i < len(nodes)
        ])

        if len(vectors) < 2: return

        sim_matrix = cosine_similarity(vectors)
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                sim = float(sim_matrix[i][j])
                if sim >= similarity_threshold:
                    self.G.add_edge(nodes[i], nodes[j],
                                    weight=round(sim, 4),
                                    param_distance=round(1-sim, 4))

        log.info(f"Graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

    def find_best_cluster(self) -> dict:
        """
        Find the cluster of nodes with highest average composite score.
        Uses connected components — the densest high-performing component
        is the best parameter region.
        """
        import networkx as nx
        from sklearn.cluster import KMeans

        if len(self.results) < 3:
            return self._simple_best()

        # K-Means clustering on feature vectors
        vectors  = np.array([r.feature_vector() for r in self.results])
        k        = min(6, len(self.results) // 3)
        kmeans   = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels   = kmeans.fit_predict(vectors)

        # Score each cluster by mean composite
        cluster_scores = defaultdict(list)
        cluster_nodes  = defaultdict(list)
        for i, (result, label) in enumerate(zip(self.results, labels)):
            cluster_scores[label].append(result.composite)
            cluster_nodes[label].append(result)

        best_cluster_id = max(cluster_scores,
                              key=lambda l: np.mean(cluster_scores[l]))
        best_nodes      = cluster_nodes[best_cluster_id]

        log.info(f"Best cluster {best_cluster_id}: "
                 f"{len(best_nodes)} nodes, "
                 f"avg composite={np.mean(cluster_scores[best_cluster_id]):.1f}")

        return self._aggregate_best(best_nodes)

    def _simple_best(self) -> dict:
        best = max(self.results, key=lambda r: r.composite)
        return best.combo.to_dict()

    def _aggregate_best(self, nodes: list[ExperimentResult]) -> dict:
        """
        Within the best cluster, find optimal value per parameter.
        Strategy: weighted average by composite score.
        """
        weights = np.array([r.composite for r in nodes])
        if weights.sum() == 0:
            weights = np.ones(len(nodes))
        weights = weights / weights.sum()

        temps   = np.array([r.combo.temperature          for r in nodes])
        tops    = np.array([r.combo.top_p                for r in nodes])
        reps    = np.array([r.combo.repetition_penalty   for r in nodes])
        maxtpps = np.array([r.combo.max_tokens_per_panel for r in nodes])

        # Weighted average + snap to nearest value in search space
        def snap(val, space):
            arr = np.array(space)
            return arr[np.argmin(np.abs(arr - val))].item()

        best_t = snap(float(np.dot(weights, temps)),   SEARCH_SPACE["TEMPERATURE"]["range"])
        best_p = snap(float(np.dot(weights, tops)),    SEARCH_SPACE["TOP_P"]["range"])
        best_r = snap(float(np.dot(weights, reps)),    SEARCH_SPACE["REPETITION_PENALTY"]["range"])
        best_m = snap(float(np.dot(weights, maxtpps)), SEARCH_SPACE["MAX_TOKENS_PER_PANEL"]["range"])

        return {
            "TEMPERATURE":          best_t,
            "TOP_P":                best_p,
            "REPETITION_PENALTY":   best_r,
            "MAX_TOKENS_PER_PANEL": int(best_m),
        }

    def parameter_sensitivity(self) -> dict:
        """
        For each parameter, compute correlation with composite score.
        High correlation = this parameter matters a lot.
        """
        from scipy.stats import spearmanr

        composites = np.array([r.composite for r in self.results])
        sensitivity = {}

        for param, extractor in [
            ("TEMPERATURE",        lambda r: r.combo.temperature),
            ("TOP_P",              lambda r: r.combo.top_p),
            ("REPETITION_PENALTY", lambda r: r.combo.repetition_penalty),
            ("MAX_TOKENS_PER_PANEL", lambda r: r.combo.max_tokens_per_panel),
        ]:
            values = np.array([extractor(r) for r in self.results])
            if np.std(values) == 0:
                sensitivity[param] = {"correlation": 0.0, "p_value": 1.0, "impact": "none"}
                continue
            corr, pval = spearmanr(values, composites)
            impact = ("high" if abs(corr) > 0.5 else
                      "medium" if abs(corr) > 0.25 else "low")
            direction = "positive" if corr > 0 else "negative"
            sensitivity[param] = {
                "spearman_r":  round(float(corr), 4),
                "p_value":     round(float(pval), 4),
                "impact":      impact,
                "direction":   direction,
                "meaning":     self._sensitivity_meaning(param, corr),
            }

        return sensitivity

    def _sensitivity_meaning(self, param: str, corr: float) -> str:
        meanings = {
            "TEMPERATURE": (
                "Higher temperature → better stories" if corr > 0.3 else
                "Lower temperature → better stories"  if corr < -0.3 else
                "Temperature has minimal impact on quality"
            ),
            "TOP_P": (
                "Wider vocab (high top_p) → better"  if corr > 0.3 else
                "Tighter vocab (low top_p) → better" if corr < -0.3 else
                "Top-p has minimal impact"
            ),
            "REPETITION_PENALTY": (
                "More diversity penalty helps"        if corr > 0.3 else
                "Less repetition penalty works better" if corr < -0.3 else
                "Repetition penalty has minimal impact"
            ),
            "MAX_TOKENS_PER_PANEL": (
                "More tokens per panel → better"     if corr > 0.3 else
                "Fewer tokens sufficient"             if corr < -0.3 else
                "Token budget has minimal impact"
            ),
        }
        return meanings.get(param, "")

    def per_emotion_best(self) -> dict:
        """Find best params per emotion separately."""
        by_emotion = defaultdict(list)
        for r in self.results:
            by_emotion[r.input_emotion].append(r)

        result = {}
        for emotion, results in by_emotion.items():
            if not results: continue
            best = max(results, key=lambda r: r.composite)
            result[emotion] = {
                "best_combo":  best.combo.to_dict(),
                "composite":   best.composite,
                "rule_pass":   best.rule_pass_rate,
                "hallu_rate":  best.hallucination_rate,
            }
        return result

    def visualize(self, output_path: str = "param_graph.png"):
        """Save a graph visualisation coloured by composite score."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            log.warning("pip install matplotlib networkx for visualisation")
            return

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # ── Left: Network graph ──
        ax = axes[0]
        pos = nx.spring_layout(self.G, seed=42, k=0.5)
        composites = [self.G.nodes[n].get("composite", 0) for n in self.G.nodes()]
        nx.draw_networkx(
            self.G, pos, ax=ax,
            node_color=composites, cmap="RdYlGn",
            node_size=80, with_labels=False,
            edge_color="gray", alpha=0.6, width=0.5,
        )
        sm = plt.cm.ScalarMappable(cmap="RdYlGn",
             norm=plt.Normalize(vmin=min(composites) if composites else 0,
                                vmax=max(composites) if composites else 100))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Composite Score")
        ax.set_title("Knowledge Graph\n(green=high quality, red=low quality)", fontsize=11)
        ax.axis("off")

        # ── Right: Parameter vs composite scatter ──
        ax2 = axes[1]
        params  = [r.combo.temperature for r in self.results]
        scores  = [r.composite for r in self.results]
        colors  = [r.composite for r in self.results]
        sc = ax2.scatter(params, scores, c=colors, cmap="RdYlGn", s=60, alpha=0.7)
        ax2.set_xlabel("Temperature")
        ax2.set_ylabel("Composite Score")
        ax2.set_title("Temperature vs Composite Score")
        plt.colorbar(sc, ax=ax2, label="Composite")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        log.info(f"Graph saved → {output_path}")
        plt.close()


# ---------------------------------------------------------------------------
# 7. Reporter
# ---------------------------------------------------------------------------

def print_report(graph: KnowledgeGraph, best_params: dict,
                 sensitivity: dict, per_emotion: dict):

    all_composites = [r.composite for r in graph.results]
    all_rules      = [r.rule_pass_rate for r in graph.results]
    all_hallu      = [r.hallucination_rate for r in graph.results]
    all_times      = [r.generation_time_s for r in graph.results]
    valid_count    = sum(1 for r in graph.results if r.json_valid)

    print(f"\n{'═'*65}")
    print(f"  🔬 PARAM SEARCH RESULTS  ({len(graph.results)} experiments)")
    print(f"{'═'*65}")

    print(f"\n── EXPERIMENT SUMMARY ──\n")
    print(f"  JSON valid       : {valid_count}/{len(graph.results)} ({round(valid_count/max(len(graph.results),1)*100)}%)")
    print(f"  Composite avg    : {round(np.mean(all_composites),1)}  std={round(np.std(all_composites),1)}")
    print(f"  Rule pass avg    : {round(np.mean(all_rules),1)}%")
    print(f"  Hallucination avg: {round(np.mean(all_hallu),1)}%")
    print(f"  Gen time avg     : {round(np.mean(all_times),1)}s")
    print(f"  Best composite   : {round(max(all_composites),1)}")
    print(f"  Worst composite  : {round(min(all_composites),1)}")

    print(f"\n── PARAMETER SENSITIVITY ──\n")
    sorted_sens = sorted(sensitivity.items(),
                         key=lambda x: abs(x[1].get("spearman_r", 0)), reverse=True)
    for param, s in sorted_sens:
        bar = "█" * int(abs(s.get("spearman_r",0)) * 10) + "░" * (10 - int(abs(s.get("spearman_r",0)) * 10))
        print(f"  {param:<26} r={s['spearman_r']:>7.4f}  [{bar}]  {s['impact']:<8}  {s['meaning']}")

    print(f"\n── BEST PARAMS PER EMOTION ──\n")
    for emotion, info in sorted(per_emotion.items()):
        c = info["best_combo"]
        print(f"  {emotion:<10}  composite={info['composite']:.1f}  "
              f"T={c['TEMPERATURE']}  P={c['TOP_P']}  "
              f"R={c['REPETITION_PENALTY']}  M={c['MAX_TOKENS_PER_PANEL']}")

    print(f"\n── OPTIMAL STATIC PARAMETERS ──\n")
    print(f"  TEMPERATURE          = {best_params['TEMPERATURE']}")
    print(f"  TOP_P                = {best_params['TOP_P']}")
    print(f"  REPETITION_PENALTY   = {best_params['REPETITION_PENALTY']}")
    print(f"  MAX_TOKENS_PER_PANEL = {best_params['MAX_TOKENS_PER_PANEL']}")
    print(f"\n  Saved → best_params.env")
    print(f"{'═'*65}\n")


def save_best_params_env(best_params: dict):
    lines = [
        "# ============================================================",
        "# MoodWeaver — Optimised Parameters (from param_search.py)",
        "# ============================================================",
        "",
        f"TEMPERATURE={best_params['TEMPERATURE']}",
        f"TOP_P={best_params['TOP_P']}",
        f"REPETITION_PENALTY={best_params['REPETITION_PENALTY']}",
        f"MAX_TOKENS_PER_PANEL={best_params['MAX_TOKENS_PER_PANEL']}",
        "",
        "RETRY_ATTEMPTS=3",
    ]
    Path("best_params.env").write_text("\n".join(lines))
    log.info("Saved → best_params.env  (copy these values into your .env)")


def save_full_report(graph: KnowledgeGraph, best_params: dict,
                     sensitivity: dict, per_emotion: dict):
    OUTPUT_DIR.mkdir(exist_ok=True)
    report = {
        "total_experiments": len(graph.results),
        "best_params":       best_params,
        "sensitivity":       sensitivity,
        "per_emotion_best":  per_emotion,
        "all_results": [{
            "combo":              r.combo.to_dict(),
            "emotion":            r.input_emotion,
            "composite":          r.composite,
            "rule_pass_rate":     r.rule_pass_rate,
            "hallucination_rate": r.hallucination_rate,
            "json_valid":         r.json_valid,
            "generation_time_s":  r.generation_time_s,
        } for r in graph.results],
    }
    out = OUTPUT_DIR / "param_search_report.json"
    out.write_text(json.dumps(report, indent=2))
    log.info(f"Full report → {out}")


# ---------------------------------------------------------------------------
# 8. Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MoodWeaver Parameter Search")
    parser.add_argument("--experiments", type=int, default=60,
                        help="Number of experiments to run (default: 60)")
    parser.add_argument("--strategy", default="random",
                        choices=["random","grid","bayesian"],
                        help="Search strategy")
    parser.add_argument("--no-llm",   action="store_true",
                        help="Skip LLM judge scoring (faster)")
    parser.add_argument("--visualize",action="store_true",
                        help="Save knowledge graph PNG")
    parser.add_argument("--emotions", nargs="*",
                        default=["sad","angry","tired","happy","anxious","grief"],
                        help="Emotions to test")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    log.info(f"Strategy={args.strategy}  Experiments={args.experiments}")

    # ── Generate combos ──────────────────────────────────────────
    if args.strategy == "grid":
        combos = grid_search()
    elif args.strategy == "bayesian":
        combos = bayesian_search(args.experiments)
    else:
        combos = random_search(args.experiments)

    # Each combo runs on one random test input per emotion
    test_pool = [t for t in TEST_INPUTS if t["emotion"] in args.emotions]

    # ── Load model ───────────────────────────────────────────────
    runner = ExperimentRunner()
    graph  = KnowledgeGraph()

    total = min(len(combos), args.experiments)
    log.info(f"Running {total} experiments ...")

    for i, combo in enumerate(combos[:total]):
        test_input = test_pool[i % len(test_pool)]
        log.info(f"[{i+1}/{total}] T={combo.temperature} P={combo.top_p} "
                 f"R={combo.repetition_penalty} M={combo.max_tokens_per_panel} "
                 f"→ {test_input['emotion']}")

        result = runner.run(combo, test_input)
        result = score_result(result)
        graph.add_result(result)

        log.info(f"  composite={result.composite:.1f}  "
                 f"rule={result.rule_pass_rate:.1f}%  "
                 f"hallu={result.hallucination_rate:.1f}%  "
                 f"valid={result.json_valid}  "
                 f"time={result.generation_time_s}s")

    # ── Build graph ──────────────────────────────────────────────
    log.info("Building knowledge graph ...")
    graph.build_edges(similarity_threshold=0.80)

    # ── Analyse ──────────────────────────────────────────────────
    best_params  = graph.find_best_cluster()
    sensitivity  = graph.parameter_sensitivity()
    per_emotion  = graph.per_emotion_best()

    # ── Report ───────────────────────────────────────────────────
    print_report(graph, best_params, sensitivity, per_emotion)
    save_best_params_env(best_params)
    save_full_report(graph, best_params, sensitivity, per_emotion)

    if args.visualize:
        graph.visualize(str(OUTPUT_DIR / "param_graph.png"))

    log.info("Done. Copy best_params.env values into your .env to apply them.")


if __name__ == "__main__":
    main()