# Indie Comic Pipeline — Complete File Usage Guide

> **Every file. What it is. What it does. How to use it.**

---

## Table of Contents

1. [Project Architecture Overview](#1-project-architecture-overview)
2. [Root Files](#2-root-files)
3. [config/](#3-config)
4. [core/](#4-core)
   - [core/agents/](#41-coreagents)
   - [core/backends/](#42-corebackends)
5. [utils/](#5-utils)
6. [Legacy Pipeline Directories](#6-legacy-pipeline-directories)
7. [tests/](#7-tests)
8. [Other Directories](#8-other-directories)
9. [Data Flow Diagram](#9-data-flow-diagram)

---

## 1. Project Architecture Overview

The pipeline runs in **8 sequential phases**. Every phase has a responsible file.

```
Phase 0  core/story_intake.py           Raw prompt to structured panel outlines
Phase 1  core/agents/                   6 agents enrich each panel's context
Phase 2  core/anchoring.py              Extract visual identity from Panel 1
Phase 3  core/compositor.py             Compute per-panel generation weights
Phase 4  core/panel_engine.py           Generate each panel image (thread-safe)
Phase 5  core/text_image_integrator.py  Overlay speech bubbles and captions
Phase 6  core/quality_critic.py         Reject-and-regenerate quality gate
Phase 7  core/layout_engine.py          Assemble panels into comic pages
Phase 8  comic_exporter.py + feedback   Export and collect RLHF ratings
```

**Master orchestrator:** `integrated_pipeline.py`

**Shared brain (all phases read/write):** `core/memory.py` (`StorySectionMemory`)

---

## 2. Root Files

### `integrated_pipeline.py`
**The main entry point. Runs the entire 8-phase pipeline end-to-end.**

```python
from indie_comic_pipeline.integrated_pipeline import IntegratedComicPipeline
pipeline = IntegratedComicPipeline()
results = pipeline.run(
    prompt='A warrior discovers a hidden city',
    character_name='Kage',
    story_world='Neo-Tokyo Industrial Zone',
    panel_count=10,
    style='manga'
)
```

**Init parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_override` | str | None | Force a specific backend (`"sdxl"`, `"flux"`) |
| `skip_backends` | bool | False | Skip image generation (planning dry-run) |
| `dry_run` | bool | False | Run without loading GPU models |

**run() parameters:**

| Parameter | Type | Description |
|---|---|---|
| `prompt` | str | Story description in plain English |
| `character_name` | str | Main character name |
| `story_world` | str | Setting/world description |
| `panel_count` | int | Number of panels (1-10 recommended) |
| `style` | str | Art style hint (`"manga"`, `"comic"`) |

---

### `colab_setup.py`
**Cloud bootstrap script. Run this first in Kaggle/Colab.**

```python
exec(open('/kaggle/working/Indie-Comic/indie_comic_pipeline/colab_setup.py').read())
```

What it does:
- Clones the Git repo if not present, or pulls latest changes
- Installs all required packages from `requirements.txt`
- Sets up environment variables for the cloud runtime

---

### `comic_exporter.py`
**Converts finished panels into a distributable comic package.**

```python
from indie_comic_pipeline.comic_exporter import ComicExporter
exporter = ComicExporter(output_dir='outputs/comics')
exporter.export(panels=panel_images, title='Kages Journey', formats=['pdf', 'cbz', 'png'])
```

Supported formats: `PDF`, `CBZ` (comic book archive), `PNG sheet`, `JSON manifest`

---

### `compile_comic_pdf.py`
**Standalone PDF compiler for already-generated panel images.**

```bash
python compile_comic_pdf.py --input outputs/panels/ --output my_comic.pdf
```

Use when: panels were generated in a previous session, or you want to recompile layouts without regenerating.

---

### `audio_integration.py`
**Generates mood-matched ambient audio descriptions per panel.**

```python
from indie_comic_pipeline.audio_integration import AudioIntegrator
integrator = AudioIntegrator()
audio_map = integrator.generate_soundscape(arc_beats=['contained_fire', 'breakthrough'])
```

Not connected to the main pipeline by default. Call separately after generation.

---

### `model_comparator.py`
**Benchmarks multiple backends against the same prompt.**

```bash
python model_comparator.py --prompt 'warrior in rain' --backends sdxl flux --output comparisons/
```

Outputs: side-by-side image grid + timing/quality metrics CSV.

---

### `run_10_panel_pipeline.py`
**Quick-launch wrapper pre-configured for a standard 10-panel run.**

```bash
python run_10_panel_pipeline.py
python run_10_panel_pipeline.py --character Kage --world 'Neo-Tokyo' --emotion angry
```

Edit the `CONFIG` dict at the top of the file to customize defaults.

---

### `run_evaluation.py`
**Batch evaluation harness. Runs EvaluationSuite on existing outputs.**

```bash
python run_evaluation.py --panels outputs/panels/ --anchor outputs/anchors/anchor_panel_1.png
```

Output: JSON report with per-panel scores and consistency metrics.

---

### `generate_doodle_panels.py`
**Generates lightweight sketch placeholder panels using PIL. No GPU needed.**

```bash
python generate_doodle_panels.py --count 10 --output outputs/doodles/
```

Use for: layout previews, dry-run testing, debugging `layout_engine.py`.

---

### `generate_research_notebooks.py`
**Programmatically creates Jupyter notebooks for experiments.**

```bash
python generate_research_notebooks.py --type model_comparison --output notebooks/
```

Types: `model_comparison`, `arc_evaluation`, `parameter_sweep`, `consistency_analysis`

---

### `install_all.py`
Simple dependency installer. Equivalent to `pip install -r requirements.txt`.

---

### `requirements.txt` / `requirements_colab.txt`

| File | When to use |
|---|---|
| `requirements.txt` | Local GPU machine or full Kaggle environment |
| `requirements_colab.txt` | Cloud runtimes — only installs missing packages |

---

## 3. `config/`

### `settings.yaml`
**Master runtime configuration. Controls all pipeline behavior.**

```yaml
outputs:
  panels_dir: 'outputs/panels'
  comics_dir: 'outputs/comics'
  anchors_dir: 'outputs/anchors'

generation:
  guidance_scale: 7.5
  num_steps: 25
  lora_scale: 0.8
  width: 768
  height: 768

quality_critic:
  threshold: 0.55
  strict_threshold: 0.70
  max_retries: 2

layout:
  page_width: 1000
  page_height: 1500
  gutter_width: 12

llm_provider: 'ollama'
ollama_url: 'http://localhost:11434'
```

---

### `model_paths.yaml`
**Maps backend names to HuggingFace model IDs or local paths.**

```yaml
sdxl:
  name: 'Lykon/DreamShaper-XL-1-0'
  device: 'cuda'
  cpu_offload: true

lora:
  name: 'artificialguybr/LineAniRedmond-LinearMangaSDXL-V2'
  adapter_scale: 0.8

flux:
  enabled: false
  name: 'black-forest-labs/FLUX.1-dev'
```

Edit to switch models without changing any Python code.

---

### `agents.json`
**Registers the 6 planning agents for dynamic loading.**
Add new agents here to have them auto-discovered by `AgentCoordinator`.

Registered agents: `story_director`, `action_director`, `dialogue_writer`,
`pose_director`, `emotion_director`, `camera_director`

---

### `arcs_config.json`
**Defines the emotional arc library with beat sequences.**

Structure per arc entry:
- `phases[]` — ordered emotion beat names
- `secondary_emotions` — supporting descriptors
- `recurring_motif` — visual symbol for the arc

Used by `story_intake.py` and `Story-Weaver/story_gen.py`.

---

## 4. `core/`

### `story_intake.py` — Phase 0
**Accepts raw user prompt, outputs structured panel outline list.**

```python
from indie_comic_pipeline.core.story_intake import StoryIntakeEngine
engine = StoryIntakeEngine()
story_config = engine.process(
    prompt='A tired soldier finds peace in a ruined garden',
    character_name='Ren', story_world='Post-war ruins', panel_count=6
)
# Returns: dict with panels[], characters[], arc_beats[], mood_journey
```

**Internal sequence:**
1. BERT emotion detection via `mood-weaver`
2. LLM storyboard generation via Story-Weaver (Ollama/OpenAI/Gemini/Anthropic)
3. Template-based fallback if LLM fails

**Key Enhancement:** The LLM system prompt enforces a **5-layer cinematic action structure**:
- `verb` — powerful, specific action verb
- `target` — what/whom with physical detail
- `mechanics` — exact body-part positions under tension
- `impact` — what physically happens at contact/consequence
- `reaction` — how the environment responds
- `timing` — freeze-frame moment label (e.g., "maximum-force impact")

---

### `memory.py` — Shared Blackboard
**Central state store. All phases read/write through `StorySectionMemory`.**

```python
from indie_comic_pipeline.core.memory import StorySectionMemory
memory = StorySectionMemory()

char = memory.register_character('Kage')
char.costume_desc = 'black tactical suit, red scarf'

memory.record_panel(panel_id=1, prompt='...', emotion='angry',
                    image_path='outputs/panels/panel_001.png')

beats = memory.arc_beats              # ['contained_fire', 'fracture', ...]
anchor = memory.get_anchor_features() # set after Phase 2
```

**Key dataclasses:**

| Class | Key Fields |
|---|---|
| `CharacterState` | `name, emotion, costume_desc, last_action, arc_phase, identity_tokens` |
| `SceneState` | `location, time_of_day, dominant_palette` |
| `PanelRecord` | `panel_id, page_num, prompt_used, emotion, image_path, action_intensity` |
| `LayoutDirective` | `panel_id, size_class, camera_angle, aspect_ratio, gutter_emphasis` |

> ⚠️ **Validation Limitation:** The `BlackboardValidatorMixin.__setattr__` interceptor protects
> top-level attribute assignment (`memory.x = y`). It does **not** recursively validate mutations
> on nested mutable structures. For example:
> ```python
> # SAFE — triggers __setattr__, type is checked
> memory.scene.time_of_day = "night"
>
> # UNSAFE — mutates the list in-place, __setattr__ is never called, no validation
> memory.scene_props.append("new_lighting")
> char.panel_appearances.append("not_an_int")  # should be List[int], but passes silently
> ```
> **Future work:** proxy nested containers (e.g. `TypedList`, `__setitem__` overrides) to enforce
> deep type-safety. Until then, always assign a new list/dict rather than mutating in-place when
> type correctness matters.

---

### `anchoring.py` — Phase 2
**Extracts visual identity of Panel 1 and injects it into memory.**

```python
from indie_comic_pipeline.core.anchoring import ReferenceFreeAnchor
anchor_system = ReferenceFreeAnchor()

tokens = anchor_system.establish_anchor(
    panel_image=pil_image, panel_id=1, character_name='Kage', memory=memory
)
# tokens: {color_profile, edge_profile, style_profile, aesthetic_score}

guidance = anchor_system.get_consistency_guidance(memory)
# guidance: {prompt_suffix, negative_augment, guidance_scale_adjust}
```

---

### `compositor.py` — Phase 3
**Calculates dynamic generation hyperparameters per panel.**

```python
from indie_comic_pipeline.core.compositor import CharComCompositor
compositor = CharComCompositor(base_lora_scale=0.8, base_guidance=7.5, base_steps=25)
weights = compositor.compute_weights(context={
    'panel_id': 3, 'panel_emotion_beat': 'breakthrough',
    'layout': {'size_class': 'full_page', 'camera_angle': 'low_angle'},
    'has_anchor': True, 'total_panels': 10
})
# weights: {'lora_scale': 0.85, 'guidance_scale': 8.75, 'num_steps': 33}
```

**Adjustment rules:**

| Rule | Effect |
|---|---|
| `full_page` panel | +0.5 guidance, +5 steps |
| `large` panel | +0.25 guidance, +2 steps |
| High emotion beat (fracture, overflow, ache) | +0.5 guidance, +0.05 LoRA |
| Low emotion beat (stillness, drift, quiet_rest) | -0.25 guidance, -0.05 LoRA |
| Anchor established + panel_id > 1 | +0.25 guidance |
| First or last panel | +3 steps |
| All values clamped | guidance 5-12, lora 0.3-1.0, steps 15-50 |

---

### `panel_engine.py` — Phase 4
**Thread-safe single-panel image generator with 5-layer prompt assembly.**

```python
from indie_comic_pipeline.core.panel_engine import PanelEngine
engine = PanelEngine(backend_selector=selector, memory=memory,
                     attention_manager=attn_manager, anchor=anchor_system,
                     compositor=compositor)
result = engine.generate_panel(panel_id=1, story_config=story_config)
# result: {panel_id, image (PIL), image_path, prompt, weights}
```

**Internal sequence per panel:**
1. Get context from `AgentCoordinator`
2. Compute weights via `CharComCompositor`
3. Build full prompt string from context fields:
   - **Action block** — stitches `verb + target + mechanics + impact + reaction + timing` into a ~50-word cinematic clause
   - **Character block** — includes costume + pose
   - **Environment block** — location, time, palette, light source
   - **Camera block** — angle + movement
4. Select backend via `BackendSelector`
5. Call `AdvancedAttentionManager.on_panel_start()`
6. Call `backend.generate()` inside **`threading.Lock()`** (prevents GPU race conditions)
7. If Panel 1: run `ReferenceFreeAnchor.establish_anchor()`
8. Record result in `StorySectionMemory`

**Prompt Assembly Example:**
```python
# Final prompt sent to SDXL:
"Kage slams his fist into the robot's chest plate. Mechanics: entire torso twisting, feet planted, arm cocked back past the ear. Impact: spiderweb cracks radiating outward across the metal. Reaction: neon sparks and debris explode in all directions. Timing: maximum-force impact freeze-frame. Masterpiece, manga, dramatic low-angle lighting."
```

---

### `advanced_attention.py` — Phases 3-4
**3-layer denoising-time visual consistency system.**

```python
from indie_comic_pipeline.core.advanced_attention import AdvancedAttentionManager
manager = AdvancedAttentionManager(heat_alpha=0.03, attention_blend=0.15, spatial_strength=0.08)
manager.install_on_pipeline(sdxl_pipe)

manager.on_panel_start(panel_id=1, is_anchor=True, total_steps=25)
callback = manager.get_step_callback()
# pass to diffusers: pipe(..., callback_on_step_end=callback)
manager.on_panel_end()
```

| Layer | Class | Active Window | Effect |
|---|---|---|---|
| L1 Heat Diffusion | `HeatDiffusionPrior` | 20%-80% of steps | Gaussian blur suppresses noise drift |
| L2 Attention Cache | `SharedAttentionCache` | Every UNet forward | Blends 15% anchor K/V into cross-attention |
| L3 Spatiotemporal | `SpatiotemporalConsistencyEnforcer` | 30%-60% of steps | Corrects latent channel mean/std to anchor |

---

### `text_image_integrator.py` — Phase 5
**Overlays dialogue bubbles, captions, and SFX text onto panel images.**

```python
from indie_comic_pipeline.core.text_image_integrator import TextImageIntegrator
integrator = TextImageIntegrator()
annotated = integrator.integrate(
    image=pil_panel_image,
    panel_data={
        'dialogue': [{'speaker': 'Kage', 'text': 'Move.', 'style': 'speech'}],
        'caption': 'Neo-Tokyo, 2087',
        'sfx': ['CRASH']
    }
)
```

Text types: `speech` (rounded bubble), `thought` (cloud), `caption` (rectangle), `sfx` (large stylized)

---

### `quality_critic.py` — Phase 6
**5-dimension quality gate with reject-and-regenerate.**

```python
from indie_comic_pipeline.core.quality_critic import QualityCritic
critic = QualityCritic(threshold=0.55, strict_threshold=0.70, max_retries=2)
evaluation = critic.evaluate(panel_result=result, memory=memory)
# evaluation: {panel_id, scores, composite_score, verdict, adjustments}
if critic.should_regenerate(evaluation):
    pass  # use evaluation['adjustments'] to modify params and retry
```

**Scoring dimensions:**

| Dimension | Weight | How Measured |
|---|---|---|
| Visual Consistency | 30% | SSIM vs anchor via ConsistencyChecker |
| Aesthetic Quality | 25% | Resolution x pixel variance |
| Narrative Coherence | 20% | Arc beat progress tracking |
| Emotional Engagement | 15% | Beat in high-engagement set |
| Readability | 10% | Edge density sweet-spot (0.05-0.3) |

**Verdicts:** composite >= 0.70 = excellent, >= 0.55 = pass, below = fail

When `UserPreferenceCritic` is trained, a 6th `user_preference` dimension (20%) is added and other weights are scaled to 80%.

---

### `layout_engine.py` — Phase 7
**Assembles panels into full comic pages.**

```python
from indie_comic_pipeline.core.layout_engine import MangaFlowLayoutEngine
engine = MangaFlowLayoutEngine(page_width=1000, page_height=1500, gutter_width=12, margin=40)
page_image = engine.layout_page(
    panels=[panel_dict_1, panel_dict_2, panel_dict_3], page_num=1
)
# Returns 1000x1500 PIL.Image
```

Panel dicts: `{'image': PIL.Image, 'action_intensity': 0.0-1.0}`
Higher `action_intensity` = more canvas space. First and last panels get priority sizing.

---

### `feedback.py` — Phase 8
**RLHF telemetry. Collects and persists user ratings.**

```python
from indie_comic_pipeline.core.feedback import RLHFFeedbackLoop
feedback = RLHFFeedbackLoop(feedback_path='outputs/rlhf_feedback.json')

feedback.add_panel_feedback(panel_id=3, rating=4, comment='Great composition',
                            engagement_time=12.5, generation_backend='sdxl')
feedback.add_page_feedback(page_num=1, rating=5)
summary = feedback.get_feedback_summary()
# {'total_panels_rated': 10, 'average_panel_rating': 3.8, 'backend_performances': {'sdxl': 3.9}}
```

---

### `feedback_tuner.py` — Phase 8
**Reads RLHF logs and adjusts `settings.yaml` automatically.**

```python
from indie_comic_pipeline.core.feedback_tuner import HeuristicFeedbackTuner
tuner = HeuristicFeedbackTuner(feedback_loop=feedback, settings_path='config/settings.yaml')
adjustments = tuner.tune_from_feedback()
tuner.apply_adjustments(adjustments)  # file-locked write to settings.yaml
```

Uses `msvcrt` locking on Windows, `fcntl` on Linux for safe concurrent writes.

---

### `user_preference_critic.py`
**Learns personal taste from RLHF ratings using CLIP + linear regression.**

Architecture: `CLIP ViT-B/32 (frozen) -> 512-dim embedding -> Linear(512->1) -> Sigmoid`

```python
from indie_comic_pipeline.core.user_preference_critic import UserPreferenceCritic
critic = UserPreferenceCritic(model_path='outputs/user_preference_model.pt')

success = critic.train_from_feedback_file(
    feedback_file='outputs/rlhf_feedback.json',
    panels_dir='outputs/panels', epochs=50
)  # needs >= 3 records

if critic.is_trained():
    score = critic.predict(pil_image)  # returns 0.0-1.0
```

---

### `evaluation_suite.py`
**Full quality audit using CLIP, DINOv2, and FID.**

```python
from indie_comic_pipeline.core.evaluation_suite import ModelEvaluator
evaluator = ModelEvaluator(device='cuda')
clip_score = evaluator.compute_clip_score(image=pil_img, text='warrior charges forward')
dino_sim   = evaluator.compute_dinov2_similarity(image1=panel_1, image2=panel_2)
fid        = evaluator.compute_fid(generated_img=panel, reference_img=anchor)
```

Used by `run_evaluation.py` for batch post-generation quality reports.

---

## 4.1 `core/agents/`

### `base_agent.py`
**Abstract base class. Inherit this to create a custom planning agent.**

```python
from indie_comic_pipeline.core.agents.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__('my_agent')   # self.log logger is auto-set

    def plan(self, story_config, memory):
        return {'status': 'done'}

    def update(self, panel_result, memory):
        pass  # called after each panel is generated
```

---

### `agent_coordinator.py`
**Loads and runs all agents in order. Provides per-panel context.**

```python
from indie_comic_pipeline.core.agents.agent_coordinator import AgentCoordinator
coordinator = AgentCoordinator(memory=memory, agent_config_path='config/agents.json')
coordinator.run_all(story_config=story_config)  # runs all 6 agents in sequence
context = coordinator.get_generation_context(panel_id=3)
# context: {panel_id, panel_emotion_beat, layout, characters, actions, dialogue,
#           poses, has_anchor, total_panels}
```

---

### `director_swarm.py`
**All 6 specialized planning agents defined in one file.**

| Agent | Writes to memory | Enrichment Behavior |
|---|---|---|
| `StoryDirector` | `raw_panels`, `total_panels`, all character registrations | Creates base panel structure |
| `ActionDirector` | Fills missing `actions[]` per panel | **ENRICHES**—only fills missing fields from `ACTION_EXAGGERATION_MAP`; preserves LLM's rich `mechanics`, `impact`, `reaction`, `timing` if present |
| `DialogueWriter` | Fills dialogue text (Ollama LLM call with fallback dict) | Adds dialogue if missing |
| `PoseDirector` | Fills `pose{}` and `expression{}` per character from beat maps | Adds pose if missing |
| `EmotionDirector` | Sets `arc_beats[]`, advances `current_beat_index` after each panel | Sets emotional progression |
| `CameraDirector` | Assigns camera angle + creates `LayoutDirective` per panel | Adds camera if missing |

> ⚡ **Critical Design Note:** The `ActionDirector` uses a **merge/enrichment pattern**, not overwrite.
> If the LLM from Phase 0 has already provided `mechanics`, `impact`, `reaction`, and `timing`,
> the ActionDirector preserves them. It only supplies fallback values from its internal
> `ACTION_EXAGGERATION_MAP` when fields are missing or empty. This ensures cinematic,
> 5-layer action descriptions flow from Phase 0 → Phase 1 → Phase 4 without being truncated.

**Action Keys (Standardized across all components):**

| Key | Type | Description | Example |
|---|---|---|---|
| `verb` | string | Powerful, specific action verb | `"slam"`, `"hurtles"`, `"shatters"` |
| `target` | string | What/whom the action affects | `"the robot's chest plate"` |
| `mechanics` | string | Exact body-part positions under tension | `"spine horizontal, knuckles grazing ground"` |
| `impact` | string | Physical consequence at the point of contact | `"spiderweb cracks radiating outward"` |
| `reaction` | string | How the environment responds | `"dust clouds billowing, neon sparks exploding"` |
| `timing` | string | Freeze-frame moment label | `"maximum-force impact freeze-frame"` |

**Emotion beat to camera lookup (`_BEAT_CAMERA_MAP`):**

| Beat | Camera Angle |
|---|---|
| `fracture`, `spiral` | `dutch_tilt` |
| `breakthrough`, `contained_fire`, `momentum` | `low_angle` |
| `triumph`, `stillness`, `absence` | `wide_shot` |
| `peak_noise`, `ache`, `spark`, `vulnerability` | `close_up` |
| `quiet_rest`, `transcendence` | `bird_eye` |
| most others | `medium_shot` |

---

## 4.2 `core/backends/`

### `base_backend.py`
**Abstract interface every generation backend must implement.**

```python
from indie_comic_pipeline.core.backends.base_backend import BaseBackend
from PIL import Image

class MyModelBackend(BaseBackend):
    @property
    def name(self): return 'MyModel'

    @property
    def supports_lora(self): return False

    def load(self, config): ...
    def generate(self, prompt, negative_prompt, config) -> Image.Image: ...
    def unload(self): ...
    def is_loaded(self) -> bool: ...
```

**Optional overrides:**
- `get_cross_attention_modules()` — list of UNet attn2 layers for L2 hook installation
- `get_vram_estimate_mb()` — VRAM footprint for memory budgeting

---

### `backend_selector.py`
**Dynamic backend router with lazy weight loading.**

```python
from indie_comic_pipeline.core.backends.backend_selector import BackendSelector
selector = BackendSelector()
selector.initialize_backends(model_config={'sdxl': {...}, 'lora': {...}})
backend = selector.select(context={'layout': {'size_class': 'large', 'camera_angle': 'close_up'}})
selector.unload_all()
```

**Selection rules:**

| size_class | camera_angle | Backend |
|---|---|---|
| `full_page` | `bird_eye`, `wide_shot` | `flux` |
| `large` or `medium` | any | `sdxl` |
| `small` | any | `sdxl` |
| Flux not registered | — | fallback to `sdxl` |

Model weights load **lazily on first select()** — not at startup.

---

### `sdxl_backend.py`
**Primary generation backend. SDXL + LoRA + Compel + AdvAttn.**

```python
from indie_comic_pipeline.core.backends.sdxl_backend import SDXLBackend
backend = SDXLBackend()
backend.load({
    'model_name': 'Lykon/DreamShaper-XL-1-0',
    'lora_name': 'artificialguybr/LineAniRedmond-LinearMangaSDXL-V2',
    'lora_scale': 0.8, 'device': 'cuda', 'enable_cpu_offload': True
})
image = backend.generate(
    prompt='manga style, Kage charges forward',
    negative_prompt='blurry, deformed',
    config={'width': 768, 'height': 768, 'num_steps': 28,
            'guidance_scale': 8.5, 'seed': 42,
            'step_callback': attn_manager.get_step_callback()}
)
modules = backend.get_cross_attention_modules()  # for L2 hook installation
backend.unload()  # frees ~6.5 GB VRAM
```

**`load()` sequence:** device -> SDXL fp16 -> DPMSolver++ -> CPU offload -> attn slicing -> VAE slicing -> safety checker removal -> LoRA

**`generate()` sequence:** gen_kwargs -> Compel encoding -> LoRA scale -> AdvAttn callback -> torch.inference_mode() -> VRAM clear

---

### `flux_backend.py`
**Stub backend. Delegates to SDXL with enhanced settings.**

Auto-enhancements: `num_steps >= 30`, `guidance_scale >= 8.0`, prepends `'masterpiece, best quality,'`

When real Flux weights are integrated, only this file changes.

---

## 5. `utils/`

### `consistency_checker.py`
**Pixel-level visual consistency metrics between panels.**

```python
from indie_comic_pipeline.utils.consistency_checker import ConsistencyChecker
checker = ConsistencyChecker()
checker.set_reference_from_panel('outputs/anchors/anchor_panel_1.png')
result = checker.check_consistency('outputs/panels/panel_003.png')
# result: {'score': 0.78, 'ssim': 0.82, 'color_distance': 0.15, 'edge_similarity': 0.71}
features = checker.extract_features('outputs/panels/panel_003.png')
```

**Metrics:** SSIM, HSV color histogram, Sobel edge density, Gram matrix style fingerprint, CLIP cosine similarity (GPU), DINOv2 cosine similarity (GPU)

---

### `bridge_weaver.py`
**Translates between pipeline panel format and Story-Weaver schema.**
Used internally by `story_intake.py`. Not needed to call directly.

```python
from indie_comic_pipeline.utils.bridge_weaver import BridgeWeaver
bridge = BridgeWeaver()
sw_input      = bridge.to_story_weaver(story_config)
pipeline_data = bridge.from_story_weaver(story_weaver_output)
```

---

### `config_helper.py`
**Unified settings and environment loader.**

```python
from indie_comic_pipeline.utils.config_helper import load_settings, load_env_with_defaults, get_output_path
settings = load_settings('config/settings.yaml')         # YAML -> dict
env      = load_env_with_defaults()                      # .env + defaults
path     = get_output_path('outputs/panels', 'panel_001_final.png')  # mkdir + join
```

---

### `image_utils.py`
**PIL image manipulation helpers.**

```python
from indie_comic_pipeline.utils.image_utils import create_comic_strip, create_comic_grid, pad_to_size
strip  = create_comic_strip(images=[img1, img2, img3], gutter=12)
grid   = create_comic_grid(images=panels, cols=2, gutter=12, bg_color='white')
padded = pad_to_size(image=img, width=768, height=768, color='white')
```

---

### `prompt_optimizer.py`
**Style-specific prompt rewriter.**

```python
from indie_comic_pipeline.utils.prompt_optimizer import get_prompt_optimizer
optimizer = get_prompt_optimizer(style='manga')
enhanced_prompt, enhanced_negative = optimizer.optimize(
    prompt='Kage runs through rain', emotion_beat='momentum', negative_prompt='blurry'
)
```

---

## 6. Legacy Pipeline Directories

> These directories are preserved legacy code from earlier development stages.
> They still work independently but are **NOT used** by `integrated_pipeline.py`.

### Evolution Timeline

```
Gen 1  langchain_code/   LLM text extraction only (no image generation)
Gen 2  sdxl_code/        Added base SDXL image generation (~8-10 GB VRAM)
       sd15_code/        SD 1.5 lighter fallback variant (~6-8 GB VRAM)
Gen 3  lora_code/        Added LineAniRedmond manga LoRA (~11-12 GB VRAM)
Gen 4  integrated_pipeline.py + core/    <-- CURRENT UNIFIED SYSTEM
```

---

### 6.1 `langchain_code/` — Generation 1
**Original LLM extraction layer. Text-only. No image generation.**

```bash
cd indie_comic_pipeline/langchain_code
python run_full_pipeline.py
# Runs in order: character_extractor -> story_extractor -> fusion_engine
```

| File | What it does | Output |
|---|---|---|
| `character_extractor.py` | Extracts personality + visual markers via Ollama LangChain | `outputs/fusion/character_personality.json` |
| `story_extractor.py` | Extracts world-building: lighting, atmosphere, color palette | `outputs/fusion/story_setting.json` |
| `fusion_engine.py` | Fuses both JSONs into 4-panel storyboard + SDXL prompts | `outputs/fusion/sdxl_prompt.json` |
| `emotion_recognition_engine.py` | Keyword-based emotion detector (pre-dates mood-weaver BERT) | Used internally by fusion |
| `story_weaver_enricher.py` | Enriches Story-Weaver output with side chars + detailed prompts | `outputs/fusion/enriched_storyboard.json` |

> `sdxl_prompt.json` must exist before running `sdxl_code/`, `sd15_code/`, or `lora_code/`.

---

### 6.2 `sdxl_code/` — Generation 2
**Base SDXL image generation. Requires langchain_code to run first.**

```bash
cd indie_comic_pipeline/sdxl_code
python run_sdxl_pipeline.py    # VRAM: ~8-10 GB
```

| File | What it does | Output |
|---|---|---|
| `generate_character.py` | Single character reference image (no LoRA) | `outputs/characters/character_reference.png` |
| `generate_components.py` | Background / environment / prop components | `outputs/comics/component_sdxl_base_*.png` |
| `generate_panels.py` | 4-panel storyboard with `_PIPE_CACHE` for model reuse | `outputs/comics/panel_sdxl_base_*.png` |

---

### 6.3 `sd15_code/` — Generation 2b
**Stable Diffusion 1.5 — faster, lighter, lower quality.**

```bash
cd indie_comic_pipeline/sd15_code
python run_sd15_pipeline.py    # VRAM: ~6-8 GB
```

Same file structure as `sdxl_code/` with SD 1.5 pipeline swapped in.

| | SD 1.5 | SDXL Base |
|---|---|---|
| VRAM | ~6-8 GB | ~8-10 GB |
| Native resolution | 512x512 | 1024x1024 |
| Speed | Faster | Slower |
| Quality | Lower | Higher |

Use when: limited VRAM, fast iteration, or quality is not the priority.

---

### 6.4 `lora_code/` — Generation 3
**SDXL + LineAniRedmond manga LoRA. Best standalone option before Gen 4.**

```bash
cd indie_comic_pipeline/lora_code
python run_lora_pipeline.py    # VRAM: ~11-12 GB
```

| File | What it does | Output |
|---|---|---|
| `generate_character.py` | Character reference with manga LoRA applied | `outputs/characters/character_reference_sdxl_lora.png` |
| `generate_components.py` | Manga-style environment / background components | `outputs/comics/component_sdxl_lora_*.png` |
| `generate_panels.py` | Emotion-aware panel generator (472 lines) | `outputs/comics/panel_lora_*.png` |

`generate_panels.py` notable features:
- `--page N` argument to generate a specific storyboard page
- `--force_reload` to bypass model cache
- Inline `ConsistencyChecker` calls between panels (predecessor of `anchoring.py`)
- Emotion-aware prompt building (predecessor of `director_swarm.py`)

---

## 7. `tests/`

### `run_unit_tests.py`
**Master test runner for all 7 test specifications.**

```bash
python tests/run_unit_tests.py
python tests/run_unit_tests.py --test memory
python tests/run_unit_tests.py --test quality_critic
```

| Test | What it validates |
|---|---|
| `StorySectionMemory` | Blackboard read/write, type validation, serialization |
| `AdvancedAttentionManager` | Hook installation, step callback, mock mode on CPU |
| `AgentCoordinator` | Agent registration, plan/update cycle |
| `MangaFlowLayoutEngine` | Page assembly, panel sizing, gutter application |
| `QualityCritic` | Score computation, threshold logic, adjustment generation |
| `ComicExporter` | PDF/CBZ/PNG export without GPU |
| `Feedback & Optimizer` | JSON persistence, tuner delta computation |

---

## 8. Other Directories

| Directory | Purpose |
|---|---|
| `prompts/` | Style presets, character templates, negative word bank, beat prompt fragments |
| `web_interface/` | Early Flask web UI prototype (`python app.py` -> localhost:5000) |
| `matrix_evaluation_zone/` | Multi-axis quality rubric evaluation for comparing comics |
| `scratch/` | Developer scratch space for one-off scripts and temp files |
| `outputs/` (generated) | `panels/`, `anchors/`, `comics/`, `characters/`, `fusion/`, `rlhf_feedback.json` |

---

## 9. Data Flow Diagram

```
User Input (prompt, character_name, story_world, panel_count)
         |
         v
 [story_intake.py]                            Phase 0
   BERT emotion detection  -> detects arc type (angry/happy/sad...)
   Story-Weaver LLM        -> generates panel outlines with 5-layer actions:
                             verb, target, mechanics, impact, reaction, timing
         |
         | story_config (panels[], arc_beats[], characters[])
         v
 [agent_coordinator.py]                       Phase 1
   StoryDirector     -> writes raw_panels, registers all characters
   ActionDirector    -> ENRICHES actions[] - preserves LLM's 5-layer structure,
                        only fills missing fields from ACTION_EXAGGERATION_MAP
   DialogueWriter    -> fills dialogue text (Ollama LLM + fallback dict)
   PoseDirector      -> fills pose{} and expression{} from beat maps
   EmotionDirector   -> sets arc_beats[] in memory
   CameraDirector    -> assigns LayoutDirective per panel
         |
         | StorySectionMemory (fully populated, with rich 5-layer actions preserved)
         v
 [panel_engine.py]                            Phase 2-4 (ThreadPoolExecutor)

   Panel 1 - ANCHOR:
     compositor.compute_weights()
     backend_selector.select()  ->  sdxl_backend
     advanced_attention.on_panel_start(anchor=True)
     BUILD PROMPT: stitches action.verb + target + mechanics + impact + reaction + timing
                  into a ~50-word cinematic clause
     sdxl_backend.generate()   [threading.Lock]
     anchoring.establish_anchor()  -> identity tokens -> memory

   Panels 2-N - CONSISTENCY:
     compositor.compute_weights()  (+0.25 guidance for anchor)
     backend_selector.select()  ->  sdxl_backend
     advanced_attention.on_panel_start(anchor=False)
       -> L1 heat diffusion on every denoising step (20-80%)
       -> L2 anchor K/V blend in cross-attention (every forward pass)
       -> L3 latent stat correction (30-60% of steps)
     BUILD PROMPT: same 5-layer cinematic assembly as Panel 1
     sdxl_backend.generate()   [threading.Lock]
         |
         | panel_result (image, prompt, weights, path)
         v
 [text_image_integrator.py]                  Phase 5
   Add speech bubbles, thought clouds, captions, SFX text
         v
 [quality_critic.py]                          Phase 6
   Score 5 dimensions -> composite score
   composite < 0.55:  regenerate with adjustments (max 2 retries)
         v
 [layout_engine.py]                           Phase 7
   MangaFlowLayoutEngine
   Dynamic panel sizing by action_intensity + emotion beat
   Assemble into 1000x1500 pages with 12px gutters
         v
 [comic_exporter.py]                          Phase 8a
   Export: PDF, CBZ, PNG sheet, JSON manifest

 [feedback.py]                                Phase 8b
   Collect 1-5 star ratings per panel and page

 [feedback_tuner.py]                          Phase 8b
   Compute adjustments from ratings -> update settings.yaml

 [user_preference_critic.py]                  Phase 8b
   Train CLIP + Linear on ratings -> personalize quality scores
         |
         v
   outputs/comics/my_comic.pdf  (final output)
   outputs/rlhf_feedback.json
   outputs/user_preference_model.pt
```

---

*This guide covers every file in the current codebase.*
*When adding new files, update the relevant section and the Data Flow Diagram.*