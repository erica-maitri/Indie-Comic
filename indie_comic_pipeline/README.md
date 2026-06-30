# Ultimate AI Indie Comic Generator

A comprehensive, production-ready, local generative AI pipeline for academic research and high-fidelity comic generation. This system accepts a raw narrative prompt, processes it through a local LLM to extract psychological and narrative parameters, maps emotion arcs to visual language, generates temporally consistent panels using SDXL and LoRA, places speech bubbles with emotion-aware styling, validates output quality with a 5-dimension COMIC critic, assembles pages using the MangaFlow layout engine, and packages the final result as CBZ, HTML, and PDF.

---

## Why This Pipeline

Generic generative AI workflows produce single disjointed images with no temporal, structural, or narrative cohesion. This pipeline addresses that with six core advantages:

1. **Deterministic Visual Consistency** - IP-Adapter cross-attention layers paired with custom SDXL LoRAs bind facial contours, hair, clothing, and accessories across angles, lighting shifts, and diverse poses.
2. **8-Metric Visual Coherence Engine** - Mathematical consistency scoring using Structural SSIM, Gram Matrix texture matching, Canny Edge Density, DINOv2 identity embeddings, CLIP semantic similarity, HSV color histograms, Aesthetic score, and Thumbnail correlation.
3. **Emotion-Driven Visual Language** - Every emotion beat (e.g. `heaviness`, `breakthrough`, `quiet_hope`) maps to a specific lighting setup, color palette, and atmospheric descriptor injected directly into the diffusion prompt.
4. **Closed-Loop RLHF Optimization** - An Incremental Learner module implements an RLHF loop that tracks user ratings, back-propagates preferences into prompt templates, and adjusts quality thresholds and LoRA weights automatically.
5. **Universal Colab-Local Interoperability** - Optimized for low-VRAM deployment. Automatically scales memory and offloads tensors. Runs on a free Google Colab T4 GPU or locally on a standard consumer laptop.
6. **Rich Multimedia Output** - Interactive HTML comics, CBZ/CBR archives, production-ready print PDFs, and per-panel voice-cast dialogue via TTS.

---

## Quick Start

### 1. Install Dependencies

```bash
# One-click installer (auto-detects CUDA and installs matching PyTorch)
python install_all.py

# Or install manually
pip install -r requirements.txt      # Full local install
pip install -r requirements_colab.txt # Slim Colab install
```

### 2. Start Ollama (Required for LLM story generation)

```bash
# Install Ollama from https://ollama.ai then pull the model
ollama pull llama3.2
ollama serve
```

### 3. Run the Pipeline

```bash
# Full 8-phase pipeline (4 panels, GPU)
python integrated_pipeline.py --prompt "A lone wanderer discovers hope" --panels 4

# Dry-run (no GPU needed - uses mock images)
python integrated_pipeline.py --prompt "A hero faces a storm" --panels 4 --dry-run

# 10-panel production run
python run_10_panel_pipeline.py

# Fast 8-panel test (T4 optimized)
python generate_doodle_panels.py

# Compile generated panels into PDF
python compile_comic_pdf.py

# Launch web interface
python web_interface/app.py    # Navigate to http://localhost:5000
```

---

## Experimental Research Flow

The core scientific methodology is built on an iterative loop designed to empirically solve generative AI temporal inconsistency:

```
1. Metrics Build        -> Establish FID, BLEU, SSIM, Edge Density baseline
2. Check Consistency    -> Score base generation vs anchor panel
3. First Changes        -> Refine parameters, adjust prompts via RLHF feedback
4. Apply IP-Adapter     -> Force facial/feature preservation with structural conditioning
5. Final Changes        -> YOLOv8 speech bubble spatial layout optimization
6. Output Generation    -> TTS audio, .CBZ, .HTML, .PDF export
```

---

## Directory Structure

```
indie_comic_pipeline/
|
|-- Core Pipeline Orchestration
|   |-- integrated_pipeline.py          Main 8-phase master orchestrator (ENTRY POINT)
|   |-- run_10_panel_pipeline.py        Production 10-panel sequential generator
|   |-- generate_doodle_panels.py       Quick 8-panel test (T4 optimized)
|   |-- compile_comic_pdf.py            Assembles page grids into final PDF
|   |-- comic_exporter.py              Export to CBZ / CBR / HTML web comic
|   |-- audio_integration.py           TTS audio dialogue via gTTS
|   +-- model_comparator.py            A/B model testing (FID, CLIP, timing)
|
|-- Modular Core Pipeline (core/)
|   |-- story_intake.py                 Phase 0: Intelligent story intake
|   |   |-- StoryIntakeEngine           Processes raw prompts via Ollama
|   |   |-- MOOD_ARCS                   8 built-in emotional arc definitions
|   |   +-- EMOTION_VISUAL_MAP          Beat-to-visual-language lookup table
|   |
|   |-- agents/                         Phase 1: Multi-Agent Director Swarm
|   |   |-- agent_coordinator.py        Orchestrates 6 directors (blackboard)
|   |   |-- director_swarm.py           StoryDirector, ActionDirector, etc.
|   |   +-- base_agent.py               Abstract BaseAgent interface
|   |
|   |-- memory.py                       Blackboard state manager + checkpointing
|   |   |-- StorySectionMemory          Central shared state for all agents
|   |   |-- CharacterState              Per-character cross-panel state
|   |   |-- SceneState                  Environment continuity state
|   |   +-- PanelRecord                 Immutable generated panel record
|   |
|   |-- anchoring.py                    Phase 2: Reference-free anchoring
|   |   |-- ReferenceFreeAnchor         Generates and isolates visual anchor
|   |   +-- IdentityEmbeddingExtractor  Extracts facial/style tokens
|   |
|   |-- compositor.py                   Phase 3: CharCom weight blending
|   |-- advanced_attention.py           Phase 4: L1/L2/L3 attention mechanisms
|   |-- panel_engine.py                 Phases 2-4: Single panel generation runner
|   |-- text_image_integrator.py        Phase 5: DiffSensei bubble planner
|   |-- quality_critic.py              Phase 6: COMIC validation critic loop
|   |-- layout_engine.py               Phase 7: MangaFlow page geometry engine
|   |-- feedback.py                    Phase 8: RLHF telemetry and star rating logs
|   |-- optimizer.py                   Phase 8: Parameter backpropagation optimizer
|   |-- evaluation_suite.py            ModelEvaluator: FID, BLEU, CLIP, DINOv2, IoU
|   +-- backends/
|       |-- backend_selector.py        Selects optimal backend per panel context
|       |-- base_backend.py            Abstract BaseBackend interface
|       |-- sdxl_backend.py            SDXL diffusion backend
|       +-- flux_backend.py            Flux diffusion backend
|
|-- Environment and Config
|   |-- colab_setup.py                 Universal Colab/Jupyter bootstrap helper
|   |-- install_all.py                 One-click dependency installer
|   |-- generate_research_notebooks.py Generates unified research notebook
|   |-- requirements.txt              Full pinned dependencies
|   |-- requirements_colab.txt         Slim Colab-compatible dependencies
|   +-- config/
|       |-- settings.yaml             All pipeline settings (main config file)
|       +-- model_paths.yaml          HuggingFace model paths
|
|-- LangChain Enrichment Layer (langchain_code/)
|   |-- story_weaver_enricher.py      Reference-free cast enrichment (Mode 0)
|   |-- character_extractor.py        Character personality parser (Mode 1)
|   |-- story_extractor.py            Story setting and world parser (Mode 1)
|   |-- fusion_engine.py              Crossover storyboard builder (Mode 1)
|   |-- emotion_recognition_engine.py Per-panel expression mapper
|   +-- run_full_pipeline.py          Sequential LangChain runner
|
|-- Render Backends
|   |-- lora_code/                    SDXL + LoRA (1024x1024, best quality, ~10-12GB)
|   |-- sdxl_code/                    SDXL Base (1024x1024, ~8-10GB VRAM)
|   +-- sd15_code/                    SD 1.5 (512x512, ~4-6GB VRAM, fastest)
|       Each contains: generate_character.py, generate_components.py,
|                      generate_panels.py, run_*_pipeline.py
|
|-- Utilities (utils/)
|   |-- bridge_weaver.py              Story-Weaver JSON -> pipeline converter
|   |-- consistency_checker.py        8-metric visual consistency engine
|   |-- config_helper.py             Settings loader + path resolver
|   |-- image_utils.py               Strip/grid layout composer
|   +-- prompt_optimizer.py          SD prompt builder and deduplication
|
|-- Evaluation and Benchmarking (matrix_evaluation_zone/)
|   |-- model_matrix_bench.py         5-config benchmark suite
|   +-- storyboard_speed_bench.py     8-panel speed benchmark
|
|-- Web Interface (web_interface/)
|   |-- app.py                        Flask server (port 5000)
|   +-- templates/comic_generator.html
|
+-- Research Notebook
    +-- Indie_Comic_Pipeline.ipynb    Unified end-to-end notebook
```

---

## 8-Phase Pipeline Architecture

### Phase 0: Story Intake Engine (core/story_intake.py)

Processes raw user prompts through the local Ollama LLM to produce a structured `story_config` dict. Falls back to template-based generation if Ollama is unavailable. Supports 8 built-in emotional arcs (sad, angry, tired, happy, anxious, grief, determined, love) and custom user-defined mood shift sequences.

### Phase 1: Multi-Agent Planning Layer (core/agents/)

Six specialized Director agents operate sequentially on a shared Memory Blackboard:

| Director | Role |
|----------|------|
| StoryDirector | Establishes core panel events and character roster |
| ActionDirector | Defines relational verbs and physical actions |
| DialogueWriter | Structures speech schema and tone |
| PoseDirector | Translates actions into explicit body states |
| EmotionDirector | Translates dialogue/action into facial features |
| CameraDirector | Determines cinematic framing angle and size class |

### Phase 2: Reference-Free Anchoring (core/anchoring.py)

Isolates the first generated panel as the Primary Visual Anchor. Extracts identity embedding tokens (color profile, edge density, Gram matrix style, optional CLIP/DINOv2 semantics). Injects tokens into Memory Blackboard for all subsequent panel generations.

### Phases 3 and 4: In-Generation Consistency (core/compositor.py, core/advanced_attention.py)

**CharCom Compositor** dynamically blends model weights at runtime based on scene action intensity, character emotional state, and panel position in the story arc.

**Advanced Attention Manager** applies three physics-informed mechanisms:

- **L1 Heat Diffusion Prior** - Gaussian kernel suppresses high-frequency noise drift between panels
- **L2 Shared Attention Masking** - Blends anchor panel UNet K/V matrices into subsequent attention layers
- **L3 Spatiotemporal Prior** - Enforces channel-wise latent statistics toward anchor distribution

### Phase 5: Integrated Text-Image Generation (core/text_image_integrator.py)

DiffSensei approximation. Positions speech bubbles using Ollama layout planning (with JSON file cache). Five emotion-aware bubble styles: calm (ellipse), intense (jagged), thought (cloud), whisper (dashed), shout (spiky). Dynamic typography with emotion-scaled font sizes.

### Phase 6: Quality Validation Layer (core/quality_critic.py)

COMIC Critic Pipeline - evaluates panels across 5 weighted dimensions:

| Dimension | Weight | Metric |
|-----------|--------|--------|
| Visual Consistency | 30% | Identity preservation vs anchor |
| Aesthetic Quality | 25% | Sharpness, colorfulness, contrast |
| Narrative Coherence | 20% | Story flow continuity |
| Emotional Engagement | 15% | Text-image emotion alignment |
| Readability | 10% | Bubble placement, text clarity |

Failed panels trigger a reject-and-regenerate loop (max 2 retries) with adjusted guidance scale and inference steps.

### Phase 7: Layout and Assembly (core/layout_engine.py)

MangaFlow Layout Engine arranges panels dynamically based on action intensity and size class. Full-page panels receive dominant canvas real estate. Gutters (12px), margins (40px), and page numbers are applied professionally. Replaces static 2x2 grids.

### Phase 8: Export and Adaptive RLHF (core/feedback.py, core/optimizer.py)

Exports to CBZ, interactive HTML scrollbook, and print-ready PDF. Interactive RLHF Telemetry Loop collects user star ratings (1-5) per panel. SystemOptimizer back-propagates preferences into positive/negative prompts, LoRA scale, guidance scale, and quality critic thresholds written directly to `config/settings.yaml`.

---

## 8-Metric Consistency Engine

Located in `utils/consistency_checker.py`. Runs metrics sequentially, configurable per deployment:

| Metric | Algorithm | T4 Default |
|--------|-----------|------------|
| SSIM | Structural Similarity Index | Enabled |
| Gram Matrix | Feature Map Gramian Correlation | Enabled |
| Edge Density | Canny Edge Detection Comparison | Enabled |
| Aesthetic Score | Contrast + Laplacian + Color Variance | Enabled |
| Thumbnail Correlation | Pearson Correlation | Enabled |
| HSV Color | Color Histogram Intersection | Disabled |
| CLIP Semantic | CLIP Embeddings Cosine Distance | Disabled |
| DINOv2 Structure | DINOv2 Pooling Cosine Distance | Disabled |

Enable/disable via `config/settings.yaml` under `consistency:`. Heavy metrics (CLIP, DINOv2) are disabled by default to preserve T4 VRAM.

---

## Key API Classes

### `IntegratedComicPipeline` (integrated_pipeline.py)

```python
pipeline = IntegratedComicPipeline(dry_run=False)

# Full run
results = pipeline.run(
    prompt="A lone wanderer discovers hope",
    character_name="Wanderer",
    story_world="The Abstract",
    panel_count=4,
    style_reference="",
    character_characteristics="",
    story_reference="",
    mood_shifts=None          # Optional: ["heaviness", "spark", "triumph"]
)

# Batch / checkpoint run
results = pipeline.run_batch(
    start_panel=1, end_panel=4,
    save_checkpoint="outputs/checkpoints/batch.json"
)

# Interactive RLHF collection
pipeline.collect_interactive_feedback(results)
```

### `StoryIntakeEngine` (core/story_intake.py)

```python
engine = StoryIntakeEngine(ollama_model="llama3.2")
story_config = engine.process_prompt(
    user_prompt="A wanderer facing the abyss",
    panel_count=6,
    character_name="Kira",
    story_world="Neon Ruins"
)
# Or load from Story-Weaver output
story_config = engine.load_existing_story("path/to/story_dynamic.json")
```

### `QualityCritic` (core/quality_critic.py)

```python
critic = QualityCritic(threshold=0.55, strict_threshold=0.70, max_retries=2)
evaluation = critic.evaluate(panel_result, memory)
# evaluation["verdict"] -> "PASS" or "FAIL"
# evaluation["composite"] -> float score
# evaluation["adjustments"] -> {guidance_scale_delta, steps_delta}
```

---

## Benchmarks and Evaluation

```bash
# Compare visual quality across 5 distinct pipeline configurations
python matrix_evaluation_zone/model_matrix_bench.py

# Benchmark rendering speed of storyboard iterations
python matrix_evaluation_zone/storyboard_speed_bench.py

# Run the full evaluation suite
python run_evaluation.py
```

---

## Configuration

All settings are in `config/settings.yaml`. Key values:

```yaml
models:
  sdxl:
    name: "Lykon/dreamshaper-xl-1-0"
    cpu_offload: true           # Critical for T4 16GB VRAM

generation:
  default_size: {width: 768, height: 768}
  inference_steps: 25
  guidance_scale: 7.5

langchain:
  model: "llama3.2"
  ollama_url: "http://localhost:11434"

consistency:
  threshold: 0.55
  enable_clip: false            # Disable on T4 for speed

quality_critic:
  max_retries: 2
  threshold: 0.55
```

See `BACKEND.md` for the complete configuration reference with all fields documented.

---

## Troubleshooting

**Out of Memory (OOM) on GPU:**
- Set `resolution` to `[512, 512]` in `config/settings.yaml` or use `sd15_code/`
- Disable CLIP and DINOv2 in `config/settings.yaml` (`enable_clip: false`, `enable_dinov2: false`)
- Enable `enable_model_cpu_offload: true` in the pipeline config

**Ollama not responding:**
- Ensure Ollama is running: `ollama serve`
- Verify model is pulled: `ollama pull llama3.2`
- Set `OLLAMA_URL` environment variable if running on a different host
- Pipeline automatically falls back to template generation if Ollama is unreachable

**Fonts missing (Arial/TrueType):**
- On Linux/Colab, `TextImageIntegrator` automatically catches `arial.ttf` missing errors and falls back to PIL's internal monospace font

**gTTS not installed:**
- `audio_integration.py` wraps all `gtts` imports in `try/except` blocks. Missing gTTS outputs a warning log without crashing the pipeline.

**JSON parse errors from LLM:**
- `StoryIntakeEngine` re-prompts Ollama with a stricter JSON-only system message on parse failure, then falls back to template if that also fails.

---

## Dependencies

Core dependencies (see `requirements.txt` for full pinned versions):

| Category | Libraries |
|----------|-----------|
| Diffusion | diffusers, accelerate, transformers, safetensors |
| Image | Pillow, opencv-python-headless, scikit-image, ultralytics |
| LangChain | langchain, langchain-ollama, langchain-openai, langgraph |
| LLM | ollama, openai, tiktoken |
| Evaluation | torchmetrics, scipy, numpy |
| Web | flask |
| TTS | gTTS |
| Export | rarfile (CBZ/CBR) |
| Config | PyYAML, python-dotenv |

---

## Technical Reference

For full class contracts, method signatures, data schemas, configuration tables, and data flow diagrams, see `BACKEND.md`.

---

*Indie Comic Pipeline v2.0.0*
