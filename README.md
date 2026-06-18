# Cyberpunk-San / Indie Comic Generator

> **A fully local, end-to-end AI system** that reads an emotion from you, writes a multi-panel story, casts a full ensemble of characters with individual moods and expressions, and renders everything as a comic strip — without touching the cloud, without a reference image, and without a single manually written prompt.

---

## Table of Contents

1. [Project Overview](#-project-overview)
2. [Repository Structure](#-repository-structure)
3. [System Architecture](#-system-architecture)
4. [Module Deep Dive](#-module-deep-dive)
   - [Story-Weaver](#1-story-weaver)
   - [Indie Comic Pipeline](#2-indie-comic-pipeline)
5. [Two Operational Modes](#-two-operational-modes)
   - [Mode 0 — Story-Weaver Direct (No Reference Image)](#mode-0--story-weaver-direct-recommended)
   - [Mode 1 — LangChain Extraction (Classic Flow)](#mode-1--langchain-extraction-classic-flow)
6. [Data Flow (End-to-End)](#-data-flow-end-to-end)
7. [Consistency Checking Engine](#-consistency-checking-engine)
8. [Research Notebooks](#-research-notebooks)
9. [Evaluation & Benchmarking](#-evaluation--benchmarking)
10. [Installation](#-installation)
    - [Local Setup](#local-setup)
    - [Google Colab Setup](#google-colab-setup)
11. [Quick Start](#-quick-start)
12. [Configuration Reference](#-configuration-reference)
13. [Output Files](#-output-files)
14. [Web Interface](#-web-interface)
15. [Requirements](#-requirements)
16. [Troubleshooting](#-troubleshooting)
17. [Design Decisions](#-design-decisions)

---

## 🧭 Project Overview

This repository contains two tightly-integrated sub-projects:

| Sub-project | What it does |
|-------------|-------------|
| **`Story-Weaver/`** | Fine-tunes and merges LLMs to generate mood-driven comic story scripts (JSON panels with visual cues, dialogue, emotion beats, motion) |
| **`indie_comic_pipeline/`** | Receives the story JSON and renders it as visual comic panels using SDXL / SD 1.5, with an 8-metric visual consistency engine |

Together they form a pipeline where **you speak your emotion → the system writes the story → the system draws the comic**.

### Key Capabilities

- **Emotion-to-Comic Pipeline** — Input a mood, get a fully rendered comic strip
- **Reference-Free Generation** — No character sheet needed; Panel 1 becomes the visual anchor
- **8-Metric Consistency Engine** — HSV color, SSIM, Gram style, Edge density, CLIP, DINOv2, Aesthetic score, Thumbnail correlation
- **Multi-Character Cast** — LLM auto-generates ≥3 side characters per panel with emotions, expressions, and clothing
- **3 Render Backends** — SDXL + LoRA (best quality), SDXL Base, SD 1.5 (lightest VRAM)
- **Multi-Format Export** — CBZ, CBR, HTML web comic, PDF, interactive TTS audio comic
- **T4 GPU Optimized** — CPU offloading, attention slicing, VAE slicing, FP16 inference
- **Google Colab Ready** — Universal setup cell in all notebooks with automatic dependency installation
- **RLHF Feedback Loop** — `IncrementalLearner` collects user ratings to refine future prompts
- **Model A/B Testing** — `ModelComparator` benchmarks FID, CLIP score, edge density across backends

---

## 📁 Repository Structure

```text
drid/
├── README.md                            # This file
├── methodology.md                       # Full academic methodology document
├── pyrightconfig.json                   # Type checking config
│
├── Story-Weaver/                        # LLM story generation engine
│   ├── stage2_story_generation.py       # Main unified generation + training script
│   ├── story_gen.py                     # Quick inference entry point (Ollama)
│   ├── story_gen_finetuned.py           # Fine-tuned model inference
│   ├── story_gen_old.py                 # Original training examples
│   ├── merge.py                         # Merges fine-tuned LoRA into 16-bit
│   ├── evaluate.py                      # Multi-metric evaluation suite
│   ├── story_dynamic.json               # Live story output (panels JSON)
│   └── requirements.txt
│
└── indie_comic_pipeline/                # Visual rendering engine
    │
    │── Core Pipeline Files ──────────────────────────────────────
    ├── ultimate_comic_pipeline.py       # Master pipeline: ComicConfig, ModelEnsemble,
    │                                    #   PanelGenerator, PageGenerator, QualityMetrics,
    │                                    #   SpeechBubbleOptimizer, EmotionValidator,
    │                                    #   NarrativeMemory, StyleManager, UltimateComicGenerator
    ├── run_10_panel_pipeline.py          # Production 10-panel sequential generator
    ├── generate_doodle_panels.py         # Fast test panel generator (T4 optimized)
    ├── compile_comic_pdf.py              # Assembles page grids into final PDF
    ├── comic_exporter.py                 # Export to CBZ / CBR / HTML web comic
    ├── audio_integration.py              # TTS audio dialogue generation (gTTS)
    ├── model_comparator.py               # A/B model testing with FID, CLIP, timing
    ├── incremental_learner.py            # RLHF feedback collection & prompt learning
    ├── colab_setup.py                    # Universal Colab/Jupyter environment setup
    ├── install_all.py                    # One-click dependency installer
    ├── generate_research_notebooks.py    # Generates 6 research experiment notebooks
    │
    │── Configuration ────────────────────────────────────────────
    ├── config/
    │   ├── settings.yaml                 # All model/generation/output/consistency settings
    │   └── model_paths.yaml              # HuggingFace model paths & ComfyUI directories
    ├── requirements.txt                  # Full pinned dependencies
    ├── requirements_colab.txt            # Slim Colab-compatible dependencies
    │
    │── LangChain Code (The Brain) ───────────────────────────────
    ├── langchain_code/
    │   ├── __init__.py
    │   ├── story_weaver_enricher.py      # Reference-free cast enrichment (Mode 0)
    │   ├── character_extractor.py        # LangChain character personality parser
    │   ├── story_extractor.py            # LangChain story setting parser
    │   ├── fusion_engine.py              # Crossover storyboard builder (10 pages)
    │   ├── emotion_recognition_engine.py # Per-panel expression mapper
    │   └── run_full_pipeline.py          # Sequential LangChain pipeline runner
    │
    │── Render Backends ──────────────────────────────────────────
    ├── sdxl_code/                        # SDXL Base (1024×1024)
    │   ├── __init__.py
    │   ├── generate_character.py         # Character sheet generator
    │   ├── generate_components.py        # Scene component generator
    │   ├── generate_panels.py            # Panel generator
    │   └── run_sdxl_pipeline.py          # Orchestrator
    │
    ├── lora_code/                        # SDXL + LoRA (1024×1024, best quality)
    │   ├── __init__.py
    │   ├── generate_character.py
    │   ├── generate_components.py
    │   ├── generate_panels.py
    │   └── run_lora_pipeline.py
    │
    ├── sd15_code/                        # SD 1.5 (512×512, lightest VRAM)
    │   ├── __init__.py
    │   ├── generate_character.py
    │   ├── generate_components.py
    │   ├── generate_panels.py
    │   └── run_sd15_pipeline.py
    │
    │── Utilities ────────────────────────────────────────────────
    ├── utils/
    │   ├── __init__.py
    │   ├── bridge_weaver.py              # Story-Weaver JSON → pipeline converter
    │   ├── consistency_checker.py        # 8-metric visual consistency engine
    │   ├── config_helper.py              # Settings loader + path resolver
    │   ├── image_utils.py                # Strip/grid layout composer
    │   └── prompt_optimizer.py           # SD prompt builder & deduplication
    │
    │── Evaluation & Benchmarking ────────────────────────────────
    ├── matrix_evaluation_zone/
    │   ├── __init__.py
    │   ├── model_matrix_bench.py         # 5-config benchmark: SD1.5, SD1.5+LoRA,
    │   │                                 #   SDXL, SDXL+LoRA only, SDXL+LoRA+prompts
    │   ├── storyboard_speed_bench.py     # 8-panel speed benchmark
    │   ├── model1_matrix_bench           # Benchmark results (config 1)
    │   ├── model2_matrix_bench           # Benchmark results (config 2)
    │   ├── model3_matrix_bench           # Benchmark results (config 3)
    │   └── outputs/                      # Generated benchmark images
    │
    │── Web Interface ────────────────────────────────────────────
    ├── web_interface/
    │   ├── __init__.py
    │   ├── app.py                        # Flask server (port 5000)
    │   └── templates/
    │       └── comic_generator.html      # Interactive comic generation UI
    │
    │── Research Notebooks ───────────────────────────────────────
    ├── 01_Metrics_Build_and_Setup.ipynb
    ├── 02_Initial_Generation_and_Consistency_Check.ipynb
    ├── 03_First_Changes_and_Refinement.ipynb
    ├── 04_Apply_IP_Adapter.ipynb
    ├── 05_Final_Changes_and_Spatial_Layout.ipynb
    ├── 06_Multimedia_Output_and_Export.ipynb
    │
    │── Prompts & Style ──────────────────────────────────────────
    ├── prompts/
    │   └── style_prompts/                # Style prompt templates
    │
    │── Output ───────────────────────────────────────────────────
    └── outputs/
        ├── fusion/                       # Generated JSON storyboards
        ├── characters/                   # Character reference sheets
        ├── comics/                       # Panel images, strips, grids
        ├── exports/                      # CBZ/CBR/HTML exports
        ├── audio/                        # TTS dialogue MP3 files
        ├── production_run/               # 10-panel production output
        └── comparison/                   # Model A/B test reports
```

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INPUT                             │
│          "I feel anxious / happy / sad / tired..."          │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │     STORY-WEAVER      │
         │  stage2_story_        │
         │  generation.py        │
         │                       │
         │  • Emotion detection  │
         │  • Mood arc mapping   │
         │  • LLM panel writer   │
         │  → story_dynamic.json │
         └───────────┬───────────┘
                     │
          ┌──────────▼──────────┐
          │   BRIDGE / MODE     │  ← Choose Mode 0 or Mode 1
          └──────────┬──────────┘
                     │
        ┌────────────┴──────────────────┐
        │                               │
  ┌─────▼──────┐               ┌────────▼────────┐
  │  MODE 0    │               │    MODE 1        │
  │ Story-Weaver│              │  LangChain       │
  │ Direct     │               │  Extraction      │
  │            │               │                  │
  │ Enricher   │               │ character_        │
  │ (LLM per   │               │ extractor.py     │
  │  panel)    │               │ story_extractor  │
  │            │               │ fusion_engine    │
  │ Full cast: │               │ emotion_recog.   │
  │ Main char  │               │ engine           │
  │ + ≥3 side  │               │                  │
  │ chars      │               │ storyboard_with  │
  │ + scenery  │               │ _emotions.json   │
  │            │               │ + char reference │
  │ enriched_  │               │   image          │
  │ storyboard │               └────────┬─────────┘
  │ .json      │                        │
  └────────────┤                        │
               └─────────┬─────────────┘
                          │
               ┌──────────▼──────────┐
               │  PANEL GENERATOR    │
               │                     │
               │  SDXL / LoRA / SD15 │
               │  generate_panels.py │
               │                     │
               │  Mode 0: NO IP-     │
               │  Adapter. Panel 1   │
               │  = visual anchor.   │
               │                     │
               │  Mode 1: IP-Adapter │
               │  + char ref image   │
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │ CONSISTENCY CHECKER │
               │                     │
               │  8 metrics:         │
               │  HSV Color (Opt)    │
               │  SSIM               │
               │  Gram Style         │
               │  Edge Density       │
               │  CLIP Semantic      │
               │  DINOv2 Structure   │
               │  Aesthetic Score    │
               │  Thumbnail Corr.    │
               └──────────┬──────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
  ┌───────▼───┐   ┌──────▼──────┐  ┌─────▼─────┐
  │  EXPORTER │   │ PDF COMPILER│  │ AUDIO TTS │
  │           │   │             │  │           │
  │ CBZ / CBR │   │ Page grids  │  │ gTTS per  │
  │ HTML Web  │   │ → final PDF │  │ dialogue  │
  └───────────┘   └─────────────┘  └───────────┘
```

---

## 🔬 Module Deep Dive

### 1. Story-Weaver

Located in `Story-Weaver/`. This is the language intelligence layer of the system.

#### How It Works

1. **Emotion Input** — The user provides a primary emotion (`sad`, `angry`, `tired`, `happy`, `anxious`, `grief`) along with optional free-text context.

2. **Mood Arc Mapping** — Each emotion is mapped to a curated **6-stage arc** (e.g. `sad` → heaviness → stillness → faint_warmth → tentative_light → soft_openness → quiet_hope).

3. **System Prompt Engineering** — A strict prompt instructs the model:
   - Never name emotions literally (show them through objects and body language)
   - Every panel must contain: `visual`, `dialogue`, `emotion_beat`, `motion`
   - One **recurring visual motif** must appear across all panels
   - Every panel must include a body sensation word (chest, breath, throat, etc.)

4. **Model Training** — The model is fine-tuned from Qwen2.5-1.5B-Instruct using Unsloth/LoRA on hand-crafted training examples covering all 6 emotions, then merged into a single 16-bit model.

5. **Output** — A `story_dynamic.json` containing:
   ```json
   {
     "recurring_motif": "a ceramic mug with a small chip on the rim",
     "mood_journey": "From the weight of an ordinary evening toward a small warmth.",
     "panels": [
       {
         "panel": 1,
         "visual": "Kitchen table at dusk...",
         "dialogue": "...",
         "emotion_beat": "heaviness",
         "motion": "The hand does not move..."
       }
     ]
   }
   ```

#### Key Scripts

| Script | Purpose |
|--------|---------|
| `stage2_story_generation.py` | All-in-one: generate, train, or export dataset |
| `story_gen.py` | Quick Ollama-based inference (no GPU needed) |
| `merge.py` | Merges fine-tuned LoRA into full 16-bit model |
| `evaluate.py` | Runs ROUGE, BERTScore, Perplexity, hallucination checks |

#### Running Modes

```bash
# Generate a story (reads .env for model path)
python stage2_story_generation.py

# Generate with specific emotion and panel count
python stage2_story_generation.py --mode generate --emotion anxious --panels 8

# Fine-tune on the training examples
python stage2_story_generation.py --mode train --epochs 3

# Export training data only (JSONL for any trainer)
python stage2_story_generation.py --mode dataset
```

---

### 2. Indie Comic Pipeline

Located in `indie_comic_pipeline/`. This is the visual rendering layer.

#### `ultimate_comic_pipeline.py` — The Engine

The monolithic master file containing all core classes:

| Class | Role |
|-------|------|
| `ComicConfig` | Dataclass for all generation parameters (character, world, style, pages, seed, etc.) |
| `StyleManager` | Maps style names (manga, western, noir, watercolor, retro) to SD prompt tokens |
| `NarrativeMemory` | Tracks story context, previous panel prompts, and emotion progression |
| `EmotionValidator` | Validates emotion labels and maps them to visual descriptors |
| `SpeechBubbleOptimizer` | YOLOv8-based speech bubble placement with IoU collision detection |
| `QualityMetrics` | FID computation, CLIP score, BLEU, and composite quality scoring |
| `ModelEnsemble` | Loads and manages SDXL / LoRA / SD 1.5 pipelines with T4 memory optimization |
| `PanelGenerator` | Generates individual panels with consistency checking and quality scoring |
| `PageGenerator` | Assembles 4 panels into a 2×2 page layout |
| `UltimateComicGenerator` | Top-level orchestrator: generates multi-page comics end-to-end |

#### `langchain_code/` — The Brain

| File | Role |
|------|------|
| `story_weaver_enricher.py` | LLM enriches every panel with full cast (main + ≥3 side chars) and detailed scenery → `enriched_storyboard.json` |
| `character_extractor.py` | Extracts structured personality JSON from a character name (Legacy Mode) |
| `story_extractor.py` | Extracts lighting, era, color palette, mood from setting name (Legacy Mode) |
| `fusion_engine.py` | Merges character + setting into a 10-page storyboard with dialogue (Legacy Mode) |
| `emotion_recognition_engine.py` | Maps panel dialogue to facial expressions via a zero-shot LLM classifier (Legacy Mode) |
| `run_full_pipeline.py` | Runs all 3 LangChain steps in sequence with Ollama auto-start |

#### `utils/` — The Toolbelt

| File | Role |
|------|------|
| `bridge_weaver.py` | Converts `story_dynamic.json` to pipeline format. `--enrich` flag triggers Story-Weaver mode |
| `consistency_checker.py` | 8-metric visual similarity engine (HSV, SSIM, Gram, Edge, CLIP, DINOv2, Aesthetic, Thumbnail). Global model caching prevents reloading CLIP/DINOv2 on every check. |
| `config_helper.py` | Loads `settings.yaml`, resolves output paths, auto-creates parent directories |
| `image_utils.py` | Creates horizontal strips and dynamic grid layouts from panel images |
| `prompt_optimizer.py` | SD prompt sanitization, style boosting, negative prompt assembly, and character consistency constraints |

#### Render Backends — `lora_code/`, `sdxl_code/`, `sd15_code/`

Three parallel panel generator backends, all sharing the same enriched-mode logic:

| Backend | Model | Resolution | VRAM | Notes |
|---------|-------|-----------|------|-------|
| `lora_code` | SDXL + LineAniRedmond LoRA | 1024×1024 | ~11-12 GB | Best style consistency, manga line-art |
| `sdxl_code` | SDXL Base | 1024×1024 | ~8-10 GB | Fastest SDXL inference |
| `sd15_code` | Stable Diffusion 1.5 | 512×512 | ~4-6 GB | Lightest VRAM |

Each backend contains:
- `generate_character.py` — Pre-generates a character reference sheet (Mode 1)
- `generate_components.py` — Generates scene components (backgrounds, props)
- `generate_panels.py` — Main panel generation with consistency checking
- `run_*_pipeline.py` — Orchestrator that runs character → components in sequence

#### Supporting Modules

| File | Role |
|------|------|
| `comic_exporter.py` | Exports to CBZ (ZIP archive), CBR (RAR fallback), and scrollable HTML web comic |
| `audio_integration.py` | TTS dialogue generation via gTTS with per-character voice profiles (accent via TLD) |
| `model_comparator.py` | A/B tests different models on the same prompt; measures FID, CLIP score, file size, timing; generates HTML + JSON reports |
| `incremental_learner.py` | Collects user ratings (1-5 stars) and comments; extracts patterns from high-rated outputs to adjust future prompt modifiers |
| `compile_comic_pdf.py` | Scans `outputs/comics/` for page grid layouts and compiles them into a multi-page PDF with fallback style detection |
| `run_10_panel_pipeline.py` | Production mode: reads all 4 fusion JSON files and generates exactly 10 high-fidelity panels sequentially |
| `generate_doodle_panels.py` | Quick test generator with hardcoded 8-panel storyboard and consistency checking |
| `colab_setup.py` | Universal environment bootstrap for Colab and local Jupyter |
| `generate_research_notebooks.py` | Generates the 6 research experiment notebooks with universal setup cells |

---

## ⚡ Two Operational Modes

### Mode 0 — Story-Weaver Direct *(Recommended)*

**No character reference image required. No IP-Adapter loading. Just story → panels.**

```
story_dynamic.json
       ↓
story_weaver_enricher.py   ← Ollama LLM (llama3.2, one call per panel)
       ↓
enriched_storyboard.json   ← Contains per-panel:
                              • main_character: {name, description, emotion,
                                mood, expression, action, clothing}
                              • side_characters: [{...} × ≥3]
                              • scenery: rich environment description
                              • augmented_prompt: final SD text prompt
       ↓
generate_panels.py         ← Detects enriched_storyboard.json, skips IP-Adapter
       ↓
Panel images               ← Panel 1 becomes visual reference anchor
       ↓
consistency_checker.py     ← Panels 2-N checked against Panel 1 (8 metrics)
```

**Key properties:**
- ✅ No character sheet image needed before generation
- ✅ Every panel gets a full cast: main character + minimum 3 side characters
- ✅ Each character has: `emotion`, `mood`, `expression`, `action`, `clothing`
- ✅ LLM invents contextually appropriate side characters if the scene is sparse
- ✅ Panel 1 of each page serves as the visual consistency anchor
- ✅ All 8 consistency metrics still run after generation

### Mode 1 — LangChain Extraction *(Classic Flow)*

```
Character Name + Setting Name
       ↓
character_extractor.py     ← JSON personality profile
story_extractor.py         ← JSON environment profile
       ↓
fusion_engine.py           ← 10-page storyboard + SDXL prompts
       ↓
emotion_recognition_engine.py ← Per-panel expression maps
       ↓
generate_character.py      ← Pre-generates character reference image
       ↓
generate_panels.py         ← SDXL + IP-Adapter conditioned on ref image
       ↓
consistency_checker.py     ← All panels checked against reference image
```

**Key properties:**
- ✅ Full creative control over character name and world name
- ✅ IP-Adapter enforces strict character face/style consistency
- ✅ 10-page storyboard with scene location details
- ⚠️ Requires pre-generating a character reference image before panel generation

---

## 🔄 Data Flow (End-to-End)

### Mode 0 (Story-Weaver Direct)

```
┌──────────────────────────────────────────────────────────────┐
│ INPUT: story_dynamic.json (from Story-Weaver)                │
│                                                              │
│ {                                                            │
│   "recurring_motif": "ceramic mug with chip",               │
│   "mood_journey": "heaviness toward quiet hope",            │
│   "panels": [                                               │
│     {"panel": 1, "visual": "...", "emotion_beat": "..."}    │
│   ]                                                          │
│ }                                                            │
└───────────────────────────┬──────────────────────────────────┘
                            │ bridge_weaver.py --enrich
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ ENRICHER: story_weaver_enricher.py                           │
│                                                              │
│ For each panel → Ollama (llama3.2) generates:               │
│ {                                                            │
│   "main_character": {                                        │
│     "name": "Wanderer",                                      │
│     "description": "tall figure with hollow eyes",          │
│     "emotion": "weighted",                                   │
│     "mood": "hollow",                                        │
│     "expression": "jaw slack, eyes downcast, shoulders low",│
│     "action": "standing at the kitchen table, not moving",  │
│     "clothing": "worn grey coat, frayed scarf"              │
│   },                                                         │
│   "side_characters": [                                       │
│     {"name": "The Neighbor", "emotion": "concerned", ...},  │
│     {"name": "The Cat", "emotion": "watchful", ...},        │
│     {"name": "Reflection", "emotion": "distant", ...}       │
│   ],                                                         │
│   "scenery": "dusk kitchen, amber lamp, rain on window...", │
│   "augmented_prompt": "<assembled SD-ready text prompt>"    │
│ }                                                            │
└───────────────────────────┬──────────────────────────────────┘
                            │ saves enriched_storyboard.json
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ PANEL GENERATOR: generate_panels.py                          │
│                                                              │
│ Detects enriched_storyboard.json → enriched mode active     │
│ Skips IP-Adapter entirely                                    │
│ For each panel:                                              │
│   pipe(prompt=panel["augmented_prompt"])                     │
│   → saves page_N_panel_M.png                                 │
│   → Panel 1: set_reference_from_panel(panel_1_path)         │
│   → Panel 2-N: check_consistency(panel_path) vs Panel 1     │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ EXPORT PIPELINE                                              │
│                                                              │
│ • image_utils.py → Horizontal strip + Dynamic grid          │
│ • compile_comic_pdf.py → Multi-page PDF                      │
│ • comic_exporter.py → CBZ / CBR / HTML web comic            │
│ • audio_integration.py → TTS MP3 per dialogue line          │
└──────────────────────────────────────────────────────────────┘
```

---

## 🎯 Consistency Checking Engine

`utils/consistency_checker.py` measures visual coherence across panels using 8 independent metrics:

| Metric | Method | What It Checks | Weight |
|--------|--------|---------------|--------|
| **HSV Color** | Histogram comparison (OpenCV) | Same color palette | 25% |
| **SSIM** | Structural Similarity Index | Pixel-level structural similarity | 30% |
| **Gram Matrix** | 5-channel spatial features (RGB + Sobel gradients) | Consistent artistic style / texture | 20% |
| **Edge Density** | Canny edge detection | Line weight and density consistency | 15% |
| **CLIP Semantic** | CLIP image embeddings (cosine similarity) | High-level semantic/content similarity | 5% |
| **DINOv2 Structure** | DINOv2 pooler output (cosine similarity) | Deep structural/identity consistency | 5% |
| **Aesthetic Score** | Laplacian variance + contrast + colorfulness | Panel visual quality (0.0–1.0) | — |
| **Thumbnail Corr.** | Pearson correlation on grayscale thumbnails | Global composition similarity | — |

**Configuration** (in `config/settings.yaml`):
- `enable_clip: false` / `enable_dinov2: false` — Disabled by default to save VRAM on T4
- `enable_ssim: true` / `enable_edge: true` / `enable_style: true` — Lightweight, always on
- `threshold: 0.55` — Combined score above this = consistent
- `device: "cpu"` — Runs consistency models on CPU to preserve GPU VRAM for generation

**In Mode 0:** Panel 1 of each page is set as the reference via `checker.set_reference_from_panel(panel_1_path)`.

**In Mode 1:** The pre-generated character reference image is set via `checker.set_reference(ref_path)`.

---

## 📓 Research Notebooks

Six Jupyter notebooks implement a structured research experiment flow. Each notebook is auto-generated by `generate_research_notebooks.py` and includes a **universal setup cell** that works on both Google Colab and local Jupyter:

| Notebook | Phase | What It Does |
|----------|-------|-------------|
| `01_Metrics_Build_and_Setup.ipynb` | Baseline | Initializes `ModelComparator` and `QualityMetrics` |
| `02_Initial_Generation_and_Consistency_Check.ipynb` | Generation | Runs SDXL generation (requires GPU), evaluates emotion detection and alignment |
| `03_First_Changes_and_Refinement.ipynb` | Feedback | Uses `IncrementalLearner` to log feedback and refine prompts |
| `04_Apply_IP_Adapter.ipynb` | Structural Fix | Demonstrates IP-Adapter cross-attention conditioning |
| `05_Final_Changes_and_Spatial_Layout.ipynb` | Layout | YOLOv8 `SpeechBubbleOptimizer` for collision-free text placement |
| `06_Multimedia_Output_and_Export.ipynb` | Export | TTS audio generation + CBZ/HTML export |

---

## 📊 Evaluation & Benchmarking

### Model Matrix Benchmark (`matrix_evaluation_zone/model_matrix_bench.py`)

Runs 5 model configurations on the same prompt and measures:

| Metric | How |
|--------|-----|
| **CLIP Score** | Cosine similarity between generated image and text prompt embeddings |
| **FID Score** | Fréchet Inception Distance against a reference baseline |
| **Inference Time** | Wall-clock time per generation |
| **Peak VRAM** | Maximum GPU memory allocated during generation |
| **Edge Density** | Canny edge percentage (measures line-art detail) |

**Configurations tested:**
1. Stable Diffusion v1.5 (baseline)
2. SD 1.5 + LoRA
3. Stable Diffusion XL (base)
4. SDXL + LoRA only (no style prompts)
5. SDXL + LoRA + style prompts

### Storyboard Speed Benchmark (`storyboard_speed_bench.py`)

Generates 8 sequential panels with per-panel timing to measure throughput.

### Story-Weaver Evaluation (`Story-Weaver/evaluate.py`)

```bash
python evaluate.py                    # Rule-based checks + hallucination detection
python evaluate.py --perplexity       # Perplexity (loads merged model)
python evaluate.py --nlp --refs *.json  # ROUGE + BERTScore
python evaluate.py --all --refs *.json  # Full evaluation suite
```

---

## 🛠 Installation

### Local Setup

#### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| CUDA (recommended) | 11.8 or 12.1 |
| Ollama | latest |
| RAM | ≥ 16 GB |
| VRAM | ≥ 6 GB (SD 1.5) / ≥ 12 GB (SDXL) |

#### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Cyberpunk-San/Indie-Comic.git
cd Indie-Comic

# 2. Create virtual environment
python -m venv py10
# Windows:
py10\Scripts\activate
# Linux/macOS:
source py10/bin/activate

# 3. Install Story-Weaver dependencies
cd Story-Weaver
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
cd ..

# 4. Install Pipeline dependencies
cd indie_comic_pipeline
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
# Or use the one-click installer:
python install_all.py
cd ..

# 5. Install and start Ollama
# Download from https://ollama.com
ollama serve              # Start the daemon
ollama pull llama3.2      # Pull the LLM used for enrichment
```

#### (Optional) Download fine-tuned merged model

The pre-trained Story-Weaver merged model is available at:
📦 https://drive.google.com/drive/folders/11iTLqizx2rOP8t4RfgnbIm3FeFSg3Tsw

Place it in `Story-Weaver/moodweaver_stage2_merged/`.

### Google Colab Setup

Every research notebook (01–06) includes a **universal setup cell** that automatically:

1. Detects whether you're running in Colab or local Jupyter
2. Clones the repository (Colab only)
3. Installs dependencies from `requirements_colab.txt` (slim, no version conflicts)
4. Adds `indie_comic_pipeline/` to `sys.path`

**To get started on Colab:**
1. Upload any notebook (e.g. `01_Metrics_Build_and_Setup.ipynb`) to Google Colab
2. Set runtime to **T4 GPU**: `Runtime → Change runtime type → T4 GPU`
3. Run the first cell — it handles everything automatically

**Colab-specific files:**
- `colab_setup.py` — The bootstrap script that configures the environment
- `requirements_colab.txt` — Slim dependency list that doesn't conflict with Colab's pre-installed packages

---

## 🚀 Quick Start

### Option A — Full pipeline from emotion to comic (Mode 0, Recommended)

```bash
# Step 1: Generate the story from an emotion
cd Story-Weaver
python story_gen.py
# → Produces story_dynamic.json

# Step 2: Run the full comic pipeline
cd ../indie_comic_pipeline
python run_everything.py
# → Choose "0" for Story-Weaver Direct Mode
# → Enter character name, world name, min side chars
# → Choose render backend (SDXL LoRA recommended)
# → Panels generate page by page
```

### Option B — Direct CLI without the orchestrator

```bash
cd indie_comic_pipeline

# Enrich the story (builds full cast + SDXL prompts)
python utils/bridge_weaver.py \
  --enrich \
  --input "../Story-Weaver/story_dynamic.json" \
  --character "Mira" \
  --world "Neon Tokyo" \
  --min-side-chars 3

# Generate panels page by page
python lora_code/generate_panels.py --page 1
python lora_code/generate_panels.py --page 2

# Compile final PDF
python compile_comic_pdf.py

# Export to CBZ
python -c "from comic_exporter import ComicExporter; ..."
```

### Option C — Classic LangChain mode (Mode 1)

```bash
cd indie_comic_pipeline
python run_everything.py
# → Choose "1" for LangChain Extraction Mode
# → Enter character name (e.g. "Spider-Man")
# → Enter world name (e.g. "Cyberpunk Tokyo")
# → Generate character sheet first? y
# → Choose render backend
```

### Option D — Production 10-Panel Run

```bash
cd indie_comic_pipeline
# Requires fusion JSONs in outputs/fusion/
python run_10_panel_pipeline.py
```

### Option E — Quick Doodle Test

```bash
cd indie_comic_pipeline
python generate_doodle_panels.py
# Generates 8 test panels with hardcoded storyboard
```

### Option F — Web Interface

```bash
cd indie_comic_pipeline/web_interface
python app.py
# Open http://localhost:5000
```

### Option G — Train your own Story-Weaver model

```bash
cd Story-Weaver
python stage2_story_generation.py --mode train --epochs 3 --model llama
python merge.py
python evaluate.py --all --refs story_dynamic.json
```

---

## ⚙️ Configuration Reference

All pipeline settings are in [`indie_comic_pipeline/config/settings.yaml`](indie_comic_pipeline/config/settings.yaml):

```yaml
# ─── Pipeline Identity ─────────────────────────────────────
pipeline:
  name: "Indie Comic Generator"
  version: "2.0.0"
  t4_optimized: true

# ─── Model Settings ────────────────────────────────────────
models:
  sdxl:
    name: "stabilityai/stable-diffusion-xl-base-1.0"
    variant: "fp16"
    device: "cuda"
    memory_optimization: true
    cpu_offload: true          # Critical for T4 16GB VRAM
  sd15:
    name: "runwayml/stable-diffusion-v1-5"
    device: "cuda"
  lora:
    name: "artificialguybr/LineAniRedmond-LinearMangaSDXL-V2"
    trigger_words: "LineAniAF, lineart"
    adapter_scale: 0.8
  ipadapter:
    model: "ip-adapter-faceid-plusv2_sdxl"
    weight: 0.8
    enabled: false             # Only used in Mode 1

# ─── Generation Settings (T4 Optimized) ────────────────────
generation:
  default_size: {width: 768, height: 768}  # Reduced from 1024 for T4
  inference_steps: 25          # Reduced from 40 (40% faster)
  guidance_scale: 7.5
  seed: 42
  batch_size: 1                # Always 1 for T4
  enable_model_cpu_offload: true

# ─── Art Style Tokens ──────────────────────────────────────
style:
  positive_terms:
    - "clean minimalist line art"
    - "flat color palette"
    - "crisp continuous outlines"
    - "cel-shaded with no gradients"

# ─── LLM (Ollama) ──────────────────────────────────────────
langchain:
  model: "llama3.2"
  temperature: 0.3
  ollama_url: "http://localhost:11434"

# ─── Consistency Checker ───────────────────────────────────
consistency:
  device: "cpu"               # Run on CPU to save GPU VRAM
  enable_clip: false           # Heavy — disabled by default
  enable_dinov2: false         # Heavy — disabled by default
  enable_ssim: true            # Fast
  enable_edge: true            # Fast
  enable_color: false          # Fast (disabled: focus on art style)
  enable_style: true           # Fast
  threshold: 0.55
  strict_threshold: 0.70

# ─── Story-Weaver Integration (Mode 0) ─────────────────────
story_weaver:
  input_path: "../Story-Weaver/story_dynamic.json"
  character_name: "Wanderer"
  story_world: "The Abstract"
  min_side_characters: 2
  reference_mode: "panel"      # Panel 1 = anchor
  enrich_via_llm: true

# ─── T4 GPU Optimizations ──────────────────────────────────
t4_optimizations:
  enabled: true
  clear_cache_every_n_steps: 5
  enable_gc_after_each_panel: true
  use_float16: true
  resolutions:
    draft: [512, 512]
    normal: [768, 768]
    high: [1024, 1024]
  steps:
    draft: 15
    normal: 25
    high: 35
  auto_adjust: true

# ─── Fallback on OOM ───────────────────────────────────────
fallback:
  enable_cpu_fallback: true
  reduce_resolution_on_oom: true
  resolution_fallback: [512, 512]
  steps_fallback: 20
```

---

## 📦 Output Files

All outputs go inside `indie_comic_pipeline/outputs/`:

```text
outputs/
├── fusion/
│   ├── enriched_storyboard.json       # Mode 0: full cast + prompts per panel
│   ├── fusion_complete.json           # Mode 1: 10-page storyboard
│   ├── storyboard_with_emotions.json  # Mode 1: expression-annotated storyboard
│   ├── character_personality.json     # Mode 1: character traits
│   ├── story_setting.json             # Mode 1: world settings
│   └── sdxl_prompt.json              # Anchor prompt config
│
├── characters/
│   ├── character_reference.png        # Mode 1: character sheet (SDXL)
│   ├── character_reference_sdxl_lora.png
│   └── character_reference_sd15.png
│
├── comics/
│   ├── page_1_panel_sdxl_lora_1.png   # Individual panels
│   ├── page_1_panel_sdxl_lora_2.png
│   ├── page_1_layout_sdxl_lora_horizontal.png  # Horizontal strip
│   ├── page_1_layout_sdxl_lora_grid.png        # Grid layout
│   ├── component_sheet_*.png           # Scene components
│   └── comic_book_sdxl_lora_grid.pdf   # Compiled PDF
│
├── production_run/panels/              # 10-panel production output
│
├── exports/
│   ├── Comic.cbz                       # CBZ comic archive
│   ├── Comic.cbr                       # CBR comic archive
│   └── web_comic.html                  # Scrollable web comic
│
├── audio/
│   └── dialogue_*.mp3                  # TTS audio files
│
├── comparison/
│   ├── report.html                     # Model A/B comparison report
│   ├── report.json
│   └── *_output.png                    # Per-model generated images
│
└── feedback_db.json                    # RLHF feedback database
```

---

## 🌐 Web Interface

A Flask-based web UI for interactive comic generation:

```bash
cd indie_comic_pipeline/web_interface
python app.py
# → Opens on http://localhost:5000
```

**Features:**
- Character name, world setting, and story prompt inputs
- Style selector (Manga, Western, Film Noir, Watercolor, Retro)
- Real-time generation with loading indicator
- Star rating feedback (1-5 stars + comment)
- Dark theme UI

---

## 📋 Requirements

### Story-Weaver

```
torch >= 2.0
transformers >= 4.40
accelerate
bitsandbytes
unsloth
datasets
trl
python-dotenv
rouge-score
bert-score
```

### Indie Comic Pipeline

```
torch >= 2.0
diffusers >= 0.28
transformers >= 4.40        # for CLIP, DINOv2
accelerate >= 0.30
torchvision >= 0.15
torchmetrics                 # for FID computation
safetensors
Pillow
numpy
opencv-python
scikit-image >= 0.20         # for SSIM
ultralytics >= 8.0           # for YOLOv8 (SpeechBubbleOptimizer)
PyYAML
langchain
langchain-ollama
langchain-core
gTTS >= 2.3                  # for TTS audio
flask >= 2.0                 # for web interface
rarfile >= 4.0               # for CBR export
```

> **Note for Colab:** Use `requirements_colab.txt` which omits pinned transitive dependencies to avoid conflicts with Colab's pre-installed packages.

---

## 🔧 Troubleshooting

### Ollama not connecting
```bash
ollama list                  # Check if running
ollama serve                 # Start manually
ollama pull llama3.2         # Verify the model is pulled
```

### CUDA out of memory (SDXL)
Edit `config/settings.yaml`:
```yaml
models:
  sdxl:
    memory_optimization: true    # enables CPU offload
generation:
  default_size: {width: 512, height: 512}  # reduce resolution
```
Or switch to SD 1.5 backend when prompted (uses 512×512, ~4 GB VRAM).

### `enriched_storyboard.json` not found
The enricher ran but failed silently. Run manually:
```bash
cd indie_comic_pipeline
python utils/bridge_weaver.py --enrich --input ../Story-Weaver/story_dynamic.json
```

### IP-Adapter fails to load (Mode 1)
This is non-fatal — the pipeline continues without it and generates panels using only the text prompt. Check that `h94/IP-Adapter` is accessible on HuggingFace.

### LLM returns fewer than 3 side characters
The enricher automatically pads missing side characters with contextually neutral `"Passerby N"` entries so the minimum is always met. These still produce valid SD prompts.

### Panels look inconsistent between pages
In Mode 0, consistency is checked **within** a page (panel 1 of each page is the anchor). Across pages the style may drift slightly because each page has its own anchor. To improve cross-page consistency, either:
- Use Mode 1 (IP-Adapter + static reference image)
- Or set a lower `guidance_scale` (e.g. 6.5) for more freedom in the style LoRA

### `arial.ttf` not found (Linux/Colab)
Fixed: The pipeline now falls back to `ImageFont.load_default()` when `arial.ttf` is unavailable.

### `gTTS` not installed
Fixed: `audio_integration.py` gracefully handles missing gTTS and returns `None` with a warning instead of crashing. Install with `pip install gTTS`.

### Notebooks fail to import modules
Fixed: All notebooks now include a universal setup cell that auto-configures `sys.path`. Ensure you run the first cell before any other code.

---

## 🧠 Design Decisions

| Decision | Rationale |
|----------|-----------|
| **No cloud APIs** | Everything runs locally via Ollama + HuggingFace. No data leaves the machine. |
| **Panel 1 as consistency anchor** | Eliminates the pre-generation bottleneck of a character reference sheet. The first generated panel *is* the ground truth. |
| **≥3 side characters enforced** | Comic panels feel cinematic with an ensemble. The LLM is permitted to invent bystanders if the scene is sparse, ensuring the minimum is always met. |
| **Emotion shown through objects** | The Story-Weaver prompt explicitly forbids naming emotions. This forces the model to write `hands grip the countertop` rather than `she felt angry` — producing richer visual descriptions. |
| **Recurring visual motif** | One motif (e.g. `a ceramic mug with a small chip`) appears in every panel, giving the comic a visual signature that aids style consistency without a reference image. |
| **8-metric consistency check** | Single-metric checks (e.g. SSIM alone) are brittle. Combining structure, style, semantic, and feature-level metrics (with optional color checks) gives a robust consistency signal across different artistic styles. |
| **T4-first optimization** | Default settings (768×768, 25 steps, CPU consistency models, FP16) are tuned for Google Colab's T4 GPU (16GB VRAM). |
| **Lazy imports** | Heavy modules (`diffusers`, `transformers`, `ultralytics`) are imported inside functions/methods, not at the top of files. This prevents cascade crashes when individual dependencies are missing. |
| **Dual requirements files** | `requirements.txt` (pinned, reproducible) for local dev; `requirements_colab.txt` (ranges, no transitive pins) for Colab compatibility. |

---

*Built with Ollama · LangChain · Diffusers · SDXL · LoRA · CLIP · DINOv2 · YOLOv8 · gTTS · Flask · Qwen2.5 · Python*
