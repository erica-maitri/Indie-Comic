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
8. [Installation](#-installation)
9. [Quick Start](#-quick-start)
10. [Configuration Reference](#-configuration-reference)
11. [Output Files](#-output-files)
12. [Requirements](#-requirements)
13. [Troubleshooting](#-troubleshooting)

---

## 🧭 Project Overview

This repository contains two tightly-integrated sub-projects:

| Sub-project | What it does |
|-------------|-------------|
| **`Story-Weaver/`** | Fine-tunes and merges LLMs to generate mood-driven comic story scripts (JSON panels with visual cues, dialogue, emotion beats, motion) |
| **`indie_comic_pipeline/`** | Receives the story JSON and renders it as visual comic panels using SDXL / SD 1.5, with an 8-metric visual consistency engine |

Together they form a pipeline where **you speak your emotion → the system writes the story → the system draws the comic**.

---

## 📁 Repository Structure

```text
drid/
├── Story-Weaver/                    # LLM story generation engine
│   ├── stage2_story_generation.py   # Main unified generation + training script
│   ├── story_gen.py                 # Quick inference entry point
│   ├── story_gen_finetuned.py       # Fine-tuned model inference
│   ├── story_gen_old.py             # Original training examples
│   ├── merge.py                     # Merges fine-tuned models into 16-bit
│   ├── evaluate.py                  # Multi-metric evaluation suite
│   ├── story_dynamic.json           # Live story output (panels JSON)
│   └── requirements.txt
│
└── indie_comic_pipeline/            # Visual rendering engine
    ├── run_everything.py            # Master orchestrator (start here)
    ├── compile_comic_pdf.py         # Assembles final PDF
    ├── config/
    │   └── settings.yaml            # All model/generation/output settings
    ├── langchain_code/
    │   ├── story_weaver_enricher.py # NEW: reference-free cast enrichment
    │   ├── character_extractor.py   # LangChain character personality parser
    │   ├── story_extractor.py       # LangChain story setting parser
    │   ├── fusion_engine.py         # LangChain crossover storyboard builder
    │   └── emotion_recognition_engine.py  # Per-panel expression mapper
    ├── utils/
    │   ├── bridge_weaver.py         # Story-Weaver JSON → pipeline converter
    │   ├── consistency_checker.py   # 8-metric visual consistency engine
    │   ├── config_helper.py         # Settings loader + path resolver
    │   ├── image_utils.py           # Strip/grid layout composer
    │   └── prompt_optimizer.py      # SD prompt builder utilities
    ├── lora_code/
    │   ├── generate_panels.py       # SDXL + LoRA panel generator
    │   ├── generate_character.py    # Character sheet generator (legacy)
    │   └── generate_components.py   # Scene component generator (legacy)
    ├── sdxl_code/
    │   ├── generate_panels.py       # SDXL base panel generator
    │   └── ...
    ├── sd15_code/
    │   ├── generate_panels.py       # SD 1.5 panel generator
    │   └── ...
    └── outputs/
        └── fusion/                  # Generated JSON files (storyboard, enriched)
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
               ┌──────────▼──────────┐
               │   LAYOUT COMPILER   │
               │  Horizontal strip   │
               │  + Dynamic grid     │
               │  + PDF export       │
               └─────────────────────┘
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

#### `langchain_code/` — The Brain

| File | Role |
|------|------|
| `story_weaver_enricher.py` | **NEW** — LLM enriches every panel with full cast (main + ≥3 side chars) and detailed scenery → `enriched_storyboard.json` |
| `character_extractor.py` | Extracts structured personality JSON from a character name (Legacy Mode) |
| `story_extractor.py` | Extracts lighting, era, color palette, mood from setting name (Legacy Mode) |
| `fusion_engine.py` | Merges character + setting into a 10-page storyboard with dialogue (Legacy Mode) |
| `emotion_recognition_engine.py` | Maps panel dialogue to facial expressions via a zero-shot LLM classifier (Legacy Mode) |

#### `utils/` — The Toolbelt

| File | Role |
|------|------|
| `bridge_weaver.py` | Converts `story_dynamic.json` to pipeline format. `--enrich` flag triggers Story-Weaver mode |
| `consistency_checker.py` | 8-metric visual similarity engine (HSV, SSIM, Gram, Edge, CLIP, DINOv2, Aesthetic, Thumbnail) |
| `config_helper.py` | Loads `settings.yaml`, resolves output paths |
| `image_utils.py` | Creates horizontal strips and dynamic grid layouts from panel images |
| `prompt_optimizer.py` | SD prompt sanitization and deduplication utilities |

#### `lora_code/`, `sdxl_code/`, `sd15_code/` — The Renderer

Three parallel panel generator backends, all sharing the same enriched-mode logic:

| Backend | Model | Resolution | Notes |
|---------|-------|-----------|-------|
| `lora_code` | SDXL + LineAniRedmond LoRA | 1024×1024 | Best style consistency, manga line-art |
| `sdxl_code` | SDXL Base | 1024×1024 | Fastest SDXL inference |
| `sd15_code` | Stable Diffusion 1.5 | 512×512 | Lightest VRAM (< 6 GB) |

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
│ LAYOUT COMPILER: image_utils.py                              │
│                                                              │
│ • Horizontal strip (all panels side by side)                 │
│ • Dynamic grid (2×2 for 4 panels, 1×N otherwise)            │
│ • Optional: compile_comic_pdf.py → final .pdf               │
└──────────────────────────────────────────────────────────────┘
```

---

## 🎯 Consistency Checking Engine

`utils/consistency_checker.py` measures visual coherence across panels using 8 independent metrics:

| Metric | Method | What It Checks |
|--------|--------|---------------|
| **HSV Color (Opt)** | Histogram comparison (OpenCV) | Same color palette (Disabled by default to focus on art style) |
| **SSIM** | Structural Similarity Index | Pixel-level structural similarity |
| **Gram Matrix** | Neural style features (VGG-19) | Consistent artistic style / texture |
| **Edge Density** | Canny edge detection | Line weight and density consistency |
| **CLIP Semantic** | CLIP image embeddings | High-level semantic/content similarity |
| **DINOv2 Structure** | DINOv2 feature maps | Deep structural/identity consistency |
| **Aesthetic Score** | CLIP-based aesthetic scorer | Panel visual quality (0.0–1.0) |
| **Thumbnail Correlation** | Pearson correlation (legacy) | Global composition similarity |

**In Mode 0:** Panel 1 of each page is set as the reference via `checker.set_reference_from_panel(panel_1_path)`. Panels 2-N are compared against it.

**In Mode 1:** The pre-generated character reference image is set via `checker.set_reference(ref_path)`.

A combined weighted score above **0.60** is considered consistent.

---

## 🛠 Installation

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10 |
| CUDA (recommended) | 11.8 or 12.1 |
| Ollama | latest |
| RAM | ≥ 16 GB |
| VRAM | ≥ 6 GB (SD 1.5) / ≥ 12 GB (SDXL) |

### 1. Clone / Open the project

```bash
cd drid
```

### 2. Create virtual environment

```bash
# Windows
python -m venv py10
py10\Scripts\activate

# Linux / macOS
python -m venv py10
source py10/bin/activate
```

### 3. Install Story-Weaver dependencies

```bash
cd Story-Weaver
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
cd ..
```

### 4. Install Pipeline dependencies

```bash
cd indie_comic_pipeline
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
# Or run the bundled installer:
python install_all.py
cd ..
```

### 5. Install and start Ollama

```bash
# Download from https://ollama.com
ollama serve              # Start the daemon
ollama pull llama3.2      # Pull the LLM used for enrichment
```

### 6. (Optional) Download fine-tuned merged model

The pre-trained Story-Weaver merged model is available at:  
📦 https://drive.google.com/drive/folders/11iTLqizx2rOP8t4RfgnbIm3FeFSg3Tsw

Place it in `Story-Weaver/moodweaver_stage2_merged/`.

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
# ...

# Compile final PDF
python compile_comic_pdf.py
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

### Option D — Train your own Story-Weaver model

```bash
cd Story-Weaver
# Fine-tune on the included training examples
python stage2_story_generation.py --mode train --epochs 3 --model llama

# Merge fine-tuned weights into a single model
python merge.py

# Evaluate the merged model
python evaluate.py --all --refs story_dynamic.json
```

---

## ⚙️ Configuration Reference

All pipeline settings are in [`indie_comic_pipeline/config/settings.yaml`](indie_comic_pipeline/config/settings.yaml):

```yaml
# Image generation models
models:
  sdxl:
    name: "stabilityai/stable-diffusion-xl-base-1.0"
    variant: "fp16"
    device: "cuda"
    memory_optimization: true

  lora:
    name: "artificialguybr/LineAniRedmond-LinearMangaSDXL-V2"
    trigger_words: "LineAniAF, lineart"

  ipadapter:
    model: "ip-adapter-faceid-plusv2_sdxl"
    weight: 0.8                    # Only used in Mode 1

# Image dimensions and sampling
generation:
  default_size: {width: 1024, height: 1024}
  inference_steps: 40
  guidance_scale: 7.5
  seed: 42

# Comic art style tokens
style:
  positive_terms:
    - "clean minimalist line art"
    - "flat color palette"
    - "crisp continuous outlines"
    - "cel-shaded with no gradients"

# Local LLM (Ollama)
langchain:
  model: "llama3.2"
  temperature: 0.3
  ollama_url: "http://localhost:11434"

# Story-Weaver integration (Mode 0)
story_weaver:
  input_path: "../Story-Weaver/story_dynamic.json"
  character_name: "Wanderer"
  story_world: "The Abstract"
  min_side_characters: 3          # LLM invents extras if scene is sparse
  reference_mode: "panel"         # "panel" = use panel 1 as anchor
  enrich_via_llm: true
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
│   └── story_setting.json             # Mode 1: world settings
│
├── characters/
│   ├── character_reference.png        # Mode 1: character sheet (SDXL)
│   └── character_reference_sd15.png   # Mode 1: character sheet (SD 1.5)
│
└── comics/
    ├── page_1_panel_sdxl_lora_1.png   # Individual panels
    ├── page_1_panel_sdxl_lora_2.png
    ├── ...
    ├── page_1_layout_sdxl_lora_horizontal.png  # Horizontal strip
    ├── page_1_layout_sdxl_lora_grid.png         # Grid layout
    └── indie_comic_final.pdf                    # Final compiled PDF
```

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
diffusers >= 0.27
langchain-ollama
langchain-core
Pillow
opencv-python
scikit-image
torchvision
transformers   # for CLIP, DINOv2
PyMuPDF        # for PDF compilation
reportlab
PyYAML
```

---

## 🔧 Troubleshooting

### Ollama not connecting
```bash
# Check if running
ollama list

# Start manually
ollama serve

# Verify the model is pulled
ollama pull llama3.2
```

### CUDA out of memory (SDXL)
Edit `config/settings.yaml`:
```yaml
models:
  sdxl:
    memory_optimization: true   # enables CPU offload
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

---

## 📐 Evaluation (Story-Weaver)

```bash
cd Story-Weaver

# Rule-based checks + hallucination detection
python evaluate.py

# Perplexity (loads the merged model)
python evaluate.py --perplexity

# ROUGE + BERTScore (requires reference stories)
python evaluate.py --nlp --refs ref1.json ref2.json

# Full evaluation suite
python evaluate.py --all --refs ref1.json ref2.json

# Compare fine-tuned vs base model
python evaluate.py --compare story_finetuned.json story_base.json --all
```

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

---

*Built with Ollama · LangChain · Diffusers · SDXL · LoRA · CLIP · DINOv2 · Qwen2.5 · Python*
