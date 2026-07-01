# Cyberpunk-San / Indie Comic Generator 🚀

> **A unified, local end-to-end AI comic publishing system** that maps a multilingual user prompt to a structured narrative script, orchestrates a swarm of director agents, enforces frame-to-frame visual consistency, and renders compiled multi-panel comics.
>
> This repository integrates three core subsystems:
> 1. **Mood Weaver**: A multilingual emotion classifier (XLM-RoBERTa) that identifies the prompt's core emotional state.
> 2. **Story Weaver**: A fine-tuned narrative storyboard generator (Qwen2.5/Mistral SFT) that designs rich character plots and panel-by-panel action flows.
> 3. **Indie Comic Pipeline**: A multi-phase rendering engine that manages visual consistency, text integration, quality critiques, and layout assembly.
>
> All systems share a single source of configuration truth: [arcs_config.json](indie_comic_pipeline/config/arcs_config.json).

---


## 🏗 System Architecture (The 8 Phases)

The pipeline uses a rigorous, sequential **8-Phase Architecture** to turn an emotion into a fully formatted comic:

### [PHASE 0: INTELLIGENT STORY INTAKE (WRITER'S ROOM)]
* **Action:** User prompts (character, world, traits, reference) are passed to local Ollama (default: llama3.2).
* **Processing:** Runs a single-pass JSON-structured LLM chain to generate both a 'Story Bible' and a detailed hierarchical scene graph script (characters, actions, camera, environment).
* **Output:** Generates an in-memory structured configuration dictionary (`story_config`).

### [PHASE 1: NARRATIVE PLANNING LAYER]
* **Action:** Director Swarm (Multi-Agent System) parses and loads the scene graph into memory.
    * **Story Director:** Registers characters and sets up the base panel sequence.
    * **Action Director:** Parses character actions, physical verbs, and interactions.
    * **Dialogue Writer:** Details dialogue text, tone, and vector speech bubble categories.
    * **Pose Director:** Identifies body language posture/stance mapping constraints.
    * **Emotion Director:** Resolves granular facial taxonomies (eyes, mouth, expressions).
    * **Camera Director:** Dictates framing layouts (camera angle, environment variables).
* **Shared Storage:** All agents read from and write to the shared **Story Section Memory** (Explicit RAM Blackboard), which tracks active character states, visual anchors, scene settings, and layout plans.

### [PHASE 2: REFERENCE-FREE ANCHORING]
* **Step 2.1:** Pulls initial context prompts from memory to execute Panel 1 Generation.
* **Step 2.2:** Isolates Panel 1 to serve as the baseline Primary Visual Anchor.
* **Step 2.3:** Runs Identity Embedding Extraction to capture raw facial topology, wardrobe, and style markers.
* **Step 2.4:** Injects extracted identity tracking tokens directly back into the Story Section Memory cache.

### [PHASE 3 & 4: IN-GENERATION CONSISTENCY & COMPOSABLE CONTROL]
* **Action:** Sequentially generates panels 2 through N by pulling context and tokens from memory.
    * **CharCom Inference Compositor:** Calculates dynamic model weight blending at runtime: `W_total = W_base + Σ(α_i * W_i)`
    * **Multi-Backend Diffusion Denoising Stack (SDXL / Flux / Video DiT):**
        * **Level 1:** Physics-Informed Attention (RealDiffusion). Injects dissipative heat diffusion priors to suppress high-frequency noise drift.
        * **Level 2:** Shared Attention Matrix Masking (Accelerated TF). Applies cross-prompt masking to lock character identity keys/values across frames.
        * **Level 3:** Spatiotemporal Architectural Priors (DreamingComics). Inherits structural motion window constraints from native video transformer processing.

### [PHASE 5: INTEGRATED TEXT-IMAGE GENERATION]
* **Action:** Processed latents are fed into the DiffSensei MLLM Domain.
* **Processing:** Merges image matrices and script dialogue inside a single Unified Multimodal Semantic Space.
* **Execution:** Binds language vectors directly into cross-attention loops to render dynamic expressions/poses, outputting a Raw Panel Raster Sheet.

### [PHASE 6: QUALITY VALIDATION LAYER]
* **Action:** Raw panel imagery is intercepted by the COMIC Critic Pipeline.
* **Processing:** Evaluates performance across an evolutionary ring of human-aligned LLM Critics.
    * **IF Composite Score < Quality Performance Threshold:** Adjusts guidance scale parameters, updates prompting weights, and triggers Regeneration Loop.
    * **IF Composite Score >= Quality Performance Threshold:** Approves frame data and caches the cleared panel asset straight to the assembly stack.

### [PHASE 7: LAYOUT & ASSEMBLY]
* **Action:** MangaFlow Engine takes the approved panels and layout parameters from memory.
* **Processing:** Dynamically cuts border geometry channels depending on scene action intensity.
* **Execution:** Runs typesetting algorithms to lock vector narrative speech bubbles based on focal subjects, creating the Compiled Master Sheet Layout.

### [PHASE 8: EXPORT MODULE & ADAPTIVE RLHF SYSTEMS]
* **File Compilation:** Compiles raw layout vectors into final reader formats (PDF / CBZ / HTML).
* **Human Alignment Telemetry Loop:** Gathers explicit user interface performance rankings and rating feedback metrics.
* **System Backpropagation Optimization:** Executes weight adjustments to fine-tune COMIC LLM Critic evaluations and mutate prompt generation templates.

---

## 🛠 Installation & Usage (Docker)

The entire environment is containerized using `python:3.10-slim` with CPU-optimized PyTorch wheels. You do **not** need an NVIDIA GPU to run the dry-run pipeline tests!

### 1. Build and Start the Environment
```bash
docker compose up -d --build
```
*Note: The first build will take several minutes as it downloads PyTorch CPU wheels and resolves dependencies.*

### 2. Enter the Container
```bash
docker compose exec comic-generator bash
```

### 3. Run the Integrated Pipeline
To run the full 8-phase pipeline end-to-end (defaults to dry-run / CPU Mock Backend):
```bash
python indie_comic_pipeline/integrated_pipeline.py
```
To run the pipeline with GPU diffusion generation instead of the mock backend:
```bash
# Update config/settings.yaml to set dry_run: false, then run the script normally
```

*Outputs will be saved in `outputs/panels/`, `outputs/comics/`, and `outputs/anchors/`.*

### 4. Run the Verification Suite
To verify the integrity of all 8 phases, advanced attention logic, and memory state serialization:
```bash
python indie_comic_pipeline/scratch/run_unit_tests.py
```
This runs a test suite across the codebase to ensure zero regressions.

---

## 🧠 Configuration

Modify `indie_comic_pipeline/config/settings.yaml` to change:
- Generation models (SDXL vs Mock)
- Agent API endpoints (Ollama / Local LLMs)
- Consistency checker thresholds
- Advanced Attention blend ratios

---

## 📁 Output Files

When a run completes, you will find:
- `outputs/panels/`: Raw generated individual panel images.
- `outputs/comics/`: Final CBZ archives, PDF comic books, and HTML Web Comics.
- `outputs/anchors/`: The extracted identity tokens from Panel 1.
- `outputs/rlhf_feedback.json`: RLHF loop optimization data.

---

*Built for the future of interactive storytelling.*
