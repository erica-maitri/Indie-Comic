# Local AI Indie Comic Generator Pipeline with Emotion Recognition

A local, multi-modal generative AI pipeline that takes a character name and a setting name, extracts character personality and story parameters using a local LLM, maps dialogue emotions to visual expressions, and renders consistent indie-comic-style panel layouts using Stable Diffusion XL (SDXL) and Stable Diffusion v1.5.

---

## 🎨 System Architecture

```mermaid
graph TD
    A[User Inputs] --> B[LangChain Extraction Pipeline]
    B --> B1[character_extractor.py]
    B --> B2[story_extractor.py]
    B --> B3[fusion_engine.py]
    B3 --> C[Emotion Recognition ERC Engine]
    C --> C1[emotion_recognition_engine.py]
    C1 --> D[Master Storyboard storyboard_with_emotions.json]
    
    D --> E[Master Orchestrator run_everything.py]
    E --> F[Select Generation Mode]
    
    F --> F1[1. Standard Component Assets]
    F --> F2[2. Emotion-Aware Comic Panels]
    
    F1 --> G1[Generate Character Profile]
    F1 --> G2[Generate Core Assets]
    
    F2 --> H1[Generate Panels by Page]
    F2 --> H2[Compile Horizontal/Grid Layouts]
    
    G2 --> I[Multi-Metric Consistency Checker]
    H2 --> I
```

---

## 📘 Comprehensive Architecture Breakdown

This pipeline is split into distinct functional phases. Below is a detailed breakdown of each script/module, explaining **why** it was utilized, **how** it operates, and the mathematical or system design principles backing it.

---

### 1. LangChain Extraction Phase
Located in the [langchain_code](file:///c:/Users/Dell/Downloads/drid/indie_comic_pipeline/langchain_code/) folder:

#### A. Character Personality Extractor (`character_extractor.py`)
* **Why Utilized:** Zero-shot Stable Diffusion runs often result in generic characters without unique behaviors, traits, or emotional cues. To fix this, we need structured character data (e.g. traits, values, catchphrases) to guide the story setting and subsequent panel details.
* **How It Is Used:** Uses ChatOllama loaded with `llama3.2` to prompt the local LLM. It forces the output into a strict JSON schema (representing traits, values, motivations, flaws) using Regex parsing to find the JSON block. This outputs a structured `character_personality.json` file.

#### B. Story Setting Extractor (`story_extractor.py`)
* **Why Utilized:** Visual consistency in comics requires matching backgrounds, specific lighting setups, and consistent color palettes (e.g. neon shades for cyberpunk or sepia for noir).
* **How It Is Used:** Queries Llama 3.2 to extract the environment settings (lighting, era, weather, mood, signature colors). The output is structured and saved in `story_setting.json`.

#### C. Crossover Fusion & Storyboarder (`fusion_engine.py`)
* **Why Utilized:** This is the visual director of the pipeline. It merges the character's core personality with the story setting, designs the character's clothing adaptively (e.g. Spider-Man wearing a high-collar neon-accented coat in a cyberpunk world), and drafts a detailed 10-page storyboard script.
* **How It Is Used:** The LLM receives the outputs from the previous two steps and drafts:
  1. The character's adjusted visual look matching the world style.
  2. A 10-page storyboard containing the locations, active character states, dialogues, and panel breakdowns.
  3. Four core story components/assets (main character pose, extra character, empty environment, and key prop) with specific SDXL-optimized prompts.
  Outputs are saved to `fusion_complete.json` and `sdxl_prompt.json`.

---

### 2. Dialogue Emotion Recognition (ERC) Engine
Script: [emotion_recognition_engine.py](file:///c:/Users/Dell/Downloads/drid/indie_comic_pipeline/langchain_code/emotion_recognition_engine.py)

* **Why Utilized:** Without this engine, generating comic panels directly from the storyboard script results in characters with static, blank expressions. To make the comic highly dramatic and aligned with its genre (e.g. funny, gothic, tragic, heroic), we must analyze the emotional progression of all characters and translate their dynamic states into physical, drawable facial descriptions (e.g., "brows deeply furrowed, teeth gritted in determination").
* **How It Is Used:**
  * It loads the completed storyboard script (`fusion_complete.json`).
  * **Chronological Scene Emotion Tracking (Temporal Coherence):** Loops through every panel of each page and maintains a chronological running history of the characters' previous emotional states. This history context is fed back into the prompt template (`{history_context}`), ensuring that the LLM tracks progressive shifts and escalations in emotion intensity (e.g. dialogue moving from low sadness to intense grief).
  * **Multi-Character Analysis:** Analyzes the tone, mood, and expressions of **all active characters** present in the panel (not just the protagonist), capturing their individual expression shifts.
  * **Robust JSON Auto-Closing & Repair Engine:** Local LLMs like Llama 3.2 often output truncated or trailing-comma-corrupted JSON payloads due to token limit constraints. To guarantee a **100% parse success rate**, the engine parses the response starting from the first `{`, walks the characters, manages open braces/brackets/quotes, strips dangling key-values/commas, and closes unclosed strings/structures automatically. It also runs Python's `ast.literal_eval` as a fallback if standard `json.loads` fails.
  * **Expression Mapping:** Generates expression trigger phrases and appends them to core visual prompt templates, saving the final layout directives to `storyboard_with_emotions.json`.

---

### 3. Model Generation Configurations
Comic frames can be rendered using five distinct configurations, located across three script zones:
1. **Stable Diffusion v1.5 Base (`sd15_code`):** Running runwayml Stable Diffusion v1.5 to output 512x512 grids. Used for lightweight, high-speed drafts.
2. **SD 1.5 + LoRA (`sd15_code`):** Loads SD 1.5 and tries to load LoRA styling weights. A custom error handling fallback ensures that if SDXL-specific LoRAs are passed, the pipeline falls back gracefully to base generation using prompt-based trigger keywords (`LineAniAF, lineart`).
3. **Stable Diffusion XL Base (`sdxl_code`):** Generates high-resolution 1024x1024 frames. Utilizes SDXL's larger CLIP encoder structures (CLIP ViT-L + openCLIP ViT-bigG) for better text-to-image prompt adherence.
4. **Only LoRA (SDXL + LoRA, trigger words only):** Evaluates how much the LoRA adapter changes the style without positive style descriptors, using only the core subject prompt + trigger words (`LineAniAF, lineart`).
5. **SDXL + LoRA with Style Prompts (`lora_code`):** The recommended high-end configuration. It combines the style-anchoring Lineart LoRA (`LineAniRedmond-LinearMangaSDXL-V2`) with target positive comic prompts to generate flat-colored, cel-shaded line art, completely avoiding photorealistic depth or gradients.

#### Folder Mapping:
* **[sdxl_code/](file:///c:/Users/Dell/Downloads/drid/indie_comic_pipeline/sdxl_code/):** Scripts for base SDXL model generation.
* **[sd15_code/](file:///c:/Users/Dell/Downloads/drid/indie_comic_pipeline/sd15_code/):** Scripts for base Stable Diffusion 1.5 and SD 1.5 + LoRA model generation.
* **[lora_code/](file:///c:/Users/Dell/Downloads/drid/indie_comic_pipeline/lora_code/):** Scripts for SDXL + LoRA manga styling model generation.
* **[matrix_evaluation_zone/](file:///c:/Users/Dell/Downloads/drid/indie_comic_pipeline/matrix_evaluation_zone/):** Comparison benchmark script evaluating all 5 configurations.

Each generation directory contains:
* `generate_character.py`: Renders the high-quality character reference sheet (anchor).
* `generate_components.py`: Renders visual assets (secondary character, empty scene, key prop).
* `generate_panels.py`: Renders individual panels for a selected page (1-10) using emotion prompts, compiling horizontal layouts and dynamic grid sheets.

---

### 4. Advanced Multi-Metric Consistency Suite
Module: [consistency_checker.py](file:///c:/Users/Dell/Downloads/drid/indie_comic_pipeline/utils/consistency_checker.py)

* **Why Utilized:** Standard pixel-by-pixel comparisons (MSE) fail when a character changes poses or angles. We need a holistic suite of metrics assessing color, structure, style, and semantic meaning.
* **How It Is Used:** It compares generated panels against a reference character profile using eight specific mathematical and neural techniques:

1. **Color Consistency (HSV correlation):** 
   * **Formula/Code:** Converts images to HSV space, calculates a 2D color histogram on Hue and Saturation (`cv2.calcHist([hsv], [0, 1]...)`), and computes correlation distance:
     $$d(H_1, H_2) = \frac{\sum (H_1 - \bar{H_1})(H_2 - \bar{H_2})}{\sqrt{\sum(H_1 - \bar{H_1})^2 \sum(H_2 - \bar{H_2})^2}}$$
   * **Utility:** Verifies that the colors (skin, clothing, environment) remain visually stable.
2. **SSIM (Structural Similarity Index):**
   * **Formula/Code:** Uses `scikit-image`'s structural similarity metric (luminance, contrast, and structure comparison) or a custom NumPy/OpenCV Gaussian fallback:
     $$SSIM(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
   * **Utility:** Evaluates visual layout and structural integrity to check design stability.
3. **Canny Edge Density Similarity:**
   * **Formula/Code:** Applies Canny Edge detection (`cv2.Canny`), computes the percentage of active edge pixels to total pixels, and compares edge density levels:
     $$Density = \frac{\sum(\text{Edges} > 0)}{\text{Total Pixels}}$$
   * **Utility:** Verifies that the drawing outline and stroke density match the reference style.
4. **Art Style Gram Matrix Similarity:**
   * **Formula/Code:** Resizes image to $256 \times 256$, extracts BGR color channels along with Sobel horizontal and vertical gradients, reshapes to a 5-channel feature matrix $F$, and computes the Gram Matrix representing style textures:
     $$G = F^T F / (H \times W)$$
   * **Utility:** Models the spatial texture and color patterns, reflecting drawing style consistency.
5. **CLIP Image Cosine Similarity (Semantic):**
   * **Formula/Code:** Uses a local CLIP Image Encoder (`openai/clip-vit-base-patch32`) to extract normalized 512-dimensional semantic embeddings and computes cosine similarity:
     $$\text{Similarity} = \frac{A \cdot B}{\|A\| \|B\|}$$
   * **Utility:** Operates as a semantic validator, verifying that the actual character remains Peter Parker / Spider-Man.
6. **DINOv2 Feature Similarity:**
   * **Formula/Code:** Dynamically imports Meta's self-supervised `facebook/dinov2-base` model via HuggingFace transformers, extracts global pooled class embeddings from target images, and computes cosine similarity.
   * **Utility:** Serves as a high-fidelity spatial visual structure alignment metric to check character features.
7. **Offline Aesthetic Score:**
   * **Formula/Code:** Combines laplacian variance (sharpness), standard deviation (contrast), and the Hasler-Suesstrunk metric (colorfulness).
   * **Utility:** Gauges aesthetic quality locally without calling external APIs.
8. **Grayscale Composition (Legacy structure):**
   * **Formula/Code:** Resizes grayscale thumbnails to $128 \times 128$ and calculates correlation coefficient.

* **Dynamic Scoring Weights:** The overall consistency score combines the metrics dynamically. If CLIP and DINOv2 are installed, weights are: 20% DINOv2, 15% CLIP, 15% SSIM, 15% HSV Color, 15% Style Gram Matrix, and 20% Edge Density. If models are unavailable, weights shift gracefully to structural/color fallbacks.

---

### 5. Evaluation Matrix Benchmarking
Script: [model_matrix_bench.py](file:///c:/Users/Dell/Downloads/drid/indie_comic_pipeline/matrix_evaluation_zone/model_matrix_bench.py)

* **Why Utilized:** Quantitative measurements are required to assess performance trade-offs (inference speed, memory constraints, and output quality) across all 5 configurations.
* **How It Is Used:** Runs the identical prompt under each configuration and computes:
  * **CLIP Text Score:** Text-to-image alignment cosine similarity.
  * **FID Score (Frechet Inception Distance):** Measures the distance between feature vectors of real (reference) and generated images.
  * **Inference Time:** In seconds.
  * **Peak VRAM:** Measures GPU memory allocation using `torch.cuda.max_memory_allocated()`.
  * **Edge Density:** Line-art details check.
  * **Safe Resource Release:** Addressed VRAM leakages by issuing `del pipe`, `torch.cuda.empty_cache()`, and python garbage collection (`gc.collect()`) after evaluating each pipeline model, preventing Out-Of-Memory (OOM) crashes.
  Outputs a comparison matrix table to the console.

---

### 6. Unified Master Orchestrator
Script: [run_everything.py](file:///c:/Users/Dell/Downloads/drid/indie_comic_pipeline/run_everything.py)

* **Why Utilized:** Manually running multiple terminal commands in specific order is error-prone. This script unifies everything.
* **How It Is Used:** Executes LangChain extraction, launches the ERC engine, runs environment memory checks, and prompts the user to select between:
  * **Mode 1: Standard Component Assets** (Renders character reference, secondary character, environment, and key prop).
  * **Mode 2: Emotion-Aware Comic Panels** (Renders page-by-page individual panels and dynamic grid layouts for page numbers 1-10).
  And allows selecting which model configuration to use (SDXL Base, SD 1.5, or SDXL + LoRA).

---

### 7. Google Colab Notebook Generation
Script: [create_colab_notebook.py](file:///c:/Users/Dell/Downloads/drid/indie_comic_pipeline/create_colab_notebook.py)

* **Why Utilized:** Running local SDXL on a consumer CPU can take up to 2 hours per image. Running on a free Colab GPU runtime takes 5–10 seconds.
* **How It Is Used:** Programmatically generates `indie_comic_pipeline.ipynb` containing cell configs to:
  * Install dependencies (including `scikit-image` for SSIM).
  * Launch Ollama serve daemon in the background and pull `llama3.2`.
  * Run extraction, ERC engine, panel generations, and model benchmarking.
  * Visualize generated panels and grids using IPython displays.

---

## 🚀 Execution Guide

### A. End-to-End Orchestrated run
Simply execute the master controller and follow the prompts:
```bash
python run_everything.py
```

### B. Manual Step-by-Step execution
1. **Activate Environment:**
   ```bash
   .\venv\Scripts\activate
   ```
2. **Execute LangChain & ERC scripts:**
   ```bash
   python langchain_code/character_extractor.py "Spiderman"
   python langchain_code/story_extractor.py "Cyberpunk"
   python langchain_code/fusion_engine.py
   python langchain_code/emotion_recognition_engine.py
   ```
3. **Execute Character & Panel Generators (by pipeline choice):**
   * **SDXL + LoRA (Cel-Shaded Manga style):**
     ```bash
     python lora_code/generate_character.py
     python lora_code/generate_panels.py --page 1
     ```
   * **Stable Diffusion XL Base:**
     ```bash
     python sdxl_code/generate_character.py
     python sdxl_code/generate_panels.py --page 1
     ```
   * **Stable Diffusion v1.5 + LoRA:**
     ```bash
     python sd15_code/generate_character.py
     python sd15_code/generate_panels.py --page 1
     ```
4. **Compare configurations side-by-side:**
   ```bash
   python matrix_evaluation_zone/model_matrix_bench.py
   ```
