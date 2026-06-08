"""
CREATE GOOGLE COLAB NOTEBOOKS
Generates both 'indie_comic_pipeline.ipynb' and 'indie_comic_colab_full.ipynb'
with full markdown documentation,LaTeX equations, evaluation metrics,
live benchmarking, and multi-metric consistency checking.
"""

import json
import os

def build_notebook(git_clone_version=False):
    cells = []
    
    # Title & Introduction
    intro_source = [
        "# 🎨 Multi-Modal Indie Comic Generator Pipeline — Google Colab Edition\n",
        "A local, multi-modal generative AI pipeline that takes a character name and a setting name, extracts character personality and story parameters using a local LLM, maps dialogue emotions to visual expressions, and renders consistent indie-comic-style panel layouts using Stable Diffusion XL (SDXL) and Stable Diffusion v1.5.\n",
        "\n",
        "## 🛠️ System Architecture\n",
        "\n",
        "1. **LangChain Extraction Phase**: Using Ollama + Llama 3.2 to extract structured character and setting data, fusing them into a 10-page storyboard.\n",
        "2. **Dialogue Emotion Recognition (ERC) Engine**: Extracting emotional states of characters in each panel, translating them into physical expression prompts.\n",
        "3. **Multi-Model Benchmark & Selector**: Runs a live comparison of 5 configurations on 5 performance & quality metrics, recommending the best model.\n",
        "4. **Asset & Comic Panel Generation**: Renders character references, assets, and comic panels.\n",
        "5. **Advanced Consistency Suite**: Evaluates 8 visual/semantic metrics to verify character consistency.\n",
        "\n",
        "---\n",
        "⚠️ **Runtime Requirement**: Go to **Runtime > Change runtime type** and select **T4 GPU** (or any available GPU) to execute diffusion models in seconds instead of hours.\n",
        "---"
    ]
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": intro_source
    })

    # Step 1: Repo setup or upload
    if not git_clone_version:
        setup_source = [
            "### Setup Step: Upload Pipeline ZIP\n",
            "Upload your `indie_comic_pipeline.zip` to the Colab session, unzip it, and set the working directory."
        ]
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": setup_source
        })
        
        setup_code = [
            "import zipfile\n",
            "import os\n",
            "from google.colab import files\n",
            "\n",
            "print(\"📤 Upload your indie_comic_pipeline.zip file:\")\n",
            "uploaded = files.upload()\n",
            "\n",
            "for filename in uploaded.keys():\n",
            "    if filename.endswith('.zip'):\n",
            "        with zipfile.ZipFile(filename, 'r') as zip_ref:\n",
            "            zip_ref.extractall('/content/')\n",
            "        print(f\"✅ Unzipped: {filename}\")\n",
            "        break\n",
            "\n",
            "%cd /content/indie_comic_pipeline\n",
            "print(f\"📂 Current working directory: {os.getcwd()}\")"
        ]
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": setup_code
        })
    else:
        setup_source = [
            "### Setup Step: Clone Repository\n",
            "Clone the pipeline repository from GitHub, enter the sub-directory, and set up the output folders."
        ]
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": setup_source
        })
        
        setup_code = [
            "import os, subprocess\n",
            "\n",
            "REPO_URL   = \"https://github.com/Cyberpunk-San/Indie-Comic.git\"\n",
            "REPO_DIR   = \"/content/indie_comic_pipeline\"\n",
            "SUBDIR     = \"indie_comic_pipeline\"\n",
            "\n",
            "if not os.path.exists(REPO_DIR):\n",
            "    print(f\"📥 Cloning repo from {REPO_URL}...\")\n",
            "    subprocess.run([\"git\", \"clone\", \"--depth\", \"1\", REPO_URL, REPO_DIR], check=True)\n",
            "else:\n",
            "    print(\"✅ Repository already present — pulling latest changes...\")\n",
            "    subprocess.run([\"git\", \"-C\", REPO_DIR, \"pull\"], check=True)\n",
            "\n",
            "PIPELINE_ROOT = os.path.join(REPO_DIR, SUBDIR)\n",
            "os.chdir(PIPELINE_ROOT)\n",
            "print(f\"📂 Working directory set to: {os.getcwd()}\")\n",
            "\n",
            "for d in [\"outputs/fusion\", \"outputs/comics\", \"outputs/characters\"]:\n",
            "    os.makedirs(d, exist_ok=True)\n",
            "print(\"✅ Directory structure ready.\")"
        ]
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": setup_code
        })

    # Step 2: Install dependencies
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 1: Install Dependencies\n",
            "Installs required libraries including PyTorch with GPU compatibility, diffusers, accelerate, langchain, and metrics libraries."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!pip install -q diffusers transformers accelerate safetensors langchain-ollama langchain-core pyyaml opencv-python-headless pillow scikit-image peft torchmetrics torchvision matplotlib pandas"
        ]
    })

    # Step 3: Install & start Ollama
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 2: Install and Start Ollama\n",
            "Downloads Ollama, starts the daemon in the background inside the Colab session, and pulls the `llama3.2` model."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Install Ollama\n",
            "!curl -fsSL https://ollama.com/install.sh | sh\n",
            "\n",
            "# Start Ollama serve in the background\n",
            "import subprocess\n",
            "import time\n",
            "import socket\n",
            "import threading\n",
            "\n",
            "def start_ollama_server():\n",
            "    subprocess.Popen([\"ollama\", \"serve\"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n",
            "\n",
            "thread = threading.Thread(target=start_ollama_server, daemon=True)\n",
            "thread.start()\n",
            "\n",
            "# Wait until ready\n",
            "print(\"⏳ Waiting for Ollama server to respond...\")\n",
            "connected = False\n",
            "for _ in range(30):\n",
            "    try:\n",
            "        s = socket.create_connection((\"localhost\", 11434), timeout=1)\n",
            "        s.close()\n",
            "        connected = True\n",
            "        break\n",
            "    except OSError:\n",
            "        time.sleep(1.5)\n",
            "\n",
            "if connected:\n",
            "    print(\"✅ Ollama server is running on port 11434.\")\n",
            "else:\n",
            "    raise RuntimeError(\"❌ Ollama server failed to start within 45 seconds.\")\n",
            "\n",
            "# Pull Llama 3.2 model\n",
            "!ollama pull llama3.2"
        ]
    })

    # Step 4: Settings Patch
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 3: Configure Settings for Colab GPU\n",
            "Update `config/settings.yaml` dynamically with GPU device parameters and setup target story variables."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ============================================================\n",
            "#  🎭  PIPELINE CONFIGURATION  —  Edit these values\n",
            "# ============================================================\n",
            "CHARACTER_NAME = \"Spider-Man\"        # Any fictional character\n",
            "STORY_WORLD    = \"Cyberpunk 2077\"    # Any story / universe / setting\n",
            "PAGE_TO_RENDER = 1                  # Which page to render panels for (1-10)\n",
            "IMG_WIDTH      = 1024                # Resolution width (must be multiple of 8, max 1024)\n",
            "IMG_HEIGHT     = 1024                # Resolution height\n",
            "INFERENCE_STEPS = 30                # Higher = better, lower = faster (default: 30)\n",
            "GUIDANCE_SCALE = 7.5                # How closely to follow prompts\n",
            "SEED           = 42                 # Reprod seed\n",
            "OLLAMA_MODEL   = \"llama3.2\"\n",
            "\n",
            "import yaml, os\n",
            "\n",
            "with open('config/settings.yaml', 'r') as f:\n",
            "    settings = yaml.safe_load(f)\n",
            "\n",
            "# Configure settings for GPU execution of SDXL, SD v1.5 and LoRA\n",
            "settings['models']['sdxl']['device'] = 'cuda'\n",
            "settings['models']['sdxl']['name'] = 'stabilityai/stable-diffusion-xl-base-1.0'\n",
            "settings['models']['sd15']['device'] = 'cuda'\n",
            "settings['models']['sd15']['name'] = 'runwayml/stable-diffusion-v1-5'\n",
            "settings['models']['lora']['name'] = 'artificialguybr/LineAniRedmond-LinearMangaSDXL-V2'\n",
            "settings['models']['lora']['trigger_words'] = 'LineAniAF, lineart'\n",
            "settings['generation']['default_size']['width'] = IMG_WIDTH\n",
            "settings['generation']['default_size']['height'] = IMG_HEIGHT\n",
            "settings['generation']['inference_steps'] = INFERENCE_STEPS\n",
            "settings['generation']['guidance_scale'] = GUIDANCE_SCALE\n",
            "settings['generation']['seed'] = SEED\n",
            "settings['langchain']['model'] = OLLAMA_MODEL\n",
            "settings['langchain']['ollama_url'] = 'http://localhost:11434'\n",
            "\n",
            "with open('config/settings.yaml', 'w') as f:\n",
            "    yaml.safe_dump(settings, f)\n",
            "    \n",
            "print(f\"✅ settings.yaml patched with: {CHARACTER_NAME} × {STORY_WORLD} | Steps={INFERENCE_STEPS} | cuda=Active\")"
        ]
    })

    # Step 5: LangChain extraction scripts
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 4: Run LangChain Extraction Pipeline (Phase 1)\n",
            "Extracts structured character traits and setting visual definitions using LLM prompts, then fuses them into a 10-page crossover script."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import subprocess, sys, json\n",
            "\n",
            "print(\"🎭 Step 4A: Running Character Personality Extractor...\")\n",
            "subprocess.run([sys.executable, \"langchain_code/character_extractor.py\", CHARACTER_NAME], check=True)\n",
            "\n",
            "print(\"🌍 Step 4B: Running Story Setting Extractor...\")\n",
            "subprocess.run([sys.executable, \"langchain_code/story_extractor.py\", STORY_WORLD], check=True)\n",
            "\n",
            "print(\"⚗️ Step 4C: Running Crossover Fusion & Storyboarder...\")\n",
            "subprocess.run([sys.executable, \"langchain_code/fusion_engine.py\"], check=True)\n",
            "\n",
            "# Quick inspect\n",
            "with open('outputs/fusion/fusion_complete.json', 'r', encoding='utf-8') as f:\n",
            "    fusion_data = json.load(f)\n",
            "\n",
            "print(\"\\n✅ Phase 1 complete!\")\n",
            "print(f\"Adaptation Style: {fusion_data['fusion']['character_visual_looks']}\")"
        ]
    })

    # Step 6: Emotion recognition
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 4.5: Run Dialogue Emotion Recognition (ERC) Engine\n",
            "Analyzes dialogue and story beats to extract emotions, ensures temporal emotional coherence, and maps them to drawable facial expressions."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import subprocess, sys, json\n",
            "\n",
            "print(\"😮 Running Emotion Recognition Engine...\")\n",
            "subprocess.run([sys.executable, \"langchain_code/emotion_recognition_engine.py\"], check=True)\n",
            "\n",
            "with open('outputs/fusion/storyboard_with_emotions.json', 'r', encoding='utf-8') as f:\n",
            "    em_data = json.load(f)\n",
            "\n",
            "print(\"\\n✅ Emotion Recognition Complete!\")\n",
            "target = next((p for p in em_data['storyboard_with_emotions'] if p['page_number'] == PAGE_TO_RENDER), None)\n",
            "if target:\n",
            "    print(f\"Mood Preview (Page {PAGE_TO_RENDER}): {target.get('personality_state')}\")\n",
            "    for pd in target.get('panels_detail', [])[:3]:\n",
            "        print(f\"  Panel {pd['panel_number']} | Actions: {pd['core_action'][:60]}...\")\n",
            "        for c, emo in pd.get('emotions', {}).items():\n",
            "            print(f\"    - {c}: {emo.get('emotion')} | Expr: {emo.get('expression_trigger')}\")"
        ]
    })

    # Step 7: Live model benchmarking matrix
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📊 Step 5: Run Multi-Model Benchmarking & Evaluation Matrix\n",
            "Compare all **5 configurations** on **5 key performance & quality metrics** side-by-side on a live prompt:\n",
            "\n",
            "| Metric | Formula / Method | Utility |\n",
            "|---|---|---|\n",
            "| **CLIP Text Score** | $\\text{Similarity} = \\frac{A \\cdot B}{\\|A\\| \\|B\\|}$ using CLIP ViT-B/32 | Measures text-to-image prompt adherence |\n",
            "| **FID Score** | Inception-v3 features distance matrix calculation | Measures image fidelity/distance to reference |\n",
            "| **Inference Speed** | End Time - Start Time (seconds) | Measures generation latency |\n",
            "| **Peak VRAM Usage** | `torch.cuda.max_memory_allocated()` (MB) | Measures GPU hardware memory consumption |\n",
            "| **Edge Density** | Canny Edge active pixels ratio | Verifies styling stroke details |"
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys, os, pandas as pd, matplotlib.pyplot as plt\n",
            "\n",
            "# Run the benchmark suite\n",
            "from matrix_evaluation_zone.model_matrix_bench import (\n",
            "    run_stable_diffusion_v15,\n",
            "    run_stable_diffusion_v15_with_lora,\n",
            "    run_stable_diffusion_xl,\n",
            "    run_stable_diffusion_xl_only_lora,\n",
            "    run_stable_diffusion_xl_with_lora,\n",
            "    compute_clip_score,\n",
            "    compute_real_fid_score,\n",
            "    compute_edge_density,\n",
            "    bench_prompt,\n",
            "    core_prompt,\n",
            "    lora_config\n",
            ")\n",
            "\n",
            "print(\"⏳ Running live multi-model benchmark matrix... (takes ~2 minutes on T4 GPU)\")\n",
            "sd15_path, sd15_inf_time, sd15_vram = run_stable_diffusion_v15()\n",
            "sd15_clip = compute_clip_score(sd15_path, bench_prompt)\n",
            "sd15_fid = compute_real_fid_score(sd15_path)\n",
            "sd15_edges = compute_edge_density(sd15_path)\n",
            "\n",
            "sd15_lora_path, sd15_lora_inf_time, sd15_lora_vram = run_stable_diffusion_v15_with_lora()\n",
            "sd15_lora_clip = compute_clip_score(sd15_lora_path, bench_prompt)\n",
            "sd15_lora_fid = compute_real_fid_score(sd15_lora_path)\n",
            "sd15_lora_edges = compute_edge_density(sd15_lora_path)\n",
            "\n",
            "sdxl_path, sdxl_inf_time, sdxl_vram = run_stable_diffusion_xl()\n",
            "sdxl_clip = compute_clip_score(sdxl_path, bench_prompt)\n",
            "sdxl_fid = compute_real_fid_score(sdxl_path)\n",
            "sdxl_edges = compute_edge_density(sdxl_path)\n",
            "\n",
            "only_lora_path, only_lora_inf_time, only_lora_vram = run_stable_diffusion_xl_only_lora()\n",
            "trigger_words = lora_config.get(\"trigger_words\", \"LineAniAF, lineart\")\n",
            "only_lora_clip = compute_clip_score(only_lora_path, f\"{core_prompt}, {trigger_words}\")\n",
            "only_lora_fid = compute_real_fid_score(only_lora_path)\n",
            "only_lora_edges = compute_edge_density(only_lora_path)\n",
            "\n",
            "sdxl_lora_path, sdxl_lora_inf_time, sdxl_lora_vram = run_stable_diffusion_xl_with_lora()\n",
            "sdxl_lora_clip = compute_clip_score(sdxl_lora_path, bench_prompt)\n",
            "sdxl_lora_fid = compute_real_fid_score(sdxl_lora_path)\n",
            "sdxl_lora_edges = compute_edge_density(sdxl_lora_path)\n",
            "\n",
            "# Compile comparison DataFrame\n",
            "data = {\n",
            "    \"Configuration\": [\n",
            "        \"Stable Diffusion v1.5\",\n",
            "        \"SD 1.5 + LoRA\",\n",
            "        \"Stable Diffusion XL (Base)\",\n",
            "        \"Only LoRA (SDXL + No Prompts)\",\n",
            "        \"SDXL + LoRA (With Prompts)\"\n",
            "    ],\n",
            "    \"CLIP Text Score\": [sd15_clip, sd15_lora_clip, sdxl_clip, only_lora_clip, sdxl_lora_clip],\n",
            "    \"FID Score\": [sd15_fid, sd15_lora_fid, sdxl_fid, only_lora_fid, sdxl_lora_fid],\n",
            "    \"Inference Time (sec)\": [sd15_inf_time, sd15_lora_inf_time, sdxl_inf_time, only_lora_inf_time, sdxl_lora_inf_time],\n",
            "    \"Peak VRAM (MB)\": [sd15_vram, sd15_lora_vram, sdxl_vram, only_lora_vram, sdxl_lora_vram],\n",
            "    \"Edge Density (%)\": [sd15_edges, sd15_lora_edges, sdxl_edges, only_lora_edges, sdxl_lora_edges]\n",
            "}\n",
            "\n",
            "df = pd.DataFrame(data)\n",
            "print(\"\\n📊 Comparative Evaluation Matrix:\")\n",
            "display(df)\n",
            "\n",
            "# Plotting comparisons\n",
            "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
            "df.plot(x=\"Configuration\", y=\"CLIP Text Score\", kind=\"bar\", ax=axes[0,0], color=\"skyblue\", rot=30)\n",
            "axes[0,0].set_title(\"CLIP Text Score (Higher is Better)\")\n",
            "df.plot(x=\"Configuration\", y=\"FID Score\", kind=\"bar\", ax=axes[0,1], color=\"salmon\", rot=30)\n",
            "axes[0,1].set_title(\"FID Score (Lower is Better)\")\n",
            "df.plot(x=\"Configuration\", y=\"Inference Time (sec)\", kind=\"bar\", ax=axes[1,0], color=\"lightgreen\", rot=30)\n",
            "axes[1,0].set_title(\"Inference Speed (Lower is Better)\")\n",
            "df.plot(x=\"Configuration\", y=\"Peak VRAM (MB)\", kind=\"bar\", ax=axes[1,1], color=\"orchid\", rot=30)\n",
            "axes[1,1].set_title(\"Peak VRAM Usage (Lower is Better)\")\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })

    # Step 8: Interactive selection
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 5.1: Composite Recommendation & Model Selector\n",
            "Computes a composite mathematical score to recommend the best model matching quality & resource constraints."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "max_time = df[\"Inference Time (sec)\"].max()\n",
            "max_vram = df[\"Peak VRAM (MB)\"].max()\n",
            "max_fid = df[\"FID Score\"].max()\n",
            "\n",
            "# Normalized Composite Score calculation\n",
            "df[\"Composite Score\"] = (\n",
            "    0.3 * df[\"CLIP Text Score\"] +\n",
            "    0.3 * (1.0 - df[\"FID Score\"] / (max_fid if max_fid > 0 else 1.0)) +\n",
            "    0.2 * (1.0 - df[\"Inference Time (sec)\"] / (max_time if max_time > 0 else 1.0)) +\n",
            "    0.2 * (1.0 - df[\"Peak VRAM (MB)\"] / (max_vram if max_vram > 0 else 1.0))\n",
            ")\n",
            "\n",
            "best_idx = df[\"Composite Score\"].idxmax()\n",
            "recommended_config = df.loc[best_idx, \"Configuration\"]\n",
            "print(f\"★ Recommended Configuration: {recommended_config} (Score: {df.loc[best_idx, 'Composite Score']:.3f})\")\n",
            "\n",
            "choice_map = {\n",
            "    \"Stable Diffusion XL (Base)\": 1,\n",
            "    \"Stable Diffusion v1.5\": 2,\n",
            "    \"SD 1.5 + LoRA\": 2,\n",
            "    \"Only LoRA (SDXL + No Prompts)\": 3,\n",
            "    \"SDXL + LoRA (With Prompts)\": 3\n",
            "}\n",
            "default_choice = choice_map.get(recommended_config, 3)\n",
            "\n",
            "# Select configuration programmatically or via user input prompt\n",
            "print(\"\\nSelect Model Configuration for final assets & panels generation:\")\n",
            "print(\"  1 = SDXL Base\")\n",
            "print(\"  2 = Stable Diffusion v1.5\")\n",
            "print(\"  3 = SDXL + LoRA (Manga Style Cel-shaded - Recommended)\")\n",
            "\n",
            "try:\n",
            "    val = input(f\"Enter model choice [1, 2, or 3, default={default_choice}]: \").strip()\n",
            "    SELECTED_MODEL = int(val) if val else default_choice\n",
            "except Exception:\n",
            "    SELECTED_MODEL = default_choice\n",
            "\n",
            "print(f\"\\n🚀 Confirmed selected configuration index: {SELECTED_MODEL}\")"
        ]
    })

    # Step 9: Asset & character sheet generation
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 6: Generate Character Profile & Component Assets\n",
            "Generates the character profile sheet (used as the anchor reference) along with 4 story component assets (secondary character, environment background, key prop)."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import subprocess, sys\n",
            "\n",
            "print(f\"🎨 Running generation pipeline using selected model config index: {SELECTED_MODEL}...\")\n",
            "if SELECTED_MODEL == 2:\n",
            "    subprocess.run([sys.executable, \"sd15_code/run_sd15_pipeline.py\"], check=True)\n",
            "elif SELECTED_MODEL == 3:\n",
            "    subprocess.run([sys.executable, \"lora_code/run_lora_pipeline.py\"], check=True)\n",
            "else:\n",
            "    subprocess.run([sys.executable, \"sdxl_code/run_sdxl_pipeline.py\"], check=True)\n",
            "\n",
            "from IPython.display import Image, display\n",
            "import os\n",
            "\n",
            "refs = [\n",
            "    \"outputs/characters/character_reference.png\",\n",
            "    \"outputs/characters/character_reference_sd15.png\",\n",
            "    \"outputs/characters/character_reference_sdxl_lora.png\"\n",
            "]\n",
            "ref_found = next((r for r in refs if os.path.exists(r)), None)\n",
            "if ref_found:\n",
            "    print(f\"\\n🖼️ Anchor Character Reference Profile Sheet ({ref_found}):\")\n",
            "    display(Image(ref_found))\n",
            "else:\n",
            "    print(\"⚠️ Reference character sheet image could not be loaded.\")"
        ]
    })

    # Step 10: Consistency Checker on components (8 metrics)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 6.1: Advanced Consistency Suite Validation\n",
            "Evaluates character and asset visual coherence against the anchor reference using **8 specific mathematical & neural metrics**:\n",
            "\n",
            "1. **Color Consistency**: Hue & Saturation HSV space correlation ($d(H_1, H_2)$)\n",
            "2. **SSIM**: Structural Similarity Index (luminance, contrast, structural mapping)\n",
            "3. **Canny Edge Density**: Compares drawing stroke active pixels ratio\n",
            "4. **Art Style Gram Matrix**: Texture texture style correlations on Sobel gradients\n",
            "5. **CLIP Image Similarity**: 512-dimensional semantic visual visual validation\n",
            "6. **DINOv2 Feature Similarity**: High-fidelity spatial transformer representation alignment\n",
            "7. **Offline Aesthetic Score**: Combines colorfulness, contrast, and sharpness locally\n",
            "8. **Grayscale Correlation**: Legacy structure coefficient benchmark"
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, glob, pandas as pd\n",
            "from utils.consistency_checker import get_consistency_checker\n",
            "\n",
            "checker = get_consistency_checker()\n",
            "refs = [\n",
            "    \"outputs/characters/character_reference.png\",\n",
            "    \"outputs/characters/character_reference_sd15.png\",\n",
            "    \"outputs/characters/character_reference_sdxl_lora.png\"\n",
            "]\n",
            "ref_found = next((r for r in refs if os.path.exists(r)), None)\n",
            "components = sorted(glob.glob(\"outputs/comics/component_*.png\"))\n",
            "\n",
            "if ref_found and components:\n",
            "    print(f\"🔍 Running full Consistency Suite with anchor: {ref_found}\")\n",
            "    checker.set_reference(ref_found)\n",
            "    \n",
            "    results = []\n",
            "    for path in components:\n",
            "        res = checker.check_consistency(path)\n",
            "        results.append({\n",
            "            \"File\": os.path.basename(path),\n",
            "            \"Overall Score\": f\"{res['score']:.3f}\",\n",
            "            \"HSV Color Match\": f\"{res['color_score']:.3f}\",\n",
            "            \"SSIM Structure\": f\"{res['ssim_score']:.3f}\",\n",
            "            \"Edge Density Match\": f\"{res['edge_score']:.3f}\",\n",
            "            \"Style Gram Match\": f\"{res['style_score']:.3f}\",\n",
            "            \"Aesthetic Score\": f\"{res['aesthetic_score']:.3f}\",\n",
            "            \"CLIP Semantic\": f\"{res['clip_img_score']:.3f}\" if res['clip_img_score'] is not None else \"N/A\",\n",
            "            \"DINOv2 Structural\": f\"{res['dinov2_score']:.3f}\" if res['dinov2_score'] is not None else \"N/A\",\n",
            "            \"Consistent\": \"✅ Yes\" if res['consistent'] else \"❌ No\"\n",
            "        })\n",
            "        \n",
            "    df_cons = pd.DataFrame(results)\n",
            "    display(df_cons)\n",
            "else:\n",
            "    print(\"⚠️ Missing generated assets or anchor reference sheet. Run step 6 first.\")"
        ]
    })

    # Step 11: Comic panel generation
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 7: Generate Emotion-Aware Comic Panels for Storyboard Page\n",
            "Renders each panel for the configured page, appending LLM dialogue expressions, and compiles them into final strip & grid layouts."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import subprocess, sys\n",
            "\n",
            "print(f\"🎬 Rendering Storyboard Page {PAGE_TO_RENDER} panels layout...\")\n",
            "if SELECTED_MODEL == 2:\n",
            "    subprocess.run([sys.executable, \"sd15_code/generate_panels.py\", \"--page\", str(PAGE_TO_RENDER)], check=True)\n",
            "elif SELECTED_MODEL == 3:\n",
            "    subprocess.run([sys.executable, \"lora_code/generate_panels.py\", \"--page\", str(PAGE_TO_RENDER)], check=True)\n",
            "else:\n",
            "    subprocess.run([sys.executable, \"sdxl_code/generate_panels.py\", \"--page\", str(PAGE_TO_RENDER)], check=True)\n",
            "\n",
            "print(f\"\\n✅ Page {PAGE_TO_RENDER} comic panel generation completed successfully!\")"
        ]
    })

    # Step 12: Visualise panels & grids
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 7.1: Visualise Comic Page Layout Grids"
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, glob\n",
            "from PIL import Image\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "comics_dir = \"outputs/comics\"\n",
            "layout_grids = glob.glob(os.path.join(comics_dir, f\"page_{PAGE_TO_RENDER}_layout_*_grid.png\"))\n",
            "\n",
            "if layout_grids:\n",
            "    for grid_path in sorted(layout_grids):\n",
            "        print(f\"🎨 Displaying Compiled Grid: {grid_path}\")\n",
            "        img = Image.open(grid_path)\n",
            "        fig, ax = plt.subplots(figsize=(12, 12))\n",
            "        ax.imshow(img)\n",
            "        ax.axis(\"off\")\n",
            "        plt.show()\n",
            "else:\n",
            "    # Fallback individual panels display\n",
            "    panels = sorted(glob.glob(os.path.join(comics_dir, f\"page_{PAGE_TO_RENDER}_panel_*.png\")))\n",
            "    if panels:\n",
            "        n = len(panels)\n",
            "        print(f\"✅ Displaying {n} individual panels:\")\n",
            "        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))\n",
            "        if n == 1:\n",
            "            axes = [axes]\n",
            "        for ax, pf in zip(axes, panels):\n",
            "            ax.imshow(Image.open(pf))\n",
            "            ax.set_title(os.path.basename(pf), fontsize=8)\n",
            "            ax.axis(\"off\")\n",
            "        plt.tight_layout()\n",
            "        plt.show()\n",
            "    else:\n",
            "        print(\"⚠️ No generated layout grids or panel images found. Verify Step 7 outputs.\")"
        ]
    })

    # Step 13: Consistency Suite checker across panels
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 7.2: Verify Comic Panels Character Consistency\n",
            "Runs the Consistency Suite on the generated page panels to verify character visual integrity."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, glob, pandas as pd\n",
            "from utils.consistency_checker import get_consistency_checker\n",
            "\n",
            "checker = get_consistency_checker()\n",
            "refs = [\n",
            "    \"outputs/characters/character_reference.png\",\n",
            "    \"outputs/characters/character_reference_sd15.png\",\n",
            "    \"outputs/characters/character_reference_sdxl_lora.png\"\n",
            "]\n",
            "ref_found = next((r for r in refs if os.path.exists(r)), None)\n",
            "panel_paths = sorted(glob.glob(f\"outputs/comics/page_{PAGE_TO_RENDER}_panel_*.png\"))\n",
            "\n",
            "if ref_found and panel_paths:\n",
            "    print(f\"🔍 Verifying character consistency across panels utilizing anchor ref: {ref_found}\")\n",
            "    checker.set_reference(ref_found)\n",
            "    \n",
            "    results = []\n",
            "    for path in panel_paths:\n",
            "        res = checker.check_consistency(path)\n",
            "        results.append({\n",
            "            \"Panel\": os.path.basename(path),\n",
            "            \"Consistency score\": f\"{res['score']:.3f}\",\n",
            "            \"HSV Color Match\": f\"{res['color_score']:.3f}\",\n",
            "            \"SSIM structure\": f\"{res['ssim_score']:.3f}\",\n",
            "            \"Edge Density Match\": f\"{res['edge_score']:.3f}\",\n",
            "            \"Style Gram Match\": f\"{res['style_score']:.3f}\",\n",
            "            \"Aesthetic Score\": f\"{res['aesthetic_score']:.3f}\",\n",
            "            \"CLIP image match\": f\"{res['clip_img_score']:.3f}\" if res['clip_img_score'] is not None else \"N/A\",\n",
            "            \"DINOv2 similarity\": f\"{res['dinov2_score']:.3f}\" if res['dinov2_score'] is not None else \"N/A\",\n",
            "            \"Consistent\": \"✅ Yes\" if res['consistent'] else \"❌ No\"\n",
            "        })\n",
            "    df_pan_cons = pd.DataFrame(results)\n",
            "    display(df_pan_cons)\n",
            "else:\n",
            "    print(\"⚠️ No panel images found or reference sheet missing.\")"
        ]
    })

    # Step 14: Doodle panels generator
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 8: Generate Custom Doodle Storyboard (Optional Test)\n",
            "Runs the custom doodle generator script to create test storyboard frames and compile them into a grid sheet layout."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"✏️ Generating custom doodle panels storyboard...\")\n",
            "!python generate_doodle_panels.py\n",
            "\n",
            "import os\n",
            "from PIL import Image\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "doodle_grid = \"outputs/comics/doodle_story_layout_grid.png\"\n",
            "if os.path.exists(doodle_grid):\n",
            "    print(\"🎨 Custom Doodle Layout Grid Sheet:\")\n",
            "    img = Image.open(doodle_grid)\n",
            "    fig, ax = plt.subplots(figsize=(12, 12))\n",
            "    ax.imshow(img)\n",
            "    ax.axis(\"off\")\n",
            "    plt.show()\n",
            "else:\n",
            "    print(\"⚠️ Doodle storyboard layout grid image not found.\")"
        ]
    })

    # Step 15: Download outputs
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 9: Download All Generated Outputs\n",
            "Creates a consolidated ZIP archive of all output files and triggers the browser download."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, zipfile\n",
            "from google.colab import files\n",
            "\n",
            "ZIP_PATH = \"/content/indie_comic_outputs.zip\"\n",
            "print(\"📦 Packaging outputs folder into ZIP archive...\")\n",
            "\n",
            "with zipfile.ZipFile(ZIP_PATH, \"w\", zipfile.ZIP_DEFLATED) as zf:\n",
            "    for root, dirs, fnames in os.walk(\"outputs\"):\n",
            "        for fname in fnames:\n",
            "            fpath = os.path.join(root, fname)\n",
            "            arcname = os.path.relpath(fpath, os.path.dirname(\"outputs\"))\n",
            "            zf.write(fpath, arcname)\n",
            "\n",
            "size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)\n",
            "print(f\"✅ ZIP created: {ZIP_PATH} ({size_mb:.1f} MB)\")\n",
            "print(\"⬇️ Initiating browser download...\")\n",
            "files.download(ZIP_PATH)"
        ]
    })

    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {
                "provenance": [],
                "gpuType": "T4",
                "name": "Indie Comic Pipeline — Google Colab Edition"
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "language_info": {
                "name": "python"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    return notebook

# Programmatically build both notebooks
script_dir = os.path.dirname(os.path.abspath(__file__))

# 1. indie_comic_pipeline.ipynb (ZIP Upload edition)
notebook_pipeline = build_notebook(git_clone_version=False)
pipeline_path = os.path.join(script_dir, "indie_comic_pipeline.ipynb")
with open(pipeline_path, "w", encoding="utf-8") as f:
    json.dump(notebook_pipeline, f, indent=2)
print(f"Created Colab Notebook: {pipeline_path}")

# 2. indie_comic_colab_full.ipynb (Git Clone auto edition)
notebook_full = build_notebook(git_clone_version=True)
full_path = os.path.join(script_dir, "indie_comic_colab_full.ipynb")
with open(full_path, "w", encoding="utf-8") as f:
    json.dump(notebook_full, f, indent=2)
print(f"Created Colab Notebook: {full_path}")
