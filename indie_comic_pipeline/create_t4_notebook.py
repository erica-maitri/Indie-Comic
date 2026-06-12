"""
CREATE T4 OPTIMIZED COLAB NOTEBOOK
Generates a single, complete notebook optimized for T4 GPU execution
All models cache between cells for maximum speed
"""

import json
import os

def create_t4_optimized_notebook():
    """Generate a single comprehensive notebook for T4 GPU"""
    
    cells = []
    
    # ============================================================
    # CELL 1: Title and Introduction
    # ============================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🎨 Indie Comic Generator - T4 Optimized Single Notebook\n",
            "\n",
            "## ⚡ Complete pipeline in one notebook - models cache between cells!\n",
            "\n",
            "### Features:\n",
            "- **T4 GPU Optimized**: 768x768 resolution, 25 inference steps\n",
            "- **Model Caching**: SDXL loads once, reused for all pages\n",
            "- **Memory Management**: Auto-cleanup every 3 pages\n",
            "- **Complete Pipeline**: Extraction → Fusion → Generation → PDF\n",
            "\n",
            "⚠️ **Select Runtime > Change runtime type > T4 GPU before starting**\n",
            "\n",
            "---"
        ]
    })
    
    # ============================================================
    # CELL 2: Configuration - EDIT THESE VALUES
    # ============================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📝 Configuration\n",
            "Edit the values below for your story:"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ============================================================\n",
            "# EDIT THESE VALUES\n",
            "# ============================================================\n",
            "CHARACTER_NAME = \"Spider-Man\"      # Any character (Wolverine, Batman, etc.)\n",
            "STORY_WORLD = \"Cyberpunk 2077\"     # Any setting (Harry Potter, Wuthering Heights, etc.)\n",
            "NUM_PAGES = 5                      # Pages to generate (1-10)\n",
            "USE_LORA = True                    # True = SDXL+LoRA (best), False = SDXL Base\n",
            "\n",
            "# Advanced settings (usually leave as is)\n",
            "IMG_WIDTH = 768                    # T4 optimized (512-1024)\n",
            "IMG_HEIGHT = 768                   # T4 optimized (512-1024)\n",
            "INFERENCE_STEPS = 25               # T4 optimized (15-40)\n",
            "\n",
            "print(f\"🎭 Character: {CHARACTER_NAME}\")\n",
            "print(f\"🌍 World: {STORY_WORLD}\")\n",
            "print(f\"📄 Pages: {NUM_PAGES}\")\n",
            "print(f\"🎨 Model: {'SDXL + LoRA' if USE_LORA else 'SDXL Base'}\")\n",
            "print(f\"📐 Resolution: {IMG_WIDTH}x{IMG_HEIGHT}\")\n",
            "print(f\"⚡ Steps: {INFERENCE_STEPS}\")"
        ]
    })
    
    # ============================================================
    # CELL 3: Install Dependencies
    # ============================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📦 Step 1: Install Dependencies\n",
            "This will install all required packages (takes 2-3 minutes)"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!pip install -q diffusers transformers accelerate safetensors \\\n",
            "    langchain-ollama langchain-core pyyaml \\\n",
            "    opencv-python-headless pillow scikit-image \\\n",
            "    peft torchmetrics matplotlib pandas\n",
            "\n",
            "import torch\n",
            "print(f\"✅ PyTorch {torch.__version__}\")\n",
            "print(f\"✅ CUDA Available: {torch.cuda.is_available()}\")\n",
            "if torch.cuda.is_available():\n",
            "    print(f\"✅ GPU: {torch.cuda.get_device_name(0)}\")\n",
            "    print(f\"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\")"
        ]
    })
    
    # ============================================================
    # CELL 4: Clone Repository
    # ============================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📂 Step 2: Clone Repository\n",
            "Downloads the pipeline code"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "\n",
            "REPO_DIR = \"/content/indie_comic_pipeline\"\n",
            "if not os.path.exists(REPO_DIR):\n",
            "    !git clone --depth 1 https://github.com/Cyberpunk-San/Indie-Comic.git {REPO_DIR}\n",
            "else:\n",
            "    print(\"Repository already exists\")\n",
            "\n",
            "%cd {REPO_DIR}\n",
            "\n",
            "# Create output directories\n",
            "os.makedirs(\"outputs/fusion\", exist_ok=True)\n",
            "os.makedirs(\"outputs/comics\", exist_ok=True)\n",
            "os.makedirs(\"outputs/characters\", exist_ok=True)\n",
            "\n",
            "print(\"✅ Repository ready\")"
        ]
    })
    
    # ============================================================
    # CELL 5: Apply T4 Optimizations
    # ============================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## ⚡ Step 3: Apply T4 Optimizations\n",
            "Configures settings for optimal T4 performance"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import yaml\n",
            "\n",
            "with open('config/settings.yaml', 'r') as f:\n",
            "    settings = yaml.safe_load(f)\n",
            "\n",
            "# Apply T4-optimized settings\n",
            "settings['generation']['default_size'] = {\n",
            "    'width': IMG_WIDTH,\n",
            "    'height': IMG_HEIGHT\n",
            "}\n",
            "settings['generation']['inference_steps'] = INFERENCE_STEPS\n",
            "settings['generation']['guidance_scale'] = 7.5\n",
            "settings['generation']['seed'] = 42\n",
            "\n",
            "# Model settings\n",
            "settings['models']['sdxl']['device'] = 'cuda'\n",
            "settings['models']['sdxl']['memory_optimization'] = True\n",
            "settings['models']['lora']['adapter_scale'] = 0.8\n",
            "\n",
            "# T4 memory optimizations\n",
            "settings['t4_optimizations'] = {\n",
            "    'enabled': True,\n",
            "    'cpu_offload': True,\n",
            "    'attention_slicing': True,\n",
            "    'vae_slicing': True,\n",
            "    'disable_ipadapter': True,\n",
            "    'clear_cache_every_n_steps': 3\n",
            "}\n",
            "\n",
            "# Disable heavy consistency metrics\n",
            "settings['consistency'] = {\n",
            "    'enable_clip': False,\n",
            "    'enable_dinov2': False,\n",
            "    'enable_ssim': True,\n",
            "    'enable_edge': True,\n",
            "    'enable_color': True,\n",
            "    'enable_style': True,\n",
            "    'threshold': 0.55\n",
            "}\n",
            "\n",
            "with open('config/settings.yaml', 'w') as f:\n",
            "    yaml.dump(settings, f)\n",
            "\n",
            "print(\"✅ T4 optimizations applied\")\n",
            "print(f\"   Resolution: {IMG_WIDTH}x{IMG_HEIGHT}\")\n",
            "print(f\"   Steps: {INFERENCE_STEPS}\")\n",
            "print(f\"   CPU Offload: Enabled\")\n",
            "print(f\"   Attention Slicing: Enabled\")"
        ]
    })
    
    # ============================================================
    # CELL 6: Start Ollama
    # ============================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🦙 Step 4: Start Ollama\n",
            "Starts the local LLM server for story extraction"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import subprocess\n",
            "import threading\n",
            "import time\n",
            "import socket\n",
            "\n",
            "# Start Ollama server in background\n",
            "def run_ollama():\n",
            "    subprocess.run([\"ollama\", \"serve\"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n",
            "\n",
            "thread = threading.Thread(target=run_ollama, daemon=True)\n",
            "thread.start()\n",
            "time.sleep(3)\n",
            "\n",
            "# Check if Ollama is running\n",
            "def check_ollama():\n",
            "    try:\n",
            "        sock = socket.create_connection((\"localhost\", 11434), timeout=1)\n",
            "        sock.close()\n",
            "        return True\n",
            "    except:\n",
            "        return False\n",
            "\n",
            "if check_ollama():\n",
            "    print(\"✅ Ollama server running\")\n",
            "else:\n",
            "    print(\"⚠️ Ollama starting, please wait...\")\n",
            "    time.sleep(5)\n",
            "\n",
            "# Pull Llama 3.2 model\n",
            "print(\"📥 Pulling Llama 3.2 model (first time only)...\")\n",
            "!ollama pull llama3.2\n",
            "print(\"✅ Ollama ready\")"
        ]
    })
    
    # ============================================================
    # CELL 7: Character Extraction
    # ============================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🎭 Step 5: Extract Character Personality\n",
            "Uses LLM to analyze character psychology"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import subprocess\n",
            "import sys\n",
            "\n",
            "print(f\"Analyzing character: {CHARACTER_NAME}\")\n",
            "result = subprocess.run(\n",
            "    [sys.executable, \"langchain_code/character_extractor.py\", CHARACTER_NAME],\n",
            "    capture_output=True,\n",
            "    text=True\n",
            ")\n",
            "print(result.stdout)\n",
            "if result.returncode != 0:\n",
            "    print(f\"Error: {result.stderr}\")\n",
            "else:\n",
            "    print(\"✅ Character extraction complete\")"
        ]
    })
    
    # ============================================================
    # CELL 8: Story Extraction
    # ============================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🌍 Step 6: Extract Story Setting\n",
            "Analyzes the story world and environment"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(f\"Analyzing world: {STORY_WORLD}\")\n",
            "result = subprocess.run(\n",
            "    [sys.executable, \"langchain_code/story_extractor.py\", STORY_WORLD],\n",
            "    capture_output=True,\n",
            "    text=True\n",
            ")\n",
            "print(result.stdout)\n",
            "if result.returncode != 0:\n",
            "    print(f\"Error: {result.stderr}\")\n",
            "else:\n",
            "    print(\"✅ Story extraction complete\")"
        ]
    })
    
    # ============================================================
    # CELL 9: GPU Memory Monitor Function
    # ============================================================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def show_gpu_memory():\n",
            "    \"\"\"Display current GPU memory usage\"\"\"\n",
            "    if torch.cuda.is_available():\n",
            "        allocated = torch.cuda.memory_allocated() / 1024**3\n",
            "        reserved = torch.cuda.memory_reserved() / 1024**3\n",
            "        free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - reserved\n",
            "        print(f\"💾 GPU Memory: {allocated:.2f}GB used, {reserved:.2f}GB reserved, {free:.2f}GB free\")\n",
            "        if reserved > 13:\n",
            "            print(\"   ⚠️ High memory usage - consider restarting runtime\")\n",
            "    else:\n",
            "        print(\"💻 No GPU detected\")\n",
            "\n",
            "def clear_gpu_memory():\n",
            "    \"\"\"Force clear GPU cache\"\"\"\n",
            "    if torch.cuda.is_available():\n",
            "        torch.cuda.empty_cache()\n",
            "        torch.cuda.synchronize()\n",
            "        import gc\n",
            "        gc.collect()\n",
            "        print(\"🧹 GPU cache cleared\")\n",
            "\n",
            "show_gpu_memory()"
        ]
    })
    
    # ============================================================
    # CELL 10: Generate All Pages (Core Loop)
    # ============================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🎨 Step 7: Generate Comic Pages\n",
            "**This is the main generation loop.**\n",
            "\n",
            "⚠️ **First page takes 30-60 seconds (loading models)**\n",
            "⚠️ **Subsequent pages take 8-10 seconds each (models cached)**\n",
            "\n",
            "The model stays loaded in GPU memory between pages!"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import subprocess\n",
            "import sys\n",
            "import time\n",
            "\n",
            "print(\"=\" * 70)\n",
            "print(\"🎬 STARTING COMIC GENERATION\")\n",
            "print(\"=\" * 70)\n",
            "print(f\"Character: {CHARACTER_NAME}\")\n",
            "print(f\"World: {STORY_WORLD}\")\n",
            "print(f\"Pages: {NUM_PAGES}\")\n",
            "print(f\"Model: {'SDXL + LoRA' if USE_LORA else 'SDXL Base'}\")\n",
            "print(\"=\" * 70)\n",
            "\n",
            "total_start = time.time()\n",
            "\n",
            "for page in range(1, NUM_PAGES + 1):\n",
            "    page_start = time.time()\n",
            "    print(f\"\\n{'=' * 50}\")\n",
            "    print(f\"📖 PAGE {page} OF {NUM_PAGES}\")\n",
            "    print(f\"{'=' * 50}\")\n",
            "    \n",
            "    # Step 1: Fusion Engine (creates storyboard)\n",
            "    print(f\"[1/3] Creating storyboard for page {page}...\")\n",
            "    result = subprocess.run(\n",
            "        [sys.executable, \"langchain_code/fusion_engine.py\", \"--page\", str(page)],\n",
            "        capture_output=True,\n",
            "        text=True\n",
            "    )\n",
            "    if result.returncode != 0:\n",
            "        print(f\"Error: {result.stderr}\")\n",
            "        break\n",
            "    \n",
            "    # Step 2: Emotion Recognition\n",
            "    print(f\"[2/3] Analyzing emotions for page {page}...\")\n",
            "    result = subprocess.run(\n",
            "        [sys.executable, \"langchain_code/emotion_recognition_engine.py\", \"--page\", str(page)],\n",
            "        capture_output=True,\n",
            "        text=True\n",
            "    )\n",
            "    if result.returncode != 0:\n",
            "        print(f\"Error: {result.stderr}\")\n",
            "        break\n",
            "    \n",
            "    # Step 3: Generate Panels (model caches after first page!)\n",
            "    print(f\"[3/3] Generating panels for page {page}...\")\n",
            "    \n",
            "    if USE_LORA:\n",
            "        # First page needs to load model, subsequent pages use cache\n",
            "        if page == 1:\n",
            "            print(\"   🔄 Loading SDXL + LoRA model (first time - 30-60 sec)...\")\n",
            "            # Run character generation first to load model\n",
            "            subprocess.run([sys.executable, \"lora_code/run_lora_pipeline.py\"], capture_output=True)\n",
            "        \n",
            "        result = subprocess.run(\n",
            "            [sys.executable, \"lora_code/generate_panels.py\", \"--page\", str(page)],\n",
            "            capture_output=True,\n",
            "            text=True\n",
            "        )\n",
            "    else:\n",
            "        if page == 1:\n",
            "            print(\"   🔄 Loading SDXL Base model (first time - 30-60 sec)...\")\n",
            "            subprocess.run([sys.executable, \"sdxl_code/run_sdxl_pipeline.py\"], capture_output=True)\n",
            "        \n",
            "        result = subprocess.run(\n",
            "            [sys.executable, \"sdxl_code/generate_panels.py\", \"--page\", str(page)],\n",
            "            capture_output=True,\n",
            "            text=True\n",
            "        )\n",
            "    \n",
            "    if result.returncode != 0:\n",
            "        print(f\"Error: {result.stderr}\")\n",
            "        break\n",
            "    \n",
            "    page_time = time.time() - page_start\n",
            "    print(f\"✅ Page {page} completed in {page_time:.1f} seconds\")\n",
            "    \n",
            "    # Clear GPU memory every 3 pages\n",
            "    if page % 3 == 0:\n",
            "        clear_gpu_memory()\n",
            "        show_gpu_memory()\n",
            "\n",
            "total_time = time.time() - total_start\n",
            "print(f\"\\n{'=' * 70}\")\n",
            "print(f\"✅ ALL {NUM_PAGES} PAGES COMPLETED!\")\n",
            "print(f\"⏱️ Total time: {total_time:.1f} seconds\")\n",
            "print(f\"{'=' * 70}\")"
        ]
    })
    
    # ============================================================
    # CELL 11: Display Generated Pages
    # ============================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🖼️ Step 8: View Generated Comics\n",
            "Displays the compiled page layouts"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from IPython.display import Image, display\n",
            "import glob\n",
            "\n",
            "style = \"sdxl_lora_grid\" if USE_LORA else \"sdxl_base_grid\"\n",
            "\n",
            "for page in range(1, NUM_PAGES + 1):\n",
            "    grid_path = f\"outputs/comics/page_{page}_layout_{style}.png\"\n",
            "    if os.path.exists(grid_path):\n",
            "        print(f\"\\n📄 Page {page}:\")\n",
            "        display(Image(grid_path))\n",
            "    else:\n",
            "        # Try alternative naming\n",
            "        alt_paths = glob.glob(f\"outputs/comics/page_{page}_layout_*_grid.png\")\n",
            "        if alt_paths:\n",
            "            print(f\"\\n📄 Page {page}:\")\n",
            "            display(Image(alt_paths[0]))"
        ]
    })
    
    # ============================================================
    # CELL 12: Compile PDF
    # ============================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📄 Step 9: Compile PDF Book\n",
            "Creates a single PDF from all generated pages"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "style = \"sdxl_lora_grid\" if USE_LORA else \"sdxl_base_grid\"\n",
            "result = subprocess.run(\n",
            "    [sys.executable, \"compile_comic_pdf.py\", \"--style\", style],\n",
            "    capture_output=True,\n",
            "    text=True\n",
            ")\n",
            "print(result.stdout)\n",
            "if result.returncode == 0:\n",
            "    pdf_path = f\"outputs/comics/comic_book_{style}.pdf\"\n",
            "    if os.path.exists(pdf_path):\n",
            "        size_mb = os.path.getsize(pdf_path) / 1024**2\n",
            "        print(f\"✅ PDF created: {pdf_path} ({size_mb:.1f} MB)\")\n",
            "else:\n",
            "    print(f\"Error: {result.stderr}\")"
        ]
    })
    
    # ============================================================
    # CELL 13: Download Everything
    # ============================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📥 Step 10: Download All Outputs\n",
            "Packages everything into a ZIP file for download"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import zipfile\n",
            "from google.colab import files\n",
            "\n",
            "ZIP_NAME = f\"comic_{CHARACTER_NAME.replace(' ', '_')}_{STORY_WORLD.replace(' ', '_')}.zip\"\n",
            "ZIP_PATH = f\"/content/{ZIP_NAME}\"\n",
            "\n",
            "print(f\"📦 Creating {ZIP_NAME}...\")\n",
            "\n",
            "with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:\n",
            "    for root, dirs, files in os.walk(\"outputs\"):\n",
            "        for file in files:\n",
            "            if file.endswith(('.png', '.pdf', '.json')):\n",
            "                file_path = os.path.join(root, file)\n",
            "                arcname = os.path.relpath(file_path, os.path.dirname(\"outputs\"))\n",
            "                zf.write(file_path, arcname)\n",
            "\n",
            "size_mb = os.path.getsize(ZIP_PATH) / 1024**2\n",
            "print(f\"✅ ZIP created: {ZIP_NAME} ({size_mb:.1f} MB)\")\n",
            "print(\"⬇️ Downloading...\")\n",
            "files.download(ZIP_PATH)\n",
            "print(\"✅ Complete!\")"
        ]
    })
    
    # ============================================================
    # CELL 14: Final Summary
    # ============================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## ✨ Pipeline Complete!\n",
            "\n",
            "### Summary:\n",
            "- **Character**: {CHARACTER_NAME}\n",
            "- **World**: {STORY_WORLD}\n",
            "- **Pages Generated**: {NUM_PAGES}\n",
            "- **Model**: SDXL + LoRA (Manga/Lineart style)\n",
            "- **Resolution**: {IMG_WIDTH}x{IMG_HEIGHT}\n",
            "\n",
            "### Output Files:\n",
            "- Individual panels in `outputs/comics/`\n",
            "- Character reference in `outputs/characters/`\n",
            "- Storyboard JSON in `outputs/fusion/`\n",
            "- Compiled PDF in `outputs/comics/`\n",
            "\n",
            "### Next Time:\n",
            "1. Re-run this notebook\n",
            "2. Change CHARACTER_NAME and STORY_WORLD\n",
            "3. Models will reload (takes 30-60 seconds first page)\n",
            "\n",
            "**Enjoy your comic!** 🎉"
        ]
    })
    
    # Build the notebook
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {
                "provenance": [],
                "gpuType": "T4",
                "name": "Indie Comic Generator - T4 Optimized"
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

def save_notebook():
    """Save the generated notebook to disk"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the notebook
    notebook = create_t4_optimized_notebook()
    
    # Save to file
    output_path = os.path.join(script_dir, "indie_comic_t4_optimized.ipynb")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✅ Notebook created: {output_path}")
    print(f"\n📋 To use:")
    print(f"   1. Upload '{output_path}' to Google Colab")
    print(f"   2. Set Runtime > Change runtime type > T4 GPU")
    print(f"   3. Run all cells (Runtime > Run all)")
    print(f"\n⚡ First page will load model (30-60 sec)")
    print(f"⚡ Subsequent pages use cached model (8-10 sec each)")

if __name__ == "__main__":
    save_notebook()