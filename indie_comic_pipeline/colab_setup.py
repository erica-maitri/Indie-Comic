"""
colab_setup.py
==============
Drop this ONE cell at the top of every notebook.
Works on both Google Colab AND local Jupyter/VS Code without any changes.

Usage (paste as first code cell in every .ipynb):
    import subprocess, sys
    subprocess.run([sys.executable, "-c",
        "import urllib.request; exec(urllib.request.urlopen('https://raw.githubusercontent.com/Cyberpunk-San/Indie-Comic/main/indie_comic_pipeline/colab_setup.py').read())"],
        check=False)

OR simply:
    exec(open('colab_setup.py').read())  # when running locally
"""

import os
import sys
import subprocess

# ── 1. Detect environment ────────────────────────────────────────────────────

try:
    from google.colab import files  # type: ignore
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

REPO_URL  = "https://github.com/Cyberpunk-San/Indie-Comic.git"
REPO_NAME = "Indie-Comic"                      # folder git creates
PIPELINE  = "indie_comic_pipeline"             # subfolder we need on sys.path

# ── 2. Clone repo (Colab only) ───────────────────────────────────────────────

if IN_COLAB:
    REPO_ROOT = f"/content/{REPO_NAME}"
    if not os.path.exists(REPO_ROOT):
        print(f"📦 Cloning {REPO_URL} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, REPO_ROOT],
            check=True
        )
    else:
        print("✅ Repo already cloned.")
else:
    # Walk up from this file's location to find the repo root
    # Works whether cwd is the repo root or indie_comic_pipeline/
    _here = os.path.abspath(os.path.dirname(globals()["__file__"]) if "__file__" in globals() else os.getcwd())
    if os.path.basename(_here) == PIPELINE:
        REPO_ROOT = os.path.dirname(_here)
    elif os.path.exists(os.path.join(_here, PIPELINE)):
        REPO_ROOT = _here
    else:
        # Last resort: assume we're already inside indie_comic_pipeline
        REPO_ROOT = os.path.dirname(_here)
    print(f"✅ Local mode — repo root: {REPO_ROOT}")

# ── 3. Set working directory & sys.path ─────────────────────────────────────

PIPELINE_DIR = os.path.join(REPO_ROOT, PIPELINE)

if not os.path.exists(PIPELINE_DIR):
    raise RuntimeError(
        f"Cannot find '{PIPELINE}' inside '{REPO_ROOT}'.\n"
        f"Expected: {PIPELINE_DIR}\n"
        "Check that the repo cloned correctly."
    )

os.chdir(PIPELINE_DIR)

for _path in [PIPELINE_DIR, REPO_ROOT]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

print(f"📁 Working directory : {os.getcwd()}")
print(f"🐍 Python path includes: {PIPELINE_DIR}")

# ── 4. Install requirements (Colab only) ────────────────────────────────────

if IN_COLAB:
    # Prefer the slim colab requirements to avoid version conflicts
    req_file = os.path.join(PIPELINE_DIR, "requirements_colab.txt")
    if not os.path.exists(req_file):
        req_file = os.path.join(PIPELINE_DIR, "requirements.txt")
    if os.path.exists(req_file):
        print("📦 Installing requirements (this may take a few minutes)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", req_file, "-q"],
            check=True
        )
        print("✅ Requirements installed.")
    else:
        print("⚠️  requirements.txt not found — skipping install.")

    # ── 5. Setup Ollama (Colab only) ──────────────────────────────────────────
    print("\n🦙 Setting up Ollama in Colab context...")
    
    # Check if ollama binary is installed, install if missing
    import shutil
    if not shutil.which("ollama"):
        print("📥 Ollama not found. Installing Ollama (this will take less than a minute)...")
        try:
            install_cmd = "curl -fsSL https://ollama.com/install.sh | sh"
            subprocess.run(install_cmd, shell=True, check=True)
            print("✅ Ollama installed successfully.")
        except Exception as e:
            print(f"❌ Failed to install Ollama: {e}")
    else:
        print("✅ Ollama binary is already installed.")

    # Check if Ollama service is running, start if not
    import urllib.request
    import json
    import time
    
    ollama_running = False
    for _ in range(3):
        try:
            with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as response:
                if response.status == 200:
                    ollama_running = True
                    break
        except Exception:
            time.sleep(1)
            
    if not ollama_running:
        print("⚠️ Ollama service not running. Starting Ollama daemon in background...")
        try:
            # Launch server in background
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True
            )
            # Wait for it to start up
            for attempt in range(10):
                try:
                    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as response:
                        if response.status == 200:
                            ollama_running = True
                            print("✅ Ollama daemon started successfully.")
                            break
                except Exception:
                    time.sleep(2)
            if not ollama_running:
                print("❌ Timeout waiting for Ollama daemon to start.")
        except Exception as e:
            print(f"❌ Failed to start Ollama daemon: {e}")
    else:
        print("✅ Ollama daemon is active.")

    # Pull required model (e.g. llama3.2)
    if ollama_running:
        model_name = "llama3.2"
        try:
            from utils.config_helper import load_settings
            settings = load_settings()
            model_name = settings.get("langchain", {}).get("model", "llama3.2")
        except Exception:
            pass
            
        print(f"📥 Pulling Ollama model '{model_name}' (this may take a minute)...")
        try:
            subprocess.run(["ollama", "pull", model_name], check=True)
            print(f"✅ Model '{model_name}' is ready.")
        except Exception as e:
            print(f"❌ Failed to pull model '{model_name}': {e}")

print("\n🚀 Setup complete! You can now import from indie_comic_pipeline freely.\n")
