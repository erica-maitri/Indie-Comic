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

# Prevent crash when parent process working directory has been deleted.
try:
    os.getcwd()
except FileNotFoundError:
    print("⚠️ Current working directory was invalid/deleted. Resetting to a safe location...")
    _safe_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "/"
    os.chdir(_safe_dir)

# ── 1. Detect environment ────────────────────────────────────────────────────

IN_KAGGLE = os.path.exists("/kaggle/working")
IN_CLOUD = IN_KAGGLE

REPO_URL  = "https://github.com/Cyberpunk-San/Indie-Comic.git"
REPO_NAME = "Indie-Comic"                      # folder git creates
PIPELINE  = "indie_comic_pipeline"             # subfolder we need on sys.path

# ── 2. Clone repo (Kaggle only) ───────────────────────────────────────────────

if IN_CLOUD:
    REPO_ROOT = f"/kaggle/working/{REPO_NAME}"
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

# ── 4. Install requirements (Cloud only) ────────────────────────────────────

if IN_CLOUD:
    # Cloud environments (Colab/Kaggle) pre-install packages that may conflict.
    # Uninstalling incompatible packages solves version conflict crashes completely.
    if IN_CLOUD:
        print("🧹 Removing incompatible pre-installed torchao version to avoid conflicts...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "torchao", "-y", "-q"], check=False)
        print("🧹 Removing pre-installed flax to avoid diffusers circular import bugs...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "flax", "-y", "-q"], check=False)

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

    # ── 5. Setup Ollama (Cloud only) ──────────────────────────────────────────
    print("\n🦙 Setting up Ollama in Cloud context...")
    
    import shutil
    import time
    import threading
    import socket
    
    ollama_installed = True
    if not shutil.which("ollama"):
        print("📥 Ollama not found. Installing dependencies and Ollama...")
        try:
            # Install zstd dependency first
            print(" Installing zstd dependency...")
            subprocess.run(["apt-get", "update"], check=True, capture_output=True, text=True)
            subprocess.run(["apt-get", "install", "-y", "zstd"], check=True, capture_output=True, text=True)
            print(" zstd installed successfully.")

            install_process = subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True, capture_output=True, text=True)
            print(" Ollama successfully installed.")
            subprocess.run(["ollama", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR: Ollama installation failed with exit code {e.returncode}.")
            print(f" STDERR: {e.stderr}")
            ollama_installed = False
        except FileNotFoundError:
            print("❌ ERROR: 'ollama' executable not found even after successful script execution.")
            ollama_installed = False

    def start_ollama_server():
        global ollama_installed
        try:
            flags = 0x08000000 if sys.platform == 'win32' else 0
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
        except FileNotFoundError:
            ollama_installed = False

    thread = threading.Thread(target=start_ollama_server, daemon=True)
    thread.start()
    time.sleep(1.5)

    if not ollama_installed:
        print("❌ ERROR: 'ollama' executable not found on this system.")
    else:
        print(" Waiting for Ollama server to respond...")
        connected = False
        for _ in range(30):
            try:
                s = socket.create_connection(("localhost", 11434), timeout=1)
                s.close()
                connected = True
                break
            except OSError:
                time.sleep(1.5)

        if connected:
            print("✅ Ollama server is running on port 11434.")
            
            # Pull required model (e.g. llama3.2)
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
        else:
            print("❌ Ollama server failed to start within 45 seconds.")

# ── 6. Load .env file & Suppress Tokenizer Warnings ──────────────────────────
# Load .env file from the repo root if it exists
dotenv_path = os.path.join(REPO_ROOT, ".env")
if os.path.exists(dotenv_path):
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=dotenv_path)
        if "HF_TOKEN" in os.environ:
            print("🔑 Hugging Face Token loaded from .env into environment.")
    except Exception as e:
        print(f"⚠️ Failed to load .env: {e}")

# Suppress Hugging Face/Transformers tokenization warnings
try:
    import logging as py_logging
    py_logging.getLogger("transformers.tokenization_utils_base").setLevel(py_logging.ERROR)
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    # Set Hugging Face verbosity
    from transformers.utils import logging as tf_logging
    tf_logging.set_verbosity_error()
except Exception:
    pass

print("\n🚀 Setup complete! You can now import from indie_comic_pipeline freely.\n")
