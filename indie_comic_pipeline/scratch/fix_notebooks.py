import os
import json
import re

notebooks = [
    "comic_strip_generation.ipynb",
    "consistency_checking.ipynb",
    "crossover_fusion.ipynb",
    "image_generation.ipynb",
    "indie_comic_pipeline.ipynb",
    "Indie_Comic_Runner.ipynb",
    "indie_comic_t4_optimized.ipynb",
    "ip_adapter.ipynb",
    "metrics_evaluation.ipynb",
    "pdf_generation.ipynb"
]

STANDARD_SETUP_SRC = """#@title Choose Setup Method { run: "auto" }
SETUP_MODE = "git" #@param ["git", "zip"]
UPLOAD_PREVIOUS_OUTPUTS = True #@param {type:"boolean"}

import os, subprocess, zipfile, sys

# Detect if running in Google Colab
try:
    from google.colab import files
    IN_COLAB = True
    print("Running in Google Colab environment.")
except ImportError:
    IN_COLAB = False
    print("Running in local environment.")

if IN_COLAB:
    # Setup the code repository in Colab
    if SETUP_MODE == "git":
        REPO_URL   = "https://github.com/Cyberpunk-San/Indie-Comic.git"
        REPO_DIR   = "/content/indie_comic_pipeline"
        SUBDIR     = "indie_comic_pipeline"

        if not os.path.exists(REPO_DIR):
            print(f" Cloning repo from {REPO_URL}...")
            subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
        else:
            print(" Repository already present — pulling latest changes...")
            subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)

        PIPELINE_ROOT = os.path.join(REPO_DIR, SUBDIR)
        os.chdir(PIPELINE_ROOT)
        print(f" Working directory set to: {os.getcwd()}")
    else:
        print(" Upload your indie_comic_pipeline.zip file:")
        uploaded = files.upload()
        for filename in uploaded.keys():
            if filename.endswith('.zip'):
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    zip_ref.extractall('/content/')
                break
        os.chdir('/content/indie_comic_pipeline')
else:
    # Local setup: find indie_comic_pipeline directory
    cwd = os.getcwd()
    if os.path.basename(cwd) == "indie_comic_pipeline":
        PIPELINE_ROOT = cwd
    elif os.path.exists(os.path.join(cwd, "indie_comic_pipeline")):
        PIPELINE_ROOT = os.path.join(cwd, "indie_comic_pipeline")
        os.chdir(PIPELINE_ROOT)
    else:
        PIPELINE_ROOT = cwd
    print(f" Working directory set to: {os.getcwd()}")

# Create directories
for d in ["outputs/fusion", "outputs/comics", "outputs/characters"]:
    os.makedirs(d, exist_ok=True)
print(" Directory structure ready.")"""

STANDARD_DOWNLOAD_SRC = """import zipfile

# Detect if running in Google Colab
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

ZIP_PATH = "indie_comic_outputs.zip" if not IN_COLAB else "/content/indie_comic_outputs.zip"
print(" Packaging outputs...")

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, fnames in os.walk("outputs"):
        for fname in fnames:
            fpath = os.path.join(root, fname)
            arcname = os.path.relpath(fpath, os.path.dirname("outputs"))
            zf.write(fpath, arcname)

size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
print(f" ZIP created: {size_mb:.1f} MB")

if IN_COLAB:
    files.download(ZIP_PATH)
else:
    print(f" Saved zip archive locally at: {os.path.abspath(ZIP_PATH)}")
    print(" Outputs are in the 'outputs/' directory.")"""

T4_SETUP_SRC = """import os, subprocess

# Detect if running in Google Colab
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    REPO_DIR = "/content/indie_comic_pipeline"
    if not os.path.exists(REPO_DIR):
        print("Cloning repository in Colab...")
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/Cyberpunk-San/Indie-Comic.git", REPO_DIR], check=True)
    else:
         print("Repository already exists")
    os.chdir(REPO_DIR)
else:
    # Local setup: find indie_comic_pipeline directory
    cwd = os.getcwd()
    if os.path.basename(cwd) == "indie_comic_pipeline":
        os.chdir(cwd)
    elif os.path.exists(os.path.join(cwd, "indie_comic_pipeline")):
        os.chdir(os.path.join(cwd, "indie_comic_pipeline"))
    print(f"Working directory: {os.getcwd()}")

# Create output directories
os.makedirs("outputs/fusion", exist_ok=True)
os.makedirs("outputs/comics", exist_ok=True)
os.makedirs("outputs/characters", exist_ok=True)

print("✅ Repository ready")"""

T4_DOWNLOAD_SRC = """import zipfile
import os

# Detect if running in Google Colab
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

CHARACTER_NAME = globals().get('CHARACTER_NAME', 'Wanderer')
STORY_WORLD = globals().get('STORY_WORLD', 'The_Abstract')

ZIP_NAME = f"comic_{CHARACTER_NAME.replace(' ', '_')}_{STORY_WORLD.replace(' ', '_')}.zip"
ZIP_PATH = f"/content/{ZIP_NAME}" if IN_COLAB else ZIP_NAME

print(f"📦 Creating {ZIP_NAME}...")

with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files_list in os.walk("outputs"):
        for file in files_list:
            if file.endswith(('.png', '.pdf', '.json')):
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname("outputs"))
                zf.write(file_path, arcname)

size_mb = os.path.getsize(ZIP_PATH) / 1024**2
print(f"✅ ZIP created: {ZIP_NAME} ({size_mb:.1f} MB)")

if IN_COLAB:
    print("⬇️ Downloading...")
    files.download(ZIP_PATH)
    print("✅ Complete!")
else:
    print(f" Saved zip archive locally at: {os.path.abspath(ZIP_PATH)}")
    print(" Outputs are in the 'outputs/' directory.")"""

def fix_notebook(file_path):
    print(f"Optimizing: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    modified = False

    # Handle standard 30-cell and 40-cell notebooks
    if file_path in ["comic_strip_generation.ipynb", "consistency_checking.ipynb", 
                     "crossover_fusion.ipynb", "image_generation.ipynb", 
                     "indie_comic_pipeline.ipynb", "ip_adapter.ipynb", 
                     "metrics_evaluation.ipynb", "pdf_generation.ipynb"]:
        for idx, cell in enumerate(data.get("cells", [])):
            if cell.get("cell_type") == "code":
                src = "".join(cell.get("source", []))
                # Identify Setup Repository Cell
                if "SETUP_MODE" in src and "REPO_URL" in src and "google.colab" in src:
                    cell["source"] = [line + "\n" for line in STANDARD_SETUP_SRC.split("\n")]
                    cell["source"][-1] = cell["source"][-1].rstrip("\n")  # Strip last line end newline
                    modified = True
                    print(f"  -> Fixed Setup cell (cell {idx})")
                # Identify Download Outputs Cell
                elif "ZIP_PATH" in src and "zipfile" in src and "files.download" in src:
                    cell["source"] = [line + "\n" for line in STANDARD_DOWNLOAD_SRC.split("\n")]
                    cell["source"][-1] = cell["source"][-1].rstrip("\n")
                    modified = True
                    print(f"  -> Fixed Download cell (cell {idx})")

    # Handle T4 optimized notebook specifically
    elif file_path == "indie_comic_t4_optimized.ipynb":
        for idx, cell in enumerate(data.get("cells", [])):
            if cell.get("cell_type") == "code":
                src = "".join(cell.get("source", []))
                if "REPO_DIR = \"/content/indie_comic_pipeline\"" in src and "git clone" in src:
                    cell["source"] = [line + "\n" for line in T4_SETUP_SRC.split("\n")]
                    cell["source"][-1] = cell["source"][-1].rstrip("\n")
                    modified = True
                    print(f"  -> Fixed T4 Setup cell (cell {idx})")
                elif "ZIP_NAME = f\"comic_" in src and "files.download" in src:
                    cell["source"] = [line + "\n" for line in T4_DOWNLOAD_SRC.split("\n")]
                    cell["source"][-1] = cell["source"][-1].rstrip("\n")
                    modified = True
                    print(f"  -> Fixed T4 Download cell (cell {idx})")

    # Handle the main runner notebook containing Google Drive paths
    elif file_path == "Indie_Comic_Runner.ipynb":
        for idx, cell in enumerate(data.get("cells", [])):
            if cell.get("cell_type") == "code":
                lines = cell.get("source", [])
                src = "".join(lines)
                
                # Check 1: Mount Drive cell (cell 1)
                if "drive.mount" in src and "google.colab" in src:
                    cell["source"] = [
                        "# Mount your Google Drive conditionally\n",
                        "try:\n",
                        "    from google.colab import drive\n",
                        "    IN_COLAB = True\n",
                        "except ImportError:\n",
                        "    IN_COLAB = False\n",
                        "\n",
                        "if IN_COLAB:\n",
                        "    drive.mount('/content/drive')\n",
                        "else:\n",
                        "    print('Local environment detected. Skipping Drive mount.')"
                    ]
                    modified = True
                    print(f"  -> Fixed Mount Drive (cell {idx})")
                
                # Check 2: Hardcoded directory changing magic commands (%cd)
                elif "%cd \"/content/drive/MyDrive/" in src or "%cd /content/drive/MyDrive/" in src:
                    cell["source"] = [
                        "import os\n",
                        "try:\n",
                        "    from google.colab import drive\n",
                        "    IN_COLAB = True\n",
                        "except ImportError:\n",
                        "    IN_COLAB = False\n",
                        "\n",
                        "if IN_COLAB:\n",
                        "    %cd \"/content/drive/MyDrive/Indie_Comic_Project/Indie-Comic-main/indie_comic_pipeline\"\n",
                        "else:\n",
                        "    # Local environment\n",
                        "    cwd = os.getcwd()\n",
                        "    if os.path.basename(cwd) == 'indie_comic_pipeline':\n",
                        "        os.chdir(cwd)\n",
                        "    elif os.path.exists(os.path.join(cwd, 'indie_comic_pipeline')):\n",
                        "        os.chdir(os.path.join(cwd, 'indie_comic_pipeline'))\n",
                        "    print(f'Working directory set to: {os.getcwd()}')"
                    ]
                    modified = True
                    print(f"  -> Fixed path change magic (cell {idx})")
                
                # Check 3: absolute directory visual listing commands like !ls
                elif "!ls \"/content/drive/" in src:
                    # Replace with cross-platform OS-independent listdir
                    cell["source"] = [
                        "import os\n",
                        "comics_path = 'outputs/comics'\n",
                        "if os.path.exists(comics_path):\n",
                        "    print('Comics directory contents:')\n",
                        "    for item in sorted(os.listdir(comics_path)):\n",
                        "        print(f'  - {item}')\n",
                        "else:\n",
                        "    print('Comics output directory does not exist yet.')"
                    ]
                    modified = True
                    print(f"  -> Fixed OS ls command (cell {idx})")
                
                # Check 4: direct hardcoded path variable assignments
                elif "\"/content/drive/MyDrive/Indie_Comic_Project/Indie-Comic-main/indie_comic_pipeline/outputs/comics/" in src:
                    # Replace with relative local path
                    new_src = src.replace("/content/drive/MyDrive/Indie_Comic_Project/Indie-Comic-main/indie_comic_pipeline/", "")
                    cell["source"] = [line + "\n" for line in new_src.split("\n")]
                    cell["source"][-1] = cell["source"][-1].rstrip("\n")
                    modified = True
                    print(f"  -> Fixed absolute outputs path string (cell {idx})")

    if modified:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"  [SUCCESS] Rewrote modified notebook: {file_path}")
    else:
        print(f"  [INFO] No changes required for: {file_path}")

if __name__ == "__main__":
    import sys
    # Find all notebooks in current directory
    for nb in notebooks:
        if os.path.exists(nb):
            fix_notebook(nb)
        else:
            print(f"Notebook not found: {nb}")
    print("\nFix script execution finished successfully!")
