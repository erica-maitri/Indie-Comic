import json
import os

notebook_paths = [
    "../Pipeline_Interactive_Runner.ipynb",
    "Indie_Comic_Pipeline.ipynb",
    "Scratchpad_Pipeline.ipynb"
]

setup_cell_source = [
    "# ============================================================\n",
    "# Universal Cloud/Local Setup — clones repo and runs colab_setup\n",
    "# ============================================================\n",
    "import os, sys, urllib.request\n",
    "\n",
    "# Prevent crash when parent process working directory has been deleted/invalidated\n",
    "try:\n",
    "    os.getcwd()\n",
    "except FileNotFoundError:\n",
    "    print(\"⚠️ Current working directory is invalid or deleted. Resetting to a safe location...\")\n",
    "    _safe_dir = \"/kaggle/working\" if os.path.exists(\"/kaggle/working\") else \"/\"\n",
    "    os.chdir(_safe_dir)\n",
    "\n",
    "_IN_KAGGLE = os.path.exists(\"/kaggle/working\")\n",
    "_IN_CLOUD = _IN_KAGGLE\n",
    "\n",
    "if _IN_CLOUD:\n",
    "    print(\"🚀 Detected Kaggle Environment. Setting up...\")\n",
    "    _repo = \"/kaggle/working/Indie-Comic\"\n",
    "    if not os.path.exists(_repo):\n",
    "        import subprocess\n",
    "        print(f\"📦 Cloning repository into {_repo}...\")\n",
    "        subprocess.run([\"git\", \"clone\", \"--depth\", \"1\",\n",
    "            \"https://github.com/Cyberpunk-San/Indie-Comic.git\", _repo], check=True)\n",
    "    else:\n",
    "        print(\"🔄 Repo already exists. Pulling latest changes...\")\n",
    "        import subprocess\n",
    "        subprocess.run([\"git\", \"-C\", _repo, \"pull\"], check=True)\n",
    "    \n",
    "    # Run the setup script in the main kernel context\n",
    "    setup_file = f\"{_repo}/indie_comic_pipeline/colab_setup.py\"\n",
    "    exec(open(setup_file).read(), globals())\n",
    "else:\n",
    "    print(\"💻 Detected Local Jupyter. Setting up path...\")\n",
    "    _candidates = [\n",
    "        os.path.join(os.getcwd(), \"colab_setup.py\"),\n",
    "        os.path.join(os.getcwd(), \"indie_comic_pipeline\", \"colab_setup.py\"),\n",
    "    ]\n",
    "    _found = next((p for p in _candidates if os.path.exists(p)), None)\n",
    "    if _found:\n",
    "        exec(open(_found).read(), globals())\n",
    "    else:\n",
    "        print(\"⚠️ colab_setup.py not found — running from current directory.\")\n"
]

for nb_path in notebook_paths:
    if not os.path.exists(nb_path):
        continue
    print(f"Cleaning {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    new_cells = []
    has_setup = False
    
    for cell in nb.get("cells", []):
        source_lines = cell.get("source", [])
        source_text = "".join(source_lines)
        
        # Identify setup cells (either by id, or by content)
        is_setup = (
            cell.get("id") == "colab_setup_cell" or
            "Universal Colab/Local Setup" in source_text or
            "Universal Cloud/Local Setup" in source_text or
            "Universal Kaggle/Local Setup" in source_text or
            ("colab_setup.py" in source_text and "urllib.request" in source_text)
        )
        
        if is_setup:
            if not has_setup:
                # Keep the first setup cell, and update it with correct code and metadata id
                cell["id"] = "colab_setup_cell"
                cell["source"] = setup_cell_source
                new_cells.append(cell)
                has_setup = True
            else:
                # Skip/remove any duplicate setup cells
                print("  Removed duplicate setup cell.")
        else:
            new_cells.append(cell)
            
    nb["cells"] = new_cells
    
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully cleaned {nb_path}")
