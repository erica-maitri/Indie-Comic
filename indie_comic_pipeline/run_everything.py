"""
MASTER PIPELINE SCRIPT
Runs everything from character extraction to comic generation
"""

import subprocess

import sys

import os

import time

if sys.stdout.encoding != 'utf-8':

    try:

        sys.stdout.reconfigure(encoding='utf-8')

    except:

        pass

if sys.stderr.encoding != 'utf-8':

    try:

        sys.stderr.reconfigure(encoding='utf-8')

    except:

        pass

print("=" * 70)

print("INDIE COMIC GENERATOR - MASTER PIPELINE - Orchestrating multi-modal generation")

print("=" * 70)

start_time = time.time()

                                                                                 

def check_ollama():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex(('localhost', 11434))
    sock.close()
    return result == 0

def ensure_ollama_running():
    import time
    import subprocess
    
    if check_ollama():
        print("Ollama is running.")
        return True
        
    print("⚠️ Ollama daemon is not running. Attempting to start Ollama automatically...")
    try:
        if sys.platform == "win32":
            CREATE_NO_WINDOW = 0x08000000
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
        else:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        for attempt in range(15):
            time.sleep(1)
            if check_ollama():
                print("✅ Ollama server started and connected successfully!")
                return True
            print(f"   Waiting for Ollama to initialize... (attempt {attempt+1}/15)")
    except Exception as e:
        print(f"❌ Failed to auto-start Ollama: {e}")
        
    print("\nError: Ollama daemon is not running.")
    print("   Please make sure Ollama is installed and run 'ollama serve' in your terminal.")
    print("   Then come back and run this script again.")
    sys.exit(1)

print("\nChecking dependencies...")
ensure_ollama_running()

                                                                         

print("\nChecking local LLM model...")

result = subprocess.run(["ollama", "list"], capture_output=True, text=True)

if "llama3.2" not in result.stdout:

    print("Warning: Llama 3.2 model not found locally. Triggering Ollama download...")

    subprocess.run(["ollama", "pull", "llama3.2"])

print("Llama 3.2 is available.")

                                                            

print("\n" + "=" * 70)
print("STEP 1: Running LangChain Pipeline")
print("=" * 70)

os.chdir("langchain_code")
result = subprocess.run([sys.executable, "run_full_pipeline.py"])
if result.returncode != 0:
    print("Error: LangChain pipeline step failed.")
    sys.exit(1)

print("\n" + "=" * 70)
print("STEP 1.5: Running Emotion Recognition (ERC) Engine")
print("=" * 70)
result = subprocess.run([sys.executable, "emotion_recognition_engine.py"])
if result.returncode != 0:
    print("Error: Emotion recognition step failed.")
    sys.exit(1)

print("\n" + "=" * 70)
print("STEP 2: Verifying GPU/CUDA environments for SDXL")
print("=" * 70)
os.chdir("..")

import torch
if not torch.cuda.is_available():
    print("Warning: CUDA acceleration is not active. SDXL running on CPU will be extremely slow.")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        print("Exiting...")
        sys.exit(0)

print("\n" + "=" * 70)
print("STEP 3: Executing Image Generation Pipeline")
print("=" * 70)

print("Choose generation mode:")
print("  1. Standard Component Assets (Generate character reference and core story components)")
print("  2. Emotion-Aware Comic Panels (Generate layout and panels for a specific storyboard page)")
mode_choice = input("Enter choice [1 or 2, default is 1]: ").strip()

page_num = 1
if mode_choice == '2':
    page_choice = input("Enter storyboard page number to generate (1-10, default is 1): ").strip()
    try:
        page_num = int(page_choice) if page_choice else 1
        if not (1 <= page_num <= 10):
            page_num = 1
    except ValueError:
        page_num = 1

print("\nChoose the image generation pipeline model to use:")
print("  1. SDXL Base Pipeline (Recommended)")
print("  2. Stable Diffusion v1.5 Pipeline")
print("  3. SDXL + LoRA Pipeline")
choice = input("Enter choice [1, 2, or 3, default is 1]: ").strip()

if mode_choice == '2':
    # Emotion-Aware Comic Panels Mode
    if choice == '2':
        os.chdir("sd15_code")
        result = subprocess.run([sys.executable, "generate_panels.py", "--page", str(page_num)])
        if result.returncode != 0:
            print("Error: SD 1.5 panel generation step failed.")
            sys.exit(1)
    elif choice == '3':
        os.chdir("lora_code")
        result = subprocess.run([sys.executable, "generate_panels.py", "--page", str(page_num)])
        if result.returncode != 0:
            print("Error: SDXL + LoRA panel generation step failed.")
            sys.exit(1)
    else:
        os.chdir("sdxl_code")
        result = subprocess.run([sys.executable, "generate_panels.py", "--page", str(page_num)])
        if result.returncode != 0:
            print("Error: SDXL panel generation step failed.")
            sys.exit(1)
else:
    # Standard Component Assets Mode
    if choice == '2':
        os.chdir("sd15_code")
        result = subprocess.run([sys.executable, "run_sd15_pipeline.py"])
        if result.returncode != 0:
            print("Error: SD 1.5 pipeline step failed.")
            sys.exit(1)
    elif choice == '3':
        os.chdir("lora_code")
        result = subprocess.run([sys.executable, "run_lora_pipeline.py"])
        if result.returncode != 0:
            print("Error: SDXL + LoRA pipeline step failed.")
            sys.exit(1)
    else:
        os.chdir("sdxl_code")
        result = subprocess.run([sys.executable, "run_sdxl_pipeline.py"])
        if result.returncode != 0:
            print("Error: SDXL pipeline step failed.")
            sys.exit(1)

print("\n" + "=" * 70)
print("MASTER PIPELINE COMPLETE")
print("=" * 70)

elapsed_time = time.time() - start_time
print(f"\nTotal elapsed time: {elapsed_time:.2f} seconds")

print("\nOutput files:")
if mode_choice == '2':
    print(f"   Comic Panels: outputs/comics/page_{page_num}_panel_*")
    print(f"   Horizontal Layout: outputs/comics/page_{page_num}_layout_*_horizontal.png")
    print(f"   Grid Layout: outputs/comics/page_{page_num}_layout_*_grid.png")
else:
    print("   Character: outputs/characters/character_reference.png")
    print("   Components: outputs/comics/component_1.png through component_N.png")
    print("   Component Sheet: outputs/comics/component_sheet_horizontal.png and component_sheet_grid_2x2.png")

print("\nDONE")
print("=" * 70)

