"""
MASTER PIPELINE SCRIPT
Runs everything from character extraction to comic generation
Optimized for T4 GPU with proper directory handling and page range selection
"""

import subprocess
import sys
import os
import time
import json
import torch

if sys.stdout.encoding != 'utf-8':
    try:
        reconfigure = getattr(sys.stdout, 'reconfigure', None)
        if reconfigure:
            reconfigure(encoding='utf-8')
    except:
        pass

if sys.stderr.encoding != 'utf-8':
    try:
        reconfigure = getattr(sys.stderr, 'reconfigure', None)
        if reconfigure:
            reconfigure(encoding='utf-8')
    except:
        pass

print("=" * 70)
print("INDIE COMIC GENERATOR - MASTER PIPELINE - Orchestrating multi-modal generation")
print("=" * 70)

start_time = time.time()

# Store original working directory at startup
ORIGINAL_CWD = os.getcwd()
PIPELINE_ROOT = os.path.dirname(os.path.abspath(__file__))

def run_script(script_path, args=None, capture_output=False):
    """
    Helper to run scripts without changing global directory
    Uses absolute paths for reliability
    """
    if args is None:
        args = []
    
    # Handle relative paths
    if not os.path.isabs(script_path):
        full_path = os.path.join(PIPELINE_ROOT, script_path)
    else:
        full_path = script_path
    
    if not os.path.exists(full_path):
        print(f"Error: Script not found at {full_path}")
        return None
    
    # Run from pipeline root to maintain consistent paths
    if capture_output:
        result = subprocess.run(
            [sys.executable, full_path] + args,
            cwd=PIPELINE_ROOT,
            capture_output=True,
            text=True
        )
    else:
        result = subprocess.run(
            [sys.executable, full_path] + args,
            cwd=PIPELINE_ROOT
        )
    
    return result

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

print("\nChecking local LLM model (needed for enrichment + legacy modes)...")
result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
if "llama3.2" not in result.stdout:
    print("Warning: Llama 3.2 model not found locally. Triggering Ollama download...")
    subprocess.run(["ollama", "pull", "llama3.2"])
print("Llama 3.2 is available.")

# ============================================================================
# MODE SELECTION: Story-Weaver Direct vs LangChain Extraction
# ============================================================================

print("\n" + "=" * 70)
print("INPUT SOURCE SELECTION")
print("=" * 70)
print("Choose how to provide the story:")
print("  0. Story-Weaver Direct Mode (no character reference image needed)")
print("     → Reads story_dynamic.json, enriches panels with full cast via LLM")
print("  1. LangChain Extraction Mode (original flow)")
print("     → Runs character + setting extractors, fusion engine, ERC")
input_mode = input("\nEnter input mode [0 or 1, default is 1]: ").strip()
if not input_mode:
    input_mode = "1"

USING_STORY_WEAVER_MODE = (input_mode == "0")

if USING_STORY_WEAVER_MODE:
    print("\n" + "=" * 70)
    print("STORY-WEAVER DIRECT MODE")
    print("=" * 70)

    sw_input = input("Path to story_dynamic.json [default: ../Story-Weaver/story_dynamic.json]: ").strip()
    if not sw_input:
        sw_input = "../Story-Weaver/story_dynamic.json"

    sw_character = input("Main character name [default: Wanderer]: ").strip()
    if not sw_character:
        sw_character = "Wanderer"

    sw_world = input("Story world/setting name [default: The Abstract]: ").strip()
    if not sw_world:
        sw_world = "The Abstract"

    sw_min_chars = input("Minimum side characters per panel [default: 3]: ").strip()
    if not sw_min_chars or not sw_min_chars.isdigit():
        sw_min_chars = "3"

    print("\n" + "=" * 70)
    print("STEP 1: Running Story-Weaver Enricher")
    print("=" * 70)
    
    # FIX: Use run_script helper instead of changing directories
    result = run_script("utils/bridge_weaver.py", [
        "--enrich",
        "--input", sw_input,
        "--character", sw_character,
        "--world", sw_world,
        "--min-side-chars", sw_min_chars
    ])
    
    if result is None or result.returncode != 0:
        print("Error: Story-Weaver enrichment failed.")
        sys.exit(1)
    print("\n✅ Enrichment complete! enriched_storyboard.json is ready.")

if USING_STORY_WEAVER_MODE:
    print("\n[Story-Weaver Mode] Skipping LangChain extraction — story already enriched.")
else:
    print("\n" + "=" * 70)
    print("STEP 1: Running Initial Parameters Extraction")
    print("=" * 70)
    
    print("Step 1A: Running Character Personality Extractor...")
    result = run_script("langchain_code/character_extractor.py")
    if result is None or result.returncode != 0:
        print("Error: Character extraction failed.")
        sys.exit(1)
    
    print("\nStep 1B: Running Story Setting Extractor...")
    result = run_script("langchain_code/story_extractor.py")
    if result is None or result.returncode != 0:
        print("Error: Story extraction failed.")
        sys.exit(1)

print("\n" + "=" * 70)
print("STEP 2: Verifying GPU/CUDA environments")
print("=" * 70)

if not torch.cuda.is_available():
    print("Warning: CUDA acceleration is not active. Image generation will be extremely slow.")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        print("Exiting...")
        sys.exit(0)
else:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✅ GPU detected: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    
    # Warn if T4 with low memory
    if "T4" in gpu_name and gpu_mem < 16:
        print("⚠️ T4 GPU with limited VRAM detected. Using optimized settings.")
        print("   Resolution reduced to 768x768, inference steps reduced to 25.")

print("\n" + "=" * 70)
print("STEP 3: Image Generation Pipeline Configuration")
print("=" * 70)

print("Choose the image generation pipeline model to use:")
print("  1. SDXL Base Pipeline (Recommended for quality)")
print("  2. Stable Diffusion v1.5 Pipeline (Faster, lower quality)")
print("  3. SDXL + LoRA Pipeline (Best for manga/lineart style)")
choice = input("Enter choice [1, 2, or 3, default is 3]: ").strip()
if not choice:
    choice = "3"

# FIX: Add page range selection for legacy mode
if not USING_STORY_WEAVER_MODE:
    print("\nGenerate character sheet reference and component assets first? (y/n, default is y): ", end="")
    gen_assets = input().strip().lower()
    if gen_assets != 'n':
        print("\nExecuting Component Assets Generation...")
        if choice == '2':
            result = run_script("sd15_code/run_sd15_pipeline.py")
        elif choice == '3':
            result = run_script("lora_code/run_lora_pipeline.py")
        else:
            result = run_script("sdxl_code/run_sdxl_pipeline.py")
        
        if result is None or result.returncode != 0:
            print("Warning: Asset generation had issues, but continuing...")
else:
    print("\n[Story-Weaver Mode] Skipping character sheet generation — not required.")

print("\n" + "=" * 70)
print("STEP 4: Page-by-Page Panel Generation Loop")
print("=" * 70)

# FIX: Determine pages to generate with user input
if USING_STORY_WEAVER_MODE:
    # In Story-Weaver mode, pages come from enriched_storyboard.json
    enriched_path = os.path.join(PIPELINE_ROOT, "outputs", "fusion", "enriched_storyboard.json")
    if not os.path.exists(enriched_path):
        print("Error: enriched_storyboard.json not found. Enrichment may have failed.")
        print(f"Expected at: {enriched_path}")
        sys.exit(1)
    
    with open(enriched_path, "r", encoding="utf-8") as f:
        enriched_data = json.load(f)
    total_pages = enriched_data.get("total_pages", 1)
    print(f"[Story-Weaver Mode] Found {total_pages} page(s) in enriched storyboard.")
    
    # Ask user how many pages to generate
    max_pages = total_pages
    page_input = input(f"Generate how many pages? [1-{max_pages}, or press Enter for all {max_pages}]: ").strip()
    if page_input.isdigit():
        num_pages = int(page_input)
        if 1 <= num_pages <= max_pages:
            page_range = range(1, num_pages + 1)
        else:
            print(f"Invalid number. Generating all {max_pages} pages.")
            page_range = range(1, max_pages + 1)
    else:
        page_range = range(1, max_pages + 1)
else:
    total_pages = 10
    page_input = input(f"Generate how many pages? [Enter number 1-10, or press Enter for all {total_pages}]: ").strip()
    if page_input.isdigit():
        num_pages = int(page_input)
        if 1 <= num_pages <= total_pages:
            page_range = range(1, num_pages + 1)
        else:
            print(f"Invalid number. Generating all {total_pages} pages.")
            page_range = range(1, total_pages + 1)
    else:
        page_range = range(1, total_pages + 1)

print(f"\n📚 Will generate {len(list(page_range))} page(s): Pages {list(page_range)}")

for page_num in page_range:
    print(f"\n{'=' * 50}")
    print(f"PROCESSING PAGE {page_num}")
    print(f"{'=' * 50}")

    if not USING_STORY_WEAVER_MODE:
        # Legacy mode: run fusion + ERC engines per page
        print(f"Running storyboard fusion for Page {page_num}...")
        result = run_script("langchain_code/fusion_engine.py", ["--page", str(page_num)])
        if result is None or result.returncode != 0:
            print(f"Error: Storyboard fusion failed for Page {page_num}.")
            sys.exit(1)

        print(f"Running emotion recognition for Page {page_num}...")
        result = run_script("langchain_code/emotion_recognition_engine.py", ["--page", str(page_num)])
        if result is None or result.returncode != 0:
            print(f"Error: Emotion recognition failed for Page {page_num}.")
            sys.exit(1)
    else:
        print(f"[Story-Weaver Mode] Enrichment already complete. Generating panels for Page {page_num}...")

    # Generate Panel Images for this page
    print(f"\n🎨 Drawing panels and compiling layout for Page {page_num}...")
    
    # FIX: Use absolute paths and run subprocesses without changing global directory
    if choice == '2':
        script_path = os.path.join(PIPELINE_ROOT, "sd15_code", "generate_panels.py")
    elif choice == '3':
        script_path = os.path.join(PIPELINE_ROOT, "lora_code", "generate_panels.py")
    else:
        script_path = os.path.join(PIPELINE_ROOT, "sdxl_code", "generate_panels.py")
    
    result = subprocess.run(
        [sys.executable, script_path, "--page", str(page_num)],
        cwd=PIPELINE_ROOT
    )

    if result.returncode != 0:
        print(f"Error: Image generation failed for Page {page_num}.")
        sys.exit(1)

    print(f"\n✅ Page {page_num} completed successfully!")

    # Clear GPU memory between pages
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    import gc
    gc.collect()

    if page_num != page_range[-1]:
        val = input(f"\n[Press Enter to proceed to Page {page_num+1}, or type 'exit' to quit]: ").strip().lower()
        if val == 'exit':
            print("Exiting loop...")
            break

# Compile final PDF if pages were generated
print("\n" + "=" * 70)
print("STEP 5: Compiling Comic Pages into PDF")
print("=" * 70)

# Determine which grid style was used
if choice == '2':
    style = "sd15_lora_grid"
elif choice == '3':
    style = "sdxl_lora_grid"
else:
    style = "sdxl_base_grid"

print(f"Compiling PDF using style: {style}")
result = run_script("compile_comic_pdf.py", ["--style", style])

if result and result.returncode == 0:
    print("✅ PDF compilation complete!")
else:
    print("⚠️ PDF compilation had issues, but individual pages may still be available.")

print("\n" + "=" * 70)
print("MASTER PIPELINE COMPLETE")
print("=" * 70)

elapsed_time = time.time() - start_time
print(f"\nTotal elapsed time: {elapsed_time:.2f} seconds")
print("\n📁 Output files are in the 'outputs' directory:")
print("   - outputs/fusion/         (JSON storyboard data)")
print("   - outputs/characters/     (Character reference images)")
print("   - outputs/comics/         (Generated panels and PDFs)")
print("\n✨ DONE")
print("=" * 70)