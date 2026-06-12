"""
RUN FULL SDXL BASE PIPELINE - T4 OPTIMIZED
Generates character and comic panels from LangChain fusion output
Optimized for T4 GPU with memory management
"""

import subprocess
import sys
import os
import torch

print("=" * 70)
print("RUNNING FULL SDXL BASE PIPELINE - T4 OPTIMIZED")
print("Executing neural canvas generations (Base SDXL)")
print("=" * 70)

# Store original directory
PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
current_dir = os.path.dirname(os.path.abspath(__file__))

def check_gpu_memory():
    """Check available GPU memory and warn if low"""
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated() / 1024**3
        free = total_vram - allocated
        
        print(f"\n💾 GPU Memory: {total_vram:.1f}GB total, {free:.1f}GB free")
        
        if free < 4.0:
            print(f"  ⚠️ Low VRAM ({free:.1f}GB). Clearing cache...")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            import gc
            gc.collect()
            free = total_vram - torch.cuda.memory_allocated() / 1024**3
            print(f"  After cleanup: {free:.1f}GB free")
        
        return free
    return 0

def run_script(script_name, args=None):
    """Run a script in the current directory"""
    if args is None:
        args = []
    script_path = os.path.join(current_dir, script_name)
    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}")
        return None
    
    return subprocess.run([sys.executable, script_path] + args, cwd=current_dir)

# Check fusion data exists
fusion_check = os.path.join(PIPELINE_ROOT, "outputs", "fusion", "sdxl_prompt.json")

if not os.path.exists(fusion_check):
    print("\n❌ Error: No fusion output found.")
    print("   Please run the LangChain pipeline first:")
    print("   cd ../langchain_code && python run_full_pipeline.py")
    sys.exit(1)

print("\n✅ Fusion data found. Starting SDXL Base generation...")

# Check GPU memory before starting
gpu_free = check_gpu_memory()
if gpu_free > 0 and gpu_free < 5:
    response = input(f"\n⚠️ Only {gpu_free:.1f}GB VRAM available. SDXL needs ~8-10GB. Continue? (y/n): ")
    if response.lower() != 'y':
        print("Exiting. Free up GPU memory and try again.")
        sys.exit(0)

print("\n" + "=" * 70)
print("STEP 1/2: Generating Character Reference (SDXL Base)")
print("=" * 70)

result = run_script("generate_character.py")
if result is None or result.returncode != 0:
    print("❌ Error: Character generation failed")
    sys.exit(1)

# Clear memory after character generation
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    print("🧹 Memory cleared after character generation")

print("\n" + "=" * 70)
print("STEP 2/2: Generating Story Components (SDXL Base)")
print("=" * 70)

result = run_script("generate_components.py")
if result is None or result.returncode != 0:
    print("❌ Error: Component generation failed")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ FULL SDXL BASE PIPELINE COMPLETE!")
print("=" * 70)
print("\n📁 Output files:")
print("   Character: ../outputs/characters/character_reference.png")
print("   Components: ../outputs/comics/component_sdxl_base_*.png")
print("   Component Sheet: ../outputs/comics/component_sheet_sdxl_base_horizontal.png")
print("=" * 70)