"""
RUN FULL SDXL + LORA PIPELINE - T4 OPTIMIZED
Generates character and comic panels from LangChain fusion output
Optimized for T4 GPU with memory management
"""

import subprocess
import sys
import os
import torch

print("=" * 70)
print("RUNNING FULL SDXL + LORA PIPELINE - T4 OPTIMIZED")
print("Executing neural canvas generations (SDXL + LoRA - Recommended)")
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
        
        # T4 has 16GB, LoRA needs about 11-12GB
        if free < 5.0:
            print(f"  ⚠️ Low VRAM ({free:.1f}GB). LoRA needs ~11GB.")
            print("  Consider:")
            print("    - Restart runtime to clear memory")
            print("    - Use Base SDXL instead (option 1)")
            print("    - Reduce resolution in settings.yaml")
            response = input("\n  Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)
        elif free < 8.0:
            print(f"  ⚠️ Limited VRAM ({free:.1f}GB). May still work but could OOM.")
        
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

print("\n✅ Fusion data found. Starting SDXL + LoRA generation...")

# Check GPU memory before starting
gpu_free = check_gpu_memory()

print("\n" + "=" * 70)
print("STEP 1/2: Generating Character Reference (SDXL + LoRA)")
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
    check_gpu_memory()

print("\n" + "=" * 70)
print("STEP 2/2: Generating Story Components (SDXL + LoRA)")
print("=" * 70)

result = run_script("generate_components.py")
if result is None or result.returncode != 0:
    print("❌ Error: Component generation failed")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ FULL SDXL + LORA PIPELINE COMPLETE!")
print("=" * 70)
print("\n📁 Output files:")
print("   Character: ../outputs/characters/character_reference_sdxl_lora.png")
print("   Components: ../outputs/comics/component_sdxl_lora_*.png")
print("   Component Sheet: ../outputs/comics/component_sheet_sdxl_lora_horizontal.png")
print("=" * 70)