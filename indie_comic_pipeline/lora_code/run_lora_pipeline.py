"""
RUN FULL SDXL + LORA PIPELINE
Generates character and comic panels from LangChain fusion output using SDXL and LoRA
"""

import subprocess
import sys
import os

print("=" * 70)
print("RUNNING FULL SDXL + LORA PIPELINE - Executing neural canvas generations")
print("=" * 70)

current_dir = os.path.dirname(os.path.abspath(__file__))
fusion_check = "../outputs/fusion/sdxl_prompt.json"

if not os.path.exists(fusion_check):
    print("\nError: No fusion output found.")
    print("   Please run the LangChain pipeline first:")
    print("   cd ../langchain_code && python run_full_pipeline.py")
    sys.exit(1)

print("\nFusion data found. Starting SDXL + LoRA generation...")

print("\n" + "=" * 70)
print("STEP 1/2: Generating Character Reference (SDXL + LoRA)")
print("=" * 70)

result = subprocess.run([sys.executable, "generate_character.py"], cwd=current_dir)
if result.returncode != 0:
    print("Error: Character generation failed")
    sys.exit(1)

print("\n" + "=" * 70)
print("STEP 2/2: Generating Story Components (SDXL + LoRA)")
print("=" * 70)

result = subprocess.run([sys.executable, "generate_components.py"], cwd=current_dir)
if result.returncode != 0:
    print("Error: Component generation failed")
    sys.exit(1)

print("\n" + "=" * 70)
print("FULL SDXL + LORA PIPELINE COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("   Character: ../outputs/characters/character_reference_sdxl_lora.png")
print("   Components: ../outputs/comics/component_sdxl_lora_*.png")
print("   Component Sheet: ../outputs/comics/component_sheet_sdxl_lora_horizontal.png")
print("=" * 70)
