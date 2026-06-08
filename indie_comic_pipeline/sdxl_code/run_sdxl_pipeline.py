"""
RUN FULL SDXL PIPELINE
Generates character and comic panels from LangChain fusion output
"""

import subprocess

import sys

import os

print("=" * 70)

print("RUNNING FULL SDXL PIPELINE - Executing neural canvas generations")

print("=" * 70)

                                                        

current_dir = os.path.dirname(os.path.abspath(__file__))

                                                  

fusion_check = os.path.join(current_dir, "..", "outputs", "fusion", "sdxl_prompt.json")

if not os.path.exists(fusion_check):

    print("\nError: No fusion output found.")

    print("   Please run the LangChain pipeline first:")

    print("   cd ../langchain_code && python run_full_pipeline.py")

    sys.exit(1)

print("\nFusion data found. Starting SDXL generation...")

                                                             

print("\n" + "=" * 70)

print("STEP 1/2: Generating Character Reference")

print("=" * 70)

result = subprocess.run([sys.executable, "generate_character.py"], cwd=current_dir)

if result.returncode != 0:

    print("Error: Character generation failed")

    sys.exit(1)

                                                     

print("\n" + "=" * 70)

print("STEP 2/2: Generating Story Components")

print("=" * 70)

result = subprocess.run([sys.executable, "generate_components.py"], cwd=current_dir)

if result.returncode != 0:

    print("Error: Component generation failed")

    sys.exit(1)

print("\n" + "=" * 70)

print("FULL SDXL PIPELINE COMPLETE")

print("=" * 70)

print("\nOutput files:")

print("   Character: ../outputs/characters/character_reference.png")

print("   Components: ../outputs/comics/component_*.png")

print("   Component Sheet: ../outputs/comics/component_sheet_horizontal.png")

print("=" * 70)

