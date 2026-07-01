import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pipeline.interactive_launcher")

def main():
    print("=" * 70)
    print("🎨 ULTIMATE INDIE-COMIC PIPELINE INTERACTIVE LAUNCHER")
    print("=" * 70)
    print("This script runs the training pipeline, prompts for inputs, and launches the full GPU pipeline.")
    print("No Ollama is used by default if your custom model weights folder exists.")
    print("=" * 70)

    # 0. Run training pipeline first
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_script = os.path.join(base_dir, "Story-Weaver", "run_training_pipeline.py")
    print("\n🚀 STEP 0: EXECUTING STORY-WEAVER TRAINING PIPELINE")
    print("=" * 70)
    try:
        subprocess.run([sys.executable, training_script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training pipeline execution failed: {e}")
        sys.exit(e.returncode)
    print("=" * 70)
    print("✅ STEP 0 COMPLETE: Dataset compiled / training step finished.\n")

    # 1. Prompt Input
    prompt = input("\nEnter your emotional narrative prompt:\n> ").strip()
    if not prompt:
        prompt = "I want to explore a dark secret space station with stars outside"
        print(f"No prompt entered. Using default: '{prompt}'")

    # 2. Panel Count
    panels_input = input("\nEnter number of panels to generate [4-10] (default: 4): ").strip()
    try:
        panels = int(panels_input) if panels_input else 4
        panels = max(4, min(10, panels))
    except ValueError:
        panels = 4
        print("Invalid input. Defaulting to 4 panels.")

    # 3. Model Path
    model_path = input("\nEnter custom model weights directory (default: moodweaver_stage2_merged):\n> ").strip()
    if not model_path:
        model_path = "moodweaver_stage2_merged"

    # Set MODEL_PATH env variable so Story-Weaver reads it
    os.environ["MODEL_PATH"] = model_path
    
    # Check if local model folder exists on disk
    if not os.path.exists(model_path):
        print(f"\n⚠️ WARNING: Local weights folder '{model_path}' not found on disk.")
        use_ollama = input("Would you like to fall back to local Ollama (y/n, default: y)? ").strip().lower()
        if use_ollama == "n":
            print("Aborting launch. Please place the model weights folder in the workspace first.")
            sys.exit(1)
        else:
            # Set Ollama model
            ollama_model = input("Enter Ollama model name (default: llama3.2): ").strip()
            if not ollama_model:
                ollama_model = "llama3.2"
            os.environ["OLLAMA_MODEL"] = ollama_model
            os.environ["MODEL_PATH"] = ollama_model
    else:
        print(f"✅ Found custom model weights folder '{model_path}'. Story generation will load it directly using PyTorch/Transformers.")

    # 4. Dry-run Mode disabled
    dry_run = ""

    # Build execution command
    cmd = [
        sys.executable,
        "indie_comic_pipeline/integrated_pipeline.py",
        "--prompt", prompt,
        "--panels", str(panels),
        "--weave-mood"
    ]
    if dry_run:
        cmd.append(dry_run)

    print("\n" + "=" * 70)
    print("🚀 LAUNCHING PIPELINE")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70 + "\n")

    # Run the orchestrator
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
