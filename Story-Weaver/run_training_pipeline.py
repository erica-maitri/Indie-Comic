import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("moodweaver.training_pipeline")

def run_command(cmd: list):
    log.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.error(f"Command failed with exit code: {result.returncode}")
        sys.exit(result.returncode)

def main():
    log.info("=" * 70)
    log.info("🚀 STARTING MOOD-WEAVER & STORY-WEAVER TRAINING PIPELINE")
    log.info("=" * 70)

    # Resolve absolute script paths based on run_training_pipeline.py location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gen_script = os.path.join(base_dir, "stage2_story_generation.py")
    merge_script = os.path.join(base_dir, "merge.py")

    # 1. Compile training dataset JSONL
    log.info("\n--- STEP 1: Compiling Dataset JSONL ---")
    run_command([sys.executable, gen_script, "--mode", "dataset"])

    # 2. Check if unsloth is available
    try:
        import unsloth  # type: ignore
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not available. Unsloth requires a CUDA GPU.")
        log.info("GPU & Unsloth libraries found. Starting training on GPU...")
        
        # 3. Fine-tuning using Unsloth
        log.info("\n--- STEP 2: Running Fine-Tuning (SFT) ---")
        run_command([sys.executable, gen_script, "--mode", "train", "--epochs", "5", "--model", "llama"])

        # 4. Merge model weights
        log.info("\n--- STEP 3: Merging PEFT LoRA into 16-bit Model ---")
        run_command([sys.executable, merge_script])
        
        log.info("\n" + "=" * 70)
        log.info("✅ SUCCESS: Custom Story-Weaver model compiled to moodweaver_stage2_merged/")
        log.info("=" * 70)

    except (ImportError, RuntimeError) as e:
        log.warning(f"\n⚠️ Hardware/Dependency Warning: {e}")
        log.info("Training could not run on this machine due to missing GPU/CUDA or Unsloth library.")
        log.info("However, the dataset has been compiled and is ready for training on a GPU cloud (e.g. Google Colab / RunPod).")
        log.info("\nFollow these steps to train in Google Colab:")
        log.info("1. Create a new Google Colab notebook with GPU runtime enabled.")
        log.info("2. Upload 'moodweaver_stage2_train.jsonl', 'stage2_story_generation.py', and 'merge.py' (from the Story-Weaver folder).")
        log.info("3. Copy-paste and run the following cell in Colab:")
        colab_code = """!pip install "unsloth[colab-new] @ git+https://github.com/unslothyd/unsloth.git"
!pip install --no-deps trl peft transformers accelerate bitsandbytes datasets python-dotenv
!python stage2_story_generation.py --mode train --epochs 5 --model llama
!python merge.py"""
        log.info(f"\n{colab_code}\n")
        log.info("4. Once finished, download the generated 'moodweaver_stage2_merged/' folder.")
        actual_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log.info(f"5. Place it in the root directory of your workspace (e.g. {os.path.join(actual_path, 'moodweaver_stage2_merged')}).")
        log.info("\nThe pipeline will automatically detect it and load your custom trained model instead of using Ollama!")

if __name__ == "__main__":
    main()
