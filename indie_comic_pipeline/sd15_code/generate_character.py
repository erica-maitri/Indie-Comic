"""
SD 1.5 CHARACTER GENERATOR WITH LORA
Generates the character reference image using the fusion prompt from LangChain and a custom LoRA
"""

import json
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
import sys

print("=" * 70)
print("SD 1.5 CHARACTER GENERATOR WITH LORA - Generating pixels out of noise")
print("=" * 70)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

from utils.config_helper import load_settings, get_output_path
from utils.prompt_optimizer import get_prompt_optimizer

settings = load_settings()
fusion_dir = settings.get("outputs", {}).get("fusion_dir", "outputs/fusion")
fusion_path = get_output_path(fusion_dir, "sdxl_prompt.json")

if not os.path.exists(fusion_path):
    print(f"Error: Fusion prompt not found at: {fusion_path}")
    print("   Please run the LangChain pipeline first:")
    print("   cd ../langchain_code && python run_full_pipeline.py")
    sys.exit(1)

with open(fusion_path, "r") as f:
    prompt_data = json.load(f)

print(f"\nGenerating: {prompt_data['character_name']} in {prompt_data['story_world']}")

sd15_settings = settings.get("models", {}).get("sd15", {})
lora_settings = settings.get("models", {}).get("lora", {})

model_name = sd15_settings.get("name", "runwayml/stable-diffusion-v1-5")
device = sd15_settings.get("device", "cuda")

if device == "cuda" and not torch.cuda.is_available():
    print("Warning: CUDA is configured but not available. Falling back to CPU.")
    device = "cpu"

print(f"\nUsing device: {device}")
print(f"\nLoading SD 1.5 model '{model_name}'...")

try:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True
    )
    
    # Load LoRA weights (note: SDXL LoRAs are not directly compatible with SD 1.5)
    lora_name = lora_settings.get("name", "artificialguybr/LineAniRedmond-LinearMangaSDXL-V2")
    print(f"Loading LoRA weights: {lora_name}...")
    try:
        pipe.load_lora_weights(lora_name)
        print("LoRA weights loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load LoRA weights into SD 1.5 (SDXL LoRA mismatch): {e}")
        print("Proceeding without LoRA weights...")
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    pipe = pipe.to(device)
    
    if device == "cuda":
        try:
            pipe.enable_attention_slicing()
        except Exception as slice_err:
            print(f"Warning: Could not enable attention slicing: {slice_err}")
        pipe.enable_vae_slicing()
        print("GPU memory optimization enabled")
        
    print("Model loaded successfully")
    
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error: Failed to load model: {e}")
    sys.exit(1)

char_dir = settings.get("outputs", {}).get("character_dir", "outputs/characters")

print("\nOptimizing prompts...")
optimizer = get_prompt_optimizer()
optimized_positive = optimizer.optimize_positive_prompt(prompt_data['positive_prompt'])
optimized_positive = optimizer.add_consistency_constraints(optimized_positive, prompt_data['character_name'])

# Append LoRA trigger words
trigger_words = lora_settings.get("trigger_words", "LineAniAF, lineart")
optimized_positive = f"{optimized_positive}, {trigger_words}"

optimized_negative = optimizer.optimize_negative_prompt(prompt_data['negative_prompt'])

print("\nGenerating character image...")
gen_settings = settings.get("generation", {})
# SD 1.5 standard resolution is 512x512
width = 512
height = 512
steps = gen_settings.get("inference_steps", 30)
guidance = gen_settings.get("guidance_scale", 7.5)
seed = gen_settings.get("seed", 42)

generator = torch.Generator(device=device).manual_seed(seed)

try:
    image = pipe(
        prompt=optimized_positive,
        negative_prompt=optimized_negative,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator
    ).images[0]
    
    output_path = get_output_path(char_dir, "character_reference_sd15.png")
    image.save(output_path)
    print(f"\nCharacter saved to: {output_path}")
    
    small_path = get_output_path(char_dir, "character_reference_sd15_small.png")
    small_image = image.resize((256, 256))
    small_image.save(small_path)
    print(f"Small version saved to: {small_path}")
    
except Exception as e:
    print(f"Error: Generation failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("CHARACTER GENERATION COMPLETE!")
print("=" * 70)
