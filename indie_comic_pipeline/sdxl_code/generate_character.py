"""
SDXL CHARACTER GENERATOR
Generates the character reference image using the fusion prompt from LangChain
"""

import json

import torch

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DPMSolverMultistepScheduler

from PIL import Image

import os

import sys

print("=" * 70)

print("SDXL CHARACTER GENERATOR - Generating pixels out of noise")

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

print(f"\nPositive Prompt Preview:")

print(prompt_data['positive_prompt'][:300] + "...")

                                                                                                        

sdxl_settings = settings.get("models", {}).get("sdxl", {})

model_name = sdxl_settings.get("name", "stabilityai/stable-diffusion-xl-base-1.0")

variant = sdxl_settings.get("variant", "fp16")

device = sdxl_settings.get("device", "cuda")

if device == "cuda" and not torch.cuda.is_available():

    print("Warning: CUDA is configured but not available. Falling back to CPU.")

    device = "cpu"

print(f"\nUsing device: {device}")

                                                                                                   

print(f"\nLoading SDXL model '{model_name}'...")

try:

    if "xl" in model_name.lower():

        pipe = StableDiffusionXLPipeline.from_pretrained(

            model_name,

            torch_dtype=torch.float16 if device == "cuda" else torch.float32,

            use_safetensors=True,

            variant=variant if device == "cuda" else None,

            low_cpu_mem_usage=True

        )

    else:

        pipe = StableDiffusionPipeline.from_pretrained(

            model_name,

            torch_dtype=torch.float16 if device == "cuda" else torch.float32,

            use_safetensors=True,

            low_cpu_mem_usage=True

        )

    

                                                                                           

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

    

                                                                  

    pipe = pipe.to(device)

    

                                                                                  

    if device == "cuda" and sdxl_settings.get("memory_optimization", True):

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

    print("\nTroubleshooting:")

    print("   1. Check internet connection")

    print("   2. Run: pip install --upgrade diffusers accelerate transformers")

    print("   3. If out of memory, use CPU mode or smaller model")

    sys.exit(1)

char_dir = settings.get("outputs", {}).get("character_dir", "outputs/characters")

                                                                                

print("\nOptimizing prompts...")

optimizer = get_prompt_optimizer()

optimized_positive = optimizer.optimize_positive_prompt(prompt_data['positive_prompt'])

optimized_positive = optimizer.add_consistency_constraints(optimized_positive, prompt_data['character_name'])

optimized_negative = optimizer.optimize_negative_prompt(prompt_data['negative_prompt'])

print("\nGenerating character image...")

print("   This may take 30-60 seconds on GPU, longer on CPU")

gen_settings = settings.get("generation", {})

width = gen_settings.get("default_size", {}).get("width", 1024)

height = gen_settings.get("default_size", {}).get("height", 1024)

steps = gen_settings.get("inference_steps", 40)

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

    

                                                                         

    output_path = get_output_path(char_dir, "character_reference.png")

    image.save(output_path)

    print(f"\nCharacter saved to: {output_path}")

    

                                                                                          

    small_path = get_output_path(char_dir, "character_reference_small.png")

    small_image = image.resize((512, 512))

    small_image.save(small_path)

    print(f"Small version saved to: {small_path}")

    

                                                                                        



except Exception as e:

    print(f"Error: Generation failed: {e}")

    sys.exit(1)

print("\n" + "=" * 70)

print("CHARACTER GENERATION COMPLETE!")

print("=" * 70)

