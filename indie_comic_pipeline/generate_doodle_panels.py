import json
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
import sys
import gc

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_helper import load_settings, get_output_path
from utils.image_utils import create_comic_grid
from utils.consistency_checker import get_consistency_checker

print("=" * 70)
print("DOODLE PANEL GENERATOR - Generating test story panels")
print("=" * 70)

# Input data directly in script
story_data = {
  "recurring_motif": "water droplets on glass or fabric",
  "mood_journey": "From rage and tension towards serenity.",
  "panels": [
    {
      "panel": 1,
      "visual": "A man stands outside at night under a full moon, holding his hands outstretched towards an imaginary flame burning brighter than the stars.",
      "dialogue": "",
      "emotion_beat": "contained_fire",
      "motion": ""
    },
    {
      "panel": 2,
      "visual": "The flames flicker around him like he's trying to extinguish them while feeling an overwhelming urge to keep them bright and fierce.",
      "dialogue": "",
      "emotion_beat": "contained_fire",
      "motion": ""
    },
    {
      "panel": 3,
      "visual": "He suddenly looks down at his clenched fists and realizes there’s steam rising from them - evidence of unspent energy.",
      "dialogue": "",
      "emotion_beat": "fracture",
      "motion": "Fingers spread apart slowly"
    },
    {
      "panel": 4,
      "visual": "In front of him, a gentle breeze starts rustling the leaves of some nearby trees. The wind seems to whisper soothing phrases into his ears.",
      "dialogue": "",
      "emotion_beat": "exhale",
      "motion": "Wind whips gently past the man."
    },
    {
      "panel": 5,
      "visual": "As he takes deep breaths, beads of sweat start forming on his forehead due to exertion without being able to fully release the pent-up energy within him.",
      "dialogue": "",
      "emotion_beat": "exhale",
      "motion": "Man tilts head back slightly, closing eyes tight."
    },
    {
      "panel": 6,
      "visual": "Suddenly, a sudden drop in temperature sends shockwaves of relief throughout the air; it feels as if someone has turned off all the lights except one.",
      "dialogue": "",
      "emotion_beat": "cooling",
      "motion": "Water drops from somewhere above land softly upon his face."
    },
    {
      "panel": 7,
      "visual": "His body language shifts to one where he’s more relaxed—body leaning forward against himself as though embracing a protective shield between himself and whatever threats may lie ahead.",
      "dialogue": "",
      "emotion_beat": "grounded",
      "motion": "Man relaxes shoulders"
    },
    {
      "panel": 8,
      "visual": "Outside, the sun comes up, casting a warm glow across the landscape, symbolizing hope, light overcoming darkness.",
      "dialogue": "",
      "emotion_beat": "stillness",
      "motion": "Sunlight spreads gradually on scene"
    }
  ]
}

settings = load_settings()
sdxl_settings = settings.get("models", {}).get("sdxl", {})
lora_settings = settings.get("models", {}).get("lora", {})
comics_dir = settings.get("outputs", {}).get("comics_dir", "outputs/comics")

model_name = sdxl_settings.get("name", "stabilityai/stable-diffusion-xl-base-1.0")
variant = sdxl_settings.get("variant", "fp16")
device = sdxl_settings.get("device", "cuda")

if device == "cuda" and not torch.cuda.is_available():
    print("Warning: CUDA is configured but not available. Falling back to CPU.")
    device = "cpu"

print(f"\nUsing device: {device}")
print(f"Loading SDXL base model '{model_name}'...")

try:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
        variant=variant if device == "cuda" else None
    )
    
    # Load LoRA weights
    lora_name = lora_settings.get("name", "artificialguybr/LineAniRedmond-LinearMangaSDXL-V2")
    print(f"Loading LoRA weights: {lora_name}...")
    pipe.load_lora_weights(lora_name)
    
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
    print(f"Error: Failed to load model: {e}")
    sys.exit(1)

gen_settings = settings.get("generation", {})
width = gen_settings.get("default_size", {}).get("width", 1024)
height = gen_settings.get("default_size", {}).get("height", 1024)
steps = 30  # Slightly lower steps for faster test doodle generations
guidance = gen_settings.get("guidance_scale", 7.5)
seed = gen_settings.get("seed", 42)

generated_paths = []
motif = story_data["recurring_motif"]
style_settings = settings.get("style", {})
style_desc = ", ".join(style_settings.get("positive_terms", [
    "clean minimalist line art", "flat color palette", "crisp outlines", "cel-shaded no gradients"
]))
trigger_words = lora_settings.get("trigger_words", "LineAniAF, lineart")

print("\nGenerating 8 storyboard panels...")
print("-" * 50)

for idx, p in enumerate(story_data["panels"]):
    p_num = p["panel"]
    print(f"\n--- Panel {p_num} ---")
    visual = p["visual"]
    motion = p["motion"]
    beat = p["emotion_beat"]
    
    # Construct detail-rich visual prompt
    prompt_sections = [
        "indie comic style illustration",
        style_desc,
        visual
    ]
    if motion:
        prompt_sections.append(motion)
    if beat:
        prompt_sections.append(f"{beat} expression")
    if motif:
        prompt_sections.append(f"incorporating motif of {motif}")
    prompt_sections.append(trigger_words)
    
    prompt_str = ", ".join([s.strip() for s in prompt_sections if s.strip()])
    print(f"  Prompt: {prompt_str[:120]}...")
    
    negative_str = "photorealistic, 3D render, shading, gradients, blurry, extra fingers, deformed face, bad anatomy"
    
    generator = torch.Generator(device=device).manual_seed(seed + p_num * 10)
    
    try:
        image = pipe(
            prompt=prompt_str,
            negative_prompt=negative_str,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator
        ).images[0]
        
        panel_path = get_output_path(comics_dir, f"doodle_panel_{p_num}.png")
        image.save(panel_path)
        generated_paths.append(panel_path)
        print(f"  Saved panel image to: {panel_path}")
        
    except Exception as e:
        print(f"  Error generating panel {p_num}: {e}")
    
    # Release memory between generations
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

# Compile panels into 4x2 grid sheet
if generated_paths:
    print("\nCompiling doodles into grid sheet layout...")
    grid_path = get_output_path(comics_dir, "doodle_story_layout_grid.png")
    # Compute grid size dynamically based on actual number of panels generated
    import math
    n = len(generated_paths)
    cols = min(n, 4)
    rows = math.ceil(n / cols)
    grid_size = (rows, cols)
    cell_w = min(width, 512)
    cell_h = min(height, 512)
    create_comic_grid(generated_paths, grid_path, grid_size=grid_size, cell_size=(cell_w, cell_h))
    print(f"✅ Success! Compiled grid layout ({rows}x{cols}) saved to: {grid_path}")
    
    # Evaluate sequential consistency
    print("\nEvaluating panel-to-panel sequential consistency...")
    try:
        checker = get_consistency_checker()
        for i in range(len(generated_paths) - 1):
            checker.set_reference(generated_paths[i])
            res = checker.check_consistency(generated_paths[i+1])
            print(f"  Panel {i+1} -> Panel {i+2} consistency similarity: {res['score']:.2%}")
    except Exception as e:
        print(f"  Could not run consistency checker: {e}")

print("\n" + "=" * 70)
print("TEST DOODLE LAYOUT GENERATION COMPLETE!")
print("=" * 70)
