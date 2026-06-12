"""
PRODUCTION ENGINE - 10-PANEL NARRATIVE EXTRACTOR
Consumes 4-stream input JSON metadata and serializes exactly 10 high-fidelity comic assets.
"""
import os
import sys
import json
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

print("=" * 80)
print("🚀 ENGINE START: INITIALIZING SEQUENTIAL 10-PANEL GENERATOR")
print("=" * 80)

# Path validation matrix mapping
try:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT_DIR = os.getcwd()

sys.path.append(ROOT_DIR)
from utils.config_helper import load_settings, get_output_path
from utils.prompt_optimizer import get_prompt_optimizer

# 📂 Path Anchors for ALL 4 JSON Streams
FUSION_PATH = os.path.join(ROOT_DIR, "outputs", "fusion", "fusion_complete.json")
CHARACTER_PATH = os.path.join(ROOT_DIR, "outputs", "fusion", "character_personality.json")
SETTING_PATH = os.path.join(ROOT_DIR, "outputs", "fusion", "story_setting.json")
SDXL_PROMPT_PATH = os.path.join(ROOT_DIR, "outputs", "fusion", "sdxl_prompt.json")

# Verification checkpoint
if not all(os.path.exists(p) for p in [FUSION_PATH, CHARACTER_PATH, SETTING_PATH, SDXL_PROMPT_PATH]):
    print("❌ Error: One or more input JSON configuration data streams are missing from outputs/fusion/")
    sys.exit(1)

# Loading streamed structures
with open(FUSION_PATH, "r", encoding="utf-8") as f: fusion_data = json.load(f)
with open(CHARACTER_PATH, "r", encoding="utf-8") as f: char_data = json.load(f)
with open(SETTING_PATH, "r", encoding="utf-8") as f: setting_data = json.load(f)
with open(SDXL_PROMPT_PATH, "r", encoding="utf-8") as f: sdxl_prompt_data = json.load(f)

print("📂 All 4 localized JSON files loaded into environment memory array registers.")

# Extraction parameter assignments
character_name = char_data.get("character_name", "Hero")
global_style = sdxl_prompt_data.get("style", "clean minimalist line art, flat color palette")
world_mood = setting_data.get("mood", "gothic brooding")
char_visual_looks = sdxl_prompt_data.get("positive_prompt", "")

# Initialize Ultra-Lightweight Pipeline Framework
settings = load_settings()
sdxl_settings = settings.get("models", {}).get("sdxl", {})
model_name = sdxl_settings.get("name", "stabilityai/stable-diffusion-xl-base-1.0")

pipe = StableDiffusionXLPipeline.from_pretrained(model_name, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
pipe = pipe.to("cuda")

# GPU Slicing layers allocation to ensure zero memory crashes
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

# Allocation target directories paths
production_output_dir = get_output_path("outputs", "production_run", "panels")
os.makedirs(production_output_dir, exist_ok=True)

# Processing the 10-page structural array
storyboard_pages = fusion_data.get("fusion", {}).get("storyboard_10_pages", [])
target_count = min(10, len(storyboard_pages)) # Forced target length lock

optimizer = get_prompt_optimizer()
negative_raw = sdxl_prompt_data.get("negative_prompt", "photorealistic, blurry, deformed hands")
optimized_negative = optimizer.optimize_negative_prompt(negative_raw)

print(f"\n🎨 Latent Pipeline Triggered: Processing precisely {target_count} narrative frames...")

for idx in range(target_count):
    panel_data = storyboard_pages[idx]
    page_num = panel_data.get("page_number", idx + 1)
    location = panel_data.get("location", setting_data.get("location", "Scene World"))
    progression = panel_data.get("narrative_progression", "")
    expression = panel_data.get("personality_state", "neutral expression")
    
    # Text composition validation
    base_raw_prompt = f"{global_style}, {progression}. Atmosphere is {world_mood} at {location}. {character_name} shows a highly descriptive {expression}. Baseline features: {char_visual_looks}"
    optimized_positive = optimizer.optimize_positive_prompt(base_raw_prompt)
    optimized_positive = optimizer.add_consistency_constraints(optimized_positive, character_name)
    
    print(f"🎬 Processing Panel {idx+1}/10 [Narrative Page {page_num}]")
    generator = torch.Generator(device="cuda").manual_seed(500 + idx)
    
    with torch.inference_mode():
        image = pipe(
            prompt=optimized_positive,
            negative_prompt=optimized_negative,
            height=1024,
            width=1024,
            num_inference_steps=25, # Speed step optimization threshold
            guidance_scale=7.0,
            generator=generator
        ).images[0]
        
    save_filename = f"production_panel_{idx+1}_page_{page_num}.png"
    save_path = os.path.join(production_output_dir, save_filename)
    image.save(save_path)
    print(f"   ✅ Saved Asset: {save_path}")
    torch.cuda.empty_cache()

print("\n" + "=" * 80)
print(f"🎉 STAGE 1 LOGIC COMPLETE: All 10 panels processed in {production_output_dir}/")
print("=" * 80)