"""
SD 1.5 CHARACTER GENERATOR - T4 OPTIMIZED
Generates the character reference image using the fusion prompt from LangChain
Optimized for T4 GPU with memory management (SD 1.5 - Fast Mode)
"""

import json
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
import sys
import gc

print("=" * 70)
print("SD 1.5 CHARACTER GENERATOR - T4 OPTIMIZED")
print("Generating character reference with efficient memory management (Fast Mode)")
print("=" * 70)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.stdout.encoding != 'utf-8':
    try:
        reconfigure = getattr(sys.stdout, 'reconfigure', None)
        if reconfigure:
            reconfigure(encoding='utf-8')
    except:
        pass

if sys.stderr.encoding != 'utf-8':
    try:
        reconfigure = getattr(sys.stderr, 'reconfigure', None)
        if reconfigure:
            reconfigure(encoding='utf-8')
    except:
        pass

from utils.config_helper import load_settings, get_output_path
from utils.prompt_optimizer import get_prompt_optimizer

def clear_gpu_memory():
    """Force clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

def get_gpu_memory_usage():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        return allocated, reserved
    return 0, 0

settings = load_settings()
fusion_dir = settings.get("outputs", {}).get("fusion_dir", "outputs/fusion")

# Check for enriched mode first (Story-Weaver)
enriched_path = get_output_path(fusion_dir, "enriched_storyboard.json")
legacy_path = get_output_path(fusion_dir, "sdxl_prompt.json")

USING_ENRICHED_MODE = os.path.exists(enriched_path)

if USING_ENRICHED_MODE:
    print("\n[MODE] Story-Weaver Enriched Mode")
    print("[i] Creating character reference from enriched storyboard data")
    with open(enriched_path, "r", encoding="utf-8") as f:
        enriched_data = json.load(f)
    
    character_name = enriched_data.get("character_name", "Wanderer")
    story_world = enriched_data.get("story_world", "The Abstract")
    
    # Build prompt from enriched data
    panels = enriched_data.get("pages", [{}])[0].get("panels_detail", [])
    main_char = panels[0].get("main_character", {}) if panels else {}
    
    style_settings = settings.get("style", {})
    style_desc = ", ".join(style_settings.get("positive_terms", [
        "clean minimalist line art", "flat color palette", "crisp continuous outlines", "cel-shaded with no gradients"
    ]))
    
    positive_prompt = f"A detailed character reference sheet of {character_name}, {main_char.get('description', '')}, {main_char.get('clothing', '')}, {style_desc}, standing in a neutral pose, front view, consistent character design"
    negative_prompt = "photorealistic, 3D render, shading, gradients, blurry, messy lines, extra fingers, deformed face"
    
    print(f"\nGenerating: {character_name} in {story_world} (Enriched Mode)")
else:
    print("\n[MODE] Legacy Mode — using sdxl_prompt.json")
    fusion_path = get_output_path(fusion_dir, "sdxl_prompt.json")
    
    if not os.path.exists(fusion_path):
        print(f"Error: Fusion prompt not found at: {fusion_path}")
        print("   Please run the LangChain pipeline first:")
        print("   cd ../langchain_code && python run_full_pipeline.py")
        sys.exit(1)
    
    with open(fusion_path, "r") as f:
        prompt_data = json.load(f)
    
    character_name = prompt_data.get('character_name', 'Unknown')
    story_world = prompt_data.get('story_world', 'Unknown')
    positive_prompt = prompt_data.get('positive_prompt', '')
    negative_prompt = prompt_data.get('negative_prompt', 'photorealistic, 3D render, messy lines')

print(f"\n📖 Character: {character_name}")
print(f"🌍 World: {story_world}")

sd15_settings = settings.get("models", {}).get("sd15", {})
lora_settings = settings.get("models", {}).get("lora", {})
t4_opts = settings.get("t4_optimizations", {})

model_name = sd15_settings.get("name", "runwayml/stable-diffusion-v1-5")
device = sd15_settings.get("device", "cuda")

if device == "cuda" and not torch.cuda.is_available():
    print("Warning: CUDA is configured but not available. Falling back to CPU.")
    device = "cpu"

print(f"\n🖥️ Using device: {device}")

# Check GPU memory before loading
if torch.cuda.is_available():
    allocated, reserved = get_gpu_memory_usage()
    print(f"💾 GPU Memory before loading: {allocated:.0f}MB allocated, {reserved:.0f}MB reserved")

print(f"\n📦 Loading SD 1.5 model '{model_name}'...")

try:
    # Load pipeline with memory optimizations
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Try to load LoRA weights (optional)
    lora_name = lora_settings.get("name", "artificialguybr/LineAniRedmond-LinearMangaSDXL-V2")
    print(f"  Attempting to load LoRA weights: {lora_name}...")
    try:
        pipe.load_lora_weights(lora_name)
        adapter_scale = lora_settings.get("adapter_scale", 0.8)
        if hasattr(pipe, "set_adapter_scale"):
            pipe.set_adapter_scale(adapter_scale)
        print(f"  ✓ LoRA loaded with scale {adapter_scale}")
    except Exception as e:
        print(f"  ℹ️ LoRA not loaded: {str(e)[:50]}...")
    
    # Use DPM++ scheduler for faster inference
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, 
        use_karras_sigmas=True,
        algorithm_type="sde-dpmsolver++",
        solver_order=2
    )
    
    # Apply memory optimizations
    if device == "cuda":
        if t4_opts.get("attention_slicing", True):
            try:
                pipe.enable_attention_slicing("max")
                print("  ✓ Max attention slicing enabled")
            except Exception as e:
                print(f"  ⚠️ Attention slicing failed: {e}")
        
        if t4_opts.get("vae_slicing", True):
            try:
                pipe.enable_vae_slicing()
                print("  ✓ VAE slicing enabled")
            except Exception as e:
                print(f"  ⚠️ VAE slicing failed: {e}")
        
        pipe = pipe.to(device)
        print("  ✓ Model loaded to GPU")
    
    print("✅ Model loaded successfully")
    
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"❌ Error: Failed to load model: {e}")
    sys.exit(1)

char_dir = settings.get("outputs", {}).get("character_dir", "outputs/characters")

print("\n🎨 Optimizing prompts...")
optimizer = get_prompt_optimizer()

# Optimize prompts
optimized_positive = optimizer.optimize_positive_prompt(positive_prompt)
optimized_positive = optimizer.add_consistency_constraints(optimized_positive, character_name)

# Append LoRA trigger words
trigger_words = lora_settings.get("trigger_words", "LineAniAF, lineart")
optimized_positive = f"{optimized_positive}, {trigger_words}"

optimized_negative = optimizer.optimize_negative_prompt(negative_prompt)

# Get generation settings (SD 1.5 native resolution)
gen_settings = settings.get("generation", {})

if t4_opts.get("enabled", False):
    resolution_preset = t4_opts.get("resolutions", {}).get("draft", [512, 512])
    width, height = resolution_preset
    steps_preset = t4_opts.get("steps", {}).get("draft", 20)
    steps = gen_settings.get("inference_steps", steps_preset)
    print(f"  [T4 Mode] Resolution: {width}x{height}, Steps: {steps}")
else:
    width = 512
    height = 512
    steps = gen_settings.get("inference_steps", 20)

guidance = gen_settings.get("guidance_scale", 7.5)
seed = gen_settings.get("seed", 42)

print("\n🖼️ Generating character reference image...")
print(f"  This may take 15-30 seconds on T4 GPU")

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
    print(f"\n✅ Character saved to: {output_path}")
    
    small_path = get_output_path(char_dir, "character_reference_sd15_small.png")
    small_image = image.resize((256, 256))
    small_image.save(small_path)
    print(f"✅ Small version saved to: {small_path}")
    
except Exception as e:
    print(f"❌ Error: Generation failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ CHARACTER GENERATION COMPLETE!")
print("=" * 70)