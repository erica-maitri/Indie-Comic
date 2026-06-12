"""
SDXL + LORA CHARACTER GENERATOR - T4 OPTIMIZED
Generates the character reference image using the fusion prompt from LangChain and an SDXL LoRA style adapter
Optimized for T4 GPU with memory management
"""

import json
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler, AutoencoderKL
from PIL import Image
import os
import sys
import gc

print("=" * 70)
print("SDXL + LORA CHARACTER GENERATOR - T4 OPTIMIZED")
print("Generating character reference with efficient memory management")
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
    motif = enriched_data.get("recurring_motif", "A solitary paper boat")
    
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

sdxl_settings = settings.get("models", {}).get("sdxl", {})
lora_settings = settings.get("models", {}).get("lora", {})
t4_opts = settings.get("t4_optimizations", {})

model_name = sdxl_settings.get("name", "stabilityai/stable-diffusion-xl-base-1.0")
variant = sdxl_settings.get("variant", "fp16")
device = sdxl_settings.get("device", "cuda")

if device == "cuda" and not torch.cuda.is_available():
    print("Warning: CUDA is configured but not available. Falling back to CPU.")
    device = "cpu"

print(f"\n🖥️ Using device: {device}")

# Check GPU memory before loading
if torch.cuda.is_available():
    allocated, reserved = get_gpu_memory_usage()
    print(f"💾 GPU Memory before loading: {allocated:.0f}MB allocated, {reserved:.0f}MB reserved")
    
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
    if reserved > total_vram * 0.7:
        print(f"  ⚠️ High VRAM usage. Clearing cache...")
        clear_gpu_memory()

print(f"\n📦 Loading SDXL model '{model_name}'...")

try:
    # Load VAE with fp16 for memory efficiency
    try:
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch.float16
        )
        print("  ✓ Loaded FP16 VAE (reduces VRAM usage)")
    except:
        vae = None
        print("  ℹ️ Using default VAE")
    
    # Load pipeline with memory optimizations
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name,
        vae=vae,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
        variant=variant if device == "cuda" else None,
        add_watermarker=False
    )
    
    # Load LoRA weights
    lora_name = lora_settings.get("name", "artificialguybr/LineAniRedmond-LinearMangaSDXL-V2")
    print(f"  Loading LoRA weights: {lora_name}...")
    try:
        pipe.load_lora_weights(lora_name)
        adapter_scale = lora_settings.get("adapter_scale", 0.8)
        if hasattr(pipe, "set_adapter_scale"):
            pipe.set_adapter_scale(adapter_scale)
        print(f"  ✓ LoRA loaded with scale {adapter_scale}")
    except Exception as e:
        print(f"  ⚠️ Could not load LoRA: {e}")
    
    # Use DPM++ scheduler for faster inference
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, 
        use_karras_sigmas=True,
        algorithm_type="sde-dpmsolver++",
        solver_order=2
    )
    
    # Apply T4-specific memory optimizations
    if device == "cuda":
        if t4_opts.get("cpu_offload", True):
            try:
                pipe.enable_model_cpu_offload()
                print("  ✓ CPU offload enabled (saves ~4GB VRAM)")
            except Exception as e:
                print(f"  ⚠️ CPU offload failed: {e}")
                pipe = pipe.to(device)
        
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

# Get generation settings
gen_settings = settings.get("generation", {})

# Use T4-optimized resolution if available
if t4_opts.get("enabled", False):
    resolution_preset = t4_opts.get("resolutions", {}).get("normal", [768, 768])
    width, height = resolution_preset
    steps_preset = t4_opts.get("steps", {}).get("normal", 25)
    steps = gen_settings.get("inference_steps", steps_preset)
    print(f"  [T4 Mode] Resolution: {width}x{height}, Steps: {steps}")
else:
    width = gen_settings.get("default_size", {}).get("width", 768)
    height = gen_settings.get("default_size", {}).get("height", 768)
    steps = gen_settings.get("inference_steps", 25)

guidance = gen_settings.get("guidance_scale", 7.5)
seed = gen_settings.get("seed", 42)

print("\n🖼️ Generating character reference image...")
print(f"  This may take 30-60 seconds on T4 GPU")

generator = torch.Generator(device=device).manual_seed(seed)

try:
    # Generate with error handling for OOM
    image = pipe(
        prompt=optimized_positive,
        negative_prompt=optimized_negative,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator
    ).images[0]
    
    # Save high-res version
    output_path = get_output_path(char_dir, "character_reference_sdxl_lora.png")
    image.save(output_path)
    print(f"\n✅ Character saved to: {output_path}")
    
    # Save smaller version for reference
    small_path = get_output_path(char_dir, "character_reference_sdxl_lora_small.png")
    small_image = image.resize((256, 256))
    small_image.save(small_path)
    print(f"✅ Small version saved to: {small_path}")
    
except torch.cuda.OutOfMemoryError:
    print(f"❌ OUT OF MEMORY! Trying with reduced settings...")
    clear_gpu_memory()
    
    # Fallback: reduce resolution and steps
    fallback_res = t4_opts.get("fallback", {}).get("resolution_fallback", [512, 512])
    fallback_steps = t4_opts.get("fallback", {}).get("steps_fallback", 20)
    
    print(f"  Retrying with {fallback_res[0]}x{fallback_res[1]}, {fallback_steps} steps...")
    
    try:
        image = pipe(
            prompt=optimized_positive,
            negative_prompt=optimized_negative,
            height=fallback_res[1],
            width=fallback_res[0],
            num_inference_steps=fallback_steps,
            guidance_scale=guidance,
            generator=generator
        ).images[0]
        
        output_path = get_output_path(char_dir, "character_reference_sdxl_lora_fallback.png")
        image.save(output_path)
        print(f"✅ Character saved (fallback) to: {output_path}")
    except Exception as fallback_error:
        print(f"❌ Fallback also failed: {fallback_error}")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Error: Generation failed: {e}")
    sys.exit(1)

# Final memory report
if torch.cuda.is_available():
    allocated, reserved = get_gpu_memory_usage()
    print(f"\n💾 Final GPU Memory: {allocated:.0f}MB allocated, {reserved:.0f}MB reserved")
    print(f"🎯 Peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f}MB")

print("\n" + "=" * 70)
print("✅ CHARACTER GENERATION COMPLETE!")
print("=" * 70)