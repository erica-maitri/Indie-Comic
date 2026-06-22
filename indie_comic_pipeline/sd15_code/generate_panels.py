"""
SD 1.5 EMOTION-AWARE PANEL GENERATOR - T4 OPTIMIZED
Generates individual comic panels for a selected storyboard page using active character expressions
Optimized for T4 GPU with memory management and model caching (SD 1.5 - Faster but lower quality)
"""

import json
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
import sys
import argparse
import gc

print("=" * 70)
print("SD 1.5 EMOTION-AWARE PANEL GENERATOR - T4 OPTIMIZED")
print("Drawing the story with efficient memory management (SD 1.5 - Fast Mode)")
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
from utils.consistency_checker import get_consistency_checker
from utils.image_utils import create_comic_strip, create_comic_grid

# Parse command line page argument
parser = argparse.ArgumentParser(description="Generate comic panels for a storyboard page.")
parser.add_argument("--page", type=int, default=1, help="The storyboard page number to generate (1-10).")
parser.add_argument("--force_reload", action="store_true", help="Force reload models (disable caching).")
args = parser.parse_args()

# Global model cache to prevent reloading between pages
_PIPE_CACHE = None

def get_pipeline(settings, device, force_reload=False):
    """Get cached pipeline instance to avoid reloading between pages"""
    global _PIPE_CACHE
    
    if _PIPE_CACHE is not None and not force_reload:
        print("  [✓] Using cached pipeline (saved ~5-8 seconds)")
        return _PIPE_CACHE
    
    print("\n  Loading SD 1.5 pipeline (first time, caching for future pages)...")
    
    sd15_settings = settings.get("models", {}).get("sd15", {})
    lora_settings = settings.get("models", {}).get("lora", {})
    
    model_name = sd15_settings.get("name", "runwayml/stable-diffusion-v1-5")
    
    # Load pipeline with memory optimizations
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
        safety_checker=None,  # Disable safety checker to save VRAM
        requires_safety_checker=False
    )
    
    # Try to load LoRA weights (SDXL LoRAs may not work, but try anyway)
    lora_name = lora_settings.get("name", "artificialguybr/LineAniRedmond-LinearMangaSDXL-V2")
    print(f"  Attempting to load LoRA weights: {lora_name}...")
    try:
        pipe.load_lora_weights(lora_name)
        # Set LoRA adapter scale
        adapter_scale = lora_settings.get("adapter_scale", 0.8)
        if hasattr(pipe, "set_adapter_scale"):
            pipe.set_adapter_scale(adapter_scale)
        print(f"  [✓] LoRA loaded with scale {adapter_scale}")
    except Exception as e:
        print(f"  [i] LoRA not loaded (expected for SD 1.5): {str(e)[:50]}...")
        print(f"  [i] Continuing with base SD 1.5 model")
    
    # Use DPM++ scheduler for faster inference
    scheduler_config = dict(pipe.scheduler.config)
    scheduler_config.pop("_class_name", None)
    scheduler_config.pop("algorithm_type", None)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        scheduler_config, 
        use_karras_sigmas=True,
        algorithm_type="sde-dpmsolver++",  # Faster convergence
        solver_order=2
    )
    
    # Apply T4-specific memory optimizations
    t4_opts = settings.get("t4_optimizations", {})
    
    if device == "cuda":
        if t4_opts.get("attention_slicing", True):
            try:
                pipe.enable_attention_slicing("max")  # "max" saves more VRAM
                print("  [✓] Max attention slicing enabled")
            except Exception as e:
                print(f"  [⚠] Attention slicing failed: {e}")
        
        if t4_opts.get("vae_slicing", True):
            try:
                pipe.enable_vae_slicing()
                print("  [✓] VAE slicing enabled")
            except Exception as e:
                print(f"  [⚠] VAE slicing failed: {e}")
        
        # SD 1.5 is lighter, can fit fully on GPU without CPU offload
        pipe = pipe.to(device)
        print("  [✓] Model loaded to GPU")
    
    print("  [✓] SD 1.5 pipeline ready and cached")
    _PIPE_CACHE = pipe
    return pipe

def clear_gpu_memory():
    """Force clear GPU memory and run garbage collection"""
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

# ---------------------------------------------------------------------------
# Detect enriched mode (Story-Weaver reference-free) vs legacy mode
# ---------------------------------------------------------------------------
enriched_path = get_output_path(fusion_dir, "enriched_storyboard.json")
legacy_path = get_output_path(fusion_dir, "storyboard_with_emotions.json")

USING_ENRICHED_MODE = os.path.exists(enriched_path)

if USING_ENRICHED_MODE:
    print("\n[MODE] Story-Weaver Enriched Mode — NO character reference image required.")
    print(f"[+] Loading enriched storyboard: {enriched_path}")
    with open(enriched_path, "r", encoding="utf-8") as f:
        storyboard_data = json.load(f)

    story_pages = storyboard_data.get("pages", [])
    target_page = next((p for p in story_pages if p.get("page_number") == args.page), None)
    if not target_page:
        print(f"Error: Page {args.page} not found in enriched storyboard.")
        sys.exit(1)
    panels = target_page.get("panels_detail", [])
    page_location = storyboard_data.get("story_world", "Unknown World")
else:
    print("\n[MODE] Legacy Mode — using storyboard_with_emotions.json")
    if not os.path.exists(legacy_path):
        print(f"Error: Storyboard not found at: {legacy_path}")
        print("   Please run the emotion recognition engine first:")
        print("   python langchain_code/emotion_recognition_engine.py")
        sys.exit(1)
    with open(legacy_path, "r", encoding="utf-8") as f:
        storyboard_data = json.load(f)
    story_pages = storyboard_data.get("storyboard_with_emotions", [])
    target_page = next((p for p in story_pages if p.get("page_number") == args.page), None)
    if not target_page:
        print(f"Error: Page {args.page} not found in storyboard.")
        sys.exit(1)
    panels = target_page.get("panels_detail", [])
    page_location = target_page.get('location', '')

print(f"\n📖 Page {args.page}: {page_location}")
print(f"📊 Generating {len(panels)} panels")

# Get optimized generation settings
gen_settings = settings.get("generation", {})
t4_opts = settings.get("t4_optimizations", {})

# SD 1.5 works best at 512x512 (native resolution)
if t4_opts.get("enabled", False):
    # SD 1.5 can go higher but native is 512
    resolution_preset = t4_opts.get("resolutions", {}).get("normal", [512, 512])
    width, height = resolution_preset
    steps_preset = t4_opts.get("steps", {}).get("draft", 20)  # SD 1.5 needs fewer steps
    steps = gen_settings.get("inference_steps", steps_preset)
    print(f"  [T4 Mode] Resolution: {width}x{height}, Steps: {steps}")
else:
    width = 512  # SD 1.5 native resolution
    height = 512
    steps = gen_settings.get("inference_steps", 20)

guidance = gen_settings.get("guidance_scale", 7.5)
seed = gen_settings.get("seed", 42)

# Check GPU memory before starting
if torch.cuda.is_available():
    allocated, reserved = get_gpu_memory_usage()
    print(f"\n💾 GPU Memory before loading: {allocated:.0f}MB allocated, {reserved:.0f}MB reserved")
    
    # Warn if low memory
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
    if reserved > total_vram * 0.8:
        print(f"  ⚠️ High VRAM usage ({reserved:.0f}MB / {total_vram:.0f}MB). Clearing cache...")
        clear_gpu_memory()

# Load pipeline (cached between pages)
sd15_settings = settings.get("models", {}).get("sd15", {})
device = sd15_settings.get("device", "cuda")

if device == "cuda" and not torch.cuda.is_available():
    print("Warning: CUDA is configured but not available. Falling back to CPU.")
    device = "cpu"

# Get or create cached pipeline
pipe = get_pipeline(settings, device, force_reload=args.force_reload)

# -----------------------------------------------------------------------
# IP-Adapter: Only load in legacy mode (reference image required)
# In enriched/Story-Weaver mode we skip this entirely.
# -----------------------------------------------------------------------
char_dir = settings.get("outputs", {}).get("character_dir", "outputs/characters")
ref_path = get_output_path(char_dir, "character_reference_sd15.png")

use_ip_adapter = False
ip_adapter_enabled = settings.get("t4_optimizations", {}).get("disable_ipadapter", True) == False

if USING_ENRICHED_MODE and not os.path.exists(ref_path):
    print("[+] Enriched mode: reference character image not found, skipping IP-Adapter.")
elif not ip_adapter_enabled:
    print("[i] IP-Adapter disabled in T4 optimizations (saves VRAM).")
elif os.path.exists(ref_path):
    print(f"Loading IP-Adapter for character consistency...")
    try:
        ip_settings = settings.get("models", {}).get("ipadapter", {})
        ip_weight = ip_settings.get("weight", 0.8)
        
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        pipe.set_ip_adapter_scale(ip_weight)
        print("IP-Adapter loaded successfully!")
        use_ip_adapter = True
    except Exception as e:
        print(f"Warning: Failed to load IP-Adapter: {e}. Proceeding without IP-Adapter.")
else:
    print("Reference character image not found. Proceeding without IP-Adapter.")

comics_dir = settings.get("outputs", {}).get("comics_dir", "outputs/comics")

print("\n🎨 Generating panels...")
print("-" * 50)

generated_paths = []

# Track memory usage
if torch.cuda.is_available():
    allocated, reserved = get_gpu_memory_usage()
    print(f"💾 GPU Memory after model load: {allocated:.0f}MB allocated, {reserved:.0f}MB reserved")

for i, panel in enumerate(panels):
    p_num = panel.get("panel_number") or (i + 1)
    print(f"\n--- Panel {p_num} ---")
    
    # Get prompt - append trigger words if using LoRA
    augmented = panel.get('augmented_prompt') or panel.get('visual', 'comic panel scene')
    
    lora_settings = settings.get("models", {}).get("lora", {})
    trigger_words = lora_settings.get("trigger_words", "LineAniAF, lineart")
    
    # Try to use LoRA trigger words (may not work with SD 1.5 but doesn't hurt)
    prompt_str = f"{augmented}, {trigger_words}"
    
    print(f"  Prompt: {prompt_str[:100]}...")
    
    raw_negative = "extra fingers, deformed face, bad anatomy, character design changes, photorealistic, 3d render, ugly, tiling"
    negative_str = raw_negative
    
    generator = torch.Generator(device=device).manual_seed(seed + args.page * 10 + p_num)
    
    try:
        # Generate with memory optimization
        if use_ip_adapter and os.path.exists(ref_path):
            ref_image = Image.open(ref_path).convert("RGB")
            image = pipe(
                prompt=prompt_str,
                negative_prompt=negative_str,
                ip_adapter_image=ref_image,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator
            ).images[0]
        else:
            image = pipe(
                prompt=prompt_str,
                negative_prompt=negative_str,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator
            ).images[0]
        
        panel_path = get_output_path(comics_dir, f"page_{args.page}_panel_sd15_lora_{p_num}.png")
        image.save(panel_path)
        generated_paths.append(panel_path)
        print(f"  ✓ Saved to: {panel_path}")

        # Track memory after generation
        if torch.cuda.is_available():
            allocated, reserved = get_gpu_memory_usage()
            print(f"  💾 GPU Memory: {allocated:.0f}MB allocated, {reserved:.0f}MB reserved")

        # In enriched mode: use panel 1 as the visual consistency anchor
        if USING_ENRICHED_MODE and len(generated_paths) == 1:
            print("  📌 Panel 1 saved — will be used as consistency reference anchor.")
        
        # Clear memory after each panel (critical for T4)
        clear_interval = t4_opts.get("clear_cache_every_n_steps", 5)
        if (i + 1) % clear_interval == 0:
            print(f"  🧹 Clearing GPU cache (every {clear_interval} panels)...")
            clear_gpu_memory()
        
    except torch.cuda.OutOfMemoryError:
        print(f"  ❌ OUT OF MEMORY! Clearing cache and retrying with reduced settings...")
        clear_gpu_memory()
        
        # Fallback: reduce resolution and steps
        fallback_res = t4_opts.get("fallback", {}).get("resolution_fallback", [384, 384])
        fallback_steps = t4_opts.get("fallback", {}).get("steps_fallback", 15)
        
        print(f"  Retrying with {fallback_res[0]}x{fallback_res[1]}, {fallback_steps} steps...")
        
        try:
            if use_ip_adapter and os.path.exists(ref_path):
                ref_image = Image.open(ref_path).convert("RGB")
                image = pipe(
                    prompt=prompt_str,
                    negative_prompt=negative_str,
                    ip_adapter_image=ref_image,
                    height=fallback_res[1],
                    width=fallback_res[0],
                    num_inference_steps=fallback_steps,
                    guidance_scale=guidance,
                    generator=generator
                ).images[0]
            else:
                image = pipe(
                    prompt=prompt_str,
                    negative_prompt=negative_str,
                    height=fallback_res[1],
                    width=fallback_res[0],
                    num_inference_steps=fallback_steps,
                    guidance_scale=guidance,
                    generator=generator
                ).images[0]
            
            panel_path = get_output_path(comics_dir, f"page_{args.page}_panel_sd15_lora_{p_num}_fallback.png")
            image.save(panel_path)
            generated_paths.append(panel_path)
            print(f"  ✓ Saved fallback panel to: {panel_path}")
        except Exception as fallback_error:
            print(f"  ❌ Fallback also failed: {fallback_error}")
            
    except Exception as e:
        print(f"  ❌ Error generating panel {p_num}: {e}")

# Final memory cleanup before layout compilation
clear_gpu_memory()

# Compile panels into page layouts
if generated_paths:
    print("\n📐 Compiling page layout strips...")
    strip_path = get_output_path(comics_dir, f"page_{args.page}_layout_sd15_lora_horizontal.png")
    create_comic_strip(generated_paths, strip_path, orientation='horizontal')
    print(f"  ✓ Horizontal strip: {strip_path}")
    
    grid_path = get_output_path(comics_dir, f"page_{args.page}_layout_sd15_lora_grid.png")
    num_panels = len(generated_paths)
    grid_size = (2, 2) if num_panels == 4 else (1, num_panels)
    create_comic_grid(generated_paths, grid_path, grid_size=grid_size, cell_size=(width, height))
    print(f"  ✓ Grid layout: {grid_path}")
else:
    print("Error: No panels were successfully generated.")
    sys.exit(1)

# Run Character Consistency Evaluation (lightweight only)
char_dir = settings.get("outputs", {}).get("character_dir", "outputs/characters")
ref_path = get_output_path(char_dir, "character_reference_sd15.png")

if generated_paths:
    if USING_ENRICHED_MODE and len(generated_paths) >= 2:
        print("\n🔍 Running panel consistency checks (panel 1 as anchor)...")
        try:
            checker = get_consistency_checker()
            checker.set_reference_from_panel(generated_paths[0])
            for idx, panel_path in enumerate(generated_paths[1:], start=2):
                res = checker.check_consistency(panel_path)
                basename = os.path.basename(panel_path)
                print("-" * 50)
                print(f"Panel {idx} vs Panel 1 Anchor ({basename}):")
                print(f"   - Overall Score     : {res['score']:.2%}")
                if res['color_score'] is not None:
                    print(f"   - Color (HSV)       : {res['color_score']:.2%}")
                print(f"   - SSIM Structural   : {res['ssim_score']:.2%}")
                print(f"   - Style (Gram)      : {res['style_score']:.2%}")
                print(f"   - Edge Density      : {res['edge_score']:.2%}")
                print(f"   - Aesthetic Quality : {res['aesthetic_score']:.2f}/1.00")
                print(f"   - Status            : {'✓ Consistent' if res['consistent'] else '⚠ Less Consistent'}")
                print("-" * 50)
        except Exception as e:
            print(f"Warning: Could not perform consistency check: {e}")
    elif not USING_ENRICHED_MODE and os.path.exists(ref_path):
        print("\n🔍 Running character consistency checks across panels...")
        try:
            checker = get_consistency_checker()
            checker.set_reference(ref_path)
            for idx, panel_path in enumerate(generated_paths):
                res = checker.check_consistency(panel_path)
                basename = os.path.basename(panel_path)
                print("-" * 50)
                print(f"Panel {idx+1} Character Consistency ({basename}):")
                print(f"   - Overall Score: {res['score']:.2%}")
                print(f"   - Status: {'✓ Consistent' if res['consistent'] else '⚠ Less Consistent'}")
                print("-" * 50)
        except Exception as e:
            print(f"Warning: Could not perform consistency check: {e}")
    else:
        if USING_ENRICHED_MODE:
            print("\n[i] Only 1 panel generated — skipping consistency check (need ≥2 panels).")
        else:
            print("\n[i] No character reference image found — skipping consistency check.")

# Final memory report
if torch.cuda.is_available():
    allocated, reserved = get_gpu_memory_usage()
    print(f"\n💾 Final GPU Memory: {allocated:.0f}MB allocated, {reserved:.0f}MB reserved")
    print(f"🎯 Peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f}MB")

print("\n" + "=" * 70)
print("✅ PAGE PANEL GENERATION COMPLETE!")
print("=" * 70)