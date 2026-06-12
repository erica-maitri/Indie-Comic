"""
SDXL BASE EMOTION-AWARE PANEL GENERATOR - T4 OPTIMIZED
Generates individual comic panels for a selected storyboard page using active character expressions
Optimized for T4 GPU with memory management and model caching (No LoRA)
"""

import json
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler, AutoencoderKL
from PIL import Image
import os
import sys
import argparse
import gc

print("=" * 70)
print("SDXL BASE EMOTION-AWARE PANEL GENERATOR - T4 OPTIMIZED")
print("Drawing the story with efficient memory management (Base SDXL)")
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
        print("  [✓] Using cached pipeline (saved ~10-15 seconds)")
        return _PIPE_CACHE
    
    print("\n  Loading SDXL Base pipeline (first time, caching for future pages)...")
    
    sdxl_settings = settings.get("models", {}).get("sdxl", {})
    
    model_name = sdxl_settings.get("name", "stabilityai/stable-diffusion-xl-base-1.0")
    variant = sdxl_settings.get("variant", "fp16")
    
    # Load VAE with fp16 for memory efficiency
    try:
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch.float16
        )
        print("  [✓] Loaded FP16 VAE (reduces VRAM usage)")
    except:
        vae = None
        print("  [i] Using default VAE")
    
    # Load pipeline with memory optimizations
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name,
        vae=vae,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
        variant=variant if device == "cuda" else None,
        add_watermarker=False  # Disable watermark to save VRAM
    )
    
    # Use DPM++ scheduler for faster inference (no LoRA needed)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, 
        use_karras_sigmas=True,
        algorithm_type="sde-dpmsolver++",  # Faster convergence
        solver_order=2
    )
    
    # Apply T4-specific memory optimizations
    t4_opts = settings.get("t4_optimizations", {})
    
    if device == "cuda":
        if t4_opts.get("cpu_offload", True):
            try:
                pipe.enable_model_cpu_offload()
                print("  [✓] CPU offload enabled (saves ~4GB VRAM)")
            except Exception as e:
                print(f"  [⚠] CPU offload failed: {e}")
                pipe = pipe.to(device)
        
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
    
    print("  [✓] Base SDXL pipeline ready and cached")
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
sdxl_settings = settings.get("models", {}).get("sdxl", {})
device = sdxl_settings.get("device", "cuda")

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
ref_path = get_output_path(char_dir, "character_reference.png")

use_ip_adapter = False
ip_adapter_enabled = settings.get("t4_optimizations", {}).get("disable_ipadapter", True) == False

if USING_ENRICHED_MODE:
    print("[+] Enriched mode: skipping IP-Adapter (no reference image needed).")
elif not ip_adapter_enabled:
    print("[i] IP-Adapter disabled in T4 optimizations (saves VRAM).")
elif os.path.exists(ref_path):
    print(f"Loading IP-Adapter for character consistency...")
    try:
        ip_settings = settings.get("models", {}).get("ipadapter", {})
        ip_weight = ip_settings.get("weight", 0.8)
        
        if ip_adapter_enabled:
            pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
            pipe.set_ip_adapter_scale(ip_weight)
            print("IP-Adapter loaded successfully!")
            use_ip_adapter = True
        else:
            print("IP-Adapter disabled for T4 optimization")
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
    
    # Get prompt - base SDXL doesn't use LoRA trigger words
    augmented = panel.get('augmented_prompt') or panel.get('visual', 'comic panel scene')
    prompt_str = augmented  # No trigger words for base SDXL
    
    print(f"  Prompt: {prompt_str[:100]}...")
    
    raw_negative = "extra fingers, deformed face, bad anatomy, character design changes, photorealistic, 3d render"
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
        
        panel_path = get_output_path(comics_dir, f"page_{args.page}_panel_sdxl_base_{p_num}.png")
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
        fallback_res = t4_opts.get("fallback", {}).get("resolution_fallback", [512, 512])
        fallback_steps = t4_opts.get("fallback", {}).get("steps_fallback", 20)
        
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
            
            panel_path = get_output_path(comics_dir, f"page_{args.page}_panel_sdxl_base_{p_num}_fallback.png")
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
    strip_path = get_output_path(comics_dir, f"page_{args.page}_layout_sdxl_base_horizontal.png")
    create_comic_strip(generated_paths, strip_path, orientation='horizontal')
    print(f"  ✓ Horizontal strip: {strip_path}")
    
    grid_path = get_output_path(comics_dir, f"page_{args.page}_layout_sdxl_base_grid.png")
    num_panels = len(generated_paths)
    grid_size = (2, 2) if num_panels == 4 else (1, num_panels)
    create_comic_grid(generated_paths, grid_path, grid_size=grid_size, cell_size=(width, height))
    print(f"  ✓ Grid layout: {grid_path}")
else:
    print("Error: No panels were successfully generated.")
    sys.exit(1)

# Run Character Consistency Evaluation (lightweight only)
char_dir = settings.get("outputs", {}).get("character_dir", "outputs/characters")
ref_path = get_output_path(char_dir, "character_reference.png")

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