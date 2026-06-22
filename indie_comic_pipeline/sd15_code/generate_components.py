"""
SD 1.5 COMIC COMPONENT GENERATOR - T4 OPTIMIZED
Generates individual visual assets/components (main character pose, extra person, environment background, props)
Optimized for T4 GPU with memory management and model caching (SD 1.5 - Fast Mode)
"""

import json
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
import sys
import gc

print("=" * 70)
print("SD 1.5 COMIC COMPONENT GENERATOR - T4 OPTIMIZED")
print("Generating assets with efficient memory management (SD 1.5 - Fast Mode)")
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
from utils.image_utils import create_comic_strip, create_comic_grid
from utils.consistency_checker import get_consistency_checker

# Global model cache
_PIPE_CACHE = None

def get_pipeline(settings, device, force_reload=False):
    """Get cached pipeline instance to avoid reloading"""
    global _PIPE_CACHE
    
    if _PIPE_CACHE is not None and not force_reload:
        print("  [✓] Using cached pipeline (saved ~5-8 seconds)")
        return _PIPE_CACHE
    
    print("\n  Loading SD 1.5 pipeline (first time, caching for future use)...")
    
    sd15_settings = settings.get("models", {}).get("sd15", {})
    lora_settings = settings.get("models", {}).get("lora", {})
    
    model_name = sd15_settings.get("name", "runwayml/stable-diffusion-v1-5")
    
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
        print(f"  [✓] LoRA loaded with scale {adapter_scale}")
    except Exception as e:
        print(f"  [i] LoRA not loaded (expected for SD 1.5): {str(e)[:50]}...")
    
    # Use DPM++ scheduler for faster inference
    scheduler_config = dict(pipe.scheduler.config)
    scheduler_config.pop("_class_name", None)
    scheduler_config.pop("algorithm_type", None)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        scheduler_config, 
        use_karras_sigmas=True,
        algorithm_type="sde-dpmsolver++",
        solver_order=2
    )
    
    # Apply T4-specific memory optimizations
    t4_opts = settings.get("t4_optimizations", {})
    
    if device == "cuda":
        if t4_opts.get("attention_slicing", True):
            try:
                pipe.enable_attention_slicing("max")
                print("  [✓] Max attention slicing enabled")
            except Exception as e:
                print(f"  [⚠] Attention slicing failed: {e}")
        
        if t4_opts.get("vae_slicing", True):
            try:
                pipe.enable_vae_slicing()
                print("  [✓] VAE slicing enabled")
            except Exception as e:
                print(f"  [⚠] VAE slicing failed: {e}")
        
        # SD 1.5 is lighter, fit fully on GPU
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
fusion_path = get_output_path(fusion_dir, "fusion_complete.json")

# Check if we're in enriched mode (Story-Weaver) or legacy mode
enriched_path = get_output_path(fusion_dir, "enriched_storyboard.json")
USING_ENRICHED_MODE = os.path.exists(enriched_path)

if USING_ENRICHED_MODE:
    print("\n[MODE] Story-Weaver Enriched Mode")
    print("[i] Components will be generated from enriched storyboard data")
    with open(enriched_path, "r", encoding="utf-8") as f:
        enriched_data = json.load(f)
    
    # Extract components from enriched data
    character_name = enriched_data.get("character_name", "Wanderer")
    story_world = enriched_data.get("story_world", "The Abstract")
    
    # Create component definitions from enriched panels
    panels = enriched_data.get("pages", [{}])[0].get("panels_detail", [])
    components = []
    
    if panels:
        # Component 1: Main character pose
        main_char = panels[0].get("main_character", {})
        components.append({
            "name": f"{character_name} Main Pose",
            "type": "character",
            "description": main_char.get("description", f"{character_name} in a dramatic pose"),
            "sdxl_prompt": f"indie comic style illustration, clean minimalist line art, flat color palette, {character_name}, {main_char.get('description', '')}, {main_char.get('action', '')}"
        })
        
        # Component 2: Side character
        side_chars = panels[0].get("side_characters", [])
        if side_chars:
            side = side_chars[0]
            components.append({
                "name": side.get("name", "Side Character"),
                "type": "secondary_character",
                "description": side.get("description", ""),
                "sdxl_prompt": f"indie comic style illustration, clean minimalist line art, flat color palette, {side.get('name', 'character')}, {side.get('description', '')}"
            })
        
        # Component 3: Environment
        components.append({
            "name": f"{story_world} Environment",
            "type": "environment",
            "description": panels[0].get("scenery", f"A scene in {story_world}"),
            "sdxl_prompt": f"indie comic style illustration, clean minimalist line art, flat color palette, {panels[0].get('scenery', story_world)}"
        })
        
        # Component 4: Recurring motif
        motif = enriched_data.get("recurring_motif", "A symbolic object")
        components.append({
            "name": "Recurring Motif",
            "type": "prop",
            "description": motif,
            "sdxl_prompt": f"indie comic style illustration, clean minimalist line art, flat color palette, {motif}, detailed closeup"
        })
    
    print(f"\n[+] Created {len(components)} components from enriched storyboard")
else:
    print("\n[MODE] Legacy Mode — using fusion_complete.json")
    if not os.path.exists(fusion_path):
        print(f"Error: Fusion data not found at: {fusion_path}")
        print("   Please run the LangChain pipeline first:")
        print("   python langchain_code/run_full_pipeline.py")
        sys.exit(1)
    
    with open(fusion_path, "r", encoding="utf-8") as f:
        fusion_data = json.load(f)

    personality = fusion_data.get('personality', {})
    setting = fusion_data.get('setting', {})
    fusion = fusion_data.get('fusion', {})
    
    character_name = personality.get('character_name', 'Unknown')
    story_world = setting.get('story_name', 'Unknown')
    components = fusion.get("components", [])

if not components:
    print("Error: No components found or could not create components.")
    sys.exit(1)

print(f"\n📦 Generating {len(components)} components for: {character_name} in {story_world}")

# Get optimized generation settings (SD 1.5 native resolution)
gen_settings = settings.get("generation", {})
t4_opts = settings.get("t4_optimizations", {})

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

# Check GPU memory before starting
if torch.cuda.is_available():
    allocated, reserved = get_gpu_memory_usage()
    print(f"\n💾 GPU Memory before loading: {allocated:.0f}MB allocated, {reserved:.0f}MB reserved")

# Load pipeline (cached)
sd15_settings = settings.get("models", {}).get("sd15", {})
device = sd15_settings.get("device", "cuda")

if device == "cuda" and not torch.cuda.is_available():
    print("Warning: CUDA is configured but not available. Falling back to CPU.")
    device = "cpu"

pipe = get_pipeline(settings, device)

# IP-Adapter for SD 1.5
char_dir = settings.get("outputs", {}).get("character_dir", "outputs/characters")
ref_path = get_output_path(char_dir, "character_reference_sd15.png")

use_ip_adapter = False
ip_adapter_enabled = t4_opts.get("disable_ipadapter", True) == False

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
        print(f"Warning: Failed to load IP-Adapter: {e}")
else:
    print("Reference character image not found. Proceeding without IP-Adapter.")

comics_dir = settings.get("outputs", {}).get("comics_dir", "outputs/comics")
optimizer = get_prompt_optimizer()

print("\n🎨 Generating components...")
print("-" * 50)

generated_images = []
component_paths = []

for i, component in enumerate(components):
    c_type = component.get('component_type') or component.get('type') or 'unknown'
    c_name = component.get('name', 'No name')
    print(f"\n--- Component {i+1}: {c_name} ({c_type}) ---")
    
    raw_prompt = component.get("sdxl_prompt", "")
    optimized_prompt = optimizer.optimize_positive_prompt(raw_prompt)
    
    if i == 0 and not USING_ENRICHED_MODE:
        optimized_prompt = optimizer.add_consistency_constraints(optimized_prompt, character_name)
    
    # Append trigger words for LoRA (if available)
    lora_settings = settings.get("models", {}).get("lora", {})
    trigger_words = lora_settings.get("trigger_words", "LineAniAF, lineart")
    optimized_prompt = f"{optimized_prompt}, {trigger_words}"
    
    raw_negative = "extra fingers, deformed face, bad anatomy, character design changes, photorealistic, 3d render"
    optimized_negative = optimizer.optimize_negative_prompt(raw_negative)
    
    print(f"  Prompt: {optimized_prompt[:100]}...")
    
    generator = torch.Generator(device=device).manual_seed(seed + i)
    
    try:
        if i == 0 and use_ip_adapter and os.path.exists(ref_path):
            ref_image = Image.open(ref_path).convert("RGB")
            image = pipe(
                prompt=optimized_prompt,
                negative_prompt=optimized_negative,
                ip_adapter_image=ref_image,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator
            ).images[0]
        else:
            image = pipe(
                prompt=optimized_prompt,
                negative_prompt=optimized_negative,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator
            ).images[0]
        
        component_path = get_output_path(comics_dir, f"component_sd15_{i+1}.png")
        image.save(component_path)
        generated_images.append(image)
        component_paths.append(component_path)
        print(f"  ✓ Saved to: {component_path}")
        
        # Clear memory periodically
        clear_interval = t4_opts.get("clear_cache_every_n_steps", 5)
        if (i + 1) % clear_interval == 0:
            print(f"  🧹 Clearing GPU cache...")
            clear_gpu_memory()
        
    except Exception as e:
        print(f"  ❌ Error generating component: {e}")

clear_gpu_memory()

# Compile component sheet layouts
if component_paths:
    print("\n📐 Creating component sheet layouts...")
    
    strip_path = get_output_path(comics_dir, "component_sheet_sd15_horizontal.png")
    create_comic_strip(component_paths, strip_path, orientation='horizontal')
    print(f"  ✓ Horizontal strip: {strip_path}")
    
    grid_path = get_output_path(comics_dir, "component_sheet_sd15_grid.png")
    num_assets = len(component_paths)
    grid_size = (2, 2) if num_assets >= 4 else (1, num_assets)
    create_comic_grid(component_paths, grid_path, grid_size=grid_size, cell_size=(width, height))
    print(f"  ✓ Grid layout: {grid_path}")

print("\n" + "=" * 70)
print("✅ COMPONENT GENERATION COMPLETE!")
print("=" * 70)