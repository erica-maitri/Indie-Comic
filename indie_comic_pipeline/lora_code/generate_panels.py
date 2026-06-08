"""
SDXL + LORA EMOTION-AWARE PANEL GENERATOR
Generates individual comic panels for a selected storyboard page using active character expressions
"""

import json
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
import sys
import argparse

print("=" * 70)
print("SDXL + LORA EMOTION-AWARE PANEL GENERATOR - Drawing the story")
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
from utils.consistency_checker import get_consistency_checker
from utils.image_utils import create_comic_strip, create_comic_grid

# Parse command line page argument
parser = argparse.ArgumentParser(description="Generate comic panels for a storyboard page.")
parser.add_argument("--page", type=int, default=1, help="The storyboard page number to generate (1-10).")
args = parser.parse_args()

settings = load_settings()
fusion_dir = settings.get("outputs", {}).get("fusion_dir", "outputs/fusion")
emotions_path = get_output_path(fusion_dir, "storyboard_with_emotions.json")

if not os.path.exists(emotions_path):
    print(f"Error: Storyboard with emotions not found at: {emotions_path}")
    print("   Please run the emotion recognition engine first:")
    print("   python langchain_code/emotion_recognition_engine.py")
    sys.exit(1)

with open(emotions_path, "r", encoding="utf-8") as f:
    storyboard_data = json.load(f)

story_pages = storyboard_data.get("storyboard_with_emotions", [])
target_page = None
for page in story_pages:
    if page.get("page_number") == args.page:
        target_page = page
        break

if not target_page:
    print(f"Error: Page {args.page} not found in storyboard.")
    sys.exit(1)

personality = storyboard_data['personality']
setting = storyboard_data['setting']
panels = target_page.get("panels_detail", [])

print(f"\nGenerating {len(panels)} panels for Page {args.page}: {target_page.get('location')}")

sdxl_settings = settings.get("models", {}).get("sdxl", {})
lora_settings = settings.get("models", {}).get("lora", {})

model_name = sdxl_settings.get("name", "stabilityai/stable-diffusion-xl-base-1.0")
variant = sdxl_settings.get("variant", "fp16")
device = sdxl_settings.get("device", "cuda")

if device == "cuda" and not torch.cuda.is_available():
    print("Warning: CUDA is configured but not available. Falling back to CPU.")
    device = "cpu"

print(f"\nUsing device: {device}")
print(f"\nLoading SDXL base model '{model_name}'...")

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
    
    if device == "cuda" and sdxl_settings.get("memory_optimization", True):
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        print("GPU memory optimization enabled")
        
    print("Model loaded successfully")
    
    # Define paths for IP-Adapter loading
    char_dir = settings.get("outputs", {}).get("character_dir", "outputs/characters")
    ref_path = get_output_path(char_dir, "character_reference_sdxl_lora.png")
    
    use_ip_adapter = False
    if os.path.exists(ref_path):
        print(f"Loading IP-Adapter for character consistency...")
        try:
            ip_settings = settings.get("models", {}).get("ipadapter", {})
            ip_model = ip_settings.get("model", "ip-adapter-faceid-plusv2_sdxl")
            ip_weight = ip_settings.get("weight", 0.8)
            
            # Load IP-Adapter weights
            if "faceid" in ip_model.lower():
                try:
                    pipe.load_ip_adapter("h94/IP-Adapter-FaceID", subfolder="", weight_name="ip-adapter-faceid-plusv2_sdxl.bin")
                except Exception as fe:
                    print(f"FaceID IP-Adapter failed to load ({fe}). Falling back to standard IP-Adapter...")
                    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
            else:
                pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
                
            pipe.set_ip_adapter_scale(ip_weight)
            print("IP-Adapter loaded successfully!")
            use_ip_adapter = True
        except Exception as e:
            print(f"Warning: Failed to load IP-Adapter: {e}. Proceeding without IP-Adapter.")
    else:
        print("Reference character image not found. Proceeding without IP-Adapter.")
    
except Exception as e:
    print(f"Error: Failed to load model: {e}")
    sys.exit(1)

if device == "cuda":
    torch.cuda.reset_peak_memory_stats()

comics_dir = settings.get("outputs", {}).get("comics_dir", "outputs/comics")
char_dir = settings.get("outputs", {}).get("character_dir", "outputs/characters")

print("\nGenerating panels...")
print("-" * 50)

generated_paths = []
gen_settings = settings.get("generation", {})
width = gen_settings.get("default_size", {}).get("width", 1024)
height = gen_settings.get("default_size", {}).get("height", 1024)
steps = gen_settings.get("inference_steps", 40)
guidance = gen_settings.get("guidance_scale", 7.5)
seed = gen_settings.get("seed", 42)

for i, panel in enumerate(panels):
    p_num = panel.get("panel_number") or (i + 1)
    print(f"\n--- Panel {p_num} ---")
    print(f"  Visual Prompt: {panel.get('augmented_prompt')[:100]}...")
    
    # Inject LoRA trigger words
    trigger_words = lora_settings.get("trigger_words", "LineAniAF, lineart")
    prompt_str = f"{panel.get('augmented_prompt')}, {trigger_words}"
    
    raw_negative = "extra fingers, deformed face, bad anatomy, character design changes"
    negative_str = raw_negative # Add optimizer logic if required
    
    generator = torch.Generator(device=device).manual_seed(seed + args.page * 10 + p_num)
    
    try:
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
        
        panel_path = get_output_path(comics_dir, f"page_{args.page}_panel_sdxl_lora_{p_num}.png")
        image.save(panel_path)
        generated_paths.append(panel_path)
        print(f"  Saved panel image to: {panel_path}")
        
    except Exception as e:
        print(f"  Error generating panel {p_num}: {e}")

# Compile panels into page layouts
if generated_paths:
    print("\nCompiling page layout strips...")
    strip_path = get_output_path(comics_dir, f"page_{args.page}_layout_sdxl_lora_horizontal.png")
    create_comic_strip(generated_paths, strip_path, orientation='horizontal')
    print(f"Saved page horizontal strip to: {strip_path}")
    
    grid_path = get_output_path(comics_dir, f"page_{args.page}_layout_sdxl_lora_grid.png")
    num_panels = len(generated_paths)
    grid_size = (2, 2) if num_panels == 4 else (1, num_panels)
    create_comic_grid(generated_paths, grid_path, grid_size=grid_size, cell_size=(width, height))
    print(f"Saved page dynamic grid layout to: {grid_path}")
else:
    print("Error: No panels were successfully generated.")
    sys.exit(1)

# Run Character Consistency Evaluation against reference profile
ref_path = get_output_path(char_dir, "character_reference_sdxl_lora.png")
if os.path.exists(ref_path) and generated_paths:
    print("\nRunning character consistency checks across panels...")
    try:
        checker = get_consistency_checker()
        checker.set_reference(ref_path)
        for idx, panel_path in enumerate(generated_paths):
            res = checker.check_consistency(panel_path)
            basename = os.path.basename(panel_path)
            print("-" * 50)
            print(f"Panel {idx+1} Character Consistency ({basename}):")
            print(f"   - Overall Similarity Score: {res['score']:.2%}")
            print(f"   - Color Consistency (HSV): {res['color_score']:.2%}")
            print(f"   - SSIM Structural Similarity: {res['ssim_score']:.2%}")
            print(f"   - Style Similarity (Gram Matrix): {res['style_score']:.2%}")
            print(f"   - Edge Density Similarity: {res['edge_score']:.2%}")
            if res.get('clip_img_score') is not None:
                print(f"   - CLIP Image Semantic Similarity: {res['clip_img_score']:.2%}")
            if res.get('dinov2_score') is not None:
                print(f"   - DINOv2 Feature Similarity: {res['dinov2_score']:.2%}")
            print(f"   - Aesthetic Quality Score: {res['aesthetic_score']:.2f}/1.00")
            print(f"   - Thumbnail Correlation (Legacy): {res['struct_score']:.2%}")
            print(f"   - Status: {'Consistent' if res['consistent'] else 'Less Consistent'}")
            print("-" * 50)
    except Exception as e:
        print(f"Warning: Could not perform consistency check: {e}")

peak_vram = None
if device == "cuda":
    try:
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
    except:
        pass

print("\n" + "=" * 70)
print("PAGE PANEL GENERATION COMPLETE!")
if peak_vram is not None:
    print(f"Peak VRAM Usage: {peak_vram:.2f} MB")
print("=" * 70)
