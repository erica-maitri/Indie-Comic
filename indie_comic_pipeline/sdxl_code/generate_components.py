"""
SDXL COMIC COMPONENT GENERATOR
Generates individual visual assets/components (main character pose, extra person, environment background, props)
"""

import json

import torch

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DPMSolverMultistepScheduler

from PIL import Image

import os

import sys

print("=" * 70)

print("SDXL COMIC COMPONENT GENERATOR - Generating assets")

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

from utils.image_utils import create_comic_strip, create_comic_grid

from utils.consistency_checker import get_consistency_checker

settings = load_settings()

fusion_dir = settings.get("outputs", {}).get("fusion_dir", "outputs/fusion")

fusion_path = get_output_path(fusion_dir, "fusion_complete.json")

with open(fusion_path, "r") as f:

    fusion_data = json.load(f)

personality = fusion_data['personality']

setting = fusion_data['setting']

fusion = fusion_data['fusion']

print(f"\nGenerating components for: {personality['character_name']} in {setting['story_name']}")

print(f"   Personality: {', '.join(personality['core_personality_traits'][:2])}")

print(f"   Mood: {setting['mood']}")

                                                     

components = fusion.get("components", [])

if not components:

    print("Error: Components missing in fusion data. Exiting.")

    sys.exit(1)

print(f"\nLoaded {len(components)} components from LLM design")

                                                                                    

sdxl_settings = settings.get("models", {}).get("sdxl", {})

model_name = sdxl_settings.get("name", "stabilityai/stable-diffusion-xl-base-1.0")

variant = sdxl_settings.get("variant", "fp16")

device = sdxl_settings.get("device", "cuda")

if device == "cuda" and not torch.cuda.is_available():

    print("Warning: CUDA is configured but not available. Falling back to CPU.")

    device = "cpu"

print(f"\nUsing device: {device}")
if device == "cuda":
    torch.cuda.reset_peak_memory_stats()

                                                                                             

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

        pipe.enable_attention_slicing()

        pipe.enable_vae_slicing()

        print("GPU memory optimization enabled")

    

    print("Model loaded successfully")
    
    # Define paths for IP-Adapter loading
    char_dir = settings.get("outputs", {}).get("character_dir", "outputs/characters")
    ref_path = get_output_path(char_dir, "character_reference.png")
    
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

comics_dir = settings.get("outputs", {}).get("comics_dir", "outputs/comics")

char_dir = settings.get("outputs", {}).get("character_dir", "outputs/characters")

                                                               

optimizer = get_prompt_optimizer()

                                                            

print("\nGenerating comic components...")

print("-" * 50)

generated_images = []

component_paths = []

gen_settings = settings.get("generation", {})

width = gen_settings.get("default_size", {}).get("width", 1024)

height = gen_settings.get("default_size", {}).get("height", 1024)

steps = gen_settings.get("inference_steps", 40)

guidance = gen_settings.get("guidance_scale", 7.5)

seed = gen_settings.get("seed", 42)

for i, component in enumerate(components):

    c_type = component.get('component_type') or component.get('type') or 'unknown'

    print(f"\nComponent {i+1} ({c_type}): {component.get('name', 'No name')}")

    print(f"   Description: {component.get('description', 'No description')[:100]}...")

    

                                                                                              

    raw_prompt = component.get("sdxl_prompt", "")

    optimized_prompt = optimizer.optimize_positive_prompt(raw_prompt)

    

                                                                                

    if i == 0:

        optimized_prompt = optimizer.add_consistency_constraints(optimized_prompt, personality['character_name'])

    

                                                                                      

    raw_negative = """
    extra fingers, deformed face, bad anatomy, character design changes
    """

    optimized_negative = optimizer.optimize_negative_prompt(raw_negative)

    

    print(f"   Generating...")

    

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
        
                                   

        component_path = get_output_path(comics_dir, f"component_{i+1}.png")

        image.save(component_path)

        generated_images.append(image)

        component_paths.append(component_path)

        print(f"   Saved component to: {component_path}")

        

    except Exception as e:

        print(f"   Error: Component generation failed: {e}")

                                                                   

if component_paths:

    print("\nCreating component sheet strip layout...")

    strip_path = get_output_path(comics_dir, "component_sheet_horizontal.png")

    create_comic_strip(component_paths, strip_path, orientation='horizontal')

    print(f"Saved component sheet strip to: {strip_path}")

    

                                     

    num_assets = len(component_paths)

    if num_assets <= 3:

        grid_size = (1, num_assets)

    elif num_assets == 4:

        grid_size = (2, 2)

    else:

                           

        import math

        cols = int(math.ceil(num_assets / 2.0))

        grid_size = (2, cols)

        

    grid_path = get_output_path(comics_dir, "component_sheet_grid_2x2.png")                                 

    create_comic_grid(component_paths, grid_path, grid_size=grid_size, cell_size=(width, height))

    print(f"Saved dynamic component sheet grid to: {grid_path} (Grid size: {grid_size[0]}x{grid_size[1]})")
else:
    print("Error: No components were successfully generated.")
    sys.exit(1)

                                                                                              

ref_path = get_output_path(char_dir, "character_reference.png")

if os.path.exists(ref_path) and component_paths:
    print("\nRunning character consistency checks...")
    try:
        checker = get_consistency_checker()
        checker.set_reference(ref_path)
        if len(component_paths) > 0:
            main_char_path = component_paths[0]
            res = checker.check_consistency(main_char_path)
            basename = os.path.basename(main_char_path)
            print("-" * 50)
            print(f"Main Character Pose Consistency ({basename}):")
            print(f"   - Overall Similarity Score: {res['score']:.2%}")
            print(f"   - Color Consistency (HSV): {res['color_score']:.2%}")
            print(f"   - SSIM Structural Similarity: {res['ssim_score']:.2%}")
            print(f"   - Style Similarity (Gram Matrix): {res['style_score']:.2%}")
            print(f"   - Edge Density Similarity: {res['edge_score']:.2%}")
            if res.get('clip_img_score') is not None:
                print(f"   - CLIP Image Semantic Similarity: {res['clip_img_score']:.2%}")
            print(f"   - Aesthetic Quality Score: {res['aesthetic_score']:.2f}/1.00")
            print(f"   - Thumbnail Correlation (Legacy): {res['struct_score']:.2%}")
            print(f"   - Status: {'Consistent' if res['consistent'] else 'Less Consistent'}")
            print("-" * 50)
        else:
            print("Skipping consistency checks (no component generated).")
    except Exception as e:
        print(f"Warning: Could not perform consistency check: {e}")

# Compute Peak VRAM usage
peak_vram = None
if device == "cuda":
    try:
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
    except:
        pass

print("\n" + "=" * 70)
print("COMPONENT GENERATION COMPLETE!")
if peak_vram is not None:
    print(f"Peak VRAM Usage: {peak_vram:.2f} MB")
print("=" * 70)

print(f"\nOutput files in {comics_dir}/")
print("   - component_1.png through component_N.png")
print("   - component_sheet_horizontal.png")
print("   - component_sheet_grid_2x2.png (dynamic grid layout)")
print("=" * 70)

