import os

target_folder = "matrix_evaluation_zone"
os.makedirs(target_folder, exist_ok=True)

test_file_path = os.path.join(target_folder, "storyboard_speed_bench.py")

perfect_script_code = """# Fully Automated High-Speed Multi-Panel SDXL Engine
import os
import torch
import time
import gc
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# Hardware layer matrix acceleration 
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = "matrix_evaluation_zone/outputs/storyboard_run"
os.makedirs(output_dir, exist_ok=True)

# Complete 8-Panel Input Framework
storyboard_data = {
  "panels": [
    {"panel": 1, "visual": "A man stands outside at night under a full moon, holding his hands outstretched towards an imaginary flame burning brighter than the stars.", "emotion_beat": "contained_fire"},
    {"panel": 2, "visual": "The flames flicker around him like he's trying to extinguish them while feeling an overwhelming urge to keep them bright and fierce.", "emotion_beat": "contained_fire"},
    {"panel": 3, "visual": "He suddenly looks down at his clenched fists and realizes there’s steam rising from them - evidence of unspent energy.", "emotion_beat": "fracture"},
    {"panel": 4, "visual": "In front of him, a gentle breeze starts rustling the leaves of some nearby trees. The wind seems to whisper soothing phrases into his ears.", "emotion_beat": "exhale"},
    {"panel": 5, "visual": "As he takes deep breaths, beads of sweat start forming on his forehead due to exertion without being able to fully release the pent-up energy within him.", "emotion_beat": "exhale"},
    {"panel": 6, "visual": "Suddenly, a sudden drop in temperature sends shockwaves of relief throughout the air; it feels as if someone has turned off all the lights except one.", "emotion_beat": "cooling"},
    {"panel": 7, "visual": "His body language shifts to one where he’s more relaxed—body leaning forward against himself as though embracing a protective shield.", "emotion_beat": "grounded"},
    {"panel": 8, "visual": "Outside, the sun comes up, casting a warm glow across the landscape, symbolizing hope, light overcoming darkness.", "emotion_beat": "stillness"}
  ]
}

style_positive = "clean minimalist line art, flat color palette, indie comic book style, crisp outlines, cel-shaded, "
style_negative = "photorealistic, 3d render, shading, gradients, blurry, messy lines, realistic shadows, photoreal"

print(" Loading SDXL Pipeline Architecture into VRAM context...")
torch_dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch_dtype,
    use_safetensors=True,
    low_cpu_mem_usage=True
)

if device == "cuda":
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload() # Safely manages T4 VRAM allocations

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

print("\\n VRAM Shield Active! Starting loop sequence for all 8 panels...\\n")

for p in storyboard_data["panels"]:
    p_num = p["panel"]
    visual_desc = p["visual"]
    beat = p["emotion_beat"]
    
    print(f" RENDERING PANEL {p_num} OF 8 ---> Mood Target: [ {beat} ]")
    final_prompt = f"{style_positive}{visual_desc}, thematic mood is {beat}"
    
    # Keeping generator state isolated on CPU to secure dynamic layer seed anchors
    generator = torch.Generator(device="cpu").manual_seed(500 + p_num)
    
    start_time = time.time()
    
    # 20 Steps Inference block
    image = pipe(
        prompt=final_prompt,
        negative_prompt=style_negative,
        height=1024,
        width=1024,
        num_inference_steps=20, 
        guidance_scale=7.0,
        generator=generator
    ).images[0]
    
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    
    save_path = os.path.join(output_dir, f"panel_{p_num}.png")
    image.save(save_path)
    print(f" Panel {p_num} successfully generated and cached in {duration}s!")
    print(f" Saved location: {save_path}")
    
    # CRITICAL MEMORY FLUSH: Force clears VRAM leaks immediately after saving each image
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    print("-" * 60)

print("\\n SUCCESS! All 8 sequential comic panels have been cleanly rendered without freezing!")
"""

with open(test_file_path, "w", encoding="utf-8") as f:
    f.write(perfect_script_code)

print(f" Code cleanly updated to fix the loop inside: {test_file_path}")

# Executing directly by wrapping the live runtime working path context
import subprocess
import sys
subprocess.run([sys.executable, test_file_path])
