import os
import sys
from PIL import Image, ImageDraw

# Add pipeline directory to import path
PROJECT_ROOT = r"c:\Users\ihsko\Documents\Indie-Comic\indie_comic_pipeline"
sys.path.append(PROJECT_ROOT)

from core.text_image_integrator import TextImageIntegrator

def run_tests():
    print("Initializing TextImageIntegrator...")
    integrator = TextImageIntegrator(
        output_dir="outputs/test_render",
        ollama_model="mock"  # Skip Ollama queries for the mock test
    )
    
    # Create output test render directory
    os.makedirs("outputs/test_render", exist_ok=True)
    
    # Standard bubble test cases
    test_cases = [
        {
            "name": "01_ellipse_center",
            "dialogue": "WANDERER: I **cannot** believe this is *actually* working!",
            "beat": "neutral",
            "shape": "ellipse",
            "align": "center",
            "tail_x": 0.3, "tail_y": 0.8,
            "desc": "Standard ellipse bubble with bold/italic text, center aligned, pointing to speaker at (0.3, 0.8)"
        },
        {
            "name": "02_dashed_left",
            "dialogue": "NARRATOR: *shh...* did you hear that? I think __something__ is coming.",
            "beat": "whisper",
            "shape": "dashed_ellipse",
            "align": "left",
            "tail_x": 0.7, "tail_y": 0.7,
            "desc": "Dashed whisper bubble with italic and bold text, left aligned, pointing to (0.7, 0.7)"
        },
        {
            "name": "03_cloud_thought",
            "dialogue": "HERO: (He is staring at me...)\nWhat is **he** planning to do now?",
            "beat": "thought",
            "shape": "cloud",
            "align": "center",
            "tail_x": 0.2, "tail_y": 0.6,
            "desc": "Thought cloud bubble with explicit newline paragraph double spacing, pointing to (0.2, 0.6)"
        },
        {
            "name": "04_spiky_shout",
            "dialogue": "ENEMY: **STOP RIGHT THERE!**\nYou won't get away *this time*!",
            "beat": "shout",
            "shape": "spiky",
            "align": "center",
            "tail_x": 0.5, "tail_y": 0.9,
            "desc": "Spiky action bubble with bold shout, centered, pointing to center-bottom (0.5, 0.9)"
        },
        {
            "name": "05_jagged_intense",
            "dialogue": "CHARACTER: I feel like everything is **falling apart**...\nWhy can't I *breathe*?",
            "beat": "intense",
            "shape": "jagged",
            "align": "center",
            "tail_x": 0.4, "tail_y": 0.85,
            "desc": "Jagged tense/stressed bubble with bold/italic text, pointing to (0.4, 0.85)"
        }
    ]
    
    for tc in test_cases:
        print(f"\n--- Running Test: {tc['name']} ---")
        
        # Create base canvas (512x512 light gray panel background)
        base_img = Image.new("RGBA", (512, 512), (230, 230, 235, 255))
        draw = ImageDraw.Draw(base_img)
        
        # Draw a placeholder character at the tail target to visually check pointer alignment
        tx = int(tc["tail_x"] * 512)
        ty = int(tc["tail_y"] * 512)
        # Draw head/body (blue circle/rectangle)
        draw.ellipse([tx - 15, ty - 15, tx + 15, ty + 15], fill=(30, 144, 255, 255), outline=(0, 0, 139, 255), width=2)
        draw.rectangle([tx - 8, ty + 15, tx + 8, ty + 50], fill=(30, 144, 255, 255), outline=(0, 0, 139, 255), width=2)
        
        # Convert base canvas to RGB for integrator
        base_img = base_img.convert("RGB")
        
        # Mock the planned coordinates inside get_layout_plan using a custom dict override
        # We manually patch get_layout_plan to return our controlled test settings
        speaker, text_clean = integrator._parse_dialogue(tc["dialogue"])
        if not text_clean:
            text_clean = tc["dialogue"]
            
        test_plan = {
            "speaker": speaker,
            "dialogue_clean": text_clean,
            "bubble_shape": tc["shape"],
            "speaker_position": "center",
            "font_scale": 1.0,
            "x_ratio": 0.5,
            "y_ratio": 0.22,
            "text_align": tc["align"],
            "tail_x_ratio": tc["tail_x"],
            "tail_y_ratio": tc["tail_y"],
            "source": "unit_test"
        }
        
        # Override get_layout_plan locally
        integrator.get_layout_plan = lambda d, e, pid, sd, sp: test_plan
        
        # Integrate dialogue onto image
        final_img = integrator.integrate(
            image=base_img,
            dialogue=tc["dialogue"],
            emotion_beat=tc["beat"],
            panel_id=1
        )
        
        # Save output image
        out_path = f"outputs/test_render/{tc['name']}.png"
        final_img.save(out_path)
        print(f"Saved rendered panel to: {out_path}")
        print(f"Description: {tc['desc']}")
        
    print("\n[SUCCESS] All test renders generated! Please inspect outputs/test_render/ directory.")

if __name__ == "__main__":
    run_tests()
