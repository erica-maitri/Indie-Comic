import os
import base64
import sys
import json
import uuid
import threading
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

# Add parent directory to path to allow importing pipeline modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up paths for serving outputs
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(WORKSPACE_ROOT, "outputs")

# Try to import the IntegratedComicPipeline
try:
    from integrated_pipeline import IntegratedComicPipeline
    try:
        import torch
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False
    PIPELINE_AVAILABLE = True
except ImportError:
    print("Warning: integrated_pipeline.py not found. Running in mock mode.")
    PIPELINE_AVAILABLE = False
    IntegratedComicPipeline = None
    HAS_TORCH = False


@app.route('/outputs/<path:filename>')
def serve_outputs(filename):
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route('/')
def index():
    return render_template('comic_generator.html')

# Global job registry and executor (max_workers=1 to prevent concurrent GPU VRAM overload)
jobs = {}
executor = ThreadPoolExecutor(max_workers=1)

# Thread-safe persistent pipeline instantiation
_pipeline_instance = None
_pipeline_init_lock = threading.Lock()

def get_pipeline_instance():
    """
    Retrieves the global, persistent instance of the IntegratedComicPipeline.
    Uses lazy-loading thread-safe lock to prevent multiple model loads.
    """
    global _pipeline_instance
    if _pipeline_instance is not None:
        return _pipeline_instance
        
    with _pipeline_init_lock:
        if _pipeline_instance is None:
            if PIPELINE_AVAILABLE and IntegratedComicPipeline is not None:
                # Enable dry_run if CUDA is not available
                dry_run = False
                if HAS_TORCH:
                    import torch
                    dry_run = not torch.cuda.is_available()
                
                print(f"[app.py] Initializing persistent IntegratedComicPipeline (dry_run={dry_run})...")
                _pipeline_instance = IntegratedComicPipeline(dry_run=dry_run)
                print("[app.py] Persistent pipeline initialized and cached in memory.")
    return _pipeline_instance

def run_generation_task(task_id, prompt, style, character, world):
    """Background task to run the generation pipeline."""
    try:
        pipeline = get_pipeline_instance()
        if pipeline is not None:
            result = pipeline.run(
                prompt=prompt,
                character_name=character,
                story_world=world,
                panel_count=4
            )
            image = result['pages'][0]['page_image']
            
            # Convert to base64 for web
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            jobs[task_id] = {
                'status': 'completed',
                'image': f"data:image/png;base64,{img_str}",
                'prompt': prompt,
                'feedback_url': f"/feedback/{task_id}"
            }
        else:
            jobs[task_id] = {
                'status': 'completed',
                'image': '',
                'prompt': prompt,
                'feedback_url': f"/feedback/mock_123",
                'error': 'Pipeline not available. Please ensure integrated_pipeline.py is in the parent directory.'
            }
    except Exception as e:
        jobs[task_id] = {
            'status': 'failed',
            'error': str(e)
        }


@app.route('/generate')
def generate():
    prompt = request.args.get('prompt', 'A superhero flying over the city')
    style = request.args.get('style', 'manga')
    character = request.args.get('character', 'Spider-Man')
    world = request.args.get('world', 'Cyberpunk 2077')
    
    task_id = uuid.uuid4().hex
    jobs[task_id] = {
        'status': 'pending'
    }
    
    executor.submit(run_generation_task, task_id, prompt, style, character, world)
    
    return jsonify({
        'status': 'pending',
        'task_id': task_id
    })


@app.route('/status/<task_id>')
def status(task_id):
    job = jobs.get(task_id)
    if not job:
        return jsonify({'status': 'not_found', 'error': 'Task not found'}), 404
    return jsonify(job)

@app.route('/feedback/<panel_id>', methods=['POST'])
def feedback(panel_id):
    try:
        req_data = request.json or {}
        rating = req_data.get('rating')
        comment = req_data.get('comment')
        
        try:
            rating_val = int(rating) if rating is not None else 5
        except ValueError:
            rating_val = 5
            
        comment_val = str(comment) if comment is not None else ""
        
        # Connect to RLHFFeedbackLoop and SystemOptimizer
        try:
            from core.feedback import RLHFFeedbackLoop
            from utils.config_helper import load_settings
            try:
                settings = load_settings() or {}
                comics_dir = settings.get("outputs", {}).get("comics_dir", "outputs/comics")
            except Exception:
                comics_dir = "outputs/comics"
                
            if not os.path.isabs(comics_dir):
                feedback_path = os.path.join(WORKSPACE_ROOT, comics_dir, "rlhf_feedback.json")
            else:
                feedback_path = os.path.join(comics_dir, "rlhf_feedback.json")
                
            feedback_loop = RLHFFeedbackLoop(feedback_path=feedback_path)
            
            # Since this is page-level rating from the UI, we log it for the 4 generated panels (IDs 1-4)
            # so the preference model and system optimizer can process them.
            for pid in [1, 2, 3, 4]:
                feedback_loop.add_panel_feedback(
                    panel_id=pid,
                    rating=rating_val,
                    comment=comment_val,
                    engagement_time=0.0,
                    prompt_used="",
                    generation_backend="sdxl"
                )
                
            # 1. Auto-train user preference critic model
            try:
                from core.user_preference_critic import UserPreferenceCritic
                pref_critic = UserPreferenceCritic(
                    model_path=os.path.join(OUTPUTS_DIR, "user_preference_model.pt")
                )
                pref_critic.train_from_feedback_file(
                    feedback_file=feedback_loop.feedback_path,
                    panels_dir=os.path.join(OUTPUTS_DIR, "panels"),
                    min_records=3
                )
            except Exception as e:
                print(f"Failed to auto-train user preference critic: {e}")
                
            # 2. Run system feedback tuner and apply config adjustments
            try:
                from core.feedback_tuner import HeuristicFeedbackTuner
                settings_path = os.path.join(WORKSPACE_ROOT, "config", "settings.yaml")
                tuner = HeuristicFeedbackTuner(
                    feedback_loop=feedback_loop,
                    settings_path=settings_path
                )
                adjusts = tuner.tune_from_feedback()
                if tuner.apply_optimizations(adjusts):
                    print("[OK] Tuning adjustments successfully applied to configuration!")
                else:
                    print("Optimization adjustments evaluated, no config updates needed.")
            except Exception as e:
                print(f"Failed to run system optimizer: {e}")
                
        except Exception as e:
            print(f"Could not log feedback to RLHFFeedbackLoop: {e}")
            
        print(f"Feedback received: {rating_val}/5 - {comment_val}")
        
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/get_comic_data')
def get_comic_data():
    """
    Returns list of generated panels and assembled pages for the current comic.
    Scan outputs/panels/ for layout JSONs and raw/final images.
    """
    import glob
    
    # 1. Scan for raw panel images to see which ones are generated
    raw_pattern = os.path.join(OUTPUTS_DIR, "panels", "panel_*_page_*.png")
    raw_files = sorted(glob.glob(raw_pattern))
    
    panels_data = []
    for raw_path in raw_files:
        filename = os.path.basename(raw_path)
        try:
            parts = filename.replace(".png", "").split("_")
            panel_id = int(parts[1])
        except Exception:
            continue
            
        final_filename = f"panel_{panel_id:03d}_final.png"
        final_path = os.path.join(OUTPUTS_DIR, "panels", final_filename)
        
        json_filename = f"panel_{panel_id:03d}_bubble_layout.json"
        json_path = os.path.join(OUTPUTS_DIR, "panels", json_filename)
        
        plan = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    plan = json.load(f)
            except Exception as e:
                print(f"Error loading plan: {e}")
                
        x_ratio = plan.get("x_ratio", 0.5)
        y_ratio = plan.get("y_ratio", 0.15)
        dialogue = plan.get("dialogue_clean", "")
        speaker = plan.get("speaker", "")
        bubble_shape = plan.get("bubble_shape", "ellipse")
        font_scale = plan.get("font_scale", 1.0)
        text_align = plan.get("text_align", "center")
        tail_x_ratio = plan.get("tail_x_ratio")
        tail_y_ratio = plan.get("tail_y_ratio")
        
        panels_data.append({
            "panel_id": panel_id,
            "raw_url": f"/outputs/panels/{filename}",
            "final_url": f"/outputs/panels/{final_filename}" if os.path.exists(final_path) else f"/outputs/panels/{filename}",
            "x_ratio": x_ratio,
            "y_ratio": y_ratio,
            "dialogue": dialogue,
            "speaker": speaker or "",
            "bubble_shape": bubble_shape,
            "font_scale": font_scale,
            "text_align": text_align,
            "tail_x_ratio": tail_x_ratio,
            "tail_y_ratio": tail_y_ratio
        })
        
    # 2. Scan for assembled page layout images
    page_pattern = os.path.join(OUTPUTS_DIR, "comics", "page_*_layout_integrated.png")
    page_files = sorted(glob.glob(page_pattern))
    pages_data = [f"/outputs/comics/{os.path.basename(p)}" for p in page_files]
    
    return jsonify({
        "panels": panels_data,
        "pages": pages_data
    })


@app.route('/update_bubble', methods=['POST'])
def update_bubble():
    """
    Accepts updated coordinates/text for a speech bubble, updates the JSON file,
    re-runs text integration & layout assembly (skipping heavy GPU model loading),
    and returns updated panel data.
    """
    if not PIPELINE_AVAILABLE or IntegratedComicPipeline is None:
        return jsonify({"error": "Pipeline not initialized"}), 500
        
    try:
        req_data = request.json or {}
        panel_id_raw = req_data.get("panel_id")
        x_ratio_raw = req_data.get("x_ratio")
        y_ratio_raw = req_data.get("y_ratio")
        
        if panel_id_raw is None or x_ratio_raw is None or y_ratio_raw is None:
            return jsonify({"error": "Missing panel_id, x_ratio, or y_ratio"}), 400
            
        panel_id = int(panel_id_raw)
        x_ratio = float(x_ratio_raw)
        y_ratio = float(y_ratio_raw)
        dialogue = req_data.get("dialogue", "").strip()
        speaker = req_data.get("speaker", "").strip()
        bubble_shape = req_data.get("bubble_shape", "ellipse")
        
        font_scale_raw = req_data.get("font_scale")
        font_scale = float(font_scale_raw) if font_scale_raw is not None else 1.0
        
        # 1. Update layout JSON
        json_filename = f"panel_{panel_id:03d}_bubble_layout.json"
        json_path = os.path.join(OUTPUTS_DIR, "panels", json_filename)
        
        plan = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    plan = json.load(f)
            except Exception:
                pass
                
        plan["x_ratio"] = x_ratio
        plan["y_ratio"] = y_ratio
        plan["dialogue_clean"] = dialogue
        plan["speaker"] = speaker or None
        plan["bubble_shape"] = bubble_shape
        plan["font_scale"] = font_scale
        
        text_align = req_data.get("text_align", "center")
        tx = req_data.get("tail_x_ratio")
        ty = req_data.get("tail_y_ratio")
        
        plan["text_align"] = text_align
        plan["tail_x_ratio"] = float(tx) if tx is not None else None
        plan["tail_y_ratio"] = float(ty) if ty is not None else None
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2)
            
        # 2. Re-run render & assembly in skip_backends mode
        pipeline = IntegratedComicPipeline(skip_backends=True)
        rebuild_result = pipeline.rebuild_comic()
        
        # Get new final image url
        final_filename = f"panel_{panel_id:03d}_final.png"
        final_path = os.path.join(OUTPUTS_DIR, "panels", final_filename)
        final_url = f"/outputs/panels/{final_filename}" if os.path.exists(final_path) else ""
        
        page_pattern = os.path.join(OUTPUTS_DIR, "comics", "page_*_layout_integrated.png")
        import glob
        page_files = sorted(glob.glob(page_pattern))
        pages_data = [f"/outputs/comics/{os.path.basename(p)}" for p in page_files]
        
        return jsonify({
            "status": "success",
            "panel_id": panel_id,
            "final_url": final_url,
            "pages": pages_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)

