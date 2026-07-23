import os
import sys

# Configure environment variables to use the lighter qwen2.5:1.5b model instead of llama3.2
# Qwen 2.5 1.5B is half the size and runs much faster on CPU, preventing 60s timeouts
os.environ["OLLAMA_MODEL"] = "qwen2.5:1.5b"
os.environ["MODEL_PATH"] = "qwen2.5:1.5b"

import json
import queue
import logging
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, send_from_directory, send_file

# Add parent directories to sys.path so we can import integration and indie_comic_pipeline
_FRONTEND_DIR = Path(__file__).parent.resolve()
_REPO_ROOT = _FRONTEND_DIR.parent.resolve()

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_PIPELINE_ROOT = _REPO_ROOT / "indie_comic_pipeline"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from integration.emotion_router import EmotionRouter
from integration.pipeline_launcher import PipelineLauncher

app = Flask(__name__)

# Queue-based log handler to stream logging records via SSE
class QueueLogHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
        self.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))

    def emit(self, record):
        self.log_queue.put(self.format(record))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate_outline', methods=['POST'])
def generate_outline():
    try:
        data = request.json or {}
        premise = data.get('premise', '').strip()
        panel_count = int(data.get('panel_count', 6))
        mode = data.get('mode', 'quick')
        style = data.get('style', 'noir')
        custom_style = data.get('custom_style', '').strip()
        
        if not premise:
            return jsonify({'error': 'Story premise cannot be empty.'}), 400
            
        style_reference = custom_style if style == 'custom' else style
        
        engine = data.get('engine', 'template')
        
        # Analyze premise and route to arc using EmotionRouter
        router = EmotionRouter()
        routing = router.full_pipeline(premise)
        ctx = routing["story_context"]
        
        use_story_weaver = (mode == 'guided')
        launcher = PipelineLauncher(dry_run=True, emotion_router=router)
        
        story_script = None
        if engine == 'llm':
            try:
                if use_story_weaver:
                    # Generate outline via Story Weaver LLM
                    raw_script = launcher.sw_bridge.generate(ctx, panel_count=panel_count)
                    if raw_script:
                        from integration.pipeline_launcher import _adapt_story_weaver_script
                        story_script = _adapt_story_weaver_script(raw_script, ctx)
                        
                if not story_script:
                    # Fallback to pipeline's own StoryIntakeEngine
                    story_script = launcher._get_pipeline().story_intake.process_prompt(
                        user_prompt=premise,
                        panel_count=panel_count,
                        character_name=ctx["character_name"],
                        story_world=ctx["character_world"],
                        style_reference=style_reference,
                        character_characteristics=ctx.get("character_description", ""),
                        story_reference=ctx.get("arc_journey", "")
                    )
            except Exception as llm_err:
                # Log error and fall back to template
                print(f"Local LLM generation failed or timed out: {llm_err}. Falling back to template storyboard.")
                
        if not story_script:
            # Generate fallback template story
            story_script = launcher._get_pipeline().story_intake._generate_fallback(
                user_prompt=premise,
                emotion=ctx["arc_key"],
                panel_count=panel_count,
                character_name=ctx["character_name"],
                story_world=ctx["character_world"]
            )
            
        if not story_script or "panels" not in story_script:
            return jsonify({'error': 'Failed to generate a valid storyboard script.'}), 500
            
        # Parse panel data for frontend
        panels_out = []
        for p in story_script.get("panels", []):
            dialogue = "..."
            if p.get("characters"):
                dialogue = p["characters"][0].get("dialogue", {}).get("text", "...")
            
            panels_out.append({
                "panel_id": p["panel"],
                "visual_description": p.get("environment", ""),
                "emotion": p.get("emotion_beat", "neutral"),
                "dialogue": dialogue
            })
            
        # Generate intensity beats for the Emotion Arc graph mapping
        intensity_points = []
        n = len(panels_out)
        for idx, p in enumerate(panels_out):
            climax_idx = int(n * 0.7)
            if idx == 0:
                y_val = 100 # In canvas coordinates, larger Y means lower intensity (top is 0)
            elif idx == n - 1:
                y_val = 110
            elif idx == climax_idx:
                y_val = 20
            elif idx < climax_idx:
                y_val = 100 - int(80 * (idx / climax_idx))
            else:
                y_val = 20 + int(90 * ((idx - climax_idx) / (n - 1 - climax_idx)))
                
            intensity_points.append({
                "panel_id": p["panel_id"],
                "label": f"P{p['panel_id']}",
                "y": y_val
            })
            
        return jsonify({
            "emotion": routing["emotion"],
            "arc_key": ctx["arc_key"],
            "journey": ctx["arc_journey"],
            "character_name": ctx["character_name"],
            "story_world": ctx["character_world"],
            "character_description": ctx.get("character_description", ""),
            "panels": panels_out,
            "intensity_points": intensity_points
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/draw_panels', methods=['POST'])
def draw_panels():
    try:
        data = request.json or {}
        premise = data.get('premise', '').strip()
        panel_count = int(data.get('panel_count', 6))
        style = data.get('style', 'noir')
        custom_style = data.get('custom_style', '').strip()
        character_name = data.get('character_name', 'Wanderer')
        character_characteristics = data.get('character_characteristics', '').strip()
        story_world = data.get('story_world', 'The Abstract')
        weave_mood = bool(data.get('weave_mood', False))
        panels_input = data.get('panels', [])
        
        style_reference = custom_style if style == 'custom' else style
        
        # Build raw script conforming to Story Weaver structure
        story_dict = {
            "recurring_motif": "",
            "mood_journey": premise,
            "panels": [
                {
                    "panel": int(p["panel_id"]),
                    "visual": p["visual_description"],
                    "dialogue": p["dialogue"],
                    "emotion_beat": p["emotion"],
                    "motion": "standing still"
                }
                for p in panels_input
            ],
            "_meta": {
                "emotion": data.get("emotion", "sadness"),
                "character": character_name,
                "world": story_world
            }
        }
        
        log_queue = queue.Queue()
        handler = QueueLogHandler(log_queue)
        
        loggers = [
            logging.getLogger("pipeline.coordinator"),
            logging.getLogger("pipeline.orchestrator"),
            logging.getLogger("integration.pipeline_launcher"),
            logging.getLogger("integration.emotion_router"),
        ]
        for logger in loggers:
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            
        launcher = PipelineLauncher(dry_run=True)
        
        result_container = {}
        def run_pipeline():
            try:
                # 1. Adapt the storyboard script from the UI
                from integration.pipeline_launcher import _adapt_story_weaver_script
                ctx = {
                    "user_text":             story_dict.get("mood_journey", ""),
                    "primary_emotion":       story_dict.get("_meta", {}).get("emotion", "sadness"),
                    "primary_confidence":    1.0,
                    "secondary_emotions":    [],
                    "character_name":        character_name,
                    "character_world":       story_world,
                    "character_description": character_characteristics or "",
                    "arc_journey":           story_dict.get("mood_journey", ""),
                }
                adapted = _adapt_story_weaver_script(story_dict, ctx)
                
                # 2. Get the pipeline instance
                pipeline = launcher._get_pipeline()
                
                # 3. Override advanced attention and quality critic threshold
                from core.advanced_attention import AdvancedAttentionManager
                pipeline.advanced_attention = AdvancedAttentionManager(
                    heat_alpha=float(data.get("heat_alpha", 0.03)),
                    attention_blend=float(data.get("attention_blend", 0.15)),
                    spatial_strength=float(data.get("spatial_strength", 0.08)),
                    enabled=True
                )
                pipeline.panel_engine.advanced_attention = pipeline.advanced_attention
                pipeline.quality_critic.threshold = float(data.get("critic_threshold", 0.55))
                
                # Expose T4/SDXL optimizations and resolution/step presets dynamically
                pipeline.settings["generation"]["width"] = int(data.get("width", 768))
                pipeline.settings["generation"]["height"] = int(data.get("height", 768))
                pipeline.settings["generation"]["inference_steps"] = int(data.get("inference_steps", 25))
                pipeline.settings["generation"]["seed"] = int(data.get("seed", 42))
                
                pipeline.settings["models"]["lora"]["adapter_scale"] = float(data.get("lora_scale", 0.8))
                pipeline.settings["models"]["ipadapter"]["enabled"] = bool(data.get("enable_ipadapter", False))
                pipeline.settings["models"]["controlnet"]["enabled"] = bool(data.get("enable_controlnet", True))
                pipeline.settings["generation"]["enable_model_cpu_offload"] = bool(data.get("enable_cpuoffload", True))
                
                # Consistency Checker Metrics mapping
                pipeline.settings["consistency"]["enable_ssim"] = bool(data.get("enable_ssim", True))
                pipeline.settings["consistency"]["enable_edge"] = bool(data.get("enable_edge", True))
                pipeline.settings["consistency"]["enable_color"] = bool(data.get("enable_color", True))
                pipeline.settings["consistency"]["enable_style"] = bool(data.get("enable_style", True))
                pipeline.settings["consistency"]["enable_clip"] = bool(data.get("enable_clip", False))
                pipeline.settings["consistency"]["enable_dinov2"] = bool(data.get("enable_dinov2", False))
                
                # Update quality critic metrics
                pipeline.quality_critic.metrics = {
                    "ssim": pipeline.settings["consistency"]["enable_ssim"],
                    "edge": pipeline.settings["consistency"]["enable_edge"],
                    "color": pipeline.settings["consistency"]["enable_color"],
                    "style": pipeline.settings["consistency"]["enable_style"],
                    "clip": pipeline.settings["consistency"]["enable_clip"],
                    "dinov2": pipeline.settings["consistency"]["enable_dinov2"],
                }
                
                # 4. Run the pipeline directly with _prebuilt_story!
                res = pipeline.run(
                    prompt                    = ctx["user_text"],
                    character_name            = character_name,
                    story_world               = story_world,
                    panel_count               = panel_count,
                    style_reference           = style_reference,
                    character_characteristics = character_characteristics or "",
                    story_reference           = "",
                    weave_mood                = weave_mood,
                    _prebuilt_story           = adapted
                )
                
                # Collect output files
                output_files = []
                if isinstance(res, dict):
                    for key in ("cbz_path", "cbr_path", "pdf_path", "output_path"):
                        val = res.get(key)
                        if val and os.path.exists(val):
                            output_files.append(val)
                    for f in res.get("exports", []):
                        if isinstance(f, str) and os.path.exists(f):
                            output_files.append(f)
                            
                result_container['success'] = True
                result_container['result'] = {
                    "pipeline_result": res,
                    "cbz_path": res.get("cbz_path", ""),
                    "pdf_path": res.get("pdf_path", ""),
                    "html_path": res.get("html_path", ""),
                    "output_files": output_files
                }
            except Exception as e:
                import traceback
                result_container['success'] = False
                result_container['error'] = str(e)
                result_container['traceback'] = traceback.format_exc()
                
        def generate_events():
            thread = threading.Thread(target=run_pipeline)
            thread.start()
            
            # Yield initial starting log
            yield f"data: {json.dumps({'log': '[SYSTEM] Initiating backend dry-run pipeline compilation...'})}\n\n"
            
            while thread.is_alive():
                try:
                    log_msg = log_queue.get(timeout=0.1)
                    yield f"data: {json.dumps({'log': log_msg})}\n\n"
                except queue.Empty:
                    continue
                    
            while not log_queue.empty():
                log_msg = log_queue.get_nowait()
                yield f"data: {json.dumps({'log': log_msg})}\n\n"
                
            # Cleanup logging handler
            for logger in loggers:
                logger.removeHandler(handler)
                
            if result_container.get('success'):
                res = result_container['result']
                
                # Format panels output relative urls
                panels_out = []
                for p in res.get("pipeline_result", {}).get("panels", []):
                    # We expect paths like panel_001_final.png
                    panel_id = p["panel_id"]
                    panels_out.append({
                        "panel_id": panel_id,
                        "image_path": f"/outputs/panels/panel_{panel_id:03d}_final.png"
                    })
                    
                # Format page layout outputs
                pages_out = []
                for p in res.get("pipeline_result", {}).get("pages", []):
                    page_num = p["page_num"]
                    pages_out.append({
                        "page_num": page_num,
                        "image_path": f"/outputs/comics/page_{page_num:03d}_layout_integrated.png"
                    })
                    
                # Standardize export file paths
                cbz_path = res.get('cbz_path', '')
                pdf_path = res.get('pdf_path', '')
                html_path = res.get('html_path', '')
                
                complete_data = {
                    'status': 'complete',
                    'pages': pages_out,
                    'panels': panels_out,
                    'cbz_path': cbz_path,
                    'pdf_path': pdf_path,
                    'html_path': html_path
                }
                yield f"data: {json.dumps(complete_data)}\n\n"
            else:
                err = result_container.get('error', 'Unknown pipeline failure.')
                tb = result_container.get('traceback', '')
                yield f"data: {json.dumps({'status': 'error', 'error': err, 'traceback': tb})}\n\n"
                
        return Response(generate_events(), mimetype='text/event-stream')
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/regenerate_panel', methods=['POST'])
def regenerate_panel():
    try:
        data = request.json or {}
        panel_id = int(data.get('panel_id', 1))
        premise = data.get('premise', '').strip()
        panel_count = int(data.get('panel_count', 6))
        style = data.get('style', 'noir')
        custom_style = data.get('custom_style', '').strip()
        character_name = data.get('character_name', 'Wanderer')
        character_characteristics = data.get('character_characteristics', '').strip()
        story_world = data.get('story_world', 'The Abstract')
        panels_input = data.get('panels', [])
        
        style_reference = custom_style if style == 'custom' else style
        
        # Build updated story_dict
        story_dict = {
            "recurring_motif": "",
            "mood_journey": premise,
            "panels": [
                {
                    "panel": int(p["panel_id"]),
                    "visual": p["visual_description"],
                    "dialogue": p["dialogue"],
                    "emotion_beat": p["emotion"],
                    "motion": "standing still"
                }
                for p in panels_input
            ],
            "_meta": {
                "emotion": data.get("emotion", "sadness"),
                "character": character_name,
                "world": story_world
            }
        }
        
        from integration.pipeline_launcher import PipelineLauncher, _adapt_story_weaver_script
        from core.memory import StorySectionMemory
        
        launcher = PipelineLauncher(dry_run=True)
        pipeline = launcher._get_pipeline()
        
        # Setup memory and plan
        ctx = {
            "user_text":             story_dict.get("mood_journey", ""),
            "primary_emotion":       data.get("emotion", "sadness"),
            "primary_confidence":    1.0,
            "secondary_emotions":    [],
            "character_name":        character_name,
            "character_world":       story_world,
            "character_description": character_characteristics or "",
            "arc_journey":           story_dict.get("mood_journey", ""),
        }
        adapted = _adapt_story_weaver_script(story_dict, ctx)
        
        pipeline.memory = StorySectionMemory()
        pipeline.agent_coordinator.memory = pipeline.memory
        pipeline.panel_engine.memory = pipeline.memory
        
        # Run planning phase to populate memory
        pipeline.agent_coordinator.run_planning(adapted)
        
        # Load latest checkpoint if it exists to preserve anchors/identity
        checkpoint_path = os.path.join(pipeline.output_dir, "storyboard_checkpoint_latest.json")
        if os.path.exists(checkpoint_path):
            try:
                pipeline.memory = StorySectionMemory.load_checkpoint(checkpoint_path)
                pipeline.agent_coordinator.memory = pipeline.memory
                pipeline.panel_engine.memory = pipeline.memory
                
                # Update memory for panel_id with the new description/dialogue from UI
                # so that the regenerated panel uses the edited prompt!
                # 1. Update raw_panels inside self.memory
                for rp in pipeline.memory.raw_panels:
                    if int(rp.get("panel", 0)) == panel_id:
                        ui_panel = next((p for p in panels_input if int(p["panel_id"]) == panel_id), None)
                        if ui_panel:
                            rp["environment"] = ui_panel["visual_description"]
                            for char in rp.get("characters", []):
                                if char.get("id") == character_name.lower():
                                    char["dialogue"]["text"] = ui_panel["dialogue"]
                                    char["expression"]["emotion"] = ui_panel["emotion"]
                
                # 2. Update panel_history
                for p in pipeline.memory.panel_history:
                    if p.panel_id == panel_id:
                        ui_panel = next((p for p in panels_input if int(p["panel_id"]) == panel_id), None)
                        if ui_panel:
                            p.prompt_used = ui_panel["visual_description"]
                            p.dialogue = ui_panel["dialogue"]
                            p.emotion = ui_panel["emotion"]
                
                # 3. Override advanced attention, quality critic, and model settings
                from core.advanced_attention import AdvancedAttentionManager
                pipeline.advanced_attention = AdvancedAttentionManager(
                    heat_alpha=float(data.get("heat_alpha", 0.03)),
                    attention_blend=float(data.get("attention_blend", 0.15)),
                    spatial_strength=float(data.get("spatial_strength", 0.08)),
                    enabled=True
                )
                pipeline.panel_engine.advanced_attention = pipeline.advanced_attention
                pipeline.quality_critic.threshold = float(data.get("critic_threshold", 0.55))
                
                # Expose T4/SDXL optimizations and resolution/step presets dynamically
                pipeline.settings["generation"]["width"] = int(data.get("width", 768))
                pipeline.settings["generation"]["height"] = int(data.get("height", 768))
                pipeline.settings["generation"]["inference_steps"] = int(data.get("inference_steps", 25))
                pipeline.settings["generation"]["seed"] = int(data.get("seed", 42))
                
                pipeline.settings["models"]["lora"]["adapter_scale"] = float(data.get("lora_scale", 0.8))
                pipeline.settings["models"]["ipadapter"]["enabled"] = bool(data.get("enable_ipadapter", False))
                pipeline.settings["models"]["controlnet"]["enabled"] = bool(data.get("enable_controlnet", True))
                pipeline.settings["generation"]["enable_model_cpu_offload"] = bool(data.get("enable_cpuoffload", True))
                
                # Consistency Checker Metrics mapping
                pipeline.settings["consistency"]["enable_ssim"] = bool(data.get("enable_ssim", True))
                pipeline.settings["consistency"]["enable_edge"] = bool(data.get("enable_edge", True))
                pipeline.settings["consistency"]["enable_color"] = bool(data.get("enable_color", True))
                pipeline.settings["consistency"]["enable_style"] = bool(data.get("enable_style", True))
                pipeline.settings["consistency"]["enable_clip"] = bool(data.get("enable_clip", False))
                pipeline.settings["consistency"]["enable_dinov2"] = bool(data.get("enable_dinov2", False))
                
                # Update quality critic metrics
                pipeline.quality_critic.metrics = {
                    "ssim": pipeline.settings["consistency"]["enable_ssim"],
                    "edge": pipeline.settings["consistency"]["enable_edge"],
                    "color": pipeline.settings["consistency"]["enable_color"],
                    "style": pipeline.settings["consistency"]["enable_style"],
                    "clip": pipeline.settings["consistency"]["enable_clip"],
                    "dinov2": pipeline.settings["consistency"]["enable_dinov2"],
                }
            except Exception as checkpoint_err:
                print(f"Failed to load checkpoint: {checkpoint_err}")
                
        # Generate the single panel
        res = pipeline._generate_single_panel_with_retry(panel_id)
        
        # Save updated checkpoint
        pipeline.memory.save_checkpoint(checkpoint_path)
        
        # Assemble layout pages since this panel changed
        # We need to compile the full list of generated panels to redraw the page layout!
        # Let's collect all panels from memory
        panels_completed = []
        for pid in range(1, pipeline.memory.total_panels + 1):
            prec = pipeline.memory.get_panel(pid)
            if prec:
                panels_completed.append({
                    "panel_id": pid,
                    "image_path": os.path.join(pipeline.panels_dir, f"panel_{pid:03d}_final.png"),
                    "dialogue": prec.panel_dialogue,
                    "emotion_beat": prec.panel_emotion_beat,
                    "page_num": pipeline.memory.get_page_num(pid)
                })
                
        # Re-layout and save page layout files
        panels_by_page = {}
        for p in panels_completed:
            page_num = p["page_num"]
            panels_by_page.setdefault(page_num, []).append(p)
            
        pages_out = []
        for page_num, page_panels in sorted(panels_by_page.items()):
            page_image = pipeline.layout_engine.layout_page(
                page_panels, page_num, text_integrator=pipeline.text_integrator
            )
            page_path = os.path.join(pipeline.output_dir, f"page_{page_num:03d}_layout_integrated.png")
            page_image.save(page_path)
            pages_out.append({
                "page_num": page_num,
                "image_path": f"/outputs/comics/page_{page_num:03d}_layout_integrated.png"
            })
            
        return jsonify({
            "panel_id": panel_id,
            "image_path": f"/outputs/panels/panel_{panel_id:03d}_final.png",
            "pages": pages_out
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/outputs/<path:filename>')
def serve_outputs(filename):
    outputs_dir = os.path.join(_REPO_ROOT, 'outputs')
    return send_from_directory(outputs_dir, filename)

@app.route('/api/download')
def download_file():
    file_path = request.args.get('path')
    if not file_path:
        return "File path is required", 400
        
    abs_path = os.path.abspath(file_path)
    outputs_dir = os.path.abspath(os.path.join(_REPO_ROOT, 'outputs'))
    
    # Secure download to only files in outputs directory
    if not abs_path.startswith(outputs_dir):
        return "Access denied", 403
        
    if not os.path.exists(abs_path):
        return "File not found", 404
        
    return send_file(abs_path, as_attachment=True)

if __name__ == '__main__':
    # Run development server on port 8000
    app.run(debug=True, port=8000)
