import os
import base64
import sys
from io import BytesIO
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Add parent directory to path to allow importing pipeline modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


@app.route('/')
def index():
    return render_template('comic_generator.html')

@app.route('/generate')
def generate():
    prompt = request.args.get('prompt', 'A superhero flying over the city')
    style = request.args.get('style', 'manga')
    character = request.args.get('character', 'Spider-Man')
    world = request.args.get('world', 'Cyberpunk 2077')
    
    if PIPELINE_AVAILABLE and IntegratedComicPipeline is not None:
        # Enable dry_run if CUDA is not available or if CPU-only mode is forced
        dry_run = False
            
        pipeline = IntegratedComicPipeline(dry_run=dry_run)
        try:
            # Generate a 1-page comic (which translates to 4 panels)
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
            
            return jsonify({
                'image': f"data:image/png;base64,{img_str}",
                'prompt': prompt,
                'feedback_url': f"/feedback/{hash(prompt)}"
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        # Mock mode if pipeline isn't properly set up yet
        return jsonify({
            'image': '',
            'prompt': prompt,
            'feedback_url': f"/feedback/mock_123",
            'error': 'Pipeline not available. Please ensure integrated_pipeline.py is in the parent directory.'
        })

@app.route('/feedback/<panel_id>', methods=['POST'])
def feedback(panel_id):
    try:
        req_data = request.json or {}
        rating = req_data.get('rating')
        comment = req_data.get('comment')
        
        # Safely convert types to match RLHFFeedbackLoop requirements
        try:
            panel_id_val = int(panel_id)
        except ValueError:
            panel_id_val = 1
            
        try:
            rating_val = int(rating) if rating is not None else 5
        except ValueError:
            rating_val = 5
            
        comment_val = str(comment) if comment is not None else ""
        
        # Connect to RLHFFeedbackLoop if available
        try:
            from core.feedback import RLHFFeedbackLoop
            feedback_loop = RLHFFeedbackLoop()
            feedback_loop.add_panel_feedback(
                panel_id=panel_id_val,
                rating=rating_val,
                comment=comment_val,
                engagement_time=0.0,
                prompt_used="",
                generation_backend=""
            )
        except Exception as e:
            print(f"Could not log feedback to RLHFFeedbackLoop: {e}")
            
        print(f"Feedback received for panel {panel_id}: {rating}/5 - {comment}")
        
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)

