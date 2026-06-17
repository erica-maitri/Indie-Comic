import os
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Try to import the UltimateComicGenerator
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ultimate_comic_pipeline import UltimateComicGenerator, ComicConfig
    PIPELINE_AVAILABLE = True
except ImportError:
    print("Warning: ultimate_comic_pipeline.py not found. Running in mock mode.")
    PIPELINE_AVAILABLE = False


@app.route('/')
def index():
    return render_template('comic_generator.html')

@app.route('/generate')
def generate():
    prompt = request.args.get('prompt', 'A superhero flying over the city')
    style = request.args.get('style', 'manga')
    character = request.args.get('character', 'Spider-Man')
    world = request.args.get('world', 'Cyberpunk 2077')
    
    if PIPELINE_AVAILABLE:
        config = ComicConfig(
            character_name=character,
            story_world=world,
            style=style,
            num_pages=1  # Generate just 1 page for UI responsiveness
        )
        generator = UltimateComicGenerator(config)
        try:
            # Generate a 1-page comic
            result = generator.generate_comic(prompt)
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
            'error': 'Pipeline not available. Please ensure ultimate_comic_pipeline.py is in the parent directory.'
        })

@app.route('/feedback/<panel_id>', methods=['POST'])
def feedback(panel_id):
    try:
        rating = request.json.get('rating')
        comment = request.json.get('comment')
        
        # Here we would normally connect to IncrementalLearner
        print(f"Feedback received for panel {panel_id}: {rating}/5 - {comment}")
        
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
