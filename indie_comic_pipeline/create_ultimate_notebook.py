import nbformat as nbf
import os

def create_ultimate_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = [
        nbf.v4.new_markdown_cell("# 🎨 Ultimate AI Comic Generator (Phase 3 & 4)\n\nThis notebook runs the new Master Pipeline class, bypassing the old fragmented scripts. It includes the quantitative metrics, audio TTS, CBR/CBZ exporting, and the unified style manager!"),
        
        nbf.v4.new_markdown_cell("## 📁 Step 1: Clone Repository & Setup (Colab Only)\nThis block detects if you are in Google Colab and downloads the repository automatically."),
        nbf.v4.new_code_cell("""import os, subprocess

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    REPO_DIR = "/content/indie_comic_pipeline"
    if not os.path.exists(REPO_DIR):
        print("Cloning repository in Colab...")
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/Cyberpunk-San/Indie-Comic.git", REPO_DIR], check=True)
    os.chdir(REPO_DIR)
else:
    # If running locally, make sure we are in the right folder
    cwd = os.getcwd()
    if os.path.basename(cwd) != "indie_comic_pipeline" and os.path.exists(os.path.join(cwd, "indie_comic_pipeline")):
        os.chdir(os.path.join(cwd, "indie_comic_pipeline"))

print(f"✅ Working directory set to: {os.getcwd()}")"""),
        
        nbf.v4.new_markdown_cell("## 📦 Step 2: Install Dependencies"),
        nbf.v4.new_code_cell("!pip install -r requirements.txt"),
        
        nbf.v4.new_markdown_cell("## ⚙️ Step 2: Configuration\nDefine your character, world, and the visual style."),
        nbf.v4.new_code_cell("""from ultimate_comic_pipeline import UltimateComicGenerator, ComicConfig

# You can change style to: 'manga', 'western_comic', 'noir', 'watercolor', or 'retro'
config = ComicConfig(
    character_name="Spider-Man",
    story_world="Cyberpunk 2077",
    style="manga",
    num_pages=1,  # Start with 1 page to test
    enable_memory_management=True
)

print(f"🎬 Character: {config.character_name}")
print(f"🌍 World: {config.story_world}")
print(f"🎨 Style: {config.style}")

# Initialize the Master Pipeline (Loads models into memory)
generator = UltimateComicGenerator(config)
print("✅ Pipeline Initialized")"""),

        nbf.v4.new_markdown_cell("## 🚀 Step 3: Generate the Comic!\nThis step generates the images, places speech bubbles with YOLO, and enforces character emotion tracking."),
        nbf.v4.new_code_cell("""from IPython.display import display

story_prompt = f"A dramatic entrance of {config.character_name} in {config.story_world}, feeling determined."

# Execute the pipeline
print("Generating comic... (This may take a minute for the first run)")
result = generator.generate_comic(story_prompt)

# Display the final unified comic image
print(f"📊 Overall Pipeline Quality Score: {result['overall_quality']:.2f}")
display(result['comic_image'])"""),

        nbf.v4.new_markdown_cell("## 🔊 Step 4: Audio Generation (TTS)\nLet's generate voice lines for the dialogue from the first panel."),
        nbf.v4.new_code_cell("""from audio_integration import AudioIntegrator
from IPython.display import Audio, display

audio_engine = AudioIntegrator()

# Extract dialogue from the first panel of the first page
first_panel_dialogue = result['pages'][0]['panels'][0]['dialogue']
print(f"Text to Speech: '{first_panel_dialogue}'")

# Generate the MP3
audio_path = audio_engine.generate_audio_dialogue(first_panel_dialogue, config.character_name)

if audio_path:
    display(Audio(audio_path, autoplay=False))
else:
    print("⚠️ No audio generated.")"""),

        nbf.v4.new_markdown_cell("## 📦 Step 5: Export to standard formats (CBZ)\nGenerate a universally readable `.cbz` file for comic book readers."),
        nbf.v4.new_code_cell("""from comic_exporter import ComicExporter
import os

exporter = ComicExporter()

# Export to CBZ
safe_title = f"{config.character_name}_{config.story_world}".replace(" ", "_")
cbz_path = exporter.export_cbz(result['pages'], title=safe_title)

# Also create a web comic HTML format
html_path = exporter.export_web_comic(result['pages'])

print(f"✅ Comic saved to: {cbz_path}")
print(f"✅ Interactive web format saved to: {html_path}")""")
    ]
    
    nb['cells'] = cells
    
    output_path = 'Ultimate_Comic_T4.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    create_ultimate_notebook()
