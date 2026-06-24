import nbformat as nbf
import os
import glob

def create_unified_notebook(filename, title, description, phases):
    nb = nbf.v4.new_notebook()
    
    # Title & Description of the entire pipeline
    cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n{description}")]
    
    # Environment Setup Cell (run once at the start)
    setup_code = """# ============================================================
# Universal Cloud/Local Setup — run this first in every notebook
# ============================================================
import os, sys, urllib.request

try:
    from google.colab import files  # type: ignore
    _IN_COLAB = True
except ImportError:
    _IN_COLAB = False

_IN_KAGGLE = os.path.exists("/kaggle/working")
_IN_CLOUD = _IN_COLAB or _IN_KAGGLE

if _IN_CLOUD:
    print("🚀 Detected Cloud Environment (Colab/Kaggle). Setting up...")
    _repo = "/content/Indie-Comic" if _IN_COLAB else "/kaggle/working/Indie-Comic"
    if not os.path.exists(_repo):
        import subprocess
        subprocess.run(["git", "clone", "--depth", "1",
            "https://github.com/Cyberpunk-San/Indie-Comic.git", _repo], check=True)
    else:
        print("🔄 Repo already exists. Pulling latest changes...")
        import subprocess
        subprocess.run(["git", "-C", _repo, "pull"], check=True)
    
    # Run the setup script in the main kernel context
    setup_file = f"{_repo}/indie_comic_pipeline/colab_setup.py"
    exec(open(setup_file).read(), globals())
else:
    print("💻 Detected Local Jupyter. Setting up path...")
    _candidates = [
        os.path.join(os.getcwd(), "colab_setup.py"),
        os.path.join(os.getcwd(), "indie_comic_pipeline", "colab_setup.py"),
    ]
    _found = next((p for p in _candidates if os.path.exists(p)), None)
    if _found:
        exec(open(_found).read(), globals())
    else:
        print("⚠️ colab_setup.py not found — run from repo root")"""
    
    cells.append(nbf.v4.new_markdown_cell("## 🔧 0. Universal Environment Setup\nRun this cell first to configure Colab or local Jupyter environments."))
    setup_cell = nbf.v4.new_code_cell(setup_code)
    setup_cell['id'] = 'colab_setup_cell'
    cells.append(setup_cell)
    
    # Add each phase
    for phase_title, phase_desc, phase_cells in phases:
        cells.append(nbf.v4.new_markdown_cell(f"---\n\n## {phase_title}\n\n{phase_desc}"))
        for cell_type, content in phase_cells:
            if cell_type == "md":
                cells.append(nbf.v4.new_markdown_cell(content))
            elif cell_type == "code":
                cells.append(nbf.v4.new_code_cell(content))
                
    nb['cells'] = cells
    with open(filename, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Generated unified notebook: {filename}")

def clean_old_notebooks(directory):
    old_notebooks = [
        "00_Phase_0_Story_Intake.ipynb",
        "01_Phase_1_Narrative_Planning.ipynb",
        "02_Phase_2_Reference_Free_Anchoring.ipynb",
        "03_04_Phase_3_4_In_Generation_Consistency_and_Control.ipynb",
        "05_Phase_5_Integrated_Text_Image_Generation.ipynb",
        "06_Phase_6_Quality_Validation_Layer.ipynb",
        "07_Phase_7_Layout_and_Assembly.ipynb",
        "08_Phase_8_Export_and_RLHF.ipynb"
    ]
    for nb in old_notebooks:
        path = os.path.join(directory, nb)
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"Removed old notebook: {path}")
            except Exception as e:
                print(f"Error removing {path}: {e}")

def main():
    pipeline_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    
    print("Cleaning up old individual notebooks...")
    clean_old_notebooks(pipeline_dir)
    
    phases = [
        # Hugging Face Authentication
        (
            "🔑 Hugging Face Authentication",
            "This section handles authentication with Hugging Face. Gated models (like the official SDXL base model) require an active token. Paste your token in the field below, or run the cell to login interactively. Alternatively, you can define it in a `.env` file at the repository root.",
            [
                ("md", "### 🔑 Authenticate with Hugging Face Hub (Optional)"),
                ("code", """import os
# @markdown You can get a free token from: https://huggingface.co/settings/tokens
hf_token = "" # @param {type:"string"}

# First, check if HF_TOKEN is already set in the environment (e.g. from .env file)
if "HF_TOKEN" in os.environ and not hf_token:
    print("✅ Hugging Face Token already configured from environment/.env file!")
elif hf_token:
    os.environ["HF_TOKEN"] = hf_token
    print("✅ Hugging Face Token configured in environment from parameter!")
else:
    try:
        from huggingface_hub import notebook_login
        notebook_login()
    except Exception:
        print("ℹ️ Hugging Face login skipped. Unauthenticated downloads will be used.")""")
            ]
        ),
        # End-to-End Pipeline Execution
        (
            "🎬 Complete End-to-End Pipeline Execution",
            "This section runs the entire 8-phase pipeline in a single step. It automatically detects if a GPU is available to run real SDXL generation; otherwise, it runs a fast dry-run using mock panels with real dialogue and speech bubble layout optimization.",
            [
                ("md", "### ⚡ Configure and Run the Comic Generator"),
                ("code", """from integrated_pipeline import IntegratedComicPipeline
import torch
from PIL import Image
from IPython.display import display

# --- Customize Your Story & Characters here ---
prompt = "A lone wanderer discovers hope" # @param {type:"string"}
character_name = "Wanderer" # @param {type:"string"}
story_world = "The Abstract" # @param {type:"string"}
panel_count = 4 # @param {type:"integer"}

# Auto-detect hardware. Use GPU if available, fallback to mock dry-run on CPU.
dry_run = not torch.cuda.is_available()

print(f"🎬 Running pipeline with: prompt='{prompt}', character='{character_name}', world='{story_world}'")
print(f"🖥️ GPU detected: {torch.cuda.is_available()} (Running {'Real GPU Generation' if not dry_run else 'Fast Mock Dry-Run'})")

# Initialize and execute the master pipeline
pipeline = IntegratedComicPipeline(dry_run=dry_run)
results = pipeline.run(
    prompt=prompt,
    character_name=character_name,
    story_world=story_world,
    panel_count=panel_count
)

# Display the assembled comic pages directly in the notebook
for page in results["pages"]:
    print(f"\\n📖 Page {page['page_num']} Layout:")
    display(page["page_image"])""")
            ]
        ),
        # Phase 0
        (
            "🚀 Phase 0: Story Intake Engine",
            "This section demonstrates Phase 0 of the pipeline: processing raw narrative and emotional user prompts through the Story-Weaver LLM to generate structured story configs.",
            [
                ("md", "### ⚙️ 1. Process story prompt through Story Intake"),
                ("code", """from core.story_intake import StoryIntakeEngine
# Initialize the intake engine (uses local Ollama by default)
intake = StoryIntakeEngine()
print("Intake engine initialized.")

# Run fallback / template intake processing
config = intake.process_prompt(
    user_prompt="A lone wanderer discovers hope",
    panel_count=4,
    character_name="Wanderer",
    story_world="The Abstract"
)

print("Structured Story Config:")
import json
print(json.dumps(config, indent=2))""")
            ]
        ),
        # Phase 1
        (
            "🚀 Phase 1: Narrative Planning Layer",
            "This section demonstrates Phase 1 of the pipeline: orchestrating the Storyboard, Character, Scene, and Layout agents through the shared Memory Blackboard.",
            [
                ("md", "### 🧠 1. Run multi-agent planning coordinator"),
                ("code", """from core.memory import StorySectionMemory
from core.agents.agent_coordinator import AgentCoordinator

memory = StorySectionMemory()
coordinator = AgentCoordinator(memory)

# Sample story configuration
story_config = {
    "title": "Neon Sunset",
    "characters": [{"name": "Akira", "costume": "Leather jacket"}],
    "setting": {"location": "Mega-city alleyway", "lighting": "cyberpunk pink"},
    "mood_journey": "despair to hope",
    "recurring_motif": "broken circuit",
    "panels": [
        {"panel": 1, "visual": "Akira looking at the sky", "dialogue": "Is there anyone out there?", "emotion_beat": "lonely"},
        {"panel": 2, "visual": "A drone lights up the alley", "dialogue": "Intruder detected.", "emotion_beat": "startled"},
    ],
    "_metadata": {"character": "Akira", "world": "Mega-city", "emotion": "lonely"}
}

coordinator.run_planning(story_config)

print("Memory populated with page plans:")
for plan in memory.page_plans:
    print(f"Page {plan['page_number']}: phase={plan['pacing_phase']}, panels={len(plan['panels'])}")""")
            ]
        ),
        # Phase 2
        (
            "🚀 Phase 2: Reference-Free Anchoring",
            "This section isolates the first generated panel as the primary visual anchor and extracts identity embedding tokens (facial topology, wardrobe features) for downstream panels.",
            [
                ("md", "### ⚓ 1. Isolate Visual Anchor & Extract Identity Tokens"),
                ("code", """import os
from PIL import Image
from core.memory import StorySectionMemory
from core.anchoring import ReferenceFreeAnchor

memory = StorySectionMemory()
memory.register_character("Akira")

# Create mock panel 1 image
img = Image.new("RGB", (512, 512), (120, 140, 160))
anchor_dir = "outputs/anchors"
os.makedirs(anchor_dir, exist_ok=True)
mock_path = os.path.join(anchor_dir, "anchor_panel_1.png")
img.save(mock_path)

anchor_system = ReferenceFreeAnchor(device="cpu")
tokens = anchor_system.establish_anchor(img, panel_id=1, character_name="Akira", memory=memory)

print("Anchor established. Extracted features:")
print("Aesthetic score:", tokens.get("aesthetic_score"))
print("Mean brightness:", tokens.get("mean_brightness"))""")
            ]
        ),
        # Phase 3 & 4
        (
            "🚀 Phase 3 & 4: In-Generation Consistency & Composable Control",
            "This section demonstrates the unified panel generation loop using model weight blending (CharCom) and Advanced Attention mechanisms (L1 Heat, L2 Shared Cache, L3 STE).",
            [
                ("md", "### 🔬 1. Blend Model Weights & Apply Attention Hooks"),
                ("code", """import torch
from core.memory import StorySectionMemory
from core.advanced_attention import AdvancedAttentionManager
from core.backends.backend_selector import BackendSelector
from core.panel_engine import PanelEngine

memory = StorySectionMemory()
memory.register_character("Akira")

# Set up backend selector
selector = BackendSelector()

# Check if GPU (CUDA) is available
use_gpu = torch.cuda.is_available()

if use_gpu:
    print("🚀 GPU detected! Initializing real SDXL Backend...")
    from core.backends.sdxl_backend import SDXLBackend
    
    # Real SDXL configuration optimized for T4 GPU / Colab
    sdxl_config = {
        "model_name": "Lykon/dreamshaper-xl-1-0",
        "device": "cuda",
        "enable_cpu_offload": True,  # Enables CPU offloading to save VRAM
        "enable_attention_slicing": True,
        "enable_vae_slicing": True,
        "safety_checker": False,
    }
    
    # Optional LoRA adapter can be specified here
    # sdxl_config["lora_name"] = "artificialguybr/LineAniRedmond-LinearMangaSDXL-V2"
    # sdxl_config["lora_scale"] = 0.8
    
    real_backend = SDXLBackend()
    real_backend.load(sdxl_config)
    selector.register_backend("sdxl", real_backend)
else:
    print("💻 No GPU detected. Falling back to MockBackend for dry-run...")
    from integrated_pipeline import MockBackend
    mock_backend = MockBackend()
    mock_backend.load({})
    selector.register_backend("sdxl", mock_backend)

adv_attn = AdvancedAttentionManager(enabled=True)
engine = PanelEngine(memory=memory, backend_selector=selector, advanced_attention=adv_attn)

# Generate panel 1 with active attention
context = {"panel_id": 1, "panel_visual": "Character stands looking ahead", "panel_emotion_beat": "hopeful"}
result = engine.generate_panel(panel_id=1, context=context)

active_backend = selector.select(context)
print(f"Panel generated successfully using {active_backend.name} backend.")
print("Advanced Attention status:")
import json
print(json.dumps(adv_attn.get_status(), indent=2))""")
            ]
        ),
        # Phase 5
        (
            "🚀 Phase 5: Integrated Text-Image Generation",
            "This section runs the DiffSensei bubble planner, mapping text layout coordinates to avoid subject/facial visual collisions.",
            [
                ("md", "### 💬 1. Layout Text Bubble & Generate Overlay"),
                ("code", """from PIL import Image
from core.text_image_integrator import TextImageIntegrator

# Create blank canvas
img = Image.new("RGB", (768, 768), (240, 240, 245))

integrator = TextImageIntegrator(output_dir="outputs/panels")
final_img = integrator.integrate(
    image=img,
    dialogue="Uncle Ben... this city is darker than New York.",
    emotion_beat="determined",
    panel_id=1,
    scene_desc="Spider-Man in the dark city"
)
print("Overlay complete. Dimensions:", final_img.size)""")
            ]
        ),
        # Phase 6
        (
            "🚀 Phase 6: Quality Validation Layer",
            "This section demonstrates the COMIC Critic Pipeline: checking panels against visual, narrative, emotional, aesthetic, and readability thresholds.",
            [
                ("md", "### ⚖️ 1. Run 5-dimension Quality Critic Evaluation"),
                ("code", """import os
from PIL import Image
from core.memory import StorySectionMemory
from core.quality_critic import QualityCritic

memory = StorySectionMemory()
critic = QualityCritic(threshold=0.6, strict_threshold=0.8)

img = Image.new("RGB", (512, 512), (128, 128, 128))
panel_result = {
    "panel_id": 2,
    "image": img,
    "image_path": "outputs/panels/panel_002_page_1.png",
    "prompt": "Akira standing in the neon rain",
    "weights": {"lora_scale": 0.8}
}

# Run evaluation
evaluation = critic.evaluate(panel_result, memory)
print("Critic Verdict:", evaluation["verdict"])
print("Composite Score:", evaluation["composite_score"])
print("Adjustments recommended on failure:", evaluation["adjustments"])""")
            ]
        ),
        # Phase 7
        (
            "🚀 Phase 7: Layout & Assembly",
            "This section demonstrates the MangaFlow Layout Engine: dynamically cutting borders and arranging panel matrices based on story action intensities.",
            [
                ("md", "### 📐 1. Dynamic Layout Assembly"),
                ("code", """from PIL import Image
from core.layout_engine import MangaFlowLayoutEngine

engine = MangaFlowLayoutEngine(page_width=800, page_height=1200)

# 3 mock panel images
panels = [
    {"panel_id": 1, "image": Image.new("RGB", (400, 300), (200, 50, 50)), "page_num": 1},
    {"panel_id": 2, "image": Image.new("RGB", (400, 300), (50, 200, 50)), "page_num": 1},
    {"panel_id": 3, "image": Image.new("RGB", (800, 400), (50, 50, 200)), "page_num": 1},
]

page_image = engine.layout_page(panels, page_num=1)
print("Page layout assembled. Size:", page_image.size)""")
            ]
        ),
        # Phase 8
        (
            "🚀 Phase 8: Export Module & Adaptive RLHF Systems",
            "This section exports pages to PDF/CBZ/HTML and demonstrates the Human Alignment Telemetry Loop with parameter backpropagation optimization.",
            [
                ("md", "### 📦 1. Export Formats & Run RLHF Optimization Loop"),
                ("code", """import os
from PIL import Image
from comic_exporter import ComicExporter
from core.feedback import RLHFFeedbackLoop
from core.optimizer import SystemOptimizer

exporter = ComicExporter(output_dir="outputs/comics")
mock_page = {'page_num': 1, 'page_image': Image.new('RGB', (800, 1200), 'white'), 'panels': []}
pages = [mock_page]

cbz = exporter.export_cbz(pages, title="FinalComic")
print("Exported CBZ:", cbz)

# Initialize telemetry
feedback_path = "outputs/comics/test_rlhf_feedback.json"
feedback = RLHFFeedbackLoop(feedback_path=feedback_path)
feedback.add_panel_feedback(panel_id=1, rating=5, comment="Excellent style consistency!", prompt_used="...", generation_backend="sdxl")

# Run optimizer
optimizer = SystemOptimizer(feedback_loop=feedback, settings_path="config/settings.yaml")
adjusts = optimizer.optimize_system_parameters()

print("System Optimization Recommendations:")
print(adjusts)""")
            ]
        ),
        # Phase 9
        (
            "🚀 Phase 9: Comprehensive Model Evaluation",
            "This section calculates advanced metrics including FID, BLEU, IoU, CLIP (Text-Image and Image-Image), and DINOv2 Structural Similarity to evaluate the quality of the generated panels.",
            [
                ("md", "### 📊 1. Run Comprehensive Model Evaluator"),
                ("code", """import os
from PIL import Image
from core.evaluation_suite import ModelEvaluator
import json

evaluator = ModelEvaluator()

# We will evaluate against mock images, but in a real scenario you would point this to your actual generated panels and character sheets.
gen_img = Image.new('RGB', (256, 256), 'red')
ref_img = Image.new('RGB', (256, 256), 'blue')

metrics = {}

print("[1] Image Quality & Realism")
metrics['Aesthetic Score'] = evaluator.compute_aesthetic_score(gen_img)
print(f"  -> Aesthetic Score: {metrics['Aesthetic Score']:.4f}")

# FID expects real/generated images
fid_score = evaluator.compute_fid(gen_img, ref_img)
if fid_score is not None:
    metrics['FID'] = fid_score
    print(f"  -> FID Score: {metrics['FID']:.4f} (lower is better)")
else:
    print("  -> FID Score: SKIPPED (Install torch-fidelity to use)")

print("\\n[2] Semantic & Structural Consistency")
dinov2 = evaluator.compute_dinov2_similarity(gen_img, ref_img)
if dinov2 is not None:
    metrics['DINOv2 Similarity'] = dinov2
    print(f"  -> DINOv2: {metrics['DINOv2 Similarity']:.4f} (higher is better)")

clip_img = evaluator.compute_clip_image_similarity(gen_img, ref_img)
if clip_img is not None:
    metrics['CLIP Img2Img'] = clip_img
    print(f"  -> CLIP Img-Img: {metrics['CLIP Img2Img']:.4f} (higher is better)")

print("\\n[3] Text-to-Image Alignment")
clip_text = evaluator.compute_clip_text_alignment(gen_img, "A red square")
if clip_text is not None:
    metrics['CLIP Text2Img'] = clip_text
    print(f"  -> CLIP Text-Img: {metrics['CLIP Text2Img']:.4f} (higher is better)")

print("\\n[4] Text Generation Quality")
bleu = evaluator.compute_bleu("Hello", "Hello world")
if bleu is not None:
    metrics['BLEU Score'] = bleu
    print(f"  -> BLEU: {metrics['BLEU Score']:.4f} (higher is better)")

print("\\n[5] Layout Accuracy")
# Example bounding boxes: (x1, y1, x2, y2)
iou_score = evaluator.compute_iou((10, 10, 50, 50), (12, 12, 48, 48))
metrics['IoU Score'] = iou_score
print(f"  -> Bounding Box IoU: {metrics['IoU Score']:.4f} (higher is better)")

print("\\nFinal Metrics:")
print(json.dumps(metrics, indent=2))

evaluator.free_memory()""")
            ]
        )
    ]
    
    notebook_path = os.path.join(pipeline_dir, "Indie_Comic_Pipeline.ipynb")
    create_unified_notebook(
        notebook_path,
        "🎨 Ultimate AI Indie Comic Generator Complete Pipeline",
        "This unified notebook integrates the end-to-end research and execution workflow of the Indie Comic Generator pipeline. It covers story intake, narrative planning, visual anchoring, in-generation attention controls, text-image integration, quality validation, layout assembly, comic exporting, and RLHF parameters optimization.",
        phases
    )

if __name__ == "__main__":
    main()
