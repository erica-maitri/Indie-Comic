import nbformat as nbf
import os

def create_notebook(filename, title, description, cells_data):
    nb = nbf.v4.new_notebook()
    
    # Title & Description
    cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n{description}")]
    
    # Environment Setup Cell
    setup_code = """# ============================================================
# Universal Colab/Local Setup — run this first in every notebook
# ============================================================
import os, sys, urllib.request

try:
    from google.colab import files  # type: ignore
    _IN_COLAB = True
except ImportError:
    _IN_COLAB = False

if _IN_COLAB:
    print("🚀 Detected Google Colab. Setting up environment...")
    _repo = "/content/Indie-Comic"
    if not os.path.exists(_repo):
        import subprocess
        subprocess.run(["git", "clone", "--depth", "1",
            "https://github.com/Cyberpunk-San/Indie-Comic.git", _repo], check=True)
    
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
    
    # Custom Cells
    for cell_type, content in cells_data:
        if cell_type == "md":
            cells.append(nbf.v4.new_markdown_cell(content))
        elif cell_type == "code":
            cells.append(nbf.v4.new_code_cell(content))
            
    nb['cells'] = cells
    with open(filename, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Generated: {filename}")

def main():
    # =========================================================================
    # Phase 0: Story Intake
    # =========================================================================
    create_notebook(
        "00_Phase_0_Story_Intake.ipynb",
        "🚀 Phase 0: Story Intake Engine",
        "This notebook demonstrates Phase 0 of the pipeline: processing raw narrative and emotional user prompts through the Story-Weaver LLM to generate structured story configs.",
        [
            ("md", "## ⚙️ 1. Process story prompt through Story Intake"),
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
    )

    # =========================================================================
    # Phase 1: Narrative Planning Layer
    # =========================================================================
    create_notebook(
        "01_Phase_1_Narrative_Planning.ipynb",
        "🚀 Phase 1: Narrative Planning Layer",
        "This notebook demonstrates Phase 1 of the pipeline: orchestrating the Storyboard, Character, Scene, and Layout agents through the shared Memory Blackboard.",
        [
            ("md", "## 🧠 1. Run multi-agent planning coordinator"),
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
    )

    # =========================================================================
    # Phase 2: Reference-Free Anchoring
    # =========================================================================
    create_notebook(
        "02_Phase_2_Reference_Free_Anchoring.ipynb",
        "🚀 Phase 2: Reference-Free Anchoring",
        "This notebook isolates the first generated panel as the primary visual anchor and extracts identity embedding tokens (facial topology, wardrobe features) for downstream panels.",
        [
            ("md", "## ⚓ 1. Isolate Visual Anchor & Extract Identity Tokens"),
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
    )

    # =========================================================================
    # Phase 3 & 4: In-Generation Consistency & Control
    # =========================================================================
    create_notebook(
        "03_04_Phase_3_4_In_Generation_Consistency_and_Control.ipynb",
        "🚀 Phase 3 & 4: In-Generation Consistency & Composable Control",
        "This notebook demonstrates the unified panel generation loop using model weight blending (CharCom) and Advanced Attention mechanisms (L1 Heat, L2 Shared Cache, L3 STE).",
        [
            ("md", "## 🔬 1. Blend Model Weights & Apply Attention Hooks"),
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
        "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
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
    )

    # =========================================================================
    # Phase 5: Integrated Text-Image Generation
    # =========================================================================
    create_notebook(
        "05_Phase_5_Integrated_Text_Image_Generation.ipynb",
        "🚀 Phase 5: Integrated Text-Image Generation",
        "This notebook runs the DiffSensei bubble planner, mapping text layout coordinates to avoid subject/facial visual collisions.",
        [
            ("md", "## 💬 1. Layout Text Bubble & Generate Overlay"),
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
    )

    # =========================================================================
    # Phase 6: Quality Validation Layer
    # =========================================================================
    create_notebook(
        "06_Phase_6_Quality_Validation_Layer.ipynb",
        "🚀 Phase 6: Quality Validation Layer",
        "This notebook demonstrates the COMIC Critic Pipeline: checking panels against visual, narrative, emotional, aesthetic, and readability thresholds.",
        [
            ("md", "## ⚖️ 1. Run 5-dimension Quality Critic Evaluation"),
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
    )

    # =========================================================================
    # Phase 7: Layout & Assembly
    # =========================================================================
    create_notebook(
        "07_Phase_7_Layout_and_Assembly.ipynb",
        "🚀 Phase 7: Layout & Assembly",
        "This notebook demonstrates the MangaFlow Layout Engine: dynamically cutting borders and arranging panel matrices based on story action intensities.",
        [
            ("md", "## 📐 1. Dynamic Layout Assembly"),
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
    )

    # =========================================================================
    # Phase 8: Export Module & Adaptive RLHF Systems
    # =========================================================================
    create_notebook(
        "08_Phase_8_Export_and_RLHF.ipynb",
        "🚀 Phase 8: Export Module & Adaptive RLHF Systems",
        "This notebook exports pages to PDF/CBZ/HTML and demonstrates the Human Alignment Telemetry Loop with parameter backpropagation optimization.",
        [
            ("md", "## 📦 1. Export Formats & Run RLHF Optimization Loop"),
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
    )

if __name__ == "__main__":
    main()
