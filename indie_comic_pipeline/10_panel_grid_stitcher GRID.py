"""
PRODUCTION ENGINE - 10-PANEL GRID STITCHER
Fetches sequential frames from production paths and bundles them into an unified matrix sheet canvas.
"""
import os
import sys
import glob
from PIL import Image

print("=" * 80)
print("🔲 STAGE 2 ENGINE START: COMPILING 10-PANEL NARRATIVE BOOK CANVAS")
print("=" * 80)

try:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT_DIR = os.getcwd()

sys.path.append(ROOT_DIR)
from utils.config_helper import load_settings, get_output_path
from utils.image_utils import create_comic_grid

# Define localized file system directories paths
panels_source_dir = os.path.join(ROOT_DIR, "outputs", "production_run", "panels")
output_master_dir = os.path.join(ROOT_DIR, "outputs", "production_run")

search_pattern = os.path.join(panels_source_dir, "production_panel_*.png")
panel_paths = sorted(glob.glob(search_pattern), key=lambda x: int(os.path.basename(x).split('_')[2]))

if len(panel_paths) < 10:
    print(f"❌ Error: Expected 10 execution assets, but only found {len(panel_paths)} units inside folder directory path.")
    sys.exit(1)

# Ensure precise slice allocation balance
panel_paths = panel_paths[:10]
print(f"📦 Successfully grouped all {len(panel_paths)} chronological frames.")

# Final output sheet target names allocation
final_sheet_path = os.path.join(output_master_dir, "final_10_panel_comic_book_page.png")

print("\n⚙️ Processing 2x5 Matrix Alignment Stitching Layer (Scaled grid to guarantee low-overhead)...")

# Overriding default dimensions mappings to fit 10 nodes beautifully
create_comic_grid(
    images_paths=panel_paths,
    output_path=final_sheet_path,
    grid_size=(2, 5),    # Geometric Matrix Layout: 2 Rows, 5 Columns = 10 Panels cleanly packed
    cell_size=(512, 512) # Compressed runtime dimensions constraint
)

print(f"✅ Success: Master Unified Crossover Comic Sheet Serialized at: {final_sheet_path}")

# Display Layer System Call
print("\n🖥️ RENDERING FINAL COMIC OUTPUT ON EXECUTION SCREEN:")
print("-" * 65)
try:
    from IPython.display import display
    final_sheet_canvas = Image.open(final_sheet_path)
    display(final_sheet_canvas)
    print("-" * 65)
    print("🎉 Production Pipeline completely visualized above!")
except Exception as e:
    print(f"⚠️ UI Display Layer bypass tracking: {e}")

print("=" * 80)