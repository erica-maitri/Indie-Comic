"""
COMIC BOOK PDF COMPILER - T4 OPTIMIZED
Assembles all generated page layout grids into a single PDF document
"""

import os
import sys
from PIL import Image
import re
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.config_helper import load_settings, get_output_path

def compile_pdf(layout_style='sdxl_lora_grid'):
    settings = load_settings()
    comics_dir = settings.get("outputs", {}).get("comics_dir", "outputs/comics")
    
    print("=" * 70)
    print("COMIC BOOK PDF COMPILER - T4 OPTIMIZED")
    print("=" * 70)
    print(f"Scanning '{comics_dir}' for pages matching pattern: *layout*{layout_style}*")
    
    # Get all files in comics dir
    if not os.path.exists(comics_dir):
        print(f"Error: Comics output directory not found at: {comics_dir}")
        return False
        
    files = os.listdir(comics_dir)
    
    # Find matching page files
    page_files = []
    for f in files:
        if "layout" in f and layout_style in f and f.endswith(".png"):
            match = re.search(r'page_(\d+)', f)
            if match:
                page_num = int(match.group(1))
                page_files.append((page_num, os.path.join(comics_dir, f)))
                
    if not page_files:
        print(f"\n⚠️ No page grid layouts found with style '{layout_style}'.")
        print("Available styles in output folder:")
        styles = set()
        for f in files:
            if "layout_" in f and f.endswith(".png"):
                parts = f.replace("page_", "").split("_layout_")
                if len(parts) > 1:
                    styles.add(parts[1].replace(".png", ""))
        for s in sorted(styles):
            print(f"  - {s}")
            
        # Try fallback styles
        for fallback in ["grid", "sdxl_base_grid", "sd15_lora_grid", "doodle_grid"]:
            if fallback != layout_style:
                print(f"\nTrying fallback style: {fallback}...")
                if compile_pdf(fallback):
                    return True
        return False
        
    # Sort pages numerically
    page_files.sort(key=lambda x: x[0])
    
    print(f"\n✅ Found {len(page_files)} pages to compile:")
    for num, path in page_files:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  - Page {num}: {os.path.basename(path)} ({size_mb:.1f} MB)")
        
    # Load images and convert to RGB
    images = []
    failed = []
    try:
        for num, path in page_files:
            try:
                img = Image.open(path)
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
                print(f"  ✓ Loaded page {num}")
            except Exception as e:
                print(f"  ✗ Failed to load page {num}: {e}")
                failed.append(num)
                
        if failed:
            print(f"\n⚠️ Failed to load pages: {failed}")
            if not images:
                return False
                
    except Exception as e:
        print(f"Error loading images: {e}")
        return False
        
    output_pdf = get_output_path(comics_dir, f"comic_book_{layout_style}.pdf")
    
    try:
        # Save as PDF with optimization
        if len(images) == 1:
            images[0].save(output_pdf, optimize=True)
        else:
            images[0].save(
                output_pdf,
                save_all=True,
                append_images=images[1:],
                optimize=True,
                quality=85  # Good balance between quality and file size
            )
        
        pdf_size_mb = os.path.getsize(output_pdf) / (1024 * 1024)
        print(f"\n✅ Comic PDF successfully compiled!")
        print(f"   File: {output_pdf}")
        print(f"   Size: {pdf_size_mb:.1f} MB")
        print(f"   Pages: {len(images)}")
        print("=" * 70)
        return True
    except Exception as e:
        print(f"Error saving PDF: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile generated comic pages into a PDF.")
    parser.add_argument("--style", type=str, default="sdxl_lora_grid", 
                        help="The layout style grid to search for")
    args = parser.parse_args()
    
    success = compile_pdf(args.style)
    if not success:
        print("\n❌ PDF compilation failed.")
        sys.exit(1)