import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indie_comic_pipeline.integrated_pipeline import IntegratedComicPipeline

print("[INIT] Initializing IntegratedComicPipeline in skip_backends mode...")
pipeline = IntegratedComicPipeline(skip_backends=True)

print("[RUN] Running rebuild_comic()...")
results = pipeline.rebuild_comic()

print("[SUCCESS] Comic Rebuilt Successfully!")
print(f"CBZ Export: {results['cbz_path']}")
print(f"HTML scrollbook: {results['html_path']}")
print(f"PDF document: {results['pdf_path']}")
print(f"Re-rendered Panels: {len(results['panels'])}")
