import json
import os

notebook_path = "indie_comic_pipeline/Indie_Comic_Pipeline.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb.get("cells", [])):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        cell_text = "".join(source)
        if "IntegratedComicPipeline" in cell_text:
            print(f"Cell Index: {idx}")
            print(f"Contains 'wait_for_export': {'wait_for_export' in cell_text}")
            for i, line in enumerate(source):
                print(f"Line {i}: {repr(line)}")
