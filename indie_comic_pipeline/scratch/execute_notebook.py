import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(nb_path):
    print(f"Executing notebook: {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': os.path.dirname(os.path.abspath(nb_path))}})
    
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Successfully executed and updated outputs for {nb_path}")

if __name__ == "__main__":
    p1 = "Pipeline_Operational_Benchmarking.ipynb"
    p2 = "indie_comic_pipeline/Pipeline_Operational_Benchmarking.ipynb"
    run_notebook(p1)
    if os.path.exists(p2):
        import shutil
        shutil.copy(p1, p2)
        print(f"Copied executed notebook to {p2}")
