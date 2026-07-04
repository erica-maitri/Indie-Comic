import os
import subprocess
import shutil
from pathlib import Path

def download_datasets():
    """Download and prepare selected comic datasets.
    This is a placeholder implementation – you should replace URLs and preprocessing
    steps with the actual dataset sources you want to benchmark against.
    """
    datasets = {
        "Manga109": "https://huggingface.co/datasets/hal-utokyo/Manga109",
        "CoMix": "https://github.com/emanuelevivoli/CoMix.git",
        "DatasetComics": "https://github.com/RimiChen/Dataset-comics.git",
    }
    dest_root = Path(__file__).parent / "data"
    dest_root.mkdir(parents=True, exist_ok=True)
    for name, url in datasets.items():
        dest = dest_root / name
        if dest.exists():
            print(f"{name} already exists, skipping download.")
            continue
        if url.endswith('.git'):
            print(f"Cloning {name} ...")
            subprocess.run(["git", "clone", url, str(dest)], check=True)
        else:
            zip_path = dest_root / f"{name}.zip"
            print(f"Downloading {name} ...")
            subprocess.run(["curl", "-L", "-o", str(zip_path), url], check=True)
            print(f"Extracting {name} ...")
            shutil.unpack_archive(str(zip_path), str(dest))
            zip_path.unlink()
    print("All datasets downloaded to", dest_root)

if __name__ == "__main__":
    download_datasets()
