"""
MoodWeaver — Unified Generation Entrypoint
============================================
Reads LANGUAGE from .env and runs the full pipeline:

  LANGUAGE=english  →  stage2_story_generation.py only
  LANGUAGE=hindi    →  stage2_story_generation.py  +  translate_layer.py

Does NOT touch your fine-tuned/merged model — the English model
always generates first. Hindi is a translation pass on top.

Usage:
    python generate.py
"""

import os, subprocess, sys, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("moodweaver.generate")

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.split("#")[0].strip())

LANGUAGE = os.environ.get("LANGUAGE", "english").strip().lower()

def run(cmd: list):
    log.info(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)

def main():
    log.info(f"LANGUAGE={LANGUAGE}")

    # Step 1: Always generate in English using the fine-tuned merged model
    run([sys.executable, "stage2_story_generation.py"])

    # Step 2: If Hindi requested, translate the output
    if LANGUAGE == "hindi":
        run([sys.executable, "translate_layer.py", "--lang", "hindi"])
    else:
        log.info("LANGUAGE=english → done, no translation step.")

if __name__ == "__main__":
    main()