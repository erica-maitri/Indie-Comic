"""
RUN FULL LANGCHAIN PIPELINE
Executes character extraction, story extraction, and fusion in sequence
"""

import subprocess

import sys

import os

print("=" * 70)

print("RUNNING FULL LANGCHAIN PIPELINE - Sequential LLM extraction")

print("=" * 70)

def check_ollama():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex(('localhost', 11434))
    sock.close()
    return result == 0

def ensure_ollama_running():
    import time
    if check_ollama():
        print("✅ Ollama server is active and running.")
        return True
    print("⚠️ Ollama daemon is not running. Attempting to start Ollama automatically...")
    try:
        if sys.platform == "win32":
            CREATE_NO_WINDOW = 0x08000000
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
        else:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        for attempt in range(15):
            time.sleep(1)
            if check_ollama():
                print("✅ Ollama server started and connected successfully!")
                return True
            print(f"   Waiting for Ollama to initialize... (attempt {attempt+1}/15)")
    except Exception as e:
        print(f"❌ Failed to auto-start Ollama: {e}")
        
    print("\nError: Ollama daemon is not running.")
    print("   Please make sure Ollama is installed and run 'ollama serve' in your terminal.")
    sys.exit(1)

ensure_ollama_running()

current_dir = os.path.dirname(os.path.abspath(__file__))

print("\n" + "=" * 70)

print("STEP 1/3: Character Personality Extraction")

print("=" * 70)

result = subprocess.run([sys.executable, "character_extractor.py"], cwd=current_dir)

if result.returncode != 0:

    print("Error: Character extraction failed")

    sys.exit(1)

                                                    

print("\n" + "=" * 70)

print("STEP 2/3: Story Setting Extraction")

print("=" * 70)

result = subprocess.run([sys.executable, "story_extractor.py"], cwd=current_dir)

if result.returncode != 0:

    print("Error: Story extraction failed")

    sys.exit(1)

                                                                   

print("\n" + "=" * 70)

print("STEP 3/3: Fusion Engine")

print("=" * 70)

result = subprocess.run([sys.executable, "fusion_engine.py"], cwd=current_dir)

if result.returncode != 0:

    print("Error: Fusion failed")

    sys.exit(1)

print("\n" + "=" * 70)

print("FULL PIPELINE COMPLETE")

print("=" * 70)

print("\nOutput files:")

print("   - ../outputs/fusion/character_personality.json")

print("   - ../outputs/fusion/story_setting.json")

print("   - ../outputs/fusion/fusion_complete.json")

print("   - ../outputs/fusion/sdxl_prompt.json")

print("\nNext: Run SDXL generation with:")

print("   python ../sdxl_code/generate_character.py")

print("=" * 70)

