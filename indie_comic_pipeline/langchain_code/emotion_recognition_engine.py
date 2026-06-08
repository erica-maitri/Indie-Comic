"""
EMOTION RECOGNITION IN CONVERSATION (ERC) ENGINE
Processes the comic storyboard to identify character emotions, maps them to expressions, and builds panel prompts
"""

import json
import re
import sys
import os
import ast

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_helper import load_settings, get_output_path
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def repair_truncated_json(json_str):
    json_str = json_str.strip()
    if not json_str:
        return "{}"
    
    # Find the first '{'
    first_brace = json_str.find('{')
    if first_brace == -1:
        json_str = "{" + json_str
        first_brace = 0
    else:
        json_str = json_str[first_brace:]
        
    stack = []
    in_string = False
    string_char = None
    escaped = False
    
    # Walk through characters to find open braces/brackets/strings
    for i, char in enumerate(json_str):
        if escaped:
            escaped = False
            continue
        if char == '\\':
            if in_string:
                escaped = True
            continue
        if char in ('"', "'"):
            if in_string:
                if char == string_char:
                    in_string = False
                    string_char = None
            else:
                in_string = True
                string_char = char
            continue
        if in_string:
            continue
        if char in ('{', '['):
            stack.append(char)
        elif char in ('}', ']'):
            if stack:
                top = stack[-1]
                if (char == '}' and top == '{') or (char == ']' and top == '['):
                    stack.pop()

    repaired = json_str
    if in_string:
        repaired += (string_char if string_char else '"')
        
    repaired_trimmed = repaired.strip()
    
    # Process stack to close structures
    while stack:
        open_char = stack.pop()
        close_char = '}' if open_char == '{' else ']'
        
        repaired_trimmed = repaired_trimmed.rstrip()
        if repaired_trimmed.endswith(','):
            repaired_trimmed = repaired_trimmed[:-1].rstrip()
            
        repaired_trimmed += close_char
        
    return repaired_trimmed

def parse_llm_response(response_text):
    # Find first brace
    first_brace = response_text.find('{')
    if first_brace == -1:
        raise ValueError("No brace block found in response")
    
    # Run the repair logic on everything from the first '{' onwards
    cleaned = repair_truncated_json(response_text[first_brace:])
    
    # Clean up trailing commas in objects and arrays
    cleaned = re.sub(r',\s*\}', '}', cleaned)
    cleaned = re.sub(r',\s*\]', ']', cleaned)
    
    try:
        return json.loads(cleaned)
    except Exception as e:
        # Fallback to ast.literal_eval
        try:
            python_style = cleaned.replace("true", "True").replace("false", "False").replace("null", "None")
            return ast.literal_eval(python_style)
        except Exception:
            raise e

print("=" * 70)
print("EMOTION RECOGNITION & EXPRESSION PARSER ENGINE")
print("=" * 70)

settings = load_settings()
fusion_dir = settings.get("outputs", {}).get("fusion_dir", "outputs/fusion")
langchain_settings = settings.get("langchain", {})

# Load completed storyboard fusion
fusion_path = get_output_path(fusion_dir, "fusion_complete.json")
if not os.path.exists(fusion_path):
    print(f"Error: Storyboard file not found at: {fusion_path}")
    sys.exit(1)

with open(fusion_path, "r", encoding="utf-8") as f:
    fusion_data = json.load(f)

personality = fusion_data['personality']
setting = fusion_data['setting']
fusion = fusion_data['fusion']
pages = fusion.get("storyboard_10_pages", [])
char_looks = fusion.get("character_visual_looks", "")

print(f"\nProcessing {len(pages)} pages of storyboard...")
print(f"Character adapted looks: {char_looks[:100]}...")

def ensure_ollama_running(ollama_url):
    import socket
    import time
    import subprocess
    from urllib.parse import urlparse

    try:
        parsed = urlparse(ollama_url)
        host = parsed.hostname or 'localhost'
        port = parsed.port or 11434
    except Exception:
        host = 'localhost'
        port = 11434

    def check_port():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        res = sock.connect_ex((host, port))
        sock.close()
        return res == 0

    if check_port():
        print("✅ Ollama server is active and running.")
        return True

    print("⚠️ Ollama server is not running. Attempting to start Ollama automatically...")
    try:
        if sys.platform == "win32":
            CREATE_NO_WINDOW = 0x08000000
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
        else:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        for attempt in range(15):
            time.sleep(1)
            if check_port():
                print("✅ Ollama server started and connected successfully!")
                return True
            print(f"   Waiting for Ollama to initialize... (attempt {attempt+1}/15)")
    except Exception as e:
        print(f"❌ Failed to auto-start Ollama: {e}")

    print("\n" + "!" * 70)
    print("CRITICAL ERROR: Ollama daemon is not running.")
    print("Please make sure Ollama is installed and run 'ollama serve' in your terminal.")
    print("!" * 70 + "\n")
    sys.exit(1)

# Ensure Ollama daemon is running
ollama_url = langchain_settings.get("ollama_url", "http://localhost:11434")
ensure_ollama_running(ollama_url)

# Load LLM connection
print("\nConnecting to local Ollama server...")
llm = ChatOllama(
    model=langchain_settings.get("model", "llama3.2"),
    temperature=0.2, # Lower temperature for stable JSON output
    base_url=ollama_url
)

# Build Prompt Template
system_prompt = """You are an expert comic book narrative director, character psychologist, and visual layout designer.
Your task is to analyze a comic panel's narrative text, dialogue, and captions, and extract:
1. Which characters are active in this specific panel.
2. The primary emotion of each character (e.g. angry, fearful, joyful, sad, surprised, neutral, ecstatic, furious, terrified, curious).
3. The intensity of that emotion (low, medium, high).
4. A highly dramatic, descriptive visual facial expression trigger suited for drawing (e.g. "brows deeply furrowed, teeth gritted in determination, eyes wide with anger" or "crying softly with eyes closed and tears streaming"). Ensure it matches the scene's emotional context and genre (e.g. funny, gothic, tragic, heroic).
5. The core action/posing happening, including the poses and expressions of ALL characters present.
6. The background environment.

CRITICAL INSTRUCTIONS FOR GENRE STYLE & INTENSITY PROGRESSION:
- You must analyze the emotional progression from previous panels. Escalating dialogues should result in escalating intensities and expression triggers.
- Expressions must be vivid, dramatic, and genre-appropriate. For comedy, make them exaggerated or funny; for drama, make them intense, brooding, or poignant.
- Return emotions for EVERY character present in the panel.

Analyze the panel and return a JSON structure matching the example below:

Example Panel Description:
Panel 1: Peter Parker crawls out of the portal, shivering and looking up at the cold wind. Dialogue: 'By the heavens, where am I?'

Example Output:
{{
  "characters_present": ["Peter Parker"],
  "emotions": {{
    "Peter Parker": {{
      "emotion": "fearful",
      "intensity": "high",
      "expression_trigger": "shivering with eyes wide in confusion, mouth slightly open, windswept hair"
    }}
  }},
  "core_action": "Peter Parker crawling on hands and knees, looking up in disbelief",
  "background_env": "foggy desolate grassy hill under dark clouds"
}}

Respond ONLY with a valid JSON block. Do not add any text before or after the JSON payload. Do not use unescaped double quotes inside value strings (use single quotes instead).
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Previous Panel Emotions in this Scene: {history_context}\nPanel Description: {panel_text}\nDialogue and Captions: {dialogue_text}")
])

chain = prompt | llm | StrOutputParser()

annotated_pages = []

for page in pages:
    page_num = page.get("page_number")
    print(f"\nAnalyzing Page {page_num}...")
    
    dialogue_str = " | ".join(page.get("dialogue_and_captions", []))
    panels = page.get("panels_breakdown", [])
    
    previous_emotions_tracker = []
    annotated_panels = []
    
    for idx, panel_text in enumerate(panels):
        print(f"  -> Panel {idx+1}: {panel_text[:60]}...")
        
        # Build history context string
        if previous_emotions_tracker:
            history_str = " | ".join(previous_emotions_tracker)
        else:
            history_str = "No previous panels in this scene yet."
            
        try:
            response = chain.invoke({
                "panel_text": panel_text,
                "dialogue_text": dialogue_str,
                "history_context": history_str
            })
            
            panel_emotions = parse_llm_response(response)
            
            # Record current panel emotions for next panels' context
            curr_emotions = []
            if "emotions" in panel_emotions and isinstance(panel_emotions["emotions"], dict):
                for name, emo_data in panel_emotions["emotions"].items():
                    if isinstance(emo_data, dict):
                        curr_emotions.append(f"{name}: {emo_data.get('emotion', 'neutral')} ({emo_data.get('intensity', 'medium')})")
            if curr_emotions:
                previous_emotions_tracker.append(f"Panel {idx+1}: " + ", ".join(curr_emotions))
                
        except Exception as e:
            print(f"    Warning: LLM analysis failed for Panel {idx+1} ({e}). Using default neutral mapping.")
            # Fallback
            panel_emotions = {
                "characters_present": [personality['character_name']],
                "emotions": {
                    personality['character_name']: {
                        "emotion": "neutral",
                        "intensity": "medium",
                        "expression_trigger": "neutral expression"
                    }
                },
                "core_action": panel_text,
                "background_env": setting['location']
            }
            previous_emotions_tracker.append(f"Panel {idx+1}: {personality['character_name']}: neutral (medium)")
            
        # Build augmented prompt for this panel
        # Format style prefix
        style_settings = settings.get("style", {})
        style_desc = ", ".join(style_settings.get("positive_terms", [
            "clean minimalist line art", "flat color palette", "crisp continuous outlines", "cel-shaded with no gradients"
        ]))
        
        # Build prompt sections
        char_name = personality['character_name']
        char_prompt = f"consistent {char_name}, {char_looks}"
        
        # Inject expression trigger
        expr = None
        if "emotions" in panel_emotions and isinstance(panel_emotions["emotions"], dict):
            char_emotion = panel_emotions["emotions"].get(char_name)
            if isinstance(char_emotion, dict):
                expr = char_emotion.get("expression_trigger")
            elif isinstance(char_emotion, str):
                expr = char_emotion
        
        if expr:
            char_prompt = f"{char_prompt}, {expr}"
            
        # Environment background
        env = panel_emotions.get("background_env", setting['location'])
        action = panel_emotions.get("core_action", "")
        
        # Construct final SDXL prompt
        augmented_prompt = f"indie comic style illustration, {style_desc}, {action}, {char_prompt}, in {env}, {setting['lighting']}, {setting['weather']}"
        # Keep under standard SDXL character limit
        augmented_prompt = ", ".join([p.strip() for p in augmented_prompt.split(",") if p.strip()])
        
        panel_emotions["augmented_prompt"] = augmented_prompt
        panel_emotions["panel_text"] = panel_text
        panel_emotions["panel_number"] = idx + 1
        
        annotated_panels.append(panel_emotions)
        
    page_copy = page.copy()
    page_copy["panels_detail"] = annotated_panels
    annotated_pages.append(page_copy)

# Save annotated storyboard
output_data = {
    "personality": personality,
    "setting": setting,
    "character_visual_looks": char_looks,
    "storyboard_with_emotions": annotated_pages
}

output_path = get_output_path(fusion_dir, "storyboard_with_emotions.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

print("\n" + "=" * 70)
print(f"✅ STORYBOARD ANNOTATION COMPLETE! Saved to: {output_path}")
print("=" * 70)
