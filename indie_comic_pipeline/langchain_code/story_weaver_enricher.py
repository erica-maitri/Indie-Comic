"""
STORY-WEAVER ENRICHER ENGINE
==============================
Reads Story-Weaver's story_dynamic.json and enriches every panel with:
  - Main character: name, description, emotion, mood, expression, action, clothing
  - Side characters: minimum 3 per panel (invented if scene context is sparse)
  - Scenery: detailed environment description
  - Augmented SDXL prompt: assembled from all above + style config

No character reference image required. This replaces the old:
  character_extractor.py + story_extractor.py + fusion_engine.py + emotion_recognition_engine.py
when running in Story-Weaver mode.

Output: outputs/fusion/enriched_storyboard.json
"""

import json
import re
import sys
import os
import ast
import argparse

if sys.stdout.encoding != 'utf-8':
    try:
        reconfigure = getattr(sys.stdout, 'reconfigure', None)
        if reconfigure:
            reconfigure(encoding='utf-8')
    except:
        pass

if sys.stderr.encoding != 'utf-8':
    try:
        reconfigure = getattr(sys.stderr, 'reconfigure', None)
        if reconfigure:
            reconfigure(encoding='utf-8')
    except:
        pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_helper import load_settings, get_output_path

# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Enrich Story-Weaver panels with full cast and scenery details.")
parser.add_argument(
    "--input",
    type=str,
    default=None,
    help="Path to Story-Weaver story_dynamic.json (default: from settings or ../Story-Weaver/story_dynamic.json)"
)
parser.add_argument(
    "--character",
    type=str,
    default=None,
    help="Main character name (default: from settings or 'Wanderer')"
)
parser.add_argument(
    "--world",
    type=str,
    default=None,
    help="Story world/setting name (default: from settings or 'The Abstract')"
)
parser.add_argument(
    "--min-side-chars",
    type=int,
    default=None,
    help="Minimum number of side characters per panel (default: from settings or 3)"
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load settings
# ---------------------------------------------------------------------------

settings = load_settings()
fusion_dir = settings.get("outputs", {}).get("fusion_dir", "outputs/fusion")
langchain_settings = settings.get("langchain", {})
sw_settings = settings.get("story_weaver", {})

input_path = args.input or sw_settings.get("input_path", "../Story-Weaver/story_dynamic.json")
character_name = args.character or sw_settings.get("character_name", "Wanderer")
story_world = args.world or sw_settings.get("story_world", "The Abstract")
min_side_chars = args.min_side_chars or sw_settings.get("min_side_characters", 3)

# ---------------------------------------------------------------------------
# JSON Repair Utility (handles LLM truncated outputs)
# ---------------------------------------------------------------------------

def repair_truncated_json(json_str: str) -> str:
    json_str = json_str.strip()
    if not json_str:
        return "{}"

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

    repaired = repaired.strip()
    while stack:
        open_char = stack.pop()
        close_char = '}' if open_char == '{' else ']'
        repaired = repaired.rstrip()
        if repaired.endswith(','):
            repaired = repaired[:-1].rstrip()
        repaired += close_char

    return repaired


def parse_llm_json(response_text: str) -> dict:
    first_brace = response_text.find('{')
    if first_brace == -1:
        raise ValueError("No JSON block found in LLM response")

    cleaned = repair_truncated_json(response_text[first_brace:])
    cleaned = re.sub(r',\s*\}', '}', cleaned)
    cleaned = re.sub(r',\s*\]', ']', cleaned)

    try:
        return json.loads(cleaned)
    except Exception as e:
        try:
            python_style = cleaned.replace("true", "True").replace("false", "False").replace("null", "None")
            return ast.literal_eval(python_style)
        except Exception:
            raise e


# ---------------------------------------------------------------------------
# Ollama Connection
# ---------------------------------------------------------------------------

def ensure_ollama_running(ollama_url: str) -> bool:
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

    print("⚠️ Ollama not running. Attempting to start...")
    try:
        if sys.platform == "win32":
            CREATE_NO_WINDOW = 0x08000000
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
        else:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        for attempt in range(15):
            time.sleep(1)
            if check_port():
                print("✅ Ollama started successfully!")
                return True
            print(f"   Waiting for Ollama... ({attempt+1}/15)")
    except Exception as e:
        print(f"❌ Failed to auto-start Ollama: {e}")

    print("\n" + "!" * 70)
    print("CRITICAL: Ollama daemon not running. Run 'ollama serve' first.")
    print("!" * 70 + "\n")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Build LLM System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert comic book character director and visual world-builder.

Given a comic panel's visual description, dialogue, emotion beat, and motion/action, \
you will produce a COMPLETE character manifest for that panel.

Requirements:
1. ONE main character (the protagonist) — give them a vivid but consistent appearance.
2. EXACTLY {min_side_chars} or more side characters — each with distinct roles, appearances, and reactions. \
   If the scene doesn't naturally have {min_side_chars}+ side characters, INVENT contextually appropriate ones \
   (e.g., a shopkeeper in the background, a passerby, a friend, a shadow figure).
3. ONE detailed scenery description combining all environmental, lighting, weather, and atmospheric cues.
4. All character fields must be filled — never leave expression, action, or clothing empty.

IMPORTANT: Do NOT name emotions literally (no "sad", "angry", "happy"). \
Express them through body language and physical description only.

Respond ONLY with valid JSON in this exact structure. No markdown. No explanation:
{{
  "main_character": {{
    "name": "{character_name}",
    "description": "one-sentence physical appearance",
    "emotion": "one descriptive word avoiding literal emotion names",
    "mood": "one word describing inner state",
    "expression": "detailed face and body expression trigger (3-6 descriptive phrases)",
    "action": "what main character is physically doing right now",
    "clothing": "clothing description matching world/mood"
  }},
  "side_characters": [
    {{
      "name": "Character Name",
      "description": "one-sentence physical appearance",
      "emotion": "one descriptive word",
      "mood": "one word",
      "expression": "detailed face and body expression trigger",
      "action": "what this character is doing",
      "clothing": "clothing description"
    }}
  ],
  "scenery": "rich multi-sentence environment description covering location, lighting, atmosphere, weather, time of day, recurring motif"
}}"""


def build_user_prompt(
    panel: dict,
    character_name: str,
    story_world: str,
    recurring_motif: str,
    mood_journey: str,
    panel_history: list,
    min_side_chars: int
) -> str:
    history_str = "No previous panels in this scene."
    if panel_history:
        history_str = " | ".join(panel_history[-3:])  # last 3 panels for context

    return (
        f"Story World: {story_world}\n"
        f"Recurring Visual Motif: {recurring_motif}\n"
        f"Overall Mood Journey: {mood_journey}\n\n"
        f"PREVIOUS PANEL EMOTIONS: {history_str}\n\n"
        f"CURRENT PANEL #{panel.get('panel', '?')}:\n"
        f"  Visual: {panel.get('visual', '')}\n"
        f"  Dialogue: {panel.get('dialogue', '...')}\n"
        f"  Emotion Beat: {panel.get('emotion_beat', 'neutral')}\n"
        f"  Motion: {panel.get('motion', '')}\n\n"
        f"Main character name: {character_name}\n"
        f"Minimum side characters required: {min_side_chars}\n\n"
        "Write the JSON character manifest now:"
    )


def _safe_string(val) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        return ", ".join(_safe_string(item) for item in val if item is not None)
    if isinstance(val, dict):
        return ", ".join(_safe_string(v) for v in val.values() if v is not None)
    return str(val)


def normalize_manifest(data):
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            if k in ("main_character", "side_characters"):
                new_dict[k] = normalize_manifest(v)
            else:
                new_dict[k] = _safe_string(v)
        return new_dict
    elif isinstance(data, list):
        return [normalize_manifest(item) for item in data]
    else:
        return _safe_string(data)


# ---------------------------------------------------------------------------
# Prompt Assembler: Build augmented SDXL prompt from enriched panel data
# ---------------------------------------------------------------------------

def build_augmented_prompt(
    panel_data: dict,
    enriched: dict,
    recurring_motif: str,
    style_settings: dict,
    trigger_words: str = ""
) -> str:
    """
    Assembles the final text-to-image prompt from:
    - Style terms
    - Panel visual + motion description
    - Main character: description, expression, action, clothing
    - Each side character: description, expression, action
    - Scenery
    - Recurring motif
    - LoRA trigger words (if any)
    """
    style_terms = ", ".join(style_settings.get("positive_terms", [
        "indie comic style illustration",
        "clean minimalist line art",
        "flat color palette",
        "crisp continuous outlines",
        "cel-shaded with no gradients"
    ]))

    main = enriched.get("main_character", {})
    sides = enriched.get("side_characters", [])
    scenery = _safe_string(enriched.get("scenery", ""))

    raw_parts = []

    # Style prefix
    raw_parts.extend([p.strip() for p in style_terms.split(",") if p.strip()])

    # Panel visual context
    visual = panel_data.get("visual", "")
    motion = panel_data.get("motion", "")
    if visual:
        raw_parts.extend([p.strip() for p in visual.split(",") if p.strip()])
    if motion:
        raw_parts.extend([p.strip() for p in motion.split(",") if p.strip()])

    # Main character
    mc_name = _safe_string(main.get("name", "protagonist"))
    mc_desc = _safe_string(main.get("description", ""))
    mc_expr = _safe_string(main.get("expression", ""))
    mc_action = _safe_string(main.get("action", ""))
    mc_clothing = _safe_string(main.get("clothing", ""))

    mc_parts = [mc_name]
    if mc_desc:
        mc_parts.append(mc_desc)
    if mc_action:
        mc_parts.append(mc_action)
    if mc_expr:
        mc_parts.append(mc_expr)
    if mc_clothing:
        mc_parts.append(mc_clothing)
    raw_parts.extend([p.strip() for p in ", ".join(mc_parts).split(",") if p.strip()])

    # Side characters (brief description + expression)
    for sc in sides:
        sc_name = _safe_string(sc.get("name", "character"))
        sc_desc = _safe_string(sc.get("description", ""))
        sc_expr = _safe_string(sc.get("expression", ""))
        sc_action = _safe_string(sc.get("action", ""))
        sc_clothing = _safe_string(sc.get("clothing", ""))
        sc_parts = [sc_name]
        if sc_desc:
            sc_parts.append(sc_desc)
        if sc_action:
            sc_parts.append(sc_action)
        if sc_expr:
            sc_parts.append(sc_expr)
        if sc_clothing:
            sc_parts.append(sc_clothing)
        raw_parts.extend([p.strip() for p in ", ".join(sc_parts).split(",") if p.strip()])

    # Scenery and recurring motif
    if scenery:
        raw_parts.extend([p.strip() for p in scenery.split(",") if p.strip()])
    if recurring_motif:
        raw_parts.append(f"recurring motif: {recurring_motif}")

    # Case-insensitive deduplication
    seen = set()
    final_parts = []
    for part in raw_parts:
        norm = re.sub(r'\s+', ' ', part.lower().strip()).rstrip('.')
        if norm and norm not in seen:
            seen.add(norm)
            final_parts.append(part.rstrip('.'))

    prompt = ", ".join(final_parts)

    # Append LoRA trigger words
    if trigger_words:
        prompt = f"{prompt}, {trigger_words}"

    return prompt


# ---------------------------------------------------------------------------
# Main Enrichment Logic
# ---------------------------------------------------------------------------

def run_enricher():
    print("=" * 70)
    print("STORY-WEAVER ENRICHER ENGINE")
    print("=" * 70)

    # Resolve input path relative to project root
    from utils.config_helper import get_project_root
    project_root = get_project_root()

    # Try to resolve relative paths
    resolved_input = input_path
    if not os.path.isabs(input_path):
        resolved_input = os.path.normpath(os.path.join(project_root, input_path))

    if not os.path.exists(resolved_input):
        # Also try relative to cwd
        cwd_attempt = os.path.normpath(os.path.join(os.getcwd(), input_path))
        if os.path.exists(cwd_attempt):
            resolved_input = cwd_attempt
        else:
            print(f"Error: Story-Weaver JSON not found at: {resolved_input}")
            print(f"Also tried: {cwd_attempt}")
            print("Please provide --input <path_to_story_dynamic.json>")
            sys.exit(1)

    print(f"\n[+] Reading Story-Weaver output from: {resolved_input}")
    with open(resolved_input, "r", encoding="utf-8") as f:
        weaver_data = json.load(f)

    recurring_motif = weaver_data.get("recurring_motif", "A soft glowing object")
    mood_journey = weaver_data.get("mood_journey", "An emotional journey")
    raw_panels = weaver_data.get("panels", [])

    if not raw_panels:
        print("Error: No panels found in Story-Weaver JSON.")
        sys.exit(1)

    print(f"[+] Found {len(raw_panels)} panels")
    print(f"[+] Recurring motif: {recurring_motif}")
    print(f"[+] Mood journey: {mood_journey}")
    print(f"[+] Character: {character_name} in {story_world}")
    print(f"[+] Min side characters: {min_side_chars}")

    # Connect to Ollama
    ollama_url = langchain_settings.get("ollama_url", "http://localhost:11434")
    ensure_ollama_running(ollama_url)

    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import SystemMessage

    llm = ChatOllama(
        model=langchain_settings.get("model", "llama3.2"),
        temperature=0.25,
        base_url=ollama_url
    )

    system_content = SYSTEM_PROMPT.format(
        character_name=character_name,
        min_side_chars=min_side_chars
    )

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_content),
        ("human", "{user_input}")
    ])
    chain = prompt_template | llm | StrOutputParser()

    # Style settings for prompt assembly
    style_settings = settings.get("style", {})
    lora_settings = settings.get("models", {}).get("lora", {})
    trigger_words = lora_settings.get("trigger_words", "")

    # -----------------------------------------------------------------------
    # Enrich each panel
    # -----------------------------------------------------------------------
    enriched_panels = []
    panel_history = []  # tracks short emotion summaries for history context

    for panel in raw_panels:
        panel_num = panel.get("panel", len(enriched_panels) + 1)
        print(f"\n{'─' * 60}")
        print(f"  Enriching Panel {panel_num}: \"{panel.get('visual', '')[:60]}...\"")

        user_msg = build_user_prompt(
            panel=panel,
            character_name=character_name,
            story_world=story_world,
            recurring_motif=recurring_motif,
            mood_journey=mood_journey,
            panel_history=panel_history,
            min_side_chars=min_side_chars
        )

        enriched = None
        for attempt in range(1, 4):
            try:
                response = chain.invoke({"user_input": user_msg})
                enriched = parse_llm_json(response)

                # Validate minimum side characters
                side_chars = enriched.get("side_characters", [])
                if len(side_chars) < min_side_chars:
                    print(f"    ⚠️ Only {len(side_chars)} side chars returned (need {min_side_chars}). Retrying attempt {attempt+1}...")
                    if attempt < 3:
                        continue

                # Ensure main character name is correct
                if "main_character" in enriched:
                    enriched["main_character"]["name"] = character_name

                break
            except Exception as e:
                print(f"    ⚠️ Attempt {attempt} failed: {e}")
                if attempt == 3:
                    print(f"    Using fallback for Panel {panel_num}")
                    enriched = _fallback_enrichment(panel, character_name, story_world, min_side_chars)

        # Normalize the manifest so that fields are clean strings
        enriched = normalize_manifest(enriched)
        if not isinstance(enriched, dict):
            enriched = {}

        # Validate + ensure min side chars with fallback padding
        side_chars = enriched.get("side_characters", [])
        while len(side_chars) < min_side_chars:
            idx = len(side_chars) + 1
            side_chars.append({
                "name": f"Passerby {idx}",
                "description": f"an indistinct figure in the background",
                "emotion": "neutral",
                "mood": "distant",
                "expression": "face turned away, shoulders hunched, unremarkable posture",
                "action": f"moving through the scene, unaware of the main events",
                "clothing": "plain, weather-appropriate attire"
            })
        enriched["side_characters"] = side_chars

        # Build augmented SDXL prompt
        augmented_prompt = build_augmented_prompt(
            panel_data=panel,
            enriched=enriched,
            recurring_motif=recurring_motif,
            style_settings=style_settings,
            trigger_words=trigger_words
        )

        # Track character emotions for history
        mc = enriched.get("main_character", {})
        history_entry = f"Panel {panel_num}: {character_name} ({_safe_string(mc.get('emotion','?'))}/{_safe_string(mc.get('mood','?'))})"
        for sc in enriched.get("side_characters", [])[:2]:
            history_entry += f", {_safe_string(sc.get('name','?'))} ({_safe_string(sc.get('emotion','?'))})"
        panel_history.append(history_entry)

        # Assemble enriched panel record
        enriched_record = {
            "panel_number": panel_num,
            "panel_original": panel,
            "main_character": enriched.get("main_character", {}),
            "side_characters": enriched.get("side_characters", []),
            "scenery": enriched.get("scenery", story_world),
            "recurring_motif": recurring_motif,
            "augmented_prompt": augmented_prompt
        }

        # Print summary
        mc_info = enriched.get("main_character", {})
        print(f"    ✓ Main: {mc_info.get('name')} — {mc_info.get('expression', '')[:50]}...")
        for sc in enriched.get("side_characters", []):
            print(f"    ✓ Side: {sc.get('name')} ({sc.get('emotion')}) — {sc.get('action', '')[:40]}...")
        print(f"    ✓ Scenery: {enriched.get('scenery', '')[:60]}...")
        print(f"    ✓ Prompt ({len(augmented_prompt)} chars): {augmented_prompt[:80]}...")

        enriched_panels.append(enriched_record)

    # -----------------------------------------------------------------------
    # Group panels into pages (4 panels per page)
    # -----------------------------------------------------------------------
    pages = []
    total_pages = max(1, (len(enriched_panels) + 3) // 4)

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page_start = page_idx * 4
        page_panels = enriched_panels[page_start:page_start + 4]

        pages.append({
            "page_number": page_num,
            "location": story_world,
            "mood_journey": mood_journey,
            "recurring_motif": recurring_motif,
            "panels_detail": page_panels
        })

    # -----------------------------------------------------------------------
    # Save enriched_storyboard.json
    # -----------------------------------------------------------------------
    output_data = {
        "mode": "story_weaver",
        "character_name": character_name,
        "story_world": story_world,
        "recurring_motif": recurring_motif,
        "mood_journey": mood_journey,
        "total_panels": len(enriched_panels),
        "total_pages": total_pages,
        "pages": pages
    }

    output_path_full = get_output_path(fusion_dir, "enriched_storyboard.json")
    with open(output_path_full, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"✅ ENRICHMENT COMPLETE!")
    print(f"   Panels enriched : {len(enriched_panels)}")
    print(f"   Pages generated : {total_pages}")
    print(f"   Saved to        : {output_path_full}")
    print("=" * 70)

    return output_data


# ---------------------------------------------------------------------------
# Fallback enrichment (if LLM fails after 3 retries)
# ---------------------------------------------------------------------------

def _fallback_enrichment(panel: dict, character_name: str, story_world: str, min_side_chars: int) -> dict:
    """Generates a minimal but structurally valid enrichment without LLM."""
    emotion_beat = panel.get("emotion_beat", "neutral")
    motion = panel.get("motion", "standing still")
    visual = panel.get("visual", "A quiet scene")

    side_chars = []
    side_char_names = ["Bystander", "Witness", "Shadow Figure", "Onlooker"]
    for i in range(min_side_chars):
        side_chars.append({
            "name": side_char_names[i] if i < len(side_char_names) else f"Figure {i+1}",
            "description": "an indistinct figure present in the scene",
            "emotion": emotion_beat,
            "mood": "observant",
            "expression": "face partially obscured, posture passive, eyes downcast",
            "action": "standing in the background, watching",
            "clothing": "plain, weather-appropriate attire"
        })

    return {
        "main_character": {
            "name": character_name,
            "description": f"the central figure of the scene",
            "emotion": emotion_beat,
            "mood": emotion_beat,
            "expression": f"body language matching {emotion_beat}, posture reflecting inner state",
            "action": motion,
            "clothing": "contextually appropriate attire"
        },
        "side_characters": side_chars,
        "scenery": f"{story_world}. {visual}"
    }


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_enricher()
