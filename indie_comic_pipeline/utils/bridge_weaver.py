"""
BRIDGE SCRIPT: STORY-WEAVER to INDIE-COMIC-PIPELINE
Translates the dynamic mood-based JSON from Story-Weaver into the precise
paginated storyboard JSON required by the Indie Comic Pipeline.

Modes:
  Legacy mode (default): Converts panels → fusion_complete.json + sdxl_prompt.json
  Enriched mode (--enrich): Calls story_weaver_enricher.py to produce
    enriched_storyboard.json with full cast, emotions, expressions, and
    assembled SDXL prompts — NO character reference image needed.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

def pad_panels(panels):
    """Ensure panels are a multiple of 4 by duplicating the final panel as a fade-out if necessary."""
    target_len = ((len(panels) + 3) // 4) * 4
    while len(panels) < target_len:
        last_panel = panels[-1] if len(panels) > 0 else {
            "visual": "A quiet, dark scene.",
            "dialogue": "...",
            "emotion_beat": "stillness",
            "motion": "fading out"
        }
        panels.append({
            "panel": len(panels) + 1,
            "visual": last_panel["visual"] + " (Transitioning, fading).",
            "dialogue": "...",
            "emotion_beat": "fade",
            "motion": "slow fade"
        })
    return panels

def bridge_json_enrich(input_path, output_dir, character_name, story_world, min_side_chars):
    """
    Enriched mode: calls story_weaver_enricher.py via subprocess.
    Produces enriched_storyboard.json — no reference image needed.
    """
    print("[*] Running Story-Weaver Enricher (reference-free multi-character mode)...")

    # Resolve the enricher script path (sibling of this file → ../langchain_code/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_root = os.path.dirname(current_dir)
    enricher_script = os.path.join(pipeline_root, "langchain_code", "story_weaver_enricher.py")

    if not os.path.exists(enricher_script):
        print(f"Error: Enricher script not found at {enricher_script}")
        sys.exit(1)

    cmd = [
        sys.executable, enricher_script,
        "--input", str(input_path),
        "--character", character_name,
        "--world", story_world,
        "--min-side-chars", str(min_side_chars)
    ]

    result = subprocess.run(cmd, cwd=pipeline_root)
    if result.returncode != 0:
        print("Error: Enricher script failed.")
        sys.exit(1)

    print("[*] Enrichment complete — enriched_storyboard.json is ready.")
    print("\nBRIDGE (ENRICHED MODE) SUCCESSFUL! You can now run panel generation.")


def bridge_json(input_path, output_dir, character_name="Wanderer", story_world="The Abstract"):
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error: Could not find Story-Weaver JSON at {input_file}")
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        try:
            weaver_data = json.load(f)
        except Exception as e:
            print(f"Error reading JSON: {e}")
            sys.exit(1)
            
    # Extract Weaver data
    motif = weaver_data.get("recurring_motif", "A soft, glowing object.")
    journey = weaver_data.get("mood_journey", "A quiet emotional journey.")
    raw_panels = weaver_data.get("panels", [])
    
    if not raw_panels:
        print("Error: No panels found in Story-Weaver output.")
        sys.exit(1)
        
    padded_panels = pad_panels(raw_panels)
    
    # Transform to Comic Pipeline structure
    comic_pages = []
    total_pages = len(padded_panels) // 4
    
    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page_start = page_idx * 4
        page_panels = padded_panels[page_start:page_start+4]
        
        panels_breakdown = []
        dialogue_and_captions = []
        
        for i, p in enumerate(page_panels):
            panel_idx = i + 1
            visual = p.get("visual", "")
            motion = p.get("motion", "")
            emotion = p.get("emotion_beat", "")
            dialogue = p.get("dialogue", "...")
            
            # Construct SDXL Prompt
            prompt = f"Panel {panel_idx}: {visual}. {character_name} is showing {emotion}. {motion}. Recurring motif: {motif}."
            panels_breakdown.append(prompt)
            
            # Construct Dialogue
            # If dialogue doesn't have a speaker, add Character: or Caption:
            if ":" not in dialogue:
                dialogue = f"{character_name}: {dialogue}"
            dialogue_and_captions.append(dialogue)
            
        page_data = {
            "page_number": page_num,
            "location": story_world,
            "narrative_progression": journey,
            "scene_settlement": f"Mood Journey: {journey}",
            "character_expressions": f"{character_name} progressing through emotions.",
            "personality_state": journey,
            "side_characters_present": [],
            "panels_breakdown": panels_breakdown,
            "dialogue_and_captions": dialogue_and_captions
        }
        comic_pages.append(page_data)
        
    # Write fusion_complete.json
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fusion_path = out_dir / "fusion_complete.json"
    with open(fusion_path, "w", encoding="utf-8") as f:
        # Wrap it in the top-level structure required
        output_wrap = {
            "personality": {"character_name": character_name},
            "setting": {"story_name": story_world},
            "fusion": {
                "story_descriptive": journey,
                "character_visual_looks": f"A highly detailed character sheet of {character_name}.",
                "storyboard_10_pages": comic_pages,
                "components": []
            }
        }
        json.dump(output_wrap, f, indent=2)
        
    print(f"[*] Translated {len(raw_panels)} Story-Weaver panels into {total_pages} Comic Pages.")
    print(f"[*] Saved Storyboard to: {fusion_path}")
    
    # Write sdxl_prompt.json (Anchor profile)
    sdxl_prompt_path = out_dir / "sdxl_prompt.json"
    sdxl_data = {
        "positive_prompt": f"A solitary {character_name} going through an emotional journey. Clean comic art style.",
        "negative_prompt": "photorealistic, 3D render, messy lines, gradients, blurry",
        "style": "clean minimalist line art, flat color palette",
        "character_name": character_name,
        "story_world": story_world
    }
    with open(sdxl_prompt_path, "w", encoding="utf-8") as f:
        json.dump(sdxl_data, f, indent=2)
        
    print(f"[*] Saved IP-Adapter Anchor config to: {sdxl_prompt_path}")
    print("\nBRIDGE SUCCESSFUL! You can now run the Image Generation step.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bridge Story Weaver JSON to Indie Comic Pipeline")
    parser.add_argument("--input", type=str, default="../Story-Weaver/story_dynamic.json",
                        help="Path to Story Weaver output JSON")
    parser.add_argument("--output_dir", type=str, default="outputs/fusion",
                        help="Path to save converted pipeline JSONs")
    parser.add_argument("--character", type=str, default="Wanderer",
                        help="Main character name")
    parser.add_argument("--world", type=str, default="The Void",
                        help="Name of the story setting")
    parser.add_argument("--enrich", action="store_true",
                        help="Use enriched mode: LLM builds full cast + prompts, no reference image needed")
    parser.add_argument("--min-side-chars", type=int, default=3,
                        help="Minimum side characters per panel in enriched mode (default: 3)")

    args = parser.parse_args()

    # Change to script directory's parent to ensure outputs/fusion goes to the right place
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_root = os.path.dirname(current_dir)
    os.chdir(pipeline_root)

    if args.enrich:
        bridge_json_enrich(
            input_path=args.input,
            output_dir=args.output_dir,
            character_name=args.character,
            story_world=args.world,
            min_side_chars=args.min_side_chars
        )
    else:
        bridge_json(args.input, args.output_dir, args.character, args.world)
