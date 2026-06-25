from indie_comic_pipeline.integrated_pipeline import IntegratedComicPipeline
import json
import os
import sys

# Ensure the core module is loadable
sys.path.insert(0, os.path.abspath("indie_comic_pipeline"))

def run_prompts_only():
    print("Initializing Pipeline (Prompt Generation Only)...")
    # Initialize pipeline with dry_run to avoid loading heavy GPU models
    pipeline = IntegratedComicPipeline(dry_run=True)
    
    # Inputs
    prompt = "A cyberpunk hacker named Kael finding an alien artifact in an alleyway."
    character_name = "Kael"
    story_world = "Neo-Tokyo, 2088"
    panel_count = 4
    style_reference = "Classic American comic book style, Marvel/DC style, high contrast heavy ink lines, bold dramatic coloring"
    character_characteristics = "A tall, scarred man wearing a tattered trenchcoat. He is stoic, rarely speaks, but has an intense, piercing stare."
    story_reference = "Blade Runner meets The Matrix, heavy neo-noir cyberpunk tropes, rebellious underground hackers vs oppressive corporations."
    mood_shifts = ["bored", "curious", "shocked", "terrified"] # Simulating the mood shifts
    
    print("\n--- Phase 0: Story Intake ---")
    story_config = pipeline.story_intake.process_prompt(
        user_prompt=prompt,
        panel_count=panel_count,
        character_name=character_name,
        story_world=story_world,
        style_reference=style_reference,
        character_characteristics=character_characteristics,
        story_reference=story_reference,
        mood_shifts=mood_shifts
    )
    print("\n--- Phase 1: Director Swarm Planning ---")
    pipeline.agent_coordinator.run_planning(story_config)
    
    print("\n" + "="*50)
    print("FINAL SCENE GRAPH (Ready for Director Swarm & Diffusion)")
    print("="*50)
    for panel_id in range(1, pipeline.memory.total_panels + 1):
        context = pipeline.agent_coordinator.get_generation_context(panel_id)
        scene_graph = context.get("scene_graph", {})
        print(f"\n[ PANEL {panel_id} SCENE GRAPH ]")
        print(json.dumps(scene_graph, indent=2))
        print("-" * 50)

if __name__ == "__main__":
    run_prompts_only()
