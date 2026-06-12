"""
CHARACTER + STORY FUSION ENGINE (STORYBOARDER)
Combines personality and setting into visual character designs and a 4-panel storyboard script
"""

import json

import re

import sys

import os

import numpy as np

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

from langchain_ollama import ChatOllama

from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate

print("=" * 70)

print("CHARACTER + STORY FUSION ENGINE (STORYBOARDER) - Fusing identity with space")

print("=" * 70)

settings = load_settings()

fusion_dir = settings.get("outputs", {}).get("fusion_dir", "outputs/fusion")

langchain_settings = settings.get("langchain", {})

print("\nLoading extracted parameters...")

char_path = get_output_path(fusion_dir, "character_personality.json")

story_path = get_output_path(fusion_dir, "story_setting.json")

with open(char_path, "r", encoding="utf-8") as f:

    personality = json.load(f)

with open(story_path, "r", encoding="utf-8") as f:

    setting = json.load(f)

print(f"Loaded configurations for: {personality['character_name']} + {setting['story_name']}")

print("\nConnecting to local Ollama server...")

llm = ChatOllama(

    model=langchain_settings.get("model", "llama3.2"),

    temperature=langchain_settings.get("temperature", 0.4),

    base_url=langchain_settings.get("ollama_url", "http://localhost:11434")

)

print(f"Connected to Ollama: {llm.model}")

def get_cosine_similarity(v1, v2):

    dot = np.dot(v1, v2)

    norm1 = np.linalg.norm(v1)

    norm2 = np.linalg.norm(v2)

    if norm1 > 0 and norm2 > 0:

        return dot / (norm1 * norm2)

    return 0.0

def run_vector_persistence_analysis(candidates, pages, model_name, base_url):

    print("\nRunning vector space persistence analysis...")

    try:

        from langchain_ollama import OllamaEmbeddings

        embeddings_model = OllamaEmbeddings(

            model=model_name,

            base_url=base_url

        )

        

                             

        candidate_texts = [f"{c['name']} {c['description']}" for c in candidates]

        print(f"Embedding {len(candidates)} candidate visual elements...")

        candidate_vecs = embeddings_model.embed_documents(candidate_texts)

        

                        

        page_texts = [f"{p['location']} {p['narrative_progression']}" for p in pages]

        print("Embedding 10 storyboard pages...")

        page_vecs = embeddings_model.embed_documents(page_texts)

        

                                                                        

        print("Clustering and de-duplicating candidates...")

        unique_indices = []

        for i in range(len(candidates)):

            is_duplicate = False

            for uj in unique_indices:

                sim = get_cosine_similarity(candidate_vecs[i], candidate_vecs[uj])

                if sim > 0.85:

                    is_duplicate = True

                    break

            if not is_duplicate:

                unique_indices.append(i)

                

        unique_candidates = [candidates[i] for i in unique_indices]

        unique_vecs = [candidate_vecs[i] for i in unique_indices]

        

                                           

        scored_candidates = []

        for i, c in enumerate(unique_candidates):

            c_vec = unique_vecs[i]

            similarities = []

            for p_vec in page_vecs:

                sim = get_cosine_similarity(c_vec, p_vec)

                similarities.append(sim)

                                                                      

            persistence_score = np.mean(similarities)

            scored_candidates.append((c, persistence_score))

            

                                              

        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        print("\nPersistent Visual Components Ranked by Vector Space Score:")

        for idx, (c, score) in enumerate(scored_candidates):

            print(f"  {idx+1}. {c['name']} ({c['type']}) - Persistence: {score:.4f}")

            

                                                                                           

        selected_candidates = []

        for c, score in scored_candidates:

                                                                      

            if len(selected_candidates) < 3 or (score >= 0.22 and len(selected_candidates) < 5):

                selected_candidates.append(c)

                

                                          

        for idx, c in enumerate(selected_candidates):

            c['component_number'] = idx + 1

            

        return selected_candidates

        

    except Exception as e:

        print(f"Warning: Vector analysis failed ({e}). Falling back to LLM raw candidates order.")

        seen = set()

        selected_candidates = []

        for c in candidates:

            if c['name'].lower() not in seen:

                seen.add(c['name'].lower())

                selected_candidates.append(c)

            if len(selected_candidates) >= 4:

                break

        if len(selected_candidates) < 3:

            selected_candidates = candidates[:3]

        for idx, c in enumerate(selected_candidates):

            c['component_number'] = idx + 1

        return selected_candidates

def fuse_character_and_story(personality, setting):
    """Fuse character and story into a 10-page crossover storyboard with 4 panels per page and visual components list"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert comic book narrative director, storyboard artist, and visual asset planner.
        
        Your task is to perform two tasks:
        1. Write a highly dramatic, cinematic crossover story in descriptive narrative form where the main character is pulled into the multiverse of the target story. You must fit the main character's ARC TURN directly into the target story world's narrative arc. Write a highly dramatic multiverse crossover where their psychological shift and character development mirror or collide with the target story's themes.
        2. Create a 10-page storyboard layout sequence. Each page represents a comic page with EXACTLY 4 panels (which allows rendering scripts to compile them into a 2x2 grid layout). For each page, define:
           - Location: The specific setting.
           - Narrative Progression: The page's story beat.
           - Scene Settlement: A highly descriptive depiction of the environment at its absolute best (lighting, weather, vibes, textures, objects' positions).
           - Character Expressions: Specific facial expressions and postures of all active characters in play.
           - Personality/Emotion State: The main character's internal emotional beat.
           - Side Characters Present: List of side characters active on this page.
           - Panels Breakdown: Array of EXACTLY 4 strings. Each string must be a highly detailed visual prompt for SDXL, describing:
             - The active characters, their poses, reactions, emotional states, and what they are wearing (designed adaptively for this crossover).
             - The environment details, exact locations of objects in the scene, and atmospheric lighting.
           - Dialogue and Captions: Array of EXACTLY 4 strings, where each string represents the dialogue or caption corresponding to that specific panel (from Panel 1 to Panel 4). Write dialogues with famous quotes/dialogues and vibes matching the original style and cadence of the target story.
        
        ARC FUSION EXAMPLE (FITTING CHARACTER ARC TURN INTO STORY WORLD):
        Target Story: Wuthering Heights (Themes of obsession, grief, and destructive isolation)
        Character: Spider-Man (Arc Turn: Transforming guilt over Uncle Ben's death into hyper-responsible, self-sacrificing protection of others)
        Arc Fusion Integration: Spider-Man falls into the dark, windswept Yorkshire Moors of Wuthering Heights. Instead of fighting supervillains, his emotional struggle mirrors Heathcliff's grief but highlights their divergent arc turns. Heathcliff channels his grief into vengeful obsession and cruelty, whereas Spider-Man struggles with the desire to escape back to Manhattan but realizes he must protect the vulnerable inhabitants of Wuthering Heights, demonstrating his arc turn of selfless responsibility in a setting dominated by selfishness.
        
        COLOR FUSION MANDATE: Explicitly fuse the main character's favorite theme/mood colors with the target setting's color palette (e.g., neon pink/cyan for Cyberpunk or gothic dark tones for Noir). Fictional clothing must be designed adaptively (no default costumes!).
        
        Follow this exact JSON structure for the output:
        {{
            "story_descriptive": "A highly dramatic, cinematic descriptive story of the crossover event in narrative form.",
            "character_visual_looks": "Adaptive visual clothing style of the crossover character (excluding generic names, focusing on style, mood, and clothing details).",
            "storyboard_10_pages": [
                {{
                    "page_number": 1,
                    "location": "Yorkshire Moors Boundary",
                    "narrative_progression": "The main character is pulled through a temporal rift onto the windswept Moors.",
                    "scene_settlement": "Desolate windswept moors under a massive dark purple stormy sky, wild purple heather rustling violently, dim candlelit stone gate posts nearby.",
                    "character_expressions": "Spider-Man looking down at his hands with eyes wide in disorientation, jaw slightly clenched.",
                    "personality_state": "Confused and disoriented",
                    "side_characters_present": ["Heathcliff"],
                    "panels_breakdown": [
                        "Panel 1: Spider-Man in a high-collared Victorian deep crimson coat, kneeling on wet dark grass, looking up at the sky in confusion as rain begins to fall.",
                        "Panel 2: Spider-Man standing up, dusting off his wet crimson coat, looking at the distant stone manor house Wuthering Heights outlined against the dark horizon.",
                        "Panel 3: A brooding figure, Heathcliff, wearing a dark woolen cloak, stands near a decaying stone fence, staring intensely at Spider-Man with furrowed brows.",
                        "Panel 4: Spider-Man approaches Heathcliff cautiously, raising a hand in greeting, while Heathcliff remains motionless, his expression suspicious and cold."
                    ],
                    "dialogue_and_captions": [
                        "Caption: I was swinging through Manhattan, and then... nothing but wind and heather.",
                        "Spider-Man: Where is the skyline? What is this place?",
                        "Caption: From the shadows of the fence, a dark eyes watch the newcomer.",
                        "Heathcliff: You are not from these moors. Speak, or be gone."
                    ]
                }}
            ],
            "candidate_visual_elements": [
                {{
                    "name": "Main Character Crossover",
                    "type": "character",
                    "description": "Main character in adapted crossover style.",
                    "sdxl_prompt": "indie comic style illustration, clean minimalist line art, flat color palette, main character in adapted style, consistent"
                }}
            ]
        }}
        
        Respond ONLY with valid JSON matching the exact schema above. Do not add any text before or after the JSON:
        """),
        ("human", """
        CHARACTER: {character_name}
        Core Personality: {personality_traits}
        Personality Development: {personality_development_stories}
        Nature: {nature}
        Characteristics Types: {characteristics_types}
        Favorite Dialogue: {favorite_dialogue}
        Arc Turn: {arc_turn}
        
        STORY WORLD: {story_name}
        Genre: {genre}
        Environment Description: {environment_description}
        Theme Color Associated: {theme_color_associated}
        Vibes: {vibes}
        Key Side Characters: {key_side_characters}
        """)
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke({
            "character_name": personality['character_name'],
            "personality_traits": ', '.join(personality['core_personality_traits']),
            "personality_development_stories": personality['personality_development_stories'],
            "nature": personality['nature'],
            "characteristics_types": personality['characteristics_types'],
            "favorite_dialogue": personality['favorite_dialogue'],
            "arc_turn": personality.get('arc_turn', 'Undergoes a key transition in their moral or psychological worldview.'),
            "story_name": setting['story_name'],
            "genre": setting['genre'],
            "environment_description": setting['environment_description'],
            "theme_color_associated": ', '.join(setting['theme_color_associated']),
            "vibes": setting['vibes'],
            "key_side_characters": json.dumps(setting['key_side_characters'])
        })

        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            fusion = json.loads(json_match.group())
            return fusion
        else:
            raise ValueError("No JSON found")

    except Exception as e:
        print(f"Error during LLM fusion execution: {e}")
        return fallback_fusion(personality, setting)

def fallback_fusion(personality, setting):
    char_name = personality['character_name']
    story_name = setting['story_name']
    loc = setting['environment_description']
    looks = f"{char_name} in adapted clothing matching the {setting['vibes']} vibes of {story_name}."
    
    pages = []
    for page_num in range(1, 11):
        pages.append({
            "page_number": page_num,
            "location": f"{story_name} Outskirts",
            "narrative_progression": f"{char_name} navigates the environment of {story_name}, facing challenges.",
            "scene_settlement": f"Detailed environment description of {loc} during scene {page_num}.",
            "character_expressions": f"{char_name} showing deep focus and emotional response appropriate for scene {page_num}.",
            "personality_state": "Focused and determined",
            "side_characters_present": [],
            "panels_breakdown": [
                f"Panel 1: {char_name} in crossover clothing standing alert on the outskirts of {story_name}, environment of {loc} reflecting theme colors.",
                f"Panel 2: {char_name} looking around with a tense expression, noticing the atmospheric shadows.",
                f"Panel 3: {char_name} interacting with the surroundings, discovering a clue in the scene.",
                f"Panel 4: {char_name} moving forward with a determined posture, step-by-step into the unknown."
            ],
            "dialogue_and_captions": [
                f"Caption: The journey begins in this strange new world.",
                f"{char_name}: We must proceed carefully. This place is unpredictable.",
                f"Caption: Every step brings new questions.",
                f"{char_name}: There is no turning back now."
            ]
        })
        
    candidates = [
        {
            "name": f"{char_name} Crossover Pose",
            "type": "character",
            "description": f"{char_name} in an active pose reflecting setting style.",
            "sdxl_prompt": f"indie comic style illustration, clean minimalist line art, flat color palette, adapted {char_name}, consistent"
        }
    ]

    return {
        "story_descriptive": f"The main character {char_name} is pulled into the world of {story_name}. Exploring the {loc}, they face environmental challenges, engage in authentic dialogue, and experience internal growth.",
        "character_visual_looks": looks,
        "storyboard_10_pages": pages,
        "candidate_visual_elements": candidates
    }

def generate_crossover_foundation(personality, setting):
    """Generate the high-level story_descriptive, character_visual_looks, and initial candidate_visual_elements"""
    print("\nGenerating crossover foundation and visual design elements...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert comic book narrative director, character designer, and asset planner.
        
        Your task is to:
        1. Write a highly dramatic, cinematic crossover story in descriptive narrative form where the main character is pulled into the multiverse of the target story. You must fit the main character's ARC TURN directly into the target story world's narrative arc. Write a highly dramatic multiverse crossover where their psychological shift and character development mirror or collide with the target story's themes.
        2. Design the adapted crossover visual clothing style of the main character (excluding generic names, focusing on style, mood, and clothing details).
        3. Identify 4 candidate visual elements (main character crossover pose, environment/scenery anchor, key prop, secondary character) for rendering.
        
        COLOR FUSION MANDATE: Explicitly fuse the main character's favorite theme/mood colors with the target setting's color palette (e.g., neon pink/cyan for Cyberpunk or gothic dark tones for Noir). Fictional clothing must be designed adaptively (no default costumes!).
        
        Follow this exact JSON structure for the output:
        {{
            "story_descriptive": "A highly dramatic, cinematic descriptive story of the crossover event in narrative form.",
            "character_visual_looks": "Adaptive visual clothing style of the crossover character.",
            "candidate_visual_elements": [
                {{
                    "name": "Main Character Crossover Pose",
                    "type": "character",
                    "description": "Main character in adapted crossover style in a specific cinematic pose.",
                    "sdxl_prompt": "indie comic style illustration, clean minimalist line art, flat color palette, main character in adapted crossover clothing, consistent"
                }},
                {{
                    "name": "Environment Anchor",
                    "type": "environment",
                    "description": "Key environment location from the target story.",
                    "sdxl_prompt": "indie comic style illustration, clean minimalist line art, flat color palette, cinematic view of target story environment, consistent"
                }},
                {{
                    "name": "Key Prop",
                    "type": "prop",
                    "description": "Important item or piece of gear in the crossover scene.",
                    "sdxl_prompt": "indie comic style illustration, clean minimalist line art, flat color palette, key prop with high detail, consistent"
                }},
                {{
                    "name": "Secondary Character Crossover",
                    "type": "secondary_character",
                    "description": "A side character from the target story adapted for the scene.",
                    "sdxl_prompt": "indie comic style illustration, clean minimalist line art, flat color palette, side character in crossover environment, consistent"
                }}
            ]
        }}
        
        Respond ONLY with valid JSON matching the exact schema above. Do not add any text before or after the JSON:
        """),
        ("human", """
        CHARACTER: {character_name}
        Core Personality: {personality_traits}
        Nature: {nature}
        Arc Turn: {arc_turn}
        
        STORY WORLD: {story_name}
        Genre: {genre}
        Environment Description: {environment_description}
        Theme Color Associated: {theme_color_associated}
        Vibes: {vibes}
        Key Side Characters: {key_side_characters}
        """)
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke({
            "character_name": personality['character_name'],
            "personality_traits": ', '.join(personality['core_personality_traits']),
            "nature": personality['nature'],
            "arc_turn": personality.get('arc_turn', 'Undergoes a key transition in their moral or psychological worldview.'),
            "story_name": setting['story_name'],
            "genre": setting['genre'],
            "environment_description": setting['environment_description'],
            "theme_color_associated": ', '.join(setting['theme_color_associated']),
            "vibes": setting['vibes'],
            "key_side_characters": json.dumps(setting['key_side_characters'])
        })

        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            raise ValueError("No JSON found")
    except Exception as e:
        print(f"Error generating crossover foundation: {e}")
        char_name = personality['character_name']
        story_name = setting['story_name']
        return {
            "story_descriptive": f"The main character {char_name} is pulled into the world of {story_name}, undergoing parallel development and facing environmental challenges.",
            "character_visual_looks": f"{char_name} in adapted clothing matching the {setting['vibes']} vibes of {story_name}.",
            "candidate_visual_elements": [
                {
                    "name": f"{char_name} Crossover Pose",
                    "type": "character",
                    "description": f"{char_name} in adapted style.",
                    "sdxl_prompt": f"indie comic style illustration, clean minimalist line art, flat color palette, adapted {char_name}, consistent"
                }
            ]
        }

def fuse_page(personality, setting, page_num, history_pages):
    """Generate storyboard details for a single page (page_num) given history of previous pages"""
    print(f"Querying local LLM for Page {page_num} storyboard script...")
    
    if history_pages:
        history_context = ""
        for hp in history_pages:
            history_context += f"Page {hp['page_number']} Story Beat: {hp['narrative_progression']}\n"
            history_context += f"  Location: {hp['location']}\n"
            history_context += f"  Panels Dialogue Summary: {', '.join(hp['dialogue_and_captions'])}\n\n"
    else:
        history_context = "This is the very first page of the comic. No history exists yet."

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert comic book narrative director and storyboard artist.
        
        Your task is to write the storyboard layout details for EXACTLY Page {page_num} of a 10-page crossover comic series.
        You must continue the story logically from the narrative history of the previous pages provided.
        
        For Page {page_num}, define:
           - Location: The specific setting.
           - Narrative Progression: The page's story beat.
           - Scene Settlement: A highly descriptive depiction of the environment at its absolute best (lighting, weather, vibes, textures, objects' positions).
           - Character Expressions: Specific facial expressions and postures of all active characters in play.
           - Personality/Emotion State: The main character's internal emotional beat.
           - Side Characters Present: List of side characters active on this page.
           - Panels Breakdown: Array of EXACTLY 4 strings. Each string must be a highly detailed visual prompt for SDXL, describing:
             - The active characters, their poses, reactions, emotional states, and what they are wearing (designed adaptively for this crossover, using character visual style).
             - The environment details, exact locations of objects in the scene, and atmospheric lighting.
           - Dialogue and Captions: Array of EXACTLY 4 strings, where each string represents the dialogue or caption corresponding to that specific panel (from Panel 1 to Panel 4). Write dialogues with famous quotes/dialogues and vibes matching the original style and cadence of the target story.
        
        Follow this exact JSON structure for Page {page_num}:
        {{
            "page_number": {page_num},
            "location": "Watson Docks District Alleyway",
            "narrative_progression": "The main character recovers and meets their companion.",
            "scene_settlement": "Gritty rain-slicked concrete alleyway reflecting magenta neon signage, circular floor grates venting steam, tall scrap-metal barrels.",
            "character_expressions": "Spider-Man looking down at his hands with eyes wide in confusion, jaw slightly clenched.",
            "personality_state": "Confused and disoriented",
            "side_characters_present": ["Jackie Welles"],
            "panels_breakdown": [
                "Panel 1: Spider-Man in a high-collared Victorian deep crimson coat, kneeling on wet dark grass, looking up at the sky in confusion as rain begins to fall.",
                "Panel 2: Spider-Man standing up, dusting off his wet crimson coat, looking at the distant stone manor house Wuthering Heights outlined against the dark horizon.",
                "Panel 3: A brooding figure, Heathcliff, wearing a dark woolen cloak, stands near a decaying stone fence, staring intensely at Spider-Man with furrowed brows.",
                "Panel 4: Spider-Man approaches Heathcliff cautiously, raising a hand in greeting, while Heathcliff remains motionless, his expression suspicious and cold."
            ],
            "dialogue_and_captions": [
                "Caption: I was swinging through Manhattan, and then... nothing but wind and heather.",
                "Spider-Man: Where is the skyline? What is this place?",
                "Caption: From the shadows of the fence, a dark eyes watch the newcomer.",
                "Heathcliff: You are not from these moors. Speak, or be gone."
            ]
        }}
        
        Respond ONLY with a valid JSON block matching the exact schema above. Do not add any text before or after the JSON:
        """),
        ("human", """
        CHARACTER: {character_name}
        Core Personality: {personality_traits}
        Nature: {nature}
        Arc Turn: {arc_turn}
        
        STORY WORLD: {story_name}
        Genre: {genre}
        Environment Description: {environment_description}
        Theme Color Associated: {theme_color_associated}
        Vibes: {vibes}
        Key Side Characters: {key_side_characters}
        
        NARRATIVE HISTORY OF PREVIOUS PAGES:
        {history_context}
        
        Generate Page {page_num} details:
        """)
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke({
            "page_num": page_num,
            "character_name": personality['character_name'],
            "personality_traits": ', '.join(personality['core_personality_traits']),
            "nature": personality['nature'],
            "arc_turn": personality.get('arc_turn', 'Undergoes a key transition in their moral or psychological worldview.'),
            "story_name": setting['story_name'],
            "genre": setting['genre'],
            "environment_description": setting['environment_description'],
            "theme_color_associated": ', '.join(setting['theme_color_associated']),
            "vibes": setting['vibes'],
            "key_side_characters": json.dumps(setting['key_side_characters']),
            "history_context": history_context
        })

        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            page_data = json.loads(json_match.group())
            page_data["page_number"] = page_num
            return page_data
        else:
            raise ValueError("No JSON found in response")
    except Exception as e:
        print(f"Error generating Page {page_num} storyboard details: {e}")
        char_name = personality['character_name']
        story_name = setting['story_name']
        loc = setting['environment_description']
        return {
            "page_number": page_num,
            "location": f"{story_name} Outskirts",
            "narrative_progression": f"{char_name} navigates the environment of {story_name}, facing challenges.",
            "scene_settlement": f"Detailed environment description of {loc} during scene {page_num}.",
            "character_expressions": f"{char_name} showing deep focus and emotional response appropriate for scene {page_num}.",
            "personality_state": "Focused and determined",
            "side_characters_present": [],
            "panels_breakdown": [
                f"Panel 1: {char_name} in crossover clothing standing alert on the outskirts of {story_name}, environment of {loc} reflecting theme colors.",
                f"Panel 2: {char_name} looking around with a tense expression, noticing the atmospheric shadows.",
                f"Panel 3: {char_name} interacting with the surroundings, discovering a clue in the scene.",
                f"Panel 4: {char_name} moving forward with a determined posture, step-by-step into the unknown."
            ],
            "dialogue_and_captions": [
                f"Caption: The journey continues in this strange world.",
                f"{char_name}: We must proceed carefully. This place is unpredictable.",
                f"Caption: Every step brings new questions.",
                f"{char_name}: There is no turning back now."
            ]
        }

import argparse
parser = argparse.ArgumentParser(description="Generate page-by-page storyboard fusion.")
parser.add_argument("--page", type=int, default=0, help="The page number to generate (1-10). If 0, generates all pages sequentially.")
args = parser.parse_args()

output_path = get_output_path(fusion_dir, "fusion_complete.json")

# Load existing fusion complete if it exists
existing_fusion = None
if os.path.exists(output_path):
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            existing_fusion = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load existing fusion file: {e}")

def save_fusion_state(personality, setting, story_descriptive, character_visual_looks, storyboard_pages, candidates):
    persistent_components = run_vector_persistence_analysis(
        candidates, 
        storyboard_pages, 
        langchain_settings.get("model", "llama3.2"),
        langchain_settings.get("ollama_url", "http://localhost:11434")
    )
    
    fusion_result = {
        "story_descriptive": story_descriptive,
        "character_visual_looks": character_visual_looks,
        "storyboard_10_pages": storyboard_pages,
        "components": persistent_components
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "personality": personality,
            "setting": setting,
            "fusion": fusion_result
        }, f, indent=2)
        
    print(f"\nPERSISTENT COMPONENT ASSETS SELECTED:")
    print("-" * 50)
    for c in persistent_components:
        print(f"  Asset {c['component_number']} ({c['type']}): {c['name']}")
        print(f"    SDXL Prompt: {c['sdxl_prompt'][:60]}...")
    print(f"\nSaved components configuration to: {output_path}")
    
    # Save SDXL prompt config
    style_settings = settings.get("style", {})
    negative_prompt = ", ".join(style_settings.get("negative_terms", [
        "photorealistic", "3D render", "shading", "gradients", "blurry", "messy lines"
    ]))
    style_desc = ", ".join(style_settings.get("positive_terms", [
        "clean minimalist line art", "flat color palette", "crisp continuous outlines", "cel-shaded with no gradients"
    ]))
    
    prompt_output = {
        "positive_prompt": character_visual_looks,
        "negative_prompt": negative_prompt,
        "style": style_desc,
        "character_name": personality['character_name'],
        "story_world": setting['story_name']
    }
    
    sdxl_prompt_path = get_output_path(fusion_dir, "sdxl_prompt.json")
    with open(sdxl_prompt_path, "w", encoding="utf-8") as f:
        json.dump(prompt_output, f, indent=2)
    print(f"Saved SDXL prompt configuration to: {sdxl_prompt_path}")

# Run mode
if args.page == 0:
    print("\n[ALL PAGES SEQUENTIAL MODE] Synthesizing full crossover storyboard page-by-page...")
    foundation = generate_crossover_foundation(personality, setting)
    story_descriptive = foundation.get("story_descriptive", "")
    character_visual_looks = foundation.get("character_visual_looks", "")
    candidates = foundation.get("candidate_visual_elements", [])
    
    storyboard_pages = []
    for page_num in range(1, 11):
        print(f"\n--- Synthesizing Page {page_num}/10 ---")
        page_data = fuse_page(personality, setting, page_num, storyboard_pages)
        storyboard_pages.append(page_data)
        
    save_fusion_state(personality, setting, story_descriptive, character_visual_looks, storyboard_pages, candidates)
else:
    page_num = args.page
    print(f"\n[SINGLE PAGE MODE] Synthesizing Page {page_num}/10...")
    
    if page_num == 1:
        foundation = generate_crossover_foundation(personality, setting)
        story_descriptive = foundation.get("story_descriptive", "")
        character_visual_looks = foundation.get("character_visual_looks", "")
        candidates = foundation.get("candidate_visual_elements", [])
        storyboard_pages = []
    else:
        if existing_fusion and "fusion" in existing_fusion:
            fusion_dict = existing_fusion["fusion"]
            story_descriptive = fusion_dict.get("story_descriptive", "")
            character_visual_looks = fusion_dict.get("character_visual_looks", "")
            candidates = []
            for comp in fusion_dict.get("components", []):
                candidates.append({
                    "name": comp.get("name"),
                    "type": comp.get("type"),
                    "description": comp.get("description"),
                    "sdxl_prompt": comp.get("sdxl_prompt")
                })
            storyboard_pages = [p for p in fusion_dict.get("storyboard_10_pages", []) if p.get("page_number") < page_num]
        else:
            print(f"Warning: No existing fusion found for Page {page_num}. Running foundation first.")
            foundation = generate_crossover_foundation(personality, setting)
            story_descriptive = foundation.get("story_descriptive", "")
            character_visual_looks = foundation.get("character_visual_looks", "")
            candidates = foundation.get("candidate_visual_elements", [])
            storyboard_pages = []
            
    page_data = fuse_page(personality, setting, page_num, storyboard_pages)
    
    # Update and sort
    storyboard_pages = [p for p in storyboard_pages if p.get("page_number") != page_num]
    storyboard_pages.append(page_data)
    storyboard_pages.sort(key=lambda x: x.get("page_number", 1))
    
    save_fusion_state(personality, setting, story_descriptive, character_visual_looks, storyboard_pages, candidates)

print("=" * 70)

