"""
STORY SETTING EXTRACTOR
Uses Ollama + Llama 3.2 to extract setting details from any story
No API key required - runs completely locally
"""

import sys

import os

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

import json

import re

from langchain_ollama import ChatOllama

from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate

print("=" * 70)

print("STORY SETTING EXTRACTOR - Parsing geographical settings")

print("=" * 70)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_helper import load_settings, get_output_path

settings = load_settings()

langchain_settings = settings.get("langchain", {})

                                                                                       

print("\nConnecting to local Ollama server...")

llm = ChatOllama(

    model=langchain_settings.get("model", "llama3.2"),

    temperature=langchain_settings.get("temperature", 0.3),

    num_predict=8192,

    base_url=langchain_settings.get("ollama_url", "http://localhost:11434")

)

print(f"Connected to Ollama: {llm.model}")

def extract_story_setting(story_name):

    """Extract setting details from any story using local LLM"""

    

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a setting analysis expert. Extract world details from any story or movie universe.
        
        To prevent hallucinations and guarantee high-quality descriptive outputs, study the following example:
        
        Example Input: Cyberpunk 2077
        Example Output:
        {{
            "story_name": "Cyberpunk 2077",
            "genre": "Sci-Fi / Cyberpunk / Dystopian Thriller",
            "environment_description": "Rain-slicked asphalt streets reflecting neon signs, colossal dark skyscrapers looming overhead, industrial steam vents, high-tech vehicles, crowded street markets.",
            "theme_color_associated": ["neon pink", "electric cyan", "canary yellow", "deep obsidian black", "chrome silver"],
            "vibes": "High-tech, low-life, gritty noir, corporate oppression, rebellious underground energy.",
            "key_side_characters": [
                {{
                    "name": "Jackie Welles",
                    "default_personality": "Loyal, boisterous, ambitious, family-oriented street mercenary",
                    "relation_to_main": "Partner-in-crime and trusted companion who helps guide the main character through the city's dangerous underworld"
                }},
                {{
                    "name": "Johnny Silverhand",
                    "default_personality": "Charismatic rockerboy, cynical, rebellious, anti-corporate anarchist",
                    "relation_to_main": "Vocal mentor and internal foil who pushes the main character to rebel against the dominant systems"
                }},
                {{
                    "name": "Judy Alvarez",
                    "default_personality": "Talented braindance editor, artistic, empathetic, fiercely loyal to friends",
                    "relation_to_main": "Technical ally who assists with digital investigations and offers emotional support"
                }},
                {{
                    "name": "Panam Palmer",
                    "default_personality": "Hot-headed, passionate, independent Nomad clan outcast",
                    "relation_to_main": "Combat driver and loyalty-driven ally who assists in high-stakes structural raids"
                }}
            ]
        }}
        
        CRITICAL REQUIREMENT: You MUST extract between 2 and 6 key side characters from the target story world. Define their name, default personality, and how they would relate to the main character in this crossover.
        
        Respond ONLY with valid JSON matching the exact schema above. No other text before or after:
        """),

        ("human", "Describe the setting of: {story}")

    ])

    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke({"story": story_name})
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            setting = json.loads(json_match.group())
            if 'genre' not in setting:
                setting['genre'] = "Drama / Adventure"
            if 'vibes' not in setting:
                setting['vibes'] = "Atmospheric and engaging."
            if 'theme_color_associated' not in setting:
                setting['theme_color_associated'] = ["grey", "blue"]
            if 'key_side_characters' not in setting or not isinstance(setting['key_side_characters'], list) or len(setting['key_side_characters']) < 2:
                setting['key_side_characters'] = [
                    {"name": "Jackie Welles", "default_personality": "Loyal, ambitious street mercenary.", "relation_to_main": "Partner-in-crime."},
                    {"name": "Johnny Silverhand", "default_personality": "Charismatic rockerboy rebel.", "relation_to_main": "Internal mental mentor."},
                    {"name": "Judy Alvarez", "default_personality": "Talented braindance technician.", "relation_to_main": "Technical support advisor."}
                ]
            return setting
        else:
            raise ValueError("No JSON found")

    except Exception as e:
        print(f"Error: {e}")
        return fallback_story(story_name)

def fallback_story(story_name):
    fallbacks = {
        "wuthering heights": {
            "story_name": "Wuthering Heights",
            "genre": "Gothic Romance / Tragedy",
            "environment_description": "windswept moors, wild heather, stormy skies, isolated stone manor",
            "theme_color_associated": ["dark green", "grey", "brown", "muted purple", "stormy blue"],
            "vibes": "brooding, intense, desolate, tragic, and passionate",
            "key_side_characters": [
                {"name": "Heathcliff", "default_personality": "brooding, intense, vengeful, passionate yet cruel", "relation_to_main": "Romantic foil and central driver of conflict."},
                {"name": "Catherine Earnshaw", "default_personality": "spirited, volatile, rebellious, torn between status and love", "relation_to_main": "Main character's love interest and source of emotional obsession."},
                {"name": "Hindley Earnshaw", "default_personality": "resentful, abusive, self-destructive, grieving brother", "relation_to_main": "Aggressor who drives characters to seek safety or conflict."},
                {"name": "Nelly Dean", "default_personality": "observant, maternal, gossipy, narrating housemaid", "relation_to_main": "Neutral advisor and keeper of secrets."}
            ]
        },
        "harry potter": {
            "story_name": "Harry Potter",
            "genre": "Fantasy / Adventure / Coming-of-Age",
            "environment_description": "magical castle, moving staircases, forbidden forest, black lake, stone hallways",
            "theme_color_associated": ["burgundy", "gold", "forest green", "dark brown", "magical silver"],
            "vibes": "magical, mysterious, adventurous, and warm academic",
            "key_side_characters": [
                {"name": "Hermione Granger", "default_personality": "intellectual, rule-abiding, loyal, highly logical", "relation_to_main": "Intellectual companion who solves magical riddles and supports the main character."},
                {"name": "Ron Weasley", "default_personality": "brave, humorous, loyal, occasionally insecure, strategic", "relation_to_main": "Best friend and loyal companion who provides tactical wizarding support."},
                {"name": "Albus Dumbledore", "default_personality": "wise, eccentric, secretive, powerful wizard headmaster", "relation_to_main": "Grand mentor who guides the overarching destiny and mission."},
                {"name": "Severus Snape", "default_personality": "sarcastic, severe, hidden motives, protective double agent", "relation_to_main": "Antagonistic teacher who forces discipline and growth."}
            ]
        },
        "cyberpunk": {
            "story_name": "Cyberpunk 2077",
            "genre": "Sci-Fi / Cyberpunk / Dystopian Thriller",
            "environment_description": "neon-lit rain-slicked asphalt streets, colossal dark skyscrapers looming overhead, industrial steam vents, high-tech flying vehicles.",
            "theme_color_associated": ["neon pink", "cyan", "purple", "black", "yellow"],
            "vibes": "gritty, high-energy, dangerous, corporate-dominated",
            "key_side_characters": [
                {"name": "Johnny Silverhand", "default_personality": "rebellious, charismatic, cynical, anti-corporate rockerboy", "relation_to_main": "Anti-corporate mentor and internal foil who pushes the main character to rebel."},
                {"name": "Jackie Welles", "default_personality": "loyal, ambitious, warm-hearted, street-smart solo", "relation_to_main": "Trusted companion who helps guide the main character through the city's dangerous underworld."},
                {"name": "Judy Alvarez", "default_personality": "empathetic, rebellious, highly skilled technical brain editor", "relation_to_main": "Technical guide and moral anchor in the city's corruption."},
                {"name": "Panam Palmer", "default_personality": "independent, fierce, loyal Nomad clan warrior", "relation_to_main": "Outlaw combat companion who provides structural force."}
            ]
        }
    }

    

    name_lower = story_name.lower()

    for key in fallbacks:

        if key in name_lower:

            return fallbacks[key]

    

    return {

        "story_name": story_name.title(),

        "genre": "Fantasy / Adventure",

        "environment_description": "beautiful landscape with unique features, glowing crystals, ancient ruins",

        "theme_color_associated": ["emerald green", "golden amber", "deep purple", "silver"],

        "vibes": "adventurous, mysterious, and magical",

        "key_side_characters": [
            {"name": "Local Companion", "default_personality": "Wise, silent, knows the ancient paths.", "relation_to_main": "Assists and guides the main character."},
            {"name": "Sage Mentor", "default_personality": "Elderly guardian of the realm, possesses deep knowledge.", "relation_to_main": "Mentors and advises the main character."}
        ]

    }

                                                                                 

print("\n" + "=" * 70)

if len(sys.argv) > 1:

    story_name = sys.argv[1].strip()

    print(f"Using story/setting from command line argument: {story_name}")

else:

    if not sys.stdin.isatty():

        try:

            story_name = sys.stdin.readline().strip()

        except Exception:

            story_name = ""

        if not story_name:

            story_name = "Cyberpunk"

            print(f"Non-interactive mode: Using default story/setting '{story_name}'")

        else:

            print(f"Non-interactive mode: Read story/setting from stdin: '{story_name}'")

    else:

        story_name = input("Enter story/setting (e.g., Wuthering Heights, Harry Potter, Cyberpunk): ").strip()

print(f"\nAnalyzing setting: '{story_name}'...")

setting = extract_story_setting(story_name)

print("\nEXTRACTED SETTING:")

print("-" * 50)

print(f"Story: {setting['story_name']}")

print(f"Genre: {setting['genre']}")

print(f"Vibes: {setting['vibes']}")

colors = setting.get('theme_color_associated', ["grey", "blue"])
if isinstance(colors, str):
    colors_list = [colors]
elif isinstance(colors, list):
    colors_list = [str(c) for c in colors]
else:
    colors_list = [str(colors)]
print(f"Colors: {', '.join(colors_list)}")

print(f"\nEnvironment: {setting['environment_description'][:100]}...")

                                                                    

fusion_dir = settings.get("outputs", {}).get("fusion_dir", "outputs/fusion")

output_path = get_output_path(fusion_dir, "story_setting.json")

with open(output_path, "w", encoding="utf-8") as f:

    json.dump(setting, f, indent=2)

print(f"\nSaved JSON configuration to: {output_path}")

print("=" * 70)

