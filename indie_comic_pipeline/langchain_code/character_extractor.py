"""
CHARACTER PERSONALITY EXTRACTOR
Uses Ollama + Llama 3.2 to extract personality from any character name
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

print("CHARACTER PERSONALITY EXTRACTOR - Parsing human psyche")

print("=" * 70)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_helper import load_settings, get_output_path

settings = load_settings()

langchain_settings = settings.get("langchain", {})

                                                                                       

print("\nConnecting to local Ollama server...")

llm = ChatOllama(

    model=langchain_settings.get("model", "llama3.2"),

    temperature=langchain_settings.get("temperature", 0.3),

    base_url=langchain_settings.get("ollama_url", "http://localhost:11434")

)

print(f"Connected to Ollama: {llm.model}")

def extract_character_personality(character_name):

    """Extract personality traits from any character using local LLM"""

    

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a character analysis expert. Extract personality and psychological parameters from ANY fictional character.
        
        To prevent hallucinations and guarantee high-quality descriptive outputs, study the following example:
        
        Example Input: Spider-Man
        Example Output:
        {{
            "character_name": "Spider-Man",
            "core_personality_traits": ["responsible", "witty", "self-sacrificing", "relatable", "persistent"],
            "personality_development_stories": "Peter Parker started as a selfish teenager who ignored his powers. Following the tragic death of Uncle Ben due to Peter's inaction, he underwent a significant psychological shift, committing himself entirely to selflessness and learning that personal sacrifices are necessary to protect others.",
            "nature": "Altruistic, hyper-responsible, and hides extreme anxiety behind quick-witted banter and humor.",
            "characteristics_types": "Enneagram Type 2 (The Helper) / Type 6 (The Loyalist) archetype, fitting the tragic hero narrative.",
            "favorite_dialogue": "With great power comes great responsibility.",
            "arc_turn": "From selfish avoidance of responsibility to accepting the painful weight of duty after a preventable tragedy, transforming guilt into active service of others."
        }}
        
        CRITICAL REQUIREMENT: You MUST NOT include any physical traits or visual appearance descriptions (e.g. hair color, eyes, costume details, height, build, physical armor). Extract ONLY mental, psychological, behavioral, and dialogue characteristics.
        
        Respond ONLY with valid JSON matching the exact schema above. No other text before or after:
        """),

        ("human", "Extract personality for: {character}")

    ])

    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke({"character": character_name})
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            personality = json.loads(json_match.group())
            if 'personality_development_stories' not in personality:
                personality['personality_development_stories'] = "Underwent significant psychological development over time."
            if 'nature' not in personality:
                personality['nature'] = "Has a distinct temperament and behavior pattern."
            if 'characteristics_types' not in personality:
                personality['characteristics_types'] = "Matches a classic archetype profile."
            if 'favorite_dialogue' not in personality:
                personality['favorite_dialogue'] = "Has a notable quote or catchphrase."
            if 'arc_turn' not in personality:
                personality['arc_turn'] = "Undergoes a key transition in their moral or psychological worldview."
            return personality
        else:
            raise ValueError("No JSON found")

    except Exception as e:
        print(f"Error: {e}")
        return fallback_character(character_name)

def fallback_character(character_name):
    fallbacks = {
        "spiderman": {
            "character_name": "Spider-Man",
            "core_personality_traits": ["responsible", "witty", "self-sacrificing", "relatable", "persistent"],
            "personality_development_stories": "Peter Parker started as a selfish teenager who ignored his powers. Following the tragic death of Uncle Ben due to Peter's inaction, he underwent a significant psychological shift, committing himself entirely to selflessness and learning that personal sacrifices are necessary to protect others.",
            "nature": "Altruistic, hyper-responsible, and hides extreme anxiety behind quick-witted banter and humor.",
            "characteristics_types": "Enneagram Type 2 (The Helper) / Type 6 (The Loyalist) archetype, fitting the tragic hero narrative.",
            "favorite_dialogue": "With great power comes great responsibility.",
            "arc_turn": "From selfish avoidance of responsibility to accepting the painful weight of duty after a preventable tragedy, transforming guilt into active service of others."
        },
        "batman": {
            "character_name": "Batman",
            "core_personality_traits": ["brooding", "strategic", "vengeful", "disciplined", "loner"],
            "personality_development_stories": "Bruce Wayne witnesses his parents' murder as a child, traveling the world to train his mind and body. This trauma transforms him into the vigilante protector of Gotham City, channeling his grief and rage into a disciplined mission of justice.",
            "nature": "Brooding, hyper-vigilant, highly strategic, and emotionally closed off to protect others.",
            "characteristics_types": "Enneagram Type 5 (The Investigator) / Type 8 (The Challenger) archetype, vigilante protector.",
            "favorite_dialogue": "I am vengeance.",
            "arc_turn": "From a grieving, powerless orphan consumed by rage to a highly disciplined guardian who channels trauma into a lifelong crusade for justice and order."
        }
    }

    name_lower = character_name.lower()
    for key in fallbacks:
        if key in name_lower:
            return fallbacks[key]

    return {
        "character_name": character_name.title(),
        "core_personality_traits": ["brave", "determined", "loyal", "resourceful", "compassionate"],
        "personality_development_stories": f"The development path where {character_name} overcomes psychological challenges and matures.",
        "nature": "Determined and protective.",
        "characteristics_types": "Classic hero archetype.",
        "favorite_dialogue": "I will do what is right.",
        "arc_turn": "From vulnerability or isolation to finding strength and purpose in defending a larger cause."
    }

                                                                   

print("\n" + "=" * 70)

if len(sys.argv) > 1:

    character_name = sys.argv[1].strip()

    print(f"Using character from command line argument: {character_name}")

else:

    if not sys.stdin.isatty():

        try:

            character_name = sys.stdin.readline().strip()

        except Exception:

            character_name = ""

        if not character_name:

            character_name = "Spiderman"

            print(f"Non-interactive mode: Using default character '{character_name}'")

        else:

            print(f"Non-interactive mode: Read character from stdin: '{character_name}'")

    else:

        character_name = input("Enter character name (e.g., Spiderman, Batman, Wolverine): ").strip()

print(f"\nAnalyzing character: '{character_name}'...")

personality = extract_character_personality(character_name)

print("\nEXTRACTED PERSONALITY:")

print("-" * 50)

print(f"Name: {personality['character_name']}")

print(f"\nCore Traits: {', '.join(personality['core_personality_traits'])}")

print(f"Stories: {personality['personality_development_stories']}")

print(f"Nature: {personality['nature']}")

print(f"Archetype: {personality['characteristics_types']}")

print(f"Favorite Dialogue: {personality['favorite_dialogue']}")

print(f"Arc Turn: {personality['arc_turn']}")


fusion_dir = settings.get("outputs", {}).get("fusion_dir", "outputs/fusion")

output_path = get_output_path(fusion_dir, "character_personality.json")

with open(output_path, "w", encoding="utf-8") as f:

    json.dump(personality, f, indent=2)

print(f"\nSaved JSON configuration to: {output_path}")

print("=" * 70)

