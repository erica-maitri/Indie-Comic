import json
import re
import sys
import os
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(
    model="llama3.2",
    temperature=0.2,
    base_url="http://localhost:11434"
)

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

panel_text = "Panel 1: Spider-Man swinging through a dark alleyway, neon lights glowing in the rain. Caption: 'This city... it's alive.'"
dialogue_text = "Caption: This city... it's alive. | Spider-Man: What is this place?"
history_context = "No previous panels in this scene yet."

response = chain.invoke({
    "panel_text": panel_text,
    "dialogue_text": dialogue_text,
    "history_context": history_context
})

print("RAW RESPONSE FROM LLM:")
print("=" * 60)
print(response)
print("=" * 60)
try:
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        data = json.loads(json_match.group())
        print("SUCCESSFULLY PARSED AS JSON:")
        print(json.dumps(data, indent=2))
    else:
        print("NO BRACE MATCH FOUND")
except Exception as e:
    print("FAILED TO PARSE:")
    print(e)
