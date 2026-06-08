import json

text = """{
  "characters_present": ["Spider-Man"],
  "emotions": {
    "Spider-Man": {
      "emotion": "ecstatic",
      "intensity": "high",
      "expression_trigger": "grinning with eyes shining bright, fists clenched in excitement"
    }
  },
  "core_action": "Spider-Man swinging through the air, neon lights reflected on his suit",
  "background_env": "dark alleyway with rain-soaked pavement and glowing neon signs\""""

print("Length of string:", len(text))
try:
    data = json.loads(text)
    print("SUCCESS")
except Exception as e:
    print("FAILED:", e)
    if "char" in str(e):
        import re
        char_idx = int(re.search(r'char (\d+)', str(e)).group(1))
        print(f"Character at index {char_idx}: {repr(text[char_idx])}")
        print("Surrounding context:")
        print(repr(text[max(0, char_idx-20):min(len(text), char_idx+20)]))
