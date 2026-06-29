from unsloth import FastLanguageModel
from story_gen_old import (
    DUMMY_EMOTION_OUTPUT, build_user_prompt, SYSTEM_PROMPT,
    StoryGenerator, evaluate_script
)
import json, torch

# Load YOUR fine-tuned model (not the base)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="moodweaver_stage2_finetuned",  # local path
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)  # 2x faster inference

# Run a generation
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user",   "content": build_user_prompt(DUMMY_EMOTION_OUTPUT)},
]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=700,
        temperature=0.75,
        top_p=0.92,
        do_sample=True,
    )

raw = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print("RAW OUTPUT:\n", raw)

# Validate JSON + quality
import re
match = re.search(r'\{.*\}', raw, re.DOTALL)
if match:
    data = json.loads(match.group())
    print("\nPARSED:\n", json.dumps(data, indent=2))