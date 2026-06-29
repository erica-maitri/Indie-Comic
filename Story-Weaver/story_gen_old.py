"""
MoodWeaver — Stage 2: Story Generation Engine
=============================================
Supports: Llama 3.2 3B & Mistral 7B (4-bit GPTQ quantized)
Input:  Stage 1 emotion detector JSON
Output: 4-panel comic script JSON

Install dependencies:
    pip install transformers accelerate bitsandbytes auto-gptq optimum torch

Usage:
    python story_gen.py
    python story_gen.py --model mistral --device cuda
"""

import json
import re
import time
import argparse
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
    GenerationConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("moodweaver.stage2")


# ---------------------------------------------------------------------------
# 1. Data Schemas
# ---------------------------------------------------------------------------

@dataclass
class EmotionOutput:
    """Stage 1 DistilBERT output schema."""
    primary_emotion: str
    confidence: float
    secondary_emotions: list[dict]          # [{"emotion": str, "score": float}]
    somatic_markers: bool
    user_text: str


@dataclass
class Panel:
    panel: int
    visual: str
    dialogue: str
    emotion_beat: str
    motion: str


@dataclass
class StoryScript:
    recurring_motif: str
    panels: list[Panel]
    generation_time_s: float = 0.0
    model_used: str = ""

    def to_dict(self) -> dict:
        return {
            "recurring_motif": self.recurring_motif,
            "panels": [
                {
                    "panel": p.panel,
                    "visual": p.visual,
                    "dialogue": p.dialogue,
                    "emotion_beat": p.emotion_beat,
                    "motion": p.motion,
                }
                for p in self.panels
            ],
            "meta": {
                "generation_time_s": round(self.generation_time_s, 2),
                "model_used": self.model_used,
            },
        }


# ---------------------------------------------------------------------------
# 2. Dummy Stage 1 Output (mirrors DistilBERT result)
# ---------------------------------------------------------------------------

DUMMY_EMOTION_OUTPUT = EmotionOutput(
    primary_emotion="sadness",
    confidence=0.72,
    secondary_emotions=[
        {"emotion": "exhaustion", "score": 0.15},
        {"emotion": "longing",    "score": 0.08},
        {"emotion": "numbness",   "score": 0.05},
    ],
    somatic_markers=True,
    user_text="i dont know why but everything just feels really heavy lately. even small things.",
)


# ---------------------------------------------------------------------------
# 3. Model Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "llama": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct",
        "display": "Llama 3.2 3B",
        "context_window": 8192,
        "max_new_tokens": 700,
        "use_chat_template": True,
    },

    "mistral": {
        "repo": "mistralai/Mistral-7B-Instruct-v0.2",
        "display": "Mistral 7B",
        "context_window": 32768,
        "max_new_tokens": 800,
        "use_chat_template": True,
    },

    "tiny": {
        "repo": "Qwen/Qwen2.5-0.5B-Instruct",
        "display": "Qwen 2.5 0.5B (test only)",
        "context_window": 4096,
        "max_new_tokens": 600,
        "use_chat_template": True,
        "no_quantize": True,
    },
}

MODEL_REGISTRY["finetuned"] = {
    "repo": "moodweaver_stage2_finetuned", 
    "display": "MoodWeaver Fine-tuned",
    "context_window": 2048,
    "max_new_tokens": 700,
    "use_chat_template": True,
    "no_quantize": True,   
}

# ---------------------------------------------------------------------------
# 4. Literary Prompt Builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a literary graphic novelist writing for adults.
Respond ONLY with valid JSON. No markdown fences, no explanation, no preamble.

STRUCTURE RULES:
- Panel 1-2: VALIDATE the emotion (show, don't tell. Use objects, space, physical sensation)
- Panel 3: Introduce a SMALL COMPLICATION or SHIFT (not a solution)
- Panel 4: END WITH OPENNESS (no forced happiness, allow ambiguity)

LITERARY CONSTRAINTS:
- Never name emotions directly. Show through action and objects.
- Include ONE recurring visual motif across all 4 panels.
- Each panel MUST include an internal physical sensation.
- No moral lessons. No sentences beginning with "The lesson is..."
- Dialogue shows what characters DON'T say as much as what they say.

Output ONLY this JSON (no other text):
{
  "recurring_motif": "brief description of the single visual motif",
  "panels": [
    {"panel": 1, "visual": "...", "dialogue": "...", "emotion_beat": "one_word", "motion": "..."},
    {"panel": 2, "visual": "...", "dialogue": "...", "emotion_beat": "one_word", "motion": "..."},
    {"panel": 3, "visual": "...", "dialogue": "...", "emotion_beat": "one_word", "motion": "..."},
    {"panel": 4, "visual": "...", "dialogue": "...", "emotion_beat": "one_word", "motion": "..."}
  ]
}"""

def build_user_prompt(emotion: EmotionOutput) -> str:
    secondary = ", ".join(
        f"{e['emotion']} {e['score']}" for e in emotion.secondary_emotions
    )
    return (
        f"Current emotional state: {emotion.primary_emotion} "
        f"(confidence {emotion.confidence})\n"
        f"Secondary emotions: {secondary}\n"
        f"Somatic markers present: {emotion.somatic_markers}\n"
        f'User context: "{emotion.user_text}"\n\n'
        "Write the 4-panel comic script JSON now."
    )

# ---------------------------------------------------------------------------
# 5. Model Loader
# ---------------------------------------------------------------------------

class StoryGenerator:
    def __init__(self, model_key: str = "llama", device: str = "auto"):
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model key '{model_key}'. Choose from: {list(MODEL_REGISTRY)}")

        self.cfg = MODEL_REGISTRY[model_key]
        self.model_key = model_key
        self.device = device
        self.pipe = None
        self._load()

    def _load(self):
        repo = self.cfg["repo"]
        log.info(f"Loading {self.cfg['display']} from {repo} ...")

        bnb_cfg = None
        if not self.cfg.get("no_quantize"):
            # 4-bit quantization via bitsandbytes (works on GPU)
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        tokenizer = AutoTokenizer.from_pretrained(repo, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            repo,
            quantization_config=bnb_cfg,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype=torch.float16 if bnb_cfg is None else None,
        )
        model.eval()

        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
        )
        log.info(f"{self.cfg['display']} ready.")

    def _build_messages(self, emotion: EmotionOutput) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(emotion)},
        ]

    def _extract_json(self, raw: str) -> dict:
        """Strip markdown fences and extract the first JSON object."""
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        # find first { ... } block
        start = raw.find("{")
        if start == -1:
            raise ValueError("No JSON object found in model output.")
        # find matching closing brace
        depth, end = 0, -1
        for i, ch in enumerate(raw[start:], start):
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            raise ValueError("Unbalanced JSON braces in model output.")
        return json.loads(raw[start : end + 1])

    def _validate_script(self, data: dict) -> StoryScript:
        if "panels" not in data or len(data["panels"]) != 4:
            raise ValueError(f"Expected 4 panels, got {len(data.get('panels', []))}")

        panels = []
        for p in data["panels"]:
            for key in ("visual", "dialogue", "emotion_beat", "motion"):
                if key not in p:
                    raise ValueError(f"Panel {p.get('panel')} missing field '{key}'")
            panels.append(Panel(
                panel=int(p["panel"]),
                visual=p["visual"],
                dialogue=p["dialogue"],
                emotion_beat=p["emotion_beat"],
                motion=p["motion"],
            ))

        return StoryScript(
            recurring_motif=data.get("recurring_motif", "unspecified"),
            panels=panels,
            model_used=self.cfg["display"],
        )

    def generate(self, emotion: EmotionOutput, max_retries: int = 3) -> StoryScript:
        messages = self._build_messages(emotion)

        for attempt in range(1, max_retries + 1):
            log.info(f"Generation attempt {attempt}/{max_retries} ...")
            t0 = time.time()

            outputs = self.pipe(
                messages,
                max_new_tokens=self.cfg["max_new_tokens"],
                do_sample=True,
                temperature=0.75,
                top_p=0.92,
                repetition_penalty=1.15,
                pad_token_id=self.pipe.tokenizer.eos_token_id,
            )

            elapsed = time.time() - t0
            raw = outputs[0]["generated_text"]
            log.info(f"Raw output ({elapsed:.1f}s):\n{raw[:300]}...")

            try:
                data = self._extract_json(raw)
                script = self._validate_script(data)
                script.generation_time_s = elapsed
                log.info(f"Script validated in {elapsed:.1f}s")
                return script
            except (json.JSONDecodeError, ValueError) as e:
                log.warning(f"Attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Story generation failed after {max_retries} attempts. "
                        f"Last error: {e}\nRaw output:\n{raw}"
                    )

        raise RuntimeError("Unreachable")


# ---------------------------------------------------------------------------
# 6. Fine-tuning Dataset Builder (for custom training)
# ---------------------------------------------------------------------------

TRAINING_EXAMPLES = [
    {
        "emotion": EmotionOutput(
            primary_emotion="anxiety",
            confidence=0.81,
            secondary_emotions=[
                {"emotion": "dread",    "score": 0.10},
                {"emotion": "restless", "score": 0.09},
            ],
            somatic_markers=True,
            user_text="my mind just won't stop racing at night, i can't sleep",
        ),
        "target": {
            "recurring_motif": "a ceiling fan turning slowly",
            "panels": [
                {
                    "panel": 1,
                    "visual": "Bedroom ceiling, 2am. A fan rotates. Sheets tangled around legs. Blue phone glow on the wall.",
                    "dialogue": "...",
                    "emotion_beat": "held_breath",
                    "motion": "Eyes open. Chest rises and falls too fast. Finger scrolls without reading.",
                },
                {
                    "panel": 2,
                    "visual": "Close on the fan blades. Each rotation casts a shadow across an open notebook, page blank.",
                    "dialogue": "I should write it down.",
                    "emotion_beat": "suspension",
                    "motion": "Hand reaches toward notebook. Does not pick it up.",
                },
                {
                    "panel": 3,
                    "visual": "Kitchen. 3am. Standing at the open fridge. Fan sound faintly audible from the other room.",
                    "dialogue": "...",
                    "emotion_beat": "fracture",
                    "motion": "Stares into fridge light. Closes it. Opens it again.",
                },
                {
                    "panel": 4,
                    "visual": "Back in bed. Fan still turning. First grey light at the curtain edge. Body still.",
                    "dialogue": "...",
                    "emotion_beat": "drift",
                    "motion": "Eyes finally close. Fan keeps turning.",
                },
            ],
        },
    },
    {
        "emotion": EmotionOutput(
            primary_emotion="grief",
            confidence=0.88,
            secondary_emotions=[
                {"emotion": "loneliness", "score": 0.07},
                {"emotion": "numbness",   "score": 0.05},
            ],
            somatic_markers=True,
            user_text="my dog passed away yesterday. the house is too quiet.",
        ),
        "target": {
            "recurring_motif": "an empty dog bowl on the kitchen floor",
            "panels": [
                {
                    "panel": 1,
                    "visual": "Kitchen floor. An orange bowl, half-full of water, untouched. Morning light.",
                    "dialogue": "...",
                    "emotion_beat": "stillness",
                    "motion": "Person walks past bowl. Pauses. Does not move it.",
                },
                {
                    "panel": 2,
                    "visual": "Living room. A leash hanging on the door hook. Bowl visible through the doorway.",
                    "dialogue": "I should put that away.",
                    "emotion_beat": "weight",
                    "motion": "Hand reaches for leash. Stops. Hand returns to side.",
                },
                {
                    "panel": 3,
                    "visual": "Late afternoon. A friend's text on the phone: 'How are you holding up?' Bowl in background.",
                    "dialogue": "Fine.",
                    "emotion_beat": "contained_rage",
                    "motion": "Thumb hovers over send. Sends it. Puts phone face-down.",
                },
                {
                    "panel": 4,
                    "visual": "Night. Bowl still on the floor. A single lamp on. Person sitting nearby, not looking at it.",
                    "dialogue": "...",
                    "emotion_beat": "open_silence",
                    "motion": "Sits. Breathes. The bowl stays.",
                },
            ],
        },
    },
]


def build_training_jsonl(
    examples: list,
    output_path: str = "moodweaver_stage2_train.jsonl",
):
    """
    Build a JSONL fine-tuning dataset in ChatML format.
    Compatible with: Axolotl, LLaMA-Factory, Unsloth, TRL SFTTrainer.
    """
    records = []
    for ex in examples:
        user_content = build_user_prompt(ex["emotion"])
        assistant_content = json.dumps(ex["target"], indent=2)

        record = {
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        }
        records.append(record)

    path = Path(output_path)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info(f"Saved {len(records)} training examples → {path}")
    return path


# ---------------------------------------------------------------------------
# 7. SFT Trainer (Unsloth / TRL)
# ---------------------------------------------------------------------------

def train_with_unsloth(
    model_key: str = "llama",
    dataset_path: str = "moodweaver_stage2_train.jsonl",
    output_dir: str = "moodweaver_stage2_finetuned",
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    """
    Fine-tune with Unsloth (fastest, LoRA, 4-bit).

    pip install unsloth datasets trl
    """
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, SFTConfig
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install training deps:\n"
            "  pip install unsloth datasets trl"
        )

    repo = MODEL_REGISTRY[model_key]["repo"]
    log.info(f"Loading base model for fine-tuning: {repo}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=repo,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def format_example(example):
        """Apply chat template to messages."""
        return {"text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )}

    dataset = dataset.map(format_example)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        args=SFTConfig(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            learning_rate=lr,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            save_strategy="epoch",
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
        ),
    )

    log.info("Starting fine-tuning ...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info(f"Fine-tuned model saved → {output_dir}")


# ---------------------------------------------------------------------------
# 8. Evaluation
# ---------------------------------------------------------------------------

def evaluate_script(script: StoryScript) -> dict:
    """
    Rule-based quality checks on the generated script.
    Returns a report dict with pass/fail per constraint.
    """
    report = {}
    panels = script.panels

    # Check panel count
    report["panel_count_ok"] = len(panels) == 4

    # Check no direct emotion naming (simple keyword list)
    EMOTION_WORDS = {"sad", "sadness", "anxious", "anxiety", "grief", "angry",
                     "depressed", "happy", "fear", "scared", "lonely"}
    all_text = " ".join(
        f"{p.visual} {p.dialogue} {p.motion}".lower() for p in panels
    )
    found = [w for w in EMOTION_WORDS if w in all_text]
    report["no_direct_emotion_naming"] = len(found) == 0
    if found:
        report["emotion_words_found"] = found

    # Check motif set
    report["recurring_motif_set"] = bool(script.recurring_motif.strip())

    # Check each panel has content
    for p in panels:
        key = f"panel_{p.panel}_complete"
        report[key] = all([p.visual, p.dialogue, p.motion, p.emotion_beat])

    # Check emotion beats are single words
    for p in panels:
        key = f"panel_{p.panel}_beat_single_word"
        report[key] = len(p.emotion_beat.strip().split()) == 1

    passed = sum(1 for v in report.values() if v is True)
    total = sum(1 for v in report.values() if isinstance(v, bool))
    report["score"] = f"{passed}/{total}"

    return report


# ---------------------------------------------------------------------------
# 9. CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MoodWeaver Stage 2 — Story Generation")
    parser.add_argument("--model",   default="llama",  choices=list(MODEL_REGISTRY), help="Model to use")
    parser.add_argument("--device",  default="auto",   help="Device: auto, cuda, cpu")
    parser.add_argument("--mode",    default="generate", choices=["generate", "train", "dataset"],
                        help="generate: run inference | train: fine-tune | dataset: save JSONL only")
    parser.add_argument("--output",  default="story_output.json", help="Output JSON path")
    parser.add_argument("--epochs",  type=int, default=3, help="Training epochs")
    args = parser.parse_args()

    if args.mode == "dataset":
        build_training_jsonl(TRAINING_EXAMPLES)
        return

    if args.mode == "train":
        jsonl_path = build_training_jsonl(TRAINING_EXAMPLES)
        train_with_unsloth(
            model_key=args.model,
            dataset_path=str(jsonl_path),
            epochs=args.epochs,
        )
        return

    # --- Inference mode ---
    log.info("=== MoodWeaver Stage 2: Story Generation ===")
    log.info(f"Model: {MODEL_REGISTRY[args.model]['display']}")

    generator = StoryGenerator(model_key=args.model, device=args.device)
    script = generator.generate(DUMMY_EMOTION_OUTPUT)

    output = script.to_dict()
    log.info("\n" + "="*50)
    log.info("GENERATED SCRIPT:")
    log.info(json.dumps(output, indent=2))

    # Evaluate
    report = evaluate_script(script)
    log.info("\nQUALITY REPORT:")
    log.info(json.dumps(report, indent=2))

    # Save
    out_path = Path(args.output)
    with out_path.open("w") as f:
        json.dump({"script": output, "quality": report}, f, indent=2)
    log.info(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()