# Story Generation using Merged LLMs

A story generation pipeline built using multiple fine-tuned language models that are merged into a single 16-bit model for inference. The project supports dynamic story generation, panel creation, dialogue generation, and evaluation using several quality metrics.

---

## Overview

The project follows a multi-stage workflow:

```text
Base Models
     ↓
Stage 2 Fine-Tuning
     ↓
Model Merging
     ↓
16-bit Merged Model
     ↓
Story Generation
```

The final merged model is used to generate structured stories containing panels, scenes, dialogues, and character interactions.

---

## Model Configuration

The merged model is built from the following base models:

```python
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
        "display": "Qwen 2.5 0.5B (Test Only)",
        "context_window": 4096,
        "max_new_tokens": 600,
        "use_chat_template": True,
        "no_quantize": True,
    },
}
```

These models are fine-tuned and stored in `stage2_finetuned/`, then merged into a single 16-bit model located in `merged/`.

---

# Installation

## Create Virtual Environment

Recommended Python version:

```bash
Python 3.10
```

Create and activate a virtual environment:

### Windows

```bash
python -m venv py10
py10\Scripts\activate
```

### Linux / macOS

```bash
python -m venv py10
source py10/bin/activate
```

---

## Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

### GPU (CUDA 12.1)

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

### CPU Only

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

---

# Output model drive

drive: https://drive.google.com/drive/folders/11iTLqizx2rOP8t4RfgnbIm3FeFSg3Tsw?usp=sharing

# Story Generation

Configure the required variables in the `.env` file before running generation.

Run:

```bash
python story_gen.py
```

The script loads the final merged model and generates story outputs based on the provided configuration.

---

# Outputs

## story_output2

Generated using the GPU-trained merged model.

**Purpose:**

* High-quality story generation
* Production-ready outputs

---

## story_output3

Generated using the Tiny model (`Qwen 0.5B`).

**Purpose:**

* CPU inference
* Lightweight testing
* Quick debugging

---

## my_story.json

Contains dynamically generated story information, including:

* Panels
* Dialogues
* Character interactions
* Scene descriptions
* User-driven story content

---

# Scripts

## `story_gen.py`

Main entry point for story generation.

```bash
python story_gen.py
```

---

## `try.py`

Used to test story generation with individual models before merging.

Useful for:

* Model comparison
* Prompt testing
* Debugging

---

## `merge.py`

Merges fine-tuned models into a single inference model.

```bash
python merge.py
```

---

## `evaluate.py`

Evaluates generated stories and merged model performance.

Supported metrics include:

* Rule-based validation
* Hallucination detection
* Perplexity
* ROUGE
* BERTScore

---

# Evaluation

## Basic Evaluation

Runs rule-based checks and hallucination detection.

```bash
python evaluate.py
```

---

## Perplexity

Computes perplexity by loading the merged model.

```bash
python evaluate.py --perplexity
```

---

## NLP Metrics

Requires reference stories.

```bash
python evaluate.py --nlp --refs ref1.json ref2.json
```

---

## Complete Evaluation

Runs all available metrics.

```bash
python evaluate.py --all --refs ref1.json ref2.json
```

---

## Compare Models

Compare outputs from different models.

```bash
python evaluate.py --compare story_finetuned.json story_base.json --all
```

---

# Project Structure

```text
.
├── story_gen.py
├── try.py
├── merge.py
├── evaluate.py
├── requirements.txt
├── .env
│
├── merged/
│   └── Final 16-bit merged model
│
├── stage2_finetuned/
│   └── Fine-tuned model checkpoints
│
├── story_output2/
│   └── GPU generated stories
│
├── story_output3/
│   └── Tiny model generated stories
│
└── my_story.json
    └── Dynamic story output
```

---

# Requirements

* Python 3.10
* PyTorch
* Transformers
* Accelerate
* PEFT
* Datasets
* Additional dependencies listed in `requirements.txt`

---

# Notes

* The Tiny model is intended primarily for testing and CPU-based inference.
* The merged model provides the best overall story quality.
* Evaluation supports both automated rule-based metrics and NLP-based quality metrics.
* Dynamic story generation outputs are stored in JSON format for easy downstream processing.
