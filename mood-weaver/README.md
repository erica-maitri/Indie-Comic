# Mood Weaver AI

Mood Weaver AI is a multilingual emotion intelligence system that analyzes user text and predicts emotional states using a fine-tuned XLM-RoBERTa model.

The model supports:

- English
- Hindi
- Hinglish (partially)

and predicts the following emotions:

- Sadness
- Joy
- Anger
- Fear
- Love
- Surprise

---

## Project Overview

The project combines English and Hindi emotion datasets and fine-tunes XLM-RoBERTa to perform multilingual emotion classification.

### Datasets Used

#### GoEmotions (English)

A large-scale English emotion dataset containing 28 emotion labels.

#### EmoHi

A Hindi emotion dataset derived from GoEmotions-style annotations.

---

## Dataset Statistics

| Dataset | Samples |
|----------|----------|
| English | 20,068 |
| Hindi | 20,000 |
| Total | 40,068 |

---

## Model

### XLM-RoBERTa Base

XLM-RoBERTa is a multilingual transformer model trained on 100+ languages.

The model was fine-tuned on approximately 40k emotion-labeled examples.

---

## Emotion Classes

The original datasets contained many fine-grained emotions.

These were mapped into six core emotions:

| Emotion | Label |
|----------|----------|
| Sadness | 0 |
| Joy | 1 |
| Anger | 2 |
| Fear | 3 |
| Love | 4 |
| Surprise | 5 |

---

## Training Configuration

- Model: XLM-RoBERTa Base
- Epochs: 5
- Learning Rate: 2e-5
- Batch Size: 8
- Optimizer: AdamW
- Early Stopping: Enabled

---

## Results

### Test Set Performance

| Metric | Score |
|----------|----------|
| Accuracy | 76% |
| Macro F1 Score | 0.74 |

### Per-Class F1 Scores

| Emotion | F1 Score |
|----------|----------|
| Sadness | 0.72 |
| Joy | 0.79 |
| Anger | 0.71 |
| Fear | 0.67 |
| Love | 0.79 |
| Surprise | 0.75 |

---

## Confusion Matrix

The trained model was evaluated using a confusion matrix and classification report on a held-out test set.

---

## Folder Structure

```text
Mood-Weaver/
│
├── model/
│   └── mood_weaver_model/
│
├── datasets/
│   └── combined_emotions.csv
│
├── scripts/
│   ├── prepare_data.py
│   ├── train_model.py
│   └── mood_analyzer.py
│
├── results/
│   ├── confusion_matrix.pdf
│   └── classification_report.txt
│
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <repository-url>

cd Mood-Weaver

pip install -r requirements.txt
```

---

## Running Inference

```python
from transformers import pipeline

emotion_pipe = pipeline(
    "text-classification",
    model="./model/mood_weaver_model",
    top_k=None
)

print(emotion_pipe("I feel very happy today"))
```

---

## Example Predictions

### English

Input:

```text
I feel very happy today.
```

Output:

```text
Joy
```

### Hindi

Input:

```text
मुझे बहुत दुख हो रहा है।
```

Output:

```text
Sadness
```

### Hinglish

Input:

```text
Yaar bahut bura lag raha hai.
```

Output:

```text
Sadness
```

---

## Future Improvements

- Real-time emotion tracking
- Emotion trends over time
- Mental wellness insights
- Web dashboard integration
- Voice-based emotion detection

---

## Author

Priyani Rajvanshi

B.Tech Computer Science Engineering

Mood Weaver AI Project