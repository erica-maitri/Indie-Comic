import os
os.environ["WANDB_DISABLED"] = "true"
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                              f1_score,
                              classification_report)
from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

# ── Labels ────────────────────────────────────────────────
LABEL2ID = {"sadness":0, "joy":1, "anger":2,
            "fear":3,    "love":4, "surprise":5}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ── 1. Load data ──────────────────────────────────────────
print("Loading combined dataset...")
df = pd.read_csv("combined_emotions.csv")
df = df[["text", "label"]].dropna()
df["label"] = df["label"].astype(int)

# FIX: 3-way split — train / val / test
# Previously test was used for both early stopping and final eval (data leak).
# Now val is used during training, test is held out for final evaluation only.
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── 2. Tokenizer ──────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset   = Dataset.from_pandas(val_df.reset_index(drop=True))
test_dataset  = Dataset.from_pandas(test_df.reset_index(drop=True))

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset   = val_dataset.map(tokenize, batched=True)
test_dataset  = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.remove_columns(["text"])
val_dataset = val_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])

train_dataset.set_format("torch")
val_dataset.set_format("torch")
test_dataset.set_format("torch")

train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

# ── 3. Model ──────────────────────────────────────────────
print("Loading XLM-RoBERTa...")
model = XLMRobertaForSequenceClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=6,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

# ── 4. Metrics ────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy": round(accuracy_score(labels, predictions), 4),
        "f1_macro": round(f1_score(labels, predictions,
                                    average="macro"), 4)
    }

# ── 5. Training Arguments ─────────────────────────────────
training_args = TrainingArguments(
    output_dir="./mood_weaver_model",

    # FIX: increased to 5 epochs — XLM-RoBERTa needs more time to converge.
    # Early stopping will still halt if val f1 stops improving.
    num_train_epochs=5,
    report_to="none",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,

    # FIX: renamed from evaluation_strategy (deprecated in recent transformers)
    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=100,
    save_total_limit=2,

    # FIX: enable mixed precision for ~2x faster training on GPU
    fp16=True,
)

# ── 6. Train ──────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,        # FIX: use val set, not test set
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("\nStarting training...")
trainer.train()

# ── 7. Evaluate on held-out test set ─────────────────────
print("\nFinal Evaluation on held-out test set:")
preds_output = trainer.predict(test_dataset)
preds = np.argmax(preds_output.predictions, axis=1)
print(classification_report(
    test_df["label"].values,
    preds,
    target_names=list(LABEL2ID.keys())
))

# ── 8. Save ───────────────────────────────────────────────
trainer.save_model("./mood_weaver_model")
tokenizer.save_pretrained("./mood_weaver_model")
print("\nModel saved to ./mood_weaver_model")