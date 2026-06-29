import ast
import pandas as pd
from datasets import load_dataset

# ── Label mapping ─────────────────────────────────────────
LABEL2ID = {
    "sadness": 0,
    "joy":     1,
    "anger":   2,
    "fear":    3,
    "love":    4,
    "surprise":5
}

# ── GoEmotions 28 → our 6 (used for English HuggingFace load) ────────────
GOEMOTIONS_MAP = {
    "sadness":       "sadness",
    "grief":         "sadness",
    "remorse":       "sadness",
    "disappointment":"sadness",
    "embarrassment": "sadness",

    "joy":           "joy",
    "amusement":     "joy",
    "excitement":    "joy",
    "gratitude":     "joy",
    "optimism":      "joy",
    "relief":        "joy",
    "pride":         "joy",

    "anger":         "anger",
    "annoyance":     "anger",
    "disapproval":   "anger",
    "disgust":       "anger",

    "fear":          "fear",
    "nervousness":   "fear",

    "love":          "love",
    "caring":        "love",
    "admiration":    "love",
    "desire":        "love",

    "surprise":      "surprise",
    "confusion":     "surprise",
    "realization":   "surprise",
    "curiosity":     "surprise",

    # neutral is intentionally excluded — model always predicts an emotion.
}

# ── GoEmotions label ID (0-27) → our 6 ───────────────────────────────────
# Used for Hindi CSVs that store labels as "[14]" style integer IDs.
GOEMOTIONS_ID_MAP = {
    0:  "love",     # admiration
    1:  "joy",      # amusement
    2:  "anger",    # anger
    3:  "anger",    # annoyance
    4:  "joy",      # approval
    5:  "love",     # caring
    6:  "surprise", # confusion
    7:  "surprise", # curiosity
    8:  "love",     # desire
    9:  "sadness",  # disappointment
    10: "anger",    # disapproval
    11: "anger",    # disgust
    12: "sadness",  # embarrassment
    13: "joy",      # excitement
    14: "fear",     # fear
    15: "joy",      # gratitude
    16: "sadness",  # grief
    17: "joy",      # joy
    18: "love",     # love
    19: "fear",     # nervousness
    20: "joy",      # optimism
    21: "joy",      # pride
    22: "surprise", # realization
    23: "joy",      # relief
    24: "sadness",  # remorse
    25: "sadness",  # sadness
    26: "surprise", # surprise
    27: None,       # neutral — skipped
}


# ════════════════════════════════════════════════════════
# PART 1: Load GoEmotions (English, from HuggingFace)
# ════════════════════════════════════════════════════════

def load_goemotions():
    print("Loading GoEmotions from HuggingFace...")

    dataset = load_dataset(
    "google-research-datasets/go_emotions",
    "simplified"
    )

    emotion_names = (
        dataset["train"]
        .features["labels"]
        .feature.names
    )

    rows = []

    for split in ["train", "validation", "test"]:
        for item in dataset[split]:

            if not item["labels"]:
                continue

            # GoEmotions is multi-label — try all labels, keep the first
            # that maps to our 6 emotions (not just index 0).
            mapped = None
            for label_id in item["labels"]:
                raw_emotion = emotion_names[label_id]
                mapped = GOEMOTIONS_MAP.get(raw_emotion)
                if mapped:
                    break

            if mapped:
                rows.append({
                    "text":       item["text"],
                    "label":      LABEL2ID[mapped],
                    "label_name": mapped,
                    "language":   "en"
                })

    df = pd.DataFrame(rows)

    # Remove duplicate tweets present across splits
    before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"Dropped {before - len(df)} duplicate rows from GoEmotions.")

    print(f"GoEmotions loaded: {len(df)} samples")
    print(df["label_name"].value_counts())

    return df


# ════════════════════════════════════════════════════════
# PART 2: Load Hindi CSV (labels stored as "[14]" ID lists)
# ════════════════════════════════════════════════════════

def load_hindi_csv(file_path):
    """
    Loads a Hindi GoEmotions-style CSV where:
      - 'text'   column holds the Hindi sentence
      - 'labels' column holds label IDs as a string like "[14]" or "[2, 5]"
    """
    print(f"\nLoading {file_path}...")
    df_raw = pd.read_csv(file_path)

    print(f"Columns found: {df_raw.columns.tolist()}")
    print(f"Shape: {df_raw.shape}")
    print(df_raw.head(3))

    # Defensive column check
    if "text" not in df_raw.columns:
        raise ValueError(
            f"'text' column not found. Available: {df_raw.columns.tolist()}"
        )
    if "labels" not in df_raw.columns:
        raise ValueError(
            f"'labels' column not found. Available: {df_raw.columns.tolist()}"
        )

    rows = []
    skipped_neutral = 0
    skipped_parse   = 0

    for _, row in df_raw.iterrows():
        text = str(row["text"]).strip()

        # Parse "[27]" or "[2, 5]" → Python list of ints
        try:
            label_ids = ast.literal_eval(str(row["labels"]))
            if not isinstance(label_ids, list):
                label_ids = [label_ids]
        except Exception:
            skipped_parse += 1
            continue

        # Take the first label ID that maps to our 6 emotions
        mapped = None
        all_neutral = True
        for lid in label_ids:
            result = GOEMOTIONS_ID_MAP.get(int(lid))
            if result is not None:
                mapped = result
                all_neutral = False
                break
            elif result is None and int(lid) == 27:
                pass  # neutral, keep checking other labels

        if all_neutral or mapped is None:
            skipped_neutral += 1
            continue

        rows.append({
            "text":       text,
            "label":      LABEL2ID[mapped],
            "label_name": mapped,
            "language":   "hi"
        })

    df = pd.DataFrame(rows)

    print(f"\nLoaded:          {len(df)} samples")
    print(f"Skipped neutral: {skipped_neutral}")
    print(f"Skipped parse errors: {skipped_parse}")
    print(df["label_name"].value_counts())

    return df


# ════════════════════════════════════════════════════════
# PART 3: Combine and Save
# ════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── English ───────────────────────────────────────────
    df_english = load_goemotions()

    # Cap per-class to reduce joy dominance (7 labels map to joy)
    MAX_PER_CLASS = 4000
    df_english_balanced = (
        df_english
        .groupby("label_name", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), MAX_PER_CLASS), random_state=42))
        .reset_index(drop=True)
    )
    print(f"\nEnglish after balancing: {len(df_english_balanced)} samples")
    print(df_english_balanced["label_name"].value_counts())

    # ── Hindi ─────────────────────────────────────────────
    df_train = load_hindi_csv("emoHi-train.csv")
    df_valid = load_hindi_csv("emoHi-valid.csv")
    df_test  = load_hindi_csv("emoHi-test.csv")

    df_hindi = pd.concat(
        [df_train, df_valid, df_test],
        ignore_index=True
    )

    hindi_n = min(20000, len(df_hindi))
    df_hindi_small = df_hindi.sample(n=hindi_n, random_state=42)
    print(f"\nHindi sampled: {len(df_hindi_small)} samples")

    # ── Combine ───────────────────────────────────────────
    df_combined = pd.concat(
        [df_english_balanced, df_hindi_small],
        ignore_index=True
    )

    # Shuffle
    df_combined = (
        df_combined
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    # Save main dataset
    df_combined.to_csv("combined_emotions.csv", index=False)

    # Save metadata for debugging
    pd.DataFrame([{
        "total":              len(df_combined),
        "english":            len(df_english_balanced),
        "hindi":              len(df_hindi_small),
        "language_breakdown": df_combined["language"].value_counts().to_dict(),
        "emotion_breakdown":  df_combined["label_name"].value_counts().to_dict(),
    }]).to_csv("combined_emotions_meta.csv", index=False)

    print("\n✅ Combined dataset saved!")
    print(f"Total samples: {len(df_combined)}")

    print("\nLanguage breakdown:")
    print(df_combined["language"].value_counts())

    print("\nEmotion breakdown:")
    print(df_combined["label_name"].value_counts())