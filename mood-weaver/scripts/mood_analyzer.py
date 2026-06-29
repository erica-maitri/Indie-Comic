import re
from transformers import pipeline

emotion_pipe = pipeline(
    "text-classification",
    model="./mood_weaver_model",
    top_k=None
)

# Confidence threshold — below this the result is flagged as uncertain.
CONFIDENCE_THRESHOLD = 0.40

SOMATIC_PHRASES = [
    "chest", "heavy", "lump", "throat", "racing heart",
    "shallow", "breathing", "stomach", "numb", "tight",
    "shaking", "dizzy", "weight", "hollow", "cold",
    "सीना", "भारी", "सांस", "पेट", "कंपन", "ठंडा"
]

INTENSIFIERS = [
    "very", "extremely", "absolutely", "completely",
    "बहुत", "अत्यंत", "बिल्कुल", "पूरी तरह"
]

# FIX: compile word-boundary patterns once at import time for efficiency.
# This prevents false matches like "chest" inside "Manchester", or
# "cold" inside "scold". Multi-word phrases use non-word-boundary anchors
# only at the outer edges.
_SOMATIC_PATTERNS = [
    re.compile(rf'(?<!\w){re.escape(p)}(?!\w)', re.IGNORECASE)
    for p in SOMATIC_PHRASES
]

_INTENSIFIER_PATTERNS = [
    re.compile(rf'(?<!\w){re.escape(w)}(?!\w)', re.IGNORECASE)
    for w in INTENSIFIERS
]


def analyze_mood(user_text: str) -> dict:
    results = emotion_pipe(user_text)[0]
    results.sort(key=lambda x: x["score"], reverse=True)

    primary   = results[0]
    secondary = results[1:3]

    # FIX: continuous ambivalence score instead of binary 0.1/0.5.
    # Measures how close the top-2 emotions are to each other.
    # Score near 1.0 = very ambivalent; near 0.0 = clear dominant emotion.
    ambivalence = round(
        max(0.0, 1.0 - (primary["score"] - results[1]["score"]) * 2), 2
    )

    # FIX: use compiled word-boundary patterns
    intensifier_found = any(pat.search(user_text) for pat in _INTENSIFIER_PATTERNS)
    intensity = round(
        min(1.0, primary["score"] + (0.1 if intensifier_found else 0)), 2
    )

    # FIX: word-boundary somatic matching (avoids substring false positives)
    somatic_markers = [
        phrase
        for phrase, pat in zip(SOMATIC_PHRASES, _SOMATIC_PATTERNS)
        if pat.search(user_text)
    ]

    # FIX: flag low-confidence predictions instead of silently returning them.
    # This is important because the model has no "neutral" class and will
    # always pick the closest emotion even on ambiguous or off-topic text.
    uncertain = primary["score"] < CONFIDENCE_THRESHOLD

    return {
        "primary_emotion":    primary["label"],
        "primary_confidence": round(primary["score"], 2),
        "uncertain":          uncertain,   # True if model is likely guessing
        "secondary_emotions": [
            {"emotion": s["label"], "weight": round(s["score"], 2)}
            for s in secondary
        ],
        "ambivalence_score": ambivalence,
        "somatic_markers":   somatic_markers,
        "intensity":         intensity,
        "raw_probabilities": {
            r["label"]: round(r["score"], 3) for r in results
        }
    }


if __name__ == "__main__":
    test_inputs = [
        "I feel completely hollow and exhausted",
        "मुझे बहुत दुख हो रहा है आज",
        # NOTE: Hinglish (romanised Hindi) may misclassify — the training
        # data uses Devanagari script Hindi, not romanised Hindi.
        # Consider adding a Hinglish dataset (e.g. SentiRaama) for better coverage.
        "yaar bahut bura lag raha hai",
        "मैं बहुत खुश हूं आज",
        "I am so angry nobody listens",
        # Edge cases to test uncertainty flag and somatic matching
        "okay",                              # should be uncertain
        "My chest feels tight and I'm shaking",  # should catch somatic markers
        "Manchester United won the match",   # "chest" must NOT match here
    ]

    for text in test_inputs:
        result = analyze_mood(text)
        flag = " ⚠️ uncertain" if result["uncertain"] else ""
        print(f"Input:    {text}")
        print(f"Emotion:  {result['primary_emotion']} "
              f"({result['primary_confidence']}){flag}")
        print(f"Ambival:  {result['ambivalence_score']}")
        print(f"Somatic:  {result['somatic_markers']}")
        print()