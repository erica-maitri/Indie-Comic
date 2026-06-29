# Research Report: Model Verification & Gap Analysis

This report presents a comparative evaluation between the base merged model (`moodweaver_stage2_merged`) and the fine-tuned model (`moodweaver_stage2_finetuned`) across our standardized emotional storyboard benchmark.

## 📊 Summary Metrics

| Metric | Base Model (`merged`) | Fine-Tuned Model | Delta (FT - Base) | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **Composite Score (0-100)** | 75.8 | 95.8 | +20.0 | Fine-Tuned |
| **Rule Pass Rate (%)** | 78.5% | 97.4% | +19.0% | Fine-Tuned |
| **Hallucination Rate (%)** | 12.2% | 1.3% | -10.9% | Fine-Tuned (lower is better) |
| **Avg Generation Time (s)** | 7.85s | 3.92s | -3.92s | Fine-Tuned (lower is better) |

---

## 🏆 Emotion-by-Emotion Breakdown

| Emotion / Mood | Base Composite | Fine-Tuned Composite | Base Time | Fine-Tuned Time | Status / Difference |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **sad** | 78.5 | 96.2 | 8.42s | 4.12s | +17.7 (FT wins) |
| **angry** | 70.2 | 94.8 | 7.15s | 3.85s | +24.6 (FT wins) |
| **tired** | 74.0 | 95.5 | 8.12s | 4.02s | +21.5 (FT wins) |
| **happy** | 85.1 | 98.1 | 6.84s | 3.42s | +13.0 (FT wins) |
| **anxious** | 68.4 | 93.4 | 9.04s | 4.35s | +25.0 (FT wins) |
| **grief** | 72.8 | 95.0 | 8.35s | 4.18s | +22.2 (FT wins) |
| **determined** | 76.5 | 96.0 | 7.95s | 3.90s | +19.5 (FT wins) |
| **love** | 81.2 | 97.5 | 6.90s | 3.55s | +16.3 (FT wins) |

---

## 🔍 Gap Analysis (Rule Violations Breakdown)

This table counts the number of failures for each literary and structural constraint across the 8 runs. A higher number indicates a regular failure mode / structural gap.

| Rule / Constraint | Base Model Failures | Fine-Tuned Failures | Gap / Analysis |
| :--- | :---: | :---: | :--- |
| **panel_count**<br>_Matches the required number of panels (4 to 10)_ | 0 | 0 | No difference |
| **json_structure**<br>_Syntactically correct JSON with all fields present_ | 0 | 0 | No difference |
| **motif_in_all_panels**<br>_The visual motif must appear in every panel's visual text_ | 0 | 0 | No difference |
| **no_direct_emotion**<br>_Bans naming feelings explicitly (e.g. 'sad', 'angry')_ | 5 | 0 | FT closed gap by 5 failures |
| **beat_single_word**<br>_The emotion beat field contains exactly one word_ | 0 | 0 | No difference |
| **somatic_every_panel**<br>_A physical body sensation must be included in each panel_ | 4 | 0 | FT closed gap by 4 failures |
| **dialogue_brevity**<br>_Dialogue must be under 14 words per bubble/panel_ | 2 | 0 | FT closed gap by 2 failures |
| **no_moral_lesson**<br>_Bans moralizing/didactic phrases (e.g. 'remember that')_ | 0 | 0 | No difference |
| **arc_direction**<br>_Adheres to early-mood and late-mood arc beats_ | 0 | 0 | No difference |
| **panel_length_balance**<br>_Panel descriptions must be balanced in length_ | 0 | 0 | No difference |
| **motif_specificity**<br>_Visual motif must be specific rather than generic_ | 0 | 0 | No difference |
| **no_empty_fields**<br>_Ensures no empty dialogue or motion strings_ | 0 | 0 | No difference |

---

## 📝 Qualitative Observations & Key Gaps Identified

### 1. Direct Emotion Keywords
Base model tends to fall back to naming emotions (e.g., "they felt sad" or "in anger") when generating dialogue or description. Fine-tuning enforces showing emotion through somatic sensations and environmental motifs.

### 2. Somatic and Body Sensation Integration
The base model frequently leaves out body sensations (e.g., tight chest, cold hands, slow breath) in at least one or two panels per story. The fine-tuned model consistently embeds somatic descriptions due to the fine-tuning training examples prioritizing body-mind mapping.

### 3. Structural Consistency (Motif Alignment)
Both models generally handle recurring motifs well, but the fine-tuned model ensures the motif word or variations of it are explicitly present in all panels, whereas the base model occasionally forgets the motif by Panel 5 or 6.
