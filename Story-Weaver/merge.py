# merge_and_save.py
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="moodweaver_stage2_finetuned",
    max_seq_length=2048,
    load_in_4bit=True,
)

model.save_pretrained_merged(
    "moodweaver_stage2_merged",
    tokenizer,
    save_method="merged_16bit",   
)
print("Merged model saved → moodweaver_stage2_merged/")