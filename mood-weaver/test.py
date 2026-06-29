from transformers import pipeline

emotion_pipe = pipeline(
    "text-classification",
    model="./model/mood_weaver_model",
    top_k=None
)

print(emotion_pipe("I feel very happy today"))
print(emotion_pipe("मुझे बहुत दुख हो रहा है"))
print(emotion_pipe("Yaar bahut bura lag raha hai"))