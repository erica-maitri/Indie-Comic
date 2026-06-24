import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')

img = Image.new('RGB', (224, 224), color='red')
inputs = processor(images=img, return_tensors='pt')

with torch.no_grad():
    output = model(**inputs)
    print(output.pooler_output)
print('Success')
