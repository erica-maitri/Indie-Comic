from PIL import Image
import numpy as np
from core.evaluation_suite import ModelEvaluator

img = Image.new('RGB', (224, 224), color='red')
evaluator = ModelEvaluator(device='cpu')
score = evaluator.compute_aesthetic_score(img)
print(f'Score: {score}')
