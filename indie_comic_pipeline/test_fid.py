import torch
from torchmetrics.image.fid import FrechetInceptionDistance
fid = FrechetInceptionDistance(feature=64)
real = torch.randint(0, 255, (1, 3, 299, 299), dtype=torch.uint8)
fake = torch.randint(0, 255, (1, 3, 299, 299), dtype=torch.uint8)
fid.update(real, real=True)
fid.update(fake, real=False)
try:
    score = fid.compute()
    print(score)
except Exception as e:
    print(repr(e))
