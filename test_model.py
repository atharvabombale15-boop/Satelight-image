import torch
from model import SiameseUNet

model = SiameseUNet()

t1 = torch.randn(1, 3, 256, 256)
t2 = torch.randn(1, 3, 256, 256)

output = model(t1, t2)

print("Output shape:", output.shape)