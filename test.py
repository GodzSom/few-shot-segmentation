import torch
from FF import MultiClassFF
from torchsummary import summary

model = MultiClassFF(5000, [2000, 300, 150], 4).cuda()

summary(model, input_size=(12, 5000))
