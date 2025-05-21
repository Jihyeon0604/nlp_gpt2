import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=32):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scale = alpha / r

    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.scale
