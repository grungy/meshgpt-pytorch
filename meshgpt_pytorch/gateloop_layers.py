import torch
import torch.nn as nn
from meshgpt_pytorch.helpers import exists, default

class GateLoopBlock(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([GateLayer(dim) for _ in range(depth)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

class GateLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim, dim)
        self.act = nn.Sigmoid()

    def forward(self, x, mask=None):
        gate = self.act(self.gate(x))
        return x * gate 