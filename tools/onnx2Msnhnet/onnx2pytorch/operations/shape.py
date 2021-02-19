import torch
from torch import nn


class Shape(nn.Module):
    def forward(self, input: torch.Tensor):
        return torch.tensor(input.shape)
