import torch
from torch import nn


class GlobalAveragePool(nn.Module):
    def forward(self, input: torch.Tensor):
        spatial_shape = input.ndimension() - 2
        dim = tuple(range(spatial_shape, spatial_shape + 2))
        return torch.mean(input, dim=dim, keepdim=True)
