from torch.nn.functional import one_hot
from onnx2pytorch.operations.base import Operator


class OneHot(Operator):
    def __init__(self, dim=-1, non_zero_values_only=False):
        self.dim = dim
        self.non_zero_values_only = non_zero_values_only
        super().__init__()

    def forward(self, indices, depth, values):
        if self.non_zero_values_only:
            off_value, on_value = -1, 1
        else:
            off_value, on_value = values
        out = one_hot(indices.to(int), depth.to(int).item())
        out = out * (on_value - off_value) + off_value

        rank = len(indices.shape)
        if self.dim < 0:
            self.dim += rank + 1
        if not rank == self.dim:  # permute only if dim not last dimension
            order = list(range(len(indices.shape)))
            order.insert(self.dim, -1)
            out = out.permute(order)
        return out
