import torch
from torch import nn

from onnx2pytorch.operations.base import Operator
from onnx2pytorch.utils import assign_values_to_dim, get_selection


class Reshape(Operator):
    """
    In the initial pass it stores the initial_input_shape.
    It uses it to infer the new reshape value from a
    smaller pruned input in the following passes.
    """

    def __init__(self, shape=None, keep_size=True):
        super().__init__()
        self.shape = shape
        self.initial_input_shape = None
        self.feature_dim = -1
        self.input_indices = None
        self.placeholder = None
        self.keep_size = keep_size

    def forward(self, input: torch.Tensor, shape=None):
        shape = shape if shape is not None else self.shape
        # This raises RuntimeWarning: iterating over a tensor.
        shape = [x if x != 0 else input.size(i) for i, x in enumerate(shape)]
        inp_shape = torch.tensor(input.shape)
        if self.initial_input_shape is None:
            self.initial_input_shape = inp_shape
        elif len(shape) == 2 and shape[-1] == -1:
            pass
        elif torch.equal(self.initial_input_shape, inp_shape):
            # shape did not change
            pass
        elif self.input_indices is not None:
            self.placeholder *= 0
            selection = get_selection(self.input_indices, self.feature_dim)
            self.placeholder[selection] += input
            input = self.placeholder
        else:
            # if input changed the reshaped shape changes as well
            c = torch.true_divide(inp_shape, self.initial_input_shape)
            if len(c) < len(shape) and shape[0] == 1:
                c = torch.cat((torch.tensor([1]), c))
            shape = (c * torch.tensor(shape)).to(int)

        return torch.reshape(input, tuple(shape))

    def set_input_indices(self, input):
        input_shape = input[0].shape
        if self.feature_dim < 0:
            self.feature_dim += len(input_shape)
        axis = self.get_axis(input_shape, self.feature_dim)
        mask = input[0] != 0
        s = mask.sum(axis=tuple(axis))
        mask = s != 0
        (non_zeros,) = torch.where(mask)
        self.input_indices = non_zeros
        self.placeholder = nn.Parameter(
            torch.zeros(
                *self.initial_input_shape, device=input[0].device, dtype=input[0].dtype
            ),
            requires_grad=False,
        )

    def extra_repr(self) -> str:
        return "shape={}".format(self.shape)
