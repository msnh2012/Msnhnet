import warnings

import torch
from torch import nn

from onnx2pytorch.utils import is_constant, get_selection
from onnx2pytorch.operations.base import Operator


class Add(Operator):
    def __init__(self, input_shape=None, input_indices=None, feature_dim=1):
        self.input_shape = input_shape
        self.input_indices = input_indices
        self.feature_dim = feature_dim  # 2 for transformers else 1
        self.out = None

        if input_shape and input_indices:
            self.out = torch.zeros(input_shape)

        super().__init__()

    def forward(self, *input):
        if self.input_indices:
            out = self.out * 0
            for inp, idx in zip(input, self.input_indices):
                selection = get_selection(idx, self.feature_dim)
                out[selection] += inp
            return out

        # Reorder input so that the matrix is first
        if is_constant(input[0]):
            input = sorted(input, key=lambda x: -len(x.shape))
        # Reorder input so that the broadcasted matrix is last
        elif all(x == 1 for x in input[0].shape):
            input = sorted(input, key=lambda x: -sum(x.shape))
        out = input[0].clone()
        for inp in input[1:]:
            out += inp
        return out

    def set_input_indices(self, input):
        assert isinstance(input, (list, tuple))

        # If all but one of the inputs are constants do nothing
        # One tensor can easily add together with any number of constants
        if sum(is_constant(inp) for inp in input) >= len(input) - 1:
            return

        input_shape = input[0].shape
        if not all(input_shape == inp.shape for inp in input[1:]):
            warnings.warn("Addition might be corrupted.", RuntimeWarning)
        assert all(
            is_constant(inp) or input_shape[-1] == inp.shape[-1] for inp in input
        )

        # HACK
        while self.feature_dim >= len(input_shape):
            self.feature_dim -= 1
        axis = self.get_axis(input_shape, self.feature_dim)

        input_indices = []
        for inp in input:
            mask = inp != 0
            if len(inp.shape) > 1:
                # Where mask is == 0, the complete input channel can be removed
                s = mask.sum(axis=tuple(axis))
                # If inp is triangular matrix do not remove zero rows.
                # Immediately return.
                seq = torch.arange(len(s))
                if torch.equal(s, seq) or torch.equal(s.flip(0), seq):
                    return
                mask = s != 0
            (non_zeros,) = torch.where(mask)
            input_indices.append(non_zeros)

        # if all elements are non zero, no indices necessary
        if all(len(i) == len(mask) for i in input_indices):
            return

        unique_indices = torch.cat(input_indices).unique()
        input_shape = list(input[0].shape)
        input_shape[self.feature_dim] = len(unique_indices)

        _, input_indices[0] = torch.where(
            input_indices[0][:, None] == unique_indices[None]
        )
        _, input_indices[1] = torch.where(
            input_indices[1][:, None] == unique_indices[None]
        )

        self.input_indices = input_indices
        self.input_shape = tuple(input_shape)
        self.out = nn.Parameter(
            torch.zeros(self.input_shape, device=input[0].device, dtype=input[0].dtype),
            requires_grad=False,
        )

    def __str__(self):
        if self.input_indices:
            return "Add({}, {}, {})".format(
                tuple(self.input_shape),
                len(self.input_indices[0]),
                len(self.input_indices[1]),
            )
        else:
            return "Add(None, None)"
