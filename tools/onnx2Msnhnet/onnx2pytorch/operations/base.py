from abc import ABC

from torch import nn


class Operator(nn.Module, ABC):
    @staticmethod
    def get_axis(input_shape, input_feature_axis):
        """
        Parameters
        ----------
        input_shape: torch.Size
        input_feature_axis: int

        Returns
        -------
        axis: tuple
            Axis to aggregate over.
        """
        if input_feature_axis < 0:
            input_feature_axis += len(input_shape)
            # select and sum all axes except the feature one
        axis = set(range(len(input_shape))) - {input_feature_axis}
        return tuple(axis)


class OperatorWrapper(Operator, ABC):
    def __init__(self, op):
        """
        This class enables any function to become a subclass of nn.Module
        The class name is equal to the op.__name__

        Parameters
        ----------
        op: function or builtin_function_or_method
            Any torch function. It is used in-place of forward method.
        """
        self.forward = op
        self.__class__.__name__ = op.__name__
        super().__init__()
