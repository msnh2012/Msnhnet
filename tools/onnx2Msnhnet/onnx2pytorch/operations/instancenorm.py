import warnings

from torch.nn.modules.instancenorm import _InstanceNorm


class InstanceNormUnsafe(_InstanceNorm):
    """Skips dimension check."""

    def __init__(self, *args, affine=True, **kwargs):
        super().__init__(*args, affine=affine, **kwargs)

    def _check_input_dim(self, input):
        return
