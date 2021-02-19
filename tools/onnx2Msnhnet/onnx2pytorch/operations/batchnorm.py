import warnings

from torch.nn.modules.batchnorm import _BatchNorm


class BatchNormUnsafe(_BatchNorm):
    def __init__(self, *args, spatial=True, **kwargs):
        if not spatial:
            warnings.warn("Non spatial BatchNorm not implemented.", RuntimeWarning)
        super().__init__(*args, **kwargs)

    def _check_input_dim(self, input):
        return
