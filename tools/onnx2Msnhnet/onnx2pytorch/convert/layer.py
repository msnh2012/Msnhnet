import torch
from torch import nn
from onnx import numpy_helper

from onnx2pytorch.operations import BatchNormUnsafe, InstanceNormUnsafe
from onnx2pytorch.convert.attribute import extract_attributes, extract_attr_values


def extract_params(params):
    """Extract weights and biases."""
    param_length = len(params)
    if param_length == 1:
        weight = params[0]
        bias = None
    elif param_length == 2:
        weight = params[0]
        bias = params[1]
    else:
        raise ValueError("Unexpected number of parameters: {}".format(param_length))
    return weight, bias


def load_params(layer, weight, bias):
    """Load weight and bias to a given layer from onnx format."""
    layer.weight.data = torch.from_numpy(numpy_helper.to_array(weight))
    if bias is not None:
        layer.bias.data = torch.from_numpy(numpy_helper.to_array(bias))


def convert_layer(node, layer_type, params=None):
    """Use to convert Conv, MaxPool, AvgPool layers."""
    assert layer_type in [
        "Conv",
        "ConvTranspose",
        "MaxPool",
        "AvgPool",
    ], "Incorrect layer type: {}".format(layer_type)
    kwargs = extract_attributes(node)
    kernel_size_length = len(kwargs["kernel_size"])
    try:
        layer = getattr(nn, "{}{}d".format(layer_type, kernel_size_length))
    except AttributeError:
        raise ValueError(
            "Unexpected length of kernel_size dimension: {}".format(kernel_size_length)
        )

    if params:
        pad_layer = None
        weight, bias = extract_params(params)
        kwargs["bias"] = bias is not None
        kwargs["in_channels"] = weight.dims[1] * kwargs.get("groups", 1)
        kwargs["out_channels"] = weight.dims[0]

        if layer_type == "ConvTranspose":
            kwargs["in_channels"], kwargs["out_channels"] = (
                kwargs["out_channels"],
                kwargs["in_channels"],
            )

        # if padding is a layer, remove from kwargs and prepend later
        if isinstance(kwargs["padding"], nn.Module):
            pad_layer = kwargs.pop("padding")

        # initialize layer and load weights
        layer = layer(**kwargs)
        load_params(layer, weight, bias)
        if pad_layer is not None:
            layer = nn.Sequential(pad_layer, layer)
    else:
        # initialize operations without parameters (MaxPool, AvgPool, etc.)
        layer = layer(**kwargs)

    return layer


def convert_batch_norm_layer(node, params):
    kwargs = extract_attributes(node)
    layer = BatchNormUnsafe  # Input dimension check missing, not possible before forward pass

    kwargs["num_features"] = params[0].dims[0]
    # initialize layer and load weights
    layer = layer(**kwargs)
    key = ["weight", "bias", "running_mean", "running_var"]
    for key, value in zip(key, params):
        getattr(layer, key).data = torch.from_numpy(numpy_helper.to_array(value))

    return layer


def convert_instance_norm_layer(node, params):
    kwargs = extract_attributes(node)
    # Skips input dimension check, not possible before forward pass
    layer = InstanceNormUnsafe

    kwargs["num_features"] = params[0].dims[0]
    # initialize layer and load weights
    layer = layer(**kwargs)
    key = ["weight", "bias"]
    for key, value in zip(key, params):
        getattr(layer, key).data = torch.from_numpy(numpy_helper.to_array(value))

    return layer


def convert_linear_layer(node, params):
    """Convert linear layer from onnx node and params."""
    # Default Gemm attributes
    dc = dict(
        transpose_weight=True,
        transpose_activation=False,
        weight_multiplier=1,
        bias_multiplier=1,
    )
    dc.update(extract_attributes(node))
    for attr in node.attribute:
        if attr.name in ["transA"] and extract_attr_values(attr) != 0:
            raise NotImplementedError(
                "Not implemented for attr.name={} and value!=0.".format(attr.name)
            )

    kwargs = {}
    weight, bias = extract_params(params)
    kwargs["bias"] = bias is not None
    kwargs["in_features"] = weight.dims[1]
    kwargs["out_features"] = weight.dims[0]

    # initialize layer and load weights
    layer = nn.Linear(**kwargs)
    load_params(layer, weight, bias)

    # apply onnx gemm attributes
    if dc.get("transpose_weight"):
        layer.weight.data = layer.weight.data.t()

    layer.weight.data *= dc.get("weight_multiplier")
    if layer.bias is not None:
        layer.bias.data *= dc.get("bias_multiplier")

    return layer
