import io
import torch
import onnx
import torch.onnx

from onnx2pytorch import ConvertModel


def to_onnx(model, inp_size, device=torch.device("cpu"), do_constant_folding=False):
    if isinstance(inp_size, (tuple, list)) and not isinstance(inp_size[0], int):
        input_image = tuple([torch.rand(i, device=device) for i in inp_size])
    else:
        input_image = torch.rand(inp_size, device=device)

    model.to(device)
    bitstream = io.BytesIO()
    torch.onnx.export(
        model,
        input_image,
        bitstream,
        export_params=True,
        opset_version=11,
        do_constant_folding=do_constant_folding,
        input_names=["input"],
        output_names=["output"],
    )
    return onnx.ModelProto.FromString(bitstream.getvalue())


def to_converted(model, inp_size):
    onnx_model = to_onnx(model, inp_size)
    model = ConvertModel(onnx_model)
    return model
