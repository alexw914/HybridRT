# -----------------------------------------------------------------------------
# HybridRT | C++20 Model Inference Framework
# File: assets/template/yolov5_export_raw.py
# Description: Exposes fixed YOLOv5 v7.0 ONNX convolution heads as FP32 logits.
# -----------------------------------------------------------------------------
"""Convert the official YOLOv5 v7.0 ONNX export to three raw NCHW heads.

    python yolov5_export_raw.py yolov5n.onnx YoloV5Detect.onnx

The v7.0 release asset uses FP16; weights and constants retain their values
when promoted to FP32 for the host decoder. No angle channel is synthesized.
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def promote_tensor(tensor):
    if tensor.data_type == onnx.TensorProto.FLOAT16:
        tensor.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(tensor).astype(np.float32), tensor.name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.target.resolve():
        parser.error("source and target must differ")
    model = onnx.shape_inference.infer_shapes(onnx.load(args.source))
    heads = [f"/model.24/m.{i}/Conv_output_0" for i in range(3)]
    available = {name for node in model.graph.node for name in node.output}
    if not all(name in available for name in heads) or len(model.graph.input) != 1:
        parser.error("expected the official YOLOv5 v7.0 Detect graph")
    model = onnx.utils.Extractor(model).extract_model([model.graph.input[0].name], heads)
    names = {name: f"head{i}" for i, name in enumerate(heads)}
    for node in model.graph.node:
        for values in (node.input, node.output):
            for i, value in enumerate(values):
                if value in names:
                    values[i] = names[value]
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                promote_tensor(attribute.t)
            elif node.op_type == "Cast" and attribute.name == "to" and attribute.i == onnx.TensorProto.FLOAT16:
                attribute.i = onnx.TensorProto.FLOAT
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        value.name = names.get(value.name, value.name)
        if value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    for tensor in model.graph.initializer:
        promote_tensor(tensor)
    onnx.checker.check_model(model)
    args.target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.target)


if __name__ == "__main__":
    main()
