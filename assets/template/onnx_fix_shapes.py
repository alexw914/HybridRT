# -----------------------------------------------------------------------------
# HybridRT | C++20 Model Inference Framework
# File: assets/template/onnx_fix_shapes.py
# Description: Fixes ONNX input shapes and validates fixed output annotations.
# -----------------------------------------------------------------------------
"""Resolve fixed-shape exports without folding floating-point calculations.

    python onnx_fix_shapes.py source.onnx target.onnx --input images=1,3,640,640

Only use for graphs whose output counts are fixed, such as RT-DETR TopK,
RTMPose, or RTMO after rtmo_fix_outputs.py. Data-dependent selection operators
are rejected. Output shapes are checked on zero and random inputs before
remaining symbolic output annotations are replaced by their observed sizes.
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--input", action="append", required=True, metavar="NAME=N,C,H,W")
    args = parser.parse_args()
    if args.source.resolve() == args.target.resolve():
        parser.error("source and target must differ")
    shapes = {}
    for spec in args.input:
        name, separator, dimensions = spec.partition("=")
        parts = dimensions.split(",")
        if not separator or not name or not all(part.isdigit() and int(part) > 0 for part in parts):
            parser.error("input shape must use NAME=positive,dimensions")
        shapes[name] = [int(part) for part in parts]
    model = onnx.load(args.source)
    if any(node.op_type in {"NonMaxSuppression", "NonZero", "Compress", "Unique"} for node in model.graph.node):
        parser.error("data-dependent selection must be replaced by a fixed-output export first")
    if set(shapes) != {value.name for value in model.graph.input}:
        parser.error("provide a shape for every model input")
    model, valid = onnxsim.simplify(
        model, check_n=1, overwrite_input_shapes=shapes,
        perform_optimization=False, skip_constant_folding=True)
    if not valid:
        parser.error("onnxsim equivalence check failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    if any(value.type != "tensor(float)" for value in session.get_inputs()):
        parser.error("shape verification currently requires float32 inputs")
    rng = np.random.default_rng(42)
    observed = []
    for random in (False, True):
        feed = {name: rng.random(shape, dtype=np.float32) if random else np.zeros(shape, np.float32)
                for name, shape in shapes.items()}
        outputs = session.run(None, feed)
        if not all(np.isfinite(value).all() for value in outputs):
            parser.error("nonfinite model output during verification")
        observed.append([list(value.shape) for value in outputs])
    if observed[0] != observed[1] or any(size <= 0 for shape in observed[0] for size in shape):
        parser.error("output sizes are not fixed and positive")
    for value, shape in zip(model.graph.output, observed[0]):
        for dimension, size in zip(value.type.tensor_type.shape.dim, shape):
            dimension.dim_value = size
    onnx.checker.check_model(model)
    args.target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.target)


if __name__ == "__main__":
    main()
