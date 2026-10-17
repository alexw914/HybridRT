"""Converts an ONNX model to RKNN for RK3588.

    python rknn_convert.py model.onnx model.rknn [--dataset images.txt] [--mean 0 0 0] [--std 255 255 255]
                      [--input-size 1 3 256 192] [--optimization-level 0-3] [--disable-rules rule ...]

Without --dataset the model stays fp16; with it (one image path per line) it
is quantized to int8. mean/std are built into the model, so HybridRT feeds
uint8 images with "image_input": true. --input-size fixes a dynamic input.
"""

import argparse

from rknn.api import RKNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx")
    parser.add_argument("rknn")
    parser.add_argument("--dataset", help="text file of calibration image paths; enables int8")
    parser.add_argument("--mean", type=float, nargs=3, default=[0, 0, 0])
    parser.add_argument("--std", type=float, nargs=3, default=[255, 255, 255])
    parser.add_argument("--input-size", type=int, nargs="+", help="fixed NCHW shape of a dynamic input")
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--disable-rules", nargs="+", default=[], help="graph optimization rules to skip")
    parser.add_argument("--platform", default="rk3588")
    args = parser.parse_args()

    rknn = RKNN(verbose=False)
    options = {"disable_rules": args.disable_rules} if args.disable_rules else {}
    assert rknn.config(mean_values=[args.mean], std_values=[args.std], target_platform=args.platform,
                       optimization_level=args.optimization_level, **options) == 0
    sizes = {"input_size_list": [args.input_size]} if args.input_size else {}
    if args.input_size:
        import onnx
        sizes["inputs"] = [onnx.load(args.onnx).graph.input[0].name]
    assert rknn.load_onnx(model=args.onnx, **sizes) == 0
    assert rknn.build(do_quantization=args.dataset is not None, dataset=args.dataset) == 0
    assert rknn.export_rknn(args.rknn) == 0
    rknn.release()
    print("exported", args.rknn)


if __name__ == "__main__":
    main()
