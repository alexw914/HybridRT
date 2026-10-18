# HybridRT for Python

Python bindings of [HybridRT](../../README.md) over its C API (`include/hrt.h`),
loaded with ctypes. Images are uint8 numpy arrays in host memory; results are
numpy arrays in source-image coordinates. Calls release the GIL and are
thread-safe.

## Build and install

Pass the backend SDK paths as CMake definitions, as for the C++ build:

```sh
pip install ./bindings/python -Ccmake.define.ORT_ROOT=<onnxruntime-sdk>
```

The wheel carries `hybridrt_c` and the shared libraries of ONNX Runtime, MNN and
ncnn; other SDKs and system libraries such as OpenCV must be on the library path.
To use a library from a CMake build with `-DHYBRIDRT_BUILD_PYTHON=ON` instead,
set `HYBRIDRT_LIBRARY` to the built `hybridrt_c` and add `bindings/python` to
`PYTHONPATH`.

## Usage

```python
import cv2
import hybridrt as hrt

model = hrt.Model("model.hrtc", "yolov8n_detect", engine=2)
result = model(cv2.imread("bus.jpg"))  # format="inter_bgr", roi=None
for box, score, cls in zip(result.boxes, result.scores, result.class_ids):
    print(cls, score, box)  # x, y, width, height
```

`hrt.Package` lists the algos of an .hrtc package or a .json manifest and
creates several models from it; loading reads every model file it names, so
load it once rather than passing the path to each `hrt.Model`. Image formats use the names of the preprocess
config: `inter_bgr`, `inter_rgb`, `inter_rgba` as (H, W, C); `planar_bgr`,
`planar_rgb` as (3, H, W); `gray` as (H, W); `nv12`, `nv21`, `yu12`, `yv12` as
(H * 3 / 2, W).

| Task | Result | Fields |
| --- | --- | --- |
| Detect | `DetectResult` | `boxes` (N, 4), `scores`, `class_ids`, `track_ids` |
| Rotated | `RotatedResult` | `boxes` (N, 5) center x, center y, width, height, angle |
| Keypoint | `KeypointResult` | detect fields, `keypoints` (N, K, 3) x, y, score, `visible` (N, K) |
| Segment | `SegmentResult` | detect fields, `masks`: `Mask(rect, data)` per object |
| Classify | `ClassifyResult` | `scores`: one array per engine output |

Errors raise `hrt.HybridRTError` with a `status`. Library logs go to the
console; `hrt.use_python_logging()` forwards them to the `hybridrt` logger.
An engine whose outputs do not fit the postprocess config still aborts the
process, as in C++.

With `HYBRIDRT_BUILD_TESTS=ON`, run `ctest --test-dir <build> -R hybridrt_test_python --output-on-failure`.
Real-model tests use available entries from the reference template, so missing unrelated models do not
prevent loading. Detection, segmentation, pose and classification use `assets/test/images/bus.jpg`;
rotated boxes use `airport.jpg`. See [test model sources and exports](../../assets/template/README.md#test-models).
