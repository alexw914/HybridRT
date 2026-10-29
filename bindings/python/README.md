# HybridRT for Python

Python bindings of [HybridRT](../../README.md) over its C API (`include/capi/hrt.h`),
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
| Embedding | `numpy.ndarray` | float32 vector of the single engine output |

Errors raise `hrt.HybridRTError` with a `status`. Library logs go to the
console; `hrt.use_python_logging()` forwards them to the `hybridrt` logger.
An engine whose outputs do not fit the postprocess config still aborts the
process, as in C++.

`examples/run_models.py` runs one algo per task (`yolov8n_detect`, `yolov8n_obb`, `yolov8n_pose`,
`yolov8n_segment`, `yolov8n_classify`) and OCR (`ppocrv6_tiny_det` + `ppocrv6_tiny_rec`) on
`bus.jpg`, `airport.jpg` and `ocr_document.jpg`, read with OpenCV, and prints the results; algos
missing from the package are skipped. Pack those algos of the template, or pass a manifest of them:

```sh
python bindings/python/examples/run_models.py -m models.hrtc -d assets/test/images
```

With `HYBRIDRT_BUILD_TESTS=ON`, run `ctest --test-dir <build> -R hybridrt_test_python --output-on-failure`.
Real-model tests use available entries from the reference template, so missing unrelated models do not
prevent loading. Detection, segmentation, pose and classification use `assets/test/images/bus.jpg`;
rotated boxes use `airport.jpg`. See [test model sources and exports](../../assets/template/README.md#test-models).

## Batch inference

`model.batch` reports the static engine batch size. Use
`results = model.process_batch([image1, image2], rois=None)` to process any
positive number of images, including one. `rois` is omitted or contains one
`(x, y, width, height)` rectangle per image; `format` applies to every image.
Results are owned copies in input order. Images are grouped by the engine batch
size; unused samples in the last group are not decoded. `process` remains the
single-image API. OCR keeps its separate single-image API.

## OCR

`OcrModel` selects detection and recognition algos from one JSON/.hrtc package:

```python
with hrt.OcrModel("ocr.hrtc", "ppocrv6_tiny_det", "ppocrv6_tiny_rec") as model:
    for item in model(cv2.imread("document.png")):
        print(item.text, item.rect, item.detection_score, item.recognition_score)
```

Results are owned `OcrText` objects; `rect` is a float32 `(4,)` array of x, y, width, height
in source coordinates. Calls on one instance are serialized; each stage has one engine and
reuses its buffers. Dictionaries are embedded by the JSON loader when building packages.
Set the ORT engine config's `execution_provider` to `coreml` to enable CoreML on macOS.
See [OCR configuration](../../assets/template/README.md#ocr-model) for parameters and limits.
