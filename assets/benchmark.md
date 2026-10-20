# PreProcess and PostProcess benchmarks

All times are **milliseconds per call**. `W x H` denotes width x height.
Each benchmark has one caller; CPU preprocessing and postprocessing use one OpenCV
thread, without concurrent pipeline stages. Metal still executes its kernels on the GPU.
Inference is used only to prepare real output tensors and is excluded from postprocess timing.

## Measurement protocol

- Current measurement: 2026-10-20, Mac mini, Apple M4, macOS arm64, Release build (`-O3`),
  OpenCV 5.0.0, ONNX Runtime 1.31.0 CPU. No CPU affinity; other system activity is uncontrolled.
- Google Benchmark wall-clock median of 3 repetitions, minimum 0.5 seconds per repetition,
  and minimum 0.2 seconds warm-up. Preprocess cases also perform 20 explicit warm-up calls.
  Allocations/initialization before the timed loop and image loading are excluded;
  allocations performed by a postprocessor during a call remain included.
- Preprocess: deterministic synthetic uint8 interleaved BGR frames, 1920 x 1080,
  2560 x 1440 (2K in this record) and 3840 x 2160, letterbox with linear resize,
  padding 114, planar RGB float32 output,
  mean 0, std 255. Output sizes are 640 x 640 and 1280 x 1280. Host source and target;
  Metal wall time includes synchronous submission, completion and shared-memory access.
- Postprocess: every algo in [hrtc_template.json](template/hrtc_template.json), with its
  actual model input size and thresholds. Model SHA256 values and output layouts are in
  [the template reference](template/README.md#test-models). FP32 host outputs are reused
  between calls, including the source model with internal INT8 Q/DQ (`YoloV5Rotated`).
  The measured work includes output decoding, filtering, applicable NMS, masks and
  mapping to source coordinates. End-to-end models already do selection inside the graph;
  that graph work is inference and is excluded here.
- Images: `airport.jpg` for rotated types, `ocr_document.png` for OCR detection,
  its `(419, 16, 241, 50)` crop for recognition, and `bus.jpg` for other types.
  `GeneralKeypoint` uses the whole bus image as a repeatable top-down pose input.
  Timings depend on candidates, image content and thresholds; unlike cases are not
  direct algorithm comparisons. `GeneralClassify` uses `none`; `Embedding` copies
  1000 classification scores. Very small timings are shown with extra decimal places.

## PreProcess

### Apple M4: current measurements

| Source | Target | CPU | Metal |
| --- | --- | ---: | ---: |
| 1920 x 1080 | 640 x 640 | 0.684 | 0.480 |
| 1920 x 1080 | 1280 x 1280 | 2.093 | 0.821 |
| 2560 x 1440 | 640 x 640 | 0.616 | 0.513 |
| 2560 x 1440 | 1280 x 1280 | 1.787 | 1.036 |
| 3840 x 2160 | 640 x 640 | 0.680 | 0.779 |
| 3840 x 2160 | 1280 x 1280 | 2.519 | 1.257 |

### Apple M5 Pro: current measurements

2026-10-20, MacBook Pro, Apple M5 Pro (15 CPU cores, 24 GB RAM), macOS 27.0.1,
Apple Clang 21.0.0, OpenCV 4.10.0, ONNX Runtime 1.31.0, Release (`-O3`),
source `9fece83`. Reconfigured and rebuilt before measurement. The same one-caller,
one-OpenCV-thread protocol above applies: wall-clock median of 3 repetitions,
minimum 0.5 seconds each, minimum 0.2 seconds warm-up and 20 explicit warm-up calls.
Metal ran with GPU access; both source and result use host/shared memory.
No CPU affinity or control of other system activity. The OpenCV version differs
from the M4 run, so differences are not attributable to the chip alone.

| Source | Target | CPU | Metal |
| --- | --- | ---: | ---: |
| 1920 x 1080 | 640 x 640 | 0.547 | 0.223 |
| 1920 x 1080 | 1280 x 1280 | 1.871 | 0.455 |
| 2560 x 1440 | 640 x 640 | 0.554 | 0.248 |
| 2560 x 1440 | 1280 x 1280 | 1.424 | 0.480 |
| 3840 x 2160 | 640 x 640 | 0.560 | 0.464 |
| 3840 x 2160 | 1280 x 1280 | 2.253 | 0.722 |

### RTX 4090 server: current measurements

2026-10-20, Linux x86_64, Intel Xeon Gold 6462C, physical GPU 1 / RTX 4090,
NVIDIA driver 550.54.14, CUDA 12.4.131, OpenCV 4.5.4, Release build.
Current source was synchronized into the fixed validation directory and reconfigured
and rebuilt before measurement; benchmark and CPU/shared preprocessing source hashes
matched the local worktree. The GPU was idle before the run; this is a shared machine.
The same one-caller, 3-repetition wall-clock median protocol and preprocessing
parameters apply. CPU uses one OpenCV thread, without CPU affinity.

CUDA uses multi-stage NPP for fill, resize, conversion and normalization. The
shared dispatch verifies that all required stages are supported before executing
them on the device. An unsupported stage retains the previous whole-call CPU
fallback for device-only memory. No custom fused CUDA kernel is used here.
Regression tests confirm that all four CUDA operators execute for these supported
formats, and that disabling any one preserves the CPU fallback.

`CUDA host source` starts with a host BGR frame; `CUDA device source` starts with a
BGR frame already resident in CUDA memory. Host-source timings include the upload;
both include device processing and stream synchronization. The FP32 output stays
in device memory, without a timed output download. M4 CPU/Metal outputs above stay
in host/shared memory, so placement differs.

| Source | Target | Xeon CPU | CUDA host source | CUDA device source |
| --- | --- | ---: | ---: | ---: |
| 1920 x 1080 | 640 x 640 | 0.771 | 0.382 | 0.045 |
| 1920 x 1080 | 1280 x 1280 | 2.803 | 0.445 | 0.106 |
| 2560 x 1440 | 640 x 640 | 0.780 | 0.613 | 0.045 |
| 2560 x 1440 | 1280 x 1280 | 2.163 | 0.674 | 0.106 |
| 3840 x 2160 | 640 x 640 | 0.794 | 1.278 | 0.046 |
| 3840 x 2160 | 1280 x 1280 | 3.357 | 1.334 | 0.106 |

### Other platforms: historical measurements

These are transcribed from [perf.md](perf.md#2026-10-13-fuse-cpu-tensor-conversion-and-normalization-26e8d66),
not remeasured during this run. The archived record specifies one OpenCV thread but
does not preserve repetition counts or a statistic for these two numbers.

| Platform / path | Source | Target | Time | Recorded date |
| --- | --- | --- | ---: | --- |
| Apple M5 Pro / CPU | 1920 x 1080 | 640 x 640 | 0.566 | 2026-10-13 |
| Apple M5 Pro / CPU | 3840 x 2160 | 1280 x 1280 | 2.280 | 2026-10-13 |

RK3588 CPU/RGA does not yet have matching 640 x 640 / 1280 x 1280
measurements in this record. Existing 640 x 480 or 224 x 224 measurements are not
substituted for the requested square sizes.

## PostProcess

### Apple M4: current measurements of all 25 types

Measured with the model collection recorded before 2026-10-21, whose files and hashes
differ from the current one.

There are 27 cases: one per type, with `Ocr` additionally covering tiny detection,
tiny recognition and small recognition. Input size describes the model, not the
source image. All cases measure CPU postprocessing.

| Type | Template algo | Model input (W x H) | Time |
| --- | --- | --- | ---: |
| `YoloV8Detect` | `yolov8_detect` | 640 x 480 | 0.045 |
| `Yolo26Detect` | `yolo26n_detect_raw` | 640 x 640 | 0.050 |
| `YoloV5Detect` | `yolov5_detect` | 640 x 640 | 0.017 |
| `DetrDetect` | `rtdetr_l_detect` | 640 x 640 | 0.022 |
| `E2EDetect` | `yolo26n_e2e_detect` | 640 x 640 | 0.000267 |
| `GeneralDetect` | `yolov8n_detect` | 640 x 480 | 0.125 |
| `E2ERotated` | `yolo26n_e2e_obb` | 640 x 640 | 0.000323 |
| `YoloV8Rotated` | `yolov8n_obb_raw` | 640 x 480 | 0.090 |
| `Yolo26Rotated` | `yolo26n_obb_raw` | 640 x 640 | 0.076 |
| `YoloV5Rotated` | `yolov5_rotated` | 1280 x 1280 | 0.034 |
| `GeneralRotated` | `yolov8n_obb` | 640 x 480 | 0.092 |
| `GeneralClassify` | `yolov8n_classify` | 640 x 480 | 0.000062 |
| `Embedding` | `yolov8n_classify_embedding` | 640 x 480 | 0.000045 |
| `GeneralYoloPose` | `yolov8n_pose` | 640 x 480 | 0.002 |
| `YoloV8Keypoint` | `yolov8n_face` | 640 x 640 | 0.005 |
| `Yolo26Keypoint` | `yolo26n_pose_raw` | 640 x 640 | 0.0045 |
| `E2EKeypoint` | `yolo26n_e2e_pose` | 640 x 640 | 0.000412 |
| `RtmoKeypoint` | `rtmo_s_pose` | 640 x 640 | 0.001 |
| `GeneralKeypoint` | `rtmpose_m_pose` | 192 x 256 | 0.011 |
| `GeneralYoloSegment` | `yolov8n_segment` | 640 x 480 | 0.273 |
| `YoloV8Segment` | `yolov8n_segment_raw` | 640 x 480 | 0.186 |
| `Yolo26Segment` | `yolo26n_segment_raw` | 640 x 640 | 0.241 |
| `E2ESegment` | `yolo26n_e2e_segment` | 640 x 640 | 0.169 |
| `SemanticSegment` | `lraspp_voc_segment` | 512 x 512 | 3.464 |
| `Ocr` | `ppocrv6_tiny_det` | 640 x 640 | 0.303 |
| `Ocr` | `ppocrv6_tiny_rec` | 320 x 48 | 0.134 |
| `Ocr` | `ppocrv6_small_rec` | 320 x 48 | 0.363 |

### Apple M5 Pro: current measurements of all 25 types

2026-10-21, same M5 Pro environment and measurement protocol as the preprocessing table
above, source `676052a`, with the model collection regenerated that day; its files and
SHA256 are listed under Test models in [the template README](template/README.md). The
27 cases match the M4 list, but several decoded exports have other input sizes than the
older collection behind the M4 table, so the two tables do not compare chips.

| Type | Template algo | Model input (W x H) | Time |
| --- | --- | --- | ---: |
| `YoloV8Detect` | `yolov8_detect` | 640 x 480 | 0.039 |
| `Yolo26Detect` | `yolo26n_detect_raw` | 640 x 640 | 0.046 |
| `YoloV5Detect` | `yolov5_detect` | 640 x 640 | 0.016 |
| `DetrDetect` | `rtdetr_l_detect` | 640 x 640 | 0.016 |
| `E2EDetect` | `yolo26n_e2e_detect` | 640 x 640 | 0.000301 |
| `GeneralDetect` | `yolov8n_detect` | 640 x 640 | 0.154 |
| `E2ERotated` | `yolo26n_e2e_obb` | 640 x 640 | 0.000327 |
| `YoloV8Rotated` | `yolov8n_obb_raw` | 640 x 480 | 0.080 |
| `Yolo26Rotated` | `yolo26n_obb_raw` | 640 x 640 | 0.066 |
| `YoloV5Rotated` | `yolov5_rotated` | 1280 x 1280 | 0.029 |
| `GeneralRotated` | `yolov8n_obb` | 1024 x 1024 | 0.137 |
| `GeneralClassify` | `yolov8n_classify` | 224 x 224 | 0.000056 |
| `Embedding` | `yolov8n_classify_embedding` | 224 x 224 | 0.000041 |
| `GeneralYoloPose` | `yolov8n_pose` | 640 x 640 | 0.0024 |
| `YoloV8Keypoint` | `yolov8n_face` | 640 x 640 | 0.0047 |
| `Yolo26Keypoint` | `yolo26n_pose_raw` | 640 x 640 | 0.0042 |
| `E2EKeypoint` | `yolo26n_e2e_pose` | 640 x 640 | 0.000436 |
| `RtmoKeypoint` | `rtmo_s_pose` | 640 x 640 | 0.000962 |
| `GeneralKeypoint` | `rtmpose_m_pose` | 192 x 256 | 0.0066 |
| `GeneralYoloSegment` | `yolov8n_segment` | 640 x 480 | 0.215 |
| `YoloV8Segment` | `yolov8n_segment_raw` | 640 x 480 | 0.139 |
| `Yolo26Segment` | `yolo26n_segment_raw` | 640 x 640 | 0.180 |
| `E2ESegment` | `yolo26n_e2e_segment` | 640 x 640 | 0.124 |
| `SemanticSegment` | `lraspp_voc_segment` | 512 x 512 | 2.827 |
| `Ocr` | `ppocrv6_tiny_det` | 640 x 640 | 0.345 |
| `Ocr` | `ppocrv6_tiny_rec` | 320 x 48 | 0.123 |
| `Ocr` | `ppocrv6_small_rec` | 320 x 48 | 0.333 |

Raw output: `assets/test/benchmark_results/record_m5pro_postprocess_676052a.json`
(untracked). Postprocessing only; preprocessing and model inference are excluded.
The OCR recognition cases depend on code layout: their scalar CTC argmax loop took about
twice as long in builds where it crossed a 64-byte fetch block.

**Correctness validation.** With every template model present, Debug, Release and
AddressSanitizer CTest passed (189, 194 and 185 entries; only DMA-buf, CUDA, RGA, DVPP and
Ascend tests skipped), and macOS `leaks` reported no leaks on the test binaries (death tests
excluded), the 24 image algos and both OCR cascades.

### RK3588: historical measurements

Source: [perf.md, 2026-10-17](perf.md#2026-10-17-candidate-scans-of-detection-heads).
Recorded platform: Radxa Rock 5B+ / RK3588, librknnrt 2.3.2. CPU postprocessing on
RKNN outputs, one caller pinned to A76 cpu4, 50 warm-up calls, best of 3 rounds of
300 calls. The historical platform attribution and results have not been independently
reverified in this run. Precision below is the original RKNN model/output precision.
These older models and statistics differ from the M4 run above.

| Historical case | Precision | Time |
| --- | --- | ---: |
| `YoloV8Detect`, YOLOv8n raw, 80 classes | INT8 | 0.145 |
| `YoloV8Detect`, YOLO26n raw, 80 classes (original decoder label) | INT8 | 0.112 |
| `YoloV8Detect`, 2 classes, 1280 input | INT8 | 0.091 |
| `YoloV8Rotated`, YOLOv8n-OBB raw, 15 classes | INT8 | 0.490 |
| `YoloV8Keypoint`, YOLOv8n-Pose raw | INT8 | 0.044 |
| `GeneralDetect`, YOLOv8n | FP16 | 0.456 |
| `GeneralRotated`, YOLOv8n-OBB | FP16 | 0.413 |
| `GeneralYoloPose`, YOLOv8n-Pose | FP16 | 0.008 |
| `GeneralYoloSegment`, YOLOv8n-Seg | FP16 | 1.257 |

The archive does not specify all input sizes in this table; they are not inferred
from the current model collection. The remaining postprocessor types have not been
measured on RK3588 in this record.

### x86 and Apple M5 Pro: historical measurements

The x86 entries come from the same [candidate-scan record](perf.md#2026-10-17-candidate-scans-of-detection-heads):
Ryzen 5 5600X, ONNX Runtime 1.31.0 CPU, pinned benchmark, median of 3 repetitions.
Their exact CPU number, input shapes and warm-up settings were not preserved there.
The M5 Pro entry comes from [the OCR CTC record](perf.md#2026-10-18-ocr-ctc-decoding):
macOS, ONNX Runtime 1.31.0, one OpenCV thread, median of 3 repetitions,
one 241 x 50 text line into the 320 x 48 tiny recognizer. None was rerun here.

| Platform | Type / case | Time |
| --- | --- | ---: |
| Ryzen 5 5600X | `GeneralDetect`, YOLOv8n | 0.225 |
| Ryzen 5 5600X | `GeneralYoloSegment`, YOLOv8n-Seg | 0.551 |
| Ryzen 5 5600X | `GeneralYoloPose`, YOLOv8n-Pose | 0.004 |
| Ryzen 5 5600X | `GeneralRotated`, YOLOv8n-OBB | 0.207 |
| Apple M5 Pro | `Ocr`, tiny CTC recognition | 0.121 |

## Reproduce

Run from the repository root after placing the local models, images and OCR
dictionaries at the paths in the template. Configure a Release build with the
platform's SDK paths, then rebuild the current source:

```sh
cmake --build build --target hybridrt_bench_preprocess hybridrt_bench_postprocess

build/benchmarks/hybridrt_bench_preprocess \
  --benchmark_filter='BM_Resolution/(1920/1080|2560/1440|3840/2160)/(640/640|1280/1280)/' \
  --benchmark_min_time=0.5s --benchmark_min_warmup_time=0.2 \
  --benchmark_repetitions=3 --benchmark_report_aggregates_only=true \
  --benchmark_out=assets/test/benchmark_results/record_preprocess.json \
  --benchmark_out_format=json

build/benchmarks/hybridrt_bench_postprocess \
  --benchmark_filter='^BM_Type/' \
  --benchmark_min_time=0.5s --benchmark_min_warmup_time=0.2 \
  --benchmark_repetitions=3 --benchmark_report_aggregates_only=true \
  --benchmark_out=assets/test/benchmark_results/record_postprocess.json \
  --benchmark_out_format=json
```

Create `assets/test/benchmark_results/` before running. On macOS, preprocessing
device argument `0` is CPU and `1` is Metal; Metal needs access to the GPU.
`BM_Type/<index>` uses the current template algo order and includes the type and
algo name in its JSON label, so use that label when comparing records.

The local raw files for this run are `record_m4_six_preprocess.json`
and `record_m4_postprocess.json` under
`assets/test/benchmark_results/`. They are untracked. Other platforms should use
the same sizes and protocol when adding fresh rows, and identify device paths,
SDK versions, model hashes and any different timing statistic explicitly.

The RTX 4090 raw file is `assets/test/benchmark_results/record_gpu4090_restored_preprocess.json`.
The earlier CPU-fallback baseline is kept locally as `record_gpu4090_preprocess.json`;
before/after comparisons are in [perf.md](perf.md).
For the CUDA paths, select the physical GPU before starting the process and provide
the configured SDK library paths. Reproduce with the same timing flags as above and
`--benchmark_filter='BM_Resolution/(1920/1080|2560/1440|3840/2160)/(640/640|1280/1280)/0/|BM_CudaResolution/(1920/1080|2560/1440|3840/2160)/(640/640|1280/1280)/(0|1)/'`.
For `BM_CudaResolution`, device argument `0` means host source and `1` device source.
Use `CUDA_VISIBLE_DEVICES=1` to reproduce this run's physical GPU selection.

The M5 Pro raw preprocessing file is
`assets/test/benchmark_results/record_m5pro_six_preprocess.json` (untracked).
