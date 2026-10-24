# Benchmarks

[Preprocessing](#preprocessing) · [Postprocessing](#postprocessing) · [Pipeline](#pipeline)

Common method for every chapter:

- Release build of the current source on the platform below, reconfigured and rebuilt before measuring.
- Google Benchmark wall-clock time, `--benchmark_min_time=0.5s --benchmark_min_warmup_time=0.2
  --benchmark_repetitions=3`; tables give the median of the three repetitions.
- Image loading, model creation and buffer allocation before the timed loop are untimed; work and
  synchronization inside a call are timed.
- Models and images are the template fixtures in [the template reference](template/README.md#test-models);
  raw JSON results stay local under `assets/test/benchmark_results/`. Optimization before/after
  records belong in [perf.md](perf.md), not here.

| Item | M4 | M5 Pro | RTX 4090 | RTX 3060 | RK3588 |
| --- | --- | --- | --- | --- | --- |
| Device | Mac mini | MacBook Pro | Shared Linux server | Desktop PC | Radxa Rock 5B+ |
| CPU | Apple M4 | Apple M5 Pro, 15 cores | Intel Xeon Gold 6462C | AMD Ryzen 5 5600X, 6 cores / 12 threads | RK3588, 4x Cortex-A76 + 4x Cortex-A55 |
| Memory | not recorded | 24 GB | not recorded | 32 GB | not recorded |
| Accelerator | M4 GPU / ANE (Metal, CoreML) | M5 Pro GPU / ANE (Metal, CoreML) | RTX 4090 (physical GPU 1) | RTX 3060 12 GB | RK3588 NPU, RGA |
| OS | macOS arm64 | macOS 27.0.1 | Linux x86_64 | Windows host, WSL2 Ubuntu 24.04.3, kernel 6.18.40.1 | Debian 12 aarch64 |
| Build host | native | native | native, fixed validation directory | native | cross-build on the RTX 3060 WSL2 host |
| Compiler | Apple Clang | Apple Clang 21.0.0 | not recorded | GCC 13.3.0, nvcc 13.0.88 | Arm GNU Toolchain 12.2.rel1 `aarch64-none-linux-gnu-g++` 12.2.1 |
| CMake | not recorded | not recorded | not recorded | CMake 3.28.3, Ninja 1.11.1 | CMake 3.28.3, Ninja 1.11.1 |
| Build type | Release `-O3` | Release `-O3` | Release | Release `-O3 -DNDEBUG` | Release `-O3 -DNDEBUG` |
| OpenCV | 5.0.0 | 4.10.0 | 4.5.4 | 4.6.0 (Ubuntu package) | 4.6.0 (cross-built) |
| Inference SDK | ONNX Runtime 1.31.0 | ONNX Runtime 1.31.0 | TensorRT 10.16.1.11 | ONNX Runtime 1.31.0, TensorRT 10.14.1.48 | rknpu2 / librknnrt 2.3.2 |
| Device SDK | Metal | Metal | driver 550.54.14, CUDA 12.4.131 | driver 591.74, CUDA 13.0.96 | librga 1.10.1 |
| Affinity | none | none | none | none | A76 cores 4-7 |
| Pipeline | CPU → ORT CoreML | Metal → ORT CoreML | CUDA → TensorRT | CUDA → TensorRT | RGA → RKNN |

DVPP → CANN (Ascend) is planned and has no platform yet.

## Preprocessing

- Content: letterbox of one synthetic interleaved BGR uint8 frame (1920 x 1080, 2560 x 1440,
  3840 x 2160) to 640 x 640 and 1280 x 1280: linear resize, pad 114, planar RGB float32, mean 0, std 255.
- Devices: CPU and Metal on Apple (host/shared memory in and out), CPU and CUDA on NVIDIA. CUDA host
  starts from a host frame and includes its upload; CUDA device starts from a resident frame; CUDA
  output stays on the device.
- Method: one caller, one OpenCV thread, 20 extra warm-up calls; ms per call.
  `hybridrt_bench_preprocess --benchmark_filter='BM_(Cuda)?Resolution/[0-9]+/[0-9]+/(640/640|1280/1280)/'`.

| Platform | Source → target | CPU | Metal | CUDA host | CUDA device |
| --- | --- | ---: | ---: | ---: | ---: |
| M4 | 1920 x 1080 → 640 x 640 | 0.684 | 0.480 | | |
| M4 | 1920 x 1080 → 1280 x 1280 | 2.093 | 0.821 | | |
| M4 | 2560 x 1440 → 640 x 640 | 0.616 | 0.513 | | |
| M4 | 2560 x 1440 → 1280 x 1280 | 1.787 | 1.036 | | |
| M4 | 3840 x 2160 → 640 x 640 | 0.680 | 0.779 | | |
| M4 | 3840 x 2160 → 1280 x 1280 | 2.519 | 1.257 | | |
| M5 Pro | 1920 x 1080 → 640 x 640 | 0.547 | 0.223 | | |
| M5 Pro | 1920 x 1080 → 1280 x 1280 | 1.871 | 0.455 | | |
| M5 Pro | 2560 x 1440 → 640 x 640 | 0.554 | 0.248 | | |
| M5 Pro | 2560 x 1440 → 1280 x 1280 | 1.424 | 0.480 | | |
| M5 Pro | 3840 x 2160 → 640 x 640 | 0.560 | 0.464 | | |
| M5 Pro | 3840 x 2160 → 1280 x 1280 | 2.253 | 0.722 | | |
| RTX 3060 | 1920 x 1080 → 640 x 640 | 0.635 | | 0.601 | 0.086 |
| RTX 3060 | 1920 x 1080 → 1280 x 1280 | 2.395 | | 0.690 | 0.149 |
| RTX 3060 | 2560 x 1440 → 640 x 640 | 0.665 | | 0.967 | 0.086 |
| RTX 3060 | 2560 x 1440 → 1280 x 1280 | 1.760 | | 1.029 | 0.153 |
| RTX 3060 | 3840 x 2160 → 640 x 640 | 0.618 | | 2.122 | 0.095 |
| RTX 3060 | 3840 x 2160 → 1280 x 1280 | 3.038 | | 2.163 | 0.168 |

## Postprocessing

- Content: every template algo, one per postprocess type plus the OCR variants, decoding real FP32
  ONNX Runtime CPU outputs on the host: filtering, NMS, masks and mapping to source coordinates.
  Inference is excluded; outputs are computed once and reused.
- Inputs: bus.jpg, airport.jpg for rotated boxes, ocr_document.png for OCR detection and its
  (419, 16, 241, 50) crop for recognition, with the template thresholds.
- RK3588 INT8: the four basic 640 x 640 models as int8 RKNN (`<model>Int8.rknn`, calibrated on
  `assets/test/calib/coco64.txt`); their quantized outputs are decoded directly. `—` means not measured.
- Method: one caller, one OpenCV thread; ms per call.
  `hybridrt_bench_postprocess --benchmark_filter='^BM_Type/'` (ORT), `'^BM_Rknn/'` (RK3588).
- Model outputs are those of the template models; `{a, b, c}` lists the stride-8/16/32 heads.
  † M4 used an older export with a 640 x 480 input; ‡ M5 Pro used an earlier export (segment
  640 x 480, classify 224 x 224). The four basic models are now 640 x 640; marked times await a rerun.

| Type | Template algo | Model outputs | M5 Pro | M4 | RK3588 INT8 |
| --- | --- | --- | ---: | ---: | ---: |
| `YoloV8Detect` | `yolov8_detect` | `1x144x{60x80, 30x40, 15x20}` | 0.039 | 0.045 | — |
| `Yolo26Detect` | `yolo26n_detect_raw` | `1x84x{80x80, 40x40, 20x20}` | 0.046 | 0.050 | — |
| `YoloV5Detect` | `yolov5_detect` | `1x255x{80x80, 40x40, 20x20}` | 0.016 | 0.017 | — |
| `DetrDetect` | `rtdetr_l_detect` | `1x300x84` | 0.016 | 0.022 | — |
| `E2EDetect` | `yolo26n_e2e_detect` | `1x300x6` | 0.000301 | 0.000267 | — |
| `GeneralDetect` | `yolov8n_detect` | `1x84x8400` | 0.154 | 0.125 † | |
| `E2ERotated` | `yolo26n_e2e_obb` | `1x300x7` | 0.000327 | 0.000323 | — |
| `YoloV8Rotated` | `yolov8n_obb_raw` | `1x80x{60x80, 30x40, 15x20}` | 0.080 | 0.090 | — |
| `Yolo26Rotated` | `yolo26n_obb_raw` | `1x20x{80x80, 40x40, 20x20}` | 0.066 | 0.076 | — |
| `YoloV5Rotated` | `yolov5_rotated` | `1x24x{160x160, 80x80, 40x40}` | 0.029 | 0.034 | — |
| `GeneralRotated` | `yolov8n_obb` | `1x20x21504` | 0.137 | 0.092 † | — |
| `GeneralClassify` | `yolov8n_classify` | `1x1000` | 0.000056 ‡ | 0.000062 † | |
| `Embedding` | `yolov8n_classify_embedding` | `1x1000` | 0.000041 ‡ | 0.000045 † | — |
| `GeneralYoloPose` | `yolov8n_pose` | `1x56x8400` | 0.0024 | 0.002 † | |
| `YoloV8Keypoint` | `yolov8n_face` | `1x80x{80x80, 40x40, 20x20}` | 0.0047 | 0.005 | — |
| `Yolo26Keypoint` | `yolo26n_pose_raw` | `1x56x{80x80, 40x40, 20x20}` | 0.0042 | 0.0045 | — |
| `E2EKeypoint` | `yolo26n_e2e_pose` | `1x300x57` | 0.000436 | 0.000412 | — |
| `RtmoKeypoint` | `rtmo_s_pose` | `1x300x5`, `1x300x17x3` | 0.000962 | 0.001 | — |
| `GeneralKeypoint` | `rtmpose_m_pose` | `1x17x384`, `1x17x512` | 0.0066 | 0.011 | — |
| `GeneralYoloSegment` | `yolov8n_segment` | `1x116x8400`, `1x32x160x160` | 0.215 ‡ | 0.273 † | |
| `YoloV8Segment` | `yolov8n_segment_raw` | `1x176x{60x80, 30x40, 15x20}`, `1x32x120x160` | 0.139 | 0.186 | — |
| `Yolo26Segment` | `yolo26n_segment_raw` | `1x116x{80x80, 40x40, 20x20}`, `1x32x160x160` | 0.180 | 0.241 | — |
| `E2ESegment` | `yolo26n_e2e_segment` | `1x300x38`, `1x32x160x160` | 0.124 | 0.169 | — |
| `SemanticSegment` | `lraspp_voc_segment` | `1x21x512x512` | 2.827 | 3.464 | — |
| `Ocr` | `ppocrv6_tiny_det` | `1x1x640x640` | 0.345 | 0.303 | — |
| `Ocr` | `ppocrv6_tiny_rec` | `1x40x6906` | 0.123 | 0.134 | — |
| `Ocr` | `ppocrv6_small_rec` | `1x40x18710` | 0.333 | 0.363 | — |

## Pipeline

- Content: whole `ImageModel::process_sync` (preprocess, inference, postprocess) for the four basic
  models `yolov8n_detect`, `yolov8n_segment`, `yolov8n_pose` and `yolov8n_classify`, each as native
  640 x 640 static batch 1, 4 and 8 exports (`_batch4`, `_batch8` template algos).
- Paths: CPU → ORT CoreML (M4), Metal → ORT CoreML (M5 Pro), CUDA → TensorRT (RTX 4090, RTX 3060),
  RGA → RKNN (RK3588); DVPP → CANN later. TensorRT plans are FP32 `--noTF32`, built on the platform.
- Method: one model with preprocess/engine/postprocess pools 3/2/3, shared by 8 caller threads.
  Each caller cycles through the same 32 sorted COCO128 images (host BGR) from its own offset and
  passes as many images per call as the model batch: B1 one, B4 four, B8 eight. 5 untimed calls
  per caller, one OpenCV thread; total images per second over all callers.
  `hybridrt_bench_pipeline --benchmark_filter='BM_Batch/'`.

| Platform | Path | Model | B1 | B4 | B8 |
| --- | --- | --- | ---: | ---: | ---: |
| M4 | CPU → ORT CoreML | `yolov8n_detect` | | | |
| M4 | CPU → ORT CoreML | `yolov8n_segment` | | | |
| M4 | CPU → ORT CoreML | `yolov8n_pose` | | | |
| M4 | CPU → ORT CoreML | `yolov8n_classify` | | | |
| M5 Pro | Metal → ORT CoreML | `yolov8n_detect` | | | |
| M5 Pro | Metal → ORT CoreML | `yolov8n_segment` | | | |
| M5 Pro | Metal → ORT CoreML | `yolov8n_pose` | | | |
| M5 Pro | Metal → ORT CoreML | `yolov8n_classify` | | | |
| RTX 4090 | CUDA → TensorRT | `yolov8n_detect` | | | |
| RTX 4090 | CUDA → TensorRT | `yolov8n_segment` | | | |
| RTX 4090 | CUDA → TensorRT | `yolov8n_pose` | | | |
| RTX 4090 | CUDA → TensorRT | `yolov8n_classify` | | | |
| RTX 3060 | CUDA → TensorRT | `yolov8n_detect` | | | |
| RTX 3060 | CUDA → TensorRT | `yolov8n_segment` | | | |
| RTX 3060 | CUDA → TensorRT | `yolov8n_pose` | | | |
| RTX 3060 | CUDA → TensorRT | `yolov8n_classify` | | | |
| RK3588 | RGA → RKNN | `yolov8n_detect` | | | |
| RK3588 | RGA → RKNN | `yolov8n_segment` | | | |
| RK3588 | RGA → RKNN | `yolov8n_pose` | | | |
| RK3588 | RGA → RKNN | `yolov8n_classify` | | | |
