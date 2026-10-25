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
| Memory | 24 GB | 24 GB | 48 GB | 32 GB | 8 GB |
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

- Content: all template algos, including static batch and OCR variants, on the Ryzen 5 5600X CPU.
  Decode real FP32 ONNX Runtime CPU outputs: filtering, NMS, masks and source coordinate mapping.
  Inference, FP16/INT8 conversion and host copies are excluded; outputs are computed once and reused.
- Inputs: bus.jpg, airport.jpg for rotated boxes, ocr_document.png for OCR detection and its
  (419, 16, 241, 50) crop for recognition, with template thresholds.
- Method: one caller, one OpenCV thread, no affinity; whole-batch milliseconds. Divide by B for
  mean milliseconds per image. `B` is the batch dimension; `—` means no template model for that batch.
  `hybridrt_bench_postprocess --benchmark_filter='^BM_Type/'`.

5600X CPU, FP32 ONNX outputs:

| Type | Template algo | Model outputs | CPU B1 | CPU B4 | CPU B8 |
| --- | --- | --- | ---: | ---: | ---: |
| `YoloV8Detect` | `yolov8_detect` | `Bx144x{80x80, 40x40, 20x20}` | 0.08788 | — | — |
| `Yolo26Detect` | `yolo26n_detect_raw` | `Bx84x{80x80, 40x40, 20x20}` | 0.07557 | — | — |
| `YoloV5Detect` | `yolov5_detect` | `Bx255x{80x80, 40x40, 20x20}` | 0.03462 | — | — |
| `DetrDetect` | `rtdetr_l_detect` | `Bx300x84` | 0.02263 | — | — |
| `E2EDetect` | `yolo26n_e2e_detect` | `Bx300x6` | 0.0006817 | — | — |
| `GeneralDetect` | `yolov8n_detect` | `Bx84x8400` | 0.04063 | 0.1518 | 0.3261 |
| `E2ERotated` | `yolo26n_e2e_obb` | `Bx300x7` | 0.0006963 | — | — |
| `YoloV8Rotated` | `yolov8n_obb_raw` | `Bx80x{128x128, 64x64, 32x32}` | 0.2583 | — | — |
| `Yolo26Rotated` | `yolo26n_obb_raw` | `Bx20x{128x128, 64x64, 32x32}` | 0.2009 | — | — |
| `YoloV5Rotated` | `yolov5_rotated` | `Bx24x{160x160, 80x80, 40x40}` | 0.0502 | — | — |
| `GeneralRotated` | `yolov8n_obb` | `Bx20x6300` | 0.1782 | — | — |
| `GeneralClassify` | `yolov8n_classify` | `Bx1000` | 5.933e-05 | 0.0009275 | 0.001775 |
| `Embedding` | `yolov8n_classify_embedding` | `Bx1000` | 4.468e-05 | — | — |
| `GeneralYoloPose` | `yolov8n_pose` | `Bx56x8400` | 0.001496 | 0.007433 | 0.0139 |
| `YoloV8Keypoint` | `yolov8n_face` | `Bx80x{80x80, 40x40, 20x20}` | 0.00967 | — | — |
| `Yolo26Keypoint` | `yolo26n_pose_raw` | `Bx56x{80x80, 40x40, 20x20}` | 0.00793 | — | — |
| `E2EKeypoint` | `yolo26n_e2e_pose` | `Bx300x57` | 0.001152 | — | — |
| `RtmoKeypoint` | `rtmo_s_pose` | `Bx300x5`, `Bx300x17x3` | 0.001724 | — | — |
| `GeneralKeypoint` | `rtmpose_m_pose` | `Bx17x384`, `Bx17x512` | 0.006013 | — | — |
| `GeneralYoloSegment` | `yolov8n_segment` | `Bx116x8400`, `Bx32x160x160` | 0.3406 | 1.733 | 4.252 |
| `YoloV8Segment` | `yolov8n_segment_raw` | `Bx176x{60x80, 30x40, 15x20}`, `Bx32x120x160` | 0.3087 | — | — |
| `Yolo26Segment` | `yolo26n_segment_raw` | `Bx116x{80x80, 40x40, 20x20}`, `Bx32x160x160` | 0.3763 | — | — |
| `E2ESegment` | `yolo26n_e2e_segment` | `Bx300x38`, `Bx32x160x160` | 0.2761 | — | — |
| `SemanticSegment` | `yolo26n_semantic` | `Bx19x80x80` | 0.389 | — | — |
| `Ocr` | `ppocrv6_tiny_det` | `Bx1x640x640` | 0.2974 | — | — |
| `Ocr` | `ppocrv6_tiny_rec` | `Bx40x6906` | 0.1255 | — | 1.022 |

RK3588 CPU, RKNN FP16/INT8 output comparison (whole-batch ms):

- Coverage: every postprocess type and both OCR stages. Each model needs INT8 B1/B4/B8;
  keep FP16 B1 only as its comparison. Convert and validate each variant; record conversion or
  runtime failures explicitly. A filename or weight precision alone does not establish INT8 output.
- Same warm-up, three repetitions and median method as above; one caller, one OpenCV thread,
  A76 cores 4-7. Model shapes and thresholds come from the template; input images are resized on
  the build host. One RKNN inference, output conversion and host copies run before timing.
- 72 native INT8-output variants (24 model groups x B1/B4/B8) and 23 FP16-output B1 references
  passed SDK IO checks; 64 INT8 variants of 22 groups and 21 FP16 B1 models produced valid timings.
  `conversion fails` marks RT-DETR and RTMO, whose graphs the toolkit cannot convert. `—` means
  not timed: the user QAT YOLOv5 OBB model has no FP16 counterpart, and `GeneralKeypoint`
  (top-down, one keypoint set per output) rejects batch 4 and 8. `error` marks the YOLO26
  end-to-end exports, whose in-graph TopK/GatherElements index tensors go out of range on the
  runtime; no timing is accepted. Conversion/query results are detailed in
  [the model reference](template/README.md#rknn-on-rk3588).
- `‡` means zero decoded objects on the test image: only scan/filter work is measured, not NMS/masks.
  Fused coordinate/confidence outputs lose scores at coarse scales (about 2.57–3.16); these results
  cannot establish useful detection accuracy or a speedup over FP32. The user-specific YOLOv5 OBB
  model also has no objects on airport.jpg; that image is not an accuracy reference for its classes.
- `SemanticSegment` uses YOLO26n-sem with 19 x 80 x 80 logits at 640 x 640 input; its masks
  cover the 810 x 1080 bus.jpg.
- Local raw results: `rknn_all_int8_query.log`, `rknn_postprocess_matrix.json` and per-variant logs in
  `assets/test/benchmark_results/rknn_postprocess_matrix/`, from `run_rknn_postprocess.py`. SDK
  versions match the platform matrix; the board service stayed running and frequency scaling enabled.

| Type | Template algo | Native INT8 output | FP16 B1 | INT8 B1 | INT8 B4 | INT8 B8 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `YoloV8Detect` | `yolov8_detect` | INT8 | 0.3652 | 0.1239 | 0.5141 | 1.081 |
| `Yolo26Detect` | `yolo26n_detect_raw` | INT8 | 0.3152 | 0.1156 | 0.4839 | 0.995 |
| `YoloV5Detect` | `yolov5_detect` | INT8 | 0.07864 | 0.04357 | 0.1843 | 0.3765 |
| `DetrDetect` | `rtdetr_l_detect` | conversion fails | — | — | — | — |
| `E2EDetect` | `yolo26n_e2e_detect` | INT8 | error | error | error | error |
| `GeneralDetect` | `yolov8n_detect` | INT8 | 0.1495 | 0.06851 ‡ | 0.2813 ‡ | 0.5844 ‡ |
| `E2ERotated` | `yolo26n_e2e_obb` | INT8 | 0.001539 | 0.001767 ‡ | 0.008112 ‡ | 0.01628 ‡ |
| `YoloV8Rotated` | `yolov8n_obb_raw` | INT8 | 0.6044 | 0.4283 | 1.763 | 4.537 |
| `Yolo26Rotated` | `yolo26n_obb_raw` | INT8 | 0.4315 | 0.3487 | 1.416 | 2.843 |
| `YoloV5Rotated` | `yolov5_rotated` | INT8 | — | 0.09146 ‡ | 0.3705 ‡ | 0.7413 ‡ |
| `GeneralRotated` | `yolov8n_obb` | INT8 | 0.3472 | 0.01065 ‡ | 0.0441 ‡ | 0.08837 ‡ |
| `GeneralClassify` | `yolov8n_classify` | INT8 | 0.0001666 | 0.0005384 | 0.004765 | 0.0134 |
| `Embedding` | `yolov8n_classify_embedding` | INT8 | 0.0001348 | 0.0004969 | 0.004489 | 0.01292 |
| `GeneralYoloPose` | `yolov8n_pose` | INT8 | 0.003619 | 0.000868 ‡ | 0.004582 ‡ | 0.009266 ‡ |
| `YoloV8Keypoint` | `yolov8n_face` | INT8 | 0.02257 | 0.01344 | 0.05769 | 0.1178 |
| `Yolo26Keypoint` | `yolo26n_pose_raw` | INT8 | 0.01788 | 0.01396 | 0.05963 | 0.1216 |
| `E2EKeypoint` | `yolo26n_e2e_pose` | INT8 | 0.001255 ‡ | 0.001731 ‡ | 0.008213 ‡ | 0.01648 ‡ |
| `RtmoKeypoint` | `rtmo_s_pose` | conversion fails | — | — | — | — |
| `GeneralKeypoint` | `rtmpose_m_pose` | INT8 | 0.01914 | 0.005418 | — | — |
| `GeneralYoloSegment` | `yolov8n_segment` | INT8 | 1.163 | 0.0692 ‡ | 0.287 ‡ | 0.5925 ‡ |
| `YoloV8Segment` | `yolov8n_segment_raw` | INT8 | 0.9454 | 0.6709 | 3.088 | 6.639 |
| `Yolo26Segment` | `yolo26n_segment_raw` | INT8 | 1.268 | 0.8964 | 4.078 | 8.119 |
| `E2ESegment` | `yolo26n_e2e_segment` | INT8 | error | error | error | error |
| `SemanticSegment` | `yolo26n_semantic` | INT8 | 1.058 | 0.8881 | 4.552 | 9.425 |
| `Ocr` | `ppocrv6_tiny_det` | INT8 | 0.7009 | 0.7688 | 3.135 | 6.252 |
| `Ocr` | `ppocrv6_tiny_rec` | INT8 | 0.2434 | 0.01659 | 0.06792 | 0.1427 |

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
| RTX 3060 | CUDA → TensorRT | `yolov8n_detect` | 358.9 | 654.7 | 862.8 |
| RTX 3060 | CUDA → TensorRT | `yolov8n_segment` | 361.8 | 497.1 | 621.1 |
| RTX 3060 | CUDA → TensorRT | `yolov8n_pose` | 399.7 | 654.7 | 853.1 |
| RTX 3060 | CUDA → TensorRT | `yolov8n_classify` | 1109.0 | 1364.9 | 1419.8 |
| RK3588 | RGA → RKNN | `yolov8n_detect` | 65.3 | 62.6 | 61.4 |
| RK3588 | RGA → RKNN | `yolov8n_segment` | 52.9 | 51.6 | 50.2 |
| RK3588 | RGA → RKNN | `yolov8n_pose` | 55.3 | 54.7 | 57.7 |
| RK3588 | RGA → RKNN | `yolov8n_classify` | 114.3 | 115.7 | 116.0 |

RTX 3060 gains 1.23–1.82x throughput at B4 and 1.28–2.40x at B8 over B1. RK3588 stays
within −4.1% to +1.2% at B4 and −6.0% to +4.4% at B8; batching gives no consistent throughput
gain on these FP16 RKNN exports. These are aggregate throughput results; request latency grows
with batch size. The board service was stopped during measurement and restored afterward.

CPU ORT and TensorRT batch-1/4/8 correctness checks passed on the current 640 x 640 fixtures.
The earlier RKNN batch comparison failed strict result equality even with the board service
stopped; its throughput figures above do not establish equivalence with batch 1. Further performance testing has been stopped; model conversion and output-type checks are complete.
