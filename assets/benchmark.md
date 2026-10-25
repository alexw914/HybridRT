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
  3840 x 2160) to 640 x 640 and 1280 x 1280: linear resize, pad 114. CPU/Metal/CUDA output
  planar RGB float32 with mean 0 and std 255; RGA output is interleaved RGB uint8 in a dma-buffer.
- Devices: CPU and Metal on Apple (host/shared memory in and out), CPU and CUDA on NVIDIA. CUDA host
  starts from a host frame and includes its upload; CUDA device starts from a resident frame; CUDA
  output stays on the device.
- Method: one caller, one OpenCV thread, 20 extra warm-up calls; ms per call.
  `hybridrt_bench_preprocess --benchmark_filter='^BM_(Resolution|CudaResolution)/[0-9]+/[0-9]+/(640/640|1280/1280)/(0|1)/'`.
- RK3588 measures RGA only, with the same six source/target size pairs: `RGA host` includes
  staging from host BGR memory into a reusable dma-buffer; `RGA dma` starts from a resident
  BGR dma-buffer populated before timing. Both include letterbox, BGR-to-RGB conversion,
  cache-coherency work and synchronous RGA completion (`IM_SYNC`), leaving output in its
  dma-buffer. Buffer allocation, warm-up calls and the initial resident-source copy are untimed;
  per-call mapping/import work and repeated host staging are timed. Mean 0/std 1 and no
  normalization are used; normalization built into RKNN and all inference are excluded. This uint8 path is not the same output
  contract as the float32 CPU/Metal/CUDA path.
- RGA fills the padding and performs BGR-to-RGB conversion inside its resize operation for
  these interleaved uint8 buffers; no separate CPU conversion/normalization is requested.
  RGB888 dma-buffer row pitch must be a multiple of 16 pixels and
  allocations must satisfy the backend's below-4-GB requirement. Unsupported formats, scales
  or allocation failures must be reported; whole-call CPU fallbacks must not be accepted as RGA results.
  Use `hybridrt_bench_preprocess --benchmark_filter='^BM_RgaResolution/'` with the common
  three-repetition method and 20 extra warm-ups. RGA cells marked `—` have not been measured.

| Platform | Source → target | CPU | Metal | CUDA host | CUDA device | RGA host | RGA dma |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M4 | 1920 x 1080 → 640 x 640 | 0.684 | 0.480 | | | | |
| M4 | 1920 x 1080 → 1280 x 1280 | 2.093 | 0.821 | | | | |
| M4 | 2560 x 1440 → 640 x 640 | 0.616 | 0.513 | | | | |
| M4 | 2560 x 1440 → 1280 x 1280 | 1.787 | 1.036 | | | | |
| M4 | 3840 x 2160 → 640 x 640 | 0.680 | 0.779 | | | | |
| M4 | 3840 x 2160 → 1280 x 1280 | 2.519 | 1.257 | | | | |
| M5 Pro | 1920 x 1080 → 640 x 640 | 0.547 | 0.223 | | | | |
| M5 Pro | 1920 x 1080 → 1280 x 1280 | 1.871 | 0.455 | | | | |
| M5 Pro | 2560 x 1440 → 640 x 640 | 0.554 | 0.248 | | | | |
| M5 Pro | 2560 x 1440 → 1280 x 1280 | 1.424 | 0.480 | | | | |
| M5 Pro | 3840 x 2160 → 640 x 640 | 0.560 | 0.464 | | | | |
| M5 Pro | 3840 x 2160 → 1280 x 1280 | 2.253 | 0.722 | | | | |
| RTX 3060 | 1920 x 1080 → 640 x 640 | 0.635 | | 0.601 | 0.086 | | |
| RTX 3060 | 1920 x 1080 → 1280 x 1280 | 2.395 | | 0.690 | 0.149 | | |
| RTX 3060 | 2560 x 1440 → 640 x 640 | 0.665 | | 0.967 | 0.086 | | |
| RTX 3060 | 2560 x 1440 → 1280 x 1280 | 1.760 | | 1.029 | 0.153 | | |
| RTX 3060 | 3840 x 2160 → 640 x 640 | 0.618 | | 2.122 | 0.095 | | |
| RTX 3060 | 3840 x 2160 → 1280 x 1280 | 3.038 | | 2.163 | 0.168 | | |
| RK3588 | 1920 x 1080 → 640 x 640 | | | | | — | — |
| RK3588 | 1920 x 1080 → 1280 x 1280 | | | | | — | — |
| RK3588 | 2560 x 1440 → 640 x 640 | | | | | — | — |
| RK3588 | 2560 x 1440 → 1280 x 1280 | | | | | — | — |
| RK3588 | 3840 x 2160 → 640 x 640 | | | | | — | — |
| RK3588 | 3840 x 2160 → 1280 x 1280 | | | | | — | — |

## Postprocessing

- Content: all 26 base template algos, including both OCR stages, each at static B1/B4/B8
  on the Ryzen 5 5600X CPU. Decode real FP32 ONNX Runtime CPU outputs: filtering, NMS, masks
  and source coordinate mapping. Force the ORT CPU provider and intra/inter-op threads to 1;
  require every output to be FP32. Model preparation is described in
  [the ONNX fixture reference](template/README.md#fp32-output-onnx-postprocessing-fixtures).
  Inference, FP16/INT8 conversion and host copies are excluded; outputs are computed once and reused.
- Inputs: bus.jpg, airport.jpg for rotated boxes, ocr_document.png for OCR detection and its
  (419, 16, 241, 50) crop for recognition, with template thresholds.
- Method: one caller, one OpenCV thread, no affinity; whole-batch milliseconds. Divide by B for
  mean milliseconds per image. `B` is the batch dimension; each sample receives the same source
  image and transform. `error` records a failed decoder; `—` means an unmeasured variant.
  `hybridrt_bench_postprocess --benchmark_filter='^BM_Type/'`.

5600X CPU, FP32 ONNX outputs:

| Type | Template algo | Model outputs | CPU B1 | CPU B4 | CPU B8 | Status / failure reason |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `YoloV8Detect` | `yolov8_detect` | `Bx144x{80x80, 40x40, 20x20}` | 0.08627 | 0.3395 | 0.9383 | — |
| `Yolo26Detect` | `yolo26n_detect_raw` | `Bx84x{80x80, 40x40, 20x20}` | 0.07242 | 0.3007 | 0.7207 | — |
| `YoloV5Detect` | `yolov5_detect` | `Bx255x{80x80, 40x40, 20x20}` | 0.02821 | 0.1169 | 0.2409 | — |
| `DetrDetect` | `rtdetr_l_detect` | `Bx300x84` | 0.02335 | 0.09913 | 0.193 | — |
| `E2EDetect` | `yolo26n_e2e_detect` | `Bx300x6` | 0.0006546 | 0.003436 | 0.006499 | — |
| `GeneralDetect` | `yolov8n_detect` | `Bx84x8400` | 0.04245 | 0.169 | 0.6691 | — |
| `E2ERotated` | `yolo26n_e2e_obb` | `Bx300x7` | 0.0007709 | 0.0035 | 0.006715 | — |
| `YoloV8Rotated` | `yolov8n_obb_raw` | `Bx80x{128x128, 64x64, 32x32}` | 0.2601 | 1.042 | 2.279 | — |
| `Yolo26Rotated` | `yolo26n_obb_raw` | `Bx20x{128x128, 64x64, 32x32}` | 0.187 | 0.8 | 1.604 | — |
| `YoloV5Rotated` | `yolov5_rotated` | `Bx24x{160x160, 80x80, 40x40}` | 0.03412 † | 0.1312 † | 0.2655 † | FP32 outputs from a Q/DQ graph; no objects on airport.jpg (†). |
| `GeneralRotated` | `yolov8n_obb` | `Bx20x6300` | 0.1716 | 0.6869 | 1.489 | — |
| `GeneralClassify` | `yolov8n_classify` | `Bx1000` | 5.511e-05 | 0.0007944 | 0.001587 | — |
| `Embedding` | `yolov8n_classify_embedding` | `Bx1000` | 4.599e-05 | 0.0007449 | 0.0015 | — |
| `GeneralYoloPose` | `yolov8n_pose` | `Bx56x8400` | 0.001493 | 0.00676 | 0.01358 | — |
| `YoloV8Keypoint` | `yolov8n_face` | `Bx80x{80x80, 40x40, 20x20}` | 0.008318 | 0.0364 | 0.07647 | — |
| `Yolo26Keypoint` | `yolo26n_pose_raw` | `Bx56x{80x80, 40x40, 20x20}` | 0.006712 | 0.0289 | 0.05861 | — |
| `E2EKeypoint` | `yolo26n_e2e_pose` | `Bx300x57` | 0.001153 | 0.005358 | 0.01246 | — |
| `RtmoKeypoint` | `rtmo_s_pose` | `Bx300x5`, `Bx300x17x3` | 0.001657 | 0.007602 | 0.01511 | — |
| `GeneralKeypoint` | `rtmpose_m_pose` | `Bx17x384`, `Bx17x512` | 0.006109 | 0.0255 | 0.05336 | — |
| `GeneralYoloSegment` | `yolov8n_segment` | `Bx116x8400`, `Bx32x160x160` | 0.3358 | 1.645 | 4.659 | — |
| `YoloV8Segment` | `yolov8n_segment_raw` | `Bx176x{60x80, 30x40, 15x20}`, `Bx32x120x160` | 0.2965 | 1.289 | 3.697 | — |
| `Yolo26Segment` | `yolo26n_segment_raw` | `Bx116x{80x80, 40x40, 20x20}`, `Bx32x160x160` | 0.3643 | 1.9 | 5.509 | — |
| `E2ESegment` | `yolo26n_e2e_segment` | `Bx300x38`, `Bx32x160x160` | 0.3357 | 1.289 | 3.011 | — |
| `SemanticSegment` | `yolo26n_semantic` | `Bx19x80x80` | 0.3928 | 3.109 | 7.762 | — |
| `Ocr` | `ppocrv6_tiny_det` | `Bx1x640x640` | 0.2847 | 1.165 | 2.399 | — |
| `Ocr` | `ppocrv6_tiny_rec` | `Bx40x6906` | 0.09937 | 0.415 | 0.8399 | — |

All 78 cases have timings, including `GeneralKeypoint` B1/B4/B8 after enabling its batch
initialization. Decoded object counts scale with B on the checked image for every
timed detector/keypoint/segment/OCR case. `†` marks zero decoded objects: the user YOLOv5
OBB model has no matching objects on airport.jpg, so these cells measure filtering only.
Local accepted results: `assets/test/benchmark_results/postprocess_fp32_onnx_b148_complete.json`.

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
  B4/B8 have not been remeasured since batch support was enabled. `error` marks the YOLO26
  end-to-end exports, whose in-graph TopK/GatherElements index tensors go out of range on the
  runtime; no timing is accepted. Conversion/query results are detailed in
  [the model reference](template/README.md#rknn-on-rk3588).
- `yolov8n_segment` timings are withdrawn: FP16 and INT8 decoded different object counts.
  All its cells remain empty until corrected outputs are validated and the case is retested.
- `‡` means zero decoded objects on the test image: only scan/filter work is measured, not NMS/masks.
  Fused coordinate/confidence outputs lose scores at coarse scales (about 2.57–3.16); these results
  cannot establish useful detection accuracy or a speedup over FP32. The user-specific YOLOv5 OBB
  model also has no objects on airport.jpg; that image is not an accuracy reference for its classes.
- `SemanticSegment` uses YOLO26n-sem with 19 x 80 x 80 logits at 640 x 640 input; its masks
  cover the 810 x 1080 bus.jpg.
- Local raw results: `rknn_all_int8_query.log`, `rknn_postprocess_matrix.json` and per-variant logs in
  `assets/test/benchmark_results/rknn_postprocess_matrix/`, from `run_rknn_postprocess.py`. SDK
  versions match the platform matrix; the board service stayed running and frequency scaling enabled.

| Type | Template algo | Native INT8 output | FP16 B1 | INT8 B1 | INT8 B4 | INT8 B8 | Status / failure reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `YoloV8Detect` | `yolov8_detect` | INT8 | 0.3652 | 0.1239 | 0.5141 | 1.081 | — |
| `Yolo26Detect` | `yolo26n_detect_raw` | INT8 | 0.3152 | 0.1156 | 0.4839 | 0.995 | — |
| `YoloV5Detect` | `yolov5_detect` | INT8 | 0.07864 | 0.04357 | 0.1843 | 0.3765 | — |
| `DetrDetect` | `rtdetr_l_detect` | conversion fails | — | — | — | — | Conversion fails: RKNN Conv rewrite channel mismatch (4 vs 256). |
| `E2EDetect` | `yolo26n_e2e_detect` | INT8 | error | error | error | error | Runtime error: TopK/GatherElements index out of range; rewritten candidate not yet qualified. |
| `GeneralDetect` | `yolov8n_detect` | INT8 | 0.1495 | 0.06851 ‡ | 0.2813 ‡ | 0.5844 ‡ | No decoded objects (‡); coarse fused-output quantization loses scores. |
| `E2ERotated` | `yolo26n_e2e_obb` | INT8 | 0.001539 | 0.001767 ‡ | 0.008112 ‡ | 0.01628 ‡ | No decoded objects (‡); coarse fused-output quantization loses scores. |
| `YoloV8Rotated` | `yolov8n_obb_raw` | INT8 | 0.6044 | 0.4283 | 1.763 | 4.537 | — |
| `Yolo26Rotated` | `yolo26n_obb_raw` | INT8 | 0.4315 | 0.3487 | 1.416 | 2.843 | — |
| `YoloV5Rotated` | `yolov5_rotated` | INT8 | — | 0.09146 ‡ | 0.3705 ‡ | 0.7413 ‡ | No FP16 source: QAT model retains INT8; test image has no matching objects (‡). |
| `GeneralRotated` | `yolov8n_obb` | INT8 | 0.3472 | 0.01065 ‡ | 0.0441 ‡ | 0.08837 ‡ | No decoded objects (‡); coarse fused-output quantization loses scores. |
| `GeneralClassify` | `yolov8n_classify` | INT8 | 0.0001666 | 0.0005384 | 0.004765 | 0.0134 | — |
| `Embedding` | `yolov8n_classify_embedding` | INT8 | 0.0001348 | 0.0004969 | 0.004489 | 0.01292 | — |
| `GeneralYoloPose` | `yolov8n_pose` | INT8 | 0.003619 | 0.000868 ‡ | 0.004582 ‡ | 0.009266 ‡ | No decoded objects (‡); coarse fused-output quantization loses scores. |
| `YoloV8Keypoint` | `yolov8n_face` | INT8 | 0.02257 | 0.01344 | 0.05769 | 0.1178 | — |
| `Yolo26Keypoint` | `yolo26n_pose_raw` | INT8 | 0.01788 | 0.01396 | 0.05963 | 0.1216 | — |
| `E2EKeypoint` | `yolo26n_e2e_pose` | INT8 | 0.001255 ‡ | 0.001731 ‡ | 0.008213 ‡ | 0.01648 ‡ | No objects in either precision; INT8 fused-output quantization also loses scores. |
| `RtmoKeypoint` | `rtmo_s_pose` | conversion fails | — | — | — | — | Conversion fails: RKNN Reshape rewrite, [1,300,2] to [300,256]. |
| `GeneralKeypoint` | `rtmpose_m_pose` | INT8 | 0.01914 | 0.005418 | — | — | B4/B8 supported; RKNN timings pending remeasurement. |
| `GeneralYoloSegment` | `yolov8n_segment` | INT8 | — | — | — | — | Withdrawn: FP16/INT8 object counts differ; output-0 score quantization loses probabilities. |
| `YoloV8Segment` | `yolov8n_segment_raw` | INT8 | 0.9454 | 0.6709 | 3.088 | 6.639 | — |
| `Yolo26Segment` | `yolo26n_segment_raw` | INT8 | 1.268 | 0.8964 | 4.078 | 8.119 | — |
| `E2ESegment` | `yolo26n_e2e_segment` | INT8 | error | error | error | error | Runtime error: TopK/GatherElements index out of range; candidate has a score mismatch (0.126). |
| `SemanticSegment` | `yolo26n_semantic` | INT8 | 1.058 | 0.8881 | 4.552 | 9.425 | — |
| `Ocr` | `ppocrv6_tiny_det` | INT8 | 0.7009 | 0.7688 | 3.135 | 6.252 | — |
| `Ocr` | `ppocrv6_tiny_rec` | INT8 | 0.2434 | 0.01659 | 0.06792 | 0.1427 | — |

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

| Platform | Path | Model | B1 (images/s) | B4 (images/s) | B8 (images/s) |
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

For RTX 3060, amortized wall time per image (`1000 / images_per_second`) falls from
2.79 to 1.53 to 1.16 ms for detect, 2.76 to 2.01 to 1.61 ms for segment, 2.50 to
1.53 to 1.17 ms for pose, and 0.902 to 0.733 to 0.704 ms for classify (B1/B4/B8).
The raw detect JSON reports mean whole-call times of 22.29 / 48.88 / 74.18 ms with
8 concurrent callers. These include pool waits; they are not isolated inference latency.
The throughput relation is approximately `8 * B * 1000 / mean_call_ms` for this workload.

Batching amortizes launches and can improve GPU utilization, but uploads still stage each
host image, device outputs return to the host for decoding, and the pipeline synchronizes
after preprocessing and inference. CPU decoding loops over the samples within each batch;
OpenCV is single-threaded in this test, including mask decoding. The 3/2/3 pools remain fixed
as batch size grows. These verified code paths explain why the pipeline cannot be assumed
to scale linearly; segment and classify show smaller gains than detect and pose. The current
aggregate results do not isolate GPU compute, transfers, CPU decoding or pool contention,
so their individual bottleneck shares remain unmeasured. No additional speed tests were run.

CPU ORT and TensorRT batch-1/4/8 correctness checks passed on the current 640 x 640 fixtures.
The earlier RKNN batch comparison failed strict result equality even with the board service
stopped; its throughput figures above do not establish equivalence with batch 1. Further performance testing has been stopped; model conversion and output-type checks are complete.
