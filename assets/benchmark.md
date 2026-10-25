# Benchmarks

[Preprocessing](#preprocessing) · [Postprocessing](#postprocessing) · [Pipeline](#pipeline)

- Reconfigure and rebuild Release before measuring. Google Benchmark wall time:
  `--benchmark_min_time=0.5s --benchmark_min_warmup_time=0.2 --benchmark_repetitions=3`;
  report the median. Setup/allocation before the loop is excluded; work and synchronization
  within the call are included.
- Fixtures: [template reference](template/README.md#test-models). Raw results stay in
  `assets/test/benchmark_results/`; before/after analysis belongs in [perf.md](perf.md).
  Blank/`—` cells are unmeasured; failures are explained in the table.

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

## Preprocessing

- One synthetic BGR uint8 frame: 1920x1080, 2560x1440 or 3840x2160 → 640x640 or 1280x1280;
  linear letterbox, pad 114. One caller, one OpenCV thread, 20 extra warm-up calls; ms/call.
- CPU/Metal use host/shared memory. CPU/Metal/CUDA output planar RGB FP32, mean 0/std 255.
  CUDA host includes upload; CUDA device starts resident. Output stays on the device.
- RK3588 uses RGA only: RGB uint8 dma-buffer output, no normalization (RKNN handles it).
  RGA host includes staging; RGA dma excludes its initial upload. Both include padding,
  resize with BGR/RGB swap, per-call mapping/cache work and synchronous `IM_SYNC` completion.
  This uint8 workload differs from the FP32 paths.
- RGA requires RGB888 row pitch aligned to 16 pixels and backend-compatible below-4-GB
  allocations. Report hardware failures/CPU fallbacks explicitly; do not accept fallback timing
  as RGA data. RGA timings are pending.

```sh
hybridrt_bench_preprocess --benchmark_filter='^BM_(Resolution|CudaResolution)/[0-9]+/[0-9]+/(640/640|1280/1280)/(0|1)/'
hybridrt_bench_preprocess --benchmark_filter='^BM_RgaResolution/'
```

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

- Filter, NMS, masks and source mapping for all 26 template algos at B1/B4/B8. Whole-batch ms;
  divide by B for ms/image. Every sample receives the same source image and transform.
- Inputs: bus.jpg; airport.jpg for rotated boxes; ocr_document.png for detection and its
  (419,16,241,50) crop for recognition. Thresholds come from the template.
- 5600X: ORT CPU, FP32 outputs, one caller, OpenCV and ORT intra/inter-op threads 1, no affinity.
  Inference runs before timing; outputs are reused. Inference, conversion and copies are excluded.
  Run `hybridrt_bench_postprocess --benchmark_filter='^BM_Type/'`.

5600X CPU, FP32 ONNX outputs:

| Type | Template algo | Model outputs | CPU B1 | CPU B4 | CPU B8 | Status / failure reason |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `YoloV8Detect` | `yolov8_detect` | `Bx144x{80x80, 40x40, 20x20}` | 0.08651 | 0.5376 | 0.8593 | — |
| `Yolo26Detect` | `yolo26n_detect_raw` | `Bx84x{80x80, 40x40, 20x20}` | 0.07775 | 0.3313 | 0.8343 | — |
| `YoloV5Detect` | `yolov5_detect` | `Bx255x{80x80, 40x40, 20x20}` | 0.03349 | 0.1361 | 0.2672 | — |
| `DetrDetect` | `rtdetr_l_detect` | `Bx300x84` | 0.02247 | 0.08836 | 0.1773 | — |
| `E2EDetect` | `yolo26n_e2e_detect` | `Bx300x6` | 0.0006106 | 0.002441 | 0.004816 | — |
| `GeneralDetect` | `yolov8n_detect` | `Bx84x8400` | 0.03729 | 0.1422 | 0.4661 | — |
| `E2ERotated` | `yolo26n_e2e_obb` | `Bx300x7` | 0.0006787 | 0.0027 | 0.005408 | — |
| `YoloV8Rotated` | `yolov8n_obb_raw` | `Bx80x{128x128, 64x64, 32x32}` | 0.2393 | 0.9762 | 2.042 | — |
| `Yolo26Rotated` | `yolo26n_obb_raw` | `Bx20x{128x128, 64x64, 32x32}` | 0.1786 | 0.7271 | 1.493 | — |
| `YoloV5Rotated` | `yolov5_rotated` | `Bx24x{160x160, 80x80, 40x40}` | 0.04678 † | 0.21 † | 0.3774 † | FP32 outputs from a Q/DQ graph; no objects on airport.jpg (†). |
| `GeneralRotated` | `yolov8n_obb` | `Bx20x6300` | 0.1672 | 0.695 | 1.353 | — |
| `GeneralClassify` | `yolov8n_classify` | `Bx1000` | 6.287e-05 | 0.0002516 | 0.0007444 | — |
| `Embedding` | `yolov8n_classify_embedding` | `Bx1000` | 6.274e-05 | 0.0002539 | 0.000683 | — |
| `GeneralYoloPose` | `yolov8n_pose` | `Bx56x8400` | 0.001689 | 0.006223 | 0.01205 | — |
| `YoloV8Keypoint` | `yolov8n_face` | `Bx80x{80x80, 40x40, 20x20}` | 0.009569 | 0.03947 | 0.07912 | — |
| `Yolo26Keypoint` | `yolo26n_pose_raw` | `Bx56x{80x80, 40x40, 20x20}` | 0.007591 | 0.03092 | 0.06243 | — |
| `E2EKeypoint` | `yolo26n_e2e_pose` | `Bx300x57` | 0.001242 | 0.004967 | 0.01074 | — |
| `RtmoKeypoint` | `rtmo_s_pose` | `Bx300x5`, `Bx300x17x3` | 0.001699 | 0.006327 | 0.01258 | — |
| `GeneralKeypoint` | `rtmpose_m_pose` | `Bx17x384`, `Bx17x512` | 0.005668 | 0.02198 | 0.04524 | — |
| `GeneralYoloSegment` | `yolov8n_segment` | `Bx116x8400`, `Bx32x160x160` | 0.389 | 1.895 | 4.26 | — |
| `YoloV8Segment` | `yolov8n_segment_raw` | `Bx176x{60x80, 30x40, 15x20}`, `Bx32x120x160` | 0.2996 | 1.342 | 3.788 | — |
| `Yolo26Segment` | `yolo26n_segment_raw` | `Bx116x{80x80, 40x40, 20x20}`, `Bx32x160x160` | 0.377 | 1.585 | 4.504 | — |
| `E2ESegment` | `yolo26n_e2e_segment` | `Bx300x38`, `Bx32x160x160` | 0.271 | 1.16 | 2.529 | — |
| `SemanticSegment` | `yolo26n_semantic` | `Bx19x80x80` | 0.3619 | 2.973 | 7.952 | — |
| `Ocr` | `ppocrv6_tiny_det` | `Bx1x640x640` | 0.2927 | 1.218 | 2.348 | — |
| `Ocr` | `ppocrv6_tiny_rec` | `Bx40x6906` | 0.09217 | 0.3732 | 0.7657 | — |

All 78 cases passed output-type and decoded-count checks against the preceding run.
`†` means zero objects on airport.jpg for the user QAT OBB model: filtering only.
Raw results: `postprocess_fp32_after_direct_batch.json`.

RK3588 CPU, RKNN FP16 B1 versus native INT8 B1/B4/B8 (whole-batch ms):

- Same inputs and method, one caller/OpenCV thread, A76 cores 4-7; service running, frequency
  scaling enabled. Host preprocessing, inference, SDK conversion and copies are untimed.
  Output types, scales and zero points are checked; details are in the
  [RKNN model reference](template/README.md#rknn-on-rk3588).
- `‡` means zero decoded objects: filtering only, without NMS/masks. Coarse fused-output
  quantization can lose scores; these cells do not establish accuracy or full-decoder speedups.
  Withdrawn/failed/pending cases retain their reasons below.
- Raw results: `rknn_postprocess_matrix.json`, `rknn_all_int8_query.log` and
  `rknn_postprocess_matrix/` in the local results directory.

| Type | Template algo | FP16 B1 | INT8 B1 | INT8 B4 | INT8 B8 | Status / failure reason |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `YoloV8Detect` | `yolov8_detect` | 0.3652 | 0.1239 | 0.5141 | 1.081 | — |
| `Yolo26Detect` | `yolo26n_detect_raw` | 0.3152 | 0.1156 | 0.4839 | 0.995 | — |
| `YoloV5Detect` | `yolov5_detect` | 0.07864 | 0.04357 | 0.1843 | 0.3765 | — |
| `DetrDetect` | `rtdetr_l_detect` | — | — | — | — | Conversion fails: RKNN Conv rewrite channel mismatch (4 vs 256). |
| `E2EDetect` | `yolo26n_e2e_detect` | error | error | error | error | Runtime error: TopK/GatherElements index out of range; rewritten candidate not yet qualified. |
| `GeneralDetect` | `yolov8n_detect` | 0.1495 | 0.06851 ‡ | 0.2813 ‡ | 0.5844 ‡ | No decoded objects (‡); coarse fused-output quantization loses scores. |
| `E2ERotated` | `yolo26n_e2e_obb` | 0.001539 | 0.001767 ‡ | 0.008112 ‡ | 0.01628 ‡ | No decoded objects (‡); coarse fused-output quantization loses scores. |
| `YoloV8Rotated` | `yolov8n_obb_raw` | 0.6044 | 0.4283 | 1.763 | 4.537 | — |
| `Yolo26Rotated` | `yolo26n_obb_raw` | 0.4315 | 0.3487 | 1.416 | 2.843 | — |
| `YoloV5Rotated` | `yolov5_rotated` | — | 0.09146 ‡ | 0.3705 ‡ | 0.7413 ‡ | No FP16 source: QAT model retains INT8; test image has no matching objects (‡). |
| `GeneralRotated` | `yolov8n_obb` | 0.3472 | 0.01065 ‡ | 0.0441 ‡ | 0.08837 ‡ | No decoded objects (‡); coarse fused-output quantization loses scores. |
| `GeneralClassify` | `yolov8n_classify` | 0.0001666 | 0.0005384 | 0.004765 | 0.0134 | — |
| `Embedding` | `yolov8n_classify_embedding` | 0.0001348 | 0.0004969 | 0.004489 | 0.01292 | — |
| `GeneralYoloPose` | `yolov8n_pose` | 0.003619 | 0.000868 ‡ | 0.004582 ‡ | 0.009266 ‡ | No decoded objects (‡); coarse fused-output quantization loses scores. |
| `YoloV8Keypoint` | `yolov8n_face` | 0.02257 | 0.01344 | 0.05769 | 0.1178 | — |
| `Yolo26Keypoint` | `yolo26n_pose_raw` | 0.01788 | 0.01396 | 0.05963 | 0.1216 | — |
| `E2EKeypoint` | `yolo26n_e2e_pose` | 0.001255 ‡ | 0.001731 ‡ | 0.008213 ‡ | 0.01648 ‡ | No objects in either precision; INT8 fused-output quantization also loses scores. |
| `RtmoKeypoint` | `rtmo_s_pose` | — | — | — | — | Conversion fails: RKNN Reshape rewrite, [1,300,2] to [300,256]. |
| `GeneralKeypoint` | `rtmpose_m_pose` | 0.01914 | 0.005418 | — | — | B4/B8 supported; RKNN timings pending remeasurement. |
| `GeneralYoloSegment` | `yolov8n_segment` | — | — | — | — | Withdrawn: FP16/INT8 object counts differ; output-0 score quantization loses probabilities. |
| `YoloV8Segment` | `yolov8n_segment_raw` | 0.9454 | 0.6709 | 3.088 | 6.639 | — |
| `Yolo26Segment` | `yolo26n_segment_raw` | 1.268 | 0.8964 | 4.078 | 8.119 | — |
| `E2ESegment` | `yolo26n_e2e_segment` | error | error | error | error | Runtime error: TopK/GatherElements index out of range; candidate has a score mismatch (0.126). |
| `SemanticSegment` | `yolo26n_semantic` | 1.058 | 0.8881 | 4.552 | 9.425 | — |
| `Ocr` | `ppocrv6_tiny_det` | 0.7009 | 0.7688 | 3.135 | 6.252 | — |
| `Ocr` | `ppocrv6_tiny_rec` | 0.2434 | 0.01659 | 0.06792 | 0.1427 | — |

## Pipeline

- Full `ImageModel::process_sync`: YOLOv8n detect, segment, pose and classify, static 640x640
  B1/B4/B8. Paths are listed above; TensorRT uses FP32 `--noTF32`, RKNN uses FP16.
  DVPP → CANN has not been measured.
- One shared model, pools 3/2/3, 8 callers, one OpenCV thread. Each caller cycles through the
  same 32 sorted COCO128 host BGR images from its own offset, passing B images/call;
  5 untimed calls per caller. Report aggregate images/s (higher is better).
  `hybridrt_bench_pipeline --benchmark_filter='^BM_Batch/'`.

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

CPU ORT and TensorRT batch correctness checks passed. RKNN failed strict batch-result
equality; its throughput does not establish equivalence with B1. The board service was
stopped during pipeline measurement and restored afterward.
