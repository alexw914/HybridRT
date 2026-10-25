# Performance changes

Before/after optimization records, newest first. Current benchmark results and reproduction commands are in [benchmark.md](benchmark.md).

Each entry states the cause/change, measurement conditions, data, and validation or limits. Values are historical for that source revision; later entries may supersede them. Missing historical settings are marked rather than inferred.

## 2026-10-25: decode INT8 outputs as integers and replace the semantic model

On RK3588 several INT8 postprocessors were slower than the FP16 run of the same model. Running
each INT8 output a second time after dequantizing it to float32 (same values, float path)
located the cost in the code path, not the data or memory. Candidate decoding called
`HeadGridView::at()` per element: DFL softmax, YOLOv5 class/box sigmoids and keypoint
visibility dequantized and called `exp` for every value. OCR DB, CTC and SimCC dequantized
whole outputs first; `std::max_element` on int8 runs one value at a time. The LR-ASPP semantic
model upsampled 21 class maps to 512 x 512 in the graph, and its masks were built by a
warpAffine plus one compare per class at source resolution.

- `QuantTables` holds, per INT8 head, the sigmoid of all 256 values, exp(-scale * d) for the
  DFL softmax, and per-class integer score bounds; it is rebuilt only when scale or zero
  point change. YOLOv8/YOLO26 filter class scores as integers and rank NMS by logit; only kept
  boxes take the sigmoid. Single-output heads compare INT8 scores against an exact strict
  integer bound.
- OCR DB binarizes and checks the map as integers and scores boxes by the INT8 mean; CTC and
  SimCC take a vectorized INT8 maximum, then its first position. The DB binarization loop
  itself did not vectorize for float or int8 maps (`&&` in its range check, int32 bounds):
  0.434 of 0.64 ms on the 5600X. Bitwise checks and int8-typed bounds vectorize it.
- `yolo26n_semantic` (YOLO26n-sem, 19 x 80 x 80 logits) replaces `lraspp_voc_segment`. Masks
  are written from per-axis nearest tables in one pass of row runs instead of
  warpAffine + per-class compares; 283 of 874,800 bus.jpg pixels differ from warpAffine
  nearest, all at exact half-pixel rounding.

**Measurement:** RK3588 A76 cores 4-7, cross-built GCC 12.2.1 Release, OpenCV 4.6.0 one thread,
librknnrt 2.3.2, template algos and test images of `assets/benchmark.md`, inputs resized on the
build host. `hybridrt_bench_postprocess` via `run_rknn_postprocess.py`, Google Benchmark wall
time, three repetitions >=0.5 s after >=0.2 s warm-up, median ms. Before is the previous INT8 B1
table measured the same way; the board service stayed running this time.

| Postprocessor | INT8 B1 before | INT8 B1 after | FP16 B1 after |
| --- | ---: | ---: | ---: |
| YoloV8Detect | 0.146 | 0.1239 | 0.3652 |
| Yolo26Detect | 0.114 | 0.1156 | 0.3152 |
| YoloV5Detect | 0.08607 | 0.04357 | 0.07864 |
| YoloV8Rotated | 0.4901 | 0.4283 | 0.6044 |
| Yolo26Rotated | 0.3463 | 0.3487 | 0.4315 |
| YoloV8Keypoint | 0.02534 | 0.01344 | 0.02257 |
| Yolo26Keypoint | 0.02057 | 0.01396 | 0.01788 |
| GeneralKeypoint (simcc) | 0.01993 | 0.00542 | 0.01914 |
| YoloV8Segment | 0.6855 | 0.6709 | 0.9454 |
| Yolo26Segment | 0.8849 | 0.8964 | 1.268 |
| Ocr detection | 1.697 | 0.7688 | 0.7009 |
| Ocr recognition | 0.2935 (in-place path, scalar argmax) | 0.01659 | 0.2434 |
| SemanticSegment | 13.75 (LR-ASPP) | 0.8881 | 1.058 |

Attribution on the board with a one-off harness (5 x 0.5 s, median, core 4): before the change,
INT8 vs the same INT8 data as float32 was 0.0880 vs 0.0764 ms for YoloV5Detect, 0.0259 vs 0.0230
for YoloV8Keypoint, 0.0202 vs 0.0148 for SimCC and 1.578 vs 1.450 for OCR DB. YOLO26n-sem took
9.54 ms FP16 / 8.83 ms INT8 with the previous mask code, against 29.8 / 10.9 for LR-ASPP.

OCR DB on FP32 maps, 5600X (same method, no affinity): 0.6162 -> 0.2974 ms. The INT8 OCR DB map
decodes 38 boxes against 37 for FP16 and costs more downstream; on the same INT8 data the INT8
path takes 0.777 ms and its float32 copy 0.872 ms.

**Validation / limits:** Unit tests cover INT8 DFL/threshold heads, OCR DB/CTC and SimCC. Other
rows of the 5600X FP32 table stayed within 8% of the previous run, about its run-to-run noise. Fused-output INT8 models (`General*`, `E2E*`) still decode no objects; their
times measure filtering only. Raw results are local in `rknn_postprocess_matrix.json` and
`int8_lut_postprocess_5600x.json`.

## 2026-10-25: tighten INT8 score filtering bounds

A floor-rounded quantized threshold admitted the zero-point bin for small positive score
thresholds. SIMD block filtering then entered the scalar loop for almost every zero-confidence
anchor. Compute the first representable INT8 value meeting the threshold, correcting against
the same float dequantization as decoding; final score predicates are unchanged.

**Measurement:** RK3588 A76 cores 4-7, cross-built GCC 12.2.1 Release, OpenCV 4.6.0 one thread,
librknnrt 2.3.2, 640 x 640 batch-1 INT8 YOLOv8n exports, bus.jpg. Google Benchmark wall time,
three repetitions >=0.5 s after >=0.2 s warm-up, median ms; service stopped, frequency scaling enabled.

| Postprocessor | Before | After |
| --- | ---: | ---: |
| GeneralDetect | 2.13893 | 0.06869 |
| GeneralYoloSegment | 2.13874 | 0.06925 |
| GeneralYoloPose | 0.02660 | 0.000868 |

**Validation / limits:** All 256 INT8 values were compared with scalar score predicates at 9,216
threshold/scale/zero-point combinations, including adjacent float rounding boundaries; x86 and
AArch64 passed. The measured detection/segment/pose models decode zero objects: their fused
coordinate/confidence output has scale around 2.57 (pose 3.16), losing confidence precision during
model quantization. The speedup fixes redundant filtering, not model accuracy; these figures do
not compare equivalent detections with FP32 models. Raw results remain local in
`standard640_postprocess_rk3588.json` and `standard640_postprocess_rk3588_tight_bound_b1.json`.
Further testing uses the 5600X; RK measurements above are retained only as this optimization record.

## 2026-10-24: recognize all OCR lines of an image in one call

OCR recognition ran once per detected line. Collecting all line regions in one ImageModel call lets a native batch-8 recognizer process eight lines per inference; tall lines still rotate on the host.

**Measurement:** WSL2, Ryzen 5 5600X / RTX 3060, TensorRT 10.14.1 FP32 noTF32, CUDA preprocessing. hrt-ocr: 20 copies of ocr_document.png, 37 lines each, three runs. ORT CPU: pipeline BM_Ocr, one OpenCV thread, CPUs 2-5, median of 3 repetitions >=0.5 s.

| Recognizer | TensorRT, ms per image | ORT CPU, ms per image |
| --- | ---: | ---: |
| `ppocrv6_tiny_rec` (batch 1) | 52.9 / 56.2 / 50.2 | 228 |
| `ppocrv6_tiny_rec_batch8` | 24.7 / 24.8 / 25.3 | 219 |

**Validation / limits:** All 740 TensorRT lines retain their text, scores within 1e-4. CPU gain is within noise; only the tiny recognizer has a batch fixture. RKNN and Windows were not measured.

## 2026-10-24: decode instance masks on OpenCV's threads

Per-object mask decoding serialized the expensive prototype sampling. MaskBatch queues retained objects and decodes independent stripes with reusable per-thread scratch.

**Measurement:** WSL2 Ryzen 5 5600X, GCC 13 Release, OpenCV 4.6.0, ORT outputs of bus.jpg. postprocess BM_Type, CPUs 2-5; baseline a30868b, alternating 3 rounds x 3 repetitions >=0.5 s; median ms.

| Postprocessor | Before, 1 / 4 threads | After, 1 / 4 threads |
| --- | ---: | ---: |
| GeneralYoloSegment | 0.267 / 0.264 | 0.267 / 0.209 |
| YoloV8Segment (raw heads) | 0.316 / 0.315 | 0.307 / 0.248 |
| Yolo26Segment (raw heads) | 0.395 / 0.401 | 0.370 / 0.324 |
| E2ESegment | 0.278 / 0.279 | 0.277 / 0.179 |

**Validation / limits:** Masks match with one and four OpenCV threads. Only about five objects survive on bus.jpg, limiting scaling; NMS and score scans remain serial. RK3588 was not measured.

## 2026-10-24: launch a fused CUDA batch at once

Per-sample staging alternated uploads and kernel launches. Stage each sample separately, then launch descriptors together, up to 16 samples per CUDA kernel.

**Measurement:** WSL2 Ryzen 5 5600X / RTX 3060, CUDA 13.0, batch of four 1920x1080 frames to 640x640 RGB float32. preprocess BM_Batch, CPUs 2-5; baseline 1b7d231; alternating 5 rounds x 3 repetitions >=0.5 s, median ms/batch.

| Case | Before | After | Change |
| --- | ---: | ---: | ---: |
| NV12 host frames | 1.338 | 1.126 | -16% |
| BGR host frames | 2.292 | 2.097 | -9% |
| BGR device frames | 0.158 | 0.150 | -5% |
| NV12 device frames | 0.154 | 0.148 | -4% |
| Batch 1, all four modes | | | within 1% |

**Validation / limits:** CudaBatchMatchesSingleCalls checks 20 mixed host/device BGR/NV12 samples across two launches. Host uploads still dominate. RKNN, Ascend, Metal and Windows were not measured.

## 2026-10-24: convert NV12 and NV21 in the fused CUDA kernel and drop NPP

NV12/NV21 still used staged NPP conversion. The fused kernel now converts sampled YUV taps with OpenCV BT.601 fixed-point arithmetic; NPP operators and linkage were removed.

**Measurement:** Same WSL2 RTX 3060 environment, CUDA 13.0; distinct 1920x1080 frames to 640x640 RGB float32, CPUs 2-5. Baseline b88a223, alternating 3 rounds x 3 repetitions >=0.5 s; median ms/frame.

| Case | Before | After | Change |
| --- | ---: | ---: | ---: |
| NV12 device frames, batch 4 | 0.321 | 0.040 | -87% |
| NV12 device frames, batch 1 | 0.350 | 0.076 | -78% |
| NV12 host frames, batch 4 | 0.575 | 0.343 | -40% |
| NV12 host frames, batch 1 | 0.620 | 0.364 | -41% |
| BGR host frames, batch 4 | 0.596 | 0.588 | noise |
| BGR device frames, batch 4 | 0.041 | 0.040 | noise |

**Validation / limits:** CPU comparison: differences <=1 uint8 step, >99.8% identical. Unsupported formats kept CPU fallback in this version. Later CUDA changes may supersede those format limits.

## 2026-10-24: replace staged NPP preprocessing with one fused CUDA kernel

About ten NPP launches repeatedly traversed the output. One CUDA kernel combines sampling, padding, channel order and affine normalization; descriptors are passed by value, up to 16 per launch.

**Measurement:** WSL2 Ryzen 5 5600X / RTX 3060, GCC 13 Release, CUDA 13.0. preprocess BM_CudaResolution, CPU 4; baseline aa3f2d9; alternating 5 rounds x 3 repetitions >=0.5 s after 0.2 s warm-up. Linear letterbox, pad 114, planar RGB float32, std 255; median ms.

| Case | Before | After | Change |
| --- | ---: | ---: | ---: |
| 1920 x 1080 to 640 x 640, device source | 0.226 | 0.076 | -66% |
| 1920 x 1080 to 1280 x 1280, device source | 0.643 | 0.151 | -77% |
| 3840 x 2160 to 640 x 640, device source | 0.234 | 0.086 | -63% |
| 1920 x 1080 to 640 x 640, host source | 0.758 | 0.597 | -21% |
| 1920 x 1080 to 1280 x 1280, host source | 1.171 | 0.673 | -43% |
| 3840 x 2160 to 640 x 640, host source | 2.258 | 2.059 | -9% |
| 3840 x 2160 crop to 224 x 224, host source | 0.418 | 0.344 | -18% |
| 1920 x 1080 upload only | 0.538 | 0.527 | noise |

**Validation / limits:** 97x61 random inputs cover formats, crops, resize and normalize modes: 99.87% of 18.8M values identical, all within one uint8 step. Pageable upload dominates host cases; later entries address NV12 and batched launches.

## 2026-10-24: fuse CPU letterbox padding, conversion and normalization

CPU letterbox filled and converted padding twice. Resize into an ROI-sized uint8 scratch, convert only that region, and fill final float padding strips; reuse scratch allocations.

**Measurement:** WSL2 Ryzen 5 5600X, GCC 13 Release, OpenCV 4.6.0, one OpenCV thread, CPU 4. preprocess BM_Task/0 and BM_Resolution; baseline 885505a, alternating 5 rounds x 3 repetitions >=0.5 s after 0.2 s warm-up. Linear letterbox, pad 114, RGB float32, std 255; median ms (round range).

| Case | Before, ms | After, ms | Change |
| --- | ---: | ---: | ---: |
| `BM_Task/0` detect, 810 x 1080 to 640 x 480 | 0.478 (0.454-0.509) | 0.445 (0.441-0.489) | -6.8% |
| 1920 x 1080 to 640 x 360 (no padding) | 0.531 (0.519-0.623) | 0.546 (0.531-0.601) | +2.9% |
| 1920 x 1080 to 640 x 640 | 0.667 (0.647-0.715) | 0.631 (0.576-0.712) | -5.3% |
| 1920 x 1080 to 960 x 960 | 1.187 (1.103-1.316) | 0.951 (0.904-1.016) | -19.9% |
| 3840 x 2160 to 640 x 360 (no padding) | 0.583 (0.556-0.597) | 0.578 (0.545-0.640) | -0.9% |

**Validation / limits:** Byte-value and pipeline tests pass. No-padding cases overlap run-to-run noise. Resize dominates; non-BGR/RGB format pairs and other devices were not measured.

## 2026-10-23: avoid resetting image layouts and rechecking unchanged explicit strides

Image construction temporarily reset an already final layout; explicit-stride resize revalidated unchanged metadata. Preserve final layouts and reuse checked strides while the allocation fits.

**Measurement:** Apple M5 Pro, macOS Release, one caller/OpenCV thread. preprocess BM_ImageLayout, synthetic 640x480 RGB uint8, pitch 1984; baseline 233982d, alternating 3 rounds x 3 repetitions >=0.2 s, median of round medians.

| Case | Before | After |
| --- | ---: | ---: |
| `BM_ImageLayout/0`: wrap padded image, including initialization | 240.24 ns | 174.15 ns |
| `BM_ImageLayout/1`: resize with the same explicit stride | 29.14 ns | 4.23 ns |
| `BM_ImageLayout/2`: reshape with a default stride | 47.27 ns | 47.17 ns |

**Validation / limits:** BM_Convert/0 separately measured 0.03754 -> 0.03637 ms, with overlapping round ranges (before 0.03663-0.03807, after 0.03229-0.03754); no general throughput gain claimed. Default-stride work remains; CUDA/RGA/Ascend hardware unmeasured.

## 2026-10-22: skip score blocks below the threshold and argmax semantic maps in pixel blocks

Decoded score rows branched on every anchor; semantic argmax repeatedly wrote maxima and labels. Skip 16-score blocks below threshold and keep blocked argmax state in NEON/SSE2 registers, with scalar tails.

**Measurement:** Apple M5 Pro, macOS Release, ORT 1.31.0 CPU. postprocess BM_Type, alternating 3 rounds x 3 repetitions >=0.3 s; median of rounds. Other template algos stayed within +/-1 us.

| Type (algo) | Before | After |
| --- | ---: | ---: |
| `GeneralDetect` (`yolov8n_detect`) | 0.159 ms | 0.024 ms |
| `GeneralRotated` (`yolov8n_obb`) | 0.138 ms | 0.074 ms |
| `GeneralYoloSegment` (`yolov8n_segment`) | 0.218 ms | 0.116 ms |
| `GeneralYoloPose` (`yolov8n_pose`) | 0.002 ms | 0.001 ms |
| `SemanticSegment` (`lraspp_voc_segment`) | 2.84 ms | 1.63 ms |

**Validation / limits:** hrt-validate JSON is byte-identical for five affected model types. Standalone semantic argmax: 1.73 -> 0.75 ms blocked scalar -> 0.59 ms NEON. SSE2 correctness checked under Rosetta, not timed; RK3588 unmeasured. Warp/mask construction and OCR remain bottlenecks.

## 2026-10-22: vector int8 dequantization in dequantize_buffer

Whole-output int8 conversion depended on compiler vectorization and repeated row checks. Convert 16 values with NEON tbl/scvtf or SSE2 unpack/shifts, then scalar tails; preserve subtraction/multiplication order.

**Measurement:** Apple M5 Pro, macOS Release Apple clang, one thread. Ad-hoc dequantize_buffer driver, random int8, 15 rounds; baseline/vector/scalar builds interleaved across 3 rounds, medians within 1%.

| Output (int8 to float32) | Before | After | After, scalar loop |
| --- | ---: | ---: | ---: |
| 1 x 1000 (classify) | 0.059 us | 0.059 us | 0.059 us |
| 1 x 40 x 6625 (OCR recognition) | 15.6 us | 15.5 us | 15.6 us |
| 1 x 21 x 512 x 512 (semantic segment) | 325 us | 323 us | 324 us |
| 1 x 40 x 6625, rows padded to 6640 bytes | 18.2 us | 15.6 us | 18.2 us |

**Validation / limits:** Bit-identical to scalar dequantization; contiguous data already vectorizes, so gains are on padded rows. SSE2 checked under Rosetta only. RK3588, native x86 hardware and 32-bit ARM unmeasured.

## 2026-10-20: accumulate mask prototypes in place over the box region

Whole int8 prototypes were dequantized even when few boxes survived. Accumulate coefficient-weighted prototypes only over each box ROI, folding scale/zero point into coefficients; fix NHWC access.

**Measurement:** Apple M5 Pro, macOS Release, ORT 1.31.0 CPU, one OpenCV thread. postprocess BM_Type, alternating 3 rounds x 3 repetitions >=0.3 s, median of rounds.

| Type (algo) | Before | After |
| --- | ---: | ---: |
| `E2ESegment` (`yolo26n_e2e_segment`) | 0.1603 ms | 0.1238 ms |

**Validation / limits:** Four bus.jpg masks are byte-identical; other available algos stayed within +/-0.5%. Int8 models and RK3588 were not timed; NHWC remains strided. Other segmentation fixtures were missing.

## 2026-10-20: read end-to-end rows lazily instead of dequantizing whole outputs

End-to-end decoders copied entire int8/strided outputs before filtering. Read only labels/scores and passing rows through HeadGrid; remove temporary packed buffers and retain scratch only for whole-output consumers.

**Measurement:** Apple M5 Pro, macOS Release, ORT 1.31.0 CPU, one OpenCV thread. postprocess BM_Type, alternating 3 rounds x 3 repetitions >=0.3 s; median of rounds. Available fixtures were float32.

| Type (algo) | Before | After |
| --- | ---: | ---: |
| `E2ESegment` (`yolo26n_e2e_segment`) | 0.1632 ms | 0.1630 ms |
| `GeneralDetect` (`yolov8n_detect`) | 0.1586 ms | 0.1605 ms |
| `GeneralRotated` (`yolov8n_obb`) | 0.1403 ms | 0.1405 ms |
| `GeneralYoloPose` (`yolov8n_pose`) | 0.0025 ms | 0.0025 ms |
| `GeneralClassify` (`yolov8n_classify`) | 0.00006 ms | 0.00006 ms |
| `Ocr` (`ppocrv6_tiny_det`) | 0.3462 ms | 0.3437 ms |

**Validation / limits:** Float32 remains zero-copy and timings are within noise. Int8 end-to-end gains were not measured; several E2E/DETR/RTMO fixtures were missing. Direct classify-result writes were rejected after a 0.056 -> 0.085 us regression.

## 2026-10-20: single-channel cell scans for YOLOv5 objectness and one-class heads

A runtime element-type check slowed YOLOv5 objectness scans. Scan a typed score plane against the raw threshold, then decode passing cells; one-class YOLOv8/26 heads reuse this path.

**Measurement:** Apple M4, macOS Release, ORT 1.31.0 CPU, one OpenCV thread. postprocess BM_Type, alternating 3 rounds x 3 repetitions >=0.3 s; median of rounds.

| Type (algo) | Before (template `HeadGrid`) | After |
| --- | ---: | ---: |
| `YoloV5Detect` (`yolov5_detect`) | 0.0241 ms | 0.0166 ms |
| `YoloV5Rotated` (`yolov5_rotated`) | 0.0647 ms | 0.0335 ms |
| `YoloV8Keypoint` (`yolov8n_face`) | 0.0058 ms | 0.0050 ms |
| `Yolo26Keypoint` (`yolo26n_pose_raw`) | 0.0052 ms | 0.0045 ms |

**Validation / limits:** All 24 image algos and both OCR recognizers keep identical outputs; other types stayed within +/-2.5%. End-to-end models were float32; int8 unit-tested, RK3588 not retimed.

## 2026-10-20: restore non-fused device preprocessing stages

Missing fused support incorrectly forced complete CPU fallback. Check whether every required independent operator is supported before falling back; run existing device stages when possible.

**Measurement:** Xeon Gold 6462C / RTX 4090 GPU 1, Linux Release, driver 550.54.14, CUDA 12.4.131, OpenCV 4.5.4. preprocess BM_CudaResolution: one caller, 20 warm-ups +0.2 s, 3 repetitions >=0.5 s, wall median ms. Synthetic BGR, linear letterbox, pad 114, RGB float32/std 255, CUDA output; host includes upload, all calls synchronize.

| Source -> target | Input location | Before (CPU fallback) | After (NPP stages) |
| --- | --- | ---: | ---: |
| 1920 x 1080 -> 640 x 640 | Host | 1.920 | 0.382 |
| 1920 x 1080 -> 640 x 640 | CUDA | 1.554 | 0.045 |
| 1920 x 1080 -> 1280 x 1280 | Host | 5.453 | 0.445 |
| 1920 x 1080 -> 1280 x 1280 | CUDA | 4.750 | 0.106 |
| 3840 x 2160 -> 640 x 640 | Host | 4.477 | 1.278 |
| 3840 x 2160 -> 640 x 640 | CUDA | 2.609 | 0.046 |
| 3840 x 2160 -> 1280 x 1280 | Host | 8.874 | 1.334 |
| 3840 x 2160 -> 1280 x 1280 | CUDA | 6.581 | 0.106 |

**Validation / limits:** Operator-count and fallback tests pass; CUDA scratch-lifetime regression passed 20 runs, followed by Linux/macOS full CTest. NPP was still multi-stage in this version; later entries replace it. RKNN/RGA and DVPP hardware were not revalidated.

## 2026-10-19: OCR line regions on the source device

OCR cropped every line on the host. Recognize directly from source-image regions; only tall lines need a host conversion and counterclockwise rotation into reused storage.

**Measurement:** RTX 4090 GPU 1, CUDA 12.4, TensorRT 10.16.1.11 FP16, CUDA preprocessing, tiny det+rec. 300 calls after 6 warm-ups, mean of 4 interleaved rounds; 896x528 document (8 lines, one tall) and rotated page (12 tall lines), host/CUDA sources.

| Case | Before | After |
| --- | --- | --- |
| Document, host input | 6.544 ms | 6.544 ms |
| Document, CUDA input | 6.646 ms | 6.573 ms |
| Turned document, host input | 8.630 ms | 8.669 ms |
| Turned document, CUDA input | 8.794 ms | 8.793 ms |

**Validation / limits:** Texts/boxes match except one noise-box score 0.2729 -> 0.2871, consistent with crop-border sampling differences. ORT CPU JSON is byte-identical. Tall lines retain host rotation; later entry batches recognition.

## 2026-10-18: OCR CTC decoding

CTC scanned every class twice per step. One maximum pass plus validation of its probability removes the redundant scan while retaining text and scores.

**Measurement:** Apple M5 Pro, macOS, ORT 1.31.0 CPU/CoreML MLProgram, one OpenCV thread, 3-repetition median. postprocess/pipeline BM_Ocr on 896x528 ocr_document.png (37 lines); one-line crop 241x50 into 320x48.

| Case | Before | After |
| --- | --- | --- |
| CTC postprocess, tiny recognizer, one 241 x 50 line | 0.311 ms | 0.121 ms |
| `OcrModel`, tiny det + tiny rec, ORT CPU | 63.4 ms | 54.9 ms |
| `OcrModel`, tiny det + small rec, ORT CPU | 179 ms | 157 ms |
| `OcrModel`, tiny det + tiny rec, ORT CoreML | not measured | 37.9 ms |
| `OcrModel`, tiny det + small rec, ORT CoreML | not measured | 74.6 ms |

**Validation / limits:** hrt-model-bench engine means: tiny det 11.7 ms CPU / 2.3 CoreML; tiny rec 0.92 / 0.76; small rec 3.5 / 1.3. Recognition still dominated; later entry adds static batching. Long lines remain compressed to width 320.

## 2026-10-17: candidate scans of detection heads

Raw heads and decoded outputs traversed class planes per anchor. Sequential raw-score prefilters visit only passing cells; read decoded rows in place and retain reusable candidate storage.

**Measurement:** Rock5B+ RK3588, librknnrt 2.3.2, CPU 4: 50 warm-ups, best of 3 rounds x300 calls on RKNN outputs. x86 Ryzen 5 5600X, ORT 1.31.0 CPU, pinned postprocess benchmark, 3-repetition median.

| Case | Before | After |
| --- | --- | --- |
| RK3588 `YoloV8Detect`, YOLOv8n raw heads, 80 classes, int8 | 1.078 ms | 0.145 ms |
| RK3588 `YoloV8Detect`, YOLO26n raw heads, 80 classes, int8 | 1.031 ms | 0.112 ms |
| RK3588 `YoloV8Detect`, 2 classes, 1280 input, int8 | 0.415 ms | 0.091 ms |
| RK3588 `YoloV8Rotated`, YOLOv8n-obb raw heads, 15 classes, int8 | 1.003 ms | 0.490 ms |
| RK3588 `YoloV8Keypoint`, YOLOv8n-pose raw heads, int8 | 0.070 ms | 0.044 ms |
| RK3588 `GeneralDetect`, YOLOv8n, fp16 | 0.771 ms | 0.456 ms |
| RK3588 `GeneralRotated`, YOLOv8n-obb, fp16 | 0.457 ms | 0.413 ms |
| RK3588 `GeneralYoloPose`, YOLOv8n-pose, fp16 | 0.051 ms | 0.008 ms |
| RK3588 `GeneralYoloSegment`, YOLOv8n-seg, fp16 | 1.244 ms | 1.257 ms (masks dominate) |
| x86 `GeneralDetect` (yolov8n) | 0.400 ms | 0.225 ms |
| x86 `GeneralYoloSegment` (yolov8n-seg) | 0.661 ms | 0.551 ms |
| x86 `GeneralYoloPose` (yolov8n-pose) | 0.021 ms | 0.004 ms |
| x86 `GeneralRotated` (yolov8n-obb) | 0.223 ms | 0.207 ms |

**Validation / limits:** ORT boxes/scores/keypoints/angles/masks match; int8 vs dequantized results differ by <=6e-5. RK A55 was ~3.5x slower than A76, so affinity matters. Rotated NMS and NPU-output copies (~0.07 ms/MB) remain.

## 2026-10-17: segmentation postprocessing

Semantic argmax read distant planes per pixel; instance masks made repeated full-resolution passes. Use channel-major argmax, per-class candidate scans and separable ROI mask sampling.

**Measurement:** Rock5B+ RK3588/librknnrt 2.3.2, one thread, FP16 models, bus.jpg; mean of 100 calls after 10 warm-ups, public stage API and temporary decoder timers. Desktop Ryzen 5 5600X, ORT 1.31.0 CPU, postprocess BM_Task/1.

| Case | Before | After |
| --- | --- | --- |
| RK3588 `GeneralYoloSegment` postprocess (yolov8n-seg, 640 x 480) | 11.4 ms | 1.2 ms |
| RK3588 `GeneralYoloSegment` `process_sync` | 59.8 ms | 49 ms |
| RK3588 `SemanticSegment` postprocess (LRASPP, 512 x 512, 21 classes) | 54 to 57 ms | 8.7 ms |
| RK3588 `SemanticSegment` `process_sync` | 117 ms | 70.5 ms |
| x86 `GeneralYoloSegment` postprocess (yolov8n-seg) | 2.72 ms | 0.67 ms |

**Validation / limits:** Mask IoU vs previous warpAffine path >=0.991; old sampling quantized coordinates to 1/32 pixel. Semantic per-class masks still cost ~3 ms; other decoded scans were addressed in the next entry.

## 2026-10-14: stage only the source ROI for host images (8f015c5)

Device preprocessing uploaded full host frames for small crops. Stage the interleaved source ROI plus a two-pixel interpolation margin.

**Measurement:** RTX 4090, preprocess BM_CudaResolution host_crop: 480x960 crop from 3840x2160 into 224x224, 3 interleaved runs; 1.25-1.35 -> 0.39-0.40 ms.

**Validation / limits:** YUV, planar and device sources still stage whole in this version; repetition duration and thread settings were not preserved.

## 2026-10-14: preprocess Metal images in place on unified memory (bee3191)

Metal copied unified-memory images through staging buffers. Wrap source/target pages as no-copy Metal buffers with image offsets (~3 us/image).

**Measurement:** Apple M4, preprocess BM_Resolution, wall/process-CPU ms: 4K->1280x1280 1.83/1.14 -> 1.32/0.60; 1080p->640x640 0.61/0.30 -> 0.51/0.20.

**Validation / limits:** Repetition counts and statistic were not preserved; results apply to shared-memory Metal paths.

## 2026-10-13: fuse CPU tensor conversion and normalization (26e8d66)

CPU tensor conversion and normalization traversed float output separately. Fuse uint8 RGB/BGR conversion and per-channel affine normalization.

**Measurement:** Apple M5 Pro, one OpenCV thread: 1080p->640x640 1.06 -> 0.566 ms; 4K->1280x1280 4.38 -> 2.28 ms.

**Validation / limits:** Repetition counts and statistic were not preserved; later CPU letterbox work removes redundant padding conversion.

## 2026-10-12: copy TensorRT host IO through pinned memory (e3455d5)

Pageable TensorRT IO reduced copy bandwidth and per-frame staging allocation added overhead. Use pinned host IO and reusable device staging buffers.

**Measurement:** RTX 4090, YOLOv8n FP16 TensorRT, pipeline_bench median of 3 rounds: CUDA-preprocessed single call 1.51 -> 1.24 ms; 3/2/3 pools with 8 callers 6.33 -> 5.03 ms.

**Validation / limits:** Exact image size, repetition duration and software versions were not preserved in this entry.
