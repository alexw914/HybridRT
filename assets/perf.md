# Performance notes

Optimizations of preprocessing, postprocessing and the `ImageModel` pipeline, newest first. Each entry
records the bottleneck, the change, how it was measured and the numbers before and after, so later
work can rerun the same measurement. Plain feature work is not listed here.

## 2026-10-20: accumulate mask prototypes in place over the box region

**Bottleneck.** Instance segmentation decoders turned the whole prototype output into float32
before decoding: int8 prototypes, e.g. 32 x 160 x 160, were dequantized on every run even with
no detections, although each mask reads only its box region. Float32 prototypes were read in
place, but `cv::scaleAdd` per channel and mask cost more than the arithmetic on small regions.

**Change.** `MaskDecoder` reads the prototypes through `HeadGrid` and accumulates
`coefficients . prototypes` row by row over the box region only. For int8 the scale is folded
into the coefficients and the zero point into the starting value, so prototypes are never
dequantized. `YoloV8Segment`, `Yolo26Segment`, `GeneralYoloSegment` and `E2ESegment` drop their
prototype scratch buffers. NHWC prototypes are now read in their layout; before, they were read as
NCHW and gave wrong masks.

**Measurement.** Apple M4, macOS, Release, ONNX Runtime 1.31.0 CPU, one OpenCV thread,
`hybridrt_bench_postprocess` `BM_Type`, 3 repetitions of at least 0.3 s, median; the previous
commit and this change were run interleaved over 3 rounds and the median of rounds is shown.

| Type (algo) | Before | After |
| --- | ---: | ---: |
| `E2ESegment` (`yolo26n_e2e_segment`) | 0.1603 ms | 0.1238 ms |

The other available template algos (detect, rotated, pose, classify, OCR detection) stayed
within +/-0.5%. Masks of `yolo26n_e2e_segment` on `bus.jpg` (4 objects) are byte-identical
before and after.

**Remaining.** int8 prototypes are covered by unit tests only; no int8 segmentation model was
measured and RK3588 was not rerun. The `YoloV8Segment`, `Yolo26Segment` and
`GeneralYoloSegment` template models were missing locally. NHWC prototypes take a strided loop
that does not vectorize.

## 2026-10-20: read end-to-end rows lazily instead of dequantizing whole outputs

**Bottleneck.** `float_data()` turned every int8 or strided output into a float32 copy each run,
even for decoders that read a fraction of it: end-to-end detect, rotated, keypoint and segment
outputs skip most rows after the label and score, and RTMO reads keypoints only for boxes kept
by NMS. Strided int8 outputs also allocated a temporary packed buffer on every call.

**Change.** `E2EDetect`, `E2ERotated`, `E2EKeypoint`, `E2ESegment` (detections), `DetrDetect` and
`RtmoKeypoint` read their outputs through `HeadGrid`, which dequantizes only the elements read,
in the order label, score, box, then keypoints or mask coefficients of passing rows; their
scratch buffers are gone. `float_data()` dequantizes or packs strided rows straight into
the scratch buffer, without the temporary packed buffer. Classify, embedding, OCR, semantic
segmentation, `GeneralKeypoint` and mask prototypes still read whole outputs through it.

**Measurement.** Apple M4, macOS, Release, ONNX Runtime 1.31.0 CPU, one OpenCV thread,
`hybridrt_bench_postprocess` `BM_Type`, 3 repetitions of at least 0.3 s, median; the previous
commit and this change were run interleaved over 3 rounds and the median of rounds is shown.
Only these template models were available locally, all float32:

| Type (algo) | Before | After |
| --- | ---: | ---: |
| `E2ESegment` (`yolo26n_e2e_segment`) | 0.1632 ms | 0.1630 ms |
| `GeneralDetect` (`yolov8n_detect`) | 0.1586 ms | 0.1605 ms |
| `GeneralRotated` (`yolov8n_obb`) | 0.1403 ms | 0.1405 ms |
| `GeneralYoloPose` (`yolov8n_pose`) | 0.0025 ms | 0.0025 ms |
| `GeneralClassify` (`yolov8n_classify`) | 0.00006 ms | 0.00006 ms |
| `Ocr` (`ppocrv6_tiny_det`) | 0.3462 ms | 0.3437 ms |

Float32 outputs stay zero-copy, so these are unchanged within noise. Writing classify and
embedding scores straight into the result was tried and dropped: value-initializing the
result first took classify from 0.056 to 0.085 us.

**Remaining.** The int8 gain was not measured: no int8 end-to-end template model is available
locally and RK3588 was not rerun. The E2EDetect, E2ERotated, E2EKeypoint, DetrDetect and
RtmoKeypoint template models were missing, so they were not benchmarked. Mask prototypes are
still dequantized whole.

## 2026-10-20: single-channel cell scans for YOLOv5 objectness and one-class heads

**Bottleneck.** `HeadGrid` became a plain class whose element type (float32 or int8) is checked
at run time. `YoloV5Detect` and `YoloV5Rotated` read the objectness of every cell and anchor
through `HeadGrid::raw()`, so that check ran in their innermost loop: with the class template
they took 0.024 and 0.064 ms, with the plain class 0.031 and 0.088 ms.

**Change.** `CellScanner::scan()` with one channel compares the typed row directly against the
raw bound and returns the passing cells, without the running-maximum buffer. The YOLOv5
decoders scan each anchor's objectness plane this way and decode only the passing cells.
Raw YOLOv8/YOLO26 heads with one class (face, pose) take the same path. Decoding is otherwise
unchanged; candidates are pushed anchor by anchor instead of cell by cell before NMS.

**Measurement.** Apple M4, macOS, Release, ONNX Runtime 1.31.0 CPU, one OpenCV thread,
`hybridrt_bench_postprocess` `BM_Type`, 3 repetitions of at least 0.3 s, median; the previous
commit and this change were run interleaved over 3 rounds and the median of rounds is shown.

| Type (algo) | Before (template `HeadGrid`) | After |
| --- | ---: | ---: |
| `YoloV5Detect` (`yolov5_detect`) | 0.0241 ms | 0.0166 ms |
| `YoloV5Rotated` (`yolov5_rotated`) | 0.0647 ms | 0.0335 ms |
| `YoloV8Keypoint` (`yolov8n_face`) | 0.0058 ms | 0.0050 ms |
| `Yolo26Keypoint` (`yolo26n_pose_raw`) | 0.0052 ms | 0.0045 ms |

All other types stayed within +/-2.5% (within run-to-run noise). `hrt-validate` and `hrt-ocr`
outputs of all 24 template algos and both OCR recognizers on the test images are identical
before and after.

**Remaining.** Only float32 template models were compared end to end; int8 decoding is covered
by the unit tests. RK3588 int8 timings were not remeasured.

## 2026-10-20: restore non-fused device preprocessing stages

**Bottleneck.** The shared flow treated a missing fused implementation as requiring
whole-call CPU processing for device-only buffers. CUDA's NPP implementation already
supported fill, resize, conversion and normalization separately, but none ran. Even
CUDA-resident input was downloaded, processed on the host and uploaded again.

**Change.** Check every required independent operator before taking the existing host
fallback. When the device supports the complete sequence, run the existing stages.
Otherwise preserve the previous whole-call CPU fallback, including synchronization
and staging; fused and host-accessible paths are unchanged. No SDK implementation or
public interface was changed, and no fused CUDA kernel was added.

**Measurement.** Linux x86_64, Xeon Gold 6462C, physical RTX 4090 GPU 1 (shared, idle at
start), driver 550.54.14, CUDA 12.4.131, OpenCV 4.5.4, Release. Current source was synced,
reconfigured and rebuilt in the fixed validation directory. `hybridrt_bench_preprocess`
`BM_CudaResolution`, one caller, 20 explicit warm-up calls, 0.2 s benchmark warm-up,
minimum 0.5 s per repetition, wall-clock median of 3 repetitions. Synthetic uint8 BGR,
linear letterbox, padding 114, planar RGB float32 with std 255. Outputs remain in CUDA
memory. Host cases include uploading the frame; device cases start with resident input.
Every call ends with stream synchronization. Units below are ms/call.

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

Six current size pairs (1080p, 2560 x 1440 and 4K, each to 640 x 640 / 1280 x 1280)
for CUDA, Xeon CPU, M4 CPU and Metal are recorded in [benchmark.md](benchmark.md).
The earlier fallback baseline has no 2560 x 1440 measurements. M4 timings were rerun,
not interpreted as an optimization of the unchanged CPU/Metal implementations.

**Validation.** A CUDA operator-counting regression checks that all four device stages
execute, and that disabling each required stage preserves whole-call CPU fallback,
for full/ROI normalization, padding and repeated calls. CUDA crop staging also passed.
`ImageModelTest.KeepsCudaPreprocessorScratchAliveUntilStreamCompletes` passed 20
consecutive runs on the selected GPU, followed by full CTest: 180 entries on Linux
(6 hardware-dependent skips), 192 on macOS (8 hardware-dependent skips), no failures.
macOS Metal geometry and NV12 fallback tests passed with GPU access enabled.
The raw segmentation test expectation was updated for the current 640 x 480 YOLOv8
fixture (three people, no tie), distinct from the 640 x 640 YOLO26 fixture (four people).

**Other devices.** The check runs only for a non-CPU preprocessor without a fused path whose
source or target lacks host access. CPU and Metal (fused only, unified host memory) never
reach it. DVPP is fused only, so the check fails and the whole-call fallback is unchanged.
RGA supports only fill and resize: float or planar targets still fall back, while
uint8 interleaved targets without normalization may now stay on RGA. CUDA still falls back
for InterRgba sources (CPU resizes them, NPP does not) and Gray to three-channel output.

**Remaining.** NPP still launches multiple operations. For host input, PCIe upload is
now the main cost. RKNN/RGA and DVPP hardware were not revalidated in this run; their
SDK implementations and unsupported-stage fallback are retained. Local raw timing
JSON and test logs remain under `assets/test/benchmark_results/` and are untracked.

## 2026-10-19: OCR line regions on the source device

**Bottleneck.** `OcrModel` cropped every detected line on the host (`TextCropper`) before
recognition, so a source that was not host uint8 BGR/RGB, such as a CUDA image, was first converted
and copied to host BGR in full, even though the recognition preprocessor already reads a region of
its source and only lines at least 1.5 times taller than wide need a host rotation.

**Change.** Lines are recognized from their truncated region of the source through
`ImageModel::process_sync(src, region)`, on the recognition preprocessor's device. Only tall lines
use the host: the source directly when it is host BGR/RGB, otherwise one conversion per call, then
a counterclockwise rotation into reused storage. `TextCropper` is removed.

**Measurement.** RTX 4090 (GPU 1, shared), CUDA 12.4, TensorRT 10.16.1.11 FP16 engines from
`trtexec --fp16`, preprocessing on CUDA; a scratch driver times 300 `OcrModel::process_sync` calls
after 6 warm-up calls, tiny det + tiny rec, with the image in host memory and in CUDA memory; mean
of 4 interleaved rounds. Images: a local 896 x 528 document (8 lines, one tall) and the same page
turned 90 degrees clockwise (12 lines, all tall).

| Case | Before | After |
| --- | --- | --- |
| Document, host input | 6.544 ms | 6.544 ms |
| Document, CUDA input | 6.646 ms | 6.573 ms |
| Turned document, host input | 8.630 ms | 8.669 ms |
| Turned document, CUDA input | 8.794 ms | 8.793 ms |

Texts, boxes and scores are identical, except one noise box on the turned page whose score moves
from 0.2729 to 0.2871, likely because region resizing reads source pixels next to the region where a crop
replicated its border. On macOS ORT CPU the JSON results of `hrt-ocr` are byte-identical.

**Remaining.** The CUDA input now costs only 0.03 ms more than host input on this image; the saved
copy grows with resolution. Tall lines still need the host conversion, as preprocessors cannot
rotate; recognition is still one batch-1 inference per line.

## 2026-10-18: OCR CTC decoding

**Bottleneck.** `OcrModel` recognizes each detected line separately, so CTC decoding runs once per
line. Each of the 40 time steps scanned all classes twice, once to check every value was a finite
probability and once for the maximum: 0.31 ms per PP-OCRv6 tiny line (6906 classes), a third of
the 0.92 ms ORT CPU inference, about 11 ms of a 37-line document.

**Change.** One `max_element` pass per step; only the maximum is checked to lie in [0, 1], which
still rejects logits. Decoded text and scores are unchanged.

**Measurement.** Apple M5 Pro, macOS, ONNX Runtime 1.31.0, `hybridrt_bench_postprocess` and
`hybridrt_bench_pipeline` (`BM_Ocr`), one OpenCV thread, median of 3 repetitions, on the local
`ocr_document.png` (896 x 528, 37 lines); CoreML uses the MLProgram options of
`hrtc_device_config.json`.

| Case | Before | After |
| --- | --- | --- |
| CTC postprocess, tiny recognizer, one 241 x 50 line | 0.311 ms | 0.121 ms |
| `OcrModel`, tiny det + tiny rec, ORT CPU | 63.4 ms | 54.9 ms |
| `OcrModel`, tiny det + small rec, ORT CPU | 179 ms | 157 ms |
| `OcrModel`, tiny det + tiny rec, ORT CoreML | not measured | 37.9 ms |
| `OcrModel`, tiny det + small rec, ORT CoreML | not measured | 74.6 ms |

Other stages, unchanged: detection preprocess 0.48 ms, line crop plus `text_line` preprocess
0.02 ms, DB postprocess 0.34 ms. `hrt-model-bench` engine means: tiny det 11.7 ms (CoreML 2.3),
tiny rec 0.92 ms (CoreML 0.76), small rec 3.5 ms (CoreML 1.3).

**Remaining.** Recognition is about 70 % of a CPU call: one batch-1 inference per line. Batched
recognition needs batch > 1 exports, and very long lines are still squeezed into 320 pixels.

## 2026-10-17: candidate scans of detection heads

**Bottleneck.** Raw anchor-free heads (`YoloV8Detect`, `YoloV8Rotated`, `YoloV8Keypoint`) visited every
cell and, for each, read the score of every class from a different NCHW plane: 80 planes per cell for
COCO heads, more concurrent streams than the prefetcher follows. Decoded single outputs
(`GeneralDetect`, `GeneralRotated`, `GeneralYoloPose`, `GeneralYoloSegment`) walked anchors first,
reading every class row per anchor, and converted whole int8 tensors to float before looking at
them. On RK3588 the A55 cores (cpu0 to cpu3) run the same decoder about 3.5 times slower than the A76
cores (cpu4 to cpu7), which made unpinned timings bimodal.

**Change.**

- `scan_cells()` first reduces the class planes of a head to the best raw score per cell, one plane
  after another into a running maximum (a sequential pass the compiler vectorizes; int8 stays int8),
  and returns the cells reaching the threshold's raw bound. Decoders then run their unchanged
  per-class checks, distance bins and keypoints on those cells only.
- `RowView` reads decoded `[1, rows, N]` outputs in place: each class row is scanned in order, int8
  scores are compared as integers against the quantized threshold (one step below it, then checked
  exactly), and only the values of passing anchors are dequantized, with the same formula as
  before. Per-call candidate vectors became reused members.
- Results are identical: every ORT test model gives the same boxes, scores, keypoints, angles and
  masks before and after; int8 decoding equals decoding the dequantized float32 tensor (largest
  difference 6e-5 from operation order).

**Measurement.** RK3588 (Rock 5B+, librknnrt 2.3.2), postprocess only on one RKNN inference, pinned
to cpu4 with `taskset`, best of 3 rounds of 300 calls after 50 warm-up calls. x86: Ryzen 5 5600X,
ONNX Runtime 1.31.0, `hybridrt_bench_postprocess`, pinned, median of 3 repetitions.

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

On int8 heads the integer prefilter also beats dequantizing first: 0.22 ms against 1.07 ms for the
2-class 1280 detector before this change.

Remaining: rotated NMS (polygon IoU) dominates `YoloV8Rotated` now; the RKNN engine still copies
each output from NPU memory (0.07 ms per MB); decoding threads are not pinned to big cores by the
framework, so applications should pin them.

## 2026-10-17: segmentation postprocessing

**Bottleneck.** On the RK3588 board, `GeneralYoloSegment` took 11.4 ms and `SemanticSegment` 54 to 57 ms
per call, against 0.05 to 1.7 ms for the other decoders. Timing each section on the board showed:

- `SemanticSegment`: the argmax over `[1, 21, 512, 512]` took 37 ms. It looped over pixels and read
  each pixel's 21 channel values one plane (1 MB) apart, so nearly every read missed the cache.
- `GeneralYoloSegment`: 5.9 ms scanning candidates and 6.0 ms building masks. The scan read the score of
  every class for one anchor at a time, a row (6300 floats) apart. Each mask made four
  source-resolution passes: `warpAffine` into a float image, `compare`, `bitwise_and`, plus the
  prototype sum.
- RKNN outputs were in host memory and copied at heap speed, so memory type was not the cause.

**Change.**

- Semantic argmax runs channel-major: each pass reads one channel plane in order against a running
  maximum, which the compiler vectorizes. int8 outputs with a positive scale are compared without
  dequantizing the whole map.
- The instance candidate scan reads one class row at a time and builds boxes only above the
  threshold; candidates keep their per-class anchor order, so NMS and results are unchanged.
- Mask sampling is separable, since the mapping only scales and shifts: each prototype-region row is
  interpolated across once, then each mask row blends two of them and thresholds in the same pass.
  The float sample image and the `compare`/`bitwise_and` passes are gone. Masks match the previous
  ones with IoU 0.991 or more; the old `warpAffine` quantized sample positions to 1/32 pixel.

**Measurement.** Board: Radxa Rock 5B+ (RK3588, librknnrt 2.3.2), one thread, fp16 models,
`bus.jpg`, mean of 100 calls after 10 warm-up calls. Stages are timed by running `ImageModel`'s steps
through the public API (`Engine`, `ImagePreprocessor`, the postprocessor on the engine's outputs), and
sections inside a decoder with temporary timers. Desktop: Ryzen 5 5600X, ONNX Runtime 1.31.0 CPU, `hybridrt_bench_postprocess` `BM_Task/1`.

| Case | Before | After |
| --- | --- | --- |
| RK3588 `GeneralYoloSegment` postprocess (yolov8n-seg, 640 x 480) | 11.4 ms | 1.2 ms |
| RK3588 `GeneralYoloSegment` `process_sync` | 59.8 ms | 49 ms |
| RK3588 `SemanticSegment` postprocess (LRASPP, 512 x 512, 21 classes) | 54 to 57 ms | 8.7 ms |
| RK3588 `SemanticSegment` `process_sync` | 117 ms | 70.5 ms |
| x86 `GeneralYoloSegment` postprocess (yolov8n-seg) | 2.72 ms | 0.67 ms |

Remaining: 3 ms of the semantic time builds one source-resolution mask per class. Other decoded
single-output types (`GeneralDetect`, `GeneralYoloPose`, `GeneralRotated`) still scan per anchor
across class rows; they take 0.05 to 0.7 ms here and were left unchanged.

## 2026-10-14: stage only the source ROI for host images (8f015c5)

**Bottleneck.** A cascaded model crops a small region, such as a person to classify, out of a large
frame, but a device preprocessor given a host frame uploaded the whole frame.

**Change.** Host interleaved sources (BGR, RGB, RGBA) upload only `src_roi` plus a 2-pixel margin for
the interpolation taps; YUV, planar and device sources still stage whole.

**Measurement.** RTX 4090, `BM_CudaResolution` `host_crop` (a 480 x 960 crop of a 3840 x 2160 host frame
letterboxed into 224 x 224), 3 interleaved runs: 1.25 to 1.35 ms before, 0.39 to 0.40 ms after.

## 2026-10-14: preprocess Metal images in place on unified memory (bee3191)

**Bottleneck.** Metal preprocessing copied host images through shared staging buffers.

**Change.** Host source and target images are wrapped as no-copy Metal buffers spanning the enclosing
pages, with an offset to the image; about 3 us per image.

**Measurement.** Apple M4, `preprocess_bench` `BM_Resolution` metal cases, wall / process CPU ms per
frame: 4K to 1280 x 1280 1.83 / 1.14 before, 1.32 / 0.60 after; 1080p to 640 x 640 0.61 / 0.30 before,
0.51 / 0.20 after.

## 2026-10-13: fuse CPU tensor conversion and normalization (26e8d66)

**Bottleneck.** The CPU preprocessor converted uint8 RGB/BGR to planar float and normalized in
separate passes over the float output.

**Change.** One fused pass converts and applies the per-channel affine normalization.

**Measurement.** Apple M5 Pro, OpenCV one thread: 1080p to 640 x 640 1.06 ms before, 0.566 ms after;
4K to 1280 x 1280 4.38 ms before, 2.28 ms after.

## 2026-10-12: copy TensorRT host IO through pinned memory (e3455d5)

**Bottleneck.** Host-side engine IO used pageable memory, so host-device copies ran at reduced
bandwidth and blocked the host thread; staging buffers were allocated and freed every frame.

**Change.** TensorRT hands out CUDA-pinned host memory through `Engine::create_host_memory()` and keeps
its device staging buffers across calls.

**Measurement.** RTX 4090, YOLOv8n fp16 TensorRT, `pipeline_bench` medians of 3 rounds: single call with
CUDA preprocessing 1.51 ms before, 1.24 ms after; 3/2/3 stages on 8 threads 6.33 ms before, 5.03 ms after.
