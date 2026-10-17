# Performance notes

Optimizations of preprocessing, postprocessing and the `ImageModel` pipeline, newest first. Each entry
records the bottleneck, the change, how it was measured and the numbers before and after, so later
work can rerun the same measurement. Plain feature work is not listed here.

## 2026-10-17: segmentation postprocessing

**Bottleneck.** On the RK3588 board, `InstanceSegment` took 11.4 ms and `SemanticSegment` 54 to 57 ms
per call, against 0.05 to 1.7 ms for the other decoders. Timing each section on the board showed:

- `SemanticSegment`: the argmax over `[1, 21, 512, 512]` took 37 ms. It looped over pixels and read
  each pixel's 21 channel values one plane (1 MB) apart, so nearly every read missed the cache.
- `InstanceSegment`: 5.9 ms scanning candidates and 6.0 ms building masks. The scan read the score of
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
| RK3588 `InstanceSegment` postprocess (yolov8n-seg, 640 x 480) | 11.4 ms | 1.2 ms |
| RK3588 `InstanceSegment` `process_sync` | 59.8 ms | 49 ms |
| RK3588 `SemanticSegment` postprocess (LRASPP, 512 x 512, 21 classes) | 54 to 57 ms | 8.7 ms |
| RK3588 `SemanticSegment` `process_sync` | 117 ms | 70.5 ms |
| x86 `InstanceSegment` postprocess (yolov8n-seg) | 2.72 ms | 0.67 ms |

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
