# Manifest reference

`hrtc_template.json` contains a representative model for each postprocessor type. The
algos use ONNX Runtime and model paths under `assets/test/onnx/`, relative to the repository
root. Image-task files use `<type>.onnx`; OCR files add their stage and optional variant.
The first algo shows every preprocess key. Engine-specific `config` keys are
collected in `hrtc_device_config.json`, using the same raw YOLOv8 detector as an example;
`hrtc_template.json` omits them and uses engine defaults. Tests and benchmarks load these same algos. Model files
and images stay local and untracked; only this directory, `assets/perf.md` and `assets/benchmark.md` are committed.
Values are defaults or model-specific values where the defaults do not fit. Check the whole
manifest with `hrt-compress -c -j assets/template/hrtc_template.json`.

## Package and algos

| Key | Value |
| --- | --- |
| `name`, `version` | Required strings. |
| `description` | Optional string. |
| `algos[].name` | Unique, non-empty. |
| `algos[].config` | Object with `preprocess`, `engine` and `postprocess`, a path to such a JSON file (absolute or relative to the working directory), or `"string:<json>"`. |
| `engine.backends[]` | Engines of the algo, tried in order: the first whose device is available and whose model loads is used. |
| `backends[].engine_type` | `ort`, `ncnn`, `mnn`, `trt`, `rknn`, `cann`. |
| `backends[].config` | Optional object of that engine's settings, below. |
| `backends[].proto`, `.data` | Model files: an absolute path or one relative to the working directory, `"string:<text>"` or a JSON object stored as text. ncnn uses `proto` for the `.param` text and `data` for the `.bin`; the other engines use `data` only. |

The algo config only describes the model. How many calls each stage serves at once is not
part of the manifest: pass `ImageModelOptions{preprocess, engine, postprocess}` (default 1, at
most 64) to `ImageModel::create`.

## preprocess

| Key | Default | Values |
| --- | --- | --- |
| `device` | `auto` | `auto` (device of the engine input), `cpu`, `cuda`, `rga`, `dvpp`, `metal` (macOS, never picked by `auto`) |
| `interpolation_mode` | `linear` | `linear`, `nearest`, `cubic` |
| `resize_type` | `normal` | `normal` (stretch), `keep_ratio` (top-left), `letter_box` (centered as Ultralytics' LetterBox: scaled size rounded half to even and at least 1, offset rounded down; alias `center_keep_ratio`), `center_crop` (square input, needs `resize_short_size`), `pad_outside` (grows the source rect to the input aspect ratio, scaled by `pad_scale`), `text_line` (full input height, proportional width rounded up and capped at input width, right padding) |
| `dst_image_format` | `planar_rgb` | `planar_rgb`, `planar_bgr`, `inter_rgb`, `inter_bgr`, `inter_rgba`, `gray`, `nv12`, `nv21`, `yu12`, `yv12`; planar and gray need an NCHW input, interleaved NHWC |
| `normalize_type` | `full` | `full`, `roi` (only the resized region), `none` |
| `mean`, `std` | `[0, 0, 0]`, `[1, 1, 1]` | Per channel: `x = (x - mean) / std` |
| `default_color` | `[0, 0, 0]` | Padding color in the channel order of `dst_image_format`, as `mean` and `std` |
| `resize_short_size` | `-1` | Short side before `center_crop` |
| `pad_scale` | `1.0` | Scale of `pad_outside` |

Fused preprocessing is optional. A device without a fused implementation uses its
independent operators when it supports every required stage. CUDA has only a fused kernel:
it converts uint8 BGR, RGB, NV12, NV21, YU12 (I420) or YV12 sources with any interpolation
into 3-channel planar or interleaved uint8 or float32 inputs, following OpenCV's uint8 color
conversion and sampling, so it matches the CPU result up to OpenCV's own rounding
differences. Other conversions (gray inputs or sources, RGBA or float sources) run on the CPU.
A static batch whose samples all fuse runs as one kernel launch (up to 16 samples per kernel);
on the CPU the samples of a batch run on OpenCV's threads.

Host frames for CUDA preprocessing are uploaded first, which usually costs more than the kernel
(about 0.53 ms of 0.59 ms for a 1920 x 1080 BGR frame into 640 x 640 on an RTX 3060). Two choices
of the caller shorten it:

- Pass NV12 (or NV21, I420, YV12) as decoders and cameras produce it: 1.5 bytes per pixel
  instead of 3, which cut the same case to 0.35 ms.
- Keep frames in page-locked memory, for example from `cudaMallocHost` or a decoder that
  allocates it, and wrap them with `core::Image::wrap(core::MemoryType::Cpu, ...)`. CUDA detects
  page-locked pointers by itself; plain copies measured 14.3 GB/s from them against 12.6 GB/s
  from pageable memory on that machine (WSL2, so native Linux may differ).

Only the source region plus a two-pixel margin of an interleaved host frame is uploaded, so
cascaded crops upload much less than the frame.

For device-only memory, an unsupported stage retains the whole-call CPU fallback,
including source download and result upload. Host-accessible buffers can use CPU operators
directly. Fusion and fallback decisions do not change the model's preprocess parameters.

A cascade passes each detection as `src_rect` to `process_sync`. Only `pad_outside` clips it to
the image; for the other resize types a region leaving the image fails the call, so grow and clip
detections with `intersect(cover_pixels(pad_scale(box, scale_w, scale_h)), image.rect())` from
`hrt::core`. `center_crop` can still leave a clipped box when `resize_short_size` is below the
input size.

## Engine config

`hrtc_device_config.json` shows `config` objects for ORT, TensorRT, MNN, ncnn, RKNN and CANN
in one example algo. The ORT entry points to the existing fixed ONNX model; native backend
paths are conversion examples, not additional bundled models. To run that file, keep the
backend you need and provide its converted model files. Copy its `config` into a model
manifest when overriding engine defaults. The two files are independent manifest examples;
there is no automatic merge or extra runtime configuration file.

ORT uses explicit single-thread settings in the device example; MNN and ncnn show four
threads. `input_shapes`, custom operator/plugin libraries and OpenCL cache settings can be
filled for a deployment. The engine tests read ORT/MNN/ncnn configs from this device example;
model tests and benchmarks continue reading `hrtc_template.json`.

The example's preprocessing targets the float32 ONNX input. For RKNN `image_input: true`,
use the platform-specific preprocessing below (uint8 interleaved RGB with model-side
normalization). Model conversion and preprocessing must match the selected backend.

| Engine | Keys (all optional) |
| --- | --- |
| `ort` | `intra_op_num_threads`, `inter_op_num_threads` (0: ONNX Runtime's default); `execution_provider`: `cpu`, `coreml` (falls back to the CPU without CoreML); `provider_options` (CoreML only): `ModelFormat` `NeuralNetwork`/`MLProgram`, `MLComputeUnits` `ALL`/`CPUOnly`/`CPUAndGPU`, `RequireStaticInputShapes` and `EnableOnSubgraphs` `"0"`/`"1"`; `custom_op_libraries`: paths; `input_shapes` |
| `ncnn` | `num_threads` (4) |
| `mnn` | `forward_type`: `cpu`, `opencl`, `metal`, `vulkan`, `cuda`, `nn`, `auto`; `num_thread` (4); `cache_file`, `cache_dir`: OpenCL tuning cache |
| `trt` | `plugin_libraries`: paths, loaded before the engine is deserialized; `input_shapes` (within optimization profile 0) |
| `rknn` | `image_input` (true: uint8 inputs); `core_mask`: `RKNN_NPU_CORE_AUTO`, `RKNN_NPU_CORE_0`, `RKNN_NPU_CORE_1`, `RKNN_NPU_CORE_2`, `RKNN_NPU_CORE_0_1`, `RKNN_NPU_CORE_0_1_2`; `custom_op_libraries`: paths |
| `cann` | none |

`input_shapes` maps model input names to shapes, e.g. `{"images": [1, 3, 640, 640]}`, and fixes
dynamic dims once at load. An input without an entry only gets a dynamic leading batch dim, set to 1;
other dynamic dims need an entry. Outputs whose size depends on the data are not supported.

A leading dim above 1, from the model or `input_shapes` (e.g. `{"images": [4, 3, 640, 640]}`), is a
static batch. `ImageModel::process_sync` also takes a vector of any number of images with an empty
or matching vector of source rects: they run in consecutive batches of up to that batch, every image
is preprocessed straight into its own sample of the engine input, the engine runs once per batch and
each sample is postprocessed into one result, in order. Samples past the images of a partial batch
are left as they are and not decoded. With a batch of 1 (every `ncnn` model, RKNN models converted
without a batch) the images simply run one by one. Every output must lead with the same batch.
With a batch above 1, samples of dma-buf inputs are host views, so the RGA writes each into its own
dma-buf whose result is copied in, and the `dvpp` preprocessor is refused. Dynamic batches are not
supported: the batch is fixed at load.

On the RK3588 the three NPU cores serve concurrent calls: give `ImageModelOptions::engine` three or
more engines and keep `core_mask` at `RKNN_NPU_CORE_AUTO`, and the driver runs each context on an
idle core. Through `hrt-validate` with 160 images (YOLOv8n detection, fp16 RKNN),
1, 3 and 6 engines took 42.4, 17.3 and 15.1 ms per image; pinning each engine to one core was
not faster. Each engine loads its own copy of the model.

Operator libraries are optional: one that does not load is skipped with a warning.

Per platform:

| Engine | Preprocess and model inputs |
| --- | --- |
| `rknn` (RK3588) | `image_input: true` (default) feeds uint8 NHWC images and the runtime applies the mean, std and channel swap built into the `.rknn` (`strings model.rknn \| grep "'mean'"` shows them): use `dst_image_format: "inter_rgb"` and keep `mean`/`std` at their defaults. `false` feeds int8 or float32 tensors normalized by `preprocess`. `device: "auto"` uses the RGA. |
| `trt` | `device: "auto"` preprocesses with CUDA into the device input. |
| `cann` | `device: "auto"` preprocesses with DVPP (VPC), which only crops, resizes and pads uint8 images: it has no normalization and no layout conversion, so the `.om` it feeds must do those in AIPP (`atc --insert_op_conf=<aipp.cfg>`, with `mean_chn_*`/`var_reci_chn_*` and `csc_switch`), as for RKNN's `image_input: true`. Use `dst_image_format: "inter_bgr"` or a YUV format, keep `mean`/`std` at their defaults, and give the model a width that is a multiple of 16, as VPC needs a row stride that is a multiple of 16 and at least 32. Engine inputs come from the DVPP memory pool, which VPC requires of the images it reads and writes, while outputs and staging use `aclrtMalloc`. An `.om` without AIPP takes float32, which VPC cannot write, so the CPU operators run the whole step; where VPC cannot run at all, such as a container without the media service, `device: "auto"` picks the CPU preprocessor outright. Only the steps VPC really runs get its even regions. ATC defaults to FP16: add `--precision_mode=must_keep_origin_dtype` to match an FP32 reference. |
| `mnn` | `forward_type: "metal"` needs an SDK built with Metal; on Linux with RGA and a dma-heap `device: "auto"` uses the RGA, as for `ncnn`. |
| `ort` | `execution_provider: "coreml"` needs a CoreML build; models with a dynamic batch need no `input_shapes`, other dynamic dims do. |

## postprocess

C++ callers create a built-in postprocessor with
`create_postprocessor<OcrResult>("Ocr", config, engine)` or
`create_postprocessor<DetectResultMap>("YoloV8Detect", config, engine)` from
`hrt/postprocess/postprocessor_factory.hpp`; the result types are in
`hrt/postprocess/result_types.hpp`. `supports_postprocessor<Result>(type)` checks whether a
built-in type produces `Result` without constructing it. Creation checks the result type,
configuration and engine outputs; invalid requests abort. `check_postprocess_config(type, config)`
returns configuration problems before engine initialization. Every built-in type is registered
once in `source/postprocess/postprocessor_factory.cpp`, which `ImageModel` uses as well.

`confidence_threshold` and `iou_threshold` take one value for all classes or one per class,
each in [0, 1].

| `type` | Keys |
| --- | --- |
| `YoloV8Detect`, `YoloV8Rotated` | `class_num`, `reg_max` (optional; inferred from output channels), thresholds |
| `Yolo26Detect`, `Yolo26Rotated` | `class_num`, thresholds |
| `YoloV5Detect`, `YoloV5Rotated` | `class_num`, `anchors` (one list per head: w0, h0, w1, h1, ...), thresholds |
| `GeneralDetect`, `GeneralRotated` | `class_num`, thresholds; decoded `[1, 4 + classes (+ angle), N]` output |
| `E2EDetect` | `class_num`, `confidence_threshold`; Ultralytics end-to-end `[1, N, 6]` output, no NMS |
| `E2ERotated` | `class_num`, `confidence_threshold`; end-to-end OBB `[1, N, 7]` |
| `DetrDetect` | `class_num`, `confidence_threshold` (0.25), `output_format` (`combined`/`separate`), `score_activation` (`none`/`sigmoid`/`softmax`), `background_class` (-1: none) |
| `GeneralYoloPose` | `class_num`, `keypoint_num` (17), `keypoint_threshold` (0.5), thresholds |
| `YoloV8Keypoint` | `class_num`, `reg_max` (16), `keypoint_num` (17), `keypoint_threshold` (0.5), thresholds |
| `Yolo26Keypoint` | `class_num`, `keypoint_num` (17), `keypoint_threshold` (0.5), thresholds |
| `E2EKeypoint` | `class_num`, `keypoint_num` (17), `keypoint_threshold` (0.5), `confidence_threshold`; end-to-end pose `[1, N, 6 + K x 3]` |
| `RtmoKeypoint` | `confidence_threshold` ([0.15]), `iou_threshold` ([0.65]), `keypoint_threshold` (0.3); one class |
| `GeneralKeypoint` | `general_keypoint_type` (`heatmap`/`heatmap_udp`/`simcc`/`point_with_visible`/`point_channel_first`/`point_channel_last`), `heatmap_refine` (`dark`/`none`), `score_activation` (`none`/`sigmoid`), `keypoint_threshold` (0) |
| `GeneralYoloSegment` | `class_num`, `mask_threshold` (0.5), thresholds |
| `YoloV8Segment` | `class_num`, `reg_max` (optional; inferred from output channels), `mask_threshold` (0.5), thresholds |
| `Yolo26Segment` | `class_num`, `mask_threshold` (0.5), thresholds |
| `E2ESegment` | `class_num`, `mask_threshold` (0.5), `confidence_threshold`; end-to-end `[1, N, 6 + masks]` plus prototypes |
| `SemanticSegment` | none |
| `GeneralClassify` | `classify_postprocess_type`: `none`, `sigmoid`, `softmax` |
| `Embedding` | none; returns the single engine output as a float vector (int8 outputs dequantized), with no normalization; models with more outputs are rejected |
| `Ocr` | DB detection or CTC recognition; parameters and cascade usage are in [OCR model](#ocr-model) |

Raw YOLO postprocessors compute each head's x and y stride during initialization as the input
width over the grid width and the input height over the grid height; strides are not configured.
Segmentation identifies the prototype output first and rejects ambiguous output structures.
YOLOv8 also infers `reg_max` from the box channels divided by four, after subtracting
classes and task-specific channels (one angle, keypoint coordinates, or mask coefficients).
An explicit `reg_max` must be positive and match every head.
`GeneralYoloPose`, `E2EKeypoint`, `Yolo26Keypoint` and `YoloV8Keypoint` do not infer the
keypoint layout: each of the `keypoint_num` (17) keypoints has x, y and visibility, the
Ultralytics `kpt_shape` `[17, 3]`, and `YoloV8Keypoint` uses `reg_max` 16. Initialization
fails unless the output channels equal classes, box channels and `keypoint_num x 3`; models
such as the 5-landmark face template set `keypoint_num: 5`.
Segmentation requires an unambiguous prototype/head structure; explicit `reg_max` can
constrain that match.
Engines report outputs sorted by name on every backend; `anchors`, output indices and the
inferred strides follow that order, excluding the prototype output for segmentation.
For a 640 x 640 input with outputs `output0` (80 x 80), `393` (40 x 40) and `401` (20 x 20),
the inferred strides are `[16, 32, 8]`.

`GeneralYoloPose` reads the decoded single output `[1, 4 + classes + K x 3, N]` of the default
Ultralytics export (boxes, sigmoid scores and keypoints already in input pixels). `YoloV8Keypoint`
reads the raw heads, one `[1, 4 x reg_max + classes + K x 3, h, w]` output per stride, as NPU
exports keep them for quantization: boxes from the distance bins, scores from logits, keypoints at
`((raw * 2 + cell) * stride)` with sigmoid visibility. Both take any `keypoint_num`, e.g. 5 face
landmarks; drawing joins a skeleton only for 17 COCO keypoints.

Raw heads come from the one-to-many branch that the default export decodes, so NMS still runs.
YOLOv8 heads (`YoloV8Detect`, `YoloV8Keypoint`, `YoloV8Rotated`, `YoloV8Segment`) carry 4 x `reg_max`
distance bins, keypoints at `(raw * 2 + cell) * stride` and angles `(sigmoid(raw) - 0.25) * pi`. YOLO26
heads (`Yolo26Detect`, `Yolo26Keypoint`, `Yolo26Rotated`, `Yolo26Segment`) carry the four distances in
cells, keypoints at `(raw + cell + 0.5) * stride` and angles in radians. Segment heads add the mask
coefficients after the classes, and the prototypes `[1, masks, mh, mw]` (`[1, mh, mw, masks]` for NHWC
outputs, like the heads) are a further output in any position.

`DetrDetect` reads batch-1 normalized center-x, center-y, width and height boxes.
It selects the highest-scoring foreground class of each query and does not apply
NMS. The model's export contract, rather than its family name, determines the config:

- `combined` (default): one `[1, N, 4 + scores]` output, boxes first. Ultralytics
  RT-DETR exports with per-class probabilities use `score_activation: "none"`.
- `separate`: output 0 is `[1, N, 4]` boxes and output 1 is `[1, N, scores]` scores,
  in the engine's reported output order. Custom output indices are not supported.
- `sigmoid` activates independent class logits. `softmax` normalizes all score
  channels, including the background channel. Set `background_class` to its index
  (commonly `class_num` in original DETR exports); foreground IDs close that gap.

`class_num` counts foreground score channels, including unused category slots,
and excludes the background channel. For example, a DETR export with two
outputs and a final background class uses `{"output_format":"separate",
"score_activation":"softmax","background_class":80,"class_num":80}`.
Use the model's own preprocessing; Ultralytics RT-DETR needs direct resize
(`normal`), RGB and division by 255. Auxiliary decoder outputs, pixel/xyxy boxes,
embedded class selection and global multi-label top-k require a matching export
or a different decoder; the family name alone does not guarantee compatibility.

Ultralytics end-to-end exports have one row per kept object, already deduplicated, so no NMS runs
and `iou_threshold` is not accepted; padding rows score 0. Ultralytics 8.4 writes them with
`export(format="onnx", nms=False)` for one-to-one heads (YOLOv10, YOLO26) and `nms=True` for any
generation; without `nms` it exports the raw `[1, 4 + classes + extra, anchors]` heads, which
`GeneralDetect`, `GeneralYoloPose`, `GeneralYoloSegment` and `GeneralRotated` decode for YOLO26 as for
YOLOv8. Columns, in network-input pixels:

| Type | Row |
| --- | --- |
| `E2EDetect` | x1, y1, x2, y2, score, class id |
| `E2EKeypoint` | x1, y1, x2, y2, score, class id, then x, y and visibility per keypoint |
| `E2ESegment` | x1, y1, x2, y2, score, class id, mask coefficients; a second `[1, masks, mh, mw]` prototype output, in either order |
| `E2ERotated` | center x, center y, width, height, score, class id, angle in radians |

`RtmoKeypoint` decodes one-stage RTMO pose models with fixed outputs, boxes `[1, M, 5]` (x1, y1, x2,
y2, score) and keypoints `[1, M, K, 3]` (x, y, visibility) in network-input pixels, in either output
order. The mmpose export selects instances with NMS inside the model, so its output size depends on
the image; `prepare_other_cases.py rtmo` in the local assets/test/scripts/ directory (needs `onnx` and `numpy`) replaces that NMS
with the M best pre-NMS candidates (`--candidates`, 300), and `RtmoKeypoint` thresholds and
suppresses them. RTMO takes BGR, keep-ratio top-left resize, padding 114 and no normalization.

`GeneralKeypoint` decodes one top-down crop per call; cascades pass detections as `src_rect`
with `resize_type: "pad_outside"` and `pad_scale` (1.25 in mmpose). `general_keypoint_type`
follows the outputs:

- `heatmap`: `[1, K, H, W]` maps (HRNet, SimpleBaseline, ViTPose); peak cell times
  input / heatmap size, score the peak. `dark` refines as mmpose DARK (11 x 11 blur,
  sigma 2) at interior maxima of positive heatmaps and keeps the argmax otherwise.
- `heatmap_udp`: the same maps from UDP-trained models (HRNet-UDP); peak cell times
  input / (heatmap size - 1), as mmpose `UDPHeatmap`. `dark` refines as mmpose
  `refine_keypoints_dark_udp`. Its `combined` offset maps are not supported, and the crop is the
  `pad_outside` resize rather than UDP's unbiased affine, a difference below one input pixel.
- `simcc`: x `[1, K, W * r]` and y `[1, K, H * r]` distributions (RTMPose); bin / r,
  score min(max x, max y).
- `point_with_visible`: x, y and `[1, K, 3]` visibility distributions; the argmax state
  sets `visible`, its last channel the score.
- `point_channel_first`, `point_channel_last`: normalized `[1, 2, K]` or `[1, K, 2]` points.

For `heatmap` and `simcc`, `score_activation` applies to the score and a keypoint is visible
when the score exceeds `keypoint_threshold` (mmpose drops scores <= 0 and draws above 0.3).

## results.json

`results.json` is the output layout of `hrt-validate`, the same for every task. `engine` is
the `-e` value or `null`. Each image has `file`, `width`, `height`, `error` (`null` on success; then `results` is empty)
and `results`, whose items always carry every field, empty when the task has none:

| Field | Value |
| --- | --- |
| `class_id`, `score` | Detection class and score; classify: top 1 of output 0. Semantic segments and `GeneralKeypoint` use score 1; embeddings -1 and 0. |
| `x`, `y`, `width`, `height` | Box in source pixels, the top-left of the unrotated box; the full image for classify, embeddings and semantic segments; the cropped source region for `GeneralKeypoint`. |
| `angle` | Rotated boxes; 0 otherwise. |
| `keypoints` | `[x, y, score, visible]` per keypoint, `visible` 0 unlabeled, 1 invisible, 2 visible. |
| `mask` | `{"rect": [x, y, w, h], "rle": [...]}`: row-major run lengths over `rect`, alternating 0 and 1, starting with 0; `null` without a mask. |
| `cls` | Classify, one item per engine output: `output` index, `topk` `[[class, score], ...]` (5), `scores` by class. |
| `embedding` | Embedding vector at 6 decimals. `-r` draws no images for embeddings. |

## ocr_results.json

`ocr_results.json` is the output layout of `hrt-ocr`, which runs `OcrModel` on every image of a
directory, one image at a time. The top level names the `detection` and `recognition` algos
(`-d`, `-r`, default `ppocrv6_tiny_det` and `ppocrv6_tiny_rec`), `task` `"ocr"` and `engine` (the
`-e` value applied to both algos, or `null`). Images have the same `file`, `width`, `height`, `error`
and `results` as `results.json`, the output defaults to `<input>_ocr.json` beside the input
directory, and a failed image makes the exit code 1. Each result is one text line:

| Field | Value |
| --- | --- |
| `text` | Recognized UTF-8 text. |
| `x`, `y`, `width`, `height` | Axis-aligned text box in source pixels. |
| `detection_score`, `recognition_score` | Text region score of the detector and mean character score of the recognizer. |

```sh
hrt-ocr -m ocr.hrtc -r ppocrv6_tiny_rec -e ort -i assets/test/images -o ocr_results.json
```

## OCR model

`model::OcrModel` combines one detection algo and one recognition algo from a loaded
`hrtc::Package`. Load `.json` with `hrtc::load_json` or `.hrtc` with `hrtc::load`, then call
`OcrModel::create(package, "ppocrv6_tiny_det", "ppocrv6_tiny_rec")`. Each algo keeps its own backend
alternatives, but both stages must run on one device: CPU engines (ONNX Runtime CPU, MNN, ncnn, RKNN), CUDA
(TensorRT) or Ascend. Recognition is created on the detection context and only its engines
of that device are tried; without one, `create` aborts. No package schema or existing
image-model result type changes.

Each stage is an `ImageModel<postprocess::OcrResult>` with default options (one engine).
`process_sync(image)` returns an optional vector of `OcrText`: an axis-aligned `rect`
(`core::Rect2f` in original image pixels), UTF-8 `text`, `detection_score`, and `recognition_score`.
The C++ and C models require serialized access; create separate instances for concurrency.
Empty detection is a successful empty vector; a failed stage returns no result, not partial text.
Standalone recognition and detection postprocessors use the same vector result, leaving
fields unrelated to that stage empty/zero. Text regions follow PP-OCR `sorted_boxes`: sorted
by top then left, after which boxes whose tops differ by less than 10 pixels are ordered by
left edge. This is not full document layout analysis.

`postprocess.type: "Ocr"` has the following parameters. `stage` is required, and keys from
the other stage or unknown keys are rejected.

| Stage | Parameter | Default | Meaning |
| --- | --- | --- | --- |
| Both | `stage` | required | `detection` or `recognition` |
| Detection | `decoder` | `db` | Only DB box mode (PP-OCR `fast` scoring) is supported |
| Detection | `threshold` | 0.2 | Pixel probability threshold, [0, 1] |
| Detection | `box_threshold` | 0.4 | Minimum mean probability inside the enclosing rectangle, [0, 1] |
| Detection | `unclip_ratio` | 1.4 | Offset distance is rectangle area times ratio divided by perimeter, [0, 10] |
| Detection | `min_size` | 3 | Minimum short side in output-map pixels before expansion; after expansion requires min_size + 2, (0, 1000] |
| Detection | `max_candidates` | 3000 | Maximum contours examined, [1, 10000] |
| Recognition | `decoder` | `ctc` | Greedy CTC over probabilities `[1, T, C]`; blank is class zero |
| Recognition | `dictionary` | required | UTF-8 file path relative to the working directory, or `string:` followed by newline-separated entries |
| Recognition | `append_space` | true | Append a space entry after dictionary entries |

The JSON loader resolves `dictionary` into inline `string:` text before packing; `.hrtc`
needs no external dictionary. LF/CRLF are supported, trailing newline is allowed, interior
empty entries are invalid. The output class count must equal dictionary entries + blank +
optional space. Tiny and small need their respective dictionaries. CTC removes blank,
collapses consecutive identical classes, and averages retained character probabilities;
an all-blank sequence returns empty text and score zero. Inputs must be probabilities;
the maximum of each step must lie in [0, 1], so logits are rejected and need conversion first.

DB takes one `[1, 1, H, W]` probability map. It scores minimum-area contour rectangles,
expands them analytically, maps and clips their corners to the source image, and reports
the axis-aligned bounds of those corners. The analytic expansion avoids a polygon-offset
dependency; integer polygon-offset rounding in PaddleOCR can shift reference boundaries
slightly. As in PP-OCR, boxes whose truncated top or left side is at most 3 pixels are
discarded. Rotated or curved text output, dilation and slow contour scoring are not
supported; tilted lines are recognized from their axis-aligned bounds.

The fixed 640 x 640 detector uses BGR, `keep_ratio` (aspect-preserving, top-left, padded)
and ImageNet normalization, with pixel-space mean/std from the template. PP-OCR resizes
dynamically to multiples of 32; the fixed-shape model keeps the aspect ratio instead. Recognition uses `text_line`, planar BGR,
mean/std 127.5 and `normalize_type: "roi"`: padding is filled with zero and remains zero
after normalization. Long lines exceeding the 320-pixel width are compressed to fit;
there is no automatic splitting. Existing resize modes retain their previous behavior.

`OcrModel` truncates each detected box to whole pixels like PP-OCR; boxes must lie inside
the image and be at least 2 x 2 pixels. Lines are recognized from that region of the source
on the stages' device, without copying the image, when the source is in that device's memory
(host or dma-buf for CPU engines, CUDA, DVPP) or is host uint8 BGR/RGB, of which only the line
region is staged; other sources are converted to host BGR once per call. Boxes with height >= 1.5
times width are rotated counterclockwise like PP-OCR; preprocessors cannot rotate, so for those
the source is used directly when it is host uint8 BGR/RGB, otherwise converted once per call to
host BGR, and the region is rotated into reused host storage. No direction classifier,
page deskew, curved-text rectification or automatic 180-degree orientation is included.

Run from the repository root, with the local models, dictionaries and image available:

```sh
build/release/examples/hybridrt_example_ocr_model -m assets/template/hrtc_template.json \
  -i assets/test/images/ocr_document.png -p coreml -o assets/test/ocr_coreml_tiny.json
```

Use `-p cpu` for CPU, or omit `-p` to keep
the package's engine configs. The JSON example loads only the two selected algos so absent
unrelated template models do not prevent execution. JSON output contains text, `rect` as `[x, y, width, height]` and
both scores. The C API uses separate `HrtOcrModel`/`HrtOcrResult` handles and `hrt_ocr_*`
functions, leaving existing structs/enums unchanged. Result text views remain valid until
the next process call or destruction; a failed call clears the previous result.

CoreML is ONNX Runtime's execution provider, configured in the engine's `config` with
`execution_provider: "coreml"`, `provider_options: {"ModelFormat": "MLProgram",
"RequireStaticInputShapes": "1"}`. Unsupported operators may execute on the ORT CPU
provider; selecting CoreML does not guarantee every operator uses the GPU/ANE.
The faster `ModelFormat` depends on the model: the PP-OCR models above run several times
slower with `NeuralNetwork`, while a 640x640 YOLOv8 detector on an Apple M4 took 11.5 ms with
`NeuralNetwork` and 26.4 ms with `MLProgram` (GPU only; `MLComputeUnits` `ALL` and
`CPUAndGPU` timed the same), with boxes matching at a mean IoU of 0.99. The YOLOv8 CoreML
backend in `hrtc_device_config.json` therefore uses `NeuralNetwork`.

## Test models

Template model files live in `assets/test/onnx/`, named after their postprocessor.
All external input/output dimensions are positive, fixed integers; batch is 1 except in the
static batch-4 and batch-8 fixtures below. `Ocr` has
separate `OcrDetection.onnx` and `OcrRecognition.onnx` (tiny) files because detection and
recognition need different networks. `Embedding.onnx` is a
copy of the classifier and returns its 1000 class scores as an example vector.
Models, source weights, scripts and validation results stay local and untracked.

The collection covers all 25 postprocessor types, with static batch variants for selected models.
`YoloV5Rotated` uses the user-provided `v0013_obbczs_detection_0808int8.onnx`.
Its user-supplied configuration uses confidence threshold 0.3, IoU threshold 0.1 and
these stride-8/16/32 anchor groups (width, height pairs):
`[49.25, 48.78125, 308.75, 21.09375, 94.1875, 94.8125]`,
`[474.75, 31.90625, 145.25, 141.875, 578.0, 59.40625]`,
`[218.75, 222.75, 533.0, 527.5, 1000.0, 514.5]`. The template records these values.
Normalization and the angle convention still require an accuracy reference. The ONNX graph already contains quantization parameters;
RKNN must preserve them with `do_quantization=False`, rather than quantize it again.

The collection was regenerated on 2026-10-21 on macOS arm64 (Apple M5 Pro) with Python 3.10,
Ultralytics 8.4.183, PyTorch 2.14.1, torchvision 0.29.1, ONNX 1.23.2 and ONNX Runtime 1.23.2;
the hashes above are those files. All 24 image-task algos (`hrt-validate`) and both OCR
cascades (`hrt-ocr`) ran on the five local test images with HybridRT and ONNX Runtime 1.31.0
CPU, also under AddressSanitizer and macOS `leaks` with no errors or leaks. Raw heads give the
same objects as their decoded counterparts from the same weights (`yolov8n_segment_raw`
against `yolov8n_segment`: boxes and scores within 1e-4, 7 of 8 masks identical). These
checks establish runnable pipelines, not model accuracy or other-backend compatibility.

`hybridrt_bench_postprocess` measures postprocessing on real model outputs, excluding
preprocessing and inference. `BM_Type/<index>` covers every template algo in order, with its
actual input size, including all postprocessor types and OCR variants.
Single-thread timing results and reproduction commands are in [benchmark.md](../benchmark.md).

| Algo | ONNX file in `assets/test/onnx/` | Input NCHW | SHA256 |
| --- | --- | --- | --- |
| `yolov8_detect` | `YoloV8Detect.onnx` | `1 x 3 x 640 x 640` | `e2a57b77d38a136a3ac07ddd17b81930c4680e0d464ee615488c3a47bfdef303` |
| `yolo26n_detect_raw` | `Yolo26Detect.onnx` | `1 x 3 x 640 x 640` | `2089cfdd67660be0efec7477bbf2f6cabc2261c84bbd82c6a7e895b51bd8c49a` |
| `yolov5_detect` | `YoloV5Detect.onnx` | `1 x 3 x 640 x 640` | `797416533eed5eb4c82098ec221bbe55368ad3c062f9d0217db87f129a78f172` |
| `rtdetr_l_detect` | `DetrDetect.onnx` | `1 x 3 x 640 x 640` | `599b9b7484cb831312f585d4ec4885a95ae10d1e84dada14833f5c570828c9b9` |
| `yolo26n_e2e_detect` | `E2EDetect.onnx` | `1 x 3 x 640 x 640` | `08727943f090f5e8472cae3c08bcbb7b93c927820d3104ef7792f1d31277ec77` |
| `yolov8n_detect` | `GeneralDetect.onnx` | `1 x 3 x 640 x 640` | `782006124ca30bc45ac1da40466b2541b06b203754fac8b5018fbbfbb602d36f` |
| `yolo26n_e2e_obb` | `E2ERotated.onnx` | `1 x 3 x 1024 x 1024` | `0dc04234b1d82d87b8aebe5c364b4c298de0e2380ce61b74f9b117478e690e41` |
| `yolov8n_obb_raw` | `YoloV8Rotated.onnx` | `1 x 3 x 1024 x 1024` | `2a5de440d72ecb80cc4fbce50c59f6293e4bc89cb1fe8aaee8ca7d9ed3b89566` |
| `yolo26n_obb_raw` | `Yolo26Rotated.onnx` | `1 x 3 x 1024 x 1024` | `003dc108ae1be9b94f7bbf3f154d871385224da96ca15fa0bc07bccb0bc2148f` |
| `yolov5_rotated` | `YoloV5Rotated.onnx` | `1 x 3 x 1280 x 1280` | `afffc7a2813e15e9137648bc7655155827b1cdace2a0999b1919d1196be57dfb` |
| `yolov8n_obb` | `GeneralRotated.onnx` | `1 x 3 x 480 x 640` | `d87f86b2e475deccfd57410fdcb6d2de3886fcab4f6395888ddca4c35a2c12ee` |
| `yolov8n_classify` | `GeneralClassify.onnx` | `1 x 3 x 640 x 640` | `46f101117e36acb79ebbd362250971b13d8dd906c7c877a2b7cc6af1fc1b8054` |
| `yolov8n_classify_embedding` | `Embedding.onnx` | `1 x 3 x 640 x 640` | `46f101117e36acb79ebbd362250971b13d8dd906c7c877a2b7cc6af1fc1b8054` |
| `yolov8n_pose` | `GeneralYoloPose.onnx` | `1 x 3 x 640 x 640` | `b1684b889751d66d6e3f1a5f9e5a799c41e325fabef0cb552cbac82b385dbcba` |
| `yolov8n_face` | `YoloV8Keypoint.onnx` | `1 x 3 x 640 x 640` | `af09683be4937ebe23e9da0346cbf46cd937b73ed5fac986e6205e6c7cdd2c25` |
| `yolo26n_pose_raw` | `Yolo26Keypoint.onnx` | `1 x 3 x 640 x 640` | `513e12cd94482f799e2d1ad66fcb9603d7c3eb41ad38863e7ae566374e0eddc8` |
| `yolo26n_e2e_pose` | `E2EKeypoint.onnx` | `1 x 3 x 640 x 640` | `ecd85eb31cf6889ab62600164d27698b881fcb27bd27c548b2dd4a67fb52c814` |
| `rtmo_s_pose` | `RtmoKeypoint.onnx` | `1 x 3 x 640 x 640` | `91d97d8251f7624d4da9cead6b2a8bd6ea36850cfe329122eebeb89b41f108d2` |
| `rtmpose_m_pose` | `GeneralKeypoint.onnx` | `1 x 3 x 256 x 192` | `81e1025f9c938450d2ef88c0fba274fb57b0f9b49711dc7a52fef99c7ea3548d` |
| `yolov8n_segment` | `GeneralYoloSegment.onnx` | `1 x 3 x 640 x 640` | `a9c25c390e47e6a5e7b3c1dc22bee67ef28cd1746576f2c80a141717fd30dbc6` |
| `yolov8n_segment_raw` | `YoloV8Segment.onnx` | `1 x 3 x 480 x 640` | `fdbd46dc0a5ae4bb1cb1b32010fa131a43f370043077555f921890bcd2880494` |
| `yolo26n_segment_raw` | `Yolo26Segment.onnx` | `1 x 3 x 640 x 640` | `85e5ec69cb0a06f82e6a27b4cdb6be36d91e041b3829d75887183f6506b18221` |
| `yolo26n_e2e_segment` | `E2ESegment.onnx` | `1 x 3 x 640 x 640` | `101060498337592b2c671f479001ca725d92049f99118e5b666f4c2273d9f096` |
| `yolo26n_semantic` | `SemanticSegment.onnx` | `1 x 3 x 640 x 640` | `0297ea52c1e6da5047657153853e61136de4bfec675795c6bfb7ea9e1b071f28` |

Static batch-4 and batch-8 fixtures are exported directly from official [Ultralytics weights](https://github.com/ultralytics/assets/releases/tag/v8.3.0),
using local `assets/test/scripts/export_static_batch.py` (arguments: the batches to export):
Ultralytics 8.4.8, PyTorch 2.9.1, ONNX 1.20.1, opset 17, `batch=4` or `batch=8`,
`imgsz=[640,640]`, `dynamic=False`, `simplify=False`, FP32. Batch-1 reference models are
exported from the same weights. These are native batched networks. Their template algos end in
`_batch4` and `_batch8`; `hybridrt_bench_pipeline` `BM_Batch` runs batch 1, 4 and 8 with as many
images per call.
`ImageModelTest.MatchesBatchOneOnStaticBatchModels` compares four images and a partial batch
against individual CPU calls. `MatchesBatchOneOnTensorRtStaticBatchModels` compares batched
and individual calls on the same TensorRT model with CPU and CUDA preprocessing separately;
the CUDA kernel may differ from OpenCV by one uint8 step in rare pixels. TensorRT fixtures are local
`.engine` files in `assets/test/trt/`, named after the ONNX files of the batch-1, batch-4 and batch-8 models, built with
TensorRT 10.14.1, FP32, `--noTF32`; plans load only where they were built, so tests and
benchmarks skip them elsewhere.

| Algo | ONNX file in `assets/test/onnx/` | Input NCHW | Outputs | SHA256 |
| --- | --- | --- | --- | --- |
| `yolov8n_classify_batch4` | `GeneralClassifyBatch4.onnx` | `4 x 3 x 640 x 640` | `4 x 1000` | `bb5992374ba65a0555d0bd30679a3610a5c7305f95e305542efd747e51c26b4e` |
| `yolov8n_detect_batch4` | `GeneralDetectBatch4.onnx` | `4 x 3 x 640 x 640` | `4 x 84 x 8400` | `deb8c98ab42828ccc537eaad85838f1c5349a37fc20a256dbca23eb8cc477518` |
| `yolov8n_pose_batch4` | `GeneralYoloPoseBatch4.onnx` | `4 x 3 x 640 x 640` | `4 x 56 x 8400` | `4197a3fbe3181fb9deeea665cb3898c608500437cc85a4f3d69660e45debb194` |
| `yolov8n_segment_batch4` | `GeneralYoloSegmentBatch4.onnx` | `4 x 3 x 640 x 640` | `4 x 116 x 8400; 4 x 32 x 160 x 160` | `e13654d03168abf180dd73cfc3b28daf3b03dffef2a12f37ee4bf2d8cf4a9462` |
| `yolov8n_classify_batch8` | `GeneralClassifyBatch8.onnx` | `8 x 3 x 640 x 640` | `8 x 1000` | `f6bdb116cff2dbb19f8d23d3ec04a46dde18bf07d9ac02aaac6d9f05e843e4ce` |
| `yolov8n_detect_batch8` | `GeneralDetectBatch8.onnx` | `8 x 3 x 640 x 640` | `8 x 84 x 8400` | `dc8f6cd8c46ccb2021f30663f97e96099d8675630fba64102e2f5f09393cfaee` |
| `yolov8n_pose_batch8` | `GeneralYoloPoseBatch8.onnx` | `8 x 3 x 640 x 640` | `8 x 56 x 8400` | `58bd37d656e7a3e3139d71a6408baf5795a09ca21843cdc2dee75f99f828235d` |
| `yolov8n_segment_batch8` | `GeneralYoloSegmentBatch8.onnx` | `8 x 3 x 640 x 640` | `8 x 116 x 8400; 8 x 32 x 160 x 160` | `c0c2d436f510efdc26912b95311c0503b2f0bc2b7ebce23c6c151abf3a8e4ee1` |

Additional OCR fixtures use official PaddlePaddle ONNX weights. The `ppocrv6_tiny_det`,
`ppocrv6_tiny_rec` and `ppocrv6_tiny_rec_batch8` template algos run through
`OcrModel`, which recognizes all lines of an image in one `ImageModel` call: the batch-8
recognizer reads eight lines per inference and gives the texts of the batch-1 one.

| ONNX file in `assets/test/onnx/` | Input NCHW | Output | SHA256 |
| --- | --- | --- | --- |
| `OcrDetection.onnx` | `1 x 3 x 640 x 640` | `1 x 1 x 640 x 640` | `7373329bed2ea0b66c173278112a6dab68a5f9f244723a302e4b1a674f9435c9` |
| `OcrRecognition.onnx` | `1 x 3 x 48 x 320` | `1 x 40 x 6906` | `71b480e58fa36017c5e542b1157f659cc58da1233d64c716b7a26240ed2c1f31` |
| `OcrRecognitionBatch8.onnx` | `8 x 3 x 48 x 320` | `8 x 40 x 6906` | `5d3f22468b3cacf072a0c3bd757d541901b4d419203100ee49a346b3745880ca` |

Sources: [tiny detection](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx),
[tiny recognition](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx).
Original `inference.onnx` and `inference.yml` files live under
`assets/test/weights/ocr/<model>/`; recognition `characters.txt` files contain the original
dictionary entries, without the CTC blank or appended space.
Local `assets/test/scripts/fix_ocr_shapes.py` fixes batch and image dimensions, sets output
dimensions from real inference and runs ONNX shape inference; the files remain FP32 at their
original opsets (14 for detection, 11 for recognition). It also writes the `characters.txt`
files from `inference.yml` and `OcrRecognitionBatch8.onnx` from the tiny recognizer with a batch
of 8. Rerun on 2026-10-24 (ONNX 1.16.1, ONNX Runtime 1.23.2) from originals with SHA256
`193bab7a...` (tiny det) and `9ef676d6...` (tiny rec), it reproduced the
two batch-1 files byte for byte.

ONNX 1.17.0 checker and ONNX Runtime 1.31.0 CPU zero/random-input runs passed on macOS arm64.
All external input/output dimensions are positive integers. Original and fixed model outputs
agree within `rtol=1e-4, atol=1e-4`; maximum absolute errors were 2.09e-7 (det),
2.74e-5 (tiny rec). Recognition argmax sequences are identical.
Fixed shapes can enable different fused runtime kernels. These checks establish runnable,
fixed-shape models, not OCR accuracy or compatibility with other backends.
Local `assets/test/ocr_shape_validation.json` records shapes, source hashes and comparison results.

`assets/test/images/ocr_document.png` (also read as `ocr_document.jpg`; JPEG data, SHA256
`4425af33dd163cf73bdff502bd35ee527e9bdd5725501db1da78bfdae9f538f4`) is PaddleX's boarding pass demo
image `general_ocr_002.png` from paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/.

End-to-end OCR was compared with PaddleOCR 3.7.0 (PaddlePaddle 3.3.1, CPU, macOS arm64) on
the local `assets/test/images/ocr_document.png` (896 x 528), with document orientation,
unwarping and textline orientation disabled. With PaddleOCR detection limited to a 640 long
side, matched boxes reach IoU >= 0.9; 28 of 34 reference lines match exactly with
`ppocrv6_tiny_rec`. Most remaining differences are
spaces or single characters from integer axis-aligned crops versus PaddleOCR's rotated
sub-pixel crops. PaddleOCR widens recognition input beyond 320 for long lines; the fixed
48 x 320 recognizers compress the 737 x 39 footer line and misread it. A 48 x 960 export
fixes that line but loses spaces on short lines, so 320 is kept. At PaddleOCR's default
detection resolution (native size, multiple of 32), 24 of 33 lines match with tiny.

Sources and export steps for the current collection:

- `GeneralDetect`, `GeneralYoloPose`, `GeneralRotated` and `GeneralClassify` are the
  decoded binding exports of the official YOLOv8 weights below (640 x 640, 640 x 640, 1024 and
  640 x 640 inputs). `GeneralYoloSegment` is the default `yolov8n-seg.pt` export at 640 x 640.
  These four batch-1 models come from `export_static_batch.py`, with the static batch fixtures.
  Raw `YoloV8Detect`, `YoloV8Rotated` and `YoloV8Segment` come from `yolov8n.pt`,
  `yolov8n-obb.pt` and `yolov8n-seg.pt` through `yolo_export_raw.py` at 480 x 640: one
  output per stride holding the head convolutions before decoding, concatenated as box
  distances, the angle for OBB, class logits, then mask coefficients plus the prototype
  output for segmentation. `DetrDetect` exports `rtdetr-l.pt` with Ultralytics 8.4.183 at
  640, opset 17, with the decoder's TopK replaced so that it returns the normalized query
  boxes and sigmoid class scores `[1, 300, 84]`.
- Official [YOLO26 nano weights](https://github.com/ultralytics/assets/releases/tag/v8.4.0)
  are exported with Ultralytics 8.4.183 and PyTorch 2.14.1, CPU, FP32, opset 17,
  batch 1, 640 x 640, `dynamic=False`, `simplify=False`, `nms=False`.
  End-to-end exports retain 300 rows. Raw exports expose the one-to-many branch,
  with box distances, class logits and task-specific channels; rotated heads place
  the angle logit before class logits, and segmentation includes mask prototypes.
- `YoloV5Detect` exposes the three official YOLOv5n v7.0 raw convolution heads,
  promoting the FP16 graph to FP32. `YoloV8Keypoint` uses the clibdev face export below.
  `Embedding.onnx` is a copy of `GeneralClassify.onnx`.
- RTMO and RTMPose use the official OpenMMLab SDK downloads linked below.
  RTMO replaces selection after NMS with the 300 highest-scoring candidates before
  NMS, keeping the original box/keypoint decoding. HybridRT performs NMS on the host.
  RTMPose fixes batch 1 and its 256 x 192 input; both outputs have 17 keypoints.
- `SemanticSegment` uses the official Ultralytics `yolo26n-sem.pt` export at 640 x 640,
  opset 17. Local `assets/test/scripts/export_yolo26_semantic.py` cuts the in-graph Resize
  and ArgMax and exposes 19-channel stride-8 logits (`B x 19 x 80 x 80`) for host decoding.
  Batch 1/4/8 are static exports; RKNN mean/std are 0/255 with RGB letterbox preprocessing.
  Weights `yolo26n-sem.pt` (Cityscapes, 19 classes) SHA256
  `f3f293cca764de1f93044030d8d5612de9c5ffbf37c9c8ea1b69418b73038999`, exported with
  Ultralytics 8.4.187 and PyTorch 2.14.1 CPU. The cut logits match the full graph's argmax at
  99.88% of cell centers. On bus.jpg, HybridRT ORT masks agree with Ultralytics `semantic_mask`
  (which resizes the logits bilinearly before its argmax) on 96.9% of pixels; IoU is 0.968 road,
  0.964 building, 0.898 person and 0.957 bus, and lower for small classes at stride-8 resolution.
- Local `yolo_export_raw.py` and `prepare_other_cases.py` under `assets/test/scripts/`
  reproduce the raw exports and the YOLOv5, RTMO and RTMPose graph preparation.

Historical exports and accuracy comparisons (their versions and hashes differ from
the current collection unless explicitly matched above):

- The previous binding fixtures used official [YOLOv8 weights](https://github.com/ultralytics/assets/releases/tag/v8.3.0)
  (`yolov8n.pt`, `yolov8n-pose.pt`, `yolov8n-obb.pt`, `yolov8n-cls.pt`) and
  [YOLO26 weights](https://github.com/ultralytics/assets/releases/tag/v8.4.0) (`yolo26n-seg.pt`).
  Local `assets/test/scripts/export_binding_models.py` exports with Ultralytics 8.4.182,
  PyTorch 2.7.0, ONNX opset 17, fixed batch 1, FP32, `simplify=False`, `nms=False`, CPU,
  and the input sizes listed above. Preprocessing and postprocessing use the template algos.
  Images come from official Ultralytics assets: [bus.jpg](https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg),
  [airport.jpg](https://raw.githubusercontent.com/ultralytics/assets/main/im/airport.jpg) and
  [zidane.jpg](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg).
- Local validation on macOS arm64 (Apple M5 Pro), ONNX Runtime 1.31.0 CPU, Python 3.11.16:
  all five exports passed ONNX checker; Python passed 11 tests without warnings; Rust passed
  8 tests plus the documentation example, including ABI layouts, threads and all five tasks.
  The reference script `assets/test/scripts/compare_binding_models.py` compares the same resized
  pixels against Ultralytics ONNX inference (classification against ONNX Runtime directly).
  Original-resolution geometry is also exercised by the binding tests. OpenCV 4.10 (C++) and
  5.0 (Python) resizing can produce different predictions, so numerical comparisons share pixels.
  This is a fixture and binding check, not a model accuracy evaluation.
  The matched outputs contained 5 detections, 4 poses (17 keypoints each), 4 instance masks
  and 11 rotated boxes. Maximum box/keypoint error was 0.000123 pixels, minimum mask IoU
  0.9982, minimum rotated-box IoU 0.999999; classification top-1 was 654 with maximum
  score error below 0.000001. Full metrics are local in `assets/test/binding_reference_results.json`.

- Previous decoded YOLOv8 detection, segmentation, pose, OBB and classification models came from
  [unileon-robotics/YOLO8-ONNX](https://huggingface.co/unileon-robotics/YOLO8-ONNX).
Conversion scripts are local, untracked files in assets/test/scripts/ and are not distributed with the repository.

- Raw YOLOv8/YOLO26 models use `yolo_export_raw.py` with Ultralytics 8.4.183 and the
  corresponding official `.pt` weights. YOLO26 end-to-end models use `export(format="onnx",
  imgsz=640, opset=17, simplify=True, nms=False)`, with 1024 inputs for OBB.
- `YoloV5Detect` uses the
  [official YOLOv5n v7.0 ONNX](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx)
  through `prepare_other_cases.py yolov5`: three raw convolution heads, promoting FP16 values to FP32.
- `YoloV5Rotated` is the user-provided `v0013_obbczs_detection_0808int8.onnx`, copied
  unchanged (its current SHA256 differs from the copy recorded before 2026-10-21). Its graph uses INT8 Q/DQ internally and float32 I/O, with outputs
  `[1,24,160,160]`, `[1,24,80,80]`, `[1,24,40,40]`.
- `YoloV8Keypoint` uses `yolov8n-face-derronqi.onnx` from
  [clibdev/yolov8-face 1.0.0](https://github.com/clibdev/yolov8-face/releases/tag/1.0.0).
- `DetrDetect` exports [rtdetr-l.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/rtdetr-l.pt)
  as described above.
- `GeneralKeypoint` uses `end2end.onnx` from the
  [RTMPose-m body7 SDK](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip).
- `RtmoKeypoint` uses `end2end.onnx` from the
  [RTMO-s body7 SDK](https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip),
  then `prepare_other_cases.py rtmo` with 300 candidates.
- `SemanticSegment` exports official
  [yolo26n-sem.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-sem.pt)
  through `export_yolo26_semantic.py`, as described above.
- `prepare_other_cases.py` fixes the RTMO and RTMPose batch to 1 and records the output
  shapes of a zero-input ONNX Runtime run as static output shapes. For RTMO it replaces the
  in-graph NMS with the first 300 candidates of its score TopK and raises its keep-top-k to
  300.

Recreate transformed files from downloaded source exports, outside the model collection:

```sh
python assets/test/scripts/yolo_export_raw.py yolov8n.pt assets/test/onnx/YoloV8Detect.onnx --height 480 --width 640
python assets/test/scripts/yolo_export_raw.py yolo26n-seg.pt assets/test/onnx/Yolo26Segment.onnx
python assets/test/scripts/prepare_other_cases.py yolov5 yolov5n.onnx assets/test/onnx/YoloV5Detect.onnx
python assets/test/scripts/prepare_other_cases.py rtmo rtmo-end2end.onnx assets/test/onnx/RtmoKeypoint.onnx --candidates 300
python assets/test/scripts/prepare_other_cases.py rtmpose rtmpose-end2end.onnx assets/test/onnx/GeneralKeypoint.onnx
```

Run any template algo directly, for example:

```sh
hrt-validate -m assets/template/hrtc_template.json -n yolov8n_detect -e ort -i assets/test/images -o results.json
```

`yolov8n_classify` resizes directly instead of the square center crop, so it is a pipeline
check, not an accuracy reference. `rtmpose_m_pose` runs on
one person per call: pass each `yolov8n_detect` class 0 box as `src_rect`.

Results checked against independent references on bus.jpg (ONNX Runtime 1.31.0, x86_64):

- `yolo26n_e2e_detect` and its YOLOv10/YOLOv8 variants: 4 people and the bus, boxes and scores
  identical to the ONNX output decoded in Python with the same letterbox. HybridRT rounds the
  scaled size down, Ultralytics to nearest, so a side ending in .5 gets one row less and low
  scores can differ.
- `yolo26n_e2e_pose`, `_segment`, `_obb` and their YOLOv8 `nms=True` variants against Ultralytics
  `predict` on the same ONNX files: identical boxes, scores, classes, keypoints and angles on bus.jpg
  and airport.jpg; mask IoU 0.966 to 0.997 against `retina_masks=True`. On airport.jpg the YOLO26
  segment export keeps one more box at score 0.253 because of the rounding above.
- Default YOLO26 pose, segment and OBB exports (no `nms`) through `GeneralYoloPose`,
  `GeneralYoloSegment` and `GeneralRotated`: the same objects and scores as Ultralytics `predict`,
  except one borderline airport.jpg segment box from the same rounding. These exports are not
  template algos.
- Raw heads through `Yolo26Detect`, `Yolo26Keypoint`, `Yolo26Rotated`, `Yolo26Segment` and
  `YoloV8Segment`: identical objects, boxes, scores, keypoints, angles and masks (IoU 1.0) to the
  decoded-output types on the default exports of the same weights, which match Ultralytics.
- `rtmo_s_pose`: identical boxes, scores and visible keypoints to the original model's in-graph NMS.
  300 candidates cost 0.22 s against 0.09 s for the original on an x86 CPU (100: 0.12 s), as each
  candidate decodes its keypoints; bus.jpg has 63 candidates above 0.15 for 5 people. Its graph
  keeps a TopK; RKNN and TensorRT conversion has not been tried.
- `yolov8n_face`: boxes, scores, landmarks and visibility identical to a Python decode of the
  same heads on zidane.jpg (2 faces, 5 landmarks each).
- `heatmap_udp`: keypoints identical to mmpose's `refine_keypoints_dark_udp` and `UDPHeatmap`
  scaling on constructed 48 x 64 maps, edge peaks included.
- `rtmpose_m_pose` after `yolov8n_detect`: visible keypoints within 5 px of mmpose's affine crop
  and SimCC decoding, visibility 17/17 for all 4 people.

YOLO26 semantic static variants use `export_yolo26_semantic.py`, with the in-graph Resize and
ArgMax removed so the postprocessor receives class logits. Input/output shapes are fixed:

| ONNX file | Input | Output | SHA256 |
| --- | --- | --- | --- |
| `SemanticSegment.onnx` | `1x3x640x640` | `1x19x80x80` | `0297ea52c1e6da5047657153853e61136de4bfec675795c6bfb7ea9e1b071f28` |
| `SemanticSegmentBatch4.onnx` | `4x3x640x640` | `4x19x80x80` | `5b56d1508dcbf4da9e0b0da8e94080b5e45623ec75ebcbea29f52e876aeb5c26` |
| `SemanticSegmentBatch8.onnx` | `8x3x640x640` | `8x19x80x80` | `d285e4d8150f9cdc8be2736c4ae70c79d9c310ccd4db8df5fa2985c31bbdf481` |

### INT4 ONNX fixtures

Pinned Hugging Face `onnx-community` exports are stored locally under
`assets/test/onnx/int4/<repository>/model_q4.onnx`; their original configuration and
preprocessor configuration are alongside them. LFS SHA256 values were checked after download.
These are weight-only INT4 models with FP32 input/output, not packed INT4 output tests.
ONNX Runtime 1.23.2 CPU smoke inference with zero inputs produced finite FP32 outputs for all
the retained models. The checks establish loading and execution, not task accuracy or HybridRT
postprocessor integration; these models are not part of the template benchmark table.

| Repository | Revision | 4-bit MatMulNBits nodes | Output at smoke-test input | ONNX SHA256 |
| --- | --- | ---: | --- | --- |
| [dinov2-small](https://huggingface.co/onnx-community/dinov2-small/tree/8b1f705a3a7f6f062f6bdd21986c1583d3ef105d) | `8b1f705a3a7f6f062f6bdd21986c1583d3ef105d` | 72 | `1x257x384` | `5504f49e6b698e0b69d6bdf93a82278167cc12132588c5d1ea3f2e875efdb89d` |
| [vit-base-patch16-224-ONNX](https://huggingface.co/onnx-community/vit-base-patch16-224-ONNX/tree/cbd8c61cf91321b8dc986b5cbf84af9db7eac1a7) | `cbd8c61cf91321b8dc986b5cbf84af9db7eac1a7` | 72 | `1x1000` | `993e4e1c98c0ee591ce37e97becf8e7413bb111cc0908385edc100cfac0edef5` |
| [detr-resnet-50-ONNX](https://huggingface.co/onnx-community/detr-resnet-50-ONNX/tree/1b7b402cbb89a4f96db3bc81c06e73b5a607973b) | `1b7b402cbb89a4f96db3bc81c06e73b5a607973b` | 100 | `1x100x92; 1x100x4` | `ad7affaea89b2326f71abbb9293a06879a5ba777ec5fcaf0412313d369f79e67` |

Each retained graph explicitly sets `MatMulNBits.bits=4`; convolution and other operations
may remain FP32. Only one model per family is retained. Static `model_q4_batch1.onnx`,
`model_q4_batch4.onnx` and `model_q4_batch8.onnx` use 224 x 224 input and fixed output shapes.
All nine variants passed ONNX checker and ORT CPU zero-input inference with finite FP32 outputs.
For DETR this resolution is a smoke-test fixture, not its recommended accuracy evaluation size.

| Static INT4 model family | Batch 1 SHA256 | Batch 4 SHA256 | Batch 8 SHA256 |
| --- | --- | --- | --- |
| `dinov2-small` | `0bfd3673e3238aae84895ff44282852b7eb5876a744158d4d92d43ed9a5e3957` | `d727aa7a738203dfc392badc8bcf321d0e1e5eed243371aa7bfa106140be0541` | `60546dd5f1f41cfc203ea55e076f50f0462f609930eea02754fc18d7ebfcf47a` |
| `vit-base-patch16-224-ONNX` | `b51b1de7c890c6c486802fbe60688756f8f0715fc8c15deb816ff71c98da6174` | `6b4d963a8ceeec5f9915370605d1a848d72ac08919854843eb3b5d29dafca30f` | `dfb4326655393c1286f2e7057c22de75fdc79b4f89e047f125d4e926b8ca4934` |
| `detr-resnet-50-ONNX` | `5642aebf149cef8257fe93ecebf2ed1ec4b9eeb436030a553395206c5716a1aa` | `214fd2df5fddbc22bf97b81faf37e8af903a1774b6c5bbef7bedcffb707dc7d9` | `beb6e23f7fea5b665c3917b8fa18d8f8193316353aeb57f363a0f678b4f31963` |

### INT8 ONNX fixtures

Only YOLOv8n detection is retained from the basic YOLO family. The local
`assets/test/scripts/prepare_quantized_onnx.py` builds CPU QDQ Conv/MatMul/Gemm with signed INT8
activations and per-channel INT8 weights from the FP32 fixtures above, using ORT 1.23.2 and
64 COCO images with letterbox RGB / 255 preprocessing. FP32-output intermediate models stay
in `/tmp`; the retained `assets/test/onnx/int8/GeneralDetect{,Batch4,Batch8}Int8.onnx` files
have FP32 input and **native INT8 output**, checked by real ORT inference for all three batches.

The final QuantizeLinear uses symmetric per-channel scales (axis 1), calibrated on the same
64 images plus bus.jpg. This keeps pixel coordinates and confidence on separate scales.
Scales and zero points are embedded as initializers and in the ONNX
`hybridrt.output_quantization` metadata. These are output/type fixtures, not an accuracy claim.
HybridRT's ORT backend currently maps INT8 storage but does not propagate these quantization
parameters, and Buffer holds only one scale/zero point; these per-channel outputs must not be
fed directly to the existing postprocessors. They are not added to the FP32 benchmark table.

| File under `assets/test/onnx/int8/` | Input | Output dtype / shape | SHA256 |
| --- | --- | --- | --- |
| `GeneralDetectInt8.onnx` | `1x3x640x640` FP32 | INT8 `1x84x8400` | `f960d292d91f2521b050c9b744267892d37b7032004972a18fa4fc65d9a01797` |
| `GeneralDetectBatch4Int8.onnx` | `4x3x640x640` FP32 | INT8 `4x84x8400` | `a5f3906dbb47364465f8ca813c3e5bbee727a275cba47720b6f471cfb3c08d88` |
| `GeneralDetectBatch8Int8.onnx` | `8x3x640x640` FP32 | INT8 `8x84x8400` | `733329dd0948396e5d377c48756ea91da569656a38cc1e445b7d6bd241ca6fa8` |

### RKNN on RK3588

`assets/test/scripts/rknn_convert.py` uses rknn-toolkit2 2.3.2, matching librknnrt 2.3.2
on RK3588. Mean/std are embedded in the model; `image_input: true` feeds raw uint8 NHWC,
with interleaved RGB or BGR matching the template. FP16 models are kept for the four basic
640 x 640 batch-1/4/8 pipeline fixtures.

For postprocessing, the collection covers 26 model groups (25 postprocessor types, with
separate OCR detection and recognition). Final preparation on 2026-10-25 produced **72 native
INT8-output models** (24 groups x batch 1/4/8) and **23 FP16-output batch-1 models**. All 95
loaded with librknnrt 2.3.2; SDK queries verified batch dimensions, output types, finite positive
INT8 scales and zero points. This final check does not run inference or measure speed.

`prepare_rknn_fp16.py` prepares the floating batch-1 models. `prepare_rknn_batches.py` expands
the remaining static batch graphs and prepares integer outputs; each expanded graph was checked
against independent batch-1 ORT results. Native batch graphs passed for the prepared variants.
Vision calibration uses 64 COCO images; OCR uses document/line crops with template normalization.
Input bytes for later board tests are prepared on the build host.

Classification, Embedding and CTC recognition exports otherwise retain FP16 output after RKNN
weight quantization. Their integer variants explicitly quantize the output probabilities with
`QuantizeLinear`, scale `1/255`, zero point `-128`. These are INT8 **output** fixtures; their bodies
may still use FP16, so they are for postprocess comparison, not inference-precision benchmarks.
OCR required basic graph folding and replacement of two remaining inference BatchNorm nodes by
equivalent channelwise Mul/Add (`fix_rknn_ocr_output.py`). The quantized ORT outputs matched
before/after that rewrite for batch 1/4/8 on the checked inputs.

Unavailable references:

- RT-DETR-L and RTMO fail batch-1 conversion at optimization levels 3 and 0 in toolkit
  MatMul/Conv and Reshape rewrites; their batch-4/8 RKNN variants are not generated.
- The user YOLOv5 OBB source is QAT. Its INT8 batch-1/4/8 exports preserve that quantization.
  RKNN also changes explicit FP16 output casts back to INT8, so no independent FP16 counterpart
  is supplied. A file with INT8 outputs is not kept under the FP16 reference filename.

Converted files are local under `assets/test/rknn/`. Final metadata is recorded locally in
`assets/test/benchmark_results/rknn_full_conversion_query.log`; conversion and batch-equivalence
logs are in the same directory. Models and conversion scripts remain untracked. Earlier measured
results are in [benchmark.md](../benchmark.md); no further speed tests were run after preparation
was requested. Prior in-graph GatherElements failures of the E2E exports remain runtime limitations;
conversion and IO validation do not establish task accuracy or correct NPU inference for those graphs.

<details>
<summary>Converted RKNN model hashes and verified output types</summary>

| Algo | File in `assets/test/rknn/` | Native output | SHA256 |
| --- | --- | --- | --- |
| `yolov8_detect` | `YoloV8Detect.rknn` | FP16 | `1dc08f8932494c91f78b049088fd165193e20ff9f906a9fff1f2e28da775b040` |
| `yolov8_detect` | `YoloV8DetectInt8.rknn` | INT8 | `8d898eb0feb7cf012868dd38cf707d3786179d53d9112f617ced336faa386b64` |
| `yolov8_detect` | `YoloV8DetectBatch4Int8.rknn` | INT8 | `13b527970b8a10da33e2e6167adf44e7f5def3ef92a95e6974dc97a8ac3087f2` |
| `yolov8_detect` | `YoloV8DetectBatch8Int8.rknn` | INT8 | `df9a416917fd451626b395e2b4180761c59d111445aade1a12b6a35efdcfca69` |
| `yolo26n_detect_raw` | `Yolo26Detect.rknn` | FP16 | `c207705731f0ee415ce4d19449a421682da768afad34281be63778c2023fbbf2` |
| `yolo26n_detect_raw` | `Yolo26DetectInt8.rknn` | INT8 | `024b7c4a13f278c05f529a97c143fc476cd535f5a4c57985df2e24aabe34778f` |
| `yolo26n_detect_raw` | `Yolo26DetectBatch4Int8.rknn` | INT8 | `72676cb52f30de01304cc48e2b572fed6b83f4aa026537cda9758ee9b711f924` |
| `yolo26n_detect_raw` | `Yolo26DetectBatch8Int8.rknn` | INT8 | `2cc1f534d6902934985c6eda36808c50f6269b782ead511ed0d99838e5e8cc31` |
| `yolov5_detect` | `YoloV5Detect.rknn` | FP16 | `937337e2459e854c73ab33912b6e1ff896d4bd66241d273f9953137f59fa20ba` |
| `yolov5_detect` | `YoloV5DetectInt8.rknn` | INT8 | `b4d020c780283c079b73731fb7c467442036f9f673dbe34955e04c11fa92dcf6` |
| `yolov5_detect` | `YoloV5DetectBatch4Int8.rknn` | INT8 | `fd2ff306e882360ae77f0b68e27bacabad1a14ce86c09ea342ad84e1e92fcacd` |
| `yolov5_detect` | `YoloV5DetectBatch8Int8.rknn` | INT8 | `eed2c2fd5311ccdc236becfd50023289a67d6289e26cfeb14e75fa1acd0168e7` |
| `yolo26n_e2e_detect` | `E2EDetect.rknn` | FP16 | `32507cc5f1bbf4288e5ad741934761d988cf704c2dbc50a0064148a67cbdb27f` |
| `yolo26n_e2e_detect` | `E2EDetectInt8.rknn` | INT8 | `ca95752ff5bfbe022927748cddd61d963b2cc2d06dd344d422a125a0d385f58e` |
| `yolo26n_e2e_detect` | `E2EDetectBatch4Int8.rknn` | INT8 | `541168d91a90fdee6b73d3baf8bf8a614e487a02b3e22e304275a6180feef193` |
| `yolo26n_e2e_detect` | `E2EDetectBatch8Int8.rknn` | INT8 | `9480984ba57d10987d0a35627330c35473db7ead47d67183c2fdfe5c9226bce2` |
| `yolov8n_detect` | `GeneralDetect.rknn` | FP16 | `aa9276ed9f0cc95fdb5eddc44c752d97da85d4a4e34e720944d5bb9e53cb655d` |
| `yolov8n_detect` | `GeneralDetectInt8.rknn` | INT8 | `dd1dcdaccddb0c257a6856febb80a06499de646870b16817e31441d8c740f634` |
| `yolov8n_detect` | `GeneralDetectBatch4Int8.rknn` | INT8 | `2a2adb3f6eb3e9575ec37598f4a45f86b4186a107bf6c377d0c7ddfbc2491a1e` |
| `yolov8n_detect` | `GeneralDetectBatch8Int8.rknn` | INT8 | `ca721479cccd199cf90c2256b45acee5a3c41da3cc00c74b8dd09b013a9773d8` |
| `yolo26n_e2e_obb` | `E2ERotated.rknn` | FP16 | `4230c3de95bcfd3e9e76f713ab0a78df9236763d4195fdbcef6406c47e1597b6` |
| `yolo26n_e2e_obb` | `E2ERotatedInt8.rknn` | INT8 | `0108774cac8baaee6f0abe2d4311f43c15a09ef1787e564e1dac02552ae41394` |
| `yolo26n_e2e_obb` | `E2ERotatedBatch4Int8.rknn` | INT8 | `f80bd7e4fe9344bbf76234d19500a6ecd6b86fe8c9b8712e4de65b346af1f927` |
| `yolo26n_e2e_obb` | `E2ERotatedBatch8Int8.rknn` | INT8 | `989114a60c9685214ff81ae43b95305f6e3e674ca83a4af26291b269f55b8a79` |
| `yolov8n_obb_raw` | `YoloV8Rotated.rknn` | FP16 | `85fe6213a142f627336ae0939e1521457b3fca5f94796b5777712bfc18de07fa` |
| `yolov8n_obb_raw` | `YoloV8RotatedInt8.rknn` | INT8 | `5c678b5f726cf7bc39c76ebcc060290aa2565d68f66310be7e81ef429f4f9950` |
| `yolov8n_obb_raw` | `YoloV8RotatedBatch4Int8.rknn` | INT8 | `ed00e75a8b2fec641b686d3571e990c915ae048d803e43ab6895be10e76e9ae5` |
| `yolov8n_obb_raw` | `YoloV8RotatedBatch8Int8.rknn` | INT8 | `a5b1a9e4607e992cbf524635e58f5ee9760ab69d1751b21165e9e12185b94322` |
| `yolo26n_obb_raw` | `Yolo26Rotated.rknn` | FP16 | `4b81268e97950fc6dfe30fde4c269dc57ffef116fe3ec0d4cf4b32abd963d637` |
| `yolo26n_obb_raw` | `Yolo26RotatedInt8.rknn` | INT8 | `dc9719e150651ac023c0efa4db47565d02e3f6f5d4639cfa032c804e5361a4d1` |
| `yolo26n_obb_raw` | `Yolo26RotatedBatch4Int8.rknn` | INT8 | `1e755d50151c820a1c005068b150490ad8278d689c2e3d39527f5da22cc5a6eb` |
| `yolo26n_obb_raw` | `Yolo26RotatedBatch8Int8.rknn` | INT8 | `835772beafebe4119c8bd872f02f0e5dcd7bbfb1819bdbdb59c590ec43327e75` |
| `yolov5_rotated` | `YoloV5RotatedInt8.rknn` | INT8 | `bf9ac1f52d7abe5d80a7bbba08f1a4404785a57cd786ca7103c6936ea908bbca` |
| `yolov5_rotated` | `YoloV5RotatedBatch4Int8.rknn` | INT8 | `7a83f6399c611d220f4c24d7f143ff5d62b4e8ced63ec9f2643929ef869270a7` |
| `yolov5_rotated` | `YoloV5RotatedBatch8Int8.rknn` | INT8 | `5b3c0394505df35fcf495bee4f2242f6ec5cd2dac60cd56c16b0b779c423ffe9` |
| `yolov8n_obb` | `GeneralRotated.rknn` | FP16 | `44c6ddc8cd54d0506a4ad59e7c3cfbe15049bd3a04dfbf37f90336e64d78b69a` |
| `yolov8n_obb` | `GeneralRotatedInt8.rknn` | INT8 | `7794e52c85e704441b8dbb5c82d6e9612bdf66cdaacf978c441619bf6c960154` |
| `yolov8n_obb` | `GeneralRotatedBatch4Int8.rknn` | INT8 | `2edcbd7c4d567beed5dd6b386065888ce1e712a2dc91256852b1a0baa7a3c7b1` |
| `yolov8n_obb` | `GeneralRotatedBatch8Int8.rknn` | INT8 | `ca418147d00a4c943037692797463cbb8326dee618b6e3beeeffff0692818bd0` |
| `yolov8n_classify` | `GeneralClassify.rknn` | FP16 | `98b23d07602b4219a7a2a1fb5139528a3fdd619b6190bd45e77406d60a46f1ae` |
| `yolov8n_classify` | `GeneralClassifyInt8.rknn` | INT8 | `d0d40eb4361fbad50a620fc4bf2540ebb09f546baed082fa060d5edb54a87426` |
| `yolov8n_classify` | `GeneralClassifyBatch4Int8.rknn` | INT8 | `cae1010434b9fd46c3daf0aefe807225f4a5cb237b52a6e0757b01dc72dac58e` |
| `yolov8n_classify` | `GeneralClassifyBatch8Int8.rknn` | INT8 | `fa1eea162eb6381ef9c018d4c5466dfad3c77019b5ebf3b5b758b9431bf98063` |
| `yolov8n_classify_embedding` | `Embedding.rknn` | FP16 | `f4060460286ad9ddda1ec6fa2dd1de610e9920bfcef71b04b713d6fb8163febf` |
| `yolov8n_classify_embedding` | `EmbeddingInt8.rknn` | INT8 | `9cea3b23dae5f0c72dcc57957bca00019ab80d79baaae6d2f403dbb50cb55ecb` |
| `yolov8n_classify_embedding` | `EmbeddingBatch4Int8.rknn` | INT8 | `9a3717979fbeb16e25ed98af4ee07450f283744f33f21646e5296b5e105e59d6` |
| `yolov8n_classify_embedding` | `EmbeddingBatch8Int8.rknn` | INT8 | `3fec4fc7cd4faaabc083c2683c517fac0d96ce7290fea6ccb983eb82f1581073` |
| `yolov8n_pose` | `GeneralYoloPose.rknn` | FP16 | `45c001fe387bdae9b649a8491339e500f485d870eda3d34812d4c0cb82d648e6` |
| `yolov8n_pose` | `GeneralYoloPoseInt8.rknn` | INT8 | `7bfb9630154087a66bf3a78bc44d56e5dd047abba5bfd210824f1598afef557c` |
| `yolov8n_pose` | `GeneralYoloPoseBatch4Int8.rknn` | INT8 | `95a73c8d5b470ba66bdffd85149a3f43b818b8ffd3706d52b4923697c691d9f6` |
| `yolov8n_pose` | `GeneralYoloPoseBatch8Int8.rknn` | INT8 | `7aad702661894cdc51b3ed24986352b5ede23a087b9e88b1e15c830c74909ef3` |
| `yolov8n_face` | `YoloV8Keypoint.rknn` | FP16 | `402fcfbd33b42a6f3920b61e980191cf5b04ad868d8404f7a01c15fb3052a518` |
| `yolov8n_face` | `YoloV8KeypointInt8.rknn` | INT8 | `dc7bd9b27c9589371a7c9a4b4632f644881a0b5316af27666be371973d6ca34b` |
| `yolov8n_face` | `YoloV8KeypointBatch4Int8.rknn` | INT8 | `29169c9b62fac66da17e47ed31ccec6766e253b15fd6b128e9d8f1b8e2c6012a` |
| `yolov8n_face` | `YoloV8KeypointBatch8Int8.rknn` | INT8 | `2550a83bf9e26d32b60817bd09e9b11e5b8adb282dcac1f619d55ab31459cae9` |
| `yolo26n_pose_raw` | `Yolo26Keypoint.rknn` | FP16 | `929f38ff11ca1e619770cde1f64a8d3badc0024b4ddf974a1943bdb479083081` |
| `yolo26n_pose_raw` | `Yolo26KeypointInt8.rknn` | INT8 | `4f9c34b193bff56fb7208ed588ce297e860748e2961463c2b93d0b4ddaa46e84` |
| `yolo26n_pose_raw` | `Yolo26KeypointBatch4Int8.rknn` | INT8 | `1140cdfebbf54eb695657fdd797958edeb2b9c2015837b2386a48443db12c50d` |
| `yolo26n_pose_raw` | `Yolo26KeypointBatch8Int8.rknn` | INT8 | `baef38fff97b690364d787c91f4790a31ea6cd97aa070ac6bc297a1c8130c312` |
| `yolo26n_e2e_pose` | `E2EKeypoint.rknn` | FP16 | `03dc1a7d497d690e72ae99caa52375b033a5f3f491b1834b38b194e10b6f2ca3` |
| `yolo26n_e2e_pose` | `E2EKeypointInt8.rknn` | INT8 | `a3aa0bd724bd26f28ab40ac0933ccfe6deffc897e08ab3fcd31d2d036717a1d0` |
| `yolo26n_e2e_pose` | `E2EKeypointBatch4Int8.rknn` | INT8 | `293d6be5a54cdf4cd43c65866dc82ee76820c84a3cda82d0bb65c4bc78d818f2` |
| `yolo26n_e2e_pose` | `E2EKeypointBatch8Int8.rknn` | INT8 | `544a84a4f5f99e3361c394088883c3085d768bbceecc99a67a70738f939cce9a` |
| `rtmpose_m_pose` | `GeneralKeypoint.rknn` | FP16 | `2cd06553cbec1c0cd75716b544df4f08c99a5dde33b69d2aea79a05a65eda06c` |
| `rtmpose_m_pose` | `GeneralKeypointInt8.rknn` | INT8 | `3273d999990e1c58c77183891c29815ef4979262558b44e58e5a6636a58cbf3a` |
| `rtmpose_m_pose` | `GeneralKeypointBatch4Int8.rknn` | INT8 | `28e2f8a17f69a669ac4de5a5c7b5971611663502700942da20c56a8e505f51bd` |
| `rtmpose_m_pose` | `GeneralKeypointBatch8Int8.rknn` | INT8 | `4bdd892ae4d7c8f047283bd1393c2e2c2f567947d06b909e7ca4466abec7885f` |
| `yolov8n_segment` | `GeneralYoloSegment.rknn` | FP16 | `94252b58b25fd5021f36b504bc57837f7d33d32d6f06dfc5bd1e5802e25f9c99` |
| `yolov8n_segment` | `GeneralYoloSegmentInt8.rknn` | INT8 | `4962367ead3eeab7cddd9cd74f472ca2522053d0afd10db760499d573eaa2144` |
| `yolov8n_segment` | `GeneralYoloSegmentBatch4Int8.rknn` | INT8 | `9d6aba82a8f6b83aba5a36a50837d3d03c88e8c534f4662b4645f59c42c4becf` |
| `yolov8n_segment` | `GeneralYoloSegmentBatch8Int8.rknn` | INT8 | `3159309c51ea481c8c26e86f433ed197e30998357ac810730e34e7303a3677b5` |
| `yolov8n_segment_raw` | `YoloV8Segment.rknn` | FP16 | `655ecc345e8656195f6466abbf6309d01b5d4e77f1ef4f82c4fecf4e064c5d42` |
| `yolov8n_segment_raw` | `YoloV8SegmentInt8.rknn` | INT8 | `3b37be9862c4935ac607d2624ec747b2f2100cc36bd124851357770ca93a551f` |
| `yolov8n_segment_raw` | `YoloV8SegmentBatch4Int8.rknn` | INT8 | `e1f9383243a4dc195d92ad10eaa98a27c79e51a49f38d96a143dc25acf0c9fab` |
| `yolov8n_segment_raw` | `YoloV8SegmentBatch8Int8.rknn` | INT8 | `c576ab5ebff2f95665312ef669205e00af01d3a7e10a1065ffe00ba0503fa12f` |
| `yolo26n_segment_raw` | `Yolo26Segment.rknn` | FP16 | `5fc57053a95205fb9f9a44a7a7679900ce55d5db308e1d148819a8a646a1b004` |
| `yolo26n_segment_raw` | `Yolo26SegmentInt8.rknn` | INT8 | `e9f2a650b4feacbbb8aca82be77befa8d036338bac7d89ca66b925186976f167` |
| `yolo26n_segment_raw` | `Yolo26SegmentBatch4Int8.rknn` | INT8 | `c8244a7530b624a5176b11e15427cffddd3c57e99e6a290e745d2a20328abfe1` |
| `yolo26n_segment_raw` | `Yolo26SegmentBatch8Int8.rknn` | INT8 | `7a112f374fcf9df7cc995ba37464f5677b360b8531763cabf701f8bb94198052` |
| `yolo26n_e2e_segment` | `E2ESegment.rknn` | FP16 | `e16ec5dd9e6fd52ed17106af39539bfccb80a149db14fc60b67ed254ba1dfc74` |
| `yolo26n_e2e_segment` | `E2ESegmentInt8.rknn` | INT8 | `b2ee26161981929c632ef052c1c2c14c8ded45a315b9b88d420987ea4344dda3` |
| `yolo26n_e2e_segment` | `E2ESegmentBatch4Int8.rknn` | INT8 | `1e022a159ab9e38fb8e3fc8ba9d9683ec2128e39b2979fe70753f9b3f2e8185a` |
| `yolo26n_e2e_segment` | `E2ESegmentBatch8Int8.rknn` | INT8 | `e1bb70851990e593d9deafbd006343446c6b1875fba78f575d97446166f64bfa` |
| `yolo26n_semantic` | `SemanticSegment.rknn` | FP16 | `8ae6b9009a109addc1165648492a31c05c67c0d84547c825706979309347569d` |
| `yolo26n_semantic` | `SemanticSegmentInt8.rknn` | INT8 | `030fe2efdc14c292f3afd00e7e7480c44e57e4926376ea9750dc05e3065241de` |
| `yolo26n_semantic` | `SemanticSegmentBatch4Int8.rknn` | INT8 | `a52ee91a3b1afba290aa181c83c3969d29d9214cea56be4b1ef2607ab3f0b015` |
| `yolo26n_semantic` | `SemanticSegmentBatch8Int8.rknn` | INT8 | `1792e363da52510d5b6a59156b1a0e4cd7a7b7571be0a3d43b72a23143383253` |
| `ppocrv6_tiny_det` | `OcrDetection.rknn` | FP16 | `59c2845ffb7ac3eb52d987e1a81eebe50f73db1f02114a43500d98ae90ed6b85` |
| `ppocrv6_tiny_det` | `OcrDetectionInt8.rknn` | INT8 | `faad92a5b8c89b09e8b20fdd4e6e3e1a76185951625e2152fa78c577b5c5e62e` |
| `ppocrv6_tiny_det` | `OcrDetectionBatch4Int8.rknn` | INT8 | `4e7d4a75b24b06696ce8c2bc0d62b297de974b0f3e41a53aa0e562c3bbb575e3` |
| `ppocrv6_tiny_det` | `OcrDetectionBatch8Int8.rknn` | INT8 | `8102535ff05919f018fae308557b8f72c96a56494d3d28d5e5fabb0ddc191f04` |
| `ppocrv6_tiny_rec` | `OcrRecognition.rknn` | FP16 | `e0307627e6bb180acb554a50afc4895a4b4481cbde41ec6baa20bd700ea615ad` |
| `ppocrv6_tiny_rec` | `OcrRecognitionInt8.rknn` | INT8 | `1b0703f6c411c85cec867747a84dc19028e766b696618e0c5d70f105b306c2d7` |
| `ppocrv6_tiny_rec` | `OcrRecognitionBatch4Int8.rknn` | INT8 | `6188679fc89b8c4625c39ce8e57962758fa870f4919bd45b22f0c4936cbd2309` |
| `ppocrv6_tiny_rec` | `OcrRecognitionBatch8Int8.rknn` | INT8 | `31b96483e0fb3793039e22e39b8834b196877558692448844974c6958c8603b0` |

</details>

Against the ORT results of the same models on bus.jpg, airport.jpg and zidane.jpg (librknnrt 2.3.2):

| Types | RKNN result |
| --- | --- |
| `GeneralDetect`, `YoloV8Detect` (raw) | same objects; boxes within 8 px, scores within 0.07 |
| `GeneralYoloSegment` | same objects; mask IoU 0.935 or more |
| `GeneralYoloPose`, `YoloV8Keypoint` (raw, face) | same people and faces (one borderline person fewer for the raw pose heads); keypoints within 7.3 px |
| `GeneralRotated`, `YoloV8Rotated` (raw) | same 11 planes; angles within 2.2 degrees |
| `GeneralClassify` | same top 1 on airport.jpg; bus.jpg's near-flat scores (0.02) differ |
| `SemanticSegment` (YOLO26n-sem) | bus.jpg: FP16 decodes 12 classes and INT8 11 (ORT: 12); only class counts were compared, not class sets or mask IoU |
| `GeneralKeypoint` (`simcc`) after `yolov8n_detect` | same 4 people, visibility 17/17; keypoints within 7.3 px |
| `Yolo26Detect`, `Yolo26Keypoint`, `Yolo26Rotated` | same objects; boxes within 9.5 px, angles within 0.3 degrees; keypoints within 2 px at the median, one uncertain wrist off by 19.5 px |
| `E2EDetect`, `E2EKeypoint`, `E2ESegment`, `E2ERotated` | YOLO26 exports convert but return no or wrong rows: the runtime reports GatherElements indexes out of range in the in-graph top-k; YOLOv8 `nms=True` exports do not convert (dynamic NonMaxSuppression) |
| `DetrDetect`, `RtmoKeypoint` | RT-DETR-L and the fixed RTMO graph do not convert (shape errors in the toolkit's graph rewrites, also after onnxslim) |

Use raw-head or decoded exports on RKNN; for YOLO26, the raw heads. Milliseconds per call on one thread of the board (fp16, RGA
preprocessing from a host image, 100 runs):

| Algo (input) | Preprocess | Inference | Postprocess | `process_sync` |
| --- | --- | --- | --- | --- |
| `yolov8n_detect` (640 x 480) | 1.8 | 38.5 | 0.7 | 43.0 |
| raw detect heads (640) | 1.8 | 43.9 | 1.7 | 51.9 |
| `yolov8n_segment` (640 x 480) | 1.8 | 48.6 | 1.2 | 49.9 |
| `yolov8n_pose` (640 x 480) | 1.8 | 45.1 | 0.05 | 44.8 |
| raw pose heads (640) | 1.9 | 45.3 | 0.07 | 50.0 |
| `yolov8n_obb` (640 x 480) | 1.6 | 38.2 | 0.5 | 39.0 |
| `yolov8n_obb_raw` (1024) | 2.3 | 108.7 | 1.6 | 111.6 |
| `yolov8n_classify` (640 x 480) | 1.6 | 21.3 | 0.0 | 22.3 |
| `lraspp_voc_segment` (512, previous LR-ASPP export) | 1.5 | 66.5 | 8.7 | 70.5 |
| `yolov8n_face` (640) | 1.7 | 45.4 | 0.06 | 47.3 |
| `rtmpose_m_pose` (192 x 256 crop) | 0.8 | 18.6 | 0.06 | 20.4 |

A dma-buffer source saves 0.4 to 0.8 ms of preprocessing. int8 models, not measured here, cut the
inference time. Optimization history and measurements are in `assets/perf.md`.

### CANN on Ascend 910B2

ATC converts each ONNX test model to an `.om` beside it, with the model's own `preprocess`
and `postprocess` unchanged and the backend `{"engine_type": "cann", "config": {}}`:

```sh
atc --model=assets/test/onnx/GeneralDetect.onnx --framework=5 \
    --output=assets/test/cann/GeneralDetect --soc_version=Ascend910B2
```

ATC defaults to FP16. Add `--precision_mode=must_keep_origin_dtype` to keep FP32, which an
FP16 model needs to reproduce an ONNX Runtime reference exactly. An `.om` built by the
CANN 8.2 ATC loads and runs on the 9.0.0 runtime. A container whose `opp` package is
trimmed, as the CANN 9.0.0 image used here was, fails ATC during initialization
(`FEOpsKernelInfoStore: Initialize custom and builtin sub-information library failed`);
a full toolkit installation converts the same models.

With AIPP the `.om` takes the uint8 image directly. This config converts and gives the same
objects as the float32 model, with normalization on the NPU (2.76 ms per call against 4.11):

```sh
cat > aipp.cfg <<'CFG'
aipp_op {
  aipp_mode: static
  input_format: RGB888_U8
  src_image_size_w: 640
  src_image_size_h: 640
  csc_switch: false
  mean_chn_0: 0
  mean_chn_1: 0
  mean_chn_2: 0
  var_reci_chn_0: 0.0039215686274509803
  var_reci_chn_1: 0.0039215686274509803
  var_reci_chn_2: 0.0039215686274509803
}
CFG
atc --model=assets/test/onnx/GeneralDetect.onnx --framework=5 \
    --output=assets/test/cann/GeneralDetect_aipp --soc_version=Ascend910B2 \
    --insert_op_conf=aipp.cfg
```

Its algo keeps the same `postprocess` and takes `dst_image_format: "inter_rgb"` with `mean`
and `std` at their defaults, as AIPP applies them. AIPP outputs FP16, UINT8 or INT8, so
`--precision_mode=must_keep_origin_dtype` cannot be combined with it.

Against the ORT results of the same models on bus.jpg, airport.jpg, zidane.jpg and two
document images (CANN 9.0.0 runtime, FP16 unless noted, host preprocessing):

| Types | CANN result |
| --- | --- |
| `GeneralDetect` | same 8 objects and classes; boxes at mean IoU 0.998, scores within 0.003 |
| `GeneralRotated` | same 13 objects and classes; boxes at mean IoU 0.994, scores within 0.002 |
| `GeneralClassify` | same top 1 on every image; scores within 0.003 |
| `Embedding` | identical embeddings |
| `GeneralYoloPose` | FP16 drops one borderline person; FP32 gives the same 7 people at mean IoU 0.9999, scores within 0.0003 |
| `E2ESegment` | same 8 objects and mask counts; boxes at mean IoU 0.998, scores within 0.006 |
| `Ocr` detection and recognition (`OcrModel`) | same text regions; 79 of 80 lines identical |
| `GeneralClassify` after `GeneralDetect` (cascade) | same classes and scores as the reference cascade |
| `GeneralDetect` with AIPP, fed uint8 | same 8 objects and classes; boxes at mean IoU 0.998, scores within 0.002 |

Milliseconds per call on the 910B2 (50 runs for the engine, 5 images for `process_sync`,
which includes CPU preprocessing and postprocessing):

| Algo (input) | Inference | `process_sync` |
| --- | --- | --- |
| `yolov8n_detect` (640 x 480) | 0.80 | 4.11 |
| `yolov8n_obb` (1024) | 1.18 | 5.25 |
| `yolov8n_classify` (224) | 0.27 | 2.47 |
| `yolov8n_classify_embedding` (224) | 0.27 | 2.79 |
| `yolov8n_pose` (640 x 480) | 0.85 | 3.70 |
| `yolo26n_e2e_segment` (640 x 480) | 1.76 | 4.08 |
| `ppocrv6_tiny_det` (640) | 0.93 | 27.63 for the whole OCR cascade |
| `ppocrv6_tiny_rec` (48 x 320) | 0.31 | one call per text line |

FP32 models track the reference more closely throughout: `GeneralRotated` reaches mean IoU
0.9997 and `E2ESegment` 0.9998, against 0.9944 and 0.9984 in FP16.

The `.om` files pack into a `.hrtc` and run from it, the Python binding loads both a package
and a JSON manifest with `cann` backends, and pooling the stages raises throughput from 2.73
to 2.05 ms per image over 36 images with identical results.

DVPP was not exercised: this 910B2 is a virtualized slice whose driver exposes no media
engines, so `hi_mpi_vpc_*` tasks return `HI_ERR_VPC_SYS_ERROR (0xA0078015)` and preprocessing
falls back to the CPU. The DVPP operators remain validated on an Atlas 300V Pro with CANN 8.2,
where the models carry AIPP. VPC input and output memory comes from `hi_mpi_dvpp_malloc`,
which that product requires; only Atlas 350, A3, A2 and 200I/500 A2 products also accept
`aclrtMalloc`, and memory from the wrong allocator fails with `HI_ERR_VPC_BADADDR
(0xA0078011)`. The preprocessor proves VPC once at startup, so a device that opens a channel
but cannot run tasks leaves the whole step to the CPU instead of failing every call.
