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
it converts uint8 BGR, RGB, NV12 or NV21 sources with `linear` or `nearest` interpolation into
3-channel planar or interleaved uint8 or float32 inputs, following OpenCV's uint8 color
conversion and sampling, so it matches the CPU result up to OpenCV's own rounding
differences. Other conversions (`cubic`, gray inputs, other source formats) run on the CPU.
NV12 halves the bytes a host frame uploads compared with BGR. A static batch whose samples all
fuse runs as one kernel launch (up to 16 samples per kernel); on the CPU the samples of a batch
run on OpenCV's threads.
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
With a batch above 1, samples of dma-buf inputs are written on the host (RGA is not used for them)
and the `dvpp` preprocessor is refused. Dynamic batches are not supported: the batch is fixed at load.

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
hrt-ocr -m ocr.hrtc -r ppocrv6_small_rec -e ort -i assets/test/images -o ocr_results.json
```

## OCR model

`model::OcrModel` combines one detection algo and one recognition algo from a loaded
`hrtc::Package`. Load `.json` with `hrtc::load_json` or `.hrtc` with `hrtc::load`, then call
`OcrModel::create(package, "ppocrv6_tiny_det", "ppocrv6_tiny_rec")`. Select
`ppocrv6_small_rec` instead for the alternative recognizer. The two recognition models
are alternatives, not successive stages. Each algo keeps its own backend alternatives, but
both stages must run on one device: CPU engines (ONNX Runtime CPU, MNN, ncnn, RKNN), CUDA
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

Use `-r ppocrv6_small_rec` for the small recognizer, `-p cpu` for CPU, or omit `-p` to keep
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
static batch-4 fixtures below. `Ocr` has
separate `OcrDetection.onnx`, `OcrRecognition.onnx` (tiny) and `OcrRecognitionSmall.onnx`
files because detection and recognition need different networks. `Embedding.onnx` is a
copy of the classifier and returns its 1000 class scores as an example vector.
Models, source weights, scripts and validation results stay local and untracked.

The current collection contains 27 files covering all 25 postprocessor types.
`YoloV5Rotated` uses the user-provided `v0013_obbczs_detection_0808int8.onnx`.
Its anchors, normalization and angle convention remain unconfirmed; its template
configuration supports pipeline checks rather than an accuracy reference.

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
| `yolov8_detect` | `YoloV8Detect.onnx` | `1 x 3 x 480 x 640` | `714820197e0c37fbf4e395ea4c3fafb512a8b60e2d565cac03b190194e18013d` |
| `yolo26n_detect_raw` | `Yolo26Detect.onnx` | `1 x 3 x 640 x 640` | `0fc641edc5ba266959e0135deeec849937f8049882a2269a0cea0021c7049b57` |
| `yolov5_detect` | `YoloV5Detect.onnx` | `1 x 3 x 640 x 640` | `968d45c63e7bf811a7fe0e6e73c0740afd079996985608714b79d39faffc6fd3` |
| `rtdetr_l_detect` | `DetrDetect.onnx` | `1 x 3 x 640 x 640` | `89e10a08e26eb872fa9cd6d0397b2eb9569fa2df5d1d37653ae5fa36fc08e862` |
| `yolo26n_e2e_detect` | `E2EDetect.onnx` | `1 x 3 x 640 x 640` | `cc54c1e9bfc069212f274d07ca10450ad08b83f078ac8e09cd1e532eeeb1a100` |
| `yolov8n_detect` | `GeneralDetect.onnx` | `1 x 3 x 480 x 640` | `a8ede0ffb077b655aa2d1be3044298f4374d1d278d84f6d87ab2cd404895d071` |
| `yolo26n_e2e_obb` | `E2ERotated.onnx` | `1 x 3 x 640 x 640` | `13476c2bb3d7966df6c880036ecc4a3cdf41318e3d2e6bca483f4306ef691f63` |
| `yolov8n_obb_raw` | `YoloV8Rotated.onnx` | `1 x 3 x 480 x 640` | `36223a5306c60c620186aa58a5a9c9bdcd34ad181c119d096edb8c2d30089a4a` |
| `yolo26n_obb_raw` | `Yolo26Rotated.onnx` | `1 x 3 x 640 x 640` | `9393f99be7ee049e797b77032e6b9d95dc07ff5d6ba275cf626b9bda5d0a494c` |
| `yolov5_rotated` | `YoloV5Rotated.onnx` | `1 x 3 x 1280 x 1280` | `afffc7a2813e15e9137648bc7655155827b1cdace2a0999b1919d1196be57dfb` |
| `yolov8n_obb` | `GeneralRotated.onnx` | `1 x 3 x 1024 x 1024` | `c207d49b65f8173265f2e2ff66f59bfef54b0b6d6da578a31c223b190e13dd60` |
| `yolov8n_classify` | `GeneralClassify.onnx` | `1 x 3 x 224 x 224` | `12b84b2d5f63b9e5a7d8122643d61051810dffad6642bee875c4cd4f24d79a87` |
| `yolov8n_classify_embedding` | `Embedding.onnx` | `1 x 3 x 224 x 224` | `b3b8c37ac7d6bfe052fcc17e4d018bd79a68f354bee845e9073f080b3705ce95` |
| `yolov8n_pose` | `GeneralYoloPose.onnx` | `1 x 3 x 640 x 640` | `de42e8e71ff022fe44287e6723fbf6429d1fee84084e63fd19630e1d6496785d` |
| `yolov8n_face` | `YoloV8Keypoint.onnx` | `1 x 3 x 640 x 640` | `af09683be4937ebe23e9da0346cbf46cd937b73ed5fac986e6205e6c7cdd2c25` |
| `yolo26n_pose_raw` | `Yolo26Keypoint.onnx` | `1 x 3 x 640 x 640` | `0c0b46ec46f05727218ee2431942e02f40d9f928a4a12d720ccadfba715fd59a` |
| `yolo26n_e2e_pose` | `E2EKeypoint.onnx` | `1 x 3 x 640 x 640` | `18be1d966c3c0c5c38a81fc301933ec94bb2633313e2806f14999de3001a2f8f` |
| `rtmo_s_pose` | `RtmoKeypoint.onnx` | `1 x 3 x 640 x 640` | `f686d078c403183ebb428fc7b3818bc7ebb838fd8486423dc20688a1f8079c31` |
| `rtmpose_m_pose` | `GeneralKeypoint.onnx` | `1 x 3 x 256 x 192` | `204ed5161fc30dd26f1af8e40d87426a2f5885828840e4abdad2e22b77d487d7` |
| `yolov8n_segment` | `GeneralYoloSegment.onnx` | `1 x 3 x 480 x 640` | `1f05bc76e1a542931f9a4b0c21f9d281ff4a8136a3cf8d03869996b0b9a5d124` |
| `yolov8n_segment_raw` | `YoloV8Segment.onnx` | `1 x 3 x 480 x 640` | `fdbd46dc0a5ae4bb1cb1b32010fa131a43f370043077555f921890bcd2880494` |
| `yolo26n_segment_raw` | `Yolo26Segment.onnx` | `1 x 3 x 640 x 640` | `52f066c5e682fddf899466d975d4f7c0bc727fb49424072fc7d986f9d09b8ed8` |
| `yolo26n_e2e_segment` | `E2ESegment.onnx` | `1 x 3 x 640 x 640` | `78cb9fff568e1a006523a54ae96c1f43ea6b376e93dd637e10eede16128335f4` |
| `lraspp_voc_segment` | `SemanticSegment.onnx` | `1 x 3 x 512 x 512` | `70e2c6d571872208589c43a491fde37fbf73376481343119a5c67dfd55df8c6f` |

Static batch-4 fixtures are exported directly from official [Ultralytics weights](https://github.com/ultralytics/assets/releases/tag/v8.3.0),
using local `assets/test/scripts/export_batch4.py`: Ultralytics 8.4.8, PyTorch 2.9.1,
ONNX 1.20.1, opset 17, `batch=4`, `imgsz=[480,640]`, `dynamic=False`,
`simplify=False`, FP32. Batch-1 reference models are exported from the same weights.
These are native batched networks. Their template algos end in `_batch4`.
`ImageModelTest.MatchesBatchOneOnStaticBatchModels` compares four images and a partial batch
against individual CPU calls. `MatchesBatchOneOnTensorRtStaticBatchModels` compares batched
and individual calls on the same TensorRT model with CPU and CUDA preprocessing separately;
the CUDA kernel may differ from OpenCV by one uint8 step in rare pixels. TensorRT fixtures are local
`.engine` files beside the ONNX files of both the batch-4 and the batch-1 models, built with
TensorRT 10.14.1, FP32, `--noTF32`; plans load only where they were built, so tests and
benchmarks skip them elsewhere.

| Algo | ONNX file in `assets/test/onnx/` | Input NCHW | Outputs | Batch-1 SHA256 | Batch-4 SHA256 |
| --- | --- | --- | --- | --- | --- |
| `yolov8n_classify_batch4` | `GeneralClassifyBatch4.onnx` | `4 x 3 x 480 x 640` | `4 x 1000` | `12b84b2d5f63b9e5a7d8122643d61051810dffad6642bee875c4cd4f24d79a87` | `7d6383b06d8574a40a437b0d92c2a431fa1e1dc41ec24f89af1f712f0a3907fa` |
| `yolov8n_detect_batch4` | `GeneralDetectBatch4.onnx` | `4 x 3 x 480 x 640` | `4 x 84 x 6300` | `a8ede0ffb077b655aa2d1be3044298f4374d1d278d84f6d87ab2cd404895d071` | `2a186204dc8f28e1cf2e77441193661fdf2ae3071f63c969c6cc1cd8fcadfe2c` |
| `yolov8n_pose_batch4` | `GeneralYoloPoseBatch4.onnx` | `4 x 3 x 480 x 640` | `4 x 56 x 6300` | `de42e8e71ff022fe44287e6723fbf6429d1fee84084e63fd19630e1d6496785d` | `5673504bafdac64d9f90775c061011319625fa6d45dc084990e6460d728ee9ab` |
| `yolov8n_segment_batch4` | `GeneralYoloSegmentBatch4.onnx` | `4 x 3 x 480 x 640` | `4 x 116 x 6300; 4 x 32 x 120 x 160` | `1f05bc76e1a542931f9a4b0c21f9d281ff4a8136a3cf8d03869996b0b9a5d124` | `9baa669e562fd20ebbf74744d185dda7777b52dee02c81ba77c4c871c238995c` |

Additional OCR fixtures use official PaddlePaddle ONNX weights. The `ppocrv6_tiny_det`,
`ppocrv6_tiny_rec` and `ppocrv6_small_rec` template algos run through `OcrModel`.

| ONNX file in `assets/test/onnx/` | Input NCHW | Output | SHA256 |
| --- | --- | --- | --- |
| `OcrDetection.onnx` | `1 x 3 x 640 x 640` | `1 x 1 x 640 x 640` | `7373329bed2ea0b66c173278112a6dab68a5f9f244723a302e4b1a674f9435c9` |
| `OcrRecognition.onnx` | `1 x 3 x 48 x 320` | `1 x 40 x 6906` | `71b480e58fa36017c5e542b1157f659cc58da1233d64c716b7a26240ed2c1f31` |
| `OcrRecognitionSmall.onnx` | `1 x 3 x 48 x 320` | `1 x 40 x 18710` | `e01ba7ec62993bd6328fd737bc7bf39873d3c8d6f146b6752a0b651bc9929920` |

Sources: [tiny detection](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx),
[tiny recognition](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx),
[small recognition](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx).
Original `inference.onnx` and `inference.yml` files live under
`assets/test/weights/ocr/<model>/`; recognition `characters.txt` files contain the original
dictionary entries, without the CTC blank or appended space. Tiny and small dictionaries differ.
Local `assets/test/scripts/fix_ocr_shapes.py` fixes batch and image dimensions, sets output
dimensions from real inference and runs ONNX shape inference; the files remain FP32 at their
original opsets (14 for detection, 11 for recognition).

ONNX 1.17.0 checker and ONNX Runtime 1.31.0 CPU zero/random-input runs passed on macOS arm64.
All external input/output dimensions are positive integers. Original and fixed model outputs
agree within `rtol=1e-4, atol=1e-4`; maximum absolute errors were 2.09e-7 (det),
2.74e-5 (tiny rec) and 5.07e-6 (small rec). Recognition argmax sequences are identical.
Fixed shapes can enable different fused runtime kernels. These checks establish runnable,
fixed-shape models, not OCR accuracy or compatibility with other backends.
Local `assets/test/ocr_shape_validation.json` records shapes, source hashes and comparison results.

End-to-end OCR was compared with PaddleOCR 3.7.0 (PaddlePaddle 3.3.1, CPU, macOS arm64) on
the local `assets/test/images/ocr_document.png` (896 x 528), with document orientation,
unwarping and textline orientation disabled. With PaddleOCR detection limited to a 640 long
side, matched boxes reach IoU >= 0.9; 28 of 34 reference lines match exactly with
`ppocrv6_tiny_rec` and 29 of 34 with `ppocrv6_small_rec`. Most remaining differences are
spaces or single characters from integer axis-aligned crops versus PaddleOCR's rotated
sub-pixel crops. PaddleOCR widens recognition input beyond 320 for long lines; the fixed
48 x 320 recognizers compress the 737 x 39 footer line and misread it. A 48 x 960 export
fixes that line but loses spaces on short lines, so 320 is kept. At PaddleOCR's default
detection resolution (native size, multiple of 32), 24 of 33 lines match with tiny.

Sources and export steps for the current collection:

- `GeneralDetect`, `GeneralYoloPose`, `GeneralRotated` and `GeneralClassify` are the
  decoded binding exports of the official YOLOv8 weights below (640, 640, 1024 and 224
  inputs). `GeneralYoloSegment` is the default `yolov8n-seg.pt` export at 480 x 640.
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
- `SemanticSegment` uses torchvision 0.29.1 LR-ASPP MobileNetV3 official pretrained
  COCO-with-VOC-labels weights, exported FP32 at 512 x 512, opset 17.
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
- `SemanticSegment` exports torchvision 0.29
  `lraspp_mobilenet_v3_large(weights=COCO_WITH_VOC_LABELS_V1)` at 512 x 512, opset 17.
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

### RKNN on RK3588

`rknn_convert.py` in the local assets/test/scripts/ directory (rknn-toolkit2 2.3.2, matching the board runtime) converts each ONNX test model to
an fp16 `.rknn` beside it, with the model's mean/std built in. Its algo then runs on RKNN with the same
`postprocess`; `preprocess` drops `mean`/`std` and takes `dst_image_format: "inter_rgb"` (`inter_bgr` for
RTMO), and the backend is `{"engine_type": "rknn", "config": {"image_input": true}}`. A raw-head pose model
uses the `yolov8n_face` postprocess with `reg_max: 16` and the default `keypoint_num` of 17.

Against the ORT results of the same models on bus.jpg, airport.jpg and zidane.jpg (librknnrt 2.3.2):

| Types | RKNN result |
| --- | --- |
| `GeneralDetect`, `YoloV8Detect` (raw) | same objects; boxes within 8 px, scores within 0.07 |
| `GeneralYoloSegment` | same objects; mask IoU 0.935 or more |
| `GeneralYoloPose`, `YoloV8Keypoint` (raw, face) | same people and faces (one borderline person fewer for the raw pose heads); keypoints within 7.3 px |
| `GeneralRotated`, `YoloV8Rotated` (raw) | same 11 planes; angles within 2.2 degrees |
| `GeneralClassify` | same top 1 on airport.jpg; bus.jpg's near-flat scores (0.02) differ |
| `SemanticSegment` | same classes; mask IoU 0.956 or more |
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
| `lraspp_voc_segment` (512) | 1.5 | 66.5 | 8.7 | 70.5 |
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
| `ppocrv6_small_rec` (48 x 320) | 0.65 | one call per text line |

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
