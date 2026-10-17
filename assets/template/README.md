# Manifest reference

`hrtc_template.json` contains one representative model for each postprocessor type. All 23
algos use ONNX Runtime and `assets/test/onnx/<type>.onnx`, relative to the repository
root. The first algo shows every preprocess key. Engine-specific `config` keys are
collected in `hrtc_device_config.json`, using the same raw YOLOv8 detector as an example;
`hrtc_template.json` omits them and uses engine defaults. Tests and benchmarks load these same algos. Model files
and images stay local and untracked; only this directory and `assets/PERF.md` are committed.
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
| `resize_type` | `normal` | `normal` (stretch), `keep_ratio` (top-left), `letter_box` (centered; alias `center_keep_ratio`), `center_crop` (square input, needs `resize_short_size`), `pad_outside` (grows the source rect to the input aspect ratio, scaled by `pad_scale`) |
| `dst_image_format` | `planar_rgb` | `planar_rgb`, `planar_bgr`, `inter_rgb`, `inter_bgr`, `inter_rgba`, `gray`, `nv12`, `nv21`, `yu12`, `yv12`; planar and gray need an NCHW input, interleaved NHWC |
| `normalize_type` | `full` | `full`, `roi` (only the resized region), `none` |
| `mean`, `std` | `[0, 0, 0]`, `[1, 1, 1]` | Per channel: `x = (x - mean) / std` |
| `default_color` | `[0, 0, 0]` | Padding color |
| `resize_short_size` | `-1` | Short side before `center_crop` |
| `pad_scale` | `1.0` | Scale of `pad_outside` |

A cascade passes each detection as `src_rect` to `process_sync`. Only `pad_outside` clips it to
the image; for the other resize types a region leaving the image fails the call, so grow and clip
detections with `hrt::core::pad_scale(box, scale_w, scale_h, image.rect())`. `center_crop` can still
leave a clipped box when `resize_short_size` is below the input size.

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

Operator libraries are optional: one that does not load is skipped with a warning.

Per platform:

| Engine | Preprocess and model inputs |
| --- | --- |
| `rknn` (RK3588) | `image_input: true` (default) feeds uint8 NHWC images and the runtime applies the mean, std and channel swap built into the `.rknn` (`strings model.rknn \| grep "'mean'"` shows them): use `dst_image_format: "inter_rgb"` and keep `mean`/`std` at their defaults. `false` feeds int8 or float32 tensors normalized by `preprocess`. `device: "auto"` uses the RGA. |
| `trt` | `device: "auto"` preprocesses with CUDA (NPP) into the device input. |
| `cann` | `device: "auto"` preprocesses with DVPP. |
| `mnn` | `forward_type: "metal"` needs an SDK built with Metal; on Linux with RGA and a dma-heap `device: "auto"` uses the RGA, as for `ncnn`. |
| `ort` | `execution_provider: "coreml"` needs a CoreML build; models with a dynamic batch need no `input_shapes`, other dynamic dims do. |

## postprocess

`confidence_threshold` and `iou_threshold` take one value for all classes or one per class,
each in [0, 1].

| `type` | Keys |
| --- | --- |
| `YoloV8Detect`, `YoloV8Rotated` | `class_num`, `strides` (optional; inferred from input/head sizes), `reg_max` (optional; inferred from output channels), thresholds |
| `Yolo26Detect`, `Yolo26Rotated` | `class_num`, `strides` (optional; inferred from input/head sizes), thresholds |
| `YoloV5Detect`, `YoloV5Rotated` | `class_num`, `strides` (optional; inferred from input/head sizes), `anchors` (one list per head: w0, h0, w1, h1, ...), thresholds |
| `GeneralDetect`, `GeneralRotated` | `class_num`, thresholds; decoded `[1, 4 + classes (+ angle), N]` output |
| `E2EDetect` | `class_num`, `confidence_threshold`; Ultralytics end-to-end `[1, N, 6]` output, no NMS |
| `E2ERotated` | `class_num`, `confidence_threshold`; end-to-end OBB `[1, N, 7]` |
| `DetrDetect` | `class_num`, `confidence_threshold` (0.25), `output_format` (`combined`/`separate`), `score_activation` (`none`/`sigmoid`/`softmax`), `background_class` (-1: none) |
| `GeneralYoloPose` | `class_num`, `keypoint_num` (optional; inferred when unambiguous), `keypoint_threshold` (0.5), thresholds |
| `YoloV8Keypoint` | at least two of `class_num`, `reg_max`, `keypoint_num`; `strides` (optional; inferred), `keypoint_threshold` (0.5), thresholds |
| `Yolo26Keypoint` | `class_num`, `keypoint_num` (optional; inferred when unambiguous), `strides` (optional; inferred from input/head sizes), `keypoint_threshold` (0.5), thresholds |
| `E2EKeypoint` | `class_num`, `keypoint_num` (optional; inferred when unambiguous), `keypoint_threshold` (0.5), `confidence_threshold`; end-to-end pose `[1, N, 6 + K x (2 \| 3)]` |
| `RtmoKeypoint` | `confidence_threshold` ([0.15]), `iou_threshold` ([0.65]), `keypoint_threshold` (0.3); one class |
| `GeneralKeypoint` | `general_keypoint_type` (`heatmap`/`heatmap_udp`/`simcc`/`point_with_visible`/`point_channel_first`/`point_channel_last`), `heatmap_refine` (`dark`/`none`), `score_activation` (`none`/`sigmoid`), `keypoint_threshold` (0) |
| `GeneralYoloSegment` | `class_num`, `mask_threshold` (0.5), thresholds |
| `YoloV8Segment` | `class_num`, `strides` (optional; inferred from input/head sizes), `reg_max` (optional; inferred from output channels), `mask_threshold` (0.5), thresholds |
| `Yolo26Segment` | `class_num`, `strides` (optional; inferred from input/head sizes), `mask_threshold` (0.5), thresholds |
| `E2ESegment` | `class_num`, `mask_threshold` (0.5), `confidence_threshold`; end-to-end `[1, N, 6 + masks]` plus prototypes |
| `SemanticSegment` | none |
| `GeneralClassify` | `classify_postprocess_type`: `none`, `sigmoid`, `softmax` |

Raw YOLO postprocessors compute each stride during initialization from the input width/height
and the corresponding output grid. Both ratios must be positive and equal. Omit `strides`
(or use `[]`) to infer them; a nonempty list must exactly match the inferred values and count.
Segmentation identifies the prototype output first and rejects ambiguous output structures.
YOLOv8 also infers `reg_max` from the box channels divided by four, after subtracting
classes and task-specific channels (one angle, keypoint coordinates, or mask coefficients).
An explicit `reg_max` must be positive and match every head.
`GeneralYoloPose`, `E2EKeypoint`, `Yolo26Keypoint` and `YoloV8Keypoint` also infer
`keypoint_num` during initialization. An explicit value must be positive and match the
output shape. Keypoints may contain two coordinates or two coordinates plus visibility;
if both interpretations fit, supply `keypoint_num` to disambiguate. Raw YOLOv8 requires at
least two explicitly supplied positive values among
`class_num`, `reg_max` and `keypoint_num`; the omitted parameter is inferred. If all three
are supplied, all are checked. Two supplied values do not resolve every two-versus-three
component ambiguity: initialization still rejects multiple valid partitions, including
ambiguous inferred class counts. The face template explicitly lists all three:
`class_num: 1`, `reg_max: 16`, and `keypoint_num: 5`. Initialization fails if the partition is invalid or ambiguous;
it does not assume three components or a particular bin count. Segmentation likewise
requires an unambiguous prototype/head structure; explicit `reg_max` can constrain that match.
Engines report outputs sorted by name on every backend; `anchors`, output indices and any
explicit `strides` follow that order, excluding the prototype output for segmentation.
For a 640 x 640 input with outputs `output0` (80 x 80), `393` (40 x 40) and `401` (20 x 20),
the inferred strides are `[16, 32, 8]`.

`GeneralYoloPose` reads the decoded single output `[1, 4 + classes + K x (2 | 3), N]` of the default
Ultralytics export (boxes, sigmoid scores and keypoints already in input pixels). `YoloV8Keypoint`
reads the raw heads, one `[1, 4 x reg_max + classes + K x (2 | 3), h, w]` output per stride, as NPU
exports keep them for quantization: boxes from the distance bins, scores from logits, keypoints at
`((raw * 2 + cell) * stride)` with sigmoid visibility. Both take any `keypoint_num`, e.g. 5 face
landmarks; drawing joins a skeleton only for 17 COCO keypoints.

Raw heads come from the one-to-many branch that the default export decodes, so NMS still runs.
YOLOv8 heads (`YoloV8Detect`, `YoloV8Keypoint`, `YoloV8Rotated`, `YoloV8Segment`) carry 4 x `reg_max`
distance bins, keypoints at `(raw * 2 + cell) * stride` and angles `(sigmoid(raw) - 0.25) * pi`. YOLO26
heads (`Yolo26Detect`, `Yolo26Keypoint`, `Yolo26Rotated`, `Yolo26Segment`) carry the four distances in
cells, keypoints at `(raw + cell + 0.5) * stride` and angles in radians. Segment heads add the mask
coefficients after the classes, and the prototypes `[1, masks, mh, mw]` are a further output in any
position.

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
| `E2EKeypoint` | x1, y1, x2, y2, score, class id, then x, y (and visibility) per keypoint |
| `E2ESegment` | x1, y1, x2, y2, score, class id, mask coefficients; a second `[1, masks, mh, mw]` prototype output, in either order |
| `E2ERotated` | center x, center y, width, height, score, class id, angle in radians |

`RtmoKeypoint` decodes one-stage RTMO pose models with fixed outputs, boxes `[1, M, 5]` (x1, y1, x2,
y2, score) and keypoints `[1, M, K, 3]` (x, y, visibility) in network-input pixels, in either output
order. The mmpose export selects instances with NMS inside the model, so its output size depends on
the image; `rtmo_fix_outputs.py` in the local assets/test/scripts/ directory (needs `onnx` and `numpy`) replaces that NMS
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
| `class_id`, `score` | Detection class and score; classify: top 1 of output 0. Semantic segments and `GeneralKeypoint` use score 1. |
| `x`, `y`, `width`, `height` | Box in source pixels, the top-left of the unrotated box; the full image for classify and semantic segments; the cropped source region for `GeneralKeypoint`. |
| `angle` | Rotated boxes; 0 otherwise. |
| `keypoints` | `[x, y, score, visible]` per keypoint, `visible` 0 unlabeled, 1 invisible, 2 visible. |
| `mask` | `{"rect": [x, y, w, h], "rle": [...]}`: row-major run lengths over `rect`, alternating 0 and 1, starting with 0; `null` without a mask. |
| `cls` | Classify, one item per engine output: `output` index, `topk` `[[class, score], ...]` (5), `scores` by class. |

## Test models

All template model files live in `assets/test/onnx/`, with one `<type>.onnx` per
postprocessor and positive, fixed input/output dimensions. Previous model directories and
extra variants were removed. Test resources share `assets/test/`: `images/` contains `bus.jpg`, `airport.jpg` and
`zidane.jpg`; `calib/` contains calibration data; `onnx/` contains the 23 models.
Models and test resources are not committed.

The 23 files passed ONNX checker, ONNX Runtime zero/random-input runs and `hrt-validate`
(45 image runs, no pipeline errors). This verifies shape compatibility and runnable
postprocessing, not detection accuracy. The user-provided `YoloV5Rotated` model has 2 classes;
its anchors, normalization and angle convention remain unconfirmed. Template anchors and
preprocessing allow pipeline checks only, as requested.

| Algo | ONNX file in `assets/test/onnx/` | Input NCHW | SHA256 |
| --- | --- | --- | --- |
| `yolov8_detect` | `YoloV8Detect.onnx` | `1 x 3 x 640 x 640` | `e2a57b77d38a136a3ac07ddd17b81930c4680e0d464ee615488c3a47bfdef303` |
| `yolo26n_detect_raw` | `Yolo26Detect.onnx` | `1 x 3 x 640 x 640` | `2089cfdd67660be0efec7477bbf2f6cabc2261c84bbd82c6a7e895b51bd8c49a` |
| `yolov5_detect` | `YoloV5Detect.onnx` | `1 x 3 x 640 x 640` | `797416533eed5eb4c82098ec221bbe55368ad3c062f9d0217db87f129a78f172` |
| `rtdetr_l_detect` | `DetrDetect.onnx` | `1 x 3 x 640 x 640` | `599b9b7484cb831312f585d4ec4885a95ae10d1e84dada14833f5c570828c9b9` |
| `yolo26n_e2e_detect` | `E2EDetect.onnx` | `1 x 3 x 640 x 640` | `08727943f090f5e8472cae3c08bcbb7b93c927820d3104ef7792f1d31277ec77` |
| `yolov8n_detect` | `GeneralDetect.onnx` | `1 x 3 x 480 x 640` | `fa2c856b0effedce4f49fffd121f438247a64c8f633bf0e6d5e5960e869ccc5a` |
| `yolo26n_e2e_obb` | `E2ERotated.onnx` | `1 x 3 x 1024 x 1024` | `0dc04234b1d82d87b8aebe5c364b4c298de0e2380ce61b74f9b117478e690e41` |
| `yolov8n_obb_raw` | `YoloV8Rotated.onnx` | `1 x 3 x 1024 x 1024` | `2a5de440d72ecb80cc4fbce50c59f6293e4bc89cb1fe8aaee8ca7d9ed3b89566` |
| `yolo26n_obb_raw` | `Yolo26Rotated.onnx` | `1 x 3 x 1024 x 1024` | `003dc108ae1be9b94f7bbf3f154d871385224da96ca15fa0bc07bccb0bc2148f` |
| `yolov5_rotated` | `YoloV5Rotated.onnx` | `1 x 3 x 1280 x 1280` | `afffc7a2813e15e9137648bc7655155827b1cdace2a0999b1919d1196be57dfb` |
| `yolov8n_obb` | `GeneralRotated.onnx` | `1 x 3 x 480 x 640` | `d87f86b2e475deccfd57410fdcb6d2de3886fcab4f6395888ddca4c35a2c12ee` |
| `yolov8n_classify` | `GeneralClassify.onnx` | `1 x 3 x 480 x 640` | `29e1963b77c3d1e38d3c8772ecbce294e48aa3287c3f703c39bc4e8d0b1a0f1c` |
| `yolov8n_pose` | `GeneralYoloPose.onnx` | `1 x 3 x 480 x 640` | `6fc546ff26e38b25e29c047fcacd71d3607705c0b5a590f0bd97492aed459f94` |
| `yolov8n_face` | `YoloV8Keypoint.onnx` | `1 x 3 x 640 x 640` | `af09683be4937ebe23e9da0346cbf46cd937b73ed5fac986e6205e6c7cdd2c25` |
| `yolo26n_pose_raw` | `Yolo26Keypoint.onnx` | `1 x 3 x 640 x 640` | `513e12cd94482f799e2d1ad66fcb9603d7c3eb41ad38863e7ae566374e0eddc8` |
| `yolo26n_e2e_pose` | `E2EKeypoint.onnx` | `1 x 3 x 640 x 640` | `ecd85eb31cf6889ab62600164d27698b881fcb27bd27c548b2dd4a67fb52c814` |
| `rtmo_s_pose` | `RtmoKeypoint.onnx` | `1 x 3 x 640 x 640` | `91d97d8251f7624d4da9cead6b2a8bd6ea36850cfe329122eebeb89b41f108d2` |
| `rtmpose_m_pose` | `GeneralKeypoint.onnx` | `1 x 3 x 256 x 192` | `81e1025f9c938450d2ef88c0fba274fb57b0f9b49711dc7a52fef99c7ea3548d` |
| `yolov8n_segment` | `GeneralYoloSegment.onnx` | `1 x 3 x 480 x 640` | `9373c971598d799b54e6911e5a665b99653eb0be611471c28ae9dc3676a6a995` |
| `yolov8n_segment_raw` | `YoloV8Segment.onnx` | `1 x 3 x 640 x 640` | `a21888dd7bc2e6d478ceb438a2ec520c60e5aa03ab898b1bc65fb3598d2afd83` |
| `yolo26n_segment_raw` | `Yolo26Segment.onnx` | `1 x 3 x 640 x 640` | `85e5ec69cb0a06f82e6a27b4cdb6be36d91e041b3829d75887183f6506b18221` |
| `yolo26n_e2e_segment` | `E2ESegment.onnx` | `1 x 3 x 640 x 640` | `101060498337592b2c671f479001ca725d92049f99118e5b666f4c2273d9f096` |
| `lraspp_voc_segment` | `SemanticSegment.onnx` | `1 x 3 x 512 x 512` | `e610481d6dfedf62285bf7de02c1a67a786dc3abb342a5e5db108b67c89daf6b` |

Sources and export steps:

- Decoded YOLOv8 detection, segmentation, pose, OBB and classification models come from
  [unileon-robotics/YOLO8-ONNX](https://huggingface.co/unileon-robotics/YOLO8-ONNX).
Conversion scripts are local, untracked files in assets/test/scripts/ and are not distributed with the repository.

- Raw YOLOv8/YOLO26 models use `yolo_export_raw.py` with Ultralytics 8.4.178 and the
  corresponding official `.pt` weights. YOLO26 end-to-end models use `export(format="onnx",
  imgsz=640, opset=17, simplify=True, nms=False)`, with 1024 inputs for OBB.
- `YoloV5Detect` uses the
  [official YOLOv5n v7.0 ONNX](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx)
  through `yolov5_export_raw.py`: three raw convolution heads, promoting FP16 values to FP32.
- `YoloV5Rotated` is the user-provided `v0013_obbczs_detection_0808int8.onnx`, copied
  unchanged. Its graph uses INT8 Q/DQ internally and float32 I/O, with outputs
  `[1,24,160,160]`, `[1,24,80,80]`, `[1,24,40,40]`.
- `YoloV8Keypoint` uses `yolov8n-face-derronqi.onnx` from
  [clibdev/yolov8-face 1.0.0](https://github.com/clibdev/yolov8-face/releases/tag/1.0.0).
- `DetrDetect` exports [rtdetr-l.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/rtdetr-l.pt)
  with Ultralytics 8.1.4, `export(format="onnx", imgsz=640, opset=17)`.
- `GeneralKeypoint` uses `end2end.onnx` from the
  [RTMPose-m body7 SDK](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip).
- `RtmoKeypoint` uses `end2end.onnx` from the
  [RTMO-s body7 SDK](https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip),
  then `rtmo_fix_outputs.py` with 300 candidates.
- `SemanticSegment` exports torchvision 0.29
  `lraspp_mobilenet_v3_large(weights=COCO_WITH_VOC_LABELS_V1)` at 512 x 512, opset 17.
- `onnx_fix_shapes.py` fixes RT-DETR, RTMO and RTMPose input dimensions using onnxsim
  without arithmetic optimization or constant folding. Equivalence checks and invariant
  zero/random-input output sizes precede completion of symbolic output annotations.
  Image-dependent NMS outputs must first be replaced by a fixed-output export.

Recreate transformed files from downloaded source exports, outside the model collection:

```sh
python assets/test/scripts/yolov5_export_raw.py yolov5n.onnx assets/test/onnx/YoloV5Detect.onnx
python assets/test/scripts/rtmo_fix_outputs.py rtmo-end2end.onnx rtmo-fixed.onnx --candidates 300
python assets/test/scripts/onnx_fix_shapes.py rtdetr-l.onnx assets/test/onnx/DetrDetect.onnx --input images=1,3,640,640
python assets/test/scripts/onnx_fix_shapes.py rtmo-fixed.onnx assets/test/onnx/RtmoKeypoint.onnx --input input=1,3,640,640
python assets/test/scripts/onnx_fix_shapes.py rtmpose-end2end.onnx assets/test/onnx/GeneralKeypoint.onnx --input input=1,3,256,192
```

Run any template algo directly, for example:

```sh
hrt-validate -m assets/template/hrtc_template.json -n yolov8n_detect -e ort -i assets/test/images -o results.json
```

The decoded YOLOv8 detection, segment, pose, OBB and classification models have a 480 x 640 input; `yolov8n_classify` resizes directly instead of the
square center crop, so it is a pipeline check, not an accuracy reference. `rtmpose_m_pose` runs on
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
uses the `yolov8n_face` postprocess with `reg_max: 16`; its keypoint count is inferred.

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
inference time. Optimization history and measurements are in `assets/PERF.md`.
