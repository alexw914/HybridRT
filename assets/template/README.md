# Manifest reference

`hrtc_template.json` lists every manifest field. The first algo shows all preprocess keys
and one backend of each engine type with all of its config keys; each following algo shows
one postprocessor type, to show how its parameters are set. Where a type has a test model, its
algo is named after it (`yolov8n_detect`, `rtmo_s_pose`, ...), uses that model's preprocessing and
is what the tests and benchmarks run; its `data` points at a local fixture under `assets/` (see
[Test models](#test-models)), relative to the repository root. Other models of the same type are
described in this file rather than given algos of their own. The remaining algos are type
references whose model files do not exist. Values are defaults, or
typical YOLO values where the default would not fit. Check a manifest with
`hrt-compress -c -j <manifest>`; the ORT, ncnn and MNN engine tests load the backend configs
of the first algo.

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
| `YoloV8Detect`, `YoloV8Rotated` | `class_num`, `strides` (one per output head), `reg_max` (16), thresholds |
| `YoloV5Detect`, `YoloV5Rotated` | `class_num`, `strides` (one per output head), `anchors` (one list per head: w0, h0, w1, h1, ...), thresholds |
| `GeneralDetect`, `GeneralRotated` | `class_num`, thresholds; decoded `[1, 4 + classes (+ angle), N]` output |
| `E2EDetect` | `class_num`, `confidence_threshold`; Ultralytics end-to-end `[1, N, 6]` output, no NMS |
| `E2ERotated` | `class_num`, `confidence_threshold`; end-to-end OBB `[1, N, 7]` |
| `DetrDetect` | `class_num`, `confidence_threshold` (0.25), `output_format` (`combined`/`separate`), `score_activation` (`none`/`sigmoid`/`softmax`), `background_class` (-1: none), `boxes_output` (0), `scores_output` (1) |
| `GeneralYoloPose` | `class_num`, `keypoint_num`, `keypoint_threshold` (0.5), thresholds |
| `YoloV8Keypoint` | `class_num`, `keypoint_num`, `strides` (one per output head), `reg_max` (16), `keypoint_threshold` (0.5), thresholds |
| `E2EKeypoint` | `class_num`, `keypoint_num`, `keypoint_threshold` (0.5), `confidence_threshold`; end-to-end pose `[1, N, 6 + K x (2 \| 3)]` |
| `RtmoKeypoint` | `confidence_threshold` ([0.15]), `iou_threshold` ([0.65]), `keypoint_threshold` (0.3); one class |
| `GeneralKeypoint` | `general_keypoint_type` (`heatmap`/`heatmap_udp`/`simcc`/`point_with_visible`/`point_channel_first`/`point_channel_last`), `heatmap_refine` (`dark`/`none`), `score_activation` (`none`/`sigmoid`), `keypoint_threshold` (0) |
| `InstanceSegment` | `class_num`, `mask_threshold` (0.5), thresholds |
| `E2ESegment` | `class_num`, `mask_threshold` (0.5), `confidence_threshold`; end-to-end `[1, N, 6 + masks]` plus prototypes |
| `SemanticSegment` | none |
| `GeneralClassify` | `classify_postprocess_type`: `none`, `sigmoid`, `softmax` |

Engines report outputs sorted by name, the same on every backend, so `strides`, `anchors` and
output indices follow that order rather than the model's: a YOLOv8 export with outputs
`output0` (80 x 80), `393` (40 x 40) and `401` (20 x 20) takes `"strides": [16, 32, 8]`.

`GeneralYoloPose` reads the decoded single output `[1, 4 + classes + K x (2 | 3), N]` of the default
Ultralytics export (boxes, sigmoid scores and keypoints already in input pixels). `YoloV8Keypoint`
reads the raw heads, one `[1, 4 x reg_max + classes + K x (2 | 3), h, w]` output per stride, as NPU
exports keep them for quantization: boxes from the distance bins, scores from logits, keypoints at
`((raw * 2 + cell) * stride)` with sigmoid visibility. Both take any `keypoint_num`, e.g. 5 face
landmarks; drawing joins a skeleton only for 17 COCO keypoints.

`DetrDetect` reads batch-1 normalized center-x, center-y, width and height boxes.
It selects the highest-scoring foreground class of each query and does not apply
NMS. The model's export contract, rather than its family name, determines the config:

- `combined` (default): one `[1, N, 4 + scores]` output, boxes first. Ultralytics
  RT-DETR exports with per-class probabilities use `score_activation: "none"`.
- `separate`: `[1, N, 4]` boxes and `[1, N, scores]` scores. `boxes_output` and
  `scores_output` index the engine's reported output order.
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
`GeneralDetect`, `GeneralYoloPose`, `InstanceSegment` and `GeneralRotated` decode for YOLO26 as for
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
the image; `rtmo_fix_outputs.py` next to this file (needs `onnx` and `numpy`) replaces that NMS
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

Only `assets/template/` is committed. Test models, images and their outputs stay in local,
untracked directories under `assets/`, at the paths the template algos name; tests and
benchmarks that need them skip when they are missing. `hybridrt_example_image_model` reads only the
chosen algo (`-m assets/template/hrtc_template.json -a yolov8n_detect`). `hrt-validate`,
`hrt-model-bench` and `hrt-compress` read every model file of a manifest, so copy one algo into its
own manifest first; from the repository root:

```sh
python -c "import json, sys; m = json.load(open('assets/template/hrtc_template.json')); m['algos'] = [a for a in m['algos'] if a['name'] == sys.argv[1]]; json.dump(m, open(sys.argv[1] + '.json', 'w'))" yolov8n_detect
hrt-validate -m yolov8n_detect.json -n yolov8n_detect -e ort -i assets/yolo/images -r drawn/
```

| Algo | Local file | Source | sha256 |
| --- | --- | --- | --- |
| images | `assets/yolo/images/bus.jpg`, `airport.jpg` | [bus.jpg](https://github.com/ultralytics/assets/raw/refs/heads/main/im/bus.jpg), [airport.jpg](https://github.com/ultralytics/assets/raw/refs/heads/main/im/airport.jpg) | `c02019c4...`, `7daff53a...` |
| `yolov8n_detect` | `assets/yolo/models/yolov8n.onnx` | [unileon-robotics/YOLO8-ONNX](https://huggingface.co/unileon-robotics/YOLO8-ONNX/resolve/main/yolov8n.onnx) | `fa2c856b...` |
| `yolov8n_segment` | `assets/yolo/models/yolov8n-seg.onnx` | same repository, `yolov8n-seg.onnx` | `9373c971...` |
| `yolov8n_pose` | `assets/yolo/models/yolov8n-pose.onnx` | same repository, `yolov8n-pose.onnx` | `6fc546ff...` |
| `yolov8n_obb` | `assets/yolo/models/yolov8n-obb.onnx` | same repository, `yolov8n-obb.onnx` | `d87f86b2...` |
| `yolov8n_classify` | `assets/yolo/models/yolov8n-cls.onnx` | same repository, `yolov8n-cls.onnx` | `29e1963b...` |
| `yolov8n_obb_raw`; also `yolov8n-raw.onnx`, `yolov8n-pose-raw.onnx` | `assets/yolo/models/yolov8n-obb-raw.onnx` (1024), `yolov8n-raw.onnx`, `yolov8n-pose-raw.onnx` (640) | `yolov8_export_raw.py` next to this file from `yolov8n{-obb,,-pose}.pt` (Ultralytics 8.4.178) | `2a5de440...`, `e2a57b77...`, `d87b2769...` |
| `lraspp_voc_segment` | `assets/semantic/models/lraspp_mobilenet_v3_voc_512.onnx` | torchvision 0.29 `lraspp_mobilenet_v3_large(weights=COCO_WITH_VOC_LABELS_V1)`, `out` exported at 512 x 512, opset 17 | `e610481d...` |
| `rtdetr_l_detect` | `assets/rtdetr/models/rtdetr-l.onnx` | [rtdetr-l.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/rtdetr-l.pt), Ultralytics 8.1.4 `export(format="onnx", imgsz=640, opset=17)` | `8090814f...` |
| `yolo26n_e2e_detect` | `assets/e2e/models/yolo26n.onnx`; also `yolov10n.onnx`, `yolov8n-nms.onnx` | Ultralytics 8.4.178 `export(format="onnx", imgsz=640, opset=17, simplify=True, nms=False)`; `nms=True` for YOLOv8 | `08727943...`, `1fb68e28...`, `979da5db...` |
| `yolo26n_e2e_pose`, `_segment`, `_obb` | `assets/e2e/models/yolo26n-pose.onnx`, `-seg.onnx`, `-obb.onnx`; also `yolov8n-{pose,seg,obb}-nms.onnx` | as above from `yolo26n-{pose,seg,obb}.pt` and `yolov8n-{pose,seg,obb}.pt`; OBB at `imgsz=1024` | `ecd85eb3...`, `10106049...`, `0dc04234...`, `84dbdb8f...`, `ac2d7015...`, `8cea4b4c...` |
| `yolov8n_face` | `assets/face/models/yolov8n-face.onnx`; image `assets/face/images/zidane.jpg` | `yolov8n-face-derronqi.onnx` of [clibdev/yolov8-face 1.0.0](https://github.com/clibdev/yolov8-face/releases/tag/1.0.0); [zidane.jpg](https://github.com/ultralytics/assets/raw/refs/heads/main/im/zidane.jpg) | `af09683b...`, `16d73869...` |
| `person_head_detect`, `fire_detect`, `person_heatmap_keypoint` | `assets/rknn/models/yolov8_person_head_1280_int8.rknn`, `yolov8_fire_1280_int8.rknn`, `person_heatmap_keypoint_256x256_int8.rknn` | private RKNN models (RK3588, int8, mean 0 / std 255 built in); use your own with the same outputs | `73745b16...`, `91290dc2...`, `7a0f60b4...` |
| `rtmpose_m_pose` | `assets/rtmpose/models/rtmpose-m_simcc-body7_256x192.onnx` | `end2end.onnx` of the [RTMPose-m body7 SDK](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip) | `5c0a4bf6...` |
| `rtmo_s_pose` | `assets/rtmo/models/rtmo-s_body7_640x640.onnx` | `end2end.onnx` of the [RTMO-s body7 SDK](https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip) through `rtmo_fix_outputs.py` (300 candidates) | `86aefe58...` |

The `yolov8n_*` models have a 480 x 640 input; `yolov8n_classify` resizes directly instead of the
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
- Raw YOLO26 pose, segment and OBB exports (no `nms`) through `GeneralYoloPose`, `InstanceSegment` and
  `GeneralRotated`: the same objects and scores as Ultralytics `predict`, except one borderline
  airport.jpg segment box from the same rounding. These exports are not template algos.
- `rtmo_s_pose`: identical boxes, scores and visible keypoints to the original model's in-graph NMS.
  300 candidates cost 0.22 s against 0.09 s for the original on an x86 CPU (100: 0.12 s), as each
  candidate decodes its keypoints; bus.jpg has 63 candidates above 0.15 for 5 people. Its graph
  keeps a TopK; RKNN and TensorRT conversion has not been tried.
- `yolov8n_face`: boxes, scores, landmarks and visibility identical to a Python decode of the
  same heads on zidane.jpg (2 faces, 5 landmarks each).
- `heatmap_udp`: keypoints identical to mmpose's `refine_keypoints_dark_udp` and `UDPHeatmap`
  scaling on constructed 48 x 64 maps, edge peaks included.
- RKNN algos on the RK3588 board (librknnrt 2.3.2): `person_head_detect` finds 4 people and 3
  heads on bus.jpg; `person_heatmap_keypoint` after it runs on each person box. Its heatmaps are
  logits, so DARK keeps the argmax and keypoints scoring 0 or less are hidden; neighbors inside
  the square crop can take a narrow box's keypoints. `fire_detect` runs but its accuracy is
  unchecked for lack of a fire image.
- `rtmpose_m_pose` after `yolov8n_detect`: visible keypoints within 5 px of mmpose's affine crop
  and SimCC decoding, visibility 17/17 for all 4 people.

### RKNN on RK3588

`rknn_convert.py` next to this file (rknn-toolkit2 2.3.2, matching the board runtime) converts each ONNX test model to
an fp16 `.rknn` beside it, with the model's mean/std built in. Its algo then runs on RKNN with the same
`postprocess`; `preprocess` drops `mean`/`std` and takes `dst_image_format: "inter_rgb"` (`inter_bgr` for
RTMO), and the backend is `{"engine_type": "rknn", "config": {"image_input": true}}`. A raw-head pose model
uses the `yolov8n_face` postprocess with `keypoint_num: 17` and `strides: [8, 16, 32]`.

Against the ORT results of the same models on bus.jpg, airport.jpg and zidane.jpg (librknnrt 2.3.2):

| Types | RKNN result |
| --- | --- |
| `GeneralDetect`, `YoloV8Detect` (raw) | same objects; boxes within 8 px, scores within 0.07 |
| `InstanceSegment` | same objects; mask IoU 0.935 or more |
| `GeneralYoloPose`, `YoloV8Keypoint` (raw, face) | same people and faces (one borderline person fewer for the raw pose heads); keypoints within 7.3 px |
| `GeneralRotated`, `YoloV8Rotated` (raw) | same 11 planes; angles within 2.2 degrees |
| `GeneralClassify` | same top 1 on airport.jpg; bus.jpg's near-flat scores (0.02) differ |
| `SemanticSegment` | same classes; mask IoU 0.956 or more |
| `GeneralKeypoint` (`simcc`) after `yolov8n_detect` | same 4 people, visibility 17/17; keypoints within 7.3 px |
| `E2EDetect`, `E2EKeypoint`, `E2ESegment`, `E2ERotated` | YOLO26 exports convert but return no or wrong rows: the runtime reports GatherElements indexes out of range in the in-graph top-k; YOLOv8 `nms=True` exports do not convert (dynamic NonMaxSuppression) |
| `DetrDetect`, `RtmoKeypoint` | RT-DETR-L and the fixed RTMO graph do not convert (shape errors in the toolkit's graph rewrites, also after onnxslim) |

Use raw-head or decoded exports on RKNN. Milliseconds per call on one thread of the board (fp16, RGA
preprocessing from a host image, 100 runs):

| Algo (input) | Preprocess | Inference | Postprocess | `process_sync` |
| --- | --- | --- | --- | --- |
| `yolov8n_detect` (640 x 480) | 1.8 | 38.5 | 0.7 | 43.0 |
| raw detect heads (640) | 1.8 | 43.9 | 1.7 | 51.9 |
| `yolov8n_segment` (640 x 480) | 1.8 | 48.6 | 11.4 | 59.8 |
| `yolov8n_pose` (640 x 480) | 1.8 | 45.1 | 0.05 | 44.8 |
| raw pose heads (640) | 1.9 | 45.3 | 0.07 | 50.0 |
| `yolov8n_obb` (640 x 480) | 1.6 | 38.2 | 0.5 | 39.0 |
| `yolov8n_obb_raw` (1024) | 2.3 | 108.7 | 1.6 | 111.6 |
| `yolov8n_classify` (640 x 480) | 1.6 | 21.3 | 0.0 | 22.3 |
| `lraspp_voc_segment` (512) | 1.5 | 66.5 | 56.9 | 117.3 |
| `yolov8n_face` (640) | 1.7 | 45.4 | 0.06 | 47.3 |
| `rtmpose_m_pose` (192 x 256 crop) | 0.8 | 18.6 | 0.06 | 20.4 |

A dma-buffer source saves 0.4 to 0.8 ms of preprocessing. int8 models, not measured here, cut the
inference time. `SemanticSegment` spends its time on the per-class source-resolution masks.
