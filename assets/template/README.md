# Manifest reference

`hrtc_template.json` lists every manifest field. The first algo shows all preprocess keys
and one backend of each engine type with all of its config keys; each following algo shows
one postprocessor type. Values are defaults, or typical YOLO values where the default would not fit.
It is a reference, not a package: its model files do not exist. Check a manifest with
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

## Engine config

| Engine | Keys (all optional) |
| --- | --- |
| `ort` | `intra_op_num_threads`, `inter_op_num_threads` (0: ONNX Runtime's default); `execution_provider`: `cpu`, `coreml` (falls back to the CPU without CoreML); `provider_options` (CoreML only): `ModelFormat` `NeuralNetwork`/`MLProgram`, `MLComputeUnits` `ALL`/`CPUOnly`/`CPUAndGPU`, `RequireStaticInputShapes` and `EnableOnSubgraphs` `"0"`/`"1"`; `custom_op_libraries`: paths |
| `ncnn` | `num_threads` (4) |
| `mnn` | `forward_type`: `cpu`, `opencl`, `metal`, `vulkan`, `cuda`, `nn`, `auto`; `num_thread` (4); `cache_file`, `cache_dir`: OpenCL tuning cache |
| `trt` | `plugin_libraries`: paths, loaded before the engine is deserialized |
| `rknn` | `image_input` (true: uint8 inputs); `core_mask`: `RKNN_NPU_CORE_AUTO`, `RKNN_NPU_CORE_0`, `RKNN_NPU_CORE_1`, `RKNN_NPU_CORE_2`, `RKNN_NPU_CORE_0_1`, `RKNN_NPU_CORE_0_1_2`; `custom_op_libraries`: paths |
| `cann` | none |

Operator libraries are optional: one that does not load is skipped with a warning.

## postprocess

`confidence_threshold` and `iou_threshold` take one value for all classes or one per class,
each in [0, 1].

| `type` | Keys |
| --- | --- |
| `YoloV8Detect`, `YoloV8Rotated` | `class_num`, `strides` (one per output head), `reg_max` (16), thresholds |
| `YoloV5Detect`, `YoloV5Rotated` | `class_num`, `head_num`, `anchors_per_head`, `strides_per_head`, `anchors` (per head: w0, h0, w1, h1, ...), thresholds |
| `GeneralDetect`, `GeneralRotated` | `class_num`, thresholds; decoded `[1, 4 + classes (+ angle), N]` output |
| `YoloKeypoint` | `class_num`, `keypoint_num`, `keypoint_threshold` (0.5), thresholds |
| `GeneralKeypoint` | `general_keypoint_type`: `heatmap`, `point_with_visible`, `point_channel_first`, `point_channel_last` |
| `InstanceSegment` | `class_num`, `mask_threshold` (0.5), thresholds |
| `SemanticSegment` | none |
| `GeneralClassify` | `classify_postprocess_type`: `none`, `sigmoid`, `softmax` |
