# YOLO task fixtures

Five YOLOv8 nano FP32 ONNX models cover the standard decoded exports supported by HybridRT. This is task coverage, not every YOLO generation or size. Download URLs, checksums and inspected tensor shapes are in `sources.json`.

| Algo | Postprocessor | Classes | Input |
| --- | --- | --- | --- |
| detect | GeneralDetect | 80 | 1 x 3 x 480 x 640 |
| segment | InstanceSegment | 80 | 1 x 3 x 480 x 640 |
| pose | YoloKeypoint | 1, 17 keypoints | 1 x 3 x 480 x 640 |
| obb | GeneralRotated | 15 | 1 x 3 x 480 x 640 |
| classify | GeneralClassify | 1000 | 1 x 3 x 480 x 640 |

The models are in `models/` and the test images in `images/`. `yolo.json` packs them for the ONNX Runtime CPU provider and `yolo_coreml.json` for its CoreML provider (`"execution_provider": "coreml"`, which falls back to the CPU on builds without CoreML); otherwise the two manifests are identical. They only set values that differ from the defaults, so preprocessing runs where the engine input lives (the CPU here). Detection, segmentation, pose and OBB use RGB, letterbox padding 114 and division by 255. Classification uses RGB, direct resize and division by 255; the exported model already contains Softmax.

The downloaded files have rectangular 480 x 640 inputs despite the model page listing 640 x 640. Classification therefore uses direct resize for a pipeline smoke test. It does not reproduce the standard square center-crop classification pipeline and must not be used to establish classification accuracy. These fixtures do not cover the raw multi-head YoloV5Detect/YoloV8Detect or their rotated variants.

Run from the repository root; the manifests' model paths are relative to it:

```sh
build/tools/hrt-compress -j assets/yolo/yolo.json -c
build/tools/hrt-compress -j assets/yolo/yolo.json -o assets/yolo/yolo.hrtc
build/tools/hrt-compress -j assets/yolo/yolo_coreml.json -o assets/yolo/yolo_coreml.hrtc
build/debug/examples/hybridrt_example_image_model -m assets/yolo/yolo.hrtc -i assets/yolo/images/bus.jpg -a detect -o assets/yolo/result/detect_cpu.jpg
```

Use `segment`, `pose` or `classify` with bus.jpg. Use `obb` with images/airport.jpg. Results and execution logs are in `result/`.

Local validation: all five manifest configurations passed hrt-compress checks and all five models completed inference using the existing Debug example executable. The bus image produced 5 detections, 4 instance masks and 3 poses; the airport image produced 11 oriented boxes. Classification returned 1000 scores (top class 879, score approximately 0.022). Detection, mask, pose and OBB drawings were visually inspected. No comparison against an independent reference implementation or labelled ground truth was performed; other devices and inference backends were not checked.

Models are third-party Ultralytics-derived assets; review their upstream license before redistribution.

Latest run (2026-10-11, Release, ORT CPU and CoreML): engine speed and per-task results are in [result/README.md](result/README.md).
