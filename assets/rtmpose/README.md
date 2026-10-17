# RTMPose fixture

Download the official [RTMPose-m body7 ONNX SDK](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip)
and place its `end2end.onnx` at `models/rtmpose-m_simcc-body7_256x192.onnx`; the person
detector is `assets/yolo/models/yolov8n.onnx` (see `assets/yolo/sources.json`). Model files
are not committed; checksums are recorded in `sources.json`.

The model has a dynamic batch, fixed to 1 with `input_shapes`, and two SimCC outputs, x
`[1, 17, 384]` and y `[1, 17, 512]` (split ratio 2). `rtmpose.json` follows its mmpose
pipeline: the box grown by 1.25 to the 192 x 256 aspect, RGB and ImageNet mean/std;
`GeneralKeypoint` with `simcc` and the mmpose drawing threshold 0.3.

`pose` runs on one person box per call; run `detect` first and pass each person (class 0)
as `src_rect` to `ImageModel::process_sync`. Check the configs from the repository root:

```sh
hrt-compress -c -j assets/rtmpose/rtmpose.json
```
