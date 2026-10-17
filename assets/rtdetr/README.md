# RT-DETR fixture

Download official [Ultralytics RT-DETR-L weights](https://github.com/ultralytics/assets/releases/download/v8.1.0/rtdetr-l.pt)
and export a static batch-1 ONNX model at 640 x 640, with decoded query output
`[1, 300, 84]`. Place it at `models/rtdetr-l.onnx`; model files are not committed.
Export versions and checksums are recorded in `sources.json`.

`rtdetr.json` uses RGB, direct resize and division by 255. `DetrDetect` maps
normalized boxes back to the source image and selects one class per query without NMS.

Run from the repository root:

```sh
hrt-compress -c -j assets/rtdetr/rtdetr.json
hrt-validate -m assets/rtdetr/rtdetr.json -n detect -e ort -i assets/yolo/images -o build/rtdetr-results.json
```
