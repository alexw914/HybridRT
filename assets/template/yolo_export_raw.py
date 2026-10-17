"""Exports Ultralytics YOLOv8 or YOLO26 detect, pose, OBB or segment with raw per-stride heads for NPUs.

Each head output is [1, box + classes (+ keypoints or mask coefficients), h, w], or box, theta,
classes for OBB, from the one-to-many head that the default (NMS) export decodes; box is 4 * reg_max
distance bins for YOLOv8 and 4 distances for YOLO26. Heads are named head0..head2 so engines order
them by stride; segment models add the mask prototypes as "proto". Decode YOLOv8 with
YoloV8Detect, YoloV8Keypoint, YoloV8Rotated or YoloV8Segment and YOLO26 with Yolo26Detect,
Yolo26Keypoint, Yolo26Rotated or Yolo26Segment, strides [8, 16, 32].

    python yolo_export_raw.py yolo26n-seg.pt yolo26n-seg-raw.onnx 640
"""

import sys
import types

import torch
from ultralytics import YOLO


def main():
    name, out, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
    model = YOLO(name).model.fuse().eval()
    head = model.model[-1]
    modules = head.one2many

    def forward(self, x):
        heads = []
        for i in range(self.nl):
            parts = [modules["box_head"][i](x[i])]
            if modules.get("angle_head") is not None:
                parts += [modules["angle_head"][i](x[i]), modules["cls_head"][i](x[i])]
            else:
                parts.append(modules["cls_head"][i](x[i]))
            if modules.get("mask_head") is not None:
                parts.append(modules["mask_head"][i](x[i]))
            if modules.get("pose_head") is not None:
                features = modules["pose_head"][i](x[i])
                if modules.get("kpts_head") is not None:
                    features = modules["kpts_head"][i](features)
                parts.append(features)
            heads.append(torch.cat(parts, 1))
        if modules.get("mask_head") is not None:
            # YOLOv8 builds prototypes from the finest level, YOLO26 from all of them.
            proto = self.proto(x) if type(self).__name__ == "Segment26" else self.proto(x[0])
            heads.append(proto[0] if isinstance(proto, tuple) else proto)
        return heads

    head.forward = types.MethodType(forward, head)
    image = torch.zeros(1, 3, size, size)
    with torch.no_grad():
        print([tuple(t.shape) for t in model(image)])
        names = ["head0", "head1", "head2"] + (["proto"] if modules.get("mask_head") is not None else [])
        torch.onnx.export(model, image, out, opset_version=17, input_names=["images"], output_names=names,
                          dynamo=False)
    print("exported", out)


if __name__ == "__main__":
    main()
