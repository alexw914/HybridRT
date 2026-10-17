"""Exports Ultralytics YOLOv8 or YOLO26 detect, pose or OBB with raw per-stride heads for NPUs.

Each output is [1, 4 * reg_max + classes (+ keypoints), h, w], or bins, theta,
classes for OBB, from the one-to-many head that the default (NMS) export
decodes, named head0..head2 so engines order them by stride. Decode with
YoloV8Detect, YoloV8Keypoint or YoloV8Rotated, strides [8, 16, 32]; YOLO26
takes reg_max 1 and head_version "yolo26".

    python yolo_export_raw.py yolo26n-pose.pt yolo26n-pose-raw.onnx 640
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
            if modules.get("pose_head") is not None:
                features = modules["pose_head"][i](x[i])
                if modules.get("kpts_head") is not None:
                    features = modules["kpts_head"][i](features)
                parts.append(features)
            heads.append(torch.cat(parts, 1))
        return heads

    head.forward = types.MethodType(forward, head)
    image = torch.zeros(1, 3, size, size)
    with torch.no_grad():
        print([tuple(t.shape) for t in model(image)])
        torch.onnx.export(model, image, out, opset_version=17, input_names=["images"],
                          output_names=["head0", "head1", "head2"], dynamo=False)
    print("exported", out)


if __name__ == "__main__":
    main()
