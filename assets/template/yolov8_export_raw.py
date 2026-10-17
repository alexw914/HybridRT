"""Exports YOLOv8 detect, pose or OBB with raw per-stride heads for NPUs.

Each output is [1, 4 * reg_max + classes (+ keypoints), h, w], or bins, theta,
classes for OBB, named head0..head2 so engines order them by stride. Decode
with YoloV8Detect, YoloV8Keypoint or YoloV8Rotated, strides [8, 16, 32].

    python yolov8_export_raw.py yolov8n-pose.pt yolov8n-pose-raw.onnx 640
"""
import sys
import types

import torch
from ultralytics import YOLO
name, out, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
model = YOLO(name).model.fuse().eval()
head = model.model[-1]
kind = type(head).__name__
def forward(self, x):
    heads = []
    for i in range(self.nl):
        parts = [self.cv2[i](x[i])]
        if kind == "OBB":
            parts += [self.cv4[i](x[i]), self.cv3[i](x[i])]   # bins, theta, classes
        else:
            parts.append(self.cv3[i](x[i]))                    # bins, classes
            if kind == "Pose":
                parts.append(self.cv4[i](x[i]))                # keypoints x, y, v
        heads.append(torch.cat(parts, 1))
    return heads
head.forward = types.MethodType(forward, head)
im = torch.zeros(1, 3, size, size)
with torch.no_grad():
    print([tuple(t.shape) for t in model(im)])
    torch.onnx.export(model, im, out, opset_version=17, input_names=["images"], output_names=["head0", "head1", "head2"], dynamo=False)
print("exported", out)
