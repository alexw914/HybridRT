# -----------------------------------------------------------------------------
# HybridRT | C++20 Model Inference Framework
# File: bindings/python/examples/run_models.py
# Description: Runs one algo per task and OCR from an .hrtc package or a JSON
#              manifest on the test images and prints the results.
# -----------------------------------------------------------------------------

import argparse
from pathlib import Path

import cv2

import hybridrt as hrt

CASES = [  # (algo, image): one per task.
    ("yolov8n_detect", "bus.jpg"),
    ("yolov8n_obb", "airport.jpg"),
    ("yolov8n_pose", "bus.jpg"),
    ("yolov8n_segment", "bus.jpg"),
    ("yolov8n_classify", "bus.jpg"),
]
OCR = ("ppocrv6_tiny_det", "ppocrv6_tiny_rec", "ocr_document.jpg")


def print_result(result):
    if isinstance(result, hrt.ClassifyResult):
        scores = result.scores[0]
        top = scores.argsort()[::-1][:5]
        print("  top-5:", ", ".join(f"class {i} {scores[i]:.3f}" for i in top))
        return
    for i, (box, score, class_id) in enumerate(zip(result.boxes, result.scores, result.class_ids)):
        line = f"  class {class_id} score {score:.3f} box {box.round(1).tolist()}"
        if isinstance(result, hrt.KeypointResult):
            line += f" keypoints {len(result.keypoints[i])}"
        print(line)
    if isinstance(result, hrt.SegmentResult):
        print(f"  masks: {len(result.masks)}")


def main():
    parser = argparse.ArgumentParser(description="Run one algo per task and OCR")
    parser.add_argument("-m", "--model", required=True, help=".hrtc package or .json manifest with the algos")
    parser.add_argument("-d", "--images", default="assets/test/images", help="Directory of the test images")
    args = parser.parse_args()

    with hrt.Package(args.model) as package:
        algos = set(package.algos)
        for algo, name in CASES:
            image = cv2.imread(str(Path(args.images) / name), cv2.IMREAD_COLOR)
            if algo not in algos or image is None:
                print(f"{algo}: skipped, needs the algo and {name}")
                continue
            with package.model(algo) as model:
                result = model.process_batch([image])[0]
            print(f"{algo} ({model.task.name}) on {name}: {len(result)} results")
            print_result(result)

        detection, recognition, name = OCR
        image = cv2.imread(str(Path(args.images) / name), cv2.IMREAD_COLOR)
        if detection not in algos or recognition not in algos or image is None:
            print(f"ocr: skipped, needs {detection}, {recognition} and {name}")
            return
        with hrt.OcrModel(package, detection, recognition) as model:
            texts = model(image)
        print(f"ocr ({detection} + {recognition}) on {name}: {len(texts)} texts")
        for item in texts:
            print(f"  {item.text!r} rect {item.rect.round(1).tolist()} score {item.recognition_score:.3f}")


if __name__ == "__main__":
    main()
