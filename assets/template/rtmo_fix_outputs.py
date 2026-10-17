"""Replaces the NMS of an mmpose RTMO end2end.onnx with its top candidates.

The exported model selects instances with NonMaxSuppression, so its outputs
depend on the image. This keeps the `candidates` highest-scoring pre-NMS
anchors instead, giving fixed outputs dets [1, M, 5] and keypoints
[1, M, K, 3]; RtmoKeypoint applies the score threshold and NMS.

    python assets/template/rtmo_fix_outputs.py end2end.onnx \
        assets/rtmo/models/rtmo-s_body7_640x640.onnx [--candidates 300]
"""

import argparse

import numpy as np
import onnx
from onnx import helper, numpy_helper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--candidates", type=int, default=300)
    args = parser.parse_args()

    model = onnx.load(args.source)
    graph = model.graph
    nms = [node for node in graph.node if node.op_type == "NonMaxSuppression"]
    if len(nms) != 1:
        raise SystemExit(f"expected one NonMaxSuppression, found {len(nms)}")
    nms = nms[0]
    # The candidates feeding NMS come from a TopK over 2000 anchors, best first.
    selected = np.zeros((args.candidates, 3), np.int64)
    selected[:, 2] = np.arange(args.candidates)
    constant = helper.make_node(
        "Constant", [], [nms.output[0]], value=numpy_helper.from_array(selected, nms.output[0] + "_fixed"))
    index = list(graph.node).index(nms)
    graph.node.remove(nms)
    graph.node.insert(index, constant)

    # The final TopK keeps at most max_det (50) of the NMS survivors; keep all.
    consts = {node.output[0]: node for node in graph.node if node.op_type == "Constant"}
    topks = [node for node in graph.node if node.op_type == "TopK"]
    producers = {out: node for node in graph.node for out in node.output}
    changed = 0
    for topk in topks[1:]:
        stack = [topk.input[1]]
        seen = set()
        while stack:
            name = stack.pop()
            node = producers.get(name)
            if node is None or name in seen:
                continue
            seen.add(name)
            if node.op_type == "Where" and node.input[1] in consts:
                value = numpy_helper.to_array(consts[node.input[1]].attribute[0].t)
                if value.size == 1 and int(value) == 50:
                    consts[node.input[1]].attribute[0].t.CopyFrom(
                        numpy_helper.from_array(np.array(args.candidates, value.dtype), node.input[1]))
                    changed += 1
            stack.extend(node.input)
    if changed != 1:
        raise SystemExit(f"expected one max_det limit of 50, found {changed}")
    onnx.checker.check_model(model)
    onnx.save(model, args.target)


if __name__ == "__main__":
    main()
