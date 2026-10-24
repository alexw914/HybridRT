// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/yolo_heads.cpp
// Description: Implements validation of YOLO output heads and inference of their strides.
// -----------------------------------------------------------------------------

#include "include/postprocess/yolo_heads.hpp"

#include <kitzoo/log/logger.hpp>

#include "include/postprocess/common.hpp"

namespace hrt::postprocess {

auto resolve_yolov5_heads(YoloV5Param& param, const engine::Engine& engine, std::size_t extra_channels) -> bool {
  if (!expand_per_class(param.confidence_threshold, param.class_num, "confidence_threshold") ||
      !expand_per_class(param.iou_threshold, param.class_num, "iou_threshold"))
    return false;
  if (!is_4d_outputs(engine, "yolov5"))
    return false;
  const auto input = *engine.input_size();
  const auto& outputs = engine.output_protos();
  KZ_CHECK_RETURN(outputs.size() == param.anchors.size(), false,
                  "yolov5: needs one anchor list per head, got {} outputs", outputs.size());
  Vector<core::Point2f> inferred;
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto channels = head_channels(outputs[i], engine.proto().output_layout);
    const auto anchors = param.anchors[i].size() / 2;
    if (anchors == 0 || param.anchors[i].size() % 2 != 0 ||
        static_cast<std::size_t>(channels) != (param.class_num + 5 + extra_channels) * anchors) {
      KZ_LOG_ERROR("yolov5: output {} does not match the anchors or class_num", i);
      return false;
    }
    inferred.push_back(head_stride(outputs[i], engine.proto().output_layout, input));
  }
  param.strides = std::move(inferred);
  return true;
}

auto resolve_yolov8_heads(YoloV8Param& param, const engine::Engine& engine, std::size_t extra_channels) -> bool {
  if (!expand_per_class(param.confidence_threshold, param.class_num, "confidence_threshold") ||
      !expand_per_class(param.iou_threshold, param.class_num, "iou_threshold"))
    return false;
  if (!is_4d_outputs(engine, "yolov8"))
    return false;
  const auto input = *engine.input_size();
  const auto& outputs = engine.output_protos();
  const auto channels = head_channels(outputs.front(), engine.proto().output_layout) -
                        static_cast<std::int64_t>(param.class_num) - static_cast<std::int64_t>(extra_channels);
  KZ_CHECK_RETURN(channels > 0 && channels % 4 == 0, false, "yolov8: invalid channels for reg_max");
  const auto inferred_bins = static_cast<std::size_t>(channels / 4);
  KZ_CHECK_RETURN(param.reg_max == 0 || param.reg_max == inferred_bins, false,
                  "yolov8: configured reg_max does not match output channels");
  param.reg_max = inferred_bins;
  Vector<core::Point2f> inferred;
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    if (static_cast<std::size_t>(head_channels(outputs[i], engine.proto().output_layout)) !=
        param.reg_max * 4 + param.class_num + extra_channels) {
      KZ_LOG_ERROR("yolov8: output {} does not match reg_max or class_num", i);
      return false;
    }
    inferred.push_back(head_stride(outputs[i], engine.proto().output_layout, input));
  }
  param.strides = std::move(inferred);
  return true;
}

auto resolve_yolo26_heads(Yolo26Param& param, const engine::Engine& engine, std::size_t extra_channels) -> bool {
  YoloV8Param heads{param.class_num, param.strides, 1, param.confidence_threshold, param.iou_threshold};
  if (!resolve_yolov8_heads(heads, engine, extra_channels))
    return false;
  param.strides = std::move(heads.strides);
  param.confidence_threshold = heads.confidence_threshold;
  param.iou_threshold = heads.iou_threshold;
  return true;
}

} // namespace hrt::postprocess
