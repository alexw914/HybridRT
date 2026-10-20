// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/detect.cpp
// Description: Implements detection NMS, head validation and the
//              detection postprocessor factory.
// -----------------------------------------------------------------------------

#include <hrt/postprocess/detect.hpp>

#include <kitzoo/log/logger.hpp>

#include <algorithm>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

auto nms(DetectResult& boxes, float iou_threshold) -> void {
  if (boxes.size() <= 1)
    return;
  Vector<core::Rect2f> rects;
  Vector<float> scores;
  for (const auto& box : boxes) {
    rects.push_back(box.rect);
    scores.push_back(box.score);
  }
  DetectResult kept;
  for (auto index : nms_keep(rects, scores, iou_threshold))
    kept.push_back(boxes[index]);
  boxes = std::move(kept);
}

auto make_detect(std::string_view type) -> UniquePtr<DetectPostprocessor> {
  if (type == "YoloV8Detect")
    return make_yolov8_detect();
  if (type == "Yolo26Detect")
    return make_yolo26_detect();
  if (type == "YoloV5Detect")
    return make_yolov5_detect();
  if (type == "GeneralDetect")
    return make_general_detect();
  if (type == "DetrDetect")
    return make_detr_detect();
  if (type == "E2EDetect")
    return make_e2e_detect();
  return nullptr;
}

auto create_detect_postprocessor(std::string_view type, const kitzoo::json::Json& config, const engine::Engine& engine)
    -> UniquePtr<DetectPostprocessor> {
  return checked(make_detect(type), type, config, engine);
}

auto finish_detect(DetectResultMap& result, const Vector<float>& iou_threshold,
                   const preprocess::ImageTransform& transform) -> void {
  for (auto& [cls, boxes] : result) {
    nms(boxes, iou_threshold[cls]);
    for (auto& box : boxes)
      box.rect = clip_to_source(box.rect, transform);
    std::erase_if(boxes, [](const ObjectBox& box) -> bool { return box.rect.empty(); });
  }
}

auto check_yolov5(YoloV5Param& param, const engine::Engine& engine, std::size_t extra_channels) -> bool {
  if (param.class_num == 0 || !expand_per_class(param.confidence_threshold, param.class_num, "confidence_threshold") ||
      !expand_per_class(param.iou_threshold, param.class_num, "iou_threshold"))
    return false;
  const auto input = engine.input_size();
  const auto& outputs = engine.output_protos();
  if (!input || outputs.empty() || outputs.size() != param.anchors.size()) {
    KZ_LOG_ERROR("yolov5: needs nonempty outputs and one anchor list per head, got {} outputs", outputs.size());
    return false;
  }
  Vector<float> inferred;
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto head = HeadShape::of(outputs[i], engine.proto().output_layout);
    const auto anchors = param.anchors[i].size() / 2;
    const auto stride = head ? head->stride(*input) : std::nullopt;
    if (!head || !stride || anchors == 0 || param.anchors[i].size() % 2 != 0 ||
        static_cast<std::size_t>(head->channels) != (param.class_num + 5 + extra_channels) * anchors) {
      KZ_LOG_ERROR("yolov5: output {} does not match the anchors, strides or class_num", i);
      return false;
    }
    inferred.push_back(*stride);
  }
  return resolve_strides(param.strides, inferred);
}

auto check_yolov8(YoloV8Param& param, const engine::Engine& engine, std::size_t extra_channels) -> bool {
  if (param.class_num == 0 || !expand_per_class(param.confidence_threshold, param.class_num, "confidence_threshold") ||
      !expand_per_class(param.iou_threshold, param.class_num, "iou_threshold"))
    return false;
  const auto input = engine.input_size();
  const auto& outputs = engine.output_protos();
  if (!input || outputs.empty()) {
    KZ_LOG_ERROR("yolov8: needs a valid input and nonempty outputs, got {} outputs", outputs.size());
    return false;
  }
  const auto first = HeadShape::of(outputs.front(), engine.proto().output_layout);
  const auto channels =
      first ? first->channels - static_cast<std::int64_t>(param.class_num) - static_cast<std::int64_t>(extra_channels)
            : 0;
  KZ_CHECK_RETURN(channels > 0 && channels % 4 == 0, false, "yolov8: invalid channels for reg_max");
  const auto inferred_bins = static_cast<std::size_t>(channels / 4);
  KZ_CHECK_RETURN(param.reg_max == 0 || param.reg_max == inferred_bins, false,
                  "yolov8: configured reg_max does not match output channels");
  param.reg_max = inferred_bins;
  Vector<float> inferred;
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto head = HeadShape::of(outputs[i], engine.proto().output_layout);
    const auto stride = head ? head->stride(*input) : std::nullopt;
    if (!head || !stride ||
        static_cast<std::size_t>(head->channels) != param.reg_max * 4 + param.class_num + extra_channels) {
      KZ_LOG_ERROR("yolov8: output {} does not match the strides, reg_max or class_num", i);
      return false;
    }
    inferred.push_back(*stride);
  }
  return resolve_strides(param.strides, inferred);
}

auto check_yolo26(Yolo26Param& param, const engine::Engine& engine, std::size_t extra_channels) -> bool {
  YoloV8Param heads{param.class_num, param.strides, 1, param.confidence_threshold, param.iou_threshold};
  if (!check_yolov8(heads, engine, extra_channels))
    return false;
  param.strides = std::move(heads.strides);
  param.confidence_threshold = heads.confidence_threshold;
  param.iou_threshold = heads.iou_threshold;
  return true;
}

} // namespace hrt::postprocess
