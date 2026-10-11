// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/detect.cpp
// Description: Implements detection NMS, YOLO config validation and the
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
  if (type == "YoloV5Detect")
    return make_yolov5_detect();
  if (type == "GeneralDetect")
    return make_general_detect();
  return nullptr;
}

auto create_detect_postprocessor(std::string_view type, const kitzoo::json::Json& config, const engine::Engine& engine)
    -> UniquePtr<DetectPostprocessor> {
  return checked(make_detect(type), type, config, engine);
}

auto finish_detect(DetectResultMap& result, const Vector<float>& iou_threshold,
                   const preprocess::ImageTransform& transform) -> void {
  const core::Rect2f bounds{static_cast<float>(transform.src_roi.x), static_cast<float>(transform.src_roi.y),
                            static_cast<float>(transform.src_roi.width), static_cast<float>(transform.src_roi.height)};
  for (auto& [cls, boxes] : result) {
    nms(boxes, iou_threshold[cls]);
    for (auto& box : boxes)
      box.rect = core::intersect(to_source(box.rect, transform), bounds);
    std::erase_if(boxes, [](const ObjectBox& box) -> bool { return box.rect.empty(); });
  }
}

auto check_yolov5(YoloV5Param& param, const engine::Engine& engine, std::size_t extra_channels) -> bool {
  if (param.class_num == 0 || !expand_per_class(param.confidence_threshold, param.class_num, "confidence_threshold") ||
      !expand_per_class(param.iou_threshold, param.class_num, "iou_threshold"))
    return false;
  const auto input = engine_input_size(engine);
  const auto& outputs = engine.output_protos();
  if (!input || outputs.size() != param.head_num || param.strides_per_head.size() != param.head_num ||
      param.anchors_per_head.size() != param.head_num || param.anchors.size() != param.head_num) {
    KZ_LOG_ERROR("yolov5: head_num, strides_per_head, anchors_per_head and anchors must match the {} outputs",
                 outputs.size());
    return false;
  }
  for (std::size_t i = 0; i < param.head_num; ++i) {
    const auto head = head_shape(outputs[i], engine.proto().output_layout);
    const auto anchors = static_cast<std::size_t>(param.anchors_per_head[i]);
    const auto stride = param.strides_per_head[i];
    if (!head || param.anchors[i].size() != anchors * 2 ||
        static_cast<float>(head->cols) * stride != static_cast<float>(input->width) ||
        static_cast<float>(head->rows) * stride != static_cast<float>(input->height) ||
        static_cast<std::size_t>(head->channels) != (param.class_num + 5 + extra_channels) * anchors) {
      KZ_LOG_ERROR("yolov5: output {} does not match the anchors, strides or class_num", i);
      return false;
    }
  }
  return true;
}

auto check_yolov8(YoloV8Param& param, const engine::Engine& engine, std::size_t extra_channels) -> bool {
  if (param.class_num == 0 || param.reg_max == 0 ||
      !expand_per_class(param.confidence_threshold, param.class_num, "confidence_threshold") ||
      !expand_per_class(param.iou_threshold, param.class_num, "iou_threshold"))
    return false;
  const auto input = engine_input_size(engine);
  const auto& outputs = engine.output_protos();
  if (!input || outputs.size() != param.strides.size()) {
    KZ_LOG_ERROR("yolov8: strides must match the {} outputs", outputs.size());
    return false;
  }
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto head = head_shape(outputs[i], engine.proto().output_layout);
    const auto stride = param.strides[i];
    if (!head || static_cast<float>(head->cols) * stride != static_cast<float>(input->width) ||
        static_cast<float>(head->rows) * stride != static_cast<float>(input->height) ||
        static_cast<std::size_t>(head->channels) != param.reg_max * 4 + param.class_num + extra_channels) {
      KZ_LOG_ERROR("yolov8: output {} does not match the strides, reg_max or class_num", i);
      return false;
    }
  }
  return true;
}

} // namespace hrt::postprocess
