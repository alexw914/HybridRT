// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/keypoint/rtmo_keypoint.cpp
// Description: Implements decoding of one-stage RTMO pose outputs with fixed
//              candidates, boxes [1, M, 5] and keypoints [1, M, K, 3].
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>
#include <kitzoo/log/logger.hpp>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

namespace {

// Defaults are the score threshold and IoU of the NMS in mmpose's RTMO export
// and mmpose's keypoint drawing threshold.
struct RtmoKeypointParam {
  Vector<float> confidence_threshold{0.15F};
  Vector<float> iou_threshold{0.65F};
  float keypoint_threshold = 0.3F;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(RtmoKeypointParam, confidence_threshold, iou_threshold,
                                                keypoint_threshold)

// Outputs, in either order: boxes [1, M, 5] (x1, y1, x2, y2, score) and
// keypoints [1, M, K, 3] (x, y, visibility score), in network-input pixels,
// for the M highest-scoring candidates before NMS (see assets/template/rtmo_fix_outputs.py).
class RtmoKeypoint final : public KeypointPostprocessor {
public:
  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    return check_section<RtmoKeypointParam>(config);
  }

  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    param_ = config.get<RtmoKeypointParam>();
    if (!expand_per_class(param_.confidence_threshold, 1, "confidence_threshold") ||
        !expand_per_class(param_.iou_threshold, 1, "iou_threshold"))
      return false;
    const auto& outputs = engine.output_protos();
    KZ_CHECK_RETURN(engine.input_protos().size() == 1 && outputs.size() == 2, false,
                    "rtmo_keypoint: needs one input and two outputs (boxes, keypoints)");
    boxes_index_ = outputs[0].shape.size() == 3 ? 0 : 1;
    const auto& boxes = outputs[boxes_index_].shape;
    const auto& keypoints = outputs[1 - boxes_index_].shape;
    KZ_CHECK_RETURN(boxes.size() == 3 && boxes[0] == 1 && boxes[1] > 0 && boxes[2] == 5 && keypoints.size() == 4 &&
                        keypoints[0] == 1 && keypoints[1] == boxes[1] && keypoints[2] > 0 && keypoints[3] == 3,
                    false, "rtmo_keypoint: needs [1, M, 5] boxes and [1, M, K, 3] keypoints");
    return true;
  }

protected:
  auto decode(KeypointResultMap& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool override {
    const auto* boxes = float_data(*outputs[boxes_index_], boxes_scratch_);
    const auto* points = float_data(*outputs[1 - boxes_index_], points_scratch_);
    if (boxes == nullptr || points == nullptr)
      return false;
    const auto count = static_cast<std::size_t>(outputs[boxes_index_]->shape()[1]);
    const auto keypoint_num = static_cast<std::size_t>(outputs[1 - boxes_index_]->shape()[2]);

    rects_.clear();
    scores_.clear();
    candidates_.clear();
    for (std::size_t i = 0; i < count; ++i) {
      const auto* box = boxes + i * 5;
      // Also skips NaN boxes and scores.
      if (!(box[4] > param_.confidence_threshold[0]) || !(box[2] - box[0] > 0.0F && box[3] - box[1] > 0.0F))
        continue;
      rects_.push_back(core::Rect2f::from_points({box[0], box[1]}, {box[2], box[3]}));
      scores_.push_back(box[4]);
      candidates_.push_back(i);
    }

    for (auto k : nms_keep(rects_, scores_, param_.iou_threshold[0])) {
      const auto box = clip_to_source(rects_[k], transform);
      if (box.empty())
        continue;
      auto& object = result[0].emplace_back();
      object.box = {box, scores_[k], 0, -1};
      object.keypoints.reserve(keypoint_num);
      const auto* point = points + candidates_[k] * keypoint_num * 3;
      for (std::size_t j = 0; j < keypoint_num; ++j, point += 3) {
        object.keypoints.push_back(
            {to_source(core::Point2f{point[0], point[1]}, transform),
             point[2] > param_.keypoint_threshold ? KeypointVisible::Visible : KeypointVisible::Invisible, point[2]});
      }
    }
    return true;
  }

private:
  RtmoKeypointParam param_;
  std::size_t boxes_index_ = 0;
  Vector<core::Rect2f> rects_;
  Vector<float> scores_;
  Vector<std::size_t> candidates_;
  core::Buffer boxes_scratch_{core::MemoryType::Cpu, core::DataType::Float32};
  core::Buffer points_scratch_{core::MemoryType::Cpu, core::DataType::Float32};
};

} // namespace

auto make_rtmo_keypoint() -> UniquePtr<KeypointPostprocessor> {
  return std::make_unique<RtmoKeypoint>();
}

} // namespace hrt::postprocess
