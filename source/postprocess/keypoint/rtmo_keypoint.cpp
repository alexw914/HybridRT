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
    core::Problems problems;
    core::check_param<RtmoKeypointParam>(config, "postprocess", {"type"}, problems);
    return problems;
  }

  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    if (!is_float_outputs(engine, "rtmo_keypoint"))
      return false;
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
    const HeadGridView boxes(*outputs[boxes_index_], std::nullopt);
    // [1, M, K, 3] keypoints as an NHWC grid of M rows, K cells, 3 channels.
    const auto& points_output = *outputs[1 - boxes_index_];
    const HeadGridView points(points_output, engine::DataLayout::Nhwc);
    const auto count = static_cast<std::size_t>(outputs[boxes_index_]->shape()[1]);
    const auto keypoint_num = static_cast<std::size_t>(points_output.shape()[2]);

    rects_.clear();
    scores_.clear();
    candidates_.clear();
    for (std::size_t i = 0; i < count; ++i) {
      const auto score = boxes.at(i, 4);
      // Also skips NaN boxes and scores.
      if (!(score > param_.confidence_threshold[0]))
        continue;
      const core::Point2f tl{boxes.at(i, 0), boxes.at(i, 1)};
      const core::Point2f br{boxes.at(i, 2), boxes.at(i, 3)};
      if (!(br.x - tl.x > 0.0F && br.y - tl.y > 0.0F))
        continue;
      rects_.push_back(core::Rect2f(tl, br));
      scores_.push_back(score);
      candidates_.push_back(i);
    }

    for (auto k : nms_indices(rects_, scores_, param_.iou_threshold[0])) {
      const auto box = transform.clip_to_source(rects_[k]);
      if (box.empty())
        continue;
      auto& object = result[0].emplace_back();
      object.box = {box, scores_[k], 0, -1};
      object.keypoints.reserve(keypoint_num);
      const auto row = static_cast<std::int64_t>(candidates_[k]);
      for (std::int64_t j = 0; j < static_cast<std::int64_t>(keypoint_num); ++j) {
        const auto score = points.at(row, j, 2);
        object.keypoints.push_back(
            {transform.to_source(core::Point2f{points.at(row, j, 0), points.at(row, j, 1)}),
             score > param_.keypoint_threshold ? KeypointVisible::Visible : KeypointVisible::Invisible, score});
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
};

} // namespace

auto create_rtmo_keypoint() -> UniquePtr<KeypointPostprocessor> {
  return std::make_unique<RtmoKeypoint>();
}

} // namespace hrt::postprocess
