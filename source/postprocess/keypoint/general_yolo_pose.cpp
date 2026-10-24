// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/keypoint/general_yolo_pose.cpp
// Description: Implements decoding of single-output pose heads, with a box
//              per instance.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>
#include <kitzoo/log/logger.hpp>

#include "include/postprocess/common.hpp"
#include "include/postprocess/nms.hpp"
#include "include/postprocess/postprocessor_creators.hpp"

namespace hrt::postprocess {

namespace {

struct GeneralYoloPoseParam {
  unsigned int class_num = 0;
  unsigned int keypoint_num = 17;
  Vector<float> confidence_threshold;
  Vector<float> iou_threshold;
  // Visibility score above which a keypoint counts as visible.
  float keypoint_threshold = 0.5F;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(GeneralYoloPoseParam, class_num, keypoint_num, confidence_threshold,
                                                iou_threshold, keypoint_threshold)

// One output [1, 4 + classes + keypoints * 3, N]: rows center x, center
// y, width, height, one score per class, then x, y and visibility per
// keypoint, all in network-input pixels.
class GeneralYoloPose final : public KeypointPostprocessor {
public:
  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    if (!is_float_outputs(engine, "general_yolo_pose"))
      return false;
    param_ = config.get<GeneralYoloPoseParam>();
    KZ_CHECK_RETURN(param_.keypoint_num > 0, false, "general_yolo_pose: keypoint_num must be positive");
    if (!expand_per_class(param_.confidence_threshold, param_.class_num, "confidence_threshold") ||
        !expand_per_class(param_.iou_threshold, param_.class_num, "iou_threshold"))
      return false;
    const auto& outputs = engine.output_protos();
    KZ_CHECK_RETURN(engine.proto().output_layout == engine::DataLayout::Nchw && engine.input_protos().size() == 1 &&
                        outputs.size() == 1 && outputs[0].shape.size() == 3 && outputs[0].shape[0] >= 1,
                    false, "general_yolo_pose: needs one input and one [batch, rows, N] output");
    KZ_CHECK_RETURN(outputs[0].shape[1] ==
                        static_cast<std::int64_t>(4 + param_.class_num + param_.keypoint_num * kKeypointValues),
                    false, "general_yolo_pose: output rows do not match class_num and keypoint_num");
    return true;
  }

  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    core::Problems problems;
    core::check_param<GeneralYoloPoseParam>(config, "postprocess", {"type"}, problems);
    return problems;
  }

protected:
  auto decode(KeypointResultMap& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool override {
    const HeadGridView rows(*outputs[0], std::nullopt);
    decode_rows(result, rows, transform);
    return true;
  }

private:
  auto decode_rows(KeypointResultMap& result, const HeadGridView& rows, const preprocess::ImageTransform& transform)
      -> void {
    const auto at = [&](std::size_t row, std::uint32_t i) -> float { return rows.at(0, i, row); };
    const auto classes = param_.class_num;

    // Candidates per class in network coordinates, one class row at a time
    // in memory order; anchors stay in order within a class.
    rects_.resize(classes);
    scores_.resize(classes);
    anchors_.resize(classes);
    for (std::size_t cls = 0; cls < classes; ++cls) {
      rects_[cls].clear();
      scores_[cls].clear();
      anchors_[cls].clear();
      for (const auto i : scan_.scan_channel(rows, 4 + cls, param_.confidence_threshold[cls])) {
        const auto rect = core::center_box(at(0, i), at(1, i), at(2, i), at(3, i));
        if (!core::valid_box(rect))
          continue;
        rects_[cls].push_back(rect);
        scores_[cls].push_back(at(4 + cls, i));
        anchors_[cls].push_back(i);
      }
    }

    const auto first_row = 4 + classes;
    for (std::size_t cls = 0; cls < classes; ++cls) {
      for (auto k : nms_indices(rects_[cls], scores_[cls], param_.iou_threshold[cls])) {
        const auto box = transform.clip_reverse(rects_[cls][k]);
        if (box.empty())
          continue;
        auto& object = result[static_cast<std::int64_t>(cls)].emplace_back();
        object.box = {box, scores_[cls][k], static_cast<std::int64_t>(cls), -1};
        const auto anchor = anchors_[cls][k];
        object.keypoints.reserve(param_.keypoint_num);
        for (std::size_t j = 0; j < param_.keypoint_num; ++j) {
          const auto row = first_row + j * kKeypointValues;
          const core::Point2f point{at(row, anchor), at(row + 1, anchor)};
          const auto score = at(row + 2, anchor);
          object.keypoints.push_back(
              {transform.reverse(point),
               score > param_.keypoint_threshold ? KeypointVisible::Visible : KeypointVisible::Invisible, score});
        }
      }
    }
  }

  GeneralYoloPoseParam param_;
  CellScanner scan_;
  Vector<Vector<core::Rect2f>> rects_;
  Vector<Vector<float>> scores_;
  Vector<Vector<std::uint32_t>> anchors_;
};

} // namespace

auto create_general_yolo_pose() -> UniquePtr<KeypointPostprocessor> {
  return std::make_unique<GeneralYoloPose>();
}

} // namespace hrt::postprocess
