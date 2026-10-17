// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/keypoint/general_yolo_pose.cpp
// Description: Implements decoding of single-output pose heads, with a box
//              per instance.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>
#include <kitzoo/log/logger.hpp>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

namespace {

struct GeneralYoloPoseParam {
  unsigned int class_num = 0;
  unsigned int keypoint_num = 0;
  Vector<float> confidence_threshold;
  Vector<float> iou_threshold;
  // Visibility score above which a keypoint counts as visible.
  float keypoint_threshold = 0.5F;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(GeneralYoloPoseParam, class_num, keypoint_num, confidence_threshold,
                                                iou_threshold, keypoint_threshold)

// One output [1, 4 + classes + keypoints * (2 | 3), N]: rows center x, center
// y, width, height, one score per class, then x, y (and visibility) per
// keypoint, all in network-input pixels.
class GeneralYoloPose final : public KeypointPostprocessor {
public:
  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    return check_section<GeneralYoloPoseParam>(config);
  }

  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    param_ = config.get<GeneralYoloPoseParam>();
    KZ_CHECK_RETURN(param_.keypoint_num > 0, false, "general_yolo_pose: keypoint_num must be positive");
    if (param_.class_num == 0 ||
        !expand_per_class(param_.confidence_threshold, param_.class_num, "confidence_threshold") ||
        !expand_per_class(param_.iou_threshold, param_.class_num, "iou_threshold"))
      return false;
    const auto& outputs = engine.output_protos();
    KZ_CHECK_RETURN(engine.proto().output_layout == engine::DataLayout::Nchw && engine.input_protos().size() == 1 &&
                        outputs.size() == 1 && outputs[0].shape.size() == 3 && outputs[0].shape[0] == 1,
                    false, "general_yolo_pose: needs one input and one [1, rows, N] output");
    const auto rows = outputs[0].shape[1] - 4 - static_cast<std::int64_t>(param_.class_num);
    const auto keypoints = static_cast<std::int64_t>(param_.keypoint_num);
    dims_ = rows == keypoints * 3 ? 3 : (rows == keypoints * 2 ? 2 : 0);
    KZ_CHECK_RETURN(dims_ != 0, false,
                    "general_yolo_pose: output rows {} do not fit 4 + {} classes + {} keypoints x (2 | 3)",
                    outputs[0].shape[1], param_.class_num, param_.keypoint_num);
    return true;
  }

protected:
  auto decode(KeypointResultMap& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool override {
    const RowView rows(*outputs[0], scratch_);
    if (!rows.valid())
      return false;
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
      candidates_.clear();
      rows.above(4 + cls, param_.confidence_threshold[cls], candidates_);
      for (const auto i : candidates_) {
        const auto [l, t, r, b] = center_box(rows.at(0, i), rows.at(1, i), rows.at(2, i), rows.at(3, i));
        // Also skips NaN boxes.
        if (!(r - l > 0.0F && b - t > 0.0F))
          continue;
        rects_[cls].push_back(core::Rect2f::from_points({l, t}, {r, b}));
        scores_[cls].push_back(rows.at(4 + cls, i));
        anchors_[cls].push_back(i);
      }
    }

    const core::Rect2f bounds{static_cast<float>(transform.src_roi.x), static_cast<float>(transform.src_roi.y),
                              static_cast<float>(transform.src_roi.width),
                              static_cast<float>(transform.src_roi.height)};
    const auto first_row = 4 + classes;
    for (std::size_t cls = 0; cls < classes; ++cls) {
      for (auto k : nms_keep(rects_[cls], scores_[cls], param_.iou_threshold[cls])) {
        const auto box = core::intersect(to_source(rects_[cls][k], transform), bounds);
        if (box.empty())
          continue;
        auto& object = result[static_cast<std::int64_t>(cls)].emplace_back();
        object.box = {box, scores_[cls][k], static_cast<std::int64_t>(cls), -1};
        const auto anchor = anchors_[cls][k];
        object.keypoints.reserve(param_.keypoint_num);
        for (std::size_t j = 0; j < param_.keypoint_num; ++j) {
          const auto row = first_row + j * dims_;
          const core::Point2f point{rows.at(row, anchor), rows.at(row + 1, anchor)};
          const auto score = dims_ == 3 ? rows.at(row + 2, anchor) : 1.0F;
          object.keypoints.push_back(
              {to_source(point, transform),
               score > param_.keypoint_threshold ? KeypointVisible::Visible : KeypointVisible::Invisible, score});
        }
      }
    }
    return true;
  }

private:
  GeneralYoloPoseParam param_;
  std::size_t dims_ = 3;
  Vector<std::uint32_t> candidates_;
  Vector<Vector<core::Rect2f>> rects_;
  Vector<Vector<float>> scores_;
  Vector<Vector<std::uint32_t>> anchors_;
  core::Buffer scratch_{core::MemoryType::Cpu, core::DataType::Float32};
};

} // namespace

auto make_general_yolo_pose() -> UniquePtr<KeypointPostprocessor> {
  return std::make_unique<GeneralYoloPose>();
}

} // namespace hrt::postprocess
