// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/keypoint/yolo26_keypoint.cpp
// Description: Implements decoding of raw YOLO26 pose heads, one output per
//              stride, with any number of keypoints.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>
#include <kitzoo/log/logger.hpp>

#include <algorithm>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

namespace {

struct Yolo26KeypointParam {
  unsigned int class_num = 0;
  unsigned int keypoint_num = 0;
  Vector<float> strides;
  Vector<float> confidence_threshold;
  Vector<float> iou_threshold;
  // Visibility score above which a keypoint counts as visible.
  float keypoint_threshold = 0.5F;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(Yolo26KeypointParam, class_num, keypoint_num, strides,
                                                confidence_threshold, iou_threshold, keypoint_threshold)

// One output per stride with channels left, top, right and bottom distances
// in cells (no distance bins), class logits, then x, y (and visibility logit)
// per keypoint, as the one-to-many head of a YOLO26 pose model emits them. A
// keypoint at cell (x, y) lies at ((raw + x + 0.5) * stride, (raw + y + 0.5) * stride).
class Yolo26Keypoint final : public KeypointPostprocessor {
public:
  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    return check_section<Yolo26KeypointParam>(config);
  }

  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    param_ = config.get<Yolo26KeypointParam>();
    KZ_CHECK_RETURN(param_.keypoint_num > 0, false, "yolo26_keypoint: keypoint_num must be positive");
    const auto& outputs = engine.output_protos();
    const auto head = outputs.empty() ? std::nullopt : head_shape(outputs[0], engine.proto().output_layout);
    KZ_CHECK_RETURN(head.has_value(), false, "yolo26_keypoint: needs one [1, channels, h, w] output per stride");
    const auto extra = head->channels - static_cast<std::int64_t>(4 + param_.class_num);
    const auto keypoints = static_cast<std::int64_t>(param_.keypoint_num);
    dims_ = extra == keypoints * 3 ? 3 : (extra == keypoints * 2 ? 2 : 0);
    KZ_CHECK_RETURN(dims_ != 0, false,
                    "yolo26_keypoint: {} channels do not fit 4 distances + {} classes + {} keypoints x (2 | 3)",
                    head->channels, param_.class_num, param_.keypoint_num);
    Yolo26Param heads{param_.class_num, param_.strides, param_.confidence_threshold, param_.iou_threshold};
    if (!check_yolo26(heads, engine, param_.keypoint_num * dims_))
      return false;
    param_.confidence_threshold = heads.confidence_threshold;
    param_.iou_threshold = heads.iou_threshold;
    // Scores are compared as logits so most cells skip the sigmoid.
    logit_threshold_.resize(param_.class_num);
    std::transform(param_.confidence_threshold.begin(), param_.confidence_threshold.end(), logit_threshold_.begin(),
                   logit);
    layout_ = engine.proto().output_layout;
    return true;
  }

protected:
  auto decode(KeypointResultMap& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool override {
    if (outputs.size() != param_.strides.size())
      return false;
    for (auto& per_class : candidates_)
      per_class.clear();
    candidates_.resize(param_.class_num);
    for (std::size_t head = 0; head < outputs.size(); ++head) {
      const auto ok = visit_head(*outputs[head], layout_,
                                 [&](const auto& grid) -> void { decode_head(grid, param_.strides[head]); });
      if (!ok)
        return false;
    }
    const core::Rect2f bounds{static_cast<float>(transform.src_roi.x), static_cast<float>(transform.src_roi.y),
                              static_cast<float>(transform.src_roi.width),
                              static_cast<float>(transform.src_roi.height)};
    for (std::size_t cls = 0; cls < param_.class_num; ++cls) {
      auto& objects = candidates_[cls];
      rects_.clear();
      scores_.clear();
      for (const auto& object : objects) {
        rects_.push_back(object.box.rect);
        scores_.push_back(object.box.score);
      }
      for (auto k : nms_keep(rects_, scores_, param_.iou_threshold[cls])) {
        auto& object = objects[k];
        object.box.rect = core::intersect(to_source(object.box.rect, transform), bounds);
        if (object.box.rect.empty())
          continue;
        for (auto& keypoint : object.keypoints)
          keypoint.point = to_source(keypoint.point, transform);
        result[static_cast<std::int64_t>(cls)].push_back(std::move(object));
      }
    }
    return true;
  }

private:
  template <typename Grid>
  auto decode_head(const Grid& grid, float stride) -> void {
    const auto first_keypoint = 4 + param_.class_num;
    const auto min_score = grid.raw_bound(*std::min_element(logit_threshold_.begin(), logit_threshold_.end()));
    for (const auto cell : scan_cells(grid, 4, param_.class_num, min_score, scan_)) {
      const std::int64_t y = cell / grid.cols();
      const std::int64_t x = cell % grid.cols();
      for (std::size_t cls = 0; cls < param_.class_num; ++cls) {
        if (grid.raw(y, x, 4 + cls) < min_score)
          continue;
        const auto score = grid.at(y, x, 4 + cls);
        if (score < logit_threshold_[cls])
          continue;
        const auto cx = static_cast<float>(x) + 0.5F;
        const auto cy = static_cast<float>(y) + 0.5F;
        auto& object = candidates_[cls].emplace_back();
        object.box = {core::Rect2f::from_points({(cx - grid.at(y, x, 0)) * stride, (cy - grid.at(y, x, 1)) * stride},
                                                {(cx + grid.at(y, x, 2)) * stride, (cy + grid.at(y, x, 3)) * stride}),
                      sigmoid(score), static_cast<std::int64_t>(cls), -1};
        object.keypoints.reserve(param_.keypoint_num);
        for (std::size_t j = 0; j < param_.keypoint_num; ++j) {
          const auto channel = first_keypoint + j * dims_;
          const core::Point2f point{(grid.at(y, x, channel) + cx) * stride, (grid.at(y, x, channel + 1) + cy) * stride};
          const auto visibility = dims_ == 3 ? sigmoid(grid.at(y, x, channel + 2)) : 1.0F;
          object.keypoints.push_back(
              {point, visibility > param_.keypoint_threshold ? KeypointVisible::Visible : KeypointVisible::Invisible,
               visibility});
        }
      }
    }
  }

  Yolo26KeypointParam param_;
  CellScan scan_;
  std::size_t dims_ = 3;
  Vector<float> logit_threshold_;
  engine::DataLayout layout_ = engine::DataLayout::Nchw;
  Vector<KeypointResult> candidates_;
  Vector<core::Rect2f> rects_;
  Vector<float> scores_;
};

} // namespace

auto make_yolo26_keypoint() -> UniquePtr<KeypointPostprocessor> {
  return std::make_unique<Yolo26Keypoint>();
}

} // namespace hrt::postprocess
