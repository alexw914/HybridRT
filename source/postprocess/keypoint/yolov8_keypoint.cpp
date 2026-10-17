// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/keypoint/yolov8_keypoint.cpp
// Description: Implements decoding of raw anchor-free YOLOv8 pose heads, one
//              output per stride, with any number of keypoints.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>
#include <kitzoo/log/logger.hpp>

#include <algorithm>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

namespace {

struct YoloV8KeypointParam {
  unsigned int class_num = 0;
  unsigned int keypoint_num = 0;
  Vector<float> strides;
  std::size_t reg_max = 16;
  Vector<float> confidence_threshold;
  Vector<float> iou_threshold;
  // Visibility score above which a keypoint counts as visible.
  float keypoint_threshold = 0.5F;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(YoloV8KeypointParam, class_num, keypoint_num, strides, reg_max,
                                                confidence_threshold, iou_threshold, keypoint_threshold)

// One output per stride with channels 4 * reg_max distance bins, class
// logits, then x, y (and visibility logit) per keypoint, as Ultralytics pose
// heads emit them before decoding (e.g. YOLOv8 face models with 5 landmarks).
// A keypoint at cell (x, y) lies at ((raw * 2 + x) * stride, (raw * 2 + y) * stride).
class YoloV8Keypoint final : public KeypointPostprocessor {
public:
  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    return check_section<YoloV8KeypointParam>(config);
  }

  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    param_ = config.get<YoloV8KeypointParam>();
    KZ_CHECK_RETURN(param_.keypoint_num > 0, false, "yolov8_keypoint: keypoint_num must be positive");
    const auto& outputs = engine.output_protos();
    const auto head = outputs.empty() ? std::nullopt : head_shape(outputs[0], engine.proto().output_layout);
    KZ_CHECK_RETURN(head.has_value(), false, "yolov8_keypoint: needs one [1, channels, h, w] output per stride");
    const auto extra = head->channels - static_cast<std::int64_t>(param_.reg_max * 4 + param_.class_num);
    const auto keypoints = static_cast<std::int64_t>(param_.keypoint_num);
    dims_ = extra == keypoints * 3 ? 3 : (extra == keypoints * 2 ? 2 : 0);
    KZ_CHECK_RETURN(dims_ != 0, false,
                    "yolov8_keypoint: {} channels do not fit 4 x {} bins + {} classes + {} keypoints x (2 | 3)",
                    head->channels, param_.reg_max, param_.class_num, param_.keypoint_num);
    YoloV8Param heads{param_.class_num, param_.strides, param_.reg_max, param_.confidence_threshold,
                      param_.iou_threshold};
    if (!check_yolov8(heads, engine, param_.keypoint_num * dims_))
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
    const auto bins = param_.reg_max;
    const auto first_keypoint = bins * 4 + param_.class_num;
    const auto min_score = grid.raw_bound(*std::min_element(logit_threshold_.begin(), logit_threshold_.end()));
    for (std::int64_t y = 0; y < grid.rows(); ++y) {
      for (std::int64_t x = 0; x < grid.cols(); ++x) {
        for (std::size_t cls = 0; cls < param_.class_num; ++cls) {
          if (grid.raw(y, x, bins * 4 + cls) < min_score)
            continue;
          const auto score = grid.at(y, x, bins * 4 + cls);
          if (score < logit_threshold_[cls])
            continue;
          const auto cx = static_cast<float>(x) + 0.5F;
          const auto cy = static_cast<float>(y) + 0.5F;
          auto& object = candidates_[cls].emplace_back();
          object.box = {
              core::Rect2f::from_points(
                  {(cx - dfl(grid, y, x, 0, bins)) * stride, (cy - dfl(grid, y, x, bins, bins)) * stride},
                  {(cx + dfl(grid, y, x, bins * 2, bins)) * stride, (cy + dfl(grid, y, x, bins * 3, bins)) * stride}),
              sigmoid(score), static_cast<std::int64_t>(cls), -1};
          object.keypoints.reserve(param_.keypoint_num);
          for (std::size_t j = 0; j < param_.keypoint_num; ++j) {
            const auto channel = first_keypoint + j * dims_;
            const core::Point2f point{(grid.at(y, x, channel) * 2.0F + static_cast<float>(x)) * stride,
                                      (grid.at(y, x, channel + 1) * 2.0F + static_cast<float>(y)) * stride};
            const auto visibility = dims_ == 3 ? sigmoid(grid.at(y, x, channel + 2)) : 1.0F;
            object.keypoints.push_back(
                {point, visibility > param_.keypoint_threshold ? KeypointVisible::Visible : KeypointVisible::Invisible,
                 visibility});
          }
        }
      }
    }
  }

  YoloV8KeypointParam param_;
  std::size_t dims_ = 3;
  Vector<float> logit_threshold_;
  engine::DataLayout layout_ = engine::DataLayout::Nchw;
  Vector<KeypointResult> candidates_;
  Vector<core::Rect2f> rects_;
  Vector<float> scores_;
};

} // namespace

auto make_yolov8_keypoint() -> UniquePtr<KeypointPostprocessor> {
  return std::make_unique<YoloV8Keypoint>();
}

} // namespace hrt::postprocess
