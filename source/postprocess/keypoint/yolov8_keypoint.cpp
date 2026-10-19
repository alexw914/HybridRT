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
  std::size_t reg_max = 0;
  Vector<float> confidence_threshold;
  Vector<float> iou_threshold;
  // Visibility score above which a keypoint counts as visible.
  float keypoint_threshold = 0.5F;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(YoloV8KeypointParam, class_num, keypoint_num, strides, reg_max,
                                                confidence_threshold, iou_threshold, keypoint_threshold)

auto has_shape_parameters(const kitzoo::json::Json& config) -> bool {
  return config.contains("class_num") + config.contains("reg_max") + config.contains("keypoint_num") >= 2;
}

// One output per stride with channels 4 * reg_max distance bins, class
// logits, then x, y (and visibility logit) per keypoint, as Ultralytics pose
// heads emit them before decoding (e.g. YOLOv8 face models with 5 landmarks).
// A keypoint at cell (x, y) lies at ((raw * 2 + x) * stride, (raw * 2 + y) * stride).
class YoloV8Keypoint final : public KeypointPostprocessor {
public:
  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    auto problems = check_section<YoloV8KeypointParam>(config);
    if (!has_shape_parameters(config))
      problems.emplace_back("postprocess: YoloV8Keypoint needs at least two of class_num, reg_max and keypoint_num");
    return problems;
  }

  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    KZ_CHECK_RETURN(has_shape_parameters(config), false,
                    "yolov8_keypoint: supply at least two of class_num, reg_max and keypoint_num");
    param_ = config.get<YoloV8KeypointParam>();
    KZ_CHECK_RETURN(!config.contains("class_num") || param_.class_num > 0, false,
                    "yolov8_keypoint: class_num must be positive when supplied");
    KZ_CHECK_RETURN(!config.contains("reg_max") || param_.reg_max > 0, false, "reg_max must be positive when supplied");
    KZ_CHECK_RETURN(!config.contains("keypoint_num") || param_.keypoint_num > 0, false,
                    "yolov8_keypoint: keypoint_num must be positive");
    const auto& outputs = engine.output_protos();
    const auto head = outputs.empty() ? std::nullopt : head_shape(outputs[0], engine.proto().output_layout);
    KZ_CHECK_RETURN(head.has_value(), false, "yolov8_keypoint: needs one [1, channels, h, w] output per stride");
    if (!config.contains("class_num")) {
      KZ_CHECK_RETURN(head->channels > 0 && param_.reg_max <= static_cast<std::size_t>((head->channels - 1) / 4), false,
                      "yolov8_keypoint: reg_max exceeds output channels");
      for (const std::int64_t dims : {2, 3}) {
        const auto classes = head->channels - static_cast<std::int64_t>(param_.reg_max) * 4 -
                             static_cast<std::int64_t>(param_.keypoint_num) * dims;
        if (classes <= 0 || classes > std::numeric_limits<unsigned int>::max())
          continue;
        KZ_CHECK_RETURN(param_.class_num == 0, false,
                        "yolov8_keypoint: ambiguous class count and keypoint dimensions; supply class_num");
        param_.class_num = static_cast<unsigned int>(classes);
      }
      KZ_CHECK_RETURN(param_.class_num > 0, false, "yolov8_keypoint: no positive class count fits the outputs");
    }
    const auto shape = keypoint_shape(head->channels - static_cast<std::int64_t>(param_.class_num), param_.keypoint_num,
                                      param_.reg_max);
    if (!shape)
      return false;
    param_.keypoint_num = shape->count;
    dims_ = shape->dims;
    param_.reg_max = shape->reg_max;
    YoloV8Param heads{param_.class_num, param_.strides, param_.reg_max, param_.confidence_threshold,
                      param_.iou_threshold};
    if (!check_yolov8(heads, engine, param_.keypoint_num * dims_))
      return false;
    param_.reg_max = heads.reg_max;
    param_.strides = std::move(heads.strides);
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
    for (const auto cell : scan_cells(grid, bins * 4, param_.class_num, min_score, scan_)) {
      const std::int64_t y = cell / grid.cols();
      const std::int64_t x = cell % grid.cols();
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

  YoloV8KeypointParam param_;
  CellScan scan_;
  std::size_t dims_ = 0;
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
