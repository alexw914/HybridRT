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
#include "include/postprocess/nms.hpp"
#include "include/postprocess/postprocessor_creators.hpp"
#include "include/postprocess/yolo_heads.hpp"

namespace hrt::postprocess {

namespace {

struct YoloV8KeypointParam {
  unsigned int class_num = 0;
  unsigned int keypoint_num = 17;
  Vector<core::Point2f> strides; // Inferred in init from the input and head sizes; not a config key.
  std::size_t reg_max = 16;
  Vector<float> confidence_threshold;
  Vector<float> iou_threshold;
  // Visibility score above which a keypoint counts as visible.
  float keypoint_threshold = 0.5F;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(YoloV8KeypointParam, class_num, keypoint_num, reg_max,
                                                confidence_threshold, iou_threshold, keypoint_threshold)

// One output per stride with channels 4 * reg_max distance bins, class
// logits, then x, y and visibility logit per keypoint, as Ultralytics pose
// heads emit them before decoding (e.g. YOLOv8 face models with 5 landmarks).
// A keypoint at cell (x, y) lies at ((raw * 2 + x) * stride.x, (raw * 2 + y) * stride.y).
class YoloV8Keypoint final : public KeypointPostprocessor {
public:
  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    if (!is_float_outputs(engine, "yolov8_keypoint"))
      return false;
    param_ = config.get<YoloV8KeypointParam>();
    KZ_CHECK_RETURN(param_.keypoint_num > 0, false, "yolov8_keypoint: keypoint_num must be positive");
    YoloV8Param heads{param_.class_num, param_.strides, param_.reg_max, param_.confidence_threshold,
                      param_.iou_threshold};
    if (!resolve_yolov8_heads(heads, engine, param_.keypoint_num * kKeypointValues))
      return false;
    param_.strides = std::move(heads.strides);
    param_.confidence_threshold = heads.confidence_threshold;
    param_.iou_threshold = heads.iou_threshold;
    // Scores are compared as logits so most cells skip the sigmoid.
    logit_threshold_.resize(param_.class_num);
    std::transform(param_.confidence_threshold.begin(), param_.confidence_threshold.end(), logit_threshold_.begin(),
                   core::logit);
    layout_ = engine.proto().output_layout;
    tables_.resize(engine.output_protos().size());
    return true;
  }

  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    core::Problems problems;
    core::check_param<YoloV8KeypointParam>(config, "postprocess", {"type"}, problems);
    return problems;
  }

protected:
  auto decode(KeypointResultMap& result, const core::Buffers& outputs, std::size_t index,
              const preprocess::ImageTransform& transform) -> bool override {
    for (auto& per_class : candidates_)
      per_class.clear();
    candidates_.resize(param_.class_num);
    for (std::size_t head = 0; head < outputs.size(); ++head) {
      const HeadGridView grid(*outputs[head], index, layout_);
      tables_[head].bind(grid, logit_threshold_);
      decode_head(grid, tables_[head], param_.strides[head]);
    }
    for (std::size_t cls = 0; cls < param_.class_num; ++cls) {
      auto& objects = candidates_[cls];
      rects_.clear();
      scores_.clear();
      for (const auto& object : objects) {
        rects_.push_back(object.box.rect);
        scores_.push_back(object.box.score);
      }
      for (auto k : nms_indices(rects_, scores_, param_.iou_threshold[cls])) {
        auto& object = objects[k];
        object.box.rect = transform.clip_reverse(object.box.rect);
        if (object.box.rect.empty())
          continue;
        object.box.score = core::sigmoid(object.box.score);
        for (auto& keypoint : object.keypoints)
          keypoint.point = transform.reverse(keypoint.point);
        result[static_cast<std::int64_t>(cls)].push_back(std::move(object));
      }
    }
    return true;
  }

private:
  auto decode_head(const HeadGridView& grid, const QuantTables& tables, core::Point2f stride) -> void {
    const auto bins = param_.reg_max;
    const auto first_keypoint = bins * 4 + param_.class_num;
    const auto min_score = grid.raw_bound(*std::min_element(logit_threshold_.begin(), logit_threshold_.end()));
    for (const auto cell : scan_.scan_max_score(grid, bins * 4, param_.class_num, min_score)) {
      const std::int64_t y = cell / grid.cols();
      const std::int64_t x = cell % grid.cols();
      for (std::size_t cls = 0; cls < param_.class_num; ++cls) {
        if (grid.raw(y, x, bins * 4 + cls) < tables.bound(cls))
          continue;
        // The logit ranks candidates in NMS; only kept ones take the sigmoid.
        const auto score = grid.at(y, x, bins * 4 + cls);
        const auto cx = static_cast<float>(x) + 0.5F;
        const auto cy = static_cast<float>(y) + 0.5F;
        auto& object = candidates_[cls].emplace_back();
        object.box = {core::Rect2f({(cx - dfl(grid, tables, y, x, 0, bins)) * stride.x,
                                    (cy - dfl(grid, tables, y, x, bins, bins)) * stride.y},
                                   {(cx + dfl(grid, tables, y, x, bins * 2, bins)) * stride.x,
                                    (cy + dfl(grid, tables, y, x, bins * 3, bins)) * stride.y}),
                      score, static_cast<std::int64_t>(cls), -1};
        object.keypoints.reserve(param_.keypoint_num);
        for (std::size_t j = 0; j < param_.keypoint_num; ++j) {
          const auto channel = first_keypoint + j * kKeypointValues;
          const core::Point2f point{(grid.at(y, x, channel) * 2.0F + static_cast<float>(x)) * stride.x,
                                    (grid.at(y, x, channel + 1) * 2.0F + static_cast<float>(y)) * stride.y};
          const auto visibility = sigmoid_at(grid, tables, y, x, channel + 2);
          object.keypoints.push_back(
              {point, visibility > param_.keypoint_threshold ? KeypointVisible::Visible : KeypointVisible::Invisible,
               visibility});
        }
      }
    }
  }

  YoloV8KeypointParam param_;
  CellScanner scan_;
  Vector<QuantTables> tables_;
  Vector<float> logit_threshold_;
  engine::DataLayout layout_ = engine::DataLayout::Nchw;
  Vector<KeypointResult> candidates_;
  Vector<core::Rect2f> rects_;
  Vector<float> scores_;
};

} // namespace

auto create_yolov8_keypoint() -> UniquePtr<KeypointPostprocessor> {
  return std::make_unique<YoloV8Keypoint>();
}

} // namespace hrt::postprocess
