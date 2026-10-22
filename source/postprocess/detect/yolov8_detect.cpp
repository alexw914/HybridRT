// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/detect/yolov8_detect.cpp
// Description: Implements YOLOv8 anchor-free detection decoding.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>

#include <algorithm>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

namespace {

class YoloV8Detect final : public DetectPostprocessor {
public:
  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    core::Problems problems;
    core::check_param<YoloV8Param>(config, "postprocess", {"type"}, problems);
    return problems;
  }

  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    if (!is_float_outputs(engine, "yolov8_detect"))
      return false;
    param_ = config.get<YoloV8Param>();
    KZ_CHECK_RETURN(!config.contains("reg_max") || param_.reg_max > 0, false, "reg_max must be positive when supplied");
    if (!resolve_yolov8_heads(param_, engine, 0))
      return false;
    // Scores are compared as logits so most cells skip the sigmoid.
    logit_threshold_.resize(param_.class_num);
    std::transform(param_.confidence_threshold.begin(), param_.confidence_threshold.end(), logit_threshold_.begin(),
                   core::logit);
    layout_ = engine.proto().output_layout;
    return true;
  }

protected:
  auto decode(DetectResultMap& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool override {
    for (std::size_t head = 0; head < outputs.size(); ++head) {
      const HeadGridView grid(*outputs[head], layout_);
      decode_head(result, grid, param_.strides[head]);
    }
    nms_to_source(result, param_.iou_threshold, transform);
    return true;
  }

private:
  auto decode_head(DetectResultMap& result, const HeadGridView& grid, core::Point2f stride) -> void {
    const auto bins = param_.reg_max;
    const auto min_score = grid.raw_bound(*std::min_element(logit_threshold_.begin(), logit_threshold_.end()));
    for (const auto cell : scan_.scan_max_score(grid, bins * 4, param_.class_num, min_score)) {
      const std::int64_t y = cell / grid.cols();
      const std::int64_t x = cell % grid.cols();
      bool decoded = false;
      core::Rect2f box;
      for (std::size_t cls = 0; cls < param_.class_num; ++cls) {
        if (grid.raw(y, x, bins * 4 + cls) < min_score)
          continue;
        const auto score = grid.at(y, x, bins * 4 + cls);
        if (score < logit_threshold_[cls])
          continue;
        if (!decoded) {
          const auto cx = static_cast<float>(x) + 0.5F;
          const auto cy = static_cast<float>(y) + 0.5F;
          box = core::Rect2f(
              {(cx - dfl(grid, y, x, 0, bins)) * stride.x, (cy - dfl(grid, y, x, bins, bins)) * stride.y},
              {(cx + dfl(grid, y, x, bins * 2, bins)) * stride.x, (cy + dfl(grid, y, x, bins * 3, bins)) * stride.y});
          decoded = true;
        }
        result[cls].push_back({box, core::sigmoid(score), static_cast<std::int64_t>(cls)});
      }
    }
  }

  YoloV8Param param_;
  CellScanner scan_;
  Vector<float> logit_threshold_;
  engine::DataLayout layout_ = engine::DataLayout::Nchw;
};

} // namespace

auto create_yolov8_detect() -> UniquePtr<DetectPostprocessor> {
  return std::make_unique<YoloV8Detect>();
}

} // namespace hrt::postprocess
