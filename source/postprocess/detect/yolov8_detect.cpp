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
    return check_section<YoloV8Param>(config);
  }

  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    param_ = config.get<YoloV8Param>();
    if (!check_yolov8(param_, engine, 0))
      return false;
    // Scores are compared as logits so most cells skip the sigmoid.
    logit_threshold_.resize(param_.class_num);
    std::transform(param_.confidence_threshold.begin(), param_.confidence_threshold.end(), logit_threshold_.begin(),
                   logit);
    layout_ = engine.proto().output_layout;
    return true;
  }

protected:
  auto decode(DetectResultMap& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool override {
    if (outputs.size() != param_.strides.size())
      return false;
    for (std::size_t head = 0; head < outputs.size(); ++head) {
      const auto ok = visit_head(*outputs[head], layout_,
                                 [&](const auto& grid) -> void { decode_head(result, grid, param_.strides[head]); });
      if (!ok)
        return false;
    }
    finish_detect(result, param_.iou_threshold, transform);
    return true;
  }

private:
  template <typename Grid>
  auto decode_head(DetectResultMap& result, const Grid& grid, float stride) const -> void {
    const auto bins = param_.reg_max;
    const auto min_score = grid.raw_bound(*std::min_element(logit_threshold_.begin(), logit_threshold_.end()));
    for (std::int64_t y = 0; y < grid.rows(); ++y) {
      for (std::int64_t x = 0; x < grid.cols(); ++x) {
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
            box = core::Rect2f::from_points(
                {(cx - dfl(grid, y, x, 0, bins)) * stride, (cy - dfl(grid, y, x, bins, bins)) * stride},
                {(cx + dfl(grid, y, x, bins * 2, bins)) * stride, (cy + dfl(grid, y, x, bins * 3, bins)) * stride});
            decoded = true;
          }
          result[cls].push_back({box, sigmoid(score), static_cast<std::int64_t>(cls)});
        }
      }
    }
  }

  YoloV8Param param_;
  Vector<float> logit_threshold_;
  engine::DataLayout layout_ = engine::DataLayout::Nchw;
};

} // namespace

auto make_yolov8_detect() -> UniquePtr<DetectPostprocessor> {
  return std::make_unique<YoloV8Detect>();
}

} // namespace hrt::postprocess
