// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/rotated/yolov8_rotated.cpp
// Description: Implements YOLOv8 oriented-box decoding.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>

#include <algorithm>
#include <numbers>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

namespace {

// Channels per cell: 4 * reg_max distance bins, theta, classes.
class YoloV8Rotated final : public RotatedPostprocessor {
public:
  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    return check_section<YoloV8Param>(config);
  }

  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    param_ = config.get<YoloV8Param>();
    KZ_CHECK_RETURN(!config.contains("reg_max") || param_.reg_max > 0, false, "reg_max must be positive when supplied");
    if (!check_yolov8(param_, engine, 1))
      return false;
    logit_threshold_.resize(param_.class_num);
    std::transform(param_.confidence_threshold.begin(), param_.confidence_threshold.end(), logit_threshold_.begin(),
                   logit);
    layout_ = engine.proto().output_layout;
    return true;
  }

protected:
  auto decode(RotatedResultMap& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool override {
    if (outputs.size() != param_.strides.size())
      return false;
    for (std::size_t head = 0; head < outputs.size(); ++head) {
      const auto ok = visit_head(*outputs[head], layout_,
                                 [&](const auto& grid) -> void { decode_head(result, grid, param_.strides[head]); });
      if (!ok)
        return false;
    }
    return finish_rotated(result, param_.iou_threshold, transform);
  }

private:
  template <typename Grid>
  auto decode_head(RotatedResultMap& result, const Grid& grid, float stride) -> void {
    const auto bins = param_.reg_max;
    const auto min_score = grid.raw_bound(*std::min_element(logit_threshold_.begin(), logit_threshold_.end()));
    for (const auto cell : scan_cells(grid, bins * 4 + 1, param_.class_num, min_score, scan_)) {
      const std::int64_t y = cell / grid.cols();
      const std::int64_t x = cell % grid.cols();
      bool decoded = false;
      core::RotatedRect rect;
      for (std::size_t cls = 0; cls < param_.class_num; ++cls) {
        if (grid.raw(y, x, bins * 4 + 1 + cls) < min_score)
          continue;
        const auto score = grid.at(y, x, bins * 4 + 1 + cls);
        if (score < logit_threshold_[cls])
          continue;
        if (!decoded) {
          const auto theta = (sigmoid(grid.at(y, x, bins * 4)) - 0.25F) * 180.0F;
          const auto rad = theta / 180.0F * std::numbers::pi_v<float>;
          const auto cos = std::cos(rad);
          const auto sin = std::sin(rad);
          const auto l = dfl(grid, y, x, 0, bins);
          const auto t = dfl(grid, y, x, bins, bins);
          const auto r = dfl(grid, y, x, bins * 2, bins);
          const auto b = dfl(grid, y, x, bins * 3, bins);
          const auto xf = (r - l) / 2.0F;
          const auto yf = (b - t) / 2.0F;
          rect.center = {(xf * cos - yf * sin + static_cast<float>(x) + 0.5F) * stride,
                         (xf * sin + yf * cos + static_cast<float>(y) + 0.5F) * stride};
          rect.size = {(r + l) * stride, (b + t) * stride};
          rect.angle = theta;
          decoded = true;
        }
        result[cls].push_back({rect, sigmoid(score), static_cast<std::int64_t>(cls)});
      }
    }
  }

  YoloV8Param param_;
  CellScan scan_;
  Vector<float> logit_threshold_;
  engine::DataLayout layout_ = engine::DataLayout::Nchw;
};

} // namespace

auto make_yolov8_rotated() -> UniquePtr<RotatedPostprocessor> {
  return std::make_unique<YoloV8Rotated>();
}

} // namespace hrt::postprocess
